from typing import Callable
import logging

import numpy as np
import pandas as pd

from .repeat_rows import repeat_rows
from .divide_by_std import divide_by_std

DEFAULT_NORMALIZATION_FUNCTION = divide_by_std

logger = logging.getLogger(__name__)

def perturbed_run(
        input_df: pd.DataFrame,
        input_variable: str,
        output_variable: str,
        forward_process: Callable,
        perturbation_process: Callable = np.random.normal,
        normalization_function: Callable = DEFAULT_NORMALIZATION_FUNCTION,
        n: int = 100,
        perturbation_mean: float = 0,
        perturbation_std: float = None,
        dropna: bool = True) -> pd.DataFrame:
    """
    Perform a Monte Carlo sensitivity analysis by perturbing an input variable and observing the effect on an output variable.

    Parameters:
        input_df (pd.DataFrame): The input DataFrame containing the data to be perturbed.
        input_variable (str): The name of the input variable to perturb.
        output_variable (str): The name of the output variable to analyze.
        forward_process (Callable): A function that processes the input DataFrame and returns a DataFrame with the output variable.
        perturbation_process (Callable, optional): A function to generate perturbations (default: np.random.normal).
        normalization_function (Callable, optional): A function to normalize perturbations (default: divide_by_std).
        n (int, optional): The number of perturbations to generate for each input row (default: 100).
        perturbation_mean (float, optional): The mean of the perturbation distribution (default: 0).
        perturbation_std (float, optional): The standard deviation of the perturbation distribution (default: None, uses input variable's std).
        dropna (bool, optional): Whether to drop rows with NaN values in the results (default: True).

    Returns:
        pd.DataFrame: A DataFrame containing the results of the sensitivity analysis, including unperturbed and perturbed inputs and outputs.

    The returned DataFrame includes the following columns:
        - "input_variable": The name of the input variable.
        - "output_variable": The name of the output variable.
        - "input_unperturbed": The unperturbed input values.
        - "input_perturbation": The perturbations applied to the input.
        - "input_perturbation_std": The normalized input perturbations.
        - "input_perturbed": The perturbed input values.
        - "output_unperturbed": The unperturbed output values.
        - "output_perturbation": The perturbations observed in the output.
        - "output_perturbation_std": The normalized output perturbations.
        - "output_perturbed": The perturbed output values.
    """
    logger.info("Starting Monte Carlo perturbed run")

    logger.info(f"calculating standard deviation of input variable: {input_variable}")
    # calculate standard deviation of the input variable
    input_std = np.nanstd(input_df[input_variable])
    logger.info(f"input variable {input_variable} standard deviation: {input_std}")

    if input_std == 0:
        input_std = np.nan

    # use standard deviation of the input variable as the perturbation standard deviation if not given
    if perturbation_std is None:
        if np.isnan(input_std) or input_std == 0:
            # For single-row or zero-variance inputs, use a default perturbation
            perturbation_std = 1.0
            logger.warning(f"Input variable {input_variable} has zero or undefined std; using default perturbation_std=1.0")
        else:
            perturbation_std = input_std

    logger.info("starting forward process")
    # forward process the unperturbed input
    unperturbed_output_df = forward_process(input_df)
    logger.info("forward process completed")

    logger.info(f"calculating standard deviation of output variable: {output_variable}")
    # calculate standard deviation of the output variable
    output_std = np.nanstd(unperturbed_output_df[output_variable])
    logger.info(f"output variable {output_variable} standard deviation: {output_std}")

    if output_std == 0:
        output_std = np.nan

    # extract output variable from unperturbed output
    unperturbed_output = unperturbed_output_df[output_variable]
    # repeat unperturbed output
    unperturbed_output = repeat_rows(unperturbed_output_df, n)[output_variable]

    logger.info("starting input perturbation generation")
    # generate input perturbation
    input_perturbation = np.concatenate([perturbation_process(perturbation_mean, perturbation_std, n) for i in range(len(input_df))])
    logger.info("input perturbation generation completed")

    # input_perturbation_std = input_perturbation / input_std

    logger.info("generating control group")
    # copy input for perturbation
    perturbed_input_df = input_df.copy()
    # repeat input for perturbation
    perturbed_input_df = repeat_rows(perturbed_input_df, n)
    # extract input variable from repeated unperturbed input
    unperturbed_input = perturbed_input_df[input_variable]

    logger.info("normalizing input perturbations")
    # normalize input perturbations
    input_perturbation_std = normalization_function(input_perturbation, unperturbed_input)

    logger.info("applying perturbations")
    # add perturbation to input
    perturbed_input_df[input_variable] = perturbed_input_df[input_variable] + input_perturbation
    # extract perturbed input
    perturbed_input = perturbed_input_df[input_variable]

    logger.info("starting forward process for perturbed input")
    # forward process the perturbed input
    perturbed_output_df = forward_process(perturbed_input_df)
    logger.info("completed forward process for perturbed input")

    # extract output variable from perturbed output
    perturbed_output = perturbed_output_df[output_variable]
    # calculate output perturbation
    output_perturbation = perturbed_output - unperturbed_output

    logger.info("normalizing output perturbations")
    # normalize output perturbations
    output_perturbation_std = normalization_function(output_perturbation, unperturbed_output)

    results_df = pd.DataFrame({
        "input_variable": input_variable,
        "output_variable": output_variable,
        "input_unperturbed": unperturbed_input,
        "input_perturbation": input_perturbation,
        "input_perturbation_std": input_perturbation_std,
        "input_perturbed": perturbed_input,
        "output_unperturbed": unperturbed_output,
        "output_perturbation": output_perturbation,
        "output_perturbation_std": output_perturbation_std,
        "output_perturbed": perturbed_output,
    })

    if dropna:
        results_df = results_df.dropna()

    logger.info("Monte Carlo run complete")

    return results_df
