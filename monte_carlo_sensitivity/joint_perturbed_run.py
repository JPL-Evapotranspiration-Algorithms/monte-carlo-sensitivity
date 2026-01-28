from typing import Callable, Union, List

import numpy as np
import pandas as pd

from monte_carlo_sensitivity.repeat_rows import repeat_rows


def joint_perturbed_run(
        input_df: pd.DataFrame,
        input_variable: Union[str, List[str]],
        output_variable: Union[str, List[str]],
        forward_process: Callable,
        perturbation_process: Callable = np.random.multivariate_normal,
        n: int = 100,
        perturbation_mean: float = None,
        perturbation_cov: float = None) -> pd.DataFrame:
    """
    Perform a joint perturbed run analysis on input data to evaluate the sensitivity of output variables
    to perturbations in input variables.

    Parameters:
        input_df (pd.DataFrame): The input DataFrame containing the input variables.
        input_variable (Union[str, List[str]]): The name(s) of the input variable(s) to perturb.
        output_variable (Union[str, List[str]]): The name(s) of the output variable(s) to analyze.
        forward_process (Callable): A function that processes the input DataFrame and returns a DataFrame
                                    with output variables.
        perturbation_process (Callable, optional): A function to generate perturbations. Defaults to
                                                   np.random.multivariate_normal.
        n (int, optional): The number of perturbations to generate for each input. Defaults to 100.
        perturbation_mean (float, optional): The mean of the perturbation distribution. Defaults to None,
                                             which assumes zero mean.
        perturbation_cov (float, optional): The covariance matrix of the perturbation distribution. Defaults
                                            to None, which assumes diagonal covariance based on input standard
                                            deviations.

    Returns:
        pd.DataFrame: A DataFrame containing the unperturbed inputs, perturbed inputs, perturbations,
                      unperturbed outputs, perturbed outputs, and standardized perturbations for both inputs
                      and outputs.
    """
    # Normalize inputs to lists for consistent handling
    if isinstance(input_variable, str):
        input_variable = [input_variable]
    if isinstance(output_variable, str):
        output_variable = [output_variable]
    
    # calculate standard deviation of the input variable
    n_input = len(input_variable)
    n_output = len(output_variable)

    input_std = np.nanstd(input_df[input_variable], axis=0)
    # Handle both scalar and array results from nanstd
    input_std = np.atleast_1d(input_std)

    # For single-row inputs or zero variance, use default std of 1.0
    if np.all(input_std == 0) or np.all(np.isnan(input_std)):
        input_std = np.ones(n_input)

    # use diagonal (independent) standard deviations of the input variables if not given
    if perturbation_cov is None:
        # For single-row inputs or zero variance, use identity matrix as default
        if np.any(np.isnan(input_std)) or np.any(input_std == 0):
            perturbation_cov = np.eye(n_input)
        else:
            perturbation_cov = np.diag(input_std)

    if perturbation_mean is None:
        perturbation_mean = np.zeros(n_input)


    # forward process the unperturbed input
    unperturbed_output_df = forward_process(input_df)
    # calculate standard deviation of the output variable
    output_std = np.nanstd(unperturbed_output_df[output_variable], axis=0)
    # Handle both scalar and array results from nanstd
    output_std = np.atleast_1d(output_std)

    if np.all(output_std == 0):
        output_std = np.full(n_output, np.nan)

    # extract output variable from unperturbed output
    unperturbed_output = unperturbed_output_df[output_variable]
    # repeat unperturbed output
    unperturbed_output = repeat_rows(unperturbed_output, n)
    # generate input perturbation
    input_perturbation = perturbation_process(perturbation_mean, perturbation_cov, n*len(input_df))
    print(input_perturbation.shape)
    # Suppress divide by zero warning - produces inf/NaN which is handled downstream
    with np.errstate(divide='ignore', invalid='ignore'):
        input_perturbation_std = input_perturbation / input_std
    # copy input for perturbation
    perturbed_input_df = input_df.copy()
    # repeat input for perturbation
    perturbed_input_df = repeat_rows(perturbed_input_df, n)
    # extract input variable from repeated unperturbed input
    unperturbed_input = perturbed_input_df[input_variable]
    # add perturbation to input
    perturbed_input_df[input_variable] = perturbed_input_df[input_variable] + input_perturbation
    # extract perturbed input
    perturbed_input = perturbed_input_df[input_variable]
    # forward process the perturbed input
    perturbed_output_df = forward_process(perturbed_input_df)
    # extract output variable from perturbed output
    perturbed_output = perturbed_output_df[output_variable]
    # calculate output perturbation
    output_perturbation = perturbed_output - unperturbed_output
    output_perturbation_std = output_perturbation / output_std

    input_perturbation_df = pd.DataFrame(input_perturbation, columns=[s+"_perturbation" for s in input_variable])
    input_perturbation_std_df = pd.DataFrame(input_perturbation_std, columns=[s+"_perturbation_std" for s in input_variable])

    unperturbed_output = unperturbed_output.loc[:,~unperturbed_output.columns.duplicated()]

    unperturbed_input.columns = [s+"_unperturbed" for s in input_variable]
    unperturbed_output.columns = [s+"_unperturbed" for s in output_variable]
    perturbed_input.columns = [s+"_perturbed" for s in input_variable]
    output_perturbation.columns = [s+"_perturbation" for s in output_variable]
    output_perturbation_std.columns = [s+"_perturbation_std" for s in output_variable]
    perturbed_output.columns = [s+"_perturbed" for s in output_variable]

    results_df = pd.concat([unperturbed_input,
                            input_perturbation_df,
                            input_perturbation_std_df,
                            perturbed_input,
                            unperturbed_output,
                            output_perturbation,
                            output_perturbation_std,
                            perturbed_output], axis=1)

    return results_df