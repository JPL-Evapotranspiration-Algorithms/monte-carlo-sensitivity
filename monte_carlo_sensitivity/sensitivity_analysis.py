from typing import Callable, Tuple, List
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.stats import mstats

from .perturbed_run import DEFAULT_NORMALIZATION_FUNCTION, perturbed_run


def sensitivity_analysis(
        input_df: pd.DataFrame,
        input_variables: List[str],
        output_variables: List[str],
        forward_process: Callable,
        perturbation_process: Callable = np.random.normal,
        normalization_function: Callable = DEFAULT_NORMALIZATION_FUNCTION,
        n: int = 100,
        perturbation_mean: float = 0,
        perturbation_std: float = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform sensitivity analysis by perturbing input variables and observing the effect on output variables.

    Args:
        input_df (pd.DataFrame): The input data as a pandas DataFrame.
        input_variables (str): List of input variable names to perturb.
        output_variables (str): List of output variable names to analyze.
        forward_process (Callable): A function that processes the input data and produces output data.
        perturbation_process (Callable, optional): A function to generate perturbations. Defaults to np.random.normal.
        normalization_function (Callable, optional): A function to normalize the data. Defaults to default_normalization_function.
        n (int, optional): Number of perturbations to generate. Defaults to 100.
        perturbation_mean (float, optional): Mean of the perturbation distribution. Defaults to 0.
        perturbation_std (float, optional): Standard deviation of the perturbation distribution. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - perturbation_df (pd.DataFrame): A DataFrame with details of the perturbations and their effects.
            - sensitivity_metrics_df (pd.DataFrame): A DataFrame with sensitivity metrics such as correlation, R², and mean normalized change.
    """
    # print(len(input_df))

    for input_variable in input_variables:
        input_df = input_df[~np.isnan(input_df[input_variable])]

    # print(len(input_df))

    sensitivity_metrics_columns = ["input_variable", "output_variable", "metric", "value"]
    sensitivity_metrics_list = []
    perturbation_list = []

    for output_variable in output_variables:
        for input_variable in input_variables:
            run_results = perturbed_run(
                input_df=input_df,
                input_variable=input_variable,
                output_variable=output_variable,
                forward_process=forward_process,
                perturbation_process=perturbation_process,
                n=n,
                perturbation_mean=perturbation_mean,
                perturbation_std=perturbation_std,
                normalization_function=normalization_function
            )

            perturbation_list.append(run_results)
            input_perturbation_std = np.array(run_results[(run_results.input_variable == input_variable) & (run_results.output_variable == output_variable)].input_perturbation_std).astype(np.float32)
            output_perturbation_std = np.array(run_results[(run_results.output_variable == output_variable) & (run_results.output_variable == output_variable)].output_perturbation_std).astype(np.float32)
            # correlation = np.corrcoef(input_perturbation_std, output_perturbation_std)[0][1]
            variable_perturbation_df = pd.DataFrame({"input_perturbation_std": input_perturbation_std, "output_perturbation_std": output_perturbation_std})
            # print(len(variable_perturbation_df))
            variable_perturbation_df = variable_perturbation_df.dropna()
            # print(len(variable_perturbation_df))
            input_perturbation_std = variable_perturbation_df.input_perturbation_std
            output_perturbation_std = variable_perturbation_df.output_perturbation_std
            correlation = mstats.pearsonr(input_perturbation_std, output_perturbation_std)[0]

            sensitivity_metrics_list.append([
                input_variable,
                output_variable,
                "correlation",
                correlation
            ])

            # Suppress expected warnings for small samples
            # Check if there are enough valid data points for regression
            if len(input_perturbation_std) >= 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r2 = scipy.stats.linregress(input_perturbation_std, output_perturbation_std)[2] ** 2
                    mean_normalized_change = np.nanmean(output_perturbation_std)
            else:
                # Not enough data points for regression (e.g., constant output)
                r2 = np.nan
                mean_normalized_change = np.nan

            sensitivity_metrics_list.append([
                input_variable,
                output_variable,
                "r2",
                r2
            ])

            sensitivity_metrics_list.append([
                input_variable,
                output_variable,
                "mean_normalized_change",
                mean_normalized_change
            ])

    perturbation_df = pd.concat(perturbation_list, ignore_index=True) if perturbation_list else pd.DataFrame(columns=[
        "input_variable", "output_variable", "input_unperturbed", "input_perturbation",
        "input_perturbation_std", "input_perturbed", "output_unperturbed", "output_perturbation",
        "output_perturbation_std", "output_perturbed"
    ])
    
    sensitivity_metrics_df = pd.DataFrame(sensitivity_metrics_list, columns=sensitivity_metrics_columns) if sensitivity_metrics_list else pd.DataFrame(columns=sensitivity_metrics_columns)

    return perturbation_df, sensitivity_metrics_df