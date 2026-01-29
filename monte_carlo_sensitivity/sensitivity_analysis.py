from typing import Callable, Tuple, List, Optional, Union, Dict
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.stats import mstats

from .perturbed_run import DEFAULT_NORMALIZATION_FUNCTION, perturbed_run
from .repeat_rows import repeat_rows


def sensitivity_analysis(
        input_df: pd.DataFrame,
        input_variables: List[str],
        output_variables: List[str],
        forward_process: Callable,
        perturbation_process: Callable = np.random.normal,
        normalization_function: Callable = DEFAULT_NORMALIZATION_FUNCTION,
        n: int = 100,
        perturbation_mean: float = 0,
        perturbation_std: float = None,
        use_joint_run: bool = True,
        input_min: Optional[Union[Dict[str, float], float]] = None,
        input_max: Optional[Union[Dict[str, float], float]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        use_joint_run (bool, optional): If True, runs forward process once on all perturbations (more efficient). 
                                       If False, uses original loop-based approach. Defaults to True.
        input_min (Optional[Union[Dict[str, float], float]], optional): Minimum allowed values for input variables.
                                       Can be a single float (applied to all variables) or dict mapping variable names to limits.
                                       Perturbed values below this limit will be clipped. Defaults to None (no constraint).
        input_max (Optional[Union[Dict[str, float], float]], optional): Maximum allowed values for input variables.
                                       Can be a single float (applied to all variables) or dict mapping variable names to limits.
                                       Perturbed values above this limit will be clipped. Defaults to None (no constraint).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - perturbation_df (pd.DataFrame): A DataFrame with details of the perturbations and their effects.
            - sensitivity_metrics_df (pd.DataFrame): A DataFrame with sensitivity metrics such as correlation, R², and mean normalized change.
    
    Notes:
        When input_min or input_max constraints are specified, perturbed values are clipped to stay within bounds.
        This uses simple post-perturbation clipping, which introduces bias when many values hit the constraints.
        The actual perturbations recorded reflect the clipped values, not the original generated perturbations.
        Consider using smaller perturbation_std if many values are being clipped (>10% of perturbations).
    """
    # Filter out NaN values for input variables and coerce to numeric to avoid object issues
    for input_variable in input_variables:
        numeric_col = pd.to_numeric(input_df[input_variable], errors="coerce")
        valid_mask = ~np.isnan(numeric_col)
        input_df = input_df.loc[valid_mask].copy()
        input_df[input_variable] = numeric_col.loc[valid_mask].values

    if use_joint_run:
        return _sensitivity_analysis_joint(
            input_df, input_variables, output_variables, forward_process,
            perturbation_process, normalization_function, n, perturbation_mean, perturbation_std,
            input_min, input_max
        )
    else:
        return _sensitivity_analysis_loop(
            input_df, input_variables, output_variables, forward_process,
            perturbation_process, normalization_function, n, perturbation_mean, perturbation_std,
            input_min, input_max
        )


def _sensitivity_analysis_joint(
        input_df: pd.DataFrame,
        input_variables: List[str],
        output_variables: List[str],
        forward_process: Callable,
        perturbation_process: Callable,
        normalization_function: Callable,
        n: int,
        perturbation_mean: float,
        perturbation_std: float,
        input_min: Optional[Union[Dict[str, float], float]] = None,
        input_max: Optional[Union[Dict[str, float], float]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Optimized sensitivity analysis that runs forward process only twice:
    once on unperturbed data, once on all combined perturbations.
    """
    # Calculate standard deviations for each input variable
    input_stds = {}
    perturbation_stds = {}
    for input_variable in input_variables:
        input_std = np.nanstd(input_df[input_variable])
        if input_std == 0 or np.isnan(input_std):
            input_std = np.nan
        input_stds[input_variable] = input_std
        
        # Use standard deviation of input variable as perturbation std if not given
        if perturbation_std is None:
            if np.isnan(input_std) or input_std == 0:
                perturbation_stds[input_variable] = 1.0
            else:
                perturbation_stds[input_variable] = input_std
        else:
            perturbation_stds[input_variable] = perturbation_std

    # Run forward process ONCE on unperturbed data
    unperturbed_output_df = forward_process(input_df)
    
    # Coerce outputs to numeric to handle object dtypes
    for col in output_variables:
        if col in unperturbed_output_df.columns:
            unperturbed_output_df[col] = pd.to_numeric(
                unperturbed_output_df[col],
                errors='coerce'
            ).astype(np.float64)
    
    # Calculate output standard deviations
    output_stds = {}
    for output_variable in output_variables:
        output_std = np.nanstd(unperturbed_output_df[output_variable])
        if output_std == 0:
            output_std = np.nan
        output_stds[output_variable] = output_std

    # Generate all perturbations for all input variables
    all_perturbations = {}
    all_unperturbed_inputs = {}
    
    for input_variable in input_variables:
        perturbations = np.concatenate([
            perturbation_process(perturbation_mean, perturbation_stds[input_variable], n) 
            for i in range(len(input_df))
        ])
        all_perturbations[input_variable] = perturbations

    # Build one large combined dataframe with all perturbation scenarios stacked
    combined_perturbed_dfs = []
    perturbation_metadata = []
    
    for input_variable in input_variables:
        # Copy and repeat input data for this variable's perturbations
        perturbed_input_df = repeat_rows(input_df.copy(), n)
        unperturbed_input = perturbed_input_df[input_variable].copy()
        
        # Apply perturbation to only this input variable
        perturbations = all_perturbations[input_variable]
        perturbed_values = perturbed_input_df[input_variable] + perturbations
        
        # Apply constraints if specified
        var_min = input_min.get(input_variable, None) if isinstance(input_min, dict) else input_min
        var_max = input_max.get(input_variable, None) if isinstance(input_max, dict) else input_max
        
        if var_min is not None or var_max is not None:
            perturbed_values = np.clip(perturbed_values, var_min, var_max)
            # Recalculate actual perturbations after clipping
            perturbations = perturbed_values.values - unperturbed_input.values
        
        perturbed_input_df[input_variable] = perturbed_values
        
        # Store metadata for later
        normalized_perturbations = normalization_function(perturbations, unperturbed_input)
        perturbation_metadata.append({
            'input_variable': input_variable,
            'unperturbed_input': unperturbed_input,
            'perturbations': perturbations,
            'normalized_perturbations': normalized_perturbations,
            'perturbed_input': perturbed_input_df[input_variable].copy()
        })
        
        combined_perturbed_dfs.append(perturbed_input_df)
    
    # Stack all perturbed scenarios into one big dataframe
    combined_perturbed_df = pd.concat(combined_perturbed_dfs, ignore_index=True)
    
    # Run forward process ONCE on all combined perturbations
    combined_perturbed_output_df = forward_process(combined_perturbed_df)
    
    # Coerce outputs to numeric to handle object dtypes
    for col in output_variables:
        if col in combined_perturbed_output_df.columns:
            combined_perturbed_output_df[col] = pd.to_numeric(
                combined_perturbed_output_df[col],
                errors='coerce'
            ).astype(np.float64)
    
    # Split the combined output back into separate results per input variable
    rows_per_scenario = len(input_df) * n
    perturbed_outputs_by_variable = {}
    
    for idx, input_variable in enumerate(input_variables):
        start_idx = idx * rows_per_scenario
        end_idx = (idx + 1) * rows_per_scenario
        perturbed_outputs_by_variable[input_variable] = combined_perturbed_output_df.iloc[start_idx:end_idx].reset_index(drop=True)
    
    # Prepare repeated unperturbed outputs
    repeated_unperturbed_output = repeat_rows(unperturbed_output_df, n)
    
    # Build results for each input-output combination
    sensitivity_metrics_list = []
    perturbation_list = []
    
    for output_variable in output_variables:
        for i, input_variable in enumerate(input_variables):
            metadata = perturbation_metadata[i]
            perturbed_output_df = perturbed_outputs_by_variable[input_variable]
            
            # Extract relevant data
            unperturbed_input = metadata['unperturbed_input']
            perturbed_input = metadata['perturbed_input']
            input_perturbation = metadata['perturbations']
            input_perturbation_std = metadata['normalized_perturbations']
            
            unperturbed_output = repeated_unperturbed_output[output_variable]
            perturbed_output = perturbed_output_df[output_variable]
            output_perturbation = perturbed_output.values - unperturbed_output.values
            output_perturbation_std = normalization_function(output_perturbation, unperturbed_output.values)
            
            # Create results dataframe for this combination
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
                "output_perturbed": perturbed_output.values,
            })
            
            perturbation_list.append(results_df)
            
            # Calculate metrics
            variable_perturbation_df = pd.DataFrame({
                "input_perturbation_std": input_perturbation_std,
                "output_perturbation_std": output_perturbation_std
            }).dropna()

            # Sanitize to numeric and filter to finite values once for all metrics
            input_pert_std = np.array(pd.to_numeric(
                variable_perturbation_df.input_perturbation_std,
                errors="coerce"
            ), dtype=np.float64)
            output_pert_std = np.array(pd.to_numeric(
                variable_perturbation_df.output_perturbation_std,
                errors="coerce"
            ), dtype=np.float64)

            valid_mask = np.isfinite(input_pert_std) & np.isfinite(output_pert_std)
            valid_input = input_pert_std[valid_mask]
            valid_output = output_pert_std[valid_mask]

            if len(valid_input) > 2:
                correlation = mstats.pearsonr(valid_input, valid_output)[0]
            else:
                correlation = np.nan

            sensitivity_metrics_list.append([
                input_variable, output_variable, "correlation", correlation
            ])

            # Calculate R² and mean normalized change using sanitized values
            if len(valid_input) >= 2:
                input_var = np.nanvar(valid_input)
                output_var = np.nanvar(valid_output)

                if input_var > 1e-10 and output_var > 1e-10:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        r2 = scipy.stats.linregress(valid_input, valid_output)[2] ** 2
                else:
                    r2 = np.nan

                mean_normalized_change = np.nanmean(valid_output)
            else:
                r2 = np.nan
                mean_normalized_change = np.nan

            sensitivity_metrics_list.append([
                input_variable, output_variable, "r2", r2
            ])
            sensitivity_metrics_list.append([
                input_variable, output_variable, "mean_normalized_change", mean_normalized_change
            ])

    # Combine all results
    perturbation_df = pd.concat(perturbation_list, ignore_index=True) if perturbation_list else pd.DataFrame(columns=[
        "input_variable", "output_variable", "input_unperturbed", "input_perturbation",
        "input_perturbation_std", "input_perturbed", "output_unperturbed", "output_perturbation",
        "output_perturbation_std", "output_perturbed"
    ])
    
    sensitivity_metrics_df = pd.DataFrame(
        sensitivity_metrics_list,
        columns=["input_variable", "output_variable", "metric", "value"]
    ) if sensitivity_metrics_list else pd.DataFrame(
        columns=["input_variable", "output_variable", "metric", "value"]
    )

    return perturbation_df, sensitivity_metrics_df


def _sensitivity_analysis_loop(
        input_df: pd.DataFrame,
        input_variables: List[str],
        output_variables: List[str],
        forward_process: Callable,
        perturbation_process: Callable,
        normalization_function: Callable,
        n: int,
        perturbation_mean: float,
        perturbation_std: float,
        input_min: Optional[Union[Dict[str, float], float]] = None,
        input_max: Optional[Union[Dict[str, float], float]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Original loop-based sensitivity analysis (for backward compatibility).
    """
    sensitivity_metrics_columns = ["input_variable", "output_variable", "metric", "value"]
    sensitivity_metrics_list = []
    perturbation_list = []

    for output_variable in output_variables:
        for input_variable in input_variables:
            # Extract constraints for this specific variable
            var_min = input_min.get(input_variable, None) if isinstance(input_min, dict) else input_min
            var_max = input_max.get(input_variable, None) if isinstance(input_max, dict) else input_max
            
            run_results = perturbed_run(
                input_df=input_df,
                input_variable=input_variable,
                output_variable=output_variable,
                forward_process=forward_process,
                perturbation_process=perturbation_process,
                n=n,
                perturbation_mean=perturbation_mean,
                perturbation_std=perturbation_std,
                normalization_function=normalization_function,
                input_min=var_min,
                input_max=var_max
            )

            perturbation_list.append(run_results)
            input_perturbation_std = np.array(run_results[(run_results.input_variable == input_variable) & (run_results.output_variable == output_variable)].input_perturbation_std).astype(np.float32)
            output_perturbation_std = np.array(run_results[(run_results.output_variable == output_variable) & (run_results.output_variable == output_variable)].output_perturbation_std).astype(np.float32)
            variable_perturbation_df = pd.DataFrame({"input_perturbation_std": input_perturbation_std, "output_perturbation_std": output_perturbation_std})
            variable_perturbation_df = variable_perturbation_df.dropna()
            
            # Coerce to float64, drop invalids once, and reuse for all metrics
            input_pert_std = np.array(pd.to_numeric(
                variable_perturbation_df.input_perturbation_std,
                errors="coerce"
            ), dtype=np.float64)
            output_pert_std = np.array(pd.to_numeric(
                variable_perturbation_df.output_perturbation_std,
                errors="coerce"
            ), dtype=np.float64)

            valid_mask = np.isfinite(input_pert_std) & np.isfinite(output_pert_std)
            valid_input = input_pert_std[valid_mask]
            valid_output = output_pert_std[valid_mask]

            if len(valid_input) > 2:
                correlation = mstats.pearsonr(valid_input, valid_output)[0]
            else:
                correlation = np.nan

            sensitivity_metrics_list.append([
                input_variable,
                output_variable,
                "correlation",
                correlation
            ])

            # Suppress expected warnings for small samples
            # Check if there are enough valid data points for regression
            if len(valid_input) >= 2:
                input_var = np.nanvar(valid_input)
                output_var = np.nanvar(valid_output)

                if input_var > 1e-10 and output_var > 1e-10:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        r2 = scipy.stats.linregress(valid_input, valid_output)[2] ** 2
                else:
                    r2 = np.nan

                mean_normalized_change = np.nanmean(valid_output)
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