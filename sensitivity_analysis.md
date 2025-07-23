
# Methodology: Monte Carlo-Based Sensitivity Analysis Framework

This methodology describes a two-level process for quantifying the sensitivity of model outputs to input variables using Monte Carlo perturbation and statistical analysis. The approach combines detailed single-variable perturbation with systematic, multi-variable assessment.

## 1. Input Preparation and Cleaning

- Begin with an input DataFrame (`input_df`) containing all relevant variables.
- Specify lists of input variables (`input_variables`) to perturb and output variables (`output_variables`) to analyze.
- For each input variable, remove rows with missing values (NaNs) to ensure valid analysis.

## 2. Monte Carlo Perturbation of Inputs (Single Variable)

For each input variable and output variable pair:

1. **Baseline Calculation:**
   - Compute the unperturbed output by applying a user-defined model or function (`forward_process`) to the input data.

2. **Perturbation Generation:**
   - For each row, generate `n` random perturbations for the selected input variable (default: normal distribution, mean zero, std equal to the variable's std).
   - Replicate each input row `n` times and add the generated perturbations to the selected input variable, creating a set of perturbed inputs.

3. **Model Evaluation:**
   - Apply the model to the perturbed inputs to obtain perturbed outputs.

4. **Effect Calculation:**
   - Compute the difference between perturbed and unperturbed values for both input and output.
   - Normalize these differences (typically by dividing by the standard deviation).

5. **Result Compilation:**
   - Aggregate the results into a DataFrame, including unperturbed and perturbed values, perturbations, and their normalized forms.

## 3. Systematic Sensitivity Analysis (Multiple Variables)

- Repeat the above Monte Carlo perturbation process for every combination of input and output variable.
- Concatenate all results into a comprehensive DataFrame (`perturbation_df`) that records all perturbations and their effects.

## 4. Sensitivity Metrics Calculation

For each input-output variable pair, calculate the following metrics using the normalized perturbations:

- **Pearson Correlation:** Measures the linear relationship between normalized input and output perturbations.
- **R² (Coefficient of Determination):** Quantifies the proportion of variance in the output explained by the input perturbation (via linear regression).
- **Mean Normalized Change:** The average normalized change in the output variable due to input perturbation.

These metrics are aggregated into a summary DataFrame (`sensitivity_metrics_df`).

## 5. Output and Interpretation

- The process returns both the detailed perturbation results and the summary sensitivity metrics.
- This enables a comprehensive, quantitative assessment of how each input variable influences each output variable, supporting robust model evaluation and interpretation.

---

This framework extends single-variable Monte Carlo sensitivity analysis to a multi-variable, multi-output context, providing both granular and summary insights into model sensitivity and robustness.
