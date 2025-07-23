# `monte-carlo-sensitivity` 

Monte-Carlo Sensitivity Analysis Python Package

[Gregory H. Halverson](https://github.com/gregory-halverson-jpl) (they/them)<br>
[gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)<br>
Lead developer and designer<br>
NASA Jet Propulsion Laboratory 329G

Margaret C. Johnson (she/her)<br>
[maggie.johnson@jpl.nasa.gov](mailto:maggie.johnson@jpl.nasa.gov)<br>
Sensitivity and uncertainty analysis<br>
NASA Jet Propulsion Laboratory 398L

## Installation

This Python package is distributed using the pip package manager. Install it with the package name `monte-carlo-sensitivity` with dashes.

```mermaid
flowchart TD
   A[Input Preparation and Cleaning] --> B[Baseline Calculation
(Apply forward_process)]
   B --> C1{Single-Variable Perturbation?}
   C1 -- Yes --> D1("Generate n Univariate Perturbations\n(For each input variable)")
   D1 --> E1("Replicate Input Rows and Add Perturbations")
   C1 -- No  --> D2("Generate n Multivariate Perturbations\n(Joint for all input variables)")
   D2 --> E2("Replicate Input Rows and Add Joint Perturbations")
   E1 --> F("Model Evaluation\n(Apply forward_process to perturbed inputs)")
   E2 --> F
   F --> G("Effect Calculation\n(Differences & Normalization)")
   G --> H("Result Compilation\n(DataFrame of all results)")
   H --> I("Systematic Sensitivity Analysis\n(Repeat for all variable pairs or jointly)")
   I --> J("Sensitivity Metrics Calculation\n(Pearson, R², Mean Change)")
   J --> K("Output & Interpretation")
```
   I --> J[Sensitivity Metrics Calculation<br>(Pearson, R², Mean Change)]
   J --> K[Output & Interpretation]
```

The `monte-carlo-sensitivity` package provides a robust, simulation-based framework for quantifying the sensitivity of model outputs to input variables using Monte Carlo perturbation and statistical analysis. The methodology supports both single-variable and joint (multi-variable) perturbation workflows.

### 1. Input Preparation and Cleaning

- Start with an input DataFrame (`input_df`) containing all relevant variables.
- Specify the input variables to perturb and the output variables to analyze. These can be single variables or lists of variables, depending on the function used.
- Remove rows with missing values (NaNs) as needed to ensure valid analysis.

### 2. Monte Carlo Perturbation of Inputs

#### Single-Variable Perturbation (`perturbed_run`)

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

#### Joint (Multivariate) Perturbation (`joint_perturbed_run`)

For joint sensitivity analysis, multiple input variables are perturbed simultaneously, optionally allowing for correlations between them:

1. **Baseline Calculation:**
   - Compute the unperturbed output using the user-defined `forward_process` function.

2. **Joint Perturbation Generation:**
   - For each row, generate `n` random perturbation vectors for the selected input variables, using a multivariate distribution (default: multivariate normal with zero mean and diagonal covariance from input stds).
   - Replicate each input row `n` times and add the generated perturbation vectors to the input variables, creating a set of jointly perturbed inputs.

3. **Model Evaluation:**
   - Apply the model to the perturbed inputs to obtain perturbed outputs.

4. **Effect Calculation:**
   - Compute the difference between perturbed and unperturbed values for both input and output variables.
   - Normalize these differences (typically by dividing by the standard deviation for each variable).

5. **Result Compilation:**
   - Aggregate the results into a DataFrame, including unperturbed and perturbed values, perturbations, and their normalized forms for all input and output variables.

### 3. Systematic Sensitivity Analysis (Multiple Variables)

- Repeat the above Monte Carlo perturbation process for every combination of input and output variable (single-variable mode), or for all variables jointly (joint mode).
- Concatenate all results into a comprehensive DataFrame (`perturbation_df`) that records all perturbations and their effects.

### 4. Sensitivity Metrics Calculation

For each input-output variable pair, calculate the following metrics using the normalized perturbations:

- **Pearson Correlation:** Measures the linear relationship between normalized input and output perturbations.
- **R² (Coefficient of Determination):** Quantifies the proportion of variance in the output explained by the input perturbation (via linear regression).
- **Mean Normalized Change:** The average normalized change in the output variable due to input perturbation.

These metrics are aggregated into a summary DataFrame (`sensitivity_metrics_df`).

### 5. Output and Interpretation

- The process returns both the detailed perturbation results and the summary sensitivity metrics.
- This enables a comprehensive, quantitative assessment of how each input variable influences each output variable, supporting robust model evaluation and interpretation.

---

## Usage

Import this package in Python as `monte_carlo_sensitivity` with underscores.

Below are examples of how to use the key functions in the package:

### Sensitivity Analysis

The `sensitivity_analysis` function performs systematic sensitivity analysis by perturbing input variables and observing the effect on output variables.

```python
import pandas as pd
from monte_carlo_sensitivity import sensitivity_analysis

# Example input DataFrame
input_df = pd.DataFrame({
    "input_var1": [1, 2, 3],
    "input_var2": [4, 5, 6],
    "output_var": [7, 8, 9]
})

# Define a forward process function
def forward_process(df):
    df["output_var"] = df["input_var1"] + df["input_var2"]
    return df

# Perform sensitivity analysis
perturbation_df, sensitivity_metrics = sensitivity_analysis(
    input_df=input_df,
    input_variables=["input_var1", "input_var2"],
    output_variables=["output_var"],
    forward_process=forward_process,
    n=100
)

print(perturbation_df)
print(sensitivity_metrics)
```

### Perturbed Run

The `perturbed_run` function performs a Monte Carlo sensitivity analysis by perturbing a single input variable and observing the effect on a single output variable.

```python
import pandas as pd
from monte_carlo_sensitivity.perturbed_run import perturbed_run

# Example input DataFrame
input_df = pd.DataFrame({
    "input_var1": [1, 2, 3],
    "input_var2": [4, 5, 6],
    "output_var": [7, 8, 9]
})

# Define a forward process function
def forward_process(df):
    df["output_var"] = df["input_var1"] + df["input_var2"]
    return df

# Perform a perturbed run
results = perturbed_run(
    input_df=input_df,
    input_variable="input_var1",
    output_variable="output_var",
    forward_process=forward_process,
    n=100
)

print(results)
```

### Joint Perturbed Run

The `joint_perturbed_run` function evaluates the sensitivity of output variables to joint perturbations in multiple input variables.

```python
import pandas as pd
from monte_carlo_sensitivity.joint_perturbed_run import joint_perturbed_run

# Example input DataFrame
input_df = pd.DataFrame({
    "input_var1": [1, 2, 3],
    "input_var2": [4, 5, 6],
    "output_var": [7, 8, 9]
})

# Define a forward process function
def forward_process(df):
    df["output_var"] = df["input_var1"] + df["input_var2"]
    return df

# Perform a joint perturbed run
results = joint_perturbed_run(
    input_df=input_df,
    input_variable=["input_var1", "input_var2"],
    output_variable=["output_var"],
    forward_process=forward_process,
    n=100
)

print(results)
```
