# Methodology: Monte Carlo Sensitivity Analysis via Input Perturbation

We performed a Monte Carlo sensitivity analysis to quantify the effect of perturbations in a selected input variable on a specified output variable. The methodology consists of the following steps:

1. **Input Preparation:**  
   The analysis begins with an input dataset (`input_df`) containing the variables of interest. A specific input variable (`input_variable`) is selected for perturbation, and an output variable (`output_variable`) is chosen for sensitivity assessment.

2. **Baseline Output Calculation:**  
   The unperturbed input data is processed using a user-defined model or function (`forward_process`), producing the baseline (unperturbed) output values.

3. **Perturbation Generation:**  
   For each row in the input dataset, a set of random perturbations is generated for the input variable. By default, these perturbations are drawn from a normal distribution with mean zero and a standard deviation equal to that of the input variable, unless otherwise specified.

4. **Input Replication and Perturbation:**  
   Each input row is replicated `n` times (default: 100), and the generated perturbations are added to the selected input variable, creating a set of perturbed input datasets.

5. **Model Evaluation on Perturbed Inputs:**  
   The perturbed input datasets are processed through the same model or function (`forward_process`) to obtain the corresponding perturbed output values.

6. **Computation of Perturbation Effects:**  
   For each perturbed instance, the following are computed:
   - The difference between the perturbed and unperturbed input values.
   - The difference between the perturbed and unperturbed output values.
   - Normalized perturbations for both input and output, typically by dividing by the standard deviation.

7. **Result Aggregation:**  
   The results are compiled into a single DataFrame, including the input and output variables, their unperturbed and perturbed values, the applied perturbations, and their normalized forms.

8. **Post-processing:**  
   Optionally, rows containing missing values (NaNs) are removed from the results.

This approach enables the estimation of the sensitivity of the output variable to random fluctuations in the input variable, providing a robust, simulation-based assessment of model behavior under uncertainty.
