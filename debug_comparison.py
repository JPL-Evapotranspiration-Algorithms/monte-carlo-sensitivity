"""
Debug script to check differences between optimized and original versions.
"""
import numpy as np
import pandas as pd
from monte_carlo_sensitivity import sensitivity_analysis

# Create test data with fixed seed for reproducibility
np.random.seed(42)
input_df = pd.DataFrame({
    'input1': np.random.randn(20),
    'input2': np.random.randn(20),
})

def forward_process(df):
    """Simple forward process."""
    result = df.copy()
    result['output1'] = df['input1'] * 2 + df['input2'] * 3
    result['output2'] = df['input1'] ** 2
    return result

input_variables = ['input1', 'input2']
output_variables = ['output1', 'output2']

# Run both versions
np.random.seed(42)
perturbation_df_opt, metrics_df_opt = sensitivity_analysis(
    input_df=input_df,
    input_variables=input_variables,
    output_variables=output_variables,
    forward_process=forward_process,
    n=50,
    use_joint_run=True
)

np.random.seed(42)
perturbation_df_orig, metrics_df_orig = sensitivity_analysis(
    input_df=input_df,
    input_variables=input_variables,
    output_variables=output_variables,
    forward_process=forward_process,
    n=50,
    use_joint_run=False
)

# Compare metrics
print("Optimized metrics:")
print(metrics_df_opt.sort_values(['input_variable', 'output_variable', 'metric']))
print()
print("Original metrics:")
print(metrics_df_orig.sort_values(['input_variable', 'output_variable', 'metric']))
print()

# Calculate differences
metrics_merged = metrics_df_opt.merge(
    metrics_df_orig,
    on=['input_variable', 'output_variable', 'metric'],
    suffixes=('_opt', '_orig')
)
metrics_merged['diff'] = metrics_merged['value_opt'] - metrics_merged['value_orig']
metrics_merged['rel_diff'] = np.abs(metrics_merged['diff'] / metrics_merged['value_orig']) * 100

print("Differences:")
print(metrics_merged[['input_variable', 'output_variable', 'metric', 'value_opt', 'value_orig', 'diff', 'rel_diff']])
print()
print(f"Max absolute difference: {metrics_merged['diff'].abs().max():.2e}")
print(f"Max relative difference: {metrics_merged['rel_diff'].max():.2f}%")

# Check individual perturbation data for one combination
print("\nChecking perturbation data for input1 -> output1:")
pert_opt = perturbation_df_opt[
    (perturbation_df_opt.input_variable == 'input1') & 
    (perturbation_df_opt.output_variable == 'output1')
].head(10)
pert_orig = perturbation_df_orig[
    (perturbation_df_orig.input_variable == 'input1') & 
    (perturbation_df_orig.output_variable == 'output1')
].head(10)

print("\nOptimized (first 10 rows):")
print(pert_opt[['input_perturbed', 'output_perturbed']])
print("\nOriginal (first 10 rows):")
print(pert_orig[['input_perturbed', 'output_perturbed']])
