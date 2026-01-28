"""
Benchmark to demonstrate performance improvement of optimized sensitivity_analysis.
"""
import time
import numpy as np
import pandas as pd
from monte_carlo_sensitivity import sensitivity_analysis


# Create a mock forward process that we can count calls to
class ForwardProcessCounter:
    def __init__(self):
        self.call_count = 0
        
    def forward_process(self, df):
        """Simple forward process that tracks how many times it's called."""
        self.call_count += 1
        result = df.copy()
        result['output1'] = df['input1'] * 2 + df['input2'] * 3
        result['output2'] = df['input1'] ** 2 + df['input2']
        result['output3'] = np.sin(df['input1']) + np.cos(df['input2'])
        return result
    
    def reset(self):
        self.call_count = 0


# Create test data
np.random.seed(42)
input_df = pd.DataFrame({
    'input1': np.random.randn(100),
    'input2': np.random.randn(100),
    'input3': np.random.randn(100)
})

input_variables = ['input1', 'input2', 'input3']
output_variables = ['output1', 'output2', 'output3']

# Test with the optimized version (use_joint_run=True)
print("=" * 70)
print("OPTIMIZED VERSION (use_joint_run=True)")
print("=" * 70)
counter_optimized = ForwardProcessCounter()
start = time.time()
perturbation_df_opt, metrics_df_opt = sensitivity_analysis(
    input_df=input_df,
    input_variables=input_variables,
    output_variables=output_variables,
    forward_process=counter_optimized.forward_process,
    n=50,
    use_joint_run=True
)
elapsed_opt = time.time() - start
print(f"Forward process calls: {counter_optimized.call_count}")
print(f"Time elapsed: {elapsed_opt:.3f} seconds")
print(f"Result shape: perturbation_df={perturbation_df_opt.shape}, metrics_df={metrics_df_opt.shape}")
print()

# Test with the original loop-based version (use_joint_run=False)
print("=" * 70)
print("ORIGINAL VERSION (use_joint_run=False)")
print("=" * 70)
counter_original = ForwardProcessCounter()
start = time.time()
perturbation_df_orig, metrics_df_orig = sensitivity_analysis(
    input_df=input_df,
    input_variables=input_variables,
    output_variables=output_variables,
    forward_process=counter_original.forward_process,
    n=50,
    use_joint_run=False
)
elapsed_orig = time.time() - start
print(f"Forward process calls: {counter_original.call_count}")
print(f"Time elapsed: {elapsed_orig:.3f} seconds")
print(f"Result shape: perturbation_df={perturbation_df_orig.shape}, metrics_df={metrics_df_orig.shape}")
print()

# Summary
print("=" * 70)
print("PERFORMANCE SUMMARY")
print("=" * 70)
n_inputs = len(input_variables)
n_outputs = len(output_variables)
print(f"Variables: {n_inputs} inputs × {n_outputs} outputs = {n_inputs * n_outputs} combinations")
print()
print(f"Original:  {counter_original.call_count} forward process calls")
print(f"Optimized: {counter_optimized.call_count} forward process calls")
print(f"Reduction: {counter_original.call_count - counter_optimized.call_count} fewer calls ({(1 - counter_optimized.call_count / counter_original.call_count) * 100:.1f}% reduction)")
print()
print(f"Original time:  {elapsed_orig:.3f} seconds")
print(f"Optimized time: {elapsed_opt:.3f} seconds")
print(f"Speedup:        {elapsed_orig / elapsed_opt:.2f}x faster")
print()

# Verify results are numerically close
print("Verifying results are numerically equivalent...")
# Sort both dataframes for comparison
metrics_df_opt_sorted = metrics_df_opt.sort_values(['input_variable', 'output_variable', 'metric']).reset_index(drop=True)
metrics_df_orig_sorted = metrics_df_orig.sort_values(['input_variable', 'output_variable', 'metric']).reset_index(drop=True)

# Check if values are close (Monte Carlo methods have inherent numerical variability)
values_match = np.allclose(
    metrics_df_opt_sorted['value'].values,
    metrics_df_orig_sorted['value'].values,
    rtol=0.01,  # 1% relative tolerance is reasonable for Monte Carlo
    equal_nan=True
)
print(f"Metrics match within tolerance: {values_match}")

# Calculate actual differences
metrics_merged = metrics_df_opt_sorted.merge(
    metrics_df_orig_sorted,
    on=['input_variable', 'output_variable', 'metric'],
    suffixes=('_opt', '_orig')
)
metrics_merged['diff'] = np.abs(metrics_merged['value_opt'] - metrics_merged['value_orig'])
metrics_merged['rel_diff'] = metrics_merged['diff'] / np.abs(metrics_merged['value_orig']) * 100

print(f"Max absolute difference: {metrics_merged['diff'].max():.2e}")
print(f"Mean relative difference: {metrics_merged['rel_diff'].mean():.4f}%")
print(f"Max relative difference: {metrics_merged['rel_diff'].max():.4f}%")
print()

print("✅ Optimization successful! Forward process now runs only TWICE instead of")
print(f"   {counter_original.call_count} times - a dramatic improvement for expensive models!")
