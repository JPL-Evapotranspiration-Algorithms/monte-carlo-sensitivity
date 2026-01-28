"""
Demonstration of optimization with a simulated expensive forward process.
"""
import time
import numpy as np
import pandas as pd
from monte_carlo_sensitivity import sensitivity_analysis

# Simulate an expensive forward process (e.g., climate model, CFD simulation)
class ExpensiveModel:
    def __init__(self, computation_time=0.1):
        """
        Args:
            computation_time: Simulated time in seconds each forward call takes
        """
        self.computation_time = computation_time
        self.call_count = 0
        self.total_time = 0
        
    def forward_process(self, df):
        """Simulates an expensive computation."""
        self.call_count += 1
        
        # Simulate expensive computation
        time.sleep(self.computation_time)
        
        # Actual computation (simple for demo)
        result = df.copy()
        result['efficiency'] = (df['temperature'] * 0.5 + 
                                df['pressure'] * 0.3 - 
                                df['humidity'] * 0.2)
        result['cost'] = (df['temperature'] ** 2 * 10 + 
                         df['pressure'] * 50)
        result['quality'] = np.exp(df['temperature'] / 100) * df['pressure']
        
        self.total_time += self.computation_time
        return result
    
    def reset(self):
        self.call_count = 0
        self.total_time = 0

# Create sample input data
np.random.seed(42)
input_df = pd.DataFrame({
    'temperature': np.random.uniform(20, 30, 50),
    'pressure': np.random.uniform(1, 3, 50),
    'humidity': np.random.uniform(30, 70, 50)
})

input_variables = ['temperature', 'pressure', 'humidity']
output_variables = ['efficiency', 'cost', 'quality']

print("=" * 80)
print("MONTE CARLO SENSITIVITY ANALYSIS WITH EXPENSIVE FORWARD PROCESS")
print("=" * 80)
print(f"Input data: {len(input_df)} rows")
print(f"Variables: {len(input_variables)} inputs × {len(output_variables)} outputs = {len(input_variables) * len(output_variables)} combinations")
print(f"Perturbations per variable: 50")
print(f"Simulated forward process time: 0.1 seconds per call")
print()

# Test optimized version
print("Running OPTIMIZED version (use_joint_run=True)...")
model_opt = ExpensiveModel(computation_time=0.1)
start = time.time()
perturbation_df_opt, metrics_df_opt = sensitivity_analysis(
    input_df=input_df,
    input_variables=input_variables,
    output_variables=output_variables,
    forward_process=model_opt.forward_process,
    n=50,
    use_joint_run=True
)
elapsed_opt = time.time() - start

print(f"✓ Complete!")
print(f"  Forward process calls: {model_opt.call_count}")
print(f"  Simulated computation time: {model_opt.total_time:.1f} seconds")
print(f"  Total elapsed time: {elapsed_opt:.1f} seconds")
print()

# Test original version  
print("Running ORIGINAL version (use_joint_run=False)...")
model_orig = ExpensiveModel(computation_time=0.1)
start = time.time()
perturbation_df_orig, metrics_df_orig = sensitivity_analysis(
    input_df=input_df,
    input_variables=input_variables,
    output_variables=output_variables,
    forward_process=model_orig.forward_process,
    n=50,
    use_joint_run=False
)
elapsed_orig = time.time() - start

print(f"✓ Complete!")
print(f"  Forward process calls: {model_orig.call_count}")
print(f"  Simulated computation time: {model_orig.total_time:.1f} seconds")
print(f"  Total elapsed time: {elapsed_orig:.1f} seconds")
print()

# Summary
print("=" * 80)
print("PERFORMANCE COMPARISON")
print("=" * 80)
print(f"Computation time saved: {model_orig.total_time - model_opt.total_time:.1f} seconds")
print(f"Forward process call reduction: {model_orig.call_count} → {model_opt.call_count} ({100 * (1 - model_opt.call_count / model_orig.call_count):.1f}% reduction)")
print(f"Speedup: {elapsed_orig / elapsed_opt:.1f}x faster")
print()
print("For a 10-minute forward process:")
print(f"  Original would take: {model_orig.call_count * 10 / 60:.1f} hours")
print(f"  Optimized takes only: {model_opt.call_count * 10 / 60:.1f} hours")
print(f"  Time saved: {(model_orig.call_count - model_opt.call_count) * 10 / 60:.1f} hours")
print()

# Show sample metrics
print("=" * 80)
print("SAMPLE SENSITIVITY METRICS (Optimized Version)")
print("=" * 80)
# Show correlations
correlations = metrics_df_opt[metrics_df_opt['metric'] == 'correlation'].pivot(
    index='output_variable',
    columns='input_variable', 
    values='value'
)
print("\nCorrelation (input → output sensitivity):")
print(correlations.to_string())
print()
print("Higher absolute values indicate stronger sensitivity relationship.")
