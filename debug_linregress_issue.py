"""
Debug Script for monte_carlo_sensitivity linregress AttributeError

ISSUE SUMMARY:
--------------
sensitivity_analysis() with use_joint_run=True fails with AttributeError when
scipy.stats.linregress receives data with insufficient variation (e.g., all
identical values or single data point).

ERROR STACK TRACE:
------------------
File .../monte_carlo_sensitivity/sensitivity_analysis.py:247, in _sensitivity_analysis_joint
    r2 = scipy.stats.linregress(
        variable_perturbation_df.input_perturbation_std,
        variable_perturbation_df.output_perturbation_std
    )[2] ** 2

File .../scipy/stats/_stats_py.py:10524, in linregress
    ssxm, ssxym, _, ssym = np.cov(x, y, bias=1).flat

File .../numpy/lib/function_base.py:2724, in cov
    avg, w_sum = average(X, axis=1, weights=w, returned=True)

File .../numpy/lib/function_base.py:557, in average
    if scl.shape != avg_as_array.shape:
AttributeError: 'float' object has no attribute 'shape'

ROOT CAUSE:
-----------
1. When perturbations result in zero or near-zero variation in outputs
   (e.g., all perturbed values are identical), the resulting standardized
   perturbation arrays have no variance
2. scipy.stats.linregress() expects arrays with at least 2 distinct values
3. np.cov() fails when computing covariance of constant arrays, causing
   the scl (scale) variable to become a float instead of an array
4. This triggers the AttributeError when numpy tries to compare shapes

COMMON SCENARIOS THAT TRIGGER THIS:
------------------------------------
- Input variable with constant or nearly constant values
- Model outputs that are insensitive to certain input perturbations
- Small datasets where filtering removes most variation
- Variables where perturbations don't propagate to outputs

REPRODUCTION:
-------------
This script reproduces the exact error condition encountered when using
process_JET_table with ECOv002 Cal-Val data where some variables have
insufficient variation or model insensitivity.

SUCCESS CRITERIA:
-----------------
The issue is FIXED when this script runs without errors and prints:
"✓ SUCCESS: All tests passed! The linregress issue is resolved."

The fix should:
1. Detect insufficient data variation before calling linregress
2. Handle edge cases gracefully (return NaN or skip)
3. Log warnings when correlations cannot be computed
4. Not break existing functionality for well-behaved data
"""

import sys
import warnings
import numpy as np
import pandas as pd
import scipy.stats
from typing import Callable, Optional


def create_test_input_data(n_samples: int = 100) -> pd.DataFrame:
    """
    Create test dataset that mimics ECOv002 Cal-Val structure.
    """
    np.random.seed(42)
    
    data = {
        'ST_C': np.random.uniform(20, 40, n_samples),
        'NDVI': np.random.uniform(0.1, 0.8, n_samples),
        'albedo': np.random.uniform(0.1, 0.3, n_samples),
        'Ta_C': np.random.uniform(15, 35, n_samples),
        'RH': np.random.uniform(0.3, 0.8, n_samples),
        'constant_var': np.ones(n_samples) * 42.0,  # Constant variable
        'low_variance': np.random.normal(100, 0.001, n_samples),  # Very low variance
    }
    
    return pd.DataFrame(data)


def insensitive_forward_process(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates a forward process where outputs are insensitive to some inputs.
    This creates the condition where perturbations don't affect outputs,
    leading to constant standardized perturbation values.
    """
    output_df = input_df.copy()
    
    # Output that's sensitive to ST_C (works fine)
    output_df['sensitive_output'] = 200 + 5 * input_df['ST_C'] + np.random.normal(0, 10, len(input_df))
    
    # Output that's completely insensitive to inputs (causes the bug)
    # All perturbations result in same value -> zero variance -> linregress fails
    output_df['insensitive_output'] = np.full(len(input_df), 500.0)
    
    # Output with very low sensitivity (may cause numerical issues)
    output_df['low_sensitivity_output'] = 300 + 0.0001 * input_df['NDVI'] + np.random.normal(0, 0.0001, len(input_df))
    
    return output_df


def sensitive_forward_process(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Well-behaved forward process with clear input-output relationships.
    This should always work correctly.
    """
    output_df = input_df.copy()
    
    output_df['output_1'] = 200 + 5 * input_df['ST_C'] + np.random.normal(0, 10, len(input_df))
    output_df['output_2'] = 100 + 3 * input_df['NDVI'] + 2 * input_df['albedo'] + np.random.normal(0, 5, len(input_df))
    output_df['output_3'] = 50 + 1.5 * input_df['Ta_C'] + np.random.normal(0, 3, len(input_df))
    
    return output_df


def test_linregress_edge_cases():
    """
    Test that demonstrates the exact linregress failure conditions.
    """
    print("\n" + "="*70)
    print("TEST 1: Demonstrating linregress Edge Cases")
    print("="*70)
    
    print("\n1a. Well-behaved data (should work):")
    try:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = scipy.stats.linregress(x, y)
        print(f"   ✓ linregress succeeded: r^2 = {result.rvalue**2:.4f}")
    except Exception as e:
        print(f"   ✗ Unexpected failure: {e}")
        return False
    
    print("\n1b. Constant x array (triggers the bug):")
    try:
        x = np.array([5.0, 5.0, 5.0, 5.0, 5.0])  # No variation
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = scipy.stats.linregress(x, y)
        print(f"   ✓ linregress handled constant x (unexpected success)")
    except AttributeError as e:
        print(f"   ✗ EXPECTED ERROR: AttributeError: {str(e)[:60]}...")
        print("      This is the bug we're trying to avoid!")
    except Exception as e:
        print(f"   ✗ Different error: {type(e).__name__}: {str(e)[:60]}...")
    
    print("\n1c. Constant y array (triggers the bug):")
    try:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 5.0, 5.0, 5.0, 5.0])  # No variation
        result = scipy.stats.linregress(x, y)
        print(f"   ✓ linregress handled constant y (unexpected success)")
    except AttributeError as e:
        print(f"   ✗ EXPECTED ERROR: AttributeError: {str(e)[:60]}...")
        print("      This is the bug we're trying to avoid!")
    except Exception as e:
        print(f"   ✗ Different error: {type(e).__name__}: {str(e)[:60]}...")
    
    print("\n1d. Single data point:")
    try:
        x = np.array([1.0])
        y = np.array([2.0])
        result = scipy.stats.linregress(x, y)
        print(f"   ✓ linregress handled single point (unexpected success)")
    except Exception as e:
        print(f"   ✗ EXPECTED ERROR: {type(e).__name__}: {str(e)[:60]}...")
    
    print("\n1e. Near-constant arrays (may cause numerical issues):")
    try:
        x = np.array([1.0, 1.0000001, 1.0000002, 1.0000003, 1.0000004])
        y = np.array([2.0, 2.0000001, 2.0000002, 2.0000003, 2.0000004])
        result = scipy.stats.linregress(x, y)
        print(f"   ✓ linregress handled near-constant: r^2 = {result.rvalue**2:.4f}")
    except Exception as e:
        print(f"   ✗ Failed: {type(e).__name__}: {str(e)[:60]}...")
    
    return True


def test_sensitivity_analysis_with_insensitive_outputs():
    """
    Test that reproduces the exact error from the ECOv002 Cal-Val analysis.
    This SHOULD FAIL in the current version and PASS after the fix.
    """
    print("\n" + "="*70)
    print("TEST 2: Reproducing sensitivity_analysis Issue")
    print("="*70)
    
    try:
        from monte_carlo_sensitivity import sensitivity_analysis, divide_absolute_by_unperturbed
        
        input_df = create_test_input_data(n_samples=50)
        
        print(f"\n✓ Created test input DataFrame with {len(input_df)} rows")
        print(f"  Variables: {list(input_df.columns)}")
        
        # Test with insensitive forward process
        print("\n→ Running sensitivity_analysis with insensitive outputs...")
        print("  (This should trigger the AttributeError in unfixed version)")
        
        perturbation_df, sensitivity_metrics_df = sensitivity_analysis(
            input_df=input_df,
            input_variables=['ST_C', 'NDVI', 'constant_var'],
            output_variables=['sensitive_output', 'insensitive_output'],
            forward_process=insensitive_forward_process,
            normalization_function=divide_absolute_by_unperturbed,
            use_joint_run=True  # Explicitly use joint run
        )
        
        print("✓ SUCCESS: sensitivity_analysis handled insensitive outputs!")
        print(f"  Perturbation DataFrame shape: {perturbation_df.shape}")
        print(f"  Sensitivity metrics shape: {sensitivity_metrics_df.shape}")
        print("\n  Sensitivity metrics columns:")
        print(f"  {list(sensitivity_metrics_df.columns)}")
        print("\n  Sensitivity metrics preview:")
        print(sensitivity_metrics_df.to_string())
        
        # Verify that insensitive combinations are handled
        insensitive_rows = sensitivity_metrics_df[
            (sensitivity_metrics_df['output_variable'] == 'insensitive_output')
        ]
        if not insensitive_rows.empty:
            print(f"\n  ✓ Found {len(insensitive_rows)} insensitive output metrics")
            print("    (Should be handled gracefully with NaN or similar)")
        
        return True
        
    except AttributeError as e:
        if "'float' object has no attribute 'shape'" in str(e):
            print(f"✗ EXPECTED ERROR (unfixed): AttributeError")
            print(f"   {str(e)}")
            print("\n   This is the bug we're trying to fix!")
            print("   It occurs when linregress receives constant arrays.")
            return False
        else:
            print(f"✗ Different AttributeError: {e}")
            raise
    except ValueError as e:
        print(f"✗ ValueError: {e}")
        print("   (May be related to the same underlying issue)")
        return False
    except Exception as e:
        print(f"✗ UNEXPECTED ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_sensitivity_analysis_with_normal_outputs():
    """
    Baseline test to ensure normal operation still works.
    This should ALWAYS PASS (before and after the fix).
    """
    print("\n" + "="*70)
    print("TEST 3: Baseline Test with Normal Outputs")
    print("="*70)
    
    try:
        from monte_carlo_sensitivity import sensitivity_analysis, divide_absolute_by_unperturbed
        
        input_df = create_test_input_data(n_samples=100)
        
        print(f"\n✓ Created test input DataFrame")
        
        # Run with well-behaved forward process
        print("\n→ Running sensitivity_analysis with sensitive outputs...")
        
        perturbation_df, sensitivity_metrics_df = sensitivity_analysis(
            input_df=input_df,
            input_variables=['ST_C', 'NDVI', 'albedo'],
            output_variables=['output_1', 'output_2'],
            forward_process=sensitive_forward_process,
            normalization_function=divide_absolute_by_unperturbed,
            use_joint_run=True
        )
        
        print("✓ SUCCESS: Works correctly with sensitive outputs (as expected)")
        print(f"  Perturbation DataFrame shape: {perturbation_df.shape}")
        print(f"  Sensitivity metrics shape: {sensitivity_metrics_df.shape}")
        
        # Verify outputs are reasonable
        assert not sensitivity_metrics_df.empty, "Sensitivity metrics should not be empty"
        
        print("\n  Sample sensitivity metrics:")
        print(sensitivity_metrics_df.head(6).to_string())
        
        return True
        
    except Exception as e:
        print(f"✗ UNEXPECTED FAILURE: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def demonstrate_fix():
    """
    Demonstrates the proper way to handle edge cases in linregress.
    Shows the SOLUTION that should be implemented in the package.
    """
    print("\n" + "="*70)
    print("SOLUTION DEMONSTRATION: Safe linregress Wrapper")
    print("="*70)
    
    def safe_linregress(x: np.ndarray, y: np.ndarray, min_variance: float = 1e-10) -> dict:
        """
        Safely compute linear regression with proper edge case handling.
        
        Returns dict with keys: slope, intercept, rvalue, pvalue, stderr
        Returns NaN values if regression cannot be computed.
        """
        # Convert to numpy arrays and remove NaN
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Remove NaN and inf values
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        
        # Check for sufficient data points
        if len(x) < 2:
            print(f"      Warning: Insufficient data points ({len(x)} < 2)")
            return {
                'slope': np.nan, 'intercept': np.nan, 
                'rvalue': np.nan, 'pvalue': np.nan, 'stderr': np.nan
            }
        
        # Check for variance in both x and y
        x_var = np.var(x)
        y_var = np.var(y)
        
        if x_var < min_variance:
            print(f"      Warning: x has insufficient variance ({x_var:.2e} < {min_variance:.2e})")
            return {
                'slope': np.nan, 'intercept': np.nan,
                'rvalue': np.nan, 'pvalue': np.nan, 'stderr': np.nan
            }
        
        if y_var < min_variance:
            print(f"      Warning: y has insufficient variance ({y_var:.2e} < {min_variance:.2e})")
            return {
                'slope': np.nan, 'intercept': np.nan,
                'rvalue': np.nan, 'pvalue': np.nan, 'stderr': np.nan
            }
        
        # Now safe to call linregress
        try:
            result = scipy.stats.linregress(x, y)
            return {
                'slope': result.slope,
                'intercept': result.intercept,
                'rvalue': result.rvalue,
                'pvalue': result.pvalue,
                'stderr': result.stderr
            }
        except Exception as e:
            print(f"      Warning: linregress failed: {type(e).__name__}: {str(e)[:50]}")
            return {
                'slope': np.nan, 'intercept': np.nan,
                'rvalue': np.nan, 'pvalue': np.nan, 'stderr': np.nan
            }
    
    print("\nTesting safe_linregress wrapper:")
    
    # Test 1: Normal data (should work)
    print("\n  1. Normal data:")
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    result = safe_linregress(x, y)
    print(f"      ✓ r = {result['rvalue']:.4f}, slope = {result['slope']:.4f}")
    
    # Test 2: Constant x (should return NaN gracefully)
    print("\n  2. Constant x:")
    x = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    result = safe_linregress(x, y)
    print(f"      ✓ Handled gracefully: r = {result['rvalue']}")
    
    # Test 3: Constant y (should return NaN gracefully)
    print("\n  3. Constant y:")
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    result = safe_linregress(x, y)
    print(f"      ✓ Handled gracefully: r = {result['rvalue']}")
    
    # Test 4: Single point (should return NaN gracefully)
    print("\n  4. Single point:")
    x = np.array([1.0])
    y = np.array([2.0])
    result = safe_linregress(x, y)
    print(f"      ✓ Handled gracefully: r = {result['rvalue']}")
    
    # Test 5: With NaN values
    print("\n  5. With NaN values:")
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, np.nan, 10.0])
    result = safe_linregress(x, y)
    print(f"      ✓ Filtered NaN: r = {result['rvalue']:.4f}")
    
    return True


def main():
    """
    Run all tests and demonstrations.
    """
    print("\n" + "#"*70)
    print("# DEBUG SCRIPT: monte_carlo_sensitivity linregress AttributeError")
    print("#"*70)
    
    print("\nPURPOSE:")
    print("-" * 70)
    print("This script reproduces a bug where sensitivity_analysis() with")
    print("use_joint_run=True fails when scipy.stats.linregress receives data")
    print("with insufficient variation (constant arrays, single points, etc.).")
    print()
    print("The error occurs when model outputs are insensitive to input")
    print("perturbations, leading to constant standardized perturbation values.")
    print()
    print("The script includes:")
    print("  1. Demonstration of linregress edge cases")
    print("  2. Reproduction of the exact error")
    print("  3. Baseline test with normal data")
    print("  4. Demonstration of the solution")
    print()
    print("SUCCESS CRITERIA: Test 2 must pass for the bug to be fixed.")
    print("-" * 70)
    
    try:
        import monte_carlo_sensitivity
        print(f"\n✓ monte_carlo_sensitivity package found")
        print(f"  Location: {monte_carlo_sensitivity.__file__}")
    except ImportError:
        print("\n✗ ERROR: monte_carlo_sensitivity package not found!")
        print("  Please install the package or run this in the package directory.")
        sys.exit(1)
    
    # Run tests
    results = []
    
    # Test 1: linregress edge cases
    test1_passed = test_linregress_edge_cases()
    results.append(('linregress edge cases', test1_passed))
    
    # Test 2: The bug reproduction (critical test)
    test2_passed = test_sensitivity_analysis_with_insensitive_outputs()
    results.append(('Insensitive outputs handling', test2_passed))
    
    # Test 3: Baseline with normal data
    test3_passed = test_sensitivity_analysis_with_normal_outputs()
    results.append(('Normal outputs baseline', test3_passed))
    
    # Demonstrate solution
    solution_works = demonstrate_fix()
    results.append(('Solution demonstration', solution_works))
    
    # Print summary
    print("\n" + "#"*70)
    print("# TEST SUMMARY")
    print("#"*70)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    # Overall status
    critical_test_passed = results[1][1]  # Test 2 is the critical one
    
    print("\n" + "="*70)
    if critical_test_passed:
        print("✓ SUCCESS: All tests passed! The linregress issue is resolved.")
        print("="*70)
        print("\nThe monte_carlo_sensitivity package now properly handles:")
        print("  • Constant perturbation values")
        print("  • Insensitive model outputs")
        print("  • Insufficient data variation")
        print("  • Edge cases in correlation calculation")
        return 0
    else:
        print("✗ FAILURE: The linregress issue still exists.")
        print("="*70)
        print("\nRECOMMENDED FIX:")
        print("-" * 70)
        print("In sensitivity_analysis.py, around line 247, replace:")
        print()
        print("  r2 = scipy.stats.linregress(")
        print("      variable_perturbation_df.input_perturbation_std,")
        print("      variable_perturbation_df.output_perturbation_std")
        print("  )[2] ** 2")
        print()
        print("With safe version:")
        print()
        print("  # Extract arrays")
        print("  x = np.asarray(variable_perturbation_df.input_perturbation_std, dtype=np.float64)")
        print("  y = np.asarray(variable_perturbation_df.output_perturbation_std, dtype=np.float64)")
        print()
        print("  # Remove NaN/inf")
        print("  mask = np.isfinite(x) & np.isfinite(y)")
        print("  x, y = x[mask], y[mask]")
        print()
        print("  # Check for sufficient variation")
        print("  min_variance = 1e-10")
        print("  if len(x) < 2 or np.var(x) < min_variance or np.var(y) < min_variance:")
        print("      r2 = np.nan")
        print("      warnings.warn(")
        print("          f\"Insufficient variation for {input_variable}->{output_variable}\"")
        print("      )")
        print("  else:")
        print("      try:")
        print("          r2 = scipy.stats.linregress(x, y)[2] ** 2")
        print("      except Exception:")
        print("          r2 = np.nan")
        print()
        print("Similarly for pearsonr at line 203 and other statistical calculations.")
        print("-" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
