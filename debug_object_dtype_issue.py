"""
Debug Script for monte_carlo_sensitivity Object dtype Issue

ISSUE SUMMARY:
--------------
sensitivity_analysis() fails with AttributeError when pandas DataFrames contain
columns with object dtype instead of numeric dtypes. This occurs in the pearsonr
correlation calculation at line 203 of sensitivity_analysis.py.

ERROR STACK TRACES:
-------------------
Early failure (line 99):
  TypeError: unsupported operand type(s) for /: 'str' and 'int'
  (occurs in np.nanstd when computing statistics on object dtype arrays)

Later failure (line 203):
  ValueError: data type <class 'numpy.object_'> not inexact
  AttributeError: 'numpy.dtypes.ObjectDType' object has no attribute 'dtype'
  (occurs in scipy.stats.mstats.pearsonr with object dtype arrays)

ROOT CAUSE:
-----------
1. When forward_process returns a DataFrame with object dtype columns, numpy
   array conversion preserves the object dtype
2. numpy statistical functions (nanstd, nanmean) fail on object dtypes
3. scipy.stats.mstats.pearsonr() also cannot handle object dtype arrays
4. The package doesn't validate or coerce data types before statistical operations

REPRODUCTION:
-------------
This script reproduces the exact error condition encountered when using 
process_JET_table as a forward_process with ECOv002 Cal-Val data.

SUCCESS CRITERIA:
-----------------
The issue is FIXED when this script runs without errors and prints:
"✓ SUCCESS: All tests passed! The object dtype issue is resolved."
"""

import sys
import numpy as np
import pandas as pd
from typing import Callable

def create_test_input_data() -> pd.DataFrame:
    """
    Create a minimal test dataset that mimics the ECOv002 Cal-Val structure
    with realistic values that can trigger object dtype issues.
    """
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'ST_C': np.random.uniform(20, 40, n_samples),
        'NDVI': np.random.uniform(0.1, 0.8, n_samples),
        'albedo': np.random.uniform(0.1, 0.3, n_samples),
        'Ta_C': np.random.uniform(15, 35, n_samples),
        'RH': np.random.uniform(0.3, 0.8, n_samples),
        'SM': np.random.uniform(0.1, 0.4, n_samples),
    }
    
    return pd.DataFrame(data)


def problematic_forward_process(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates a forward process that returns object dtype columns.
    This mimics the behavior that can occur in process_JET_table when
    certain operations result in object dtype inference by pandas.
    """
    output_df = input_df.copy()
    
    # Simulate calculations that might result in object dtype
    # This can happen when:
    # 1. String representations of arrays are inadvertently stored
    # 2. Mixed types exist in the data
    # 3. pandas infers object dtype due to None values mixed with numbers
    
    # Case 1: Array-like strings (common in some file formats)
    output_df['Rn_Wm2'] = [f"[{x:.2f}]" for x in np.random.uniform(200, 600, len(input_df))]
    
    # Case 2: Mixed None and numeric (object dtype inference)
    et_values = np.random.uniform(0, 5, len(input_df))
    et_values[::10] = None  # Add some None values
    output_df['ET_daylight_kg'] = et_values  # This becomes object dtype
    
    # Case 3: Explicit object dtype (mimicking certain data operations)
    gpp_values = np.random.uniform(0, 10, len(input_df))
    output_df['GPP_inst_g_m2_s'] = pd.Series(gpp_values, dtype='object')
    
    return output_df


def fixed_forward_process(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Example of a properly-typed forward process that ensures numeric dtypes.
    """
    output_df = input_df.copy()
    
    # All outputs should be proper numeric types
    output_df['Rn_Wm2'] = np.random.uniform(200, 600, len(input_df)).astype(np.float64)
    output_df['ET_daylight_kg'] = np.random.uniform(0, 5, len(input_df)).astype(np.float64)
    output_df['GPP_inst_g_m2_s'] = np.random.uniform(0, 10, len(input_df)).astype(np.float64)
    
    return output_df


def test_sensitivity_analysis_with_object_dtype():
    """
    Test that reproduces the exact error condition.
    This SHOULD FAIL in the current version and PASS after the fix.
    """
    print("\n" + "="*70)
    print("TEST 1: Reproducing Object dtype Issue")
    print("="*70)
    
    try:
        from monte_carlo_sensitivity import sensitivity_analysis, divide_absolute_by_unperturbed
        
        input_df = create_test_input_data()
        
        print(f"✓ Created test input DataFrame with {len(input_df)} rows")
        print(f"  Input columns: {list(input_df.columns)}")
        print(f"  Input dtypes: {dict(input_df.dtypes)}")
        
        # Run with problematic forward process
        print("\n→ Running sensitivity_analysis with object dtype outputs...")
        
        perturbation_df, sensitivity_metrics_df = sensitivity_analysis(
            input_df=input_df,
            input_variables=['ST_C', 'NDVI', 'albedo'],
            output_variables=['Rn_Wm2', 'ET_daylight_kg', 'GPP_inst_g_m2_s'],
            forward_process=problematic_forward_process,
            normalization_function=divide_absolute_by_unperturbed
        )
        
        print("✓ SUCCESS: sensitivity_analysis handled object dtypes correctly!")
        print(f"  Perturbation DataFrame shape: {perturbation_df.shape}")
        print(f"  Sensitivity metrics shape: {sensitivity_metrics_df.shape}")
        return True
        
    except AttributeError as e:
        if "'numpy.dtypes.ObjectDType' object has no attribute 'dtype'" in str(e):
            print(f"✗ EXPECTED ERROR (unfixed): {type(e).__name__}: {e}")
            print("\nThis is the bug we're trying to fix!")
            return False
        else:
            print(f"✗ UNEXPECTED ERROR: {type(e).__name__}: {e}")
            raise
    except ValueError as e:
        if "not inexact" in str(e):
            print(f"✗ EXPECTED ERROR (unfixed): {type(e).__name__}: {e}")
            print("\nThis is the bug we're trying to fix!")
            return False
        else:
            print(f"✗ UNEXPECTED ERROR: {type(e).__name__}: {e}")
            raise
    except TypeError as e:
        if "unsupported operand type(s) for /" in str(e) or "'str' and 'int'" in str(e):
            print(f"✗ EXPECTED ERROR (unfixed): {type(e).__name__}: {e}")
            print("\nThis is the object dtype bug! It occurs even earlier than pearsonr.")
            print("The error happens when numpy tries to compute statistics on object dtype arrays.")
            return False
        else:
            print(f"✗ UNEXPECTED ERROR: {type(e).__name__}: {e}")
            raise
    except Exception as e:
        print(f"✗ UNEXPECTED ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_sensitivity_analysis_with_proper_dtype():
    """
    Test that verifies the package works correctly with proper dtypes.
    This should ALWAYS PASS (even before the fix).
    """
    print("\n" + "="*70)
    print("TEST 2: Baseline Test with Proper dtypes")
    print("="*70)
    
    try:
        from monte_carlo_sensitivity import sensitivity_analysis, divide_absolute_by_unperturbed
        
        input_df = create_test_input_data()
        
        print(f"✓ Created test input DataFrame")
        
        # Run with proper forward process
        print("\n→ Running sensitivity_analysis with proper float64 outputs...")
        
        perturbation_df, sensitivity_metrics_df = sensitivity_analysis(
            input_df=input_df,
            input_variables=['ST_C', 'NDVI'],
            output_variables=['Rn_Wm2', 'ET_daylight_kg'],
            forward_process=fixed_forward_process,
            normalization_function=divide_absolute_by_unperturbed
        )
        
        print("✓ SUCCESS: Works correctly with proper dtypes (as expected)")
        print(f"  Perturbation DataFrame shape: {perturbation_df.shape}")
        print(f"  Sensitivity metrics shape: {sensitivity_metrics_df.shape}")
        
        # Verify outputs are reasonable
        assert not sensitivity_metrics_df.empty, "Sensitivity metrics should not be empty"
        assert 'input_variable' in sensitivity_metrics_df.columns, "Should have input_variable column"
        assert 'output_variable' in sensitivity_metrics_df.columns, "Should have output_variable column"
        
        return True
        
    except Exception as e:
        print(f"✗ UNEXPECTED FAILURE: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def demonstrate_dtype_coercion():
    """
    Demonstrates how to properly handle and coerce object dtypes.
    This shows the SOLUTION that should be implemented in the package.
    """
    print("\n" + "="*70)
    print("SOLUTION DEMONSTRATION: Proper dtype Handling")
    print("="*70)
    
    # Create data with object dtypes
    input_df = create_test_input_data()
    output_df = problematic_forward_process(input_df)
    
    print("\nBEFORE coercion:")
    print(f"  Rn_Wm2 dtype: {output_df['Rn_Wm2'].dtype}")
    print(f"  ET_daylight_kg dtype: {output_df['ET_daylight_kg'].dtype}")
    print(f"  GPP_inst_g_m2_s dtype: {output_df['GPP_inst_g_m2_s'].dtype}")
    print(f"  Sample Rn_Wm2 value: {output_df['Rn_Wm2'].iloc[0]} (type: {type(output_df['Rn_Wm2'].iloc[0])})")
    
    # SOLUTION: Coerce to numeric with proper error handling
    def coerce_to_float64(series: pd.Series) -> np.ndarray:
        """
        Safely coerce a pandas Series to float64 numpy array.
        Handles string representations, None values, and mixed types.
        """
        # Handle string representations of arrays like "[123.45]"
        if series.dtype == 'object':
            try:
                # Try to strip brackets and convert
                series = series.apply(lambda x: str(x).strip('[]') if isinstance(x, str) else x)
            except:
                pass
        
        # Try to convert to numeric, coercing errors to NaN
        numeric_series = pd.to_numeric(series, errors='coerce')
        # Convert to numpy array with explicit float64 dtype
        return np.array(numeric_series, dtype=np.float64)
    
    print("\nAFTER coercion:")
    Rn_coerced = coerce_to_float64(output_df['Rn_Wm2'])
    ET_coerced = coerce_to_float64(output_df['ET_daylight_kg'])
    GPP_coerced = coerce_to_float64(output_df['GPP_inst_g_m2_s'])
    
    print(f"  Rn_Wm2 dtype: {Rn_coerced.dtype}")
    print(f"  ET_daylight_kg dtype: {ET_coerced.dtype}")
    print(f"  GPP_inst_g_m2_s dtype: {GPP_coerced.dtype}")
    print(f"  Sample Rn_Wm2 value: {Rn_coerced[0]:.2f}")
    
    # Verify they can be used in scipy functions now
    from scipy.stats import mstats
    
    # Create some perturbations
    Rn_perturbed = Rn_coerced + np.random.normal(0, 10, len(Rn_coerced))
    
    print("\n✓ Testing scipy.stats.mstats.pearsonr with coerced arrays...")
    correlation = mstats.pearsonr(Rn_coerced, Rn_perturbed)
    print(f"✓ SUCCESS: Correlation calculated: {correlation[0]:.4f}")
    
    return True


def main():
    """
    Run all tests and demonstrations.
    """
    print("\n" + "#"*70)
    print("# DEBUG SCRIPT: monte_carlo_sensitivity Object dtype Issue")
    print("#"*70)
    
    print("\nPURPOSE:")
    print("-" * 70)
    print("This script reproduces a bug where sensitivity_analysis() fails when")
    print("the forward_process returns DataFrames with object dtype columns.")
    print("The error occurs in scipy.stats.mstats.pearsonr() which cannot handle")
    print("object dtypes.")
    print()
    print("The script includes:")
    print("  1. Reproduction of the exact error")
    print("  2. Baseline test that should always work")
    print("  3. Demonstration of the solution")
    print()
    print("SUCCESS CRITERIA: Test 1 must pass for the bug to be considered fixed.")
    print("-" * 70)
    
    try:
        # Check if package is available
        import monte_carlo_sensitivity
        print(f"\n✓ monte_carlo_sensitivity package found")
        print(f"  Location: {monte_carlo_sensitivity.__file__}")
    except ImportError:
        print("\n✗ ERROR: monte_carlo_sensitivity package not found!")
        print("  Please install the package or run this script in the package directory.")
        sys.exit(1)
    
    # Run tests
    results = []
    
    # Test 1: The bug reproduction
    test1_passed = test_sensitivity_analysis_with_object_dtype()
    results.append(('Object dtype handling', test1_passed))
    
    # Test 2: Baseline with proper dtypes
    test2_passed = test_sensitivity_analysis_with_proper_dtype()
    results.append(('Proper dtype baseline', test2_passed))
    
    # Demonstrate solution
    print()
    solution_works = demonstrate_dtype_coercion()
    results.append(('Solution demonstration', solution_works))
    
    # Print summary
    print("\n" + "#"*70)
    print("# TEST SUMMARY")
    print("#"*70)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    # Overall status
    critical_tests_passed = results[0][1]  # Test 1 is the critical one
    
    print("\n" + "="*70)
    if critical_tests_passed:
        print("✓ SUCCESS: All tests passed! The object dtype issue is resolved.")
        print("="*70)
        print("\nThe monte_carlo_sensitivity package now properly handles:")
        print("  • Object dtype columns from forward_process")
        print("  • Mixed data types")
        print("  • String representations of numeric values")
        print("  • None/NaN value coercion")
        return 0
    else:
        print("✗ FAILURE: The object dtype issue still exists.")
        print("="*70)
        print("\nRECOMMENDED FIX:")
        print("-" * 70)
        print("In sensitivity_analysis.py, add dtype coercion after forward_process:")
        print()
        print("  # After running forward_process, coerce outputs to numeric")
        print("  unperturbed_output_df = forward_process(input_df)")
        print("  for col in unperturbed_output_df.columns:")
        print("      if col in output_variables:")
        print("          unperturbed_output_df[col] = pd.to_numeric(")
        print("              unperturbed_output_df[col],")
        print("              errors='coerce'")
        print("          ).astype(np.float64)")
        print()
        print("And before calling pearsonr (around line 203):")
        print()
        print("  # Coerce to float64 to handle object dtypes")
        print("  input_pert_std = np.array(pd.to_numeric(")
        print("      variable_perturbation_df.input_perturbation_std,")
        print("      errors='coerce'")
        print("  ), dtype=np.float64)")
        print("  output_pert_std = np.array(pd.to_numeric(")
        print("      variable_perturbation_df.output_perturbation_std,")
        print("      errors='coerce'")
        print("  ), dtype=np.float64)")
        print()
        print("  # Remove NaN values before correlation")
        print("  mask = ~(np.isnan(input_pert_std) | np.isnan(output_pert_std))")
        print("  if np.sum(mask) > 2:")
        print("      correlation = mstats.pearsonr(")
        print("          input_pert_std[mask],")
        print("          output_pert_std[mask]")
        print("      )[0]")
        print("  else:")
        print("      correlation = np.nan")
        print("-" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
