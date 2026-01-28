# Unit Tests for monte-carlo-sensitivity

This directory contains comprehensive unit tests for the `monte-carlo-sensitivity` package.

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run tests with coverage
```bash
pytest tests/ -v --cov=monte_carlo_sensitivity --cov-report=term-missing --cov-report=html
```

Or use the makefile targets:
```bash
make test          # Run tests verbosely
make test-cov      # Run with coverage report
make test-fast     # Skip slow tests
make test-verbose  # Extra verbose output
```

## Test Structure

### Test Files

- **test_import_monte_carlo_sensitivity.py** - Verifies package can be imported
- **test_import_dependencies.py** - Validates required dependencies are available
- **conftest.py** - Shared pytest fixtures and test data
- **test_repeat_rows.py** - Tests for the DataFrame row duplication utility
- **test_normalization.py** - Tests for normalization functions:
  - `divide_by_std` - Normalize by standard deviation
  - `divide_by_unperturbed` - Normalize by baseline values
  - `divide_absolute_by_unperturbed` - Normalize absolute values by baseline
- **test_perturbed_run.py** - Tests for the core univariate sensitivity analysis function
- **test_sensitivity_analysis.py** - Tests for the high-level orchestrator function
- **test_joint_perturbed_run.py** - Tests for multivariate sensitivity analysis

### Fixtures (conftest.py)

The `conftest.py` file provides reusable test fixtures:

**DataFrames:**
- `simple_dataframe` - Clean DataFrame with no missing values
- `dataframe_with_nans` - DataFrame containing NaN values
- `dataframe_with_zeros` - DataFrame containing zero values

**Forward Process Functions:**
- `linear_forward_process` - Simple linear transformation: y = 2*x + 1
- `identity_forward_process` - Returns input unchanged
- `quadratic_forward_process` - Quadratic transformation: y = x²
- `multivar_forward_process` - Multi-variable: z = 2*x + 3*y

**NumPy Arrays:**
- `sample_array_normal` - Standard numeric array
- `sample_array_with_zeros` - Array with zero values
- `sample_array_with_nans` - Array with NaN values
- `sample_array_with_inf` - Array with infinite values

**Utilities:**
- `random_seed` - Sets numpy random seed for reproducible tests

## Test Coverage

Current test coverage focuses on:

1. **Utility Functions** (repeat_rows, normalization)
   - Basic functionality
   - Edge cases (zeros, NaNs, infinities)
   - Data type handling

2. **Core Functions** (perturbed_run)
   - Output structure validation
   - Mathematical correctness
   - Parameter handling
   - Reproducibility

3. **High-Level Functions** (sensitivity_analysis)
   - Multiple input/output combinations
   - Metrics calculation
   - Format validation

4. **Multivariate Analysis** (joint_perturbed_run)
   - Correlated perturbations
   - Multiple variables
   - Custom covariance matrices

## Known Implementation Details

When writing new tests, be aware of these implementation specifics:

1. **`divide_by_std`** uses `ddof=0` (population standard deviation), not `ddof=1` (sample std)
2. **`sensitivity_analysis`** returns metrics in "long" format with columns:
   - `input_variable`, `output_variable`, `metric`, `value`
   - Metrics are: `correlation`, `r2`, `mean_normalized_change`
3. **`repeat_rows`** uses `np.repeat` which may change data types (e.g., int64 → object)
4. **`perturbed_run`** always uses `perturbation_mean=0` internally (parameter is ignored)
5. **`joint_perturbed_run`** may fail with single output variables due to iteration issues

## Adding New Tests

When adding new tests:

1. Use existing fixtures when possible
2. Test both normal operation and edge cases
3. Use descriptive test names: `test_function_name_specific_behavior`
4. Include docstrings explaining what each test validates
5. Use appropriate numpy testing functions for array comparisons:
   - `np.testing.assert_array_almost_equal` for floating point arrays
   - `pd.testing.assert_frame_equal` for DataFrames
   - `pd.testing.assert_series_equal` for Series

## Test Organization

Tests are organized into classes by the function being tested:
```python
class TestFunctionName:
    def test_basic_functionality(self, fixture):
        """Test normal operation."""
        ...
    
    def test_edge_case_zeros(self, fixture):
        """Test behavior with zero values."""
        ...
    
    def test_edge_case_nans(self, fixture):
        """Test behavior with NaN values."""
        ...
```

## Coverage Reports

HTML coverage reports are generated in the `htmlcov/` directory. Open `htmlcov/index.html` in a browser to view detailed line-by-line coverage information.

## Continuous Integration

These tests are designed to run in CI/CD pipelines. Ensure all tests pass before merging code changes.
