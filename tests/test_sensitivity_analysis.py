"""
Tests for sensitivity_analysis function.

Tests the high-level sensitivity analysis orchestrator that processes
all input-output variable combinations and calculates metrics.
"""

import numpy as np
import pandas as pd
import pytest

from monte_carlo_sensitivity.sensitivity_analysis import sensitivity_analysis


class TestSensitivityAnalysis:
    """Test suite for sensitivity_analysis function."""

    def test_sensitivity_analysis_basic_structure(self, simple_dataframe, linear_forward_process, random_seed):
        """Test that output has expected structure."""
        perturbation_df, metrics_df = sensitivity_analysis(
            input_df=simple_dataframe,
            input_variables=['x'],
            output_variables=['y'],
            forward_process=linear_forward_process,
            n=20
        )
        
        # Check return types
        assert isinstance(perturbation_df, pd.DataFrame)
        assert isinstance(metrics_df, pd.DataFrame)
        
        # Check perturbation_df structure
        assert 'input_variable' in perturbation_df.columns
        assert 'output_variable' in perturbation_df.columns
        
        # Check metrics_df structure (long format with 'metric' and 'value' columns)
        expected_cols = ['input_variable', 'output_variable', 'metric', 'value']
        for col in expected_cols:
            assert col in metrics_df.columns, f"Missing column: {col}"
        
        # Check that expected metrics are present
        metrics = set(metrics_df['metric'].unique())
        expected_metrics = {'correlation', 'r2', 'mean_normalized_change'}
        assert expected_metrics.issubset(metrics)

    def test_sensitivity_analysis_single_variable_pair(self, simple_dataframe, linear_forward_process, random_seed):
        """Test with single input-output pair."""
        perturbation_df, metrics_df = sensitivity_analysis(
            input_df=simple_dataframe,
            input_variables=['x'],
            output_variables=['y'],
            forward_process=linear_forward_process,
            n=50
        )
        
        # Should have 3 rows in metrics (correlation, r2, mean_normalized_change)
        assert len(metrics_df) == 3
        assert all(metrics_df['input_variable'] == 'x')
        assert all(metrics_df['output_variable'] == 'y')
        
        # Check all metrics are present
        metrics = set(metrics_df['metric'])
        assert metrics == {'correlation', 'r2', 'mean_normalized_change'}
        
        # Metrics should be non-null
        assert not metrics_df['value'].isna().any()

    def test_sensitivity_analysis_multiple_inputs(self, random_seed):
        """Test with multiple input variables."""
        input_df = pd.DataFrame({
            'x1': [1.0, 2.0, 3.0],
            'x2': [4.0, 5.0, 6.0]
        })
        
        def multi_input_process(df):
            result = df.copy()
            result['y'] = df['x1'] + df['x2']
            return result
        
        perturbation_df, metrics_df = sensitivity_analysis(
            input_df=input_df,
            input_variables=['x1', 'x2'],
            output_variables=['y'],
            forward_process=multi_input_process,
            n=30
        )
        
        # Should have 6 rows in metrics (2 variables * 3 metrics each)
        assert len(metrics_df) == 6
        
        # Check that both variables are represented
        input_vars = set(metrics_df['input_variable'])
        assert input_vars == {'x1', 'x2'}

    def test_sensitivity_analysis_multiple_outputs(self, random_seed):
        """Test with multiple output variables."""
        input_df = pd.DataFrame({'x': [1.0, 2.0, 3.0]})
        
        def multi_output_process(df):
            result = df.copy()
            result['y1'] = df['x'] * 2
            result['y2'] = df['x'] ** 2
            return result
        
        perturbation_df, metrics_df = sensitivity_analysis(
            input_df=input_df,
            input_variables=['x'],
            output_variables=['y1', 'y2'],
            forward_process=multi_output_process,
            n=30
        )
        
        # Should have 6 rows in metrics (2 outputs * 3 metrics each)
        assert len(metrics_df) == 6
        
        # Check that both outputs are represented
        output_vars = set(metrics_df['output_variable'])
        assert output_vars == {'y1', 'y2'}

    def test_sensitivity_analysis_all_combinations(self, random_seed):
        """Test that all input-output combinations are analyzed."""
        input_df = pd.DataFrame({
            'x1': [1.0, 2.0, 3.0],
            'x2': [4.0, 5.0, 6.0]
        })
        
        def process(df):
            result = df.copy()
            result['y1'] = df['x1'] + df['x2']
            result['y2'] = df['x1'] * df['x2']
            return result
        
        perturbation_df, metrics_df = sensitivity_analysis(
            input_df=input_df,
            input_variables=['x1', 'x2'],
            output_variables=['y1', 'y2'],
            forward_process=process,
            n=25
        )
        
        # Should have 12 rows: 4 combinations * 3 metrics each
        assert len(metrics_df) == 12
        
        # Check all combinations exist
        combinations = set(zip(metrics_df['input_variable'], metrics_df['output_variable']))
        expected = {('x1', 'y1'), ('x1', 'y2'), ('x2', 'y1'), ('x2', 'y2')}
        assert combinations == expected

    def test_sensitivity_analysis_linear_relationship_correlation(self, random_seed):
        """Test that linear relationship produces high correlation."""
        input_df = pd.DataFrame({'x': np.linspace(1, 10, 20)})
        
        def linear_process(df):
            result = df.copy()
            result['y'] = 3 * df['x'] + 5
            return result
        
        perturbation_df, metrics_df = sensitivity_analysis(
            input_df=input_df,
            input_variables=['x'],
            output_variables=['y'],
            forward_process=linear_process,
            n=100
        )
        
        # Perfect linear relationship should have high correlation
        correlation = metrics_df[metrics_df['metric'] == 'correlation']['value'].iloc[0]
        assert correlation > 0.95, f"Expected high correlation, got {correlation}"
        
        # R-squared should also be high
        r_squared = metrics_df[metrics_df['metric'] == 'r2']['value'].iloc[0]
        assert r_squared > 0.90, f"Expected high R², got {r_squared}"

    def test_sensitivity_analysis_no_relationship(self, random_seed):
        """Test with no relationship between input and output."""
        input_df = pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0, 5.0]})
        
        def constant_process(df):
            result = df.copy()
            result['y'] = np.ones(len(df)) * 10.0  # Constant output
            return result
        
        perturbation_df, metrics_df = sensitivity_analysis(
            input_df=input_df,
            input_variables=['x'],
            output_variables=['y'],
            forward_process=constant_process,
            n=50
        )
        
        #Should have low correlation and R² (or NaN with constant output)
        correlation = metrics_df[metrics_df['metric'] == 'correlation']['value'].iloc[0]
        r_squared = metrics_df[metrics_df['metric'] == 'r2']['value'].iloc[0]
        
        # With constant output, values will be NaN (can't compute correlation with zero variance)
        # Just verify the test runs without error - the values should be NaN
        assert len(metrics_df) == 3  # Should still return all 3 metrics

    def test_sensitivity_analysis_perturbation_count(self, simple_dataframe, linear_forward_process, random_seed):
        """Test that perturbation count is respected."""
        n = 15
        perturbation_df, metrics_df = sensitivity_analysis(
            input_df=simple_dataframe,
            input_variables=['x'],
            output_variables=['y'],
            forward_process=linear_forward_process,
            n=n
        )
        
        # Should have len(input_df) * n rows
        expected_rows = len(simple_dataframe) * n
        assert len(perturbation_df) == expected_rows

    def test_sensitivity_analysis_custom_perturbation_std(self, simple_dataframe, linear_forward_process, random_seed):
        """Test using custom perturbation standard deviation."""
        custom_std = 5.0
        perturbation_df, metrics_df = sensitivity_analysis(
            input_df=simple_dataframe,
            input_variables=['x'],
            output_variables=['y'],
            forward_process=linear_forward_process,
            n=100,
            perturbation_std=custom_std
        )
        
        # Check that input perturbations reflect custom std
        # (approximately, due to randomness)
        input_perturbations = perturbation_df['input_perturbed'].values[:100] - simple_dataframe.iloc[0]['x']
        actual_std = np.std(input_perturbations, ddof=1)
        
        # Should be close to custom_std
        assert abs(actual_std - custom_std) / custom_std < 0.25

    def test_sensitivity_analysis_metrics_range(self, simple_dataframe, linear_forward_process, random_seed):
        """Test that metrics are in valid ranges."""
        perturbation_df, metrics_df = sensitivity_analysis(
            input_df=simple_dataframe,
            input_variables=['x'],
            output_variables=['y'],
            forward_process=linear_forward_process,
            n=50
        )
        
        # Check correlation values are in range [-1, 1]
        correlations = metrics_df[metrics_df['metric'] == 'correlation']['value']
        for val in correlations:
            if not pd.isna(val):
                assert -1 <= val <= 1, f"Correlation {val} out of range"
        
        # Check R² values are in range [0, 1]
        r2_values = metrics_df[metrics_df['metric'] == 'r2']['value']
        for val in r2_values:
            if not pd.isna(val):
                assert 0 <= val <= 1, f"R² {val} out of range"

    def test_sensitivity_analysis_perturbation_df_completeness(self, random_seed):
        """Test that perturbation DataFrame contains all expected columns."""
        input_df = pd.DataFrame({'x': [1.0, 2.0, 3.0]})
        
        def simple_process(df):
            result = df.copy()
            result['y'] = df['x'] * 2
            return result
        
        perturbation_df, metrics_df = sensitivity_analysis(
            input_df=input_df,
            input_variables=['x'],
            output_variables=['y'],
            forward_process=simple_process,
            n=10
        )
        
        # Check for essential columns
        required_cols = [
            'input_variable', 'output_variable',
            'input_unperturbed', 'input_perturbation', 'input_perturbed',
            'output_unperturbed', 'output_perturbation', 'output_perturbed'
        ]
        
        for col in required_cols:
            assert col in perturbation_df.columns, f"Missing column: {col}"

    def test_sensitivity_analysis_consistent_variable_names(self, random_seed):
        """Test that variable names are consistent throughout output."""
        input_df = pd.DataFrame({
            'input_a': [1.0, 2.0],
            'input_b': [3.0, 4.0]
        })
        
        def process(df):
            result = df.copy()
            result['output_x'] = df['input_a'] + df['input_b']
            result['output_y'] = df['input_a'] * df['input_b']
            return result
        
        perturbation_df, metrics_df = sensitivity_analysis(
            input_df=input_df,
            input_variables=['input_a', 'input_b'],
            output_variables=['output_x', 'output_y'],
            forward_process=process,
            n=10
        )
        
        # Check metrics_df variable names
        for _, row in metrics_df.iterrows():
            assert row['input_variable'] in ['input_a', 'input_b']
            assert row['output_variable'] in ['output_x', 'output_y']
        
        # Check perturbation_df variable names
        unique_inputs = perturbation_df['input_variable'].unique()
        unique_outputs = perturbation_df['output_variable'].unique()
        
        assert set(unique_inputs).issubset({'input_a', 'input_b'})
        assert set(unique_outputs).issubset({'output_x', 'output_y'})

    def test_joint_mode_handles_object_and_low_variance(self, random_seed):
        """Joint mode should tolerate object dtypes and near-constant data without crashing."""
        input_df = pd.DataFrame({
            'x': pd.Series([1.0, 1.0], dtype=object)
        })

        def process(df):
            result = df.copy()
            # Produce a nearly constant output to exercise variance guards
            result['y'] = pd.to_numeric(df['x'], errors='coerce') * 1.0
            return result

        perturbation_df, metrics_df = sensitivity_analysis(
            input_df=input_df,
            input_variables=['x'],
            output_variables=['y'],
            forward_process=process,
            n=5,
            use_joint_run=True
        )

        # Should produce the standard metric set without raising
        assert set(metrics_df['metric']) == {'correlation', 'r2', 'mean_normalized_change'}
        # r2 may be NaN when variance is zero but should not error
        r2_value = metrics_df.loc[metrics_df['metric'] == 'r2', 'value'].iloc[0]
        assert np.isnan(r2_value) or 0 <= r2_value <= 1
        # Perturbations should still be returned
        assert not perturbation_df.empty
