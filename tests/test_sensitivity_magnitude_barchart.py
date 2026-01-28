"""
Tests for sensitivity_magnitude_barchart module.
"""

import pytest
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid display
import matplotlib.pyplot as plt
from monte_carlo_sensitivity.sensitivity_magnitude_barchart import sensitivity_magnitude_barchart


class TestSensitivityMagnitudeBarchart:
    """Tests for sensitivity_magnitude_barchart function."""

    def test_sensitivity_magnitude_barchart_single_output(self, tmp_path, monkeypatch):
        """Test barchart with single output variable."""
        # Change to temp directory to avoid creating files in project root
        monkeypatch.chdir(tmp_path)
        
        # Create sample sensitivity metrics DataFrame
        df = pd.DataFrame({
            'output_variable': ['y', 'y', 'y'],
            'input_variable': ['x1', 'x2', 'x3'],
            'metric': ['mean_normalized_change', 'mean_normalized_change', 'mean_normalized_change'],
            'value': [0.5, 0.3, 0.1]
        })
        
        # Mock plt.show() to prevent display
        monkeypatch.setattr(plt, 'show', lambda: None)
        
        # Should execute without error
        sensitivity_magnitude_barchart(
            model_name='test_model',
            output_variable='y',
            df=df,
            metric='mean_normalized_change'
        )
        
        # Verify files were created
        assert (tmp_path / 'test_model Sensitivity Magnitude Multi-Panel.jpeg').exists()
        assert (tmp_path / 'test_model Sensitivity Magnitude Multi-Panel.svg').exists()
        
        # Clean up
        plt.close('all')

    def test_sensitivity_magnitude_barchart_multiple_outputs(self, tmp_path, monkeypatch):
        """Test barchart with multiple output variables."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)
        
        # Create sample sensitivity metrics DataFrame with two outputs
        df = pd.DataFrame({
            'output_variable': ['y1', 'y1', 'y2', 'y2'],
            'input_variable': ['x1', 'x2', 'x1', 'x2'],
            'metric': ['mean_normalized_change'] * 4,
            'value': [0.6, 0.4, 0.3, 0.7]
        })
        
        # Mock plt.show()
        monkeypatch.setattr(plt, 'show', lambda: None)
        
        # Should execute without error
        sensitivity_magnitude_barchart(
            model_name='test_model_multi',
            output_variable=['y1', 'y2'],
            df=df,
            metric='mean_normalized_change'
        )
        
        # Verify files were created
        assert (tmp_path / 'test_model_multi Sensitivity Magnitude Multi-Panel.jpeg').exists()
        assert (tmp_path / 'test_model_multi Sensitivity Magnitude Multi-Panel.svg').exists()
        
        # Clean up
        plt.close('all')

    def test_sensitivity_magnitude_barchart_output_as_list(self, tmp_path, monkeypatch):
        """Test that single output as list works same as string."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)
        
        # Create sample sensitivity metrics DataFrame
        df = pd.DataFrame({
            'output_variable': ['y', 'y'],
            'input_variable': ['x1', 'x2'],
            'metric': ['mean_normalized_change', 'mean_normalized_change'],
            'value': [0.5, 0.3]
        })
        
        # Mock plt.show()
        monkeypatch.setattr(plt, 'show', lambda: None)
        
        # Should execute without error when output_variable is a list
        sensitivity_magnitude_barchart(
            model_name='test_list',
            output_variable=['y'],
            df=df,
            metric='mean_normalized_change'
        )
        
        # Verify files were created
        assert (tmp_path / 'test_list Sensitivity Magnitude Multi-Panel.jpeg').exists()
        assert (tmp_path / 'test_list Sensitivity Magnitude Multi-Panel.svg').exists()
        
        # Clean up
        plt.close('all')
