"""
Tests for normalization functions.

Tests divide_by_std, divide_by_unperturbed, and divide_absolute_by_unperturbed
functions used to normalize perturbations in sensitivity analysis.
"""

import numpy as np
import pytest

from monte_carlo_sensitivity.divide_by_std import divide_by_std
from monte_carlo_sensitivity.divide_by_unperturbed import divide_by_unperturbed
from monte_carlo_sensitivity.divide_absolute_by_unperturbed import divide_absolute_by_unperturbed


class TestDivideByStd:
    """Test suite for divide_by_std normalization function."""

    def test_divide_by_std_basic(self):
        """Test basicnormalization by standard deviation."""
        perturbations = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        unperturbed = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        
        result = divide_by_std(perturbations, unperturbed)
        
        # Should divide by std of unperturbed values (population std, ddof=0)
        expected_std = np.std(unperturbed, ddof=0)
        expected = perturbations / expected_std
        
        np.testing.assert_array_almost_equal(result, expected)

    def test_divide_by_std_zero_std(self):
        """Test behavior when standard deviation is zero."""
        perturbations = np.array([1.0, 2.0, 3.0])
        unperturbed = np.array([5.0, 5.0, 5.0])  # Constant values, std=0
        
        result = divide_by_std(perturbations, unperturbed)
        
        # Division by zero should result in inf
        assert np.all(np.isinf(result))

    def test_divide_by_std_with_nans(self):
        """Test handling of NaN values."""
        perturbations = np.array([1.0, np.nan, 3.0, 4.0])
        unperturbed = np.array([10.0, 20.0, 30.0, 40.0])
        
        result = divide_by_std(perturbations, unperturbed)
        
        # NaN should be preserved in result
        assert np.isnan(result[1])
        assert not np.isnan(result[0])
        assert not np.isnan(result[2])

    def test_divide_by_std_negative_values(self):
        """Test normalization with negative perturbations."""
        perturbations = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        unperturbed = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        
        result = divide_by_std(perturbations, unperturbed)
        
        expected_std = np.std(unperturbed, ddof=0)
        expected = perturbations / expected_std
        
        np.testing.assert_array_almost_equal(result, expected)


class TestDivideByUnperturbed:
    """Test suite for divide_by_unperturbed normalization function."""

    def test_divide_by_unperturbed_basic(self):
        """Test basic normalization by unperturbed values."""
        perturbations = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        unperturbed = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        
        result = divide_by_unperturbed(perturbations, unperturbed)
        
        expected = perturbations / unperturbed
        np.testing.assert_array_almost_equal(result, expected)

    def test_divide_by_unperturbed_zeros(self):
        """Test that division by zero produces NaN."""
        perturbations = np.array([1.0, 2.0, 3.0])
        unperturbed = np.array([10.0, 0.0, 30.0])
        
        result = divide_by_unperturbed(perturbations, unperturbed)
        
        # Division by zero should be replaced with NaN
        assert not np.isnan(result[0])
        assert np.isnan(result[1])  # Zero denominator
        assert not np.isnan(result[2])

    def test_divide_by_unperturbed_inf_handling(self):
        """Test that infinite values are replaced with NaN."""
        perturbations = np.array([1.0, 2.0, 3.0])
        unperturbed = np.array([10.0, 0.0, 30.0])
        
        result = divide_by_unperturbed(perturbations, unperturbed)
        
        # Should replace inf with NaN
        assert np.isfinite(result[0])
        assert np.isnan(result[1])
        assert np.isfinite(result[2])

    def test_divide_by_unperturbed_negative_values(self):
        """Test normalization with negative values."""
        perturbations = np.array([-5.0, 10.0, -15.0])
        unperturbed = np.array([10.0, 20.0, 30.0])
        
        result = divide_by_unperturbed(perturbations, unperturbed)
        
        expected = np.array([-0.5, 0.5, -0.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_divide_by_unperturbed_preserves_nans(self):
        """Test that NaN values in input are preserved."""
        perturbations = np.array([1.0, np.nan, 3.0])
        unperturbed = np.array([10.0, 20.0, 30.0])
        
        result = divide_by_unperturbed(perturbations, unperturbed)
        
        assert not np.isnan(result[0])
        assert np.isnan(result[1])
        assert not np.isnan(result[2])

    def test_divide_by_unperturbed_dtype(self):
        """Test that result is float64."""
        perturbations = np.array([1, 2, 3], dtype=np.int32)
        unperturbed = np.array([10, 20, 30], dtype=np.int32)
        
        result = divide_by_unperturbed(perturbations, unperturbed)
        
        assert result.dtype == np.float64


class TestDivideAbsoluteByUnperturbed:
    """Test suite for divide_absolute_by_unperturbed normalization function."""

    def test_divide_absolute_by_unperturbed_basic(self):
        """Test basic normalization of absolute perturbations."""
        perturbations = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        unperturbed = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        
        result = divide_absolute_by_unperturbed(perturbations, unperturbed)
        
        expected = np.abs(perturbations) / unperturbed
        np.testing.assert_array_almost_equal(result, expected)

    def test_divide_absolute_by_unperturbed_negative_values(self):
        """Test that negative perturbations become positive."""
        perturbations = np.array([-5.0, 10.0, -15.0])
        unperturbed = np.array([10.0, 20.0, 30.0])
        
        result = divide_absolute_by_unperturbed(perturbations, unperturbed)
        
        # Should take absolute value before dividing
        expected = np.array([0.5, 0.5, 0.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_divide_absolute_by_unperturbed_zeros(self):
        """Test that division by zero produces NaN."""
        perturbations = np.array([1.0, -2.0, 3.0])
        unperturbed = np.array([10.0, 0.0, 30.0])
        
        result = divide_absolute_by_unperturbed(perturbations, unperturbed)
        
        assert not np.isnan(result[0])
        assert np.isnan(result[1])  # Zero denominator
        assert not np.isnan(result[2])

    def test_divide_absolute_by_unperturbed_all_results_positive(self):
        """Test that all results are non-negative (or NaN)."""
        perturbations = np.array([-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0])
        unperturbed = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        
        result = divide_absolute_by_unperturbed(perturbations, unperturbed)
        
        # All non-NaN values should be >= 0
        finite_results = result[np.isfinite(result)]
        assert np.all(finite_results >= 0)

    def test_divide_absolute_by_unperturbed_inf_handling(self):
        """Test that infinite values are replaced with NaN."""
        perturbations = np.array([1.0, -2.0, 3.0])
        unperturbed = np.array([10.0, 0.0, 30.0])
        
        result = divide_absolute_by_unperturbed(perturbations, unperturbed)
        
        assert np.isfinite(result[0])
        assert np.isnan(result[1])
        assert np.isfinite(result[2])

    def test_divide_absolute_by_unperturbed_dtype(self):
        """Test that result is float64."""
        perturbations = np.array([-1, 2, -3], dtype=np.int32)
        unperturbed = np.array([10, 20, 30], dtype=np.int32)
        
        result = divide_absolute_by_unperturbed(perturbations, unperturbed)
        
        assert result.dtype == np.float64

    def test_divide_absolute_by_unperturbed_preserves_nans(self):
        """Test that NaN values in input are preserved."""
        perturbations = np.array([-1.0, np.nan, 3.0])
        unperturbed = np.array([10.0, 20.0, 30.0])
        
        result = divide_absolute_by_unperturbed(perturbations, unperturbed)
        
        assert not np.isnan(result[0])
        assert np.isnan(result[1])
        assert not np.isnan(result[2])
