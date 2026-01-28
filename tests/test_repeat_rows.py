"""
Tests for repeat_rows function.

Tests the DataFrame row duplication utility used for Monte Carlo sampling.
"""

import numpy as np
import pandas as pd
import pytest

from monte_carlo_sensitivity.repeat_rows import repeat_rows


class TestRepeatRows:
    """Test suite for repeat_rows function."""

    def test_repeat_rows_basic(self, simple_dataframe):
        """Test basic row repetition functionality."""
        n = 3
        result = repeat_rows(simple_dataframe, n)
        
        # Check total rows
        assert len(result) == len(simple_dataframe) * n
        
        # Check that each original row appears n times consecutively
        for i in range(len(simple_dataframe)):
            for j in range(n):
                row_idx = i * n + j
                pd.testing.assert_series_equal(
                    result.iloc[row_idx],
                    simple_dataframe.iloc[i],
                    check_names=False
                )

    def test_repeat_rows_single(self, simple_dataframe):
        """Test repeating rows once (n=1) returns equivalent DataFrame."""
        result = repeat_rows(simple_dataframe, 1)
        
        assert len(result) == len(simple_dataframe)
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True),
            simple_dataframe.reset_index(drop=True)
        )

    def test_repeat_rows_preserves_columns(self, simple_dataframe):
        """Test that column names and order are preserved."""
        result = repeat_rows(simple_dataframe, 5)
        
        assert list(result.columns) == list(simple_dataframe.columns)

    def test_repeat_rows_with_nans(self, dataframe_with_nans):
        """Test that NaN values are preserved during repetition."""
        n = 2
        result = repeat_rows(dataframe_with_nans, n)
        
        # Check that NaN values are in expected positions
        for i in range(len(dataframe_with_nans)):
            for j in range(n):
                row_idx = i * n + j
                for col in dataframe_with_nans.columns:
                    if pd.isna(dataframe_with_nans.iloc[i][col]):
                        assert pd.isna(result.iloc[row_idx][col])
                    else:
                        assert result.iloc[row_idx][col] == dataframe_with_nans.iloc[i][col]

    def test_repeat_rows_with_zeros(self, dataframe_with_zeros):
        """Test that zero values are preserved during repetition."""
        n = 4
        result = repeat_rows(dataframe_with_zeros, n)
        
        assert len(result) == len(dataframe_with_zeros) * n
        
        # Verify zeros are preserved
        for i in range(len(dataframe_with_zeros)):
            for j in range(n):
                row_idx = i * n + j
                pd.testing.assert_series_equal(
                    result.iloc[row_idx],
                    dataframe_with_zeros.iloc[i],
                    check_names=False
                )

    def test_repeat_rows_empty_dataframe(self):
        """Test repeating an empty DataFrame."""
        df = pd.DataFrame()
        result = repeat_rows(df, 5)
        
        assert len(result) == 0
        assert result.empty

    def test_repeat_rows_single_row(self):
        """Test repeating a single-row DataFrame."""
        df = pd.DataFrame({'a': [1.0], 'b': [2.0]})
        n = 10
        result = repeat_rows(df, n)
        
        assert len(result) == n
        for i in range(n):
            assert result.iloc[i]['a'] == 1.0
            assert result.iloc[i]['b'] == 2.0

    def test_repeat_rows_large_n(self, simple_dataframe):
        """Test repeating with large n value."""
        n = 1000
        result = repeat_rows(simple_dataframe, n)
        
        assert len(result) == len(simple_dataframe) * n
        assert list(result.columns) == list(simple_dataframe.columns)

    def test_repeat_rows_values_preserved(self):
        """Test that values are correctly preserved."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c']
        })
        
        result = repeat_rows(df, 3)
        
        # Check that values are preserved (dtypes may change due to np.repeat)
        assert list(result['int_col'].values) == [1, 1, 1, 2, 2, 2, 3, 3, 3]
        assert list(result['str_col'].values) == ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c']
