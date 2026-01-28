"""
Pytest fixtures for monte-carlo-sensitivity tests.

Provides shared test data, mock functions, and utilities for testing
sensitivity analysis functions.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def simple_dataframe():
    """Create a simple DataFrame with no missing values."""
    return pd.DataFrame({
        'x': [1.0, 2.0, 3.0, 4.0, 5.0],
        'y': [2.0, 4.0, 6.0, 8.0, 10.0],
        'z': [1.5, 2.5, 3.5, 4.5, 5.5]
    })


@pytest.fixture
def dataframe_with_nans():
    """Create a DataFrame containing NaN values."""
    return pd.DataFrame({
        'x': [1.0, 2.0, np.nan, 4.0, 5.0],
        'y': [2.0, np.nan, 6.0, 8.0, 10.0],
        'z': [1.5, 2.5, 3.5, np.nan, 5.5]
    })


@pytest.fixture
def dataframe_with_zeros():
    """Create a DataFrame containing zero values."""
    return pd.DataFrame({
        'x': [0.0, 1.0, 2.0, 3.0, 0.0],
        'y': [1.0, 0.0, 3.0, 4.0, 5.0],
        'z': [0.0, 0.0, 1.0, 2.0, 3.0]
    })


@pytest.fixture
def linear_forward_process():
    """Create a simple linear forward process: y = 2*x + 1."""
    def process(df):
        result = df.copy()
        if 'x' in df.columns:
            result['y'] = 2 * df['x'] + 1
        return result
    return process


@pytest.fixture
def identity_forward_process():
    """Create an identity forward process that returns input unchanged."""
    def process(df):
        return df.copy()
    return process


@pytest.fixture
def quadratic_forward_process():
    """Create a quadratic forward process: y = x^2."""
    def process(df):
        result = df.copy()
        if 'x' in df.columns:
            result['y'] = df['x'] ** 2
        return result
    return process


@pytest.fixture
def multivar_forward_process():
    """Create a multi-variable forward process: z = 2*x + 3*y."""
    def process(df):
        result = df.copy()
        if 'x' in df.columns and 'y' in df.columns:
            result['z'] = 2 * df['x'] + 3 * df['y']
        return result
    return process


@pytest.fixture
def sample_array_normal():
    """Create a sample numpy array with normal values."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def sample_array_with_zeros():
    """Create a numpy array containing zeros."""
    return np.array([0.0, 1.0, 2.0, 0.0, 3.0])


@pytest.fixture
def sample_array_with_nans():
    """Create a numpy array containing NaN values."""
    return np.array([1.0, np.nan, 3.0, 4.0, np.nan])


@pytest.fixture
def sample_array_with_inf():
    """Create a numpy array containing infinite values."""
    return np.array([1.0, 2.0, np.inf, 4.0, -np.inf])


@pytest.fixture
def random_seed():
    """Fixture to set and reset random seed for reproducible tests."""
    original_state = np.random.get_state()
    np.random.seed(42)
    yield 42
    np.random.set_state(original_state)
