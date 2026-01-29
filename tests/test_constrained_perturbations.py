"""
Tests for constrained perturbations in sensitivity_analysis and perturbed_run.

These tests validate the min/max constraint functionality that prevents perturbed
values from exceeding specified bounds.
"""
import numpy as np
import pandas as pd
import pytest

from monte_carlo_sensitivity import sensitivity_analysis, perturbed_run


def simple_linear_process(df: pd.DataFrame) -> pd.DataFrame:
    """Simple linear transformation: y = 2x + 3"""
    result = df.copy()
    result['y'] = 2 * df['x'] + 3
    return result


def multivariate_process(df: pd.DataFrame) -> pd.DataFrame:
    """Multivariate process: z = a + 2*b - c"""
    result = df.copy()
    result['z'] = df['a'] + 2*df['b'] - df['c']
    return result


class TestConstrainedPerturbedRun:
    """Tests for perturbed_run with input_min and input_max constraints."""

    def test_perturbed_run_with_max_constraint(self):
        """Test that perturbed values respect max constraint."""
        input_df = pd.DataFrame({'x': [5.0, 10.0, 15.0]})
        
        result = perturbed_run(
            input_df=input_df,
            input_variable='x',
            output_variable='y',
            forward_process=simple_linear_process,
            n=50,
            perturbation_std=10.0,  # Large std to ensure we'd exceed max without clipping
            input_max=20.0
        )
        
        # All perturbed input values should be <= 20.0
        assert (result['input_perturbed'] <= 20.0).all(), "Some perturbed values exceed max"
        # Should have some values at or near the boundary
        assert result['input_perturbed'].max() >= 19.0, "No values near maximum boundary"

    def test_perturbed_run_with_min_constraint(self):
        """Test that perturbed values respect min constraint."""
        input_df = pd.DataFrame({'x': [5.0, 10.0, 15.0]})
        
        result = perturbed_run(
            input_df=input_df,
            input_variable='x',
            output_variable='y',
            forward_process=simple_linear_process,
            n=50,
            perturbation_std=10.0,  # Large std to ensure we'd go below min without clipping
            input_min=0.0
        )
        
        # All perturbed input values should be >= 0.0
        assert (result['input_perturbed'] >= 0.0).all(), "Some perturbed values go below min"
        # Should have some values at or near the boundary
        assert result['input_perturbed'].min() <= 1.0, "No values near minimum boundary"

    def test_perturbed_run_with_both_constraints(self):
        """Test that perturbed values respect both min and max constraints."""
        input_df = pd.DataFrame({'x': [10.0, 20.0, 30.0]})
        
        result = perturbed_run(
            input_df=input_df,
            input_variable='x',
            output_variable='y',
            forward_process=simple_linear_process,
            n=100,
            perturbation_std=50.0,  # Very large std to test clipping
            input_min=5.0,
            input_max=35.0
        )
        
        # All perturbed values should be within bounds
        assert (result['input_perturbed'] >= 5.0).all(), "Some values below min"
        assert (result['input_perturbed'] <= 35.0).all(), "Some values above max"

    def test_perturbed_run_constraints_affect_perturbation_column(self):
        """Test that actual perturbations are recalculated after clipping."""
        input_df = pd.DataFrame({'x': [10.0]})
        
        result = perturbed_run(
            input_df=input_df,
            input_variable='x',
            output_variable='y',
            forward_process=simple_linear_process,
            n=100,
            perturbation_std=20.0,
            input_min=0.0,
            input_max=20.0
        )
        
        # Verify that input_perturbation = input_perturbed - input_unperturbed
        expected_perturbation = result['input_perturbed'] - result['input_unperturbed']
        np.testing.assert_allclose(
            result['input_perturbation'], 
            expected_perturbation, 
            rtol=1e-10,
            err_msg="Perturbations not recalculated correctly after clipping"
        )

    def test_perturbed_run_no_constraints_unchanged(self):
        """Test that behavior is unchanged when constraints are not specified."""
        input_df = pd.DataFrame({'x': [5.0, 10.0, 15.0]})
        
        # Set seed for reproducibility
        np.random.seed(42)
        result_no_constraints = perturbed_run(
            input_df=input_df,
            input_variable='x',
            output_variable='y',
            forward_process=simple_linear_process,
            n=50,
            perturbation_std=5.0
        )
        
        # Reset seed
        np.random.seed(42)
        result_with_none = perturbed_run(
            input_df=input_df,
            input_variable='x',
            output_variable='y',
            forward_process=simple_linear_process,
            n=50,
            perturbation_std=5.0,
            input_min=None,
            input_max=None
        )
        
        # Results should be identical
        pd.testing.assert_frame_equal(result_no_constraints, result_with_none)


class TestConstrainedSensitivityAnalysis:
    """Tests for sensitivity_analysis with input_min and input_max constraints."""

    def test_sensitivity_analysis_scalar_max_constraint(self):
        """Test sensitivity analysis with scalar max constraint applied to all variables."""
        input_df = pd.DataFrame({
            'a': [10.0, 20.0, 30.0],
            'b': [5.0, 10.0, 15.0],
            'c': [2.0, 4.0, 6.0]
        })
        
        perturbations_df, metrics_df = sensitivity_analysis(
            input_df=input_df,
            input_variables=['a', 'b'],
            output_variables=['z'],
            forward_process=multivariate_process,
            n=50,
            perturbation_std=20.0,
            input_max=40.0
        )
        
        # Check all perturbed values are below max
        assert (perturbations_df['input_perturbed'] <= 40.0).all()
        # Metrics should be computed
        assert len(metrics_df) == 6  # 2 input variables * 3 metrics each
        assert not metrics_df['value'].isna().all()

    def test_sensitivity_analysis_scalar_min_constraint(self):
        """Test sensitivity analysis with scalar min constraint applied to all variables."""
        input_df = pd.DataFrame({
            'a': [10.0, 20.0, 30.0],
            'b': [5.0, 10.0, 15.0],
            'c': [2.0, 4.0, 6.0]
        })
        
        perturbations_df, metrics_df = sensitivity_analysis(
            input_df=input_df,
            input_variables=['a', 'b'],
            output_variables=['z'],
            forward_process=multivariate_process,
            n=50,
            perturbation_std=20.0,
            input_min=0.0
        )
        
        # Check all perturbed values are above min
        assert (perturbations_df['input_perturbed'] >= 0.0).all()

    def test_sensitivity_analysis_dict_constraints(self):
        """Test sensitivity analysis with per-variable dict constraints."""
        input_df = pd.DataFrame({
            'a': [10.0, 20.0, 30.0],
            'b': [5.0, 10.0, 15.0],
            'c': [2.0, 4.0, 6.0]
        })
        
        # Different constraints for each variable
        input_min = {'a': 0.0, 'b': 0.0}
        input_max = {'a': 50.0, 'b': 20.0}
        
        perturbations_df, metrics_df = sensitivity_analysis(
            input_df=input_df,
            input_variables=['a', 'b'],
            output_variables=['z'],
            forward_process=multivariate_process,
            n=50,
            perturbation_std=20.0,
            input_min=input_min,
            input_max=input_max
        )
        
        # Check constraints for 'a'
        a_perturbed = perturbations_df[perturbations_df['input_variable'] == 'a']['input_perturbed']
        assert (a_perturbed >= 0.0).all()
        assert (a_perturbed <= 50.0).all()
        
        # Check constraints for 'b'
        b_perturbed = perturbations_df[perturbations_df['input_variable'] == 'b']['input_perturbed']
        assert (b_perturbed >= 0.0).all()
        assert (b_perturbed <= 20.0).all()

    def test_sensitivity_analysis_asymmetric_constraints(self):
        """Test with asymmetric bounds (different distances from mean)."""
        input_df = pd.DataFrame({
            'x': [10.0, 15.0, 20.0]
        })
        
        # Asymmetric: can go down by 5 but up by 15
        perturbations_df, metrics_df = sensitivity_analysis(
            input_df=input_df,
            input_variables=['x'],
            output_variables=['y'],
            forward_process=simple_linear_process,
            n=100,
            perturbation_std=10.0,
            input_min=5.0,
            input_max=35.0
        )
        
        assert (perturbations_df['input_perturbed'] >= 5.0).all()
        assert (perturbations_df['input_perturbed'] <= 35.0).all()

    def test_sensitivity_analysis_partial_dict_constraints(self):
        """Test dict constraints that only specify limits for some variables."""
        input_df = pd.DataFrame({
            'a': [10.0, 20.0, 30.0],
            'b': [5.0, 10.0, 15.0],
            'c': [2.0, 4.0, 6.0]
        })
        
        # Only constrain 'a', leave 'b' unconstrained
        input_min = {'a': 0.0}
        input_max = {'a': 50.0}
        
        perturbations_df, metrics_df = sensitivity_analysis(
            input_df=input_df,
            input_variables=['a', 'b'],
            output_variables=['z'],
            forward_process=multivariate_process,
            n=50,
            perturbation_std=10.0,
            input_min=input_min,
            input_max=input_max
        )
        
        # 'a' should be constrained
        a_perturbed = perturbations_df[perturbations_df['input_variable'] == 'a']['input_perturbed']
        assert (a_perturbed >= 0.0).all()
        assert (a_perturbed <= 50.0).all()
        
        # 'b' should be unconstrained (can have any values from perturbation)
        b_perturbed = perturbations_df[perturbations_df['input_variable'] == 'b']['input_perturbed']
        # Just verify it exists and has reasonable values
        assert len(b_perturbed) > 0

    def test_sensitivity_analysis_joint_mode_with_constraints(self):
        """Test that constraints work with optimized joint execution mode."""
        input_df = pd.DataFrame({
            'a': [10.0, 20.0, 30.0],
            'b': [5.0, 10.0, 15.0],
            'c': [2.0, 4.0, 6.0]
        })
        
        perturbations_df, metrics_df = sensitivity_analysis(
            input_df=input_df,
            input_variables=['a', 'b'],
            output_variables=['z'],
            forward_process=multivariate_process,
            n=50,
            perturbation_std=20.0,
            input_min=0.0,
            input_max=40.0,
            use_joint_run=True  # Explicitly use joint mode
        )
        
        assert (perturbations_df['input_perturbed'] >= 0.0).all()
        assert (perturbations_df['input_perturbed'] <= 40.0).all()

    def test_sensitivity_analysis_loop_mode_with_constraints(self):
        """Test that constraints work with loop-based execution mode."""
        input_df = pd.DataFrame({
            'a': [10.0, 20.0, 30.0],
            'b': [5.0, 10.0, 15.0],
            'c': [2.0, 4.0, 6.0]
        })
        
        perturbations_df, metrics_df = sensitivity_analysis(
            input_df=input_df,
            input_variables=['a', 'b'],
            output_variables=['z'],
            forward_process=multivariate_process,
            n=50,
            perturbation_std=20.0,
            input_min=0.0,
            input_max=40.0,
            use_joint_run=False  # Explicitly use loop mode
        )
        
        assert (perturbations_df['input_perturbed'] >= 0.0).all()
        assert (perturbations_df['input_perturbed'] <= 40.0).all()

    def test_sensitivity_analysis_backward_compatibility(self):
        """Test that results are identical when constraints are not used."""
        input_df = pd.DataFrame({
            'a': [10.0, 20.0, 30.0],
            'b': [5.0, 10.0, 15.0],
            'c': [2.0, 4.0, 6.0]
        })
        
        np.random.seed(42)
        perturb_old, metrics_old = sensitivity_analysis(
            input_df=input_df,
            input_variables=['a', 'b'],
            output_variables=['z'],
            forward_process=multivariate_process,
            n=30,
            perturbation_std=5.0
        )
        
        np.random.seed(42)
        perturb_new, metrics_new = sensitivity_analysis(
            input_df=input_df,
            input_variables=['a', 'b'],
            output_variables=['z'],
            forward_process=multivariate_process,
            n=30,
            perturbation_std=5.0,
            input_min=None,
            input_max=None
        )
        
        # Results should be identical
        pd.testing.assert_frame_equal(perturb_old, perturb_new)
        pd.testing.assert_frame_equal(metrics_old, metrics_new)

    def test_sensitivity_analysis_tight_constraints_affect_sensitivity(self):
        """Test that tight constraints can reduce observed sensitivity."""
        input_df = pd.DataFrame({'x': [10.0, 20.0, 30.0]})
        
        # Unconstrained - should see full sensitivity
        np.random.seed(123)
        perturb_unconstrained, _ = sensitivity_analysis(
            input_df=input_df,
            input_variables=['x'],
            output_variables=['y'],
            forward_process=simple_linear_process,
            n=100,
            perturbation_std=5.0
        )
        
        # Very tight constraints - will clip most perturbations
        np.random.seed(123)
        perturb_constrained, _ = sensitivity_analysis(
            input_df=input_df,
            input_variables=['x'],
            output_variables=['y'],
            forward_process=simple_linear_process,
            n=100,
            perturbation_std=5.0,
            input_min=9.5,
            input_max=30.5
        )
        
        # With tight constraints, the range of perturbations should be smaller
        unconstrained_range = perturb_unconstrained['input_perturbation'].abs().mean()
        constrained_range = perturb_constrained['input_perturbation'].abs().mean()
        assert constrained_range < unconstrained_range, \
            "Tight constraints should reduce perturbation magnitude"


class TestConstraintEdgeCases:
    """Test edge cases and boundary conditions for constraints."""

    def test_constraint_at_exactly_boundary(self):
        """Test when initial values are at constraint boundaries."""
        input_df = pd.DataFrame({'x': [0.0, 10.0, 20.0]})
        
        result = perturbed_run(
            input_df=input_df,
            input_variable='x',
            output_variable='y',
            forward_process=simple_linear_process,
            n=50,
            perturbation_std=5.0,
            input_min=0.0,
            input_max=20.0
        )
        
        # Should handle boundary values correctly
        assert (result['input_perturbed'] >= 0.0).all()
        assert (result['input_perturbed'] <= 20.0).all()

    def test_constraint_with_zero_perturbation_std(self):
        """Test constraints when perturbation std is very small."""
        input_df = pd.DataFrame({'x': [10.0]})
        
        result = perturbed_run(
            input_df=input_df,
            input_variable='x',
            output_variable='y',
            forward_process=simple_linear_process,
            n=20,
            perturbation_std=0.01,  # Very small
            input_min=9.0,
            input_max=11.0
        )
        
        # Should work without issues
        assert len(result) == 20
        assert (result['input_perturbed'] >= 9.0).all()
        assert (result['input_perturbed'] <= 11.0).all()

    def test_constraint_only_max(self):
        """Test constraint with only max specified (min is None)."""
        input_df = pd.DataFrame({'x': [10.0, 20.0]})
        
        result = perturbed_run(
            input_df=input_df,
            input_variable='x',
            output_variable='y',
            forward_process=simple_linear_process,
            n=50,
            perturbation_std=15.0,
            input_min=None,
            input_max=25.0
        )
        
        # Max should be enforced
        assert (result['input_perturbed'] <= 25.0).all()
        # Some values should be well below max (no min constraint)
        assert result['input_perturbed'].min() < 10.0

    def test_constraint_only_min(self):
        """Test constraint with only min specified (max is None)."""
        input_df = pd.DataFrame({'x': [10.0, 20.0]})
        
        result = perturbed_run(
            input_df=input_df,
            input_variable='x',
            output_variable='y',
            forward_process=simple_linear_process,
            n=50,
            perturbation_std=15.0,
            input_min=5.0,
            input_max=None
        )
        
        # Min should be enforced
        assert (result['input_perturbed'] >= 5.0).all()
        # Some values should be well above min (no max constraint)
        assert result['input_perturbed'].max() > 25.0
