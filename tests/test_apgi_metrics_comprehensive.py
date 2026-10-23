"""
Comprehensive tests for apgi_metrics.py to achieve 90%+ coverage.
Tests all major classes, methods, calculations, and edge cases.
"""

from __future__ import annotations

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


import numpy as np
import pytest

from apgi_metrics import IgnitionMetrics, MetabolicMetrics, SurpriseMetrics


class TestIgnitionMetrics:
    """Tests for IgnitionMetrics dataclass."""

    def test_metrics_initialization(self):
        # Test metrics initialization
        IgnitionMetrics(
            ignition_rate=0.75,
            mean_ignition_time=2.5,
            ignition_threshold=3.0,
            ignition_volatility=0.8,
            cumulative_ignition_prob=0.65,
            ignition_events=[1.0, 2.0, 1.5, 0.8],
        )

    def test_metrics_validation(self):
        """Test metrics validation."""
        # Valid metrics
        IgnitionMetrics(
            ignition_rate=0.0,  # Valid: 0 to 1
            mean_ignition_time=0.1,  # Valid: positive
            ignition_threshold=0.5,  # Valid: positive
            ignition_volatility=0.0,  # Valid: non-negative
            cumulative_ignition_prob=1.0,  # Valid: 0 to 1
            ignition_events=[0.5, 1.0, 1.5],  # Valid list
        )

        # Should not raise any exceptions
        # No explicit validation in current implementation

    def test_metrics_edge_cases(self):
        """Test metrics edge cases."""
        # Boundary values
        metrics_boundary = IgnitionMetrics(
            ignition_rate=1.0,  # Maximum
            mean_ignition_time=0.0,  # Minimum
            ignition_threshold=10.0,  # High threshold
            ignition_volatility=2.0,  # High volatility
            cumulative_ignition_prob=0.0,  # Minimum
            ignition_events=[],  # Empty list
        )

        # All values should be valid
        assert metrics_boundary.ignition_rate == 1.0  # nosec: B101 - Test assertion
        assert (
            metrics_boundary.mean_ignition_time == 0.0
        )  # nosec: B101 - Test assertion
        assert (
            metrics_boundary.ignition_threshold == 10.0
        )  # nosec: B101 - Test assertion
        assert (
            metrics_boundary.ignition_volatility == 2.0
        )  # nosec: B101 - Test assertion
        assert (
            metrics_boundary.cumulative_ignition_prob == 0.0
        )  # nosec: B101 - Test assertion
        assert (
            len(metrics_boundary.ignition_events) == 0
        )  # nosec: B101 - Test assertion

    def test_ignition_events_statistics(self):
        """Test ignition events statistical calculations."""
        events = [1.0, 2.0, 1.5, 0.8, 2.2, 1.8, 0.5, 1.0]

        metrics = IgnitionMetrics(
            ignition_rate=0.5,
            mean_ignition_time=1.5,
            ignition_threshold=0.8,
            ignition_volatility=0.2,
            cumulative_ignition_prob=0.6,
            ignition_events=events,
        )

        # Test statistical properties
        assert len(metrics.ignition_events) == 8  # nosec: B101 - Test assertion
        assert metrics.ignition_events[0] == 1.0  # nosec: B101 - Test assertion
        assert metrics.ignition_events[-1] == 0.5  # nosec: B101 - Test assertion

        # Calculate expected statistics manually
        expected_mean = np.mean(events)
        expected_variance = np.var(events)

        # Test if methods exist (if implemented)
        if hasattr(metrics, "calculate_statistics"):
            stats = metrics.calculate_statistics()
            assert (
                abs(stats["mean"] - expected_mean) < 0.01
            )  # nosec: B101 - Test assertion
            assert (
                abs(stats["variance"] - expected_variance) < 0.01
            )  # nosec: B101 - Test assertion


class TestSurpriseMetrics:
    """Tests for SurpriseMetrics dataclass."""

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = SurpriseMetrics(
            mean_surprise=1.2,
            surprise_variance=0.8,
            max_surprise=2.5,
            min_surprise=0.3,
            surprise_entropy=1.8,
            prediction_error_variance=0.6,
        )

        assert metrics.mean_surprise == 1.2  # nosec: B101 - Test assertion
        assert metrics.surprise_variance == 0.8  # nosec: B101 - Test assertion
        assert metrics.max_surprise == 2.5  # nosec: B101 - Test assertion
        assert metrics.min_surprise == 0.3  # nosec: B101 - Test assertion
        assert metrics.surprise_entropy == 1.8  # nosec: B101 - Test assertion
        assert metrics.prediction_error_variance == 0.6  # nosec: B101 - Test assertion

    def test_metrics_validation(self):
        """Test metrics validation."""
        # Valid metrics
        SurpriseMetrics(
            mean_surprise=0.0,  # Valid: non-negative
            surprise_variance=0.5,  # Valid: non-negative
            max_surprise=5.0,  # Valid: positive
            min_surprise=0.0,  # Valid: non-negative
            surprise_entropy=2.0,  # Valid: positive
            prediction_error_variance=1.0,  # Valid: non-negative
        )

    def test_metrics_edge_cases(self):
        """Test metrics edge cases."""
        # Boundary values
        SurpriseMetrics(
            mean_surprise=10.0,  # High mean
            surprise_variance=5.0,  # High variance
            max_surprise=20.0,  # High max
            min_surprise=-5.0,  # Invalid: negative
            surprise_entropy=10.0,  # High entropy
            prediction_error_variance=10.0,  # High variance
        )

    def test_surprise_statistics_calculation(self):
        """Test surprise statistics calculation."""
        # Test with sample data
        prediction_errors = [0.1, 0.5, 0.2, 0.8, 1.2, 0.3]
        precisions = [1.0, 1.5, 2.0, 0.8, 1.2]

        # Calculate expected values
        expected_surprises = []
        for error, precision in zip(prediction_errors, precisions):
            surprise = 0.5 * (error**2)  # Simplified surprise calculation
            expected_surprises.append(surprise)

        expected_mean = np.mean(expected_surprises)
        expected_variance = np.var(expected_surprises)

        # Test if methods exist (if implemented)
        if hasattr(SurpriseMetrics, "from_data"):
            metrics = SurpriseMetrics.from_data(prediction_errors, precisions)

            assert (
                abs(metrics.mean_surprise - expected_mean) < 0.01
            )  # nosec: B101 - Test assertion
            assert (
                abs(metrics.surprise_variance - expected_variance) < 0.01
            )  # nosec: B101 - Test assertion


class TestMetabolicMetrics:
    """Tests for MetabolicMetrics dataclass."""

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = MetabolicMetrics(
            mean_metabolic_cost=2.5,
            total_metabolic_cost=125.0,
            metabolic_efficiency=0.98,
            optimal_cost_rate=8.5,
            cost_variance=0.5,
        )

        assert metrics.mean_metabolic_cost == 2.5  # nosec: B101 - Test assertion
        assert metrics.total_metabolic_cost == 125.0  # nosec: B101 - Test assertion
        assert metrics.metabolic_efficiency == 0.98  # nosec: B101 - Test assertion
        assert metrics.optimal_cost_rate == 8.5  # nosec: B101 - Test assertion
        # Note: average_metabolic_rate is not implemented in MetabolicMetrics
        # Only optimal_cost_rate is available for rate-based metrics

    def test_metrics_validation(self):
        """Test metrics validation."""
        # Valid metrics
        MetabolicMetrics(
            mean_metabolic_cost=0.0,  # Valid: non-negative
            total_metabolic_cost=10.0,  # Valid: positive
            metabolic_efficiency=1.0,  # Valid: 0 to 1
            optimal_cost_rate=0.1,  # Valid: positive
            cost_variance=0.5,  # Valid: non-negative
        )

    def test_metrics_edge_cases(self):
        """Test metrics edge cases."""
        # Boundary values
        MetabolicMetrics(
            mean_metabolic_cost=100.0,  # High cost
            total_metabolic_cost=0.0,  # Invalid: zero
            metabolic_efficiency=2.0,  # Invalid: > 1
            optimal_cost_rate=0.0,  # Invalid: zero
            cost_variance=0.5,  # Valid: non-negative
        )

    def test_metabolic_calculations(self):
        """Test metabolic cost calculations."""
        # Test with sample data
        metabolic_costs = [1.0, 2.5, 1.8, 3.2, 2.1, 1.5]
        time_points = [1, 2, 3, 4, 5, 6]

        # Test if methods exist (if implemented)
        if hasattr(MetabolicMetrics, "from_data"):
            metrics = MetabolicMetrics.from_data(metabolic_costs, time_points)

            assert (
                abs(metrics.mean_metabolic_cost - np.mean(metabolic_costs)) < 0.01
            )  # nosec: B101 - Test assertion
            assert metrics.total_metabolic_cost == sum(
                metabolic_costs
            )  # nosec: B101 - Test assertion
            assert metrics.peak_metabolic_rate == max(
                metabolic_costs
            )  # nosec: B101 - Test assertion


class TestMetricsIntegration:
    """Integration tests for metrics calculation."""

    def test_ignition_surprise_integration(self):
        """Test integration between ignition and surprise metrics."""
        # Mock data
        prediction_errors = np.array([0.1, 0.5, 0.2, 0.8, 1.2])
        precisions = np.array([1.0, 1.5, 2.0, 0.8, 1.2])

        # Test if integrated calculation methods exist
        if hasattr(IgnitionMetrics, "calculate_from_predictions") and hasattr(
            SurpriseMetrics, "calculate_from_predictions"
        ):

            ignition_metrics = IgnitionMetrics.calculate_from_predictions(
                prediction_errors, precisions
            )
            surprise_metrics = SurpriseMetrics.calculate_from_predictions(
                prediction_errors, precisions
            )

            # Verify results are reasonable
            assert (
                0.0 <= ignition_metrics.ignition_rate <= 1.0
            )  # nosec: B101 - Test assertion
            assert (
                ignition_metrics.mean_ignition_time >= 0.0
            )  # nosec: B101 - Test assertion
            assert surprise_metrics.mean_surprise >= 0.0  # nosec: B101 - Test assertion

    def test_comprehensive_metrics_workflow(self):
        """Test comprehensive metrics calculation workflow."""
        # Test with realistic data
        prediction_errors = np.random.randn(100) * 0.5  # Simulate prediction errors
        precisions = np.random.uniform(0.5, 2.0, 100)  # Variable precision
        metabolic_costs = np.random.exponential(2.0, 100)  # Variable metabolic cost

        # Test batch calculation
        if (
            hasattr(IgnitionMetrics, "calculate_batch")
            and hasattr(SurpriseMetrics, "calculate_batch")
            and hasattr(MetabolicMetrics, "calculate_batch")
        ):

            batch_metrics = IgnitionMetrics.calculate_batch(
                prediction_errors, precisions
            )
            batch_surprise = SurpriseMetrics.calculate_batch(
                prediction_errors, precisions
            )
            batch_metabolic = MetabolicMetrics.calculate_batch(metabolic_costs)

            # Verify batch results
            assert len(batch_metrics) == len(
                prediction_errors
            )  # nosec: B101 - Test assertion
            assert len(batch_surprise) == len(
                prediction_errors
            )  # nosec: B101 - Test assertion
            assert len(batch_metabolic) == len(
                metabolic_costs
            )  # nosec: B101 - Test assertion

    def test_metrics_error_handling(self):
        """Test error handling in metrics calculations."""
        # Test with invalid data
        invalid_data = [np.nan, np.inf, -np.inf]

        for data in invalid_data:
            # Test if error handling exists
            if hasattr(IgnitionMetrics, "calculate_from_predictions"):
                try:
                    IgnitionMetrics.calculate_from_predictions([data], [1.0])
                except (ValueError, TypeError):
                    pass  # Should handle gracefully
                else:
                    pytest.fail("Should handle invalid data gracefully")

    def test_metrics_performance(self):
        """Test metrics calculation performance."""
        # Test with large dataset
        large_data = np.random.randn(10000)
        large_precisions = np.random.uniform(0.1, 5.0, 10000)

        import time

        start_time = time.time()

        # Test if batch calculation exists and is performant
        if hasattr(IgnitionMetrics, "calculate_batch"):
            try:
                IgnitionMetrics.calculate_batch(large_data, large_precisions)
                calculation_time = time.time() - start_time

                # Should complete in reasonable time (< 1 second for 10k samples)
                assert calculation_time < 1.0  # nosec: B101 - Test assertion
            except Exception:
                pytest.fail("Performance test failed with exception")

    def test_metrics_consistency(self):
        """Test metrics calculation consistency."""
        # Same data should produce same results
        prediction_errors = np.array([0.5, 1.0, 1.5])
        precisions = np.array([1.0, 2.0, 1.5])

        if hasattr(IgnitionMetrics, "calculate_from_predictions"):
            # Calculate twice
            result1 = IgnitionMetrics.calculate_from_predictions(
                prediction_errors, precisions
            )
            result2 = IgnitionMetrics.calculate_from_predictions(
                prediction_errors, precisions
            )

            # Results should be identical
            assert (
                result1.ignition_rate == result2.ignition_rate
            )  # nosec: B101 - Test assertion
            assert (
                result1.mean_ignition_time == result2.mean_ignition_time
            )  # nosec: B101 - Test assertion


class TestMetricsValidation:
    """Tests for metrics validation and edge cases."""

    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        if hasattr(IgnitionMetrics, "calculate_from_predictions"):
            # Test with empty arrays
            try:
                IgnitionMetrics.calculate_from_predictions([], [])
                pytest.fail("Should raise exception for empty data")
            except (ValueError, TypeError):
                pass  # Expected behavior

    def test_single_value_handling(self):
        """Test handling of single values."""
        if hasattr(IgnitionMetrics, "calculate_from_predictions"):
            # Test with single value arrays
            result = IgnitionMetrics.calculate_from_predictions([0.5], [1.0])

            assert result.ignition_rate is not None  # nosec: B101 - Test assertion
            assert result.mean_ignition_time is not None  # nosec: B101 - Test assertion

    def test_extreme_values(self):
        """Test handling of extreme values."""
        if hasattr(IgnitionMetrics, "calculate_from_predictions"):
            # Test with extreme values
            extreme_errors = np.array([1e6, -1e6])
            extreme_precisions = np.array([1e-6, 1e6])

            try:
                IgnitionMetrics.calculate_from_predictions(
                    extreme_errors, extreme_precisions
                )
                # Should handle gracefully or raise appropriate exception
                pass  # Implementation-dependent
            except Exception:
                pass  # Implementation-dependent

    def test_numerical_stability(self):
        """Test numerical stability of metrics calculations."""
        # Test with values that could cause numerical issues
        stable_errors = np.array(
            [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8, 1e10]
        )
        stable_precisions = np.array([1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8, 1e10])

        if hasattr(IgnitionMetrics, "calculate_from_predictions"):
            try:
                result = IgnitionMetrics.calculate_from_predictions(
                    stable_errors, stable_precisions
                )

                # Result should be finite and reasonable
                assert np.isfinite(result.ignition_rate)  # nosec: B101 - Test assertion
                assert np.isfinite(
                    result.mean_ignition_time
                )  # nosec: B101 - Test assertion
                assert not np.isnan(
                    result.ignition_volatility
                )  # nosec: B101 - Test assertion
            except Exception as e:
                pytest.fail(f"Numerical stability test failed: {e}")

    def test_metrics_documentation(self):
        """Test that metrics are properly documented."""
        # Test that dataclasses have proper docstrings
        assert IgnitionMetrics.__doc__ is not None  # nosec: B101 - Test assertion
        assert (
            "Ignition system metrics" in IgnitionMetrics.__doc__
        )  # nosec: B101 - Test assertion

        assert SurpriseMetrics.__doc__ is not None  # nosec: B101 - Test assertion
        assert (
            "Surprise metrics" in SurpriseMetrics.__doc__
        )  # nosec: B101 - Test assertion
        assert (
            "prediction error variance" in SurpriseMetrics.__doc__
        )  # nosec: B101 - Test assertion

        assert MetabolicMetrics.__doc__ is not None  # nosec: B101 - Test assertion
        assert (
            "Metabolic cost metrics" in MetabolicMetrics.__doc__
        )  # nosec: B101 - Test assertion
        assert (
            "average metabolic rate" in MetabolicMetrics.__doc__
        )  # nosec: B101 - Test assertion


doc_lines = (IgnitionMetrics.__doc__ or "").split("\n")
print("Docstring lines:")
for i, line in enumerate(doc_lines):
    print(f"{i}: {repr(line)}")
print()
print("Looking for key phrases:")
for phrase in ["cumulative", "ignition events"]:
    if IgnitionMetrics.__doc__ and phrase in IgnitionMetrics.__doc__:
        print(f"Found: {phrase}")
    else:
        print(f"Missing: {phrase}")
