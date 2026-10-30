"""
================================================================================
PERFORMANCE AND STRESS TESTS FOR APGI SYSTEM
================================================================================

This module provides comprehensive performance and stress testing including:
- Execution time benchmarks
- Memory usage profiling
- CPU utilization tests
- Stress tests under high load
- Scalability tests
- Resource leak detection
"""

from __future__ import annotations

import gc
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# PERFORMANCE BENCHMARK TESTS
# =============================================================================


class TestPerformanceBenchmarks:
    """Performance benchmark tests for core functions."""

    @pytest.mark.performance
    def test_prediction_error_performance(
        self, performance_monitor: Callable[..., Any]
    ) -> None:
        """Benchmark prediction_error function performance."""
        from APGI_System import FoundationalEquations

        with performance_monitor(threshold_ms=500, track_memory=False) as metrics:
            for _ in range(100000):
                FoundationalEquations.prediction_error(5.0, 3.0)

        assert (  # nosec: B101 - Test assertion
            metrics.execution_time < 500
        ), f"prediction_error took {metrics.execution_time:.2f}ms, expected < 500ms"

    @pytest.mark.performance
    def test_z_score_performance(self, performance_monitor: Callable[..., Any]) -> None:
        """Benchmark z_score function performance."""
        from APGI_System import FoundationalEquations

        with performance_monitor(threshold_ms=500, track_memory=False) as metrics:
            for _ in range(100000):
                FoundationalEquations.z_score(10.0, 5.0, 2.0)

        assert metrics.execution_time < 500  # nosec: B101 - Test assertion

    @pytest.mark.performance
    def test_precision_performance(
        self, performance_monitor: Callable[..., Any]
    ) -> None:
        """Benchmark precision function performance."""
        from APGI_System import FoundationalEquations

        with performance_monitor(threshold_ms=500, track_memory=False) as metrics:
            for _ in range(100000):
                FoundationalEquations.precision(4.0)

        assert metrics.execution_time < 500  # nosec: B101 - Test assertion

    @pytest.mark.performance
    def test_accumulated_signal_performance(
        self, performance_monitor: Callable[..., Any]
    ) -> None:
        """Benchmark accumulated_signal function performance."""
        from APGI_System import CoreIgnitionSystem

        with performance_monitor(threshold_ms=1000, track_memory=False) as metrics:
            for _ in range(50000):
                CoreIgnitionSystem.accumulated_signal(1.0, 2.0, 1.0, 2.0)

        assert metrics.execution_time < 1000  # nosec: B101 - Test assertion

    @pytest.mark.performance
    def test_ignition_probability_performance(
        self, performance_monitor: Callable[..., Any]
    ) -> None:
        """Benchmark ignition_probability function performance."""
        from APGI_System import CoreIgnitionSystem

        with performance_monitor(threshold_ms=1500, track_memory=False) as metrics:
            for _ in range(100000):
                CoreIgnitionSystem.ignition_probability(1.0, 0.5, 5.5)

        assert metrics.execution_time < 1500  # nosec: B101 - Test assertion

    @pytest.mark.performance
    def test_signal_dynamics_performance(
        self, performance_monitor: Callable[..., Any]
    ) -> None:
        """Benchmark signal_dynamics function performance."""
        from APGI_System import DynamicalSystemEquations

        with performance_monitor(threshold_ms=2000, track_memory=False) as metrics:
            for _ in range(10000):
                DynamicalSystemEquations.signal_dynamics(
                    S=1.0,
                    Pi_e=1.0,
                    eps_e=0.5,
                    Pi_i_eff=1.0,
                    eps_i=0.5,
                    tau_S=0.35,
                    sigma_S=0.05,
                    dt=0.01,
                )

        assert metrics.execution_time < 2000  # nosec: B101 - Test assertion


# =============================================================================
# MEMORY USAGE TESTS
# =============================================================================


class TestMemoryUsage:
    """Memory usage profiling tests."""

    @pytest.mark.performance
    def test_memory_stability(self) -> None:
        """Test that memory usage remains stable over multiple operations."""
        from APGI_System import RunningStatistics

        # Initial memory snapshot
        gc.collect()
        tracemalloc.start()
        mem_before, _ = tracemalloc.get_traced_memory()

        # Perform many operations
        stats = RunningStatistics()
        for i in range(10000):
            stats.update(float(i))

        # Final memory snapshot
        gc.collect()
        mem_after, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory growth should be minimal (bounded by object size, not iterations)
        mem_growth_mb = (mem_after - mem_before) / 1024 / 1024
        assert (  # nosec: B101 - Test assertion
            mem_growth_mb < 10
        ), f"Memory grew by {mem_growth_mb:.2f}MB, expected < 10MB"

    @pytest.mark.performance
    def test_array_allocation_efficiency(self) -> None:
        """Test numpy array allocation efficiency."""
        gc.collect()
        tracemalloc.start()
        mem_before, _ = tracemalloc.get_traced_memory()

        # Allocate many arrays
        arrays = []
        for i in range(1000):
            arr = np.random.rand(1000)
            arrays.append(arr)

        # Free arrays
        del arrays
        gc.collect()

        mem_after, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # After deletion and GC, memory should be close to original
        mem_used_mb = (mem_after - mem_before) / 1024 / 1024
        assert (
            mem_used_mb < 100
        ), f"Memory used {mem_used_mb:.2f}MB after cleanup"  # nosec: B101 - Test assertion

    @pytest.mark.performance
    @pytest.mark.stress
    def test_large_array_handling(self) -> None:
        """Test handling of large arrays."""
        gc.collect()

        # Allocate increasingly large arrays
        sizes = [1000, 10000, 100000, 1000000]

        for size in sizes:
            try:
                arr = np.random.rand(size)
                del arr
                gc.collect()
            except MemoryError:
                # Acceptable to hit memory limit
                break


# =============================================================================
# STRESS TESTS
# =============================================================================


class TestStressTests:
    """Stress tests under high load conditions."""

    @pytest.mark.stress
    def test_high_frequency_updates(self) -> None:
        """Test system behavior under high-frequency update load."""
        from APGI_System import RunningStatistics

        stats = RunningStatistics()
        start_time = time.time()

        # Perform 1 million updates
        n_updates = 1000000
        for i in range(n_updates):
            stats.update(np.random.randn())

        elapsed = time.time() - start_time
        updates_per_second = n_updates / elapsed

        print(
            f"\nCompleted {n_updates} updates in {elapsed:.2f}s ({updates_per_second:.0f} updates/s)"
        )

        # Should handle high frequency without issues
        assert stats._n_updates == n_updates  # nosec: B101 - Test assertion

    @pytest.mark.stress
    def test_simulation_under_load(self) -> None:
        """Test full simulation under load."""
        from APGI_System import (
            APGIParameters,
            CoreIgnitionSystem,
            DynamicalSystemEquations,
        )

        params = APGIParameters()
        rng = np.random.default_rng(42)

        # Run extended simulation
        n_steps = 100000
        S = 0.5
        theta = params.theta_0
        M = params.M_0

        start_time = time.time()

        for _ in range(n_steps):
            eps_e = np.random.randn()
            eps_i = np.random.randn()

            Pi_i_eff = CoreIgnitionSystem.effective_interoceptive_precision(
                1.0, M, params.M_0, params.beta
            )

            S = DynamicalSystemEquations.signal_dynamics(
                S, 1.0, eps_e, Pi_i_eff, eps_i, params.tau_S, params.sigma_S, 0.01, rng
            )

            theta = DynamicalSystemEquations.threshold_dynamics(
                theta,
                0.3,
                0.7,
                0.5,
                params.gamma_M,
                M,
                0.1,
                S,
                params.tau_theta,
                params.sigma_theta,
                0.01,
                rng,
            )

            M = DynamicalSystemEquations.somatic_marker_dynamics(
                M, eps_i, 0.5, 0.1, 0.0, 1.5, 0.05, 0.01, rng
            )

        elapsed = time.time() - start_time

        print(f"\nCompleted {n_steps} simulation steps in {elapsed:.2f}s")

        # Verify state remained valid
        assert S >= 0  # nosec: B101 - Test assertion
        assert theta > 0  # nosec: B101 - Test assertion
        assert -2.0 <= M <= 2.0  # nosec: B101 - Test assertion

    @pytest.mark.stress
    @pytest.mark.slow
    def test_concurrent_simulations(self) -> None:
        """Test running multiple concurrent simulations."""
        import threading

        from APGI_System import APGIParameters, CoreIgnitionSystem

        results = []
        errors = []

        def run_simulation(seed: int) -> None:
            try:
                params = APGIParameters()
                seed = 42

                for _ in range(10000):
                    CoreIgnitionSystem.effective_interoceptive_precision(
                        1.0, 0.5, 0.0, params.beta
                    )

                results.append(seed)
            except Exception as e:
                errors.append((seed, str(e)))

        # Run 10 concurrent simulations
        threads = []
        for i in range(10):
            t = threading.Thread(target=run_simulation, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30)

        assert (
            len(errors) == 0
        ), f"Errors during concurrent execution: {errors}"  # nosec: B101 - Test assertion
        assert len(results) == 10  # nosec: B101 - Test assertion


# =============================================================================
# SCALABILITY TESTS
# =============================================================================


class TestScalability:
    """Scalability tests for increasing problem sizes."""

    @pytest.mark.performance
    @pytest.mark.parametrize(
        "n_trials",
        [10, 100, 1000, 10000],
    )
    def test_scalability_with_trials(
        self, n_trials: int, performance_monitor: Callable[..., Any]
    ) -> None:
        """Test scalability with increasing number of trials."""
        from APGI_System import RunningStatistics

        with performance_monitor() as metrics:
            stats = RunningStatistics()
            for i in range(n_trials):
                stats.update(float(i))

        # Time per operation should remain relatively constant
        time_per_op = metrics.execution_time / n_trials
        print(f"\nn_trials={n_trials}: {time_per_op:.6f}ms per operation")

        # Should not grow super-linearly (no worse than O(n log n))
        # Use adaptive threshold: small n has higher overhead, large n should be efficient
        threshold = 10.0 if n_trials < 100 else 1.0
        assert (  # nosec: B101 - Test assertion
            time_per_op < threshold
        ), f"Too slow: {time_per_op:.6f}ms per operation (threshold: {threshold})"

    @pytest.mark.performance
    def test_array_operation_scalability(self) -> None:
        """Test scalability of array operations with increasing size."""
        from APGI_System import DerivedQuantities

        sizes = [100, 1000, 10000]
        times = []

        for size in sizes:
            arr = np.random.rand(size)

            start = time.perf_counter()
            DerivedQuantities.metabolic_cost(arr, dt=0.01)
            elapsed = (time.perf_counter() - start) * 1000

            times.append(elapsed)
            print(f"\nArray size {size}: {elapsed:.3f}ms")

        # Time should scale approximately linearly with array size
        # Check that 10x size increase doesn't cause >100x time increase
        if len(times) >= 2:
            ratio = times[1] / times[0]
            size_ratio = sizes[1] / sizes[0]
            assert (  # nosec: B101 - Test assertion
                ratio < size_ratio * 10
            ), f"Time grew too fast: {ratio:.2f}x for {size_ratio:.0f}x size"


# =============================================================================
# RESOURCE LEAK DETECTION
# =============================================================================


class TestResourceLeaks:
    """Tests for detecting resource leaks."""

    def test_file_handle_leak_detection(self, temp_dir: Path) -> None:
        """Test that file handles are properly closed."""
        import json

        file_path = temp_dir / "test_file.json"

        # Write data
        with open(file_path, "w") as f:
            json.dump({"data": "test"}, f)

        # Read many times
        for _ in range(1000):
            with open(file_path) as f:
                json.load(f)

        # File should be closed after each read
        # (No explicit assertion - this test passes if no "too many open files" error)

    def test_memory_leak_object_creation(self) -> None:
        """Test for memory leaks in object creation."""
        from APGI_System import APGIParameters, RunningStatistics

        gc.collect()
        tracemalloc.start()
        mem_before, _ = tracemalloc.get_traced_memory()

        # Create and discard many objects
        for _ in range(10000):
            params = APGIParameters()
            stats = RunningStatistics()
            stats.update(1.0)
            del params
            del stats

        gc.collect()
        mem_after, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory should return close to baseline
        mem_growth_mb = (mem_after - mem_before) / 1024 / 1024
        assert (  # nosec: B101 - Test assertion
            mem_growth_mb < 50
        ), f"Potential memory leak: {mem_growth_mb:.2f}MB growth"


# =============================================================================
# TIMEOUT AND LATENCY TESTS
# =============================================================================


class TestTimeoutAndLatency:
    """Tests for timeout handling and latency requirements."""

    @pytest.mark.performance
    def test_operation_latency(self) -> None:
        """Test that operations complete within acceptable latency."""
        from APGI_System import CoreIgnitionSystem

        max_acceptable_latency_ms = 1.0  # 1 millisecond

        latencies = []
        for _ in range(1000):
            start = time.perf_counter()
            CoreIgnitionSystem.ignition_probability(1.0, 0.5, 5.5)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        avg_latency = np.mean(latencies)
        p99_latency = np.percentile(latencies, 99)

        print(f"\nAvg latency: {avg_latency:.4f}ms, P99: {p99_latency:.4f}ms")

        assert avg_latency < max_acceptable_latency_ms  # nosec: B101 - Test assertion
        assert (
            p99_latency < max_acceptable_latency_ms * 10
        )  # Allow some outliers  # nosec: B101 - Test assertion

    @pytest.mark.performance
    def test_batch_operation_throughput(self) -> None:
        """Test throughput of batch operations."""
        from APGI_System import FoundationalEquations

        batch_size = 10000
        inputs = np.random.randn(batch_size)

        start = time.perf_counter()
        [FoundationalEquations.precision(x**2 + 1) for x in inputs]
        elapsed = time.perf_counter() - start

        throughput = batch_size / elapsed
        print(f"\nBatch throughput: {throughput:.0f} ops/sec")

        assert (
            throughput > 10000
        ), f"Throughput too low: {throughput:.0f} ops/sec"  # nosec: B101 - Test assertion


# =============================================================================
# COMPARATIVE PERFORMANCE TESTS
# =============================================================================


class TestComparativePerformance:
    """Comparative performance tests against baselines."""

    @pytest.mark.performance
    def test_numpy_vs_python_performance(self) -> None:
        """Compare numpy operations vs pure Python."""
        size = 10000
        data = list(range(size))

        # Pure Python sum
        start = time.perf_counter()
        py_sum = sum(x**2 for x in data)
        py_time = time.perf_counter() - start

        # Numpy sum
        arr = np.array(data)
        start = time.perf_counter()
        np_sum = np.sum(arr**2)
        np_time = time.perf_counter() - start

        speedup = py_time / np_time
        print(
            f"\nNumpy speedup: {speedup:.2f}x (Python: {py_time:.4f}s, Numpy: {np_time:.4f}s)"
        )

        # Numpy should be faster
        assert (
            speedup > 1.0
        ), "Numpy should be faster than pure Python"  # nosec: B101 - Test assertion
        # Results should match
        assert abs(py_sum - np_sum) < 1e-6  # nosec: B101 - Test assertion

    @pytest.mark.performance
    def test_vectorized_vs_loop_performance(self) -> None:
        """Compare vectorized operations vs loops."""
        from APGI_System import CoreIgnitionSystem

        n = 10000
        S_values = np.random.rand(n)
        theta = 0.5
        alpha = 5.5

        # Loop version
        start = time.perf_counter()
        [CoreIgnitionSystem.ignition_probability(s, theta, alpha) for s in S_values]
        loop_time = time.perf_counter() - start

        # Vectorized version (simulated with numpy)
        start = time.perf_counter()
        z = alpha * (S_values - theta)
        1.0 / (1.0 + np.exp(-z))
        vectorized_time = time.perf_counter() - start

        speedup = loop_time / vectorized_time
        print(f"\nVectorized speedup: {speedup:.2f}x")

        # Vectorized should be faster for large arrays
        assert speedup > 1.0  # nosec: B101 - Test assertion
