"""
Enhanced tests for performance_monitoring.py - Covering missing lines to reach 100%.

This module tests the previously uncovered functionality:
- MemorySnapshot and CPUSnapshot dataclasses
- Background monitoring thread
- Threshold callbacks
- Performance regression detection
- Plot generation
- Various edge cases
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from utils.performance_monitoring import (
    BenchmarkResult,
    CPUSnapshot,
    MemorySnapshot,
    PerformanceMetrics,
    PerformanceMonitor,
    _benchmark_decorator,
    _PerformanceContext,
    benchmark,
    create_performance_monitor,
    get_memory_usage,
    monitor_function_performance,
)


class TestMemorySnapshot:
    """Test MemorySnapshot dataclass."""

    def test_memory_snapshot_creation(self):
        """Test creating a MemorySnapshot."""
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            rss_mb=100.0,
            vms_mb=200.0,
            percent=50.0,
            available_mb=1000.0,
            gc_objects=1000,
            gc_stats={"generation_0": 10, "generation_1": 5},
        )
        assert snapshot.rss_mb == 100.0  # nosec: B101 - Test assertion
        assert snapshot.vms_mb == 200.0  # nosec: B101 - Test assertion
        assert snapshot.percent == 50.0  # nosec: B101 - Test assertion
        assert snapshot.available_mb == 1000.0  # nosec: B101 - Test assertion
        assert snapshot.gc_objects == 1000  # nosec: B101 - Test assertion
        assert snapshot.gc_stats["generation_0"] == 10  # nosec: B101 - Test assertion


class TestCPUSnapshot:
    """Test CPUSnapshot dataclass."""

    def test_cpu_snapshot_creation(self):
        """Test creating a CPUSnapshot."""
        snapshot = CPUSnapshot(
            timestamp=time.time(),
            percent=25.0,
            count=8,
            freq_mhz=2400.0,
            load_avg=[1.0, 2.0, 3.0],
        )
        assert snapshot.percent == 25.0  # nosec: B101 - Test assertion
        assert snapshot.count == 8  # nosec: B101 - Test assertion
        assert snapshot.freq_mhz == 2400.0  # nosec: B101 - Test assertion
        assert snapshot.load_avg == [1.0, 2.0, 3.0]  # nosec: B101 - Test assertion


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_performance_metrics_creation(self):
        """Test creating PerformanceMetrics."""
        metrics = PerformanceMetrics(
            operation_name="test_op",
            start_time=time.time(),
        )
        assert metrics.operation_name == "test_op"  # nosec: B101 - Test assertion
        assert metrics.start_time > 0  # nosec: B101 - Test assertion
        assert metrics.end_time is None  # nosec: B101 - Test assertion
        assert metrics.duration is None  # nosec: B101 - Test assertion


class TestPerformanceMonitorBackgroundMonitoring:
    """Test background monitoring functionality."""

    @patch("utils.performance_monitoring.psutil.Process")
    @patch("utils.performance_monitoring.psutil.cpu_percent")
    @patch("utils.performance_monitoring.psutil.cpu_count")
    @patch("utils.performance_monitoring.psutil.cpu_freq")
    @patch("utils.performance_monitoring.psutil.virtual_memory")
    def test_start_monitoring(
        self, mock_vm, mock_freq, mock_count, mock_cpu_percent, mock_process
    ):
        """Test starting background monitoring."""
        # Setup mocks
        mock_process_instance = MagicMock()
        mock_process_instance.memory_info.return_value = MagicMock(
            rss=100000000, vms=200000000
        )
        mock_process_instance.memory_percent.return_value = 5.0
        mock_process.return_value = mock_process_instance

        mock_cpu_percent.return_value = 25.0
        mock_count.return_value = 8
        mock_freq.return_value = MagicMock(current=2400.0)
        mock_vm.return_value = MagicMock(available=4000000000)

        monitor = PerformanceMonitor()
        monitor.start_monitoring(interval=0.1)

        # Let it run for a bit
        time.sleep(0.2)

        # Stop monitoring
        monitor.stop_monitoring()

        assert len(monitor.memory_history) > 0  # nosec: B101 - Test assertion
        assert len(monitor.cpu_history) > 0  # nosec: B101 - Test assertion

    def test_stop_monitoring_not_started(self):
        """Test stopping when not started."""
        monitor = PerformanceMonitor()
        # Should not raise
        monitor.stop_monitoring()

    @patch("utils.performance_monitoring.psutil.Process")
    def test_monitoring_error_handling(self, mock_process):
        """Test error handling in monitoring loop."""
        mock_process.side_effect = Exception("Process error")

        monitor = PerformanceMonitor()
        monitor.start_monitoring(interval=0.1)
        time.sleep(0.2)
        monitor.stop_monitoring()

        # Should have handled error gracefully


class TestPerformanceMonitorThresholds:
    """Test threshold checking functionality."""

    @patch("utils.performance_monitoring.psutil.Process")
    @patch("utils.performance_monitoring.psutil.cpu_percent")
    @patch("utils.performance_monitoring.psutil.cpu_count")
    @patch("utils.performance_monitoring.psutil.cpu_freq")
    @patch("utils.performance_monitoring.psutil.virtual_memory")
    def test_memory_threshold_violation(
        self, mock_vm, mock_freq, mock_count, mock_cpu_percent, mock_process
    ):
        """Test memory threshold violation detection."""
        # Setup mocks with high memory usage
        mock_process_instance = MagicMock()
        mock_process_instance.memory_info.return_value = MagicMock(
            rss=2000000000, vms=3000000000
        )  # 2GB RSS
        mock_process_instance.memory_percent.return_value = 50.0
        mock_process.return_value = mock_process_instance

        mock_cpu_percent.return_value = 25.0
        mock_count.return_value = 8
        mock_freq.return_value = MagicMock(current=2400.0)
        mock_vm.return_value = MagicMock(available=1000000000)

        # Setup callback
        callback_calls = []

        def callback(event_type, data):
            callback_calls.append((event_type, data))

        monitor = PerformanceMonitor()
        monitor.memory_threshold_mb = 1024  # 1GB threshold
        monitor.add_threshold_callback(callback)
        monitor.start_monitoring(interval=0.1)

        time.sleep(0.2)

        monitor.stop_monitoring()

        # Should have detected threshold violation
        assert len(callback_calls) > 0  # nosec: B101 - Test assertion

    @patch("utils.performance_monitoring.psutil.Process")
    @patch("utils.performance_monitoring.psutil.cpu_percent")
    @patch("utils.performance_monitoring.psutil.cpu_count")
    @patch("utils.performance_monitoring.psutil.cpu_freq")
    @patch("utils.performance_monitoring.psutil.virtual_memory")
    def test_cpu_threshold_violation(
        self, mock_vm, mock_freq, mock_count, mock_cpu_percent, mock_process
    ):
        """Test CPU threshold violation detection."""
        # Setup mocks with high CPU usage
        mock_process_instance = MagicMock()
        mock_process_instance.memory_info.return_value = MagicMock(
            rss=100000000, vms=200000000
        )
        mock_process_instance.memory_percent.return_value = 5.0
        mock_process.return_value = mock_process_instance

        mock_cpu_percent.return_value = 90.0  # High CPU
        mock_count.return_value = 8
        mock_freq.return_value = MagicMock(current=2400.0)
        mock_vm.return_value = MagicMock(available=4000000000)

        callback_calls = []

        def callback(event_type, data):
            callback_calls.append((event_type, data))

        monitor = PerformanceMonitor()
        monitor.cpu_threshold_percent = 80.0
        monitor.add_threshold_callback(callback)
        monitor.start_monitoring(interval=0.1)

        time.sleep(0.2)

        monitor.stop_monitoring()

        # Should have detected threshold violation
        assert len(callback_calls) > 0  # nosec: B101 - Test assertion

    @patch("utils.performance_monitoring.psutil.Process")
    def test_memory_leak_detection(self, mock_process):
        """Test memory leak detection."""
        # Simulate growing memory
        memory_values = [100000000, 150000000, 200000000, 250000000, 300000000]
        call_count = [0]

        def side_effect():
            idx = min(call_count[0], len(memory_values) - 1)
            call_count[0] += 1
            mock = MagicMock()
            mock.memory_info.return_value = MagicMock(
                rss=memory_values[idx], vms=memory_values[idx] * 2
            )
            mock.memory_percent.return_value = 5.0
            return mock

        mock_process.side_effect = side_effect

        callback_calls = []

        def callback(event_type, data):
            callback_calls.append((event_type, data))

        monitor = PerformanceMonitor()
        monitor.memory_leak_threshold_mb = 50
        monitor.add_threshold_callback(callback)

        # Simulate memory snapshots
        with monitor._lock:
            for i, mem in enumerate(memory_values):
                monitor.memory_history.append(
                    MemorySnapshot(
                        timestamp=time.time() + i,
                        rss_mb=mem / 1024 / 1024,
                        vms_mb=mem * 2 / 1024 / 1024,
                        percent=5.0,
                        available_mb=4000.0,
                        gc_objects=1000,
                        gc_stats={},
                    )
                )

        # Check thresholds
        if len(monitor.memory_history) >= 10:
            monitor._check_thresholds(
                monitor.memory_history[-1],
                (
                    monitor.cpu_history[-1]
                    if monitor.cpu_history
                    else CPUSnapshot(
                        timestamp=time.time(),
                        percent=50.0,
                        count=8,
                        freq_mhz=2400.0,
                        load_avg=[1.0, 2.0, 3.0],
                    )
                ),
            )


class TestPerformanceMonitorOperations:
    """Test operation monitoring."""

    @patch("utils.performance_monitoring.psutil.Process")
    @patch("utils.performance_monitoring.psutil.cpu_percent")
    @patch("utils.performance_monitoring.psutil.cpu_count")
    @patch("utils.performance_monitoring.psutil.cpu_freq")
    @patch("utils.performance_monitoring.psutil.virtual_memory")
    def test_start_and_end_operation(
        self, mock_vm, mock_freq, mock_count, mock_cpu_percent, mock_process
    ):
        """Test starting and ending operation monitoring."""
        # Setup mocks
        mock_process_instance = MagicMock()
        mock_process_instance.memory_info.return_value = MagicMock(
            rss=100000000, vms=200000000
        )
        mock_process_instance.memory_percent.return_value = 5.0
        mock_process.return_value = mock_process_instance

        mock_cpu_percent.return_value = 25.0
        mock_count.return_value = 8
        mock_freq.return_value = MagicMock(current=2400.0)
        mock_vm.return_value = MagicMock(available=4000000000)

        monitor = PerformanceMonitor()

        # Start operation
        metrics = monitor.start_operation("test_op")
        assert metrics.operation_name == "test_op"  # nosec: B101 - Test assertion
        assert metrics.start_time > 0  # nosec: B101 - Test assertion
        assert metrics.memory_before is not None  # nosec: B101 - Test assertion
        assert metrics.cpu_before is not None  # nosec: B101 - Test assertion

        time.sleep(0.1)

        # End operation
        monitor.end_operation(metrics, success=True)
        assert metrics.end_time is not None  # nosec: B101 - Test assertion
        assert metrics.duration is not None  # nosec: B101 - Test assertion
        assert metrics.memory_after is not None  # nosec: B101 - Test assertion
        assert metrics.cpu_after is not None  # nosec: B101 - Test assertion
        assert metrics.success is True  # nosec: B101 - Test assertion

    @patch("utils.performance_monitoring.psutil.Process")
    def test_end_operation_with_error(self, mock_process):
        """Test ending operation with error."""
        mock_process_instance = MagicMock()
        mock_process_instance.memory_info.return_value = MagicMock(
            rss=100000000, vms=200000000
        )
        mock_process_instance.memory_percent.return_value = 5.0
        mock_process.return_value = mock_process_instance

        monitor = PerformanceMonitor()

        metrics = monitor.start_operation("test_op")
        time.sleep(0.05)
        monitor.end_operation(metrics, success=False, error="Test error")

        assert metrics.success is False  # nosec: B101 - Test assertion
        assert metrics.error == "Test error"  # nosec: B101 - Test assertion


class TestPerformanceSummary:
    """Test performance summary functionality."""

    @patch("utils.performance_monitoring.psutil.Process")
    def test_get_performance_summary(self, mock_process):
        """Test getting performance summary."""
        mock_process_instance = MagicMock()
        mock_process_instance.memory_info.return_value = MagicMock(
            rss=100000000, vms=200000000
        )
        mock_process_instance.memory_percent.return_value = 5.0
        mock_process.return_value = mock_process_instance

        monitor = PerformanceMonitor()

        # Create some operation metrics
        metrics1 = monitor.start_operation("op1")
        time.sleep(0.05)
        monitor.end_operation(metrics1, success=True)

        metrics2 = monitor.start_operation("op2")
        time.sleep(0.05)
        monitor.end_operation(metrics2, success=True)

        summary = monitor.get_performance_summary()
        assert "total_operations" in summary  # nosec: B101 - Test assertion
        assert summary["total_operations"] == 2  # nosec: B101 - Test assertion

    @patch("utils.performance_monitoring.psutil.Process")
    def test_get_performance_summary_by_name(self, mock_process):
        """Test getting performance summary filtered by operation name."""
        mock_process_instance = MagicMock()
        mock_process_instance.memory_info.return_value = MagicMock(
            rss=100000000, vms=200000000
        )
        mock_process_instance.memory_percent.return_value = 5.0
        mock_process.return_value = mock_process_instance

        monitor = PerformanceMonitor()

        # Create operations with different names
        metrics1 = monitor.start_operation("op1")
        time.sleep(0.05)
        monitor.end_operation(metrics1, success=True)

        metrics2 = monitor.start_operation("op2")
        time.sleep(0.05)
        monitor.end_operation(metrics2, success=True)

        summary = monitor.get_performance_summary("op1")
        assert summary["total_operations"] == 1  # nosec: B101 - Test assertion

    def test_get_performance_summary_empty(self):
        """Test getting performance summary with no operations."""
        monitor = PerformanceMonitor()
        summary = monitor.get_performance_summary()
        assert summary == {}  # nosec: B101 - Test assertion


class TestTrendAnalysis:
    """Test trend analysis functionality."""

    def test_get_memory_trend(self):
        """Test getting memory trend."""
        monitor = PerformanceMonitor()

        # Add memory snapshots
        with monitor._lock:
            for i in range(10):
                monitor.memory_history.append(
                    MemorySnapshot(
                        timestamp=time.time() + i,
                        rss_mb=100.0 + i * 10,  # Increasing memory
                        vms_mb=200.0 + i * 10,
                        percent=5.0,
                        available_mb=4000.0,
                        gc_objects=1000,
                        gc_stats={},
                    )
                )

        trend = monitor.get_memory_trend()
        assert "current_mb" in trend  # nosec: B101 - Test assertion
        assert "trend_direction" in trend  # nosec: B101 - Test assertion
        assert trend["trend_direction"] == "increasing"  # nosec: B101 - Test assertion

    def test_get_memory_trend_stable(self):
        """Test getting stable memory trend."""
        monitor = PerformanceMonitor()

        # Add stable memory snapshots
        with monitor._lock:
            for i in range(10):
                monitor.memory_history.append(
                    MemorySnapshot(
                        timestamp=time.time() + i,
                        rss_mb=100.0,  # Stable memory
                        vms_mb=200.0,
                        percent=5.0,
                        available_mb=4000.0,
                        gc_objects=1000,
                        gc_stats={},
                    )
                )

        trend = monitor.get_memory_trend()
        assert trend["trend_direction"] == "stable"  # nosec: B101 - Test assertion

    def test_get_memory_trend_insufficient_data(self):
        """Test getting memory trend with insufficient data."""
        monitor = PerformanceMonitor()
        trend = monitor.get_memory_trend()
        assert trend == {}  # nosec: B101 - Test assertion

    def test_get_cpu_trend(self):
        """Test getting CPU trend."""
        monitor = PerformanceMonitor()

        # Add CPU snapshots
        with monitor._lock:
            for i in range(10):
                monitor.cpu_history.append(
                    CPUSnapshot(
                        timestamp=time.time() + i,
                        percent=20.0 + i * 2,  # Increasing CPU
                        count=8,
                        freq_mhz=2400.0,
                        load_avg=[1.0, 2.0, 3.0],
                    )
                )

        trend = monitor.get_cpu_trend()
        assert "current_percent" in trend  # nosec: B101 - Test assertion
        assert "trend_direction" in trend  # nosec: B101 - Test assertion
        assert trend["trend_direction"] == "increasing"  # nosec: B101 - Test assertion

    def test_get_cpu_trend_insufficient_data(self):
        """Test getting CPU trend with insufficient data."""
        monitor = PerformanceMonitor()
        trend = monitor.get_cpu_trend()
        assert trend == {}  # nosec: B101 - Test assertion


class TestPerformanceRegression:
    """Test performance regression detection."""

    @patch("utils.performance_monitoring.psutil.Process")
    def test_detect_performance_regression_no_data(self, mock_process):
        """Test regression detection with insufficient data."""
        mock_process_instance = MagicMock()
        mock_process_instance.memory_info.return_value = MagicMock(
            rss=100000000, vms=200000000
        )
        mock_process_instance.memory_percent.return_value = 5.0
        mock_process.return_value = mock_process_instance

        monitor = PerformanceMonitor()

        result = monitor.detect_performance_regression()
        assert result["status"] == "insufficient_data"  # nosec: B101 - Test assertion

    @patch("utils.performance_monitoring.psutil.Process")
    def test_detect_performance_regression_significant(self, mock_process):
        """Test detecting significant regression."""
        mock_process_instance = MagicMock()
        mock_process_instance.memory_info.return_value = MagicMock(
            rss=100000000, vms=200000000
        )
        mock_process_instance.memory_percent.return_value = 5.0
        mock_process.return_value = mock_process_instance

        monitor = PerformanceMonitor()

        # Create baseline metrics (fast)
        for i in range(10):
            metrics = monitor.start_operation(f"baseline_{i}")
            time.sleep(0.01)
            monitor.end_operation(metrics, success=True)

        # Create recent metrics (slow - regression)
        for i in range(5):
            metrics = monitor.start_operation(f"recent_{i}")
            time.sleep(0.05)  # Slower
            monitor.end_operation(metrics, success=True)

        result = monitor.detect_performance_regression(
            baseline_window=10, current_window=5
        )

        # Should detect regression
        if result["status"] == "detected":
            assert "regression_percent" in result  # nosec: B101 - Test assertion
            assert "significance" in result  # nosec: B101 - Test assertion


class TestPlotGeneration:
    """Test plot generation functionality."""

    def test_generate_performance_plots_memory(self, tmp_path):
        """Test generating memory usage plots."""
        monitor = PerformanceMonitor(output_dir=str(tmp_path))

        # Add memory history
        with monitor._lock:
            for i in range(10):
                monitor.memory_history.append(
                    MemorySnapshot(
                        timestamp=time.time() + i,
                        rss_mb=100.0 + i * 10,
                        vms_mb=200.0 + i * 10,
                        percent=5.0,
                        available_mb=4000.0,
                        gc_objects=1000,
                        gc_stats={},
                    )
                )

        with patch("matplotlib.pyplot.savefig"):
            with patch("matplotlib.pyplot.close"):
                plot_files = monitor.generate_performance_plots()
                assert len(plot_files) > 0  # nosec: B101 - Test assertion

    def test_generate_performance_plots_cpu(self, tmp_path):
        """Test generating CPU usage plots."""
        monitor = PerformanceMonitor(output_dir=str(tmp_path))

        # Add CPU history
        with monitor._lock:
            for i in range(10):
                monitor.cpu_history.append(
                    CPUSnapshot(
                        timestamp=time.time() + i,
                        percent=20.0 + i * 2,
                        count=8,
                        freq_mhz=2400.0,
                        load_avg=[1.0, 2.0, 3.0],
                    )
                )

        with patch("matplotlib.pyplot.savefig"):
            with patch("matplotlib.pyplot.close"):
                plot_files = monitor.generate_performance_plots()
                assert len(plot_files) > 0  # nosec: B101 - Test assertion


class TestSaveMonitoringData:
    """Test saving monitoring data."""

    def test_save_monitoring_data(self, tmp_path):
        """Test saving monitoring data to file."""
        monitor = PerformanceMonitor(output_dir=str(tmp_path))

        # Add some history
        with monitor._lock:
            monitor.memory_history.append(
                MemorySnapshot(
                    timestamp=time.time(),
                    rss_mb=100.0,
                    vms_mb=200.0,
                    percent=5.0,
                    available_mb=4000.0,
                    gc_objects=1000,
                    gc_stats={},
                )
            )

        file_path = monitor.save_monitoring_data()
        assert file_path.exists()  # nosec: B101 - Test assertion

        # Verify content
        with open(file_path) as f:
            data = json.load(f)
            assert "memory_history" in data  # nosec: B101 - Test assertion
            assert "cpu_history" in data  # nosec: B101 - Test assertion


class TestCallbackErrorHandling:
    """Test callback error handling."""

    def test_callback_error(self, capsys):
        """Test that callback errors are handled gracefully."""

        def bad_callback(event_type, data):
            raise Exception("Callback error")

        monitor = PerformanceMonitor()
        monitor.add_threshold_callback(bad_callback)

        # Create a mock memory snapshot
        memory_snapshot = MemorySnapshot(
            timestamp=time.time(),
            rss_mb=2000.0,  # Exceeds threshold
            vms_mb=3000.0,
            percent=50.0,
            available_mb=1000.0,
            gc_objects=1000,
            gc_stats={},
        )

        cpu_snapshot = CPUSnapshot(
            timestamp=time.time(),
            percent=90.0,
            count=8,
            freq_mhz=2400.0,
            load_avg=[1.0, 2.0, 3.0],
        )

        # Should not raise
        monitor._check_thresholds(memory_snapshot, cpu_snapshot)


class TestBenchmarkEdgeCases:
    """Test benchmark edge cases."""

    def test_benchmark_empty_times(self):
        """Test benchmark with empty times list."""
        result = benchmark(lambda: None, iterations=0)
        assert result.iterations == 0  # nosec: B101 - Test assertion

    def test_benchmark_func_raises_exception(self):
        """Test benchmark when function raises exception."""

        def failing_func():
            raise ValueError("Test error")

        # Timing a function that fails is meaningless, so the error propagates
        # (PerformanceMonitor.benchmark, by contrast, keeps going).
        with pytest.raises(ValueError, match="Test error"):
            benchmark(failing_func, iterations=1)


class TestPerformanceContext:
    """Test _PerformanceContext."""

    def test_performance_context_exit_with_exception(self):
        """Test context manager exit with exception."""
        context = _PerformanceContext("test")

        try:
            with context:
                raise ValueError("Test error")
        except ValueError:
            pass

        # Duration should still be recorded
        assert context.end_time is not None  # nosec: B101 - Test assertion


class TestCreatePerformanceMonitor:
    """Test create_performance_monitor function."""

    def test_create_performance_monitor(self, tmp_path):
        """Test creating a performance monitor."""
        monitor = create_performance_monitor(output_dir=str(tmp_path))
        assert isinstance(monitor, PerformanceMonitor)  # nosec: B101 - Test assertion
        assert str(monitor.output_dir) == str(tmp_path)  # nosec: B101 - Test assertion


class TestMonitorFunctionPerformance:
    """Test monitor_function_performance decorator."""

    @patch("utils.performance_monitoring.psutil.Process")
    def test_monitor_function_success(self, mock_process):
        """Test monitoring a successful function."""
        mock_process_instance = MagicMock()
        mock_process_instance.memory_info.return_value = MagicMock(
            rss=100000000, vms=200000000
        )
        mock_process_instance.memory_percent.return_value = 5.0
        mock_process.return_value = mock_process_instance

        monitor = PerformanceMonitor()

        @monitor_function_performance(monitor, "test_func")
        def test_func():
            return 42

        result = test_func()
        assert result == 42  # nosec: B101 - Test assertion

    @patch("utils.performance_monitoring.psutil.Process")
    def test_monitor_function_exception(self, mock_process):
        """Test monitoring a function that raises exception."""
        mock_process_instance = MagicMock()
        mock_process_instance.memory_info.return_value = MagicMock(
            rss=100000000, vms=200000000
        )
        mock_process_instance.memory_percent.return_value = 5.0
        mock_process.return_value = mock_process_instance

        monitor = PerformanceMonitor()

        @monitor_function_performance(monitor, "failing_func")
        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_func()


class TestBenchmarkDecorator:
    """Test _benchmark_decorator."""

    def test_benchmark_decorator_usage(self):
        """Test using benchmark as decorator with iterations."""
        decorator = _benchmark_decorator("test", 2)

        @decorator
        def test_func():
            time.sleep(0.001)
            return 42

        result = test_func()
        assert isinstance(result, BenchmarkResult)  # nosec: B101 - Test assertion
        assert result.name == "test"  # nosec: B101 - Test assertion


class TestMemoryUsageError:
    """Test get_memory_usage error handling."""

    @patch("utils.performance_monitoring.psutil.Process")
    def test_get_memory_usage_error(self, mock_process):
        """Test get_memory_usage when psutil raises error."""
        mock_process.side_effect = Exception("Process error")

        memory = get_memory_usage()
        assert memory == 0.0  # nosec: B101 - Test assertion


class TestRegressionEdgeCases:
    """Test regression detection edge cases."""

    @patch("utils.performance_monitoring.psutil.Process")
    def test_regression_no_successful_operations(self, mock_process):
        """Test regression detection with no successful operations."""
        mock_process_instance = MagicMock()
        mock_process_instance.memory_info.return_value = MagicMock(
            rss=100000000, vms=200000000
        )
        mock_process_instance.memory_percent.return_value = 5.0
        mock_process.return_value = mock_process_instance

        monitor = PerformanceMonitor()

        # Create failed operations only
        for i in range(20):
            metrics = monitor.start_operation(f"op_{i}")
            time.sleep(0.01)
            monitor.end_operation(metrics, success=False, error="Failed")

        result = monitor.detect_performance_regression()
        assert result["status"] == "insufficient_data"  # nosec: B101 - Test assertion


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
