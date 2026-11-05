"""
Unit tests for APGI components.

Comprehensive test suite for APGI integration, validation, and monitoring modules.
"""

import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

# Import APGI modules
from apgi_integration import (
    APGIIntegration,
    APGIParameters,
    CoreEquations,
    DynamicalSystem,
    RunningStatistics,
)
from progress_tracking import ProgressTracker, TrialResult
from utils.git_operations import GitRollbackManager
from utils.performance_monitoring import MemorySnapshot, PerformanceMonitor
from validation import (
    validate_code_modification,
    validate_modifications_before_apply,
    validate_module_name,
)


class TestAPGIParameters:
    """Test cases for APGIParameters."""

    def test_default_parameters(self):
        """Test default parameter values."""
        params = APGIParameters()

        assert 0.2 <= params.tau_S <= 0.5  # nosec: B101 - Test assertion
        assert 0.5 <= params.beta <= 2.5  # nosec: B101 - Test assertion
        assert 3.0 <= params.alpha <= 8.0  # nosec: B101 - Test assertion
        assert 0.3 <= params.rho <= 0.9  # nosec: B101 - Test assertion

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Valid parameters
        params = APGIParameters()
        violations = params.validate()
        assert len(violations) == 0  # nosec: B101 - Test assertion

        # Invalid parameters
        params.tau_S = 1.0  # Outside range
        violations = params.validate()
        assert len(violations) > 0  # nosec: B101 - Test assertion
        assert any("tau_S" in v for v in violations)  # nosec: B101 - Test assertion

    def test_domain_thresholds(self):
        """Test domain-specific thresholds."""
        params = APGIParameters()

        survival_threshold = params.get_domain_threshold("survival")
        neutral_threshold = params.get_domain_threshold("neutral")
        default_threshold = params.get_domain_threshold("other")

        assert survival_threshold < neutral_threshold  # nosec: B101 - Test assertion
        assert default_threshold == params.theta_0  # nosec: B101 - Test assertion

    def test_neuromodulator_effects(self):
        """Test neuromodulator effects calculation."""
        params = APGIParameters()
        effects = params.apply_neuromodulator_effects()

        expected_keys = ["Pi_e_mod", "theta_mod", "beta_mod", "Pi_i_mod"]
        for key in expected_keys:
            assert key in effects  # nosec: B101 - Test assertion
            assert isinstance(
                effects[key], (int, float)
            )  # nosec: B101 - Test assertion

    def test_precision_expectation_gap(self):
        """Test precision expectation gap calculation."""
        params = APGIParameters()

        gap = params.compute_precision_expectation_gap(2.0, 1.5)
        assert isinstance(gap, (int, float))  # nosec: B101 - Test assertion


class TestCoreEquations:
    """Test cases for core APGI equations."""

    def test_prediction_error(self):
        """Test prediction error calculation."""
        assert (
            CoreEquations.prediction_error(10, 8) == 2
        )  # nosec: B101 - Test assertion
        assert (
            CoreEquations.prediction_error(5, 10) == -5
        )  # nosec: B101 - Test assertion
        assert CoreEquations.prediction_error(0, 0) == 0  # nosec: B101 - Test assertion

    def test_precision(self):
        """Test precision calculation."""
        assert CoreEquations.precision(1.0) == 1.0  # nosec: B101 - Test assertion
        assert CoreEquations.precision(0.5) == 2.0  # nosec: B101 - Test assertion
        assert CoreEquations.precision(0.1) == 10.0  # nosec: B101 - Test assertion
        assert (
            CoreEquations.precision(0.0) == 1e6
        )  # Cap for zero variance  # nosec: B101 - Test assertion

    def test_z_score(self):
        """Test z-score calculation."""
        # Normal case
        assert (
            CoreEquations.z_score(2.0, 1.0, 1.0) == 1.0
        )  # nosec: B101 - Test assertion
        assert (
            CoreEquations.z_score(0.0, 1.0, 1.0) == -1.0
        )  # nosec: B101 - Test assertion

        # Edge case: zero standard deviation
        assert (
            CoreEquations.z_score(1.0, 0.0, 0.0) == 0.0
        )  # nosec: B101 - Test assertion

    def test_accumulated_signal(self):
        """Test accumulated signal calculation."""
        signal = CoreEquations.accumulated_signal(2.0, 1.0, 1.5, 0.5)
        expected = 0.5 * 2.0 * (1.0**2) + 0.5 * 1.5 * (0.5**2)
        assert abs(signal - expected) < 1e-10  # nosec: B101 - Test assertion

    def test_effective_interoceptive_precision(self):
        """Test effective interoceptive precision."""
        precision = CoreEquations.effective_interoceptive_precision(1.0, 0.5, 0.0, 1.5)
        expected = 1.0 * (1.0 + 1.5 * (1.0 / (1.0 + np.exp(-(0.5 - 0.0)))))
        assert abs(precision - expected) < 1e-10  # nosec: B101 - Test assertion

    def test_ignition_probability(self):
        """Test ignition probability calculation."""
        # High signal, low threshold
        prob = CoreEquations.ignition_probability(10.0, 1.0, 5.0)
        assert prob > 0.9  # nosec: B101 - Test assertion

        # Low signal, high threshold
        prob = CoreEquations.ignition_probability(1.0, 10.0, 5.0)
        assert prob < 0.1  # nosec: B101 - Test assertion

        # Edge case: equal signal and threshold
        prob = CoreEquations.ignition_probability(5.0, 5.0, 5.0)
        assert 0.4 < prob < 0.6  # Should be around 0.5  # nosec: B101 - Test assertion


class TestRunningStatistics:
    """Test cases for RunningStatistics."""

    def test_initialization(self):
        """Test RunningStatistics initialization."""
        stats = RunningStatistics()

        assert stats.mean == 0.0  # nosec: B101 - Test assertion
        assert stats.var == 1.0  # nosec: B101 - Test assertion
        assert stats.count == 0  # nosec: B101 - Test assertion

    def test_updates(self):
        """Test statistics updates."""
        stats = RunningStatistics()

        # First update
        mean, std = stats.update(5.0)
        assert mean == 5.0  # nosec: B101 - Test assertion
        assert (
            std == 0.0
        )  # Single value has zero variance  # nosec: B101 - Test assertion

        # Second update
        mean, std = stats.update(7.0)
        assert mean == 6.0  # (5 + 7) / 2  # nosec: B101 - Test assertion
        assert std > 0.0  # nosec: B101 - Test assertion

    def test_z_score(self):
        """Test z-score calculation."""
        stats = RunningStatistics()

        # Before any updates
        z = stats.z_score(5.0)
        assert z == 0.0  # nosec: B101 - Test assertion

        # After updates
        stats.update(10.0)
        stats.update(15.0)

        z = stats.z_score(10.0)
        assert (
            abs(z - (-1.0)) < 1e-10
        )  # Mean=12.5, so z = (10 - 12.5) / std = -1.0  # nosec: B101 - Test assertion

    def test_reset(self):
        """Test statistics reset."""
        stats = RunningStatistics()
        stats.update(5.0)
        stats.update(10.0)

        assert stats.count == 2  # nosec: B101 - Test assertion
        assert stats.mean == 7.5  # nosec: B101 - Test assertion

        stats.reset()
        assert stats.count == 0  # nosec: B101 - Test assertion
        assert stats.mean == 0.0  # nosec: B101 - Test assertion
        assert stats.var == 1.0  # nosec: B101 - Test assertion


class TestDynamicalSystem:
    """Test cases for DynamicalSystem."""

    def test_initialization(self):
        """Test DynamicalSystem initialization."""
        params = APGIParameters()
        system = DynamicalSystem(params)

        assert system.S == 0.0  # nosec: B101 - Test assertion
        assert system.theta == params.theta_0  # nosec: B101 - Test assertion
        assert system.M == 0.0  # nosec: B101 - Test assertion
        assert len(system.S_history) == 0  # nosec: B101 - Test assertion
        assert len(system.ignition_history) == 0  # nosec: B101 - Test assertion

    def test_step(self):
        """Test single time step."""
        params = APGIParameters()
        system = DynamicalSystem(params)

        state = system.step(
            prediction_error_ext=1.0,
            prediction_error_int=0.3,
            precision_ext=2.0,
            precision_int_baseline=1.5,
            dt=0.01,
        )

        expected_keys = [
            "S",
            "theta",
            "M",
            "z_e",
            "z_i",
            "Pi_i_eff",
            "ignition_prob",
            "ignited",
        ]
        for key in expected_keys:
            assert key in state  # nosec: B101 - Test assertion
            # Handle numpy bool types as well as Python bool
            val = state[key]
            assert isinstance(  # nosec: B101 - Test assertion
                val, (int, float, bool, np.bool_)
            ), f"{key} has type {type(val)}"

    def test_reset(self):
        """Test system reset."""
        import APGI_System

        APGI_System.set_run_seed(0)  # step() is stochastic; fix the recorded seed
        params = APGIParameters()
        system = DynamicalSystem(params)

        # Run a few steps to change state
        system.step(1.0, 0.3, 2.0, 1.5, 0.01)
        system.step(0.5, 0.15, 1.8, 1.2, 0.01)
        assert system.S > 0.0  # nosec: B101 - Test assertion
        assert len(system.S_history) > 0  # nosec: B101 - Test assertion
        # Reset
        system.reset()
        assert system.S == 0.0  # nosec: B101 - Test assertion
        assert system.theta == params.theta_0  # nosec: B101 - Test assertion
        assert system.M == 0.0  # nosec: B101 - Test assertion
        assert len(system.S_history) == 0  # nosec: B101 - Test assertion
        assert len(system.ignition_history) == 0  # nosec: B101 - Test assertion

    def test_metabolic_cost(self):
        """Test metabolic cost calculation."""
        params = APGIParameters()
        system = DynamicalSystem(params)

        # Run a few steps
        for _ in range(10):
            system.step(1.0, 0.3, 2.0, 1.5, 0.01)

        cost = system.get_metabolic_cost()
        assert cost >= 0.0  # nosec: B101 - Test assertion

    def test_ignition_rate(self):
        """Test ignition rate calculation."""
        params = APGIParameters()
        system = DynamicalSystem(params)

        # Run a few steps
        for _ in range(10):
            system.step(1.0, 0.3, 2.0, 1.5, 0.01)

        rate = system.get_ignition_rate()
        assert 0.0 <= rate <= 1.0  # nosec: B101 - Test assertion


class TestAPGIIntegration:
    """Test cases for APGIIntegration."""

    def test_initialization(self):
        """Test APGIIntegration initialization."""
        apgi = APGIIntegration()

        assert apgi.S == 0.0  # nosec: B101 - Test assertion
        assert apgi.theta == 0.5  # Default theta_0  # nosec: B101 - Test assertion
        assert apgi.M == 0.0  # nosec: B101 - Test assertion
        assert len(apgi.trial_metrics) == 0  # nosec: B101 - Test assertion

    def test_proxy_properties(self):
        """Test proxy properties to dynamical system."""
        apgi = APGIIntegration()

        # Test setting and getting
        apgi.S = 1.0
        assert apgi.S == 1.0  # nosec: B101 - Test assertion
        assert apgi.dynamics.S == 1.0  # nosec: B101 - Test assertion

        apgi.theta = 0.8
        assert apgi.theta == 0.8  # nosec: B101 - Test assertion
        assert apgi.dynamics.theta == 0.8  # nosec: B101 - Test assertion

    def test_core_equations(self):
        """Test core equation methods."""
        apgi = APGIIntegration()

        # Test prediction error
        error = apgi.compute_prediction_error(10, 8)
        assert error == 2.0  # nosec: B101 - Test assertion

        # Test precision
        precision = apgi.compute_precision(0.5)
        assert precision == 1.0  # 0.5 * 0.5  # nosec: B101 - Test assertion

        # Test surprise
        surprise = apgi.compute_surprise(1.0, 2.0)
        assert surprise == 1.0  # 0.5 * 2.0 * (1.0^2)  # nosec: B101 - Test assertion

    def test_ignition_probability(self):
        """Test ignition probability calculation."""
        apgi = APGIIntegration()

        prob = apgi.compute_ignition_probability(1.0, 2.0, 0.5)
        assert 0.0 <= prob <= 1.0  # nosec: B101 - Test assertion

    def test_process_trial(self):
        """Test trial processing."""
        apgi = APGIIntegration()

        state = apgi.process_trial(observed=1.0, predicted=0.0, trial_type="neutral")

        expected_keys = [
            "S",
            "theta",
            "M",
            "z_e",
            "z_i",
            "Pi_i_eff",
            "ignition_prob",
            "ignited",
        ]
        for key in expected_keys:
            assert key in state  # nosec: B101 - Test assertion

    def test_finalize(self):
        """Test experiment finalization."""
        apgi = APGIIntegration()

        # Process some trials
        for i in range(5):
            apgi.process_trial(i * 0.1, 0.0, "neutral")

        summary = apgi.finalize()

        expected_keys = [
            "mean_surprise",
            "mean_threshold",
            "mean_somatic_marker",
            "ignition_rate",
            "metabolic_cost",
            "final_surprise",
            "final_threshold",
            "final_somatic_marker",
        ]
        for key in expected_keys:
            assert key in summary  # nosec: B101 - Test assertion


class TestValidation:
    """Test cases for validation module."""

    def test_validate_modifications_before_apply(self):
        """Test modification validation."""
        # Valid modifications
        valid_mods = {
            "time_budget": 300.0,
            "participant_id": "test_001",
            "stimulus_type": "visual",
        }
        result = validate_modifications_before_apply(valid_mods)
        assert result.is_valid  # nosec: B101 - Test assertion
        assert len(result.errors) == 0  # nosec: B101 - Test assertion

        # Invalid modifications
        invalid_mods = {
            "time_budget": -100.0,  # Negative
            "participant_id": "test_001" * 100,  # Too long
            "stimulus_type": "invalid_type",
        }
        result = validate_modifications_before_apply(invalid_mods)
        assert not result.is_valid  # nosec: B101 - Test assertion
        assert len(result.errors) >= 2  # nosec: B101 - Test assertion

    def test_validate_code_modification(self):
        """Test code modification validation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("# original content")
            temp_path = f.name

        try:
            # Valid code
            valid_code = "print('Hello, world!')\nprint('This is safe.')"
            result = validate_code_modification(temp_path, valid_code)
            assert result.is_valid  # nosec: B101 - Test assertion

            # Dangerous code
            dangerous_code = "import os; os.system('rm -rf /')\nprint('Dangerous!')"
            result = validate_code_modification(temp_path, dangerous_code)
            assert not result.is_valid  # nosec: B101 - Test assertion
            assert len(result.errors) > 0  # nosec: B101 - Test assertion
        finally:
            os.unlink(temp_path)

    def test_validate_module_name(self):
        """Test module name validation."""
        # Safe modules
        safe_modules = ["math", "random", "datetime", "collections", "eval", "exec"]
        for module in safe_modules:
            assert validate_module_name(module)  # nosec: B101 - Test assertion

        # Dangerous modules
        dangerous_modules = ["os", "sys", "subprocess"]
        for module in dangerous_modules:
            assert not validate_module_name(module)  # nosec: B101 - Test assertion

        # Invalid names
        invalid_names = ["..module", "module$hack", "module;cmd"]
        for name in invalid_names:
            assert not validate_module_name(name)  # nosec: B101 - Test assertion


class TestGitOperations:
    """Test cases for git operations with rollback."""

    def test_git_rollback_manager_initialization(self):
        """Test GitRollbackManager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize a git repository
            subprocess.run(["git", "init"], cwd=temp_dir, check=True)

            manager = GitRollbackManager(temp_dir)
            assert (
                manager.repo_path == Path(temp_dir).resolve()
            )  # nosec: B101 - Test assertion
            assert len(manager.operations_history) == 0  # nosec: B101 - Test assertion

    def test_get_current_commit_and_branch(self):
        """Test getting current commit and branch."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize a git repository
            subprocess.run(["git", "init"], cwd=temp_dir, check=True)
            subprocess.run(
                ["git", "config", "user.name", "Test"], cwd=temp_dir, check=True
            )
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=temp_dir,
                check=True,
            )

            # Create initial commit
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("test content")
            subprocess.run(["git", "add", "test.txt"], cwd=temp_dir, check=True)
            subprocess.run(
                ["git", "commit", "-m", "Initial commit"], cwd=temp_dir, check=True
            )

            manager = GitRollbackManager(temp_dir)
            commit_hash = manager.get_current_commit()
            branch_name = manager.get_current_branch()

            assert len(commit_hash) == 40  # SHA-1 hash  # nosec: B101 - Test assertion
            assert (
                branch_name == "main" or branch_name == "master"
            )  # nosec: B101 - Test assertion

    def test_stage_files(self):
        """Test file staging with rollback capability."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize git repository
            subprocess.run(
                ["git", "init"], cwd=temp_dir, check=True, capture_output=True
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"],
                cwd=temp_dir,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=temp_dir,
                check=True,
                capture_output=True,
            )

            # Create initial commit so there's something to rollback to
            readme = Path(temp_dir) / "README.md"
            readme.write_text("# Test repo")
            subprocess.run(
                ["git", "add", "README.md"],
                cwd=temp_dir,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "Initial commit"],
                cwd=temp_dir,
                check=True,
                capture_output=True,
            )

            manager = GitRollbackManager(temp_dir)

            # Create test files
            test_file1 = Path(temp_dir) / "test1.py"
            test_file2 = Path(temp_dir) / "test2.py"
            test_file1.write_text("# content 1")
            test_file2.write_text("# content 2")

            result = manager.stage_files(["test1.py", "test2.py"])
            assert result.is_valid  # nosec: B101 - Test assertion
            assert len(manager.operations_history) == 1  # nosec: B101 - Test assertion
            assert (
                manager.operations_history[0].operation_type == "stage"
            )  # nosec: B101 - Test assertion

    def test_create_branch(self):
        """Test branch creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize git repository
            subprocess.run(["git", "init"], cwd=temp_dir, check=True)
            subprocess.run(
                ["git", "config", "user.name", "Test"], cwd=temp_dir, check=True
            )
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=temp_dir,
                check=True,
            )

            manager = GitRollbackManager(temp_dir)

            # Create new branch
            result = manager.create_branch("test-branch")
            assert result.is_valid  # nosec: B101 - Test assertion
            assert (
                "Created and checked out branch: test-branch" in result.warnings[0]
            )  # nosec: B101 - Test assertion


class TestProgressTracking:
    """Test cases for progress tracking."""

    def test_progress_tracker_initialization(self):
        """Test ProgressTracker initialization."""
        tracker = ProgressTracker("test_exp", "test_participant", 100)

        assert tracker.experiment_name == "test_exp"  # nosec: B101 - Test assertion
        assert (
            tracker.participant_id == "test_participant"
        )  # nosec: B101 - Test assertion
        assert tracker.progress.total_trials == 100  # nosec: B101 - Test assertion
        assert tracker.progress.current_trial == 0  # nosec: B101 - Test assertion
        assert tracker.progress.completed_trials == 0  # nosec: B101 - Test assertion
        assert tracker.progress.status == "running"  # nosec: B101 - Test assertion

    def test_trial_recording(self):
        """Test trial recording."""
        tracker = ProgressTracker("test_exp", "test_participant", 10)

        # Record a trial
        trial_result = TrialResult(
            trial_number=1,
            timestamp=time.time(),
            response_time=500.0,
            accuracy=0.8,
            apgi_metrics={"ignition_rate": 0.3, "surprise": 0.5},
        )

        tracker.complete_trial(trial_result)

        assert tracker.progress.completed_trials == 1  # nosec: B101 - Test assertion
        assert len(tracker.progress.trials) == 1  # nosec: B101 - Test assertion
        assert (
            tracker.progress.trials[0].trial_number == 1
        )  # nosec: B101 - Test assertion

    def test_progress_calculations(self):
        """Test progress calculations."""
        tracker = ProgressTracker("test_exp", "test_participant", 10)

        # Add some trials
        for i in range(5):
            trial_result = TrialResult(
                trial_number=i + 1,
                timestamp=time.time(),
                response_time=500.0 + i * 10,
                accuracy=0.8 + i * 0.02,
            )
            tracker.complete_trial(trial_result)

        # Test progress percentage
        assert (
            tracker.get_progress_percentage() == 50.0
        )  # 5/10  # nosec: B101 - Test assertion

        # Test trial statistics
        stats = tracker.get_trial_statistics()
        assert stats["completed_trials"] == 5  # nosec: B101 - Test assertion
        assert (
            stats["avg_response_time"] == 520.0
        )  # (500 + 510 + 520 + 530 + 540) / 5  # nosec: B101 - Test assertion
        assert stats["avg_accuracy"] == pytest.approx(  # nosec: B101 - Test assertion
            0.84
        )  # (0.80 + 0.82 + 0.84 + 0.86 + 0.88) / 5

    def test_experiment_completion(self):
        """Test experiment completion."""
        tracker = ProgressTracker("test_exp", "test_participant", 10)

        tracker.complete_experiment("completed")

        assert tracker.progress.status == "completed"  # nosec: B101 - Test assertion
        assert tracker.progress.end_time is not None  # nosec: B101 - Test assertion
        assert tracker.get_elapsed_time() > 0  # nosec: B101 - Test assertion

    def test_export_summary(self):
        """Test summary export."""
        tracker = ProgressTracker("test_exp", "test_participant", 10)

        # Add a trial
        trial_result = TrialResult(
            trial_number=1,
            timestamp=time.time(),
            response_time=500.0,
            accuracy=0.8,
            apgi_metrics={"ignition_rate": 0.3},
        )
        tracker.complete_trial(trial_result)

        summary = tracker.export_summary()

        expected_sections = [
            "experiment_info",
            "progress",
            "trial_statistics",
            "apgi_statistics",
            "errors",
        ]
        for section in expected_sections:
            assert section in summary  # nosec: B101 - Test assertion

        assert summary["progress"]["total_trials"] == 10  # nosec: B101 - Test assertion
        assert (
            summary["progress"]["completed_trials"] == 1
        )  # nosec: B101 - Test assertion


class TestPerformanceMonitoring:
    """Test cases for performance monitoring."""

    def test_performance_monitor_initialization(self):
        """Test PerformanceMonitor initialization."""
        monitor = PerformanceMonitor()

        assert monitor.output_dir.exists()  # nosec: B101 - Test assertion
        assert not monitor.is_monitoring  # nosec: B101 - Test assertion
        assert len(monitor.memory_history) == 0  # nosec: B101 - Test assertion
        assert len(monitor.cpu_history) == 0  # nosec: B101 - Test assertion
        assert len(monitor.operation_metrics) == 0  # nosec: B101 - Test assertion

    def test_memory_snapshot(self):
        """Test memory snapshot creation."""
        monitor = PerformanceMonitor()

        snapshot = monitor._take_memory_snapshot()

        assert isinstance(snapshot.rss_mb, (int, float))  # nosec: B101 - Test assertion
        assert isinstance(snapshot.vms_mb, (int, float))  # nosec: B101 - Test assertion
        assert isinstance(
            snapshot.percent, (int, float)
        )  # nosec: B101 - Test assertion
        assert isinstance(
            snapshot.available_mb, (int, float)
        )  # nosec: B101 - Test assertion
        assert isinstance(snapshot.gc_objects, int)  # nosec: B101 - Test assertion
        assert isinstance(snapshot.gc_stats, dict)  # nosec: B101 - Test assertion

    def test_cpu_snapshot(self):
        """Test CPU snapshot creation."""
        monitor = PerformanceMonitor()

        snapshot = monitor._take_cpu_snapshot()

        assert isinstance(
            snapshot.percent, (int, float)
        )  # nosec: B101 - Test assertion
        assert isinstance(snapshot.count, int)  # nosec: B101 - Test assertion
        assert isinstance(
            snapshot.freq_mhz, (int, float)
        )  # nosec: B101 - Test assertion
        assert isinstance(snapshot.load_avg, list)  # nosec: B101 - Test assertion

    def test_operation_monitoring(self):
        """Test operation monitoring."""
        monitor = PerformanceMonitor()

        # Start operation
        metrics = monitor.start_operation("test_operation")

        assert (
            metrics.operation_name == "test_operation"
        )  # nosec: B101 - Test assertion
        assert metrics.start_time > 0  # nosec: B101 - Test assertion
        assert metrics.end_time is None  # nosec: B101 - Test assertion
        assert metrics.memory_before is not None  # nosec: B101 - Test assertion
        assert metrics.cpu_before is not None  # nosec: B101 - Test assertion

        # Simulate some work
        time.sleep(0.01)

        # End operation
        monitor.end_operation(metrics, success=True)

        assert metrics.end_time is not None  # nosec: B101 - Test assertion
        assert metrics.duration is not None  # type: ignore[unreachable]  # nosec: B101 - Test assertion
        assert metrics.duration > 0  # nosec: B101 - Test assertion
        assert metrics.success is True  # nosec: B101 - Test assertion

    def test_performance_summary(self):
        """Test performance summary generation."""
        monitor = PerformanceMonitor()

        # Add some operations
        for i in range(5):
            metrics = monitor.start_operation(f"operation_{i}")
            time.sleep(0.001)
            monitor.end_operation(metrics, success=True)

        summary = monitor.get_performance_summary()

        assert summary["total_operations"] == 5  # nosec: B101 - Test assertion
        assert summary["successful_operations"] == 5  # nosec: B101 - Test assertion
        assert summary["failed_operations"] == 0  # nosec: B101 - Test assertion
        assert summary["success_rate"] == 1.0  # nosec: B101 - Test assertion
        assert "avg_duration" in summary  # nosec: B101 - Test assertion

    def test_memory_trend(self):
        """Test memory trend analysis."""
        monitor = PerformanceMonitor()

        # Add some memory snapshots
        for i in range(10):
            snapshot = MemorySnapshot(
                timestamp=time.time() + i,
                rss_mb=100.0 + i * 10,
                vms_mb=150.0 + i * 5,
                percent=50.0 + i * 2,
                available_mb=4000.0 - i * 100,
                gc_objects=1000 + i * 100,
                gc_stats={},
            )
            with monitor._lock:
                monitor.memory_history.append(snapshot)

        trend = monitor.get_memory_trend()

        assert "current_mb" in trend  # nosec: B101 - Test assertion
        assert "trend_direction" in trend  # nosec: B101 - Test assertion
        assert trend["trend_direction"] == "increasing"  # nosec: B101 - Test assertion
        assert trend["samples"] == 10  # nosec: B101 - Test assertion

    def test_performance_regression_detection(self):
        """Test performance regression detection."""
        monitor = PerformanceMonitor()

        # Add baseline operations (consistent duration)
        for i in range(15):
            metrics = monitor.start_operation(f"baseline_{i}")
            time.sleep(0.005)  # 5ms baseline
            monitor.end_operation(metrics, success=True)

        # Add recent operations (significantly slower)
        for i in range(10):
            metrics = monitor.start_operation(f"recent_{i}")
            time.sleep(0.1)  # 100ms - much slower than 5ms
            monitor.end_operation(metrics, success=True)

        # Use windows that match our data: 15 baselines and 10 recents
        regression = monitor.detect_performance_regression(
            baseline_window=15, current_window=10
        )

        assert regression["status"] == "detected"  # nosec: B101 - Test assertion
        assert (
            regression["regression_percent"] > 30
        )  # Allow for timing variability  # nosec: B101 - Test assertion
        assert regression["significance"] in [  # nosec: B101 - Test assertion
            "moderate_regression",
            "significant_regression",
        ]


# Test configuration
if __name__ == "__main__":
    pytest.main([__file__])
