"""
Focused tests for experiment modules to improve coverage.

Tests the key constants and classes across experiment files.
"""

import sys
from pathlib import Path

import pytest

# Add experiments directory to path
experiments_dir = Path(__file__).parent.parent / "experiments"
sys.path.insert(0, str(experiments_dir))

# Import specific experiment modules that we know exist
import prepare_ai_benchmarking
import prepare_attentional_blink
import prepare_go_no_go
import prepare_stroop_effect


class TestPrepareAIBenchmarking:
    """Test prepare_ai_benchmarking module."""

    def test_constants_exist(self):
        """Test that required constants are defined."""
        assert hasattr(
            prepare_ai_benchmarking, "TIME_BUDGET"
        )  # nosec: B101 - Test assertion
        assert hasattr(
            prepare_ai_benchmarking, "NUM_TRIALS"
        )  # nosec: B101 - Test assertion
        assert hasattr(
            prepare_ai_benchmarking, "APGI_ENABLED"
        )  # nosec: B101 - Test assertion

    def test_constants_values(self):
        """Test constant values."""
        assert (
            prepare_ai_benchmarking.TIME_BUDGET == 600
        )  # nosec: B101 - Test assertion
        assert prepare_ai_benchmarking.NUM_TRIALS == 50  # nosec: B101 - Test assertion
        assert (
            prepare_ai_benchmarking.APGI_ENABLED is True
        )  # nosec: B101 - Test assertion

    def test_benchmark_type_enum(self):
        """Test BenchmarkType enum."""
        assert hasattr(
            prepare_ai_benchmarking, "BenchmarkType"
        )  # nosec: B101 - Test assertion
        benchmark_types = list(prepare_ai_benchmarking.BenchmarkType)
        assert (
            prepare_ai_benchmarking.BenchmarkType.REASONING in benchmark_types
        )  # nosec: B101 - Test assertion
        assert (
            prepare_ai_benchmarking.BenchmarkType.MEMORY in benchmark_types
        )  # nosec: B101 - Test assertion
        assert (
            prepare_ai_benchmarking.BenchmarkType.ATTENTION in benchmark_types
        )  # nosec: B101 - Test assertion

    def test_difficulty_enum(self):
        """Test Difficulty enum."""
        assert hasattr(
            prepare_ai_benchmarking, "Difficulty"
        )  # nosec: B101 - Test assertion
        difficulties = list(prepare_ai_benchmarking.Difficulty)
        assert (
            prepare_ai_benchmarking.Difficulty.EASY in difficulties
        )  # nosec: B101 - Test assertion
        assert (
            prepare_ai_benchmarking.Difficulty.MEDIUM in difficulties
        )  # nosec: B101 - Test assertion
        assert (
            prepare_ai_benchmarking.Difficulty.HARD in difficulties
        )  # nosec: B101 - Test assertion

    def test_ai_benchmark_generator_class(self):
        """Test AIBenchmarkGenerator class."""
        assert hasattr(
            prepare_ai_benchmarking, "AIBenchmarkGenerator"
        )  # nosec: B101 - Test assertion

        generator = prepare_ai_benchmarking.AIBenchmarkGenerator()
        assert hasattr(generator, "create_trial")  # nosec: B101 - Test assertion
        assert hasattr(generator, "reset")  # nosec: B101 - Test assertion
        assert callable(generator.create_trial)  # nosec: B101 - Test assertion
        assert callable(generator.reset)  # nosec: B101 - Test assertion

    def test_ai_benchmark_trial_dataclass(self):
        """Test AIBenchmarkTrial dataclass."""
        assert hasattr(
            prepare_ai_benchmarking, "AIBenchmarkTrial"
        )  # nosec: B101 - Test assertion

        # Test creating a trial with correct parameters
        trial = prepare_ai_benchmarking.AIBenchmarkTrial(
            trial_number=1,
            benchmark_type=prepare_ai_benchmarking.BenchmarkType.REASONING,
            difficulty=prepare_ai_benchmarking.Difficulty.MEDIUM,
            task_description="Test task",
            correct_answer="4",
        )
        assert trial.trial_number == 1  # nosec: B101 - Test assertion
        assert (
            trial.benchmark_type == prepare_ai_benchmarking.BenchmarkType.REASONING
        )  # nosec: B101 - Test assertion
        assert (
            trial.difficulty == prepare_ai_benchmarking.Difficulty.MEDIUM
        )  # nosec: B101 - Test assertion
        assert trial.task_description == "Test task"  # nosec: B101 - Test assertion
        assert trial.correct_answer == "4"  # nosec: B101 - Test assertion

    def test_ai_benchmark_experiment_class(self):
        """Test AIBenchmarkExperiment class."""
        assert hasattr(
            prepare_ai_benchmarking, "AIBenchmarkExperiment"
        )  # nosec: B101 - Test assertion

        experiment = prepare_ai_benchmarking.AIBenchmarkExperiment(num_trials=10)
        assert experiment.num_trials == 10  # nosec: B101 - Test assertion
        assert hasattr(experiment, "get_next_trial")  # nosec: B101 - Test assertion
        assert hasattr(experiment, "run_trial")  # nosec: B101 - Test assertion
        assert callable(experiment.get_next_trial)  # nosec: B101 - Test assertion

    def test_generator_create_trial(self):
        """Test trial creation."""
        generator = prepare_ai_benchmarking.AIBenchmarkGenerator(seed=42)
        trial = generator.create_trial(1)

        assert isinstance(
            trial, prepare_ai_benchmarking.AIBenchmarkTrial
        )  # nosec: B101 - Test assertion
        assert trial.trial_number == 1  # nosec: B101 - Test assertion
        assert isinstance(
            trial.benchmark_type, prepare_ai_benchmarking.BenchmarkType
        )  # nosec: B101 - Test assertion
        assert isinstance(
            trial.difficulty, prepare_ai_benchmarking.Difficulty
        )  # nosec: B101 - Test assertion
        assert trial.task_description != ""  # nosec: B101 - Test assertion
        assert trial.correct_answer != ""  # nosec: B101 - Test assertion

    def test_experiment_get_next_trial(self):
        """Test getting next trial from experiment."""
        experiment = prepare_ai_benchmarking.AIBenchmarkExperiment(
            num_trials=5, seed=42
        )

        trial = experiment.get_next_trial()
        assert trial is not None  # nosec: B101 - Test assertion
        assert isinstance(
            trial, prepare_ai_benchmarking.AIBenchmarkTrial
        )  # nosec: B101 - Test assertion

        # Get all trials
        trials = []
        while trial is not None:
            trials.append(trial)
            trial = experiment.get_next_trial()

        assert len(trials) == 5  # nosec: B101 - Test assertion


class TestPrepareAttentionalBlink:
    """Test prepare_attentional_blink module."""

    def test_constants_exist(self):
        """Test that required constants are defined."""
        assert hasattr(
            prepare_attentional_blink, "TIME_BUDGET"
        )  # nosec: B101 - Test assertion
        assert hasattr(
            prepare_attentional_blink, "NUM_TRIALS"
        )  # nosec: B101 - Test assertion
        assert hasattr(
            prepare_attentional_blink, "APGI_ENABLED"
        )  # nosec: B101 - Test assertion

    def test_constants_values(self):
        """Test constant values."""
        assert (
            prepare_attentional_blink.TIME_BUDGET == 600
        )  # nosec: B101 - Test assertion
        assert (
            prepare_attentional_blink.NUM_TRIALS == 100
        )  # nosec: B101 - Test assertion
        assert (
            prepare_attentional_blink.APGI_ENABLED is True
        )  # nosec: B101 - Test assertion


class TestPrepareGoNoGo:
    """Test prepare_go_no_go module."""

    def test_constants_exist(self):
        """Test that required constants are defined."""
        assert hasattr(prepare_go_no_go, "TIME_BUDGET")  # nosec: B101 - Test assertion
        assert hasattr(prepare_go_no_go, "NUM_TRIALS")  # nosec: B101 - Test assertion
        assert hasattr(prepare_go_no_go, "APGI_ENABLED")  # nosec: B101 - Test assertion

    def test_constants_values(self):
        """Test constant values."""
        assert prepare_go_no_go.TIME_BUDGET == 600  # nosec: B101 - Test assertion
        # NUM_TRIALS is tuned by the XPR AUTO optimiser, so assert what must
        # hold (a positive count that the experiment uses by default) rather
        # than a value that goes stale on every parameter update.
        assert isinstance(prepare_go_no_go.NUM_TRIALS, int)  # nosec: B101
        assert prepare_go_no_go.NUM_TRIALS > 0  # nosec: B101 - Test assertion
        assert (  # nosec: B101 - Test assertion
            prepare_go_no_go.GoNoGoExperiment().num_trials
            == prepare_go_no_go.NUM_TRIALS
        )
        assert prepare_go_no_go.APGI_ENABLED is True  # nosec: B101 - Test assertion


class TestPrepareStroopEffect:
    """Test prepare_stroop_effect module."""

    def test_constants_exist(self):
        """Test that required constants are defined."""
        assert hasattr(
            prepare_stroop_effect, "TIME_BUDGET"
        )  # nosec: B101 - Test assertion
        assert hasattr(
            prepare_stroop_effect, "NUM_TRIALS"
        )  # nosec: B101 - Test assertion
        assert hasattr(
            prepare_stroop_effect, "APGI_ENABLED"
        )  # nosec: B101 - Test assertion

    def test_constants_values(self):
        """Test constant values."""
        assert prepare_stroop_effect.TIME_BUDGET == 600  # nosec: B101 - Test assertion
        # NUM_TRIALS is tuned by the XPR AUTO optimiser; see the Go/No-Go test.
        assert isinstance(prepare_stroop_effect.NUM_TRIALS, int)  # nosec: B101
        assert prepare_stroop_effect.NUM_TRIALS > 0  # nosec: B101 - Test assertion
        assert (  # nosec: B101 - Test assertion
            prepare_stroop_effect.StroopExperiment().num_trials
            == prepare_stroop_effect.NUM_TRIALS
        )
        assert (
            prepare_stroop_effect.APGI_ENABLED is True
        )  # nosec: B101 - Test assertion


class TestExperimentIntegration:
    """Test integration between prepare modules."""

    def test_parameter_consistency(self):
        """Test that parameters are consistent between prepare modules."""
        # Check that TIME_BUDGET is consistent
        assert (
            prepare_ai_benchmarking.TIME_BUDGET == 600
        )  # nosec: B101 - Test assertion
        assert (
            prepare_attentional_blink.TIME_BUDGET == 600
        )  # nosec: B101 - Test assertion
        assert prepare_go_no_go.TIME_BUDGET == 600  # nosec: B101 - Test assertion
        assert prepare_stroop_effect.TIME_BUDGET == 600  # nosec: B101 - Test assertion

    def test_apgi_enabled_all_experiments(self):
        """Test that APGI is enabled in all experiment modules."""
        prepare_modules = [
            prepare_ai_benchmarking,
            prepare_attentional_blink,
            prepare_go_no_go,
            prepare_stroop_effect,
        ]

        for module in prepare_modules:
            assert hasattr(module, "APGI_ENABLED")  # nosec: B101 - Test assertion
            assert module.APGI_ENABLED is True  # nosec: B101 - Test assertion

    def test_all_modules_have_cli_integration(self):
        """Test that all modules have CLI integration."""
        prepare_modules = [
            prepare_ai_benchmarking,
            prepare_attentional_blink,
            prepare_go_no_go,
            prepare_stroop_effect,
        ]

        for module in prepare_modules:
            # Check that CLI functions are imported
            assert hasattr(module, "cli_entrypoint")  # nosec: B101 - Test assertion
            assert hasattr(
                module, "create_standard_parser"
            )  # nosec: B101 - Test assertion


if __name__ == "__main__":
    pytest.main([__file__])
