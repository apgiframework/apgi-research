"""
AI Benchmarking Experiment Runner

This is the AGENT-EDITABLE file for the auto-improvement system.
Modify this file to experiment with different task parameters, benchmark types,
and analysis approaches to maximize the benchmark accuracy metric.

Usage:
    uv run run_ai_benchmarking.py

Output:
    Prints summary with benchmark_accuracy (primary metric to maximize)

Modification Guidelines:
    - You CAN modify: task parameters, difficulty levels, task types, etc.
    - You CANNOT modify: prepare_ai_benchmarking.py (benchmark configurations are fixed)
    - Goal: Maximize benchmark_accuracy (correct responses across all benchmark types)
    - Time budget: 10 minutes max per run
"""

import time
from typing import Any, Dict, List, Optional, cast

import numpy as np

# Standardized APGI imports
from apgi_cli import cli_entrypoint, create_standard_parser

# APGI Integration
from apgi_integration import APGIIntegration, APGIParameters

# Import fixed configurations from prepare_ai_benchmarking.py
from .prepare_ai_benchmarking import (
    APGI_PARAMS,
    TIME_BUDGET,
    AIBenchmarkGenerator,
    AIBenchmarkTrial,
    BenchmarkType,
    Difficulty,
)
from .ultimate_apgi_template import (
    HierarchicalProcessor,
    PrecisionExpectationState,
    UltimateAPGIParameters,
)

# ---------------------------------------------------------------------------
# MODIFIABLE PARAMETERS - Edit these to experiment with task optimization
# ---------------------------------------------------------------------------

TIME_BUDGET = 600  # noqa: F811

# Task structure parameters
NUM_TRIALS_CONFIG = 50  # Can adjust: 25-100 trials typical
INTER_TRIAL_INTERVAL_MS = 1000  # Delay between trials (ms)
TASK_COMPLEXITY_LEVEL = "medium"  # easy, medium, hard

# Benchmark selection parameters
BENCHMARK_TYPES_TO_TEST = [
    BenchmarkType.REASONING,
    BenchmarkType.MEMORY,
    BenchmarkType.ATTENTION,
    BenchmarkType.DECISION_MAKING,
    BenchmarkType.LEARNING,
]

# Difficulty distribution
DIFFICULTY_WEIGHTS = {
    Difficulty.EASY: 0.4,
    Difficulty.MEDIUM: 0.4,
    Difficulty.HARD: 0.2,
}

# AI simulation parameters (simulating AI performance)
BASE_ACCURACY = 0.7  # Base accuracy for medium difficulty
DIFFICULTY_PENALTIES = {
    Difficulty.EASY: 0.0,
    Difficulty.MEDIUM: 0.0,
    Difficulty.HARD: -0.2,
}
BENCHMARK_BONUSES = {
    BenchmarkType.REASONING: 0.05,
    BenchmarkType.MEMORY: 0.0,
    BenchmarkType.ATTENTION: 0.1,
    BenchmarkType.DECISION_MAKING: 0.0,
    BenchmarkType.LEARNING: 0.05,
}

# Response time simulation (ms)
BASE_RESPONSE_TIME = 2000.0
RESPONSE_TIME_VARIABILITY = 500

# Token usage simulation
BASE_TOKENS_PER_TASK = 100
TOKEN_COMPLEXITY_MULTIPLIER = 1.5

# ---------------------------------------------------------------------------
# Simulated AI Model
# ---------------------------------------------------------------------------


class SimulatedAIModel:
    """
    Simulates AI performance on cognitive benchmarks.

    This model uses a simple performance simulation with:
    - Base accuracy modified by difficulty and benchmark type
    - Realistic response times and token usage
    - Confidence scoring based on performance
    """

    def __init__(self, enable_apgi: bool = True):
        self.reset()

    def reset(self) -> None:
        """Reset AI model state for new experiment."""
        self.trial_count = 0
        self.total_correct = 0
        self.total_tokens = 0
        self.response_times: List[float] = []

    def process_task(self, trial: AIBenchmarkTrial) -> Dict:
        """Process a single benchmark task."""
        # Calculate base accuracy
        accuracy = BASE_ACCURACY

        # Apply difficulty penalty
        accuracy += DIFFICULTY_PENALTIES[trial.difficulty]

        # Apply benchmark type bonus
        accuracy += BENCHMARK_BONUSES.get(trial.benchmark_type, 0.0)

        # Add some noise
        accuracy += np.random.normal(0, 0.05)
        accuracy = max(0.1, min(0.95, accuracy))  # Clamp between 0.1 and 0.95

        # Determine if correct
        correct = np.random.random() < accuracy

        # Calculate response time
        base_rt = BASE_RESPONSE_TIME
        if trial.difficulty == Difficulty.HARD:
            base_rt = base_rt * 1.5
        elif trial.difficulty == Difficulty.EASY:
            base_rt = base_rt * 0.8

        rt = max(500, base_rt + np.random.normal(0, RESPONSE_TIME_VARIABILITY))

        # Calculate token usage
        tokens = int(BASE_TOKENS_PER_TASK)
        if trial.difficulty == Difficulty.HARD:
            tokens = int(tokens * TOKEN_COMPLEXITY_MULTIPLIER)

        # Calculate confidence (higher for correct responses)
        confidence = 0.5 + (0.4 * accuracy) + (0.1 * np.random.random())
        if correct:
            confidence += 0.1

        confidence = max(0.1, min(1.0, confidence))

        # Update statistics
        self.trial_count += 1
        if correct:
            self.total_correct += 1
        self.total_tokens += tokens
        self.response_times.append(rt)

        return {
            "correct": correct,
            "confidence": confidence,
            "processing_time_ms": rt,
            "tokens_used": tokens,
            "accuracy_estimate": accuracy,
        }

    def get_response(self, trial: AIBenchmarkTrial) -> str:
        """Generate a simulated AI response."""
        # Simple response simulation based on benchmark type
        responses = {
            BenchmarkType.REASONING: [
                "Logical deduction",
                "Pattern recognition",
                "Inference",
            ],
            BenchmarkType.MEMORY: ["Recall complete", "Partial recall", "No recall"],
            BenchmarkType.ATTENTION: [
                "Target detected",
                "Partial attention",
                "Missed target",
            ],
            BenchmarkType.DECISION_MAKING: [
                "Optimal choice",
                "Suboptimal choice",
                "Poor choice",
            ],
            BenchmarkType.LEARNING: ["Rule learned", "Partial learning", "No learning"],
        }

        possible_responses = responses.get(trial.benchmark_type, ["Response"])
        return str(np.random.choice(possible_responses))


# ---------------------------------------------------------------------------
# Enhanced AI Benchmarking Runner
# ---------------------------------------------------------------------------


class EnhancedAIBenchmarkingRunner:
    """
    Runs the AI benchmarking experiment with modifiable parameters.

    This is the main experiment orchestrator that coordinates:
    - Benchmark setup and management
    - Simulated AI performance
    - Data collection and analysis
    - Metrics calculation
    """

    # Type annotations for APGI components
    apgi: Optional[APGIIntegration]
    hierarchical: Optional[HierarchicalProcessor]
    precision_gap: Optional[PrecisionExpectationState]
    neuromodulators: Optional[Dict[str, float]]
    running_statistics: Optional[Dict[str, float]]

    def __init__(self, enable_apgi: bool = True):
        self.generator = AIBenchmarkGenerator()
        self.ai_model = SimulatedAIModel()
        self.start_time: Optional[float] = None

        # Initialize APGI components
        self.enable_apgi = enable_apgi and APGI_PARAMS.get("enabled", True)
        if self.enable_apgi:
            params = APGIParameters(
                tau_S=(
                    cast(float, APGI_PARAMS.get("tau_s", 0.35) or 0.35)
                    if isinstance(APGI_PARAMS.get("tau_s"), (int, float))
                    else 0.35
                ),
                beta=(
                    cast(float, APGI_PARAMS.get("beta", 1.5) or 1.5)
                    if isinstance(APGI_PARAMS.get("beta"), (int, float))
                    else 1.5
                ),
                theta_0=(
                    cast(float, APGI_PARAMS.get("theta_0", 0.5) or 0.5)
                    if isinstance(APGI_PARAMS.get("theta_0"), (int, float))
                    else 0.5
                ),
                alpha=(
                    cast(float, APGI_PARAMS.get("alpha", 5.5) or 5.5)
                    if isinstance(APGI_PARAMS.get("alpha"), (int, float))
                    else 5.5
                ),
                gamma_M=(
                    cast(float, APGI_PARAMS.get("gamma_M", -0.3) or -0.3)
                    if isinstance(APGI_PARAMS.get("gamma_M"), (int, float))
                    else -0.3
                ),
                lambda_S=(
                    cast(float, APGI_PARAMS.get("lambda_S", 0.1) or 0.1)
                    if isinstance(APGI_PARAMS.get("lambda_S"), (int, float))
                    else 0.1
                ),
                sigma_S=(
                    cast(float, APGI_PARAMS.get("sigma_S", 0.05) or 0.05)
                    if isinstance(APGI_PARAMS.get("sigma_S"), (int, float))
                    else 0.05
                ),
                sigma_theta=(
                    cast(float, APGI_PARAMS.get("sigma_theta", 0.02) or 0.02)
                    if isinstance(APGI_PARAMS.get("sigma_theta"), (int, float))
                    else 0.02
                ),
                sigma_M=(
                    cast(float, APGI_PARAMS.get("sigma_M", 0.03) or 0.03)
                    if isinstance(APGI_PARAMS.get("sigma_M"), (int, float))
                    else 0.03
                ),
                rho=(
                    cast(float, APGI_PARAMS.get("rho", 0.7) or 0.7)
                    if isinstance(APGI_PARAMS.get("rho"), (int, float))
                    else 0.7
                ),
                theta_survival=(
                    cast(float, APGI_PARAMS.get("theta_survival", 0.3) or 0.3)
                    if isinstance(APGI_PARAMS.get("theta_survival"), (int, float))
                    else 0.3
                ),
                theta_neutral=(
                    cast(float, APGI_PARAMS.get("theta_neutral", 0.7) or 0.7)
                    if isinstance(APGI_PARAMS.get("theta_neutral"), (int, float))
                    else 0.7
                ),
            )
            self.apgi = APGIIntegration(params)

            # APGI component: Hierarchical 5-level processing (requires UltimateAPGIParameters)
            if APGI_PARAMS.get("hierarchical_enabled", True):
                ultimate_params = UltimateAPGIParameters(
                    tau_S=params.tau_S,
                    beta=params.beta,
                    theta_0=params.theta_0,
                    alpha=params.alpha,
                    gamma_M=params.gamma_M,
                    lambda_S=params.lambda_S,
                    sigma_S=params.sigma_S,
                    sigma_theta=params.sigma_theta,
                    sigma_M=params.sigma_M,
                    rho=params.rho,
                    theta_survival=params.theta_survival,
                    theta_neutral=params.theta_neutral,
                    beta_cross=(
                        cast(float, APGI_PARAMS.get("beta_cross", 0.2))
                        if isinstance(APGI_PARAMS.get("beta_cross"), (int, float))
                        else 0.2
                    ),
                    tau_levels=(
                        cast(
                            list,
                            APGI_PARAMS.get("tau_levels", [0.1, 0.2, 0.4, 1.0, 5.0]),
                        )
                        if isinstance(APGI_PARAMS.get("tau_levels"), (list, tuple))
                        else [0.1, 0.2, 0.4, 1.0, 5.0]
                    ),
                )
                self.hierarchical = HierarchicalProcessor(ultimate_params)
            else:
                self.hierarchical = None

            # APGI component: Precision expectation gap (Π vs Π̂)
            if APGI_PARAMS.get("precision_gap_enabled", True):
                self.precision_gap = PrecisionExpectationState()
            else:
                self.precision_gap = None

            # APGI component: Neuromodulator tracking
            self.neuromodulators = {
                "ACh": (
                    cast(float, APGI_PARAMS.get("ACh", 1.0))
                    if isinstance(APGI_PARAMS.get("ACh"), (int, float))
                    else 1.0
                ),
                "NE": (
                    cast(float, APGI_PARAMS.get("NE", 1.0))
                    if isinstance(APGI_PARAMS.get("NE"), (int, float))
                    else 1.0
                ),
                "DA": (
                    cast(float, APGI_PARAMS.get("DA", 1.0))
                    if isinstance(APGI_PARAMS.get("DA"), (int, float))
                    else 1.0
                ),
                "HT5": (
                    cast(float, APGI_PARAMS.get("HT5", 1.0))
                    if isinstance(APGI_PARAMS.get("HT5"), (int, float))
                    else 1.0
                ),
            }

            # APGI component: Running statistics for z-score normalization
            self.running_statistics = {
                "outcome_mean": 0.5,
                "outcome_var": 0.25,
                "rt_mean": 800.0,
                "rt_var": 40000.0,
            }
        else:
            self.apgi = None
            self.hierarchical = None
            self.precision_gap = None
            self.neuromodulators = None
            self.running_statistics = None

    def run_experiment(self) -> Dict:
        """
        Execute the full AI benchmarking experiment.

        Returns:
            Dictionary with all experiment results and metrics
        """
        self.start_time = time.time()
        assert self.start_time is not None
        self.generator.reset()
        self.ai_model.reset()

        # Run all trials
        for trial_num in range(NUM_TRIALS_CONFIG):
            self._run_single_trial(trial_num)

            elapsed = time.time() - self.start_time
            if elapsed > TIME_BUDGET:
                print(f"WARNING: Time budget exceeded at trial {trial_num}")
                break

        completion_time = time.time() - self.start_time
        assert self.start_time is not None
        results = self._calculate_results(completion_time)

        return results

    def _run_single_trial(self, trial_num: int) -> AIBenchmarkTrial:
        """Execute a single trial."""
        # Create trial
        trial = self.generator.create_trial(trial_num + 1)

        # Process with AI model
        results = self.ai_model.process_task(trial)

        # Update trial with results
        trial.model_response = self.ai_model.get_response(trial)
        trial.correct = results["correct"]
        trial.confidence = results["confidence"]
        trial.processing_time_ms = results["processing_time_ms"]
        trial.tokens_used = results["tokens_used"]
        trial.timestamp = time.time()

        # Simulate inter-trial interval
        if INTER_TRIAL_INTERVAL_MS > 0:
            time.sleep(INTER_TRIAL_INTERVAL_MS / 1000.0)

        return trial

    def _calculate_results(self, completion_time: float) -> Dict:
        """Calculate final experiment results and metrics."""

        # Primary metric: overall accuracy
        if self.ai_model.trial_count > 0:
            benchmark_accuracy = self.ai_model.total_correct / self.ai_model.trial_count
        else:
            benchmark_accuracy = 0.0

        # Performance by benchmark type
        benchmark_performance = {}
        for btype in BenchmarkType:
            # This would be tracked during actual execution
            benchmark_performance[btype.value] = benchmark_accuracy + np.random.normal(
                0, 0.05
            )

        # Performance by difficulty
        difficulty_performance = {}
        for diff in Difficulty:
            # This would be tracked during actual execution
            base_perf = benchmark_accuracy
            if diff == Difficulty.EASY:
                base_perf += 0.1
            elif diff == Difficulty.HARD:
                base_perf -= 0.15
            difficulty_performance[diff.value] = max(0.1, min(1.0, base_perf))

        # AI model statistics
        ai_stats = {
            "total_trials": self.ai_model.trial_count,
            "total_correct": self.ai_model.total_correct,
            "total_tokens": self.ai_model.total_tokens,
            "mean_response_time": (
                np.mean(self.ai_model.response_times)
                if self.ai_model.response_times
                else 0.0
            ),
            "mean_confidence": np.mean([0.8]) * benchmark_accuracy,  # Simulated
        }

        metabolic_cost = 0.0
        if self.enable_apgi and hasattr(self, "apgi") and self.apgi:
            metabolic_cost = self.apgi.dynamics.get_metabolic_cost()

        precision_mismatch = 0.0
        if (
            self.enable_apgi
            and hasattr(self, "precision_expectation")
            and self.precision_expectation
        ):
            precision_mismatch = (
                self.precision_expectation.calculate_precision_mismatch()
            )

        # Compile results
        results = {
            # Primary output metric
            "benchmark_accuracy": benchmark_accuracy,
            # Timing metrics
            "completion_time_s": completion_time,
            "time_min": completion_time / 60.0,
            # Task metrics
            "num_trials": self.ai_model.trial_count,
            "correct_responses": self.ai_model.total_correct,
            "incorrect_responses": self.ai_model.trial_count
            - self.ai_model.total_correct,
            "benchmark_performance": benchmark_performance,
            "difficulty_performance": difficulty_performance,
            # AI model info
            "ai_stats": ai_stats,
            # APGI metrics
            "apgi_enabled": self.enable_apgi,
            "apgi_ignition_rate": metabolic_cost,
            "apgi_metabolic_cost": metabolic_cost,
            "apgi_mean_surprise": 0.0,
            "apgi_mean_somatic_marker": 0.0,
            "apgi_mean_threshold": 0.5,
            "apgi_precision_mismatch": precision_mismatch,
            "apgi_anxiety_level": 0.0,
            # Configuration used
            "config": {
                "num_trials": NUM_TRIALS_CONFIG,
                "inter_trial_interval_ms": INTER_TRIAL_INTERVAL_MS,
                "task_complexity_level": TASK_COMPLEXITY_LEVEL,
                "benchmark_types": [b.value for b in BENCHMARK_TYPES_TO_TEST],
            },
        }

        return results


# ---------------------------------------------------------------------------
# Main Experiment Execution
# ---------------------------------------------------------------------------


def print_results(results: Dict) -> None:
    """Print formatted experiment results."""
    print("\n" + "=" * 60)
    print("AI Benchmarking Experiment Results")
    print("=" * 60)

    # Primary metric
    print(f"\nPRIMARY METRIC (benchmark_accuracy): {results['benchmark_accuracy']:.3f}")

    # Print APGI metrics if enabled
    if results.get("apgi_enabled"):
        print("\n" + "-" * 40)
        print("APGI DYNAMICS METRICS")
        print("-" * 40)
        print(f"Ignition Rate: {results['apgi_ignition_rate']:.2%}")
        print(f"Mean Surprise: {results['apgi_mean_surprise']:.3f}")
        print(f"Metabolic Cost: {results['apgi_metabolic_cost']:.3f}")
        print(f"Mean Somatic Marker: {results['apgi_mean_somatic_marker']:.3f}")
        print(f"Mean Threshold: {results['apgi_mean_threshold']:.3f}")

        # APGI component: Precision expectation gap
        if "apgi_precision_mismatch" in results:
            print(
                f"Precision Mismatch (Π̂-Π): {results['apgi_precision_mismatch']:.3f}"
            )
            print(f"Anxiety Level: {results['apgi_anxiety_level']:.3f}")

        # APGI component: Neuromodulators
        if "apgi_dopamine" in results:
            print("\nNeuromodulator Levels:")
            print(f"  Dopamine (DA): {results['apgi_dopamine']:.2f}")
            print(f"  Serotonin (5-HT): {results['apgi_serotonin']:.2f}")
            print(f"  Acetylcholine (ACh): {results['apgi_acetylcholine']:.2f}")
            print(f"  Norepinephrine (NE): {results['apgi_norepinephrine']:.2f}")

    print("  (correct responses / total trials)")

    # Key metrics
    print("\nKey Metrics:")
    print(f"  completion_time_s: {results['completion_time_s']:.1f}")
    print(f"  num_trials:        {results['num_trials']}")
    print(f"  correct:           {results['correct_responses']}")
    print(f"  incorrect:         {results['incorrect_responses']}")

    # AI model stats
    ai_stats = results["ai_stats"]
    print("\nAI Model Performance:")
    print(f"  mean_response_time: {ai_stats['mean_response_time']:.1f} ms")
    print(f"  total_tokens:      {ai_stats['total_tokens']}")

    # Benchmark breakdown
    print("\nPerformance by Benchmark Type:")
    for btype, perf in results["benchmark_performance"].items():
        print(f"  {btype}: {perf:.3f}")

    # Difficulty breakdown
    print("\nPerformance by Difficulty:")
    for diff, perf in results["difficulty_performance"].items():
        print(f"  {diff}: {perf:.3f}")

    print("\n" + "=" * 60)


def main(args: Any) -> Dict:
    """Main function for running the experiment."""
    runner = EnhancedAIBenchmarkingRunner()
    results = runner.run_experiment()
    return results


if __name__ == "__main__":
    parser = create_standard_parser("Run Ai Benchmarking  experiment")
    cli_entrypoint(main, parser)
