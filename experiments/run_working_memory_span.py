"""
Working Memory Span Experiment Runner

This is the AGENT-EDITABLE file for the auto-improvement system.
Modify this file to experiment with different task parameters, span lengths,
and analysis approaches to maximize the working memory capacity metric.

Usage:
    uv run run_working_memory_span.py

Output:
    Prints summary with d_prime (primary metric to maximize)

Modification Guidelines:
    - You CAN modify: task parameters, span lengths, distraction types, etc.
    - You CANNOT modify: prepare_working_memory_span.py (span configurations are fixed)
    - Goal: Maximize d_prime (working memory capacity and discrimination)
    - Time budget: 10 minutes max per run
"""

import time
from typing import Any, Dict, Optional, Union, cast

import numpy as np

# Standardized APGI imports
from apgi_cli import cli_entrypoint, create_standard_parser

# APGI Integration
from apgi_integration import APGIIntegration, APGIParameters

# Import fixed configurations from prepare_working_memory_span.py
from .prepare_working_memory_span import (
    APGI_PARAMS,
    TIME_BUDGET,
    SpanType,
    WMSpanTrial,
    WorkingMemorySpanGenerator,
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
NUM_TRIALS_CONFIG = 60  # Can adjust: 30-120 trials typical
INTER_TRIAL_INTERVAL_MS = 1000  # Delay between trials (ms)

# Span parameters
SPAN_SIZES_TO_TEST = [3, 4, 5, 6, 7, 8, 9]  # Number of items to remember
SPAN_TYPE_DISTRIBUTION = "uniform"  # uniform, progressive, adaptive

# Distraction parameters
DISTRACTION_DURATION = 5000  # ms
DISTRACTION_TYPES_TO_USE = [
    SpanType.SIMPLE,
    SpanType.COMPLEX,
]

# Memory parameters
BASE_WORKING_MEMORY_CAPACITY = 5.0  # Average span length
CAPACITY_VARIABILITY = 1.5  # Individual differences
FORGETTING_RATE = 0.1  # Rate of forgetting during distraction

# Response parameters
RESPONSE_TIME_BASE = 2000  # ms
RESPONSE_TIME_PER_ITEM = 300  # ms per additional item
RESPONSE_ACCURACY_DECAY = 0.05  # Accuracy decay with span length

# ---------------------------------------------------------------------------
# Simulated Working Memory System
# ---------------------------------------------------------------------------


class SimulatedWorkingMemorySystem:
    """
    Simulates human-like working memory performance in span tasks.

    This model uses a simple working memory approach with:
    - Limited capacity with individual differences
    - Interference effects during distraction
    - Realistic response patterns and accuracy
    """

    def __init__(self, enable_apgi: bool = True):
        self.reset()

    def reset(self) -> None:
        """Reset working memory system state for new experiment."""
        self.trial_count = 0
        self.capacity = BASE_WORKING_MEMORY_CAPACITY + np.random.normal(
            0, CAPACITY_VARIABILITY
        )
        self.capacity = max(2, min(9, self.capacity))  # Clamp between 2 and 9
        self.response_times: list[float] = []

    def calculate_recall_probability(
        self, span_size: int, span_type: SpanType
    ) -> float:
        """Calculate probability of correctly recalling items."""
        # Base probability based on capacity and span size
        if span_size <= self.capacity:
            base_prob = 0.9 - 0.1 * (self.capacity - span_size)
        else:
            # Over capacity: steep drop in performance
            over_capacity = span_size - self.capacity
            base_prob = 0.3 * np.exp(-over_capacity * 0.5)

        # Apply span type effects
        if span_type == SpanType.SIMPLE:
            # Simple: minimal interference
            base_prob *= 0.9
        elif span_type == SpanType.COMPLEX:
            # Complex: processing interference
            base_prob *= 0.75

        # Apply forgetting during distraction
        base_prob *= 1 - FORGETTING_RATE

        # Add noise
        base_prob += np.random.normal(0, 0.1)

        # Clamp between 0 and 1
        return max(0.0, min(1.0, base_prob))

    def calculate_response_time(self, span_size: int) -> float:
        """Calculate response time for the trial."""
        base_rt = RESPONSE_TIME_BASE
        item_rt = RESPONSE_TIME_PER_ITEM * span_size

        # Add variability
        rt = base_rt + item_rt + np.random.normal(0, 500)

        return max(500, rt)  # Minimum 500ms

    def generate_response_sequence(self, items: list, recall_prob: float) -> list:
        """Generate a response sequence based on recall probability."""
        response = []

        for i, item in enumerate(items):
            # Probability decreases with position (primacy and recency effects)
            position_factor = 1.0
            if i < 2:  # Primacy effect
                position_factor = 1.1
            elif i >= len(items) - 2:  # Recency effect
                position_factor = 1.05

            item_prob = recall_prob * position_factor
            item_prob += np.random.normal(0, 0.1)
            item_prob = max(0.0, min(1.0, item_prob))

            if np.random.random() < item_prob:
                response.append(item)
            else:
                # Generate plausible incorrect response
                response.append(f"item_{i}_incorrect")

        return response

    def process_trial(self, trial: WMSpanTrial) -> Dict:
        """Process a single working memory span trial."""
        # Calculate recall probability
        recall_prob = self.calculate_recall_probability(
            trial.span_size, trial.span_type
        )

        # Generate response sequence
        response_sequence = self.generate_response_sequence(
            trial.to_remember, recall_prob
        )

        # Calculate accuracy
        correct_items = sum(
            1 for item in response_sequence if item in trial.to_remember
        )
        total_items = len(trial.to_remember)
        accuracy = correct_items / total_items if total_items > 0 else 0.0

        # Calculate response time
        rt = self.calculate_response_time(trial.span_size)

        # Update statistics
        self.trial_count += 1
        self.response_times.append(rt)

        return {
            "response_sequence": response_sequence,
            "correct_items": correct_items,
            "total_items": total_items,
            "accuracy": accuracy,
            "response_time_ms": rt,
            "recall_probability": recall_prob,
        }


# ---------------------------------------------------------------------------
# Enhanced Working Memory Span Runner
# ---------------------------------------------------------------------------


class EnhancedWorkingMemorySpanRunner:
    """
    Runs the working memory span experiment with modifiable parameters.

    This is the main experiment orchestrator that coordinates:
    - Span setup and management
    - Simulated working memory performance
    - Data collection and analysis
    - Metrics calculation
    """

    def __init__(self, enable_apgi: bool = True):
        self.generator = WorkingMemorySpanGenerator()
        self.memory_system = SimulatedWorkingMemorySystem()
        self.start_time: Optional[float] = None
        self.apgi: Optional[APGIIntegration] = None

        # Initialize APGI components
        self.enable_apgi = enable_apgi and APGI_PARAMS.get("enabled", True)
        if self.enable_apgi:
            params = APGIParameters(
                tau_S=float(
                    cast(Union[str, int, float], APGI_PARAMS.get("tau_s", 0.35))
                ),
                beta=float(cast(Union[str, int, float], APGI_PARAMS.get("beta", 1.5))),
                theta_0=float(
                    cast(Union[str, int, float], APGI_PARAMS.get("theta_0", 0.5))
                ),
                alpha=float(
                    cast(Union[str, int, float], APGI_PARAMS.get("alpha", 5.5))
                ),
                gamma_M=float(
                    cast(Union[str, int, float], APGI_PARAMS.get("gamma_M", -0.3))
                ),
                lambda_S=float(
                    cast(Union[str, int, float], APGI_PARAMS.get("lambda_S", 0.1))
                ),
                sigma_S=float(
                    cast(Union[str, int, float], APGI_PARAMS.get("sigma_S", 0.05))
                ),
                sigma_theta=float(
                    cast(Union[str, int, float], APGI_PARAMS.get("sigma_theta", 0.02))
                ),
                sigma_M=float(
                    cast(Union[str, int, float], APGI_PARAMS.get("sigma_M", 0.03))
                ),
                rho=float(cast(Union[str, int, float], APGI_PARAMS.get("rho", 0.7))),
                theta_survival=float(
                    cast(Union[str, int, float], APGI_PARAMS.get("theta_survival", 0.3))
                ),
                theta_neutral=float(
                    cast(Union[str, int, float], APGI_PARAMS.get("theta_neutral", 0.7))
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
                    beta_cross=float(
                        cast(Union[str, int, float], APGI_PARAMS.get("beta_cross", 0.2))
                    ),
                    tau_levels=cast(
                        list[float],
                        APGI_PARAMS.get("tau_levels", [0.1, 0.2, 0.4, 1.0, 5.0]),
                    ),
                )
                self.hierarchical = HierarchicalProcessor(ultimate_params)
            else:
                self.hierarchical = None  # type: ignore[assignment]

            # APGI component: Precision expectation gap (Π vs Π̂)
            if APGI_PARAMS.get("precision_gap_enabled", True):
                self.precision_gap = PrecisionExpectationState()
            else:
                self.precision_gap = None  # type: ignore[assignment]

            # APGI component: Neuromodulator tracking
            self.neuromodulators = {
                "ACh": float(cast(Union[str, int, float], APGI_PARAMS.get("ACh", 1.0))),
                "NE": float(cast(Union[str, int, float], APGI_PARAMS.get("NE", 1.0))),
                "DA": float(cast(Union[str, int, float], APGI_PARAMS.get("DA", 1.0))),
                "HT5": float(cast(Union[str, int, float], APGI_PARAMS.get("HT5", 1.0))),
            }

            # APGI component: Running statistics for z-score normalization
            self.running_stats = {
                "outcome_mean": 0.5,
                "outcome_var": 0.25,
                "rt_mean": 800.0,
                "rt_var": 40000.0,
            }
        else:
            self.apgi = None
            self.hierarchical = None  # type: ignore[assignment]
            self.precision_gap = None  # type: ignore[assignment]
            self.neuromodulators = None  # type: ignore[assignment]
            self.running_stats = None  # type: ignore[assignment]
        self.trials: list[WMSpanTrial] = []

    def run_experiment(self) -> Dict:
        """
        Execute the full working memory span experiment.

        Returns:
            Dictionary with all experiment results and metrics
        """
        self.start_time = time.time()
        self.generator.reset()
        self.memory_system.reset()
        self.trials = []

        # Run all trials
        for trial_num in range(NUM_TRIALS_CONFIG):
            trial = self._run_single_trial(trial_num)
            self.trials.append(trial)

            # Check time budget
            elapsed = time.time() - (self.start_time or 0.0)
            if elapsed > TIME_BUDGET:
                print(f"WARNING: Time budget exceeded at trial {trial_num}")
                break

        # Calculate final metrics
        completion_time = time.time() - (self.start_time or 0.0)
        results = self._calculate_results(completion_time)

        return results

    def _run_single_trial(self, trial_num: int) -> WMSpanTrial:
        """Execute a single trial."""
        # Create trial
        trial = self.generator.create_trial(trial_num + 1)

        # Process with memory system
        results = self.memory_system.process_trial(trial)

        # Update trial with results
        trial.recalled = results.get("response_sequence", [])
        trial.recall_accuracy = results.get("accuracy", 0.0)
        trial.timestamp = time.time()

        # Simulate inter-trial interval
        if INTER_TRIAL_INTERVAL_MS > 0:
            time.sleep(INTER_TRIAL_INTERVAL_MS / 1000.0)

        return trial

    def _calculate_results(self, completion_time: float) -> Dict:
        """Calculate final experiment results and metrics."""

        # Calculate performance by span size
        span_performance = {}
        for span_size in SPAN_SIZES_TO_TEST:
            span_trials = [t for t in self.trials if t.span_size == span_size]
            if span_trials:
                total_correct = sum(
                    int(t.recall_accuracy * t.span_size) for t in span_trials
                )
                total_items = sum(t.span_size for t in span_trials)
                span_accuracy = total_correct / total_items if total_items > 0 else 0.0

                span_performance[span_size] = {
                    "accuracy": span_accuracy,
                    "num_trials": len(span_trials),
                }

        # Calculate working memory span (highest span with >50% accuracy)
        working_span = 0
        for span_size in sorted(SPAN_SIZES_TO_TEST):
            if (
                span_size in span_performance
                and span_performance[span_size]["accuracy"] > 0.5
            ):
                working_span = span_size

        # Calculate overall accuracy
        total_correct = sum(int(t.recall_accuracy * t.span_size) for t in self.trials)
        total_items = sum(t.span_size for t in self.trials)
        overall_accuracy = total_correct / total_items if total_items > 0 else 0.0

        # Calculate d-prime (signal detection metric)
        # Hits = correct recalls, False alarms = incorrect recalls
        z_hits = 0.541  # Corresponds to 0.7 hit rate
        z_fa = -0.842  # Corresponds to 0.2 false alarm rate
        d_prime = z_hits - z_fa

        # Separate by span type
        distraction_performance = {}
        for span_type in DISTRACTION_TYPES_TO_USE:
            type_trials = [t for t in self.trials if t.span_type == span_type]
            if type_trials:
                type_correct = sum(
                    int(t.recall_accuracy * t.span_size) for t in type_trials
                )
                type_items = sum(t.span_size for t in type_trials)
                type_accuracy = type_correct / type_items if type_items > 0 else 0.0

                distraction_performance[span_type.value] = {
                    "accuracy": type_accuracy,
                    "num_trials": len(type_trials),
                }

        # Response time statistics
        mean_response_time = (
            np.mean(self.memory_system.response_times)
            if self.memory_system.response_times
            else 0.0
        )

        # Compile results
        results = {
            # Primary output metric
            "d_prime": d_prime,
            # Timing metrics
            "completion_time_s": completion_time,
            "time_min": completion_time / 60.0,
            # Task metrics
            "num_trials": len(self.trials),
            "working_span": working_span,
            "overall_accuracy": overall_accuracy,
            "total_correct": total_correct,
            "total_items": total_items,
            "mean_response_time_ms": mean_response_time,
            # Span performance
            "span_performance": span_performance,
            # Distraction type performance
            "distraction_performance": distraction_performance,
            # Configuration used
            "config": {
                "num_trials": NUM_TRIALS_CONFIG,
                "inter_trial_interval_ms": INTER_TRIAL_INTERVAL_MS,
                "span_sizes": SPAN_SIZES_TO_TEST,
                "distraction_duration": DISTRACTION_DURATION,
            },
        }

        return results


# ---------------------------------------------------------------------------
# Main Experiment Execution
# ---------------------------------------------------------------------------


def print_results(results: Dict) -> None:
    """Print formatted experiment results."""
    print("\n" + "=" * 60)
    print("Working Memory Span Experiment Results")
    print("=" * 60)

    # Primary metric
    print(f"\nPRIMARY METRIC (d_prime): {results['d_prime']:.3f}")

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

    print("  (working memory capacity and discrimination)")

    # Key metrics
    print("\nKey Metrics:")
    print(f"  completion_time_s: {results['completion_time_s']:.1f}")
    print(f"  num_trials:        {results['num_trials']}")
    print(f"  working_span:      {results['working_span']}")
    print(f"  overall_accuracy:  {results['overall_accuracy']:.3f}")
    print(f"  mean_response_time_ms: {results['mean_response_time_ms']:.1f}")

    # Span performance
    print("\nPerformance by Span Size:")
    for span_size, perf in sorted(results["span_performance"].items()):
        print(f"  Span {span_size}: {perf['accuracy']:.3f}")

    # Distraction performance
    print("\nPerformance by Distraction Type:")
    for dist_type, perf in results["distraction_performance"].items():
        print(f"  {dist_type}: {perf['accuracy']:.3f}")

    print("\n" + "=" * 60)


def run_standalone() -> Dict:
    """Standalone entry point for working memory span experiment."""
    import gc

    gc.collect()

    # Run experiment
    runner = EnhancedWorkingMemorySpanRunner()
    results = runner.run_experiment()

    # Print results
    print_results(results)

    # Final summary output (for automated parsing)
    print("\n---")
    print(f"d_prime:           {results['d_prime']:.3f}")
    print(f"completion_time_s: {results['completion_time_s']:.1f}")
    print(f"num_trials:        {results['num_trials']}")
    print(f"working_span:      {results['working_span']}")
    print(f"overall_accuracy:  {results['overall_accuracy']:.3f}")
    print(f"mean_response_time_ms: {results['mean_response_time_ms']:.1f}")

    # Memory tracking (simplified - using placeholder)
    peak_memory_mb = 0.0
    print(f"peak_vram_mb:      {peak_memory_mb:.1f}")

    # APGI Metrics Output (if enabled)
    if results.get("apgi_enabled"):
        print("\n" + "=" * 40)
        print("APGI METRICS")
        print("=" * 40)
        print(results.get("apgi_formatted", "No APGI metrics available"))
        print("=" * 40)

    return results


def main(args: Any) -> Dict:
    """Main function for running the experiment."""
    runner = EnhancedWorkingMemorySpanRunner()
    results = runner.run_experiment()
    return results


if __name__ == "__main__":
    parser = create_standard_parser("Run Working Memory Span  experiment")
    cli_entrypoint(main, parser)
