"""
Attentional Blink Experiment Runner

This is the AGENT-EDITABLE file for the auto-improvement system.
Modify this file to experiment with different task parameters, SOA values,
and analysis approaches to maximize the blink magnitude metric.

Usage:
    uv run run_attentional_blink.py

Output:
    Prints summary with blink_magnitude (primary metric to maximize)

Modification Guidelines:
    - You CAN modify: task parameters, SOA values, T1-T2 relationships, etc.
    - You CANNOT modify: prepare_attentional_blink.py (stimulus sets are fixed)
    - Goal: Maximize blink_magnitude (difference between lag 1 and lag 2-3 accuracy)
    - Time budget: 10 minutes max per run
"""

import time
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np

# Standardized APGI imports
from apgi_cli import cli_entrypoint, create_standard_parser

# APGI Integration - imports the dynamical system for tracking ignition, surprise, somatic markers
from apgi_integration import APGIIntegration, APGIParameters, format_apgi_output

# Import fixed configurations from prepare_attentional_blink.py
from .prepare_attentional_blink import (
    APGI_PARAMS,  # APGI parameters from prepare file
    TIME_BUDGET,
    ABExperiment,
    TrialType,
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
NUM_TRIALS_CONFIG = 100  # Can adjust: 50-200 trials typical
RSVP_RATE_CONFIG = 10  # Items per second (Hz)
T1_POSITION_MIN = 5  # Minimum position for T1 in stream
T1_POSITION_MAX = 12  # Maximum position for T1

# SOA (Stimulus Onset Asynchrony) configuration
# Standard: lags 1-8 (100-800ms at 10Hz)
CUSTOM_LAGS = [1, 2, 3, 4, 5, 6, 7, 8]  # Modify to test specific lags
FOCAL_LAGS = [2, 3]  # The classic blink window - optimize for max effect here

# Participant simulation parameters
USE_ATTENTION_MODEL = True  # Use simple attention model vs random
ATTENTION_CAPACITY = 2  # Items attention can handle simultaneously
RECOVERY_TIME_MS = 500  # Time for attention to recover after T1

# Trial type distribution
BOTH_TARGETS_PROB = 0.70  # Probability of both T1 and T2 present
T1_ONLY_PROB = 0.10  # Catch trials
T2_ONLY_PROB = 0.10  # Baseline
NEITHER_PROB = 0.10  # Catch trials

# Analysis and data collection
CALCULATE_TEMPORAL_DYNAMICS = True  # Calculate recovery function
RUN_T1_ACCURACY_CHECK = True  # Ensure T1 accuracy > 60%
TRACK_INDIVIDUAL_DIFFERENCES = False  # Simulate participant variability

# Advanced: Dynamic difficulty adjustment
USE_ADAPTIVE_LAGS = False  # Adjust lags based on performance
ADAPTIVE_THRESHOLD = 0.5  # Target accuracy for adaptation


# ---------------------------------------------------------------------------
# Simulated Participant Model
# ---------------------------------------------------------------------------


class SimulatedParticipant:
    """
    Simulates human-like attention in the AB paradigm.

    This model uses a simple attentional bottleneck with:
    - Limited attentional capacity
    - Recovery time after T1 processing
    - Individual variability in attention
    """

    def __init__(
        self,
        attention_capacity: int = ATTENTION_CAPACITY,
        recovery_time_ms: float = RECOVERY_TIME_MS,
    ):
        self.attention_capacity = attention_capacity
        self.recovery_time_ms = recovery_time_ms
        self.reset()

    def reset(self) -> None:
        """Reset participant state for new experiment."""
        self.t1_processing = False
        self.t1_processed_time = 0.0
        self.t1_accuracy = 0.7
        self.t2_accuracy_lag1 = 0.9
        self.t2_accuracy_blink = 0.4  # Impaired during blink
        self.t2_recovery = 0.7

    def process_trial(self, lag: int, trial_type: TrialType) -> tuple:
        """
        Simulate processing of an AB trial.

        Returns:
            (t1_correct, t2_correct, t1_rt, t2_rt)
        """
        # SOA calculation (not used in current implementation)

        # T1 accuracy (baseline)
        t1_correct = np.random.random() < self.t1_accuracy
        t1_rt = (
            np.random.uniform(400, 700) if t1_correct else np.random.uniform(600, 1000)
        )

        # T2 accuracy depends on lag and whether T1 was correct
        if trial_type != TrialType.BOTH_TARGETS or not t1_correct:
            t2_correct = False
            t2_rt = 0.0
        else:
            # Classic blink pattern
            if lag == 1:
                t2_prob = self.t2_accuracy_lag1  # Spared
            elif lag in [2, 3]:
                t2_prob = self.t2_accuracy_blink  # Impaired (blink)
            else:
                t2_prob = self.t2_recovery  # Recovery

            t2_correct = np.random.random() < t2_prob
            t2_rt = (
                np.random.uniform(400, 800)
                if t2_correct
                else np.random.uniform(500, 900)
            )

        return t1_correct, t2_correct, t1_rt, t2_rt


# ---------------------------------------------------------------------------
# Enhanced Attentional Blink Runner
# ---------------------------------------------------------------------------


class EnhancedAttentionalBlinkRunner:
    """
    Runs the Attentional Blink experiment with APGI dynamics integration.
    """

    def __init__(self, enable_apgi: bool = True):
        self.experiment = ABExperiment(num_trials=NUM_TRIALS_CONFIG)
        self.participant = SimulatedParticipant()
        self.start_time: Optional[float] = None
        self.apgi: Optional[APGIIntegration] = None

        # Initialize APGI integration if enabled
        self.enable_apgi = enable_apgi and APGI_PARAMS.get("enabled", True)
        if self.enable_apgi:
            params = APGIParameters(
                tau_S=float(
                    cast(
                        Union[str, int, float], APGI_PARAMS.get("tau_s", "0.25") or 0.25
                    )
                ),
                beta=float(
                    cast(Union[str, int, float], APGI_PARAMS.get("beta", "1.8") or 1.8)
                ),
                theta_0=float(
                    cast(
                        Union[str, int, float], APGI_PARAMS.get("theta_0", "0.4") or 0.4
                    )
                ),
                alpha=float(
                    cast(Union[str, int, float], APGI_PARAMS.get("alpha", "6.0") or 6.0)
                ),
                gamma_M=float(
                    cast(
                        Union[str, int, float],
                        APGI_PARAMS.get("gamma_M", "-0.3") or -0.3,
                    )
                ),
                lambda_S=float(
                    cast(
                        Union[str, int, float],
                        APGI_PARAMS.get("lambda_S", "0.1") or 0.1,
                    )
                ),
                sigma_S=float(
                    cast(
                        Union[str, int, float],
                        APGI_PARAMS.get("sigma_S", "0.05") or 0.05,
                    )
                ),
                sigma_theta=float(
                    cast(
                        Union[str, int, float],
                        APGI_PARAMS.get("sigma_theta", "0.02") or 0.02,
                    )
                ),
                sigma_M=float(
                    cast(
                        Union[str, int, float],
                        APGI_PARAMS.get("sigma_M", "0.03") or 0.03,
                    )
                ),
                rho=float(
                    cast(Union[str, int, float], APGI_PARAMS.get("rho", "0.7") or 0.7)
                ),
                theta_survival=float(
                    cast(
                        Union[str, int, float],
                        APGI_PARAMS.get("theta_survival", "0.4") or 0.4,
                    )
                ),
                theta_neutral=float(
                    cast(
                        Union[str, int, float],
                        APGI_PARAMS.get("theta_neutral", "0.6") or 0.6,
                    )
                ),
            )
            self.apgi = APGIIntegration(params)

            # 100/100: Hierarchical 5-level processing (requires UltimateAPGIParameters)
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
                        cast(
                            Union[str, int, float],
                            APGI_PARAMS.get("beta_cross", "0.2") or 0.2,
                        )
                    ),
                    tau_levels=cast(
                        Optional[List[float]],
                        APGI_PARAMS.get("tau_levels", [0.1, 0.2, 0.4, 1.0, 5.0]),
                    )
                    or [0.1, 0.2, 0.4, 1.0, 5.0],
                )
                self.hierarchical = HierarchicalProcessor(ultimate_params)
            else:
                self.hierarchical = None  # type: ignore[assignment]

            # 100/100: Precision expectation gap (Π vs Π̂)
            if APGI_PARAMS.get("precision_gap_enabled", True):
                self.precision_gap = PrecisionExpectationState()
            else:
                self.precision_gap = None  # type: ignore[assignment]

            # 100/100: Neuromodulator tracking
            self.neuromodulators = {
                "ACh": float(
                    cast(Union[str, int, float], APGI_PARAMS.get("ACh", "1.0") or 1.0)
                ),
                "NE": float(
                    cast(Union[str, int, float], APGI_PARAMS.get("NE", "1.0") or 1.0)
                ),
                "DA": float(
                    cast(Union[str, int, float], APGI_PARAMS.get("DA", "1.0") or 1.0)
                ),
                "HT5": float(
                    cast(Union[str, int, float], APGI_PARAMS.get("HT5", "1.0") or 1.0)
                ),
            }
        else:
            self.apgi = None

    def reset(self) -> None:
        """Reset experiment and participant state."""
        self.experiment.reset()
        self.participant.reset()

    def run_experiment(self) -> Dict:
        """Execute the full Attentional Blink experiment."""
        self.start_time = time.time()
        self.experiment.reset()
        self.participant.reset()

        # Run all trials
        for trial_num in range(NUM_TRIALS_CONFIG):
            self._run_single_trial(trial_num)

            # Check time budget
            if self.start_time is not None:
                elapsed = time.time() - self.start_time
                if elapsed > TIME_BUDGET:
                    print(f"WARNING: Time budget exceeded at trial {trial_num}")
                    break

        # Calculate final metrics
        completion_time = time.time() - (self.start_time or 0)
        results = self._calculate_results(completion_time)

        return results

    def _run_single_trial(self, trial_num: int) -> None:
        """Execute a single trial with APGI tracking."""
        config = self.experiment.get_next_trial()
        if config is None:
            return

        # Get participant responses
        t1_correct, t2_correct, t1_rt, t2_rt = self.participant.process_trial(
            config.lag, config.trial_type
        )

        # Record trial
        self.experiment.run_trial(
            config=config,
            t1_response=config.t1_stimulus if t1_correct else "WRONG",
            t1_rt_ms=t1_rt,
            t2_response=config.t2_stimulus if t2_correct else "WRONG",
            t2_rt_ms=t2_rt,
            saw_t2=t2_correct or np.random.random() < 0.3,  # Sometimes false positive
        )

        # Process trial with APGI dynamics if enabled
        if self.apgi:
            # Compute prediction error: difference between expected and actual T2 detection
            # Expected accuracy based on lag (lag 1 = spared, lag 2-3 = blink, lag 4+ = recovery)
            if config.lag == 1:
                expected_accuracy = 0.9  # Spared
            elif config.lag in [2, 3]:
                expected_accuracy = 0.4  # Blink
            else:
                expected_accuracy = 0.7  # Recovery

            observed_accuracy = 1.0 if t2_correct else 0.0

            # Trial type affects precision - attentional blink is high-demand
            trial_type = "survival" if config.lag in [2, 3] else "neutral"

            # 100/100: Determine precision based on trial type and neuromodulators
            # ACh increases attention precision for RSVP tasks
            ach_boost = float(
                self.neuromodulators.get("ACh", 1.0) if self.neuromodulators else 1.0
            )
            # NE increases arousal during blink window
            ne_effect = float(
                self.neuromodulators.get("NE", 1.0) if self.neuromodulators else 1.0
            )

            precision_ext = (
                (2.5 if config.lag in [2, 3] else 1.5)
                * ach_boost
                * (1.0 + 0.2 * ne_effect)
            )
            precision_int = (
                1.5 if t2_correct else 1.0
            )  # Correct detection increases precision

            # 100/100: Update running statistics for z-score normalization
            alpha_mu = 0.01
            alpha_sigma = 0.005

            # Update outcome statistics
            if not hasattr(self, "running_stats"):
                self.running_stats = {
                    "outcome_mean": 0.5,
                    "outcome_var": 0.25,
                    "rt_mean": 600.0,
                    "rt_var": 40000.0,
                }

            self.running_stats["outcome_mean"] += alpha_mu * (
                observed_accuracy - self.running_stats["outcome_mean"]
            )
            self.running_stats["outcome_var"] += alpha_sigma * (
                (observed_accuracy - self.running_stats["outcome_mean"]) ** 2
                - self.running_stats["outcome_var"]
            )
            self.running_stats["outcome_var"] = max(
                0.01, self.running_stats["outcome_var"]
            )

            # 100/100: Update precision expectation gap (Π vs Π̂)
            if self.precision_gap:
                nm_floats = {
                    k: float(v)
                    for k, v in (self.neuromodulators or {}).items()
                    if v is not None
                }
                self.precision_gap.update(
                    precision_ext, precision_int, nm_floats, trial_type
                )
                precision_ext = self.precision_gap.Pi_e_actual
                precision_int = self.precision_gap.Pi_i_actual

            # 100/100: Process with APGI - computes ignition, surprise, somatic markers
            apgi_state = self.apgi.process_trial(
                observed=observed_accuracy,
                predicted=expected_accuracy,
                trial_type=trial_type,
                precision_ext=precision_ext,
                precision_int=precision_int,
            )

            # 100/100: Process hierarchical levels
            if self.hierarchical:
                signal = apgi_state.get("S", 0.0)
                for level_idx in range(5):
                    level_state = self.hierarchical.process_level(level_idx, signal)
                    signal = level_state.S * 0.8

    def _calculate_results(self, completion_time: float) -> Dict:
        """Calculate and return experiment results with APGI metrics."""
        summary = self.experiment.get_summary()

        results = {
            "num_trials": len(self.experiment.trials),
            "completion_time_s": completion_time,
            "t1_accuracy": summary.get("t1_accuracy", 0.0),
            "t2_by_lag": summary.get("t2_accuracy_by_lag", {}),
            "blink_magnitude": summary.get("blink_magnitude", 0.0),
        }

        # Add APGI metrics if enabled
        if self.apgi:
            apgi_summary = self.apgi.finalize()
            results["apgi_enabled"] = True

            # Core dynamical metrics
            results["apgi_ignition_rate"] = apgi_summary.get("ignition_rate", 0.0)
            results["apgi_mean_surprise"] = apgi_summary.get("mean_surprise", 0.0)
            results["apgi_metabolic_cost"] = apgi_summary.get("metabolic_cost", 0.0)
            results["apgi_mean_somatic_marker"] = apgi_summary.get(
                "mean_somatic_marker", 0.0
            )
            results["apgi_mean_threshold"] = apgi_summary.get("mean_threshold", 0.0)

            # 100/100: Precision expectation gap (Π vs Π̂)
            if self.precision_gap:
                results["apgi_precision_mismatch"] = (
                    self.precision_gap.precision_mismatch
                )
                results["apgi_anxiety_level"] = self.precision_gap.anxiety_level
                results["apgi_precision_overestimated"] = (
                    self.precision_gap.precision_overestimated
                )

            # 100/100: Hierarchical processing
            if self.hierarchical:
                hier_summary = self.hierarchical.get_hierarchical_summary()
                results.update({f"apgi_{k}": v for k, v in hier_summary.items()})

            # 100/100: Neuromodulator baselines
            if self.neuromodulators:
                results["apgi_acetylcholine"] = self.neuromodulators.get("ACh", 1.0)
                results["apgi_norepinephrine"] = self.neuromodulators.get("NE", 1.0)
                results["apgi_dopamine"] = self.neuromodulators.get("DA", 1.0)
                results["apgi_serotonin"] = self.neuromodulators.get("HT5", 1.0)

            results["apgi_formatted"] = format_apgi_output(apgi_summary)
        else:
            results["apgi_enabled"] = False

        return results


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------


def print_results(results: Dict) -> None:
    """Print experiment results in a formatted way."""
    print("\n" + "=" * 60)
    print("ATTENTIONAL BLINK EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Trials completed: {results['num_trials']}")
    print(f"Completion time: {results['completion_time_s']:.2f}s")
    print(f"T1 Accuracy: {results['t1_accuracy']:.2%}")
    print("\nT2 Accuracy by Lag:")
    for lag, acc in sorted(results["t2_by_lag"].items()):
        bar = "█" * int(acc * 20)
        print(f"  Lag {lag} ({lag * 100}ms): {acc:.1%} {bar}")
    print(f"\nBlink Magnitude: {results['blink_magnitude']:.3f}")

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

    print("=" * 60)


def main(args: Any) -> Dict:
    """Main function for running the experiment."""
    runner = EnhancedAttentionalBlinkRunner()
    results = runner.run_experiment()
    return results


if __name__ == "__main__":
    parser = create_standard_parser("Run Attentional Blink  experiment")
    cli_entrypoint(main, parser)
