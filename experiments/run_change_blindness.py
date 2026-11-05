"""
Change Blindness Experiment Runner

This is the AGENT-EDITABLE file for the auto-improvement system.
Modify this file to experiment with different task parameters, mask durations,
and analysis approaches to maximize the detection_rate metric.

Usage:
    uv run run_change_blindness.py

Output:
    Prints summary with detection_rate (primary metric to maximize)

Modification Guidelines:
    - You CAN modify: task parameters, mask timing, stimulus presentation, etc.
    - You CANNOT modify: prepare_change_blindness.py (stimulus sets are fixed)
    - Goal: Maximize detection_rate (ability to detect changes)
    - Time budget: 10 minutes max per run
"""

import time
from typing import Any, Dict, List, Optional, cast

import numpy as np

# Standardized APGI imports
from apgi_cli import cli_entrypoint, create_standard_parser

# APGI Integration
from apgi_integration import APGIIntegration, APGIParameters

from .prepare_change_blindness import (
    APGI_PARAMS,
    TIME_BUDGET,
    ChangeBlindnessExperiment,
)
from .ultimate_apgi_template import (
    HierarchicalProcessor,
    PrecisionExpectationState,
    UltimateAPGIParameters,
)

# ---------------------------------------------------------------------------
# MODIFIABLE PARAMETERS
# ---------------------------------------------------------------------------

TIME_BUDGET = 600  # noqa: F811

NUM_TRIALS_CONFIG = 60
CHANGE_PROBABILITY = 0.50

# Detection parameters
BASE_DETECTION_RATE = 0.40
MASK_DURATION_EFFECT = -0.002  # Per ms of mask

# ---------------------------------------------------------------------------
# Simulated Participant
# ---------------------------------------------------------------------------


class SimulatedParticipant:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        pass

    def process_trial(self, trial_type: Any, mask_duration_ms: int) -> tuple:
        if trial_type.value == "change":
            prob = BASE_DETECTION_RATE + MASK_DURATION_EFFECT * mask_duration_ms
            detected = np.random.random() < max(0.1, prob)
        else:
            # No change trial: false alarm rate
            detected = np.random.random() < 0.15

        rt = 800 + np.random.normal(0, 200) if detected else 0
        return detected, max(400, rt)


# ---------------------------------------------------------------------------
# Enhanced Runner
# ---------------------------------------------------------------------------


class EnhancedChangeBlindnessRunner:
    def __init__(self, enable_apgi: bool = True):
        self.experiment = ChangeBlindnessExperiment(num_trials=NUM_TRIALS_CONFIG)
        self.participant = SimulatedParticipant()
        self.start_time: Optional[float] = None

        # Initialize APGI components
        self.enable_apgi = enable_apgi and APGI_PARAMS.get("enabled", True)

        # Helper function to safely get float values from APGI_PARAMS
        def safe_float(key: str, default: float) -> float:
            value = APGI_PARAMS.get(key, default)
            if isinstance(value, (int, float)):
                return float(value)
            elif value is not None:
                return float(str(value))
            else:
                return default

        self.apgi: Optional[APGIIntegration] = None
        self.hierarchical: Optional[HierarchicalProcessor] = None
        self.precision_gap: Optional[PrecisionExpectationState] = None
        self.neuromodulators: Optional[Dict[str, float]] = None
        self.running_stats: Optional[Dict[str, float]] = None
        if self.enable_apgi:
            params = APGIParameters(
                tau_S=safe_float("tau_s", 0.35),
                beta=safe_float("beta", 1.5),
                theta_0=safe_float("theta_0", 0.5),
                alpha=safe_float("alpha", 5.5),
                gamma_M=safe_float("gamma_M", -0.3),
                lambda_S=safe_float("lambda_S", 0.1),
                sigma_S=safe_float("sigma_S", 0.05),
                sigma_theta=safe_float("sigma_theta", 0.02),
                sigma_M=safe_float("sigma_M", 0.03),
                rho=safe_float("rho", 0.7),
                theta_survival=safe_float("theta_survival", 0.3),
                theta_neutral=safe_float("theta_neutral", 0.7),
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
                    beta_cross=safe_float("beta_cross", 0.2),
                    tau_levels=cast(
                        List[float],
                        APGI_PARAMS.get("tau_levels") or [0.1, 0.2, 0.4, 1.0, 5.0],
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
                "ACh": safe_float("ACh", 1.0),
                "NE": safe_float("NE", 1.0),
                "DA": safe_float("DA", 1.0),
                "HT5": safe_float("HT5", 1.0),
            }

            # APGI component: Running statistics for z-score normalization
            self.running_stats = {
                "outcome_mean": 0.5,
                "outcome_var": 0.25,
                "rt_mean": 500.0,
                "rt_var": 25000.0,
            }
        else:
            self.apgi = None
            self.hierarchical = None
            self.precision_gap = None
            self.neuromodulators = None
            self.running_stats = None

    def run_experiment(self) -> Dict:
        self.start_time = time.time()
        self.experiment.reset()
        self.participant.reset()

        for trial_num in range(NUM_TRIALS_CONFIG):
            self._run_single_trial(trial_num)

            if time.time() - self.start_time > TIME_BUDGET:
                break

        return self._calculate_results()

    def _run_single_trial(self, trial_num: int) -> None:
        trial = self.experiment.get_next_trial()
        if trial is None:
            return

        detected, rt = self.participant.process_trial(
            trial.trial_type, trial.display_duration_ms
        )

        self.experiment.run_trial(
            trial=trial,
            change_detected=detected,
            rt_ms=rt,
        )

        # APGI component: Process with APGI if enabled
        if (
            self.apgi
            and self.neuromodulators is not None
            and self.running_stats is not None
        ):
            # Compute prediction error from trial outcome
            observed_accuracy = 1.0 if detected else 0.0
            expected_accuracy = 0.5  # Baseline

            # Determine trial type
            trial_type = "neutral"

            # APGI component: Determine precision based on neuromodulators
            ach_boost = (
                self.neuromodulators.get("ACh", 1.0) if self.neuromodulators else 1.0
            )
            ne_effect = (
                self.neuromodulators.get("NE", 1.0) if self.neuromodulators else 1.0
            )
            da_effect = (
                self.neuromodulators.get("DA", 1.0) if self.neuromodulators else 1.0
            )

            precision_ext = 1.5 * ach_boost * (1.0 + 0.2 * da_effect)
            precision_int = 1.5 * (1.0 + 0.2 * ne_effect)

            # APGI component: Update running statistics
            if self.running_stats is not None:
                alpha_mu = 0.01
                alpha_sigma = 0.005
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

            # APGI component: Update precision expectation gap (Π vs Π̂)
            if self.precision_gap:
                self.precision_gap.update(
                    precision_ext, precision_int, self.neuromodulators, trial_type
                )
                precision_ext = self.precision_gap.Pi_e_actual
                precision_int = self.precision_gap.Pi_i_actual

            # APGI component: Process with APGI
            apgi_state = self.apgi.process_trial(
                observed=observed_accuracy,
                predicted=expected_accuracy,
                trial_type=trial_type,
                precision_ext=precision_ext,
                precision_int=precision_int,
            )

            # APGI component: Process hierarchical levels
            if self.hierarchical:
                signal = apgi_state.get("S", 0.0) or 0.0
                for level_idx in range(5):
                    level_state = self.hierarchical.process_level(level_idx, signal)
                    signal = level_state.S * 0.8

    def _calculate_results(self) -> Dict:
        summary = self.experiment.get_summary()
        start_time = self.start_time if self.start_time is not None else time.time()
        completion_time = time.time() - start_time

        apgi_metrics = {}
        if self.apgi and hasattr(self.apgi, "finalize"):
            apgi_metrics = self.apgi.finalize()

        return {
            **{
                "num_trials": len(self.experiment.trials),
                "completion_time_s": completion_time,
                "d_prime": summary.get("d_prime", 0.0),
                "detection_rate": summary.get("detection_rate", 0.5),
                "correct_rejection_rate": summary.get("correct_rejection_rate", 0.5),
                "threshold_ms": summary.get("threshold_ms", 500.0),
                "mean_rt_ms": summary.get("mean_rt_ms", 600.0),
            },
            **apgi_metrics,
        }


def print_results(results: Dict) -> None:
    print("\n" + "=" * 60)
    print("CHANGE BLINDNESS EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Trials: {results['num_trials']}")
    print(f"Time: {results['completion_time_s']:.2f}s")
    print(f"Detection Rate: {results['detection_rate']:.1%}")
    print(f"Correct Rejection Rate: {results['correct_rejection_rate']:.1%}")
    print(f"Threshold: {results['threshold_ms']:.0f}ms")
    print(f"Mean RT: {results['mean_rt_ms']:.1f}ms")
    print("=" * 60)


def main(args: Any) -> Dict:
    """Main function for running the experiment."""
    runner = EnhancedChangeBlindnessRunner()
    results = runner.run_experiment()
    return results


if __name__ == "__main__":
    parser = create_standard_parser("Run Change Blindness  experiment")
    cli_entrypoint(main, parser)
