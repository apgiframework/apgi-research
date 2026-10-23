"""
Visual Search Experiment Runner

This is the AGENT-EDITABLE file for the auto-improvement system.
Modify this file to experiment with different task parameters, display sizes,
and analysis approaches to maximize the conjunction_present_slope metric.

Usage:
    uv run run_visual_search.py

Output:
    Prints summary with conjunction_present_slope (primary metric to maximize)

Modification Guidelines:
    - You CAN modify: task parameters, display sizes, stimulus timing, etc.
    - You CANNOT modify: prepare_visual_search.py (stimulus sets are fixed)
    - Goal: Maximize conjunction_present_slope (ms/item search efficiency)
    - Time budget: 10 minutes max per run
"""

import time
from typing import Any, Dict, List, Optional, cast

import numpy as np

# Standardized APGI imports
from apgi_cli import cli_entrypoint, create_standard_parser

# APGI Integration - imports the dynamical system for tracking ignition, surprise, somatic markers
from apgi_integration import APGIIntegration, APGIParameters, format_apgi_output

from .prepare_visual_search import (
    APGI_PARAMS,  # APGI parameters from prepare file
    TIME_BUDGET,
    SearchType,
    VSExperiment,
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

NUM_TRIALS_CONFIG = 80
DISPLAY_SIZES_CONFIG = [8, 16, 24, 32]
TARGET_PRESENT_PROB = 0.5

# Search types to test
TEST_FEATURE_SEARCH = True
TEST_CONJUNCTION_SEARCH = True

# RT parameters (ms)
FEATURE_SEARCH_INTERCEPT = 250
CONJUNCTION_SEARCH_INTERCEPT = 300
FEATURE_SLOPE = 5  # ms/item
CONJUNCTION_SLOPE = 25  # ms/item

# ---------------------------------------------------------------------------
# Simulated Participant
# ---------------------------------------------------------------------------


class SimulatedParticipant:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.feature_intercept = FEATURE_SEARCH_INTERCEPT
        self.feature_slope = FEATURE_SLOPE
        self.conjunction_intercept = CONJUNCTION_SEARCH_INTERCEPT
        self.conjunction_slope = CONJUNCTION_SLOPE

    def process_trial(
        self, search_type: SearchType, display_size: int, target_present: bool
    ) -> tuple:
        if search_type == SearchType.FEATURE:
            base_rt = float(self.feature_intercept + self.feature_slope * display_size)
        else:
            base_rt = float(
                self.conjunction_intercept + self.conjunction_slope * display_size
            )

        if not target_present:
            base_rt *= 1.2  # Absent trials ~20% slower

        rt = base_rt + np.random.normal(0, 50)
        correct = np.random.random() < 0.95

        return correct, max(200, rt)


# ---------------------------------------------------------------------------
# Enhanced Runner
# ---------------------------------------------------------------------------


class EnhancedVisualSearchRunner:
    """Visual Search Runner with APGI dynamics integration."""

    def __init__(self, enable_apgi: bool = True):
        self.experiment = VSExperiment(num_trials=NUM_TRIALS_CONFIG)
        self.participant = SimulatedParticipant()
        self.start_time: Optional[float] = None
        self.apgi: Optional[APGIIntegration] = None

        # Initialize APGI integration if enabled
        self.enable_apgi = enable_apgi and APGI_PARAMS.get("enabled", True)
        if self.enable_apgi:
            params = APGIParameters(
                tau_S=float(cast(float, APGI_PARAMS.get("tau_s", 0.35))),
                beta=float(cast(float, APGI_PARAMS.get("beta", 1.3))),
                theta_0=float(cast(float, APGI_PARAMS.get("theta_0", 0.5))),
                alpha=float(cast(float, APGI_PARAMS.get("alpha", 5.0))),
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
                    beta_cross=float(cast(float, APGI_PARAMS.get("beta_cross", 0.2))),
                    tau_levels=cast(
                        Optional[List[float]],
                        APGI_PARAMS.get("tau_levels", [0.1, 0.2, 0.4, 1.0, 5.0]),
                    ),
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
            self.neuromodulators: Dict[str, float] = {
                "ACh": float(cast(float, APGI_PARAMS.get("ACh", 1.0))),
                "NE": float(cast(float, APGI_PARAMS.get("NE", 1.0))),
                "DA": float(cast(float, APGI_PARAMS.get("DA", 1.0))),
                "HT5": float(cast(float, APGI_PARAMS.get("HT5", 1.0))),
            }
        else:
            self.apgi = None

    def run_experiment(self) -> Dict:
        self.start_time = time.time()
        self.experiment.reset()
        self.participant.reset()

        for trial_num in range(NUM_TRIALS_CONFIG):
            self._run_single_trial(trial_num)

            if (
                self.start_time is not None
                and time.time() - self.start_time > TIME_BUDGET
            ):
                break

        return self._calculate_results()

    def _run_single_trial(self, trial_num: int) -> None:
        trial = self.experiment.get_next_trial()
        if trial is None:
            return

        correct, rt = self.participant.process_trial(
            trial.search_type, trial.display_size, trial.target_present
        )

        response = (
            "present"
            if (trial.target_present and correct)
            or (not trial.target_present and not correct)
            else "absent"
        )

        self.experiment.run_trial(trial=trial, response=response, rt_ms=rt)

        # Process trial with APGI dynamics if enabled
        if self.apgi:
            # Prediction error based on reaction time deviation from expected
            if trial.search_type == SearchType.FEATURE:
                expected_rt = (
                    FEATURE_SEARCH_INTERCEPT + FEATURE_SLOPE * trial.display_size
                )
            else:
                expected_rt = (
                    CONJUNCTION_SEARCH_INTERCEPT
                    + CONJUNCTION_SLOPE * trial.display_size
                )

            observed_rt = rt / 1000.0  # Convert to seconds
            expected_rt_sec = expected_rt / 1000.0

            # Trial type: conjunction search is more demanding (higher precision)
            trial_type = (
                "survival" if trial.search_type == SearchType.CONJUNCTION else "neutral"
            )

            # 100/100: Determine precision based on neuromodulators
            ach_boost = float(self.neuromodulators.get("ACh", 1.0))
            ne_effect = float(self.neuromodulators.get("NE", 1.0))
            da_effect = float(self.neuromodulators.get("DA", 1.0))

            precision_ext = 1.5 * ach_boost * (1.0 + 0.2 * da_effect)
            precision_int = 1.5 * (1.0 + 0.2 * ne_effect)

            # 100/100: Update precision expectation gap (Π vs Π̂)
            if self.precision_gap:
                nm_floats = {
                    k: float(v) for k, v in (self.neuromodulators or {}).items()
                }
                self.precision_gap.update(
                    precision_ext, precision_int, nm_floats, trial_type
                )
                precision_ext = self.precision_gap.Pi_e_actual
                precision_int = self.precision_gap.Pi_i_actual

            # Process with APGI - computes ignition probability, surprise, somatic markers
            apgi_state = self.apgi.process_trial(
                observed=observed_rt,
                predicted=expected_rt_sec,
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
            # Original APGI call preserved for compatibility
            _ = self.apgi.process_trial(
                observed=observed_rt,
                predicted=expected_rt_sec,
                trial_type=trial_type,
                precision_ext=(
                    2.0 if trial.search_type == SearchType.CONJUNCTION else 1.0
                ),
                precision_int=(
                    1.5 if correct else 1.0
                ),  # Correct detection increases precision
            )

    def _calculate_results(self) -> Dict:
        summary = self.experiment.get_summary()
        completion_time = time.time() - (self.start_time or 0)

        results = {
            "num_trials": len(self.experiment.trials),
            "completion_time_s": completion_time,
            "feature_present_slope": summary.get("feature_present_slope", 0.0),
            "conjunction_present_slope": summary.get("conjunction_present_slope", 0.0),
            "slope_ratio": summary.get("conjunction_present_slope", 0.0)
            / max(summary.get("feature_present_slope", 1.0), 1.0),
            "overall_accuracy": summary.get("overall_accuracy", 0.0),
        }

        # Add APGI metrics if enabled
        if self.apgi:
            apgi_summary = self.apgi.finalize()
            results["apgi_enabled"] = True
            results["apgi_ignition_rate"] = apgi_summary.get("ignition_rate", 0.0)
            results["apgi_mean_surprise"] = apgi_summary.get("mean_surprise", 0.0)
            results["apgi_metabolic_cost"] = apgi_summary.get("metabolic_cost", 0.0)
            results["apgi_mean_somatic_marker"] = apgi_summary.get(
                "mean_somatic_marker", 0.0
            )
            results["apgi_mean_threshold"] = apgi_summary.get("mean_threshold", 0.0)
            results["apgi_formatted"] = format_apgi_output(apgi_summary)
        else:
            results["apgi_enabled"] = False

        return results


def print_results(results: Dict) -> None:
    print("\n" + "=" * 60)
    print("VISUAL SEARCH EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Trials: {results['num_trials']}")
    print(f"Time: {results['completion_time_s']:.2f}s")
    print(f"Feature Search Slope: {results['feature_present_slope']:.2f} ms/item")
    print(
        f"Conjunction Search Slope: {results['conjunction_present_slope']:.2f} ms/item"
    )
    print(f"Slope Ratio (C/F): {results['slope_ratio']:.2f}x")
    print(f"Accuracy: {results['overall_accuracy']:.1%}")

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
    runner = EnhancedVisualSearchRunner()
    results = runner.run_experiment()
    return results


if __name__ == "__main__":
    parser = create_standard_parser("Run Visual Search  experiment")
    cli_entrypoint(main, parser)
