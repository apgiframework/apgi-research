"""
Somatic Marker Priming Experiment Runner

This is the AGENT-EDITABLE file for the auto-improvement system.
Modify this file to experiment with different task parameters, marker presentations,
and analysis approaches to maximize the priming_effect_ms metric.

Usage:
    uv run run_somatic_marker_priming.py

Output:
    Prints summary with priming_effect_ms (primary metric to maximize)

Modification Guidelines:
    - You CAN modify: task parameters, marker presentations, timing intervals, etc.
    - You CANNOT modify: prepare_somatic_marker_priming.py (marker sets are fixed)
    - Goal: Maximize priming_effect_ms (somatic influence on perception)
    - Time budget: 10 minutes max per run
"""

import time
from typing import Any, Dict, List, Optional, cast

import numpy as np

# Standardized APGI imports
from apgi_cli import cli_entrypoint, create_standard_parser

# APGI Integration
from apgi_integration import APGIIntegration, APGIParameters

from .prepare_somatic_marker_priming import (
    APGI_PARAMS,
    TIME_BUDGET,
    EmotionType,
    PrimeType,
    SomaticMarkerExperiment,
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

NUM_TRIALS_CONFIG = 100

# Priming parameters
BASE_PRIMING_EFFECT = 20  # ms
PRIMING_VARIABILITY = 10  # ms

# RT parameters (ms)
SAME_MARKER_RT = 400
DIFFERENT_MARKER_RT = 450
RT_VARIABILITY = 60

# ---------------------------------------------------------------------------
# Simulated Participant
# ---------------------------------------------------------------------------


class SimulatedParticipant:
    def __init__(self, enable_apgi: bool = True):
        self.reset()

    def reset(self) -> None:
        pass

    def process_trial(self, marker_type: PrimeType) -> tuple:
        # Priming effect: same marker types are processed faster
        if marker_type == PrimeType.POSITIVE:
            # Simulated bias towards positive outcomes in POSITIVE prime
            rt = SAME_MARKER_RT + np.random.normal(0, RT_VARIABILITY)
            correct = np.random.random() < 0.98  # Higher accuracy
        elif marker_type == PrimeType.NEGATIVE:
            rt = DIFFERENT_MARKER_RT + np.random.normal(0, RT_VARIABILITY)
            correct = np.random.random() < 0.92
        else:
            rt = (SAME_MARKER_RT + DIFFERENT_MARKER_RT) / 2 + np.random.normal(
                0, RT_VARIABILITY
            )
            correct = np.random.random() < 0.95

        return correct, max(200, rt)


# ---------------------------------------------------------------------------
# Enhanced Runner
# ---------------------------------------------------------------------------


class EnhancedSomaticMarkerRunner:
    def __init__(self, enable_apgi: bool = True):
        self.experiment = SomaticMarkerExperiment(num_trials=NUM_TRIALS_CONFIG)
        self.participant = SimulatedParticipant()
        self.start_time: Optional[float] = None

        # APGI component placeholders
        self.hierarchical: Optional[HierarchicalProcessor] = None
        self.precision_gap: Optional[PrecisionExpectationState] = None

        # Initialize APGI components
        self.enable_apgi = enable_apgi and APGI_PARAMS.get("enabled", True)
        if self.enable_apgi:
            params = APGIParameters(
                tau_S=float(str(APGI_PARAMS.get("tau_s", 0.35) or 0.35)),
                beta=float(str(APGI_PARAMS.get("beta", 1.5) or 1.5)),
                theta_0=float(str(APGI_PARAMS.get("theta_0", 0.5) or 0.5)),
                alpha=float(str(APGI_PARAMS.get("alpha", 5.5) or 5.5)),
                gamma_M=float(str(APGI_PARAMS.get("gamma_M", -0.3) or -0.3)),
                lambda_S=float(str(APGI_PARAMS.get("lambda_S", 0.1) or 0.1)),
                sigma_S=float(str(APGI_PARAMS.get("sigma_S", 0.05) or 0.05)),
                sigma_theta=float(str(APGI_PARAMS.get("sigma_theta", 0.02) or 0.02)),
                sigma_M=float(str(APGI_PARAMS.get("sigma_M", 0.03) or 0.03)),
                rho=float(str(APGI_PARAMS.get("rho", 0.7) or 0.7)),
                theta_survival=float(
                    str(APGI_PARAMS.get("theta_survival", 0.3) or 0.3)
                ),
                theta_neutral=float(
                    str(APGI_PARAMS.get("theta_neutral", "0.7") or 0.7)
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
                    theta_neutral=float(params.theta_neutral),
                    beta_cross=float(str(APGI_PARAMS.get("beta_cross", 0.2) or 0.2)),
                    tau_levels=cast(
                        List[float],
                        APGI_PARAMS.get("tau_levels", [0.1, 0.2, 0.4, 1.0, 5.0])
                        or [0.1, 0.2, 0.4, 1.0, 5.0],
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
                "ACh": float(str(APGI_PARAMS.get("ACh", "1.0") or 1.0)),
                "NE": float(str(APGI_PARAMS.get("NE", "1.0") or 1.0)),
                "DA": float(str(APGI_PARAMS.get("DA", "1.0") or 1.0)),
                "HT5": float(str(APGI_PARAMS.get("HT5", "1.0") or 1.0)),
            }

            # APGI component: Running statistics for z-score normalization
            self.running_stats = {
                "outcome_mean": 0.5,
                "outcome_var": 0.25,
                "rt_mean": 800.0,
                "rt_var": 40000.0,
            }
        # APGI tracking across trials
        self.apgi_states: List[Dict] = []
        self.ignition_history: List[float] = []
        self.surprise_history: List[float] = []
        self.threshold_history: List[float] = []
        self.somatic_marker_history: List[float] = []
        self.metabolic_cost_history: List[float] = []

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

        # Convert EmotionType to PrimeType for compatibility
        emotion_str = (
            trial.emotion_type.value
            if hasattr(trial.emotion_type, "value")
            else str(trial.emotion_type)
        )
        prime_type = (
            PrimeType.POSITIVE
            if emotion_str == "positive"
            else PrimeType.NEGATIVE if emotion_str == "negative" else PrimeType.NEUTRAL
        )
        correct, rt = self.participant.process_trial(prime_type)

        self.experiment.run_trial(
            trial=trial,
            selected="A" if correct else "B",
            outcome=100.0 if correct else 50.0,
            somatic=0.5 if correct else 0.2,
            rt_ms=rt,
        )

        # APGI component: Process with APGI if enabled
        if self.apgi:
            # Compute prediction error from trial outcome
            observed_accuracy = 1.0 if correct else 0.0
            expected_accuracy = 0.5  # Baseline

            # Determine trial type
            trial_type = "neutral"

            # APGI component: Determine precision based on neuromodulators
            ach_boost = self.neuromodulators.get("ACh", 1.0)
            ne_effect = self.neuromodulators.get("NE", 1.0)
            da_effect = self.neuromodulators.get("DA", 1.0)

            precision_ext = 1.5 * ach_boost * (1.0 + 0.2 * da_effect)
            precision_int = 1.5 * (1.0 + 0.2 * ne_effect)

            # APGI component: Update running statistics
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

            # APGI component: Track APGI state
            self.apgi_states.append(apgi_state)
            self.ignition_history.append(
                float(apgi_state.get("ignition_prob", 0.0) or 0.0)
            )
            self.surprise_history.append(float(apgi_state.get("surprise", 0.0) or 0.0))
            self.threshold_history.append(float(apgi_state.get("theta", 0.5) or 0.5))
            self.somatic_marker_history.append(float(apgi_state.get("M", 0.0) or 0.0))
            self.metabolic_cost_history.append(
                float(apgi_state.get("metabolic_cost", 0.0) or 0.0)
            )

    def _calculate_results(self) -> Dict:
        summary = self.experiment.get_summary()
        if self.start_time is not None:
            completion_time = time.time() - self.start_time
        else:
            completion_time = 0.0

        # Calculate priming effect from RT data
        positive_trials = [
            t for t in self.experiment.trials if t.emotion_type == EmotionType.POSITIVE
        ]
        negative_trials = [
            t for t in self.experiment.trials if t.emotion_type == EmotionType.NEGATIVE
        ]

        same_marker_rt = (
            np.mean([t.rt_ms for t in positive_trials]) if positive_trials else 0.0
        )
        different_marker_rt = (
            np.mean([t.rt_ms for t in negative_trials]) if negative_trials else 0.0
        )
        priming_effect = different_marker_rt - same_marker_rt

        # Calculate accuracy as proportion of advantageous choices
        accuracy = summary.get("overall_advantageous_rate", 0.0)

        results = {
            "num_trials": len(self.experiment.trials),
            "completion_time_s": completion_time,
            "d_prime": 0.0,  # Not calculated in this experiment
            "accuracy": accuracy,
            "priming_effect_ms": priming_effect,
            "same_marker_rt_ms": same_marker_rt,
            "different_marker_rt_ms": different_marker_rt,
        }

        # Calculate APGI metrics from history
        if self.apgi_states:
            apgi_metrics = {
                "apgi_enabled": True,
                "apgi_ignition_rate": min(
                    1.0, max(0.0, np.mean(self.ignition_history))
                ),
                "apgi_mean_surprise": np.mean(self.surprise_history),
                "apgi_metabolic_cost": np.mean(self.metabolic_cost_history),
                "apgi_mean_somatic_marker": np.mean(self.somatic_marker_history),
                "apgi_mean_threshold": (
                    np.mean(self.threshold_history) if self.threshold_history else 0.5
                ),
            }

            if self.precision_gap:
                apgi_metrics["apgi_precision_mismatch"] = (
                    self.precision_gap.precision_mismatch
                )
                apgi_metrics["apgi_anxiety_level"] = self.precision_gap.anxiety_level

            if self.neuromodulators:
                apgi_metrics["apgi_acetylcholine"] = float(
                    self.neuromodulators.get("ACh", 1.0) or 1.0
                )
                apgi_metrics["apgi_norepinephrine"] = float(
                    self.neuromodulators.get("NE", 1.0) or 1.0
                )
                apgi_metrics["apgi_dopamine"] = float(
                    self.neuromodulators.get("DA", 1.0) or 1.0
                )
                apgi_metrics["apgi_serotonin"] = float(
                    self.neuromodulators.get("HT5", 1.0) or 1.0
                )
        else:
            apgi_metrics = {
                "apgi_enabled": float(bool(self.enable_apgi)),
                "apgi_ignition_rate": 0.0,
                "apgi_mean_surprise": 0.0,
                "apgi_metabolic_cost": 0.0,
                "apgi_mean_somatic_marker": 0.0,
                "apgi_mean_threshold": 0.5,
            }

        results.update(apgi_metrics)
        return results


def print_results(results: Dict) -> None:
    print("\n" + "=" * 60)
    print("SOMATIC MARKER PRIMING EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Trials: {results['num_trials']}")

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

    print(f"Time: {results['completion_time_s']:.2f}s")
    print(f"Priming Effect: {results['priming_effect_ms']:.1f}ms")
    print(f"Same Marker RT: {results['same_marker_rt_ms']:.1f}ms")
    print(f"Different Marker RT: {results['different_marker_rt_ms']:.1f}ms")
    print(f"Accuracy: {results['accuracy']:.1%}")
    print("=" * 60)


def main(args: Any) -> Dict:
    """Main function for running the experiment."""
    runner = EnhancedSomaticMarkerRunner()
    results = runner.run_experiment()
    return results


if __name__ == "__main__":
    parser = create_standard_parser("Run Somatic Marker Priming  experiment")
    cli_entrypoint(main, parser)
