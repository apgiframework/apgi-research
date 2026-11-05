"""
Interoceptive Gating Experiment Runner

This is the AGENT-EDITABLE file for the auto-improvement system.
Modify this file to experiment with different task parameters, stimulus intensities,
and analysis approaches to maximize the gating_threshold metric.

Usage:
    uv run run_interoceptive_gating.py

Output:
    Prints summary with gating_threshold (primary metric to maximize)

Modification Guidelines:
    - You CAN modify: task parameters, stimulus intensities, detection thresholds, etc.
    - You CANNOT modify: prepare_interoceptive_gating.py (stimulus sets are fixed)
    - Goal: Maximize gating_threshold (interoceptive signal detection)
    - Time budget: 10 minutes max per run
"""

import time
from typing import Any, Dict, List, Optional, cast

import numpy as np

# Standardized APGI imports
from apgi_cli import cli_entrypoint, create_standard_parser

# APGI Integration
from apgi_integration import APGIIntegration, APGIParameters

from .prepare_interoceptive_gating import (
    APGI_PARAMS,
    TIME_BUDGET,
    Condition,
    InteroGatingExperiment,
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

# Gating parameters
BASE_THRESHOLD = 0.5
THRESHOLD_VARIABILITY = 0.1

# RT parameters (ms)
CUE_RT = 400
RESPONSE_RT = 350
RT_VARIABILITY = 50

# ---------------------------------------------------------------------------
# Simulated Participant
# ---------------------------------------------------------------------------


class SimulatedParticipant:
    def __init__(self, enable_apgi: bool = True):
        self.reset()

    def reset(self) -> None:
        self.threshold = BASE_THRESHOLD + np.random.normal(0, THRESHOLD_VARIABILITY)
        self.cue_rt = CUE_RT + np.random.normal(0, RT_VARIABILITY)
        self.response_rt = RESPONSE_RT + np.random.normal(0, RT_VARIABILITY)

    def process_trial(self, trial_type: Any) -> tuple:
        if trial_type == "cue":
            rt = self.cue_rt + np.random.normal(0, RT_VARIABILITY)
        else:
            rt = self.response_rt + np.random.normal(0, RT_VARIABILITY)

        # Simulate interoceptive gating
        intensity = np.random.uniform(0, 1)
        detected = intensity > self.threshold

        return detected, max(200, rt)


# ---------------------------------------------------------------------------
# Enhanced Runner
# ---------------------------------------------------------------------------


class EnhancedInteroceptiveGatingRunner:
    def __init__(self, enable_apgi: bool = True):
        self.experiment = InteroGatingExperiment(num_trials=NUM_TRIALS_CONFIG)
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

        # Declare APGI component types
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
                "rt_mean": 800.0,
                "rt_var": 40000.0,
            }

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

        detected, rt = self.participant.process_trial(trial.condition)

        self.experiment.run_trial(
            trial=trial,
            detected=detected,
            confidence=5 if detected else 3,  # Confidence rating 1-9
            rt_ms=rt,
        )

        # APGI component: Process with APGI if enabled
        if self.apgi:
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
            if self.running_stats:
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
                    precision_ext, precision_int, self.neuromodulators or {}, trial_type
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
                signal = apgi_state.get("S", 0.0)
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

        # Calculate signal detection metrics
        trials = self.experiment.trials

        # Initialize variables
        hit_rate: float = 0.0
        false_alarm_rate: float = 0.0
        d_prime: float = 0.0
        mean_rt_ms: float = 0.0

        if trials:
            # Hit rate: proportion of interoceptive trials detected
            interoceptive_trials = [
                t for t in trials if t.condition == Condition.INTEROCEPTIVE
            ]
            hit_rate = float(
                np.mean([t.response_detected for t in interoceptive_trials])
                if interoceptive_trials
                else 0.0
            )

            # False alarm rate: proportion of exteroceptive trials detected when they shouldn't be
            exteroceptive_trials = [
                t for t in trials if t.condition == Condition.EXTEROCEPTIVE
            ]
            false_alarm_rate = float(
                np.mean([t.response_detected for t in exteroceptive_trials])
                if exteroceptive_trials
                else 0.0
            )

            # Calculate d' from hit rate and false alarm rate
            # Avoid log(0) by using small epsilon
            epsilon = 1e-10
            hit_rate_safe = max(epsilon, float(hit_rate))
            fa_rate_safe = max(epsilon, float(false_alarm_rate))
            # Use scipy.stats.norm.ppf for inverse normal distribution
            try:
                from scipy.stats import norm

                z_hit = norm.ppf(hit_rate_safe)
                z_fa = norm.ppf(fa_rate_safe)
                d_prime = z_hit - z_fa
            except ImportError:
                # Fallback if scipy not available
                d_prime = 0.0

            # Calculate mean RT
            mean_rt_ms = float(np.mean([t.rt_ms for t in trials]))

        return {
            **{
                "num_trials": len(self.experiment.trials),
                "completion_time_s": completion_time,
                "d_prime": d_prime,
                "gating_effect": summary.get("gating_effect", 0.0),
                "gating_threshold": summary.get("interoceptive_sensitivity", 0.5),
                "hit_rate": hit_rate,
                "false_alarm_rate": false_alarm_rate,
                "mean_rt_ms": mean_rt_ms,
            },
            **apgi_metrics,
        }


def print_results(results: Dict) -> None:
    print("\n" + "=" * 60)
    print("INTEROCEPTIVE GATING EXPERIMENT RESULTS")
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
    print(f"Gating Effect: {results['gating_effect']:.2f}")
    print(f"Hit Rate: {results['hit_rate']:.1%}")
    print(f"False Alarm Rate: {results['false_alarm_rate']:.1%}")
    print(f"Mean RT: {results['mean_rt_ms']:.1f}ms")
    print("=" * 60)


def main(args: Any) -> Dict:
    """Main function for running the experiment."""
    runner = EnhancedInteroceptiveGatingRunner()
    results = runner.run_experiment()
    return results


if __name__ == "__main__":
    parser = create_standard_parser("Run Interoceptive Gating  experiment")
    cli_entrypoint(main, parser)
