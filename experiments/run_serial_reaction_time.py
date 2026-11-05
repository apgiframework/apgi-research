"""
Serial Reaction Time Experiment Runner

This is the AGENT-EDITABLE file for the auto-improvement system.
Modify this file to experiment with different task parameters, sequence patterns,
and analysis approaches to maximize the learning_effect_ms metric.

Usage:
    uv run run_serial_reaction_time.py

Output:
    Prints summary with learning_effect_ms (primary metric to maximize)

Modification Guidelines:
    - You CAN modify: task parameters, sequence patterns, learning rates, etc.
    - You CANNOT modify: prepare_serial_reaction_time.py (stimulus sets are fixed)
    - Goal: Maximize learning_effect_ms (procedural learning improvement)
    - Time budget: 10 minutes max per run
"""

import time
from typing import Any, Dict, List, cast

import numpy as np

# Standardized APGI imports
from apgi_cli import cli_entrypoint, create_standard_parser

# APGI Integration
from apgi_integration import APGIIntegration, APGIParameters

from .prepare_serial_reaction_time import APGI_PARAMS, TIME_BUDGET, SRTExperiment
from .ultimate_apgi_template import (
    HierarchicalProcessor,
    PrecisionExpectationState,
    UltimateAPGIParameters,
)

# ---------------------------------------------------------------------------
# MODIFIABLE PARAMETERS
# ---------------------------------------------------------------------------

TIME_BUDGET = 600  # noqa: F811

NUM_TRIALS_CONFIG = 120
SEQUENCE_LENGTH_CONFIG = 10
RANDOM_PROB_CONFIG = 0.20

# RT parameters (ms)
SEQUENTIAL_RT = 450
RANDOM_RT = 520
RT_VARIABILITY = 70

# ---------------------------------------------------------------------------
# Simulated Participant
# ---------------------------------------------------------------------------


class SimulatedParticipant:
    def __init__(self, enable_apgi: bool = True):
        self.reset()

    def reset(self) -> None:
        self.sequential_rt = SEQUENTIAL_RT
        self.random_rt = RANDOM_RT

    def process_trial(self, is_sequential: bool) -> tuple:
        if is_sequential:
            rt = self.sequential_rt + np.random.normal(0, RT_VARIABILITY)
        else:
            rt = self.random_rt + np.random.normal(0, RT_VARIABILITY)

        correct = np.random.random() < 0.95
        return correct, max(250, rt)


# ---------------------------------------------------------------------------
# Enhanced Runner
# ---------------------------------------------------------------------------


class EnhancedSRTRunner:
    def __init__(self, enable_apgi: bool = True):
        self.experiment = SRTExperiment(num_trials=NUM_TRIALS_CONFIG)
        self.participant = SimulatedParticipant()
        self.start_time: float = 0.0

        # Initialize APGI components
        self.enable_apgi = enable_apgi and APGI_PARAMS.get("enabled", True)
        if self.enable_apgi:
            params = APGIParameters(
                tau_S=float(APGI_PARAMS.get("tau_s", 0.35) or 0.35),  # type: ignore[arg-type]
                beta=float(APGI_PARAMS.get("beta", 1.5) or 1.5),  # type: ignore[arg-type]
                theta_0=float(APGI_PARAMS.get("theta_0", 0.5) or 0.5),  # type: ignore[arg-type]
                alpha=float(APGI_PARAMS.get("alpha", 5.5) or 5.5),  # type: ignore[arg-type]
                gamma_M=float(APGI_PARAMS.get("gamma_M", -0.3) or -0.3),  # type: ignore[arg-type]
                lambda_S=float(APGI_PARAMS.get("lambda_S", 0.1) or 0.1),  # type: ignore[arg-type]
                sigma_S=float(APGI_PARAMS.get("sigma_S", 0.05) or 0.05),  # type: ignore[arg-type]
                sigma_theta=float(APGI_PARAMS.get("sigma_theta", 0.02) or 0.02),  # type: ignore[arg-type]
                sigma_M=float(APGI_PARAMS.get("sigma_M", 0.03) or 0.03),  # type: ignore[arg-type]
                rho=float(APGI_PARAMS.get("rho", 0.7) or 0.7),  # type: ignore[arg-type]
                theta_survival=float(APGI_PARAMS.get("theta_survival", 0.3) or 0.3),  # type: ignore[arg-type]
                theta_neutral=float(APGI_PARAMS.get("theta_neutral", 0.7) or 0.7),  # type: ignore[arg-type]
            )
            self.apgi = APGIIntegration(params)

            # APGI component: Hierarchical 5-level processing (requires UltimateAPGIParameters)
            self.hierarchical: Any = None
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
                    beta_cross=float(APGI_PARAMS.get("beta_cross", 0.2) or 0.2),  # type: ignore[arg-type]
                    tau_levels=list(  # type: ignore[call-overload]
                        APGI_PARAMS.get("tau_levels", [0.1, 0.2, 0.4, 1.0, 5.0])
                        or [0.1, 0.2, 0.4, 1.0, 5.0]
                    ),
                )
                self.hierarchical = HierarchicalProcessor(ultimate_params)

            # APGI component: Precision expectation gap (Π vs Π̂)
            self.precision_gap: Any = None
            if APGI_PARAMS.get("precision_gap_enabled", True):
                self.precision_gap = PrecisionExpectationState()

            # APGI component: Neuromodulator tracking
            self.neuromodulators: Dict[str, float] = {
                "ACh": float(APGI_PARAMS.get("ACh", 1.0) or 1.0),  # type: ignore[arg-type]
                "NE": float(APGI_PARAMS.get("NE", 1.0) or 1.0),  # type: ignore[arg-type]
                "DA": float(APGI_PARAMS.get("DA", 1.0) or 1.0),  # type: ignore[arg-type]
                "HT5": float(APGI_PARAMS.get("HT5", 1.0) or 1.0),  # type: ignore[arg-type]
            }

            self.running_stats = {
                "outcome_mean": 0.5,
                "outcome_var": 0.25,
                "rt_mean": 800.0,
                "rt_var": 40000.0,
            }
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

        correct, rt = self.participant.process_trial(trial.is_sequential)

        self.experiment.run_trial(
            trial=trial,
            response=str(trial.stimulus_position) if correct else "wrong",
            rt_ms=rt,
        )

        if self.apgi:
            observed_accuracy = 1.0 if correct else 0.0
            expected_accuracy = 0.5  # Baseline

            trial_type = "neutral"

            ach_boost = self.neuromodulators.get("ACh", 1.0)
            ne_effect = self.neuromodulators.get("NE", 1.0)
            da_effect = self.neuromodulators.get("DA", 1.0)

            precision_ext = 1.5 * ach_boost * (1.0 + 0.2 * da_effect)
            precision_int = 1.5 * (1.0 + 0.2 * ne_effect)

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

            if self.precision_gap:
                self.precision_gap.update(
                    precision_ext, precision_int, self.neuromodulators, trial_type
                )
                precision_ext = self.precision_gap.Pi_e_actual
                precision_int = self.precision_gap.Pi_i_actual

            apgi_state = self.apgi.process_trial(
                observed=observed_accuracy,
                predicted=expected_accuracy,
                trial_type=trial_type,
                precision_ext=precision_ext,
                precision_int=precision_int,
            )

            self.apgi_states.append(apgi_state)
            self.ignition_history.append(apgi_state.get("ignition_prob", 0.0))
            self.surprise_history.append(apgi_state.get("surprise", 0.0))
            self.threshold_history.append(apgi_state.get("theta", 0.5))
            self.somatic_marker_history.append(apgi_state.get("M", 0.0))
            self.metabolic_cost_history.append(apgi_state.get("metabolic_cost", 0.0))

    def _calculate_results(self) -> Dict:
        summary = self.experiment.get_summary()
        completion_time = time.time() - self.start_time
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
                apgi_metrics["apgi_precision_overestimated"] = (
                    self.precision_gap.precision_overestimated
                )

            if self.neuromodulators:
                apgi_metrics["apgi_dopamine"] = self.neuromodulators.get("DA", 1.0)
                apgi_metrics["apgi_serotonin"] = self.neuromodulators.get("HT5", 1.0)
                apgi_metrics["apgi_acetylcholine"] = self.neuromodulators.get(
                    "ACh", 1.0
                )
                apgi_metrics["apgi_norepinephrine"] = self.neuromodulators.get(
                    "NE", 1.0
                )
        else:
            apgi_metrics = {
                "apgi_enabled": float(cast(Any, self.enable_apgi)),
                "apgi_ignition_rate": 0.0,
                "apgi_mean_surprise": 0.0,
                "apgi_metabolic_cost": 0.0,
                "apgi_mean_somatic_marker": 0.0,
                "apgi_mean_threshold": 0.5,
                "apgi_precision_mismatch": 0.0,
                "apgi_anxiety_level": 0.0,
            }

        return {
            **{
                "num_trials": len(self.experiment.trials),
                "completion_time_s": completion_time,
                "d_prime": summary.get("d_prime", 0.0),
                "learning_effect_ms": summary.get("learning_effect_ms", 0.0),
                "sequential_rt_ms": summary.get("sequential_rt_ms", 0.0),
                "random_rt_ms": summary.get("random_rt_ms", 0.0),
                "accuracy": summary.get("accuracy", 0.0),
            },
            **apgi_metrics,
        }


def print_results(results: Dict) -> None:
    print("\n" + "=" * 60)
    print("SERIAL REACTION TIME EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Trials: {results['num_trials']}")

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
    print(f"Learning Effect: {results['learning_effect_ms']:.1f}ms")
    print(f"Sequential RT: {results['sequential_rt_ms']:.1f}ms")
    print(f"Random RT: {results['random_rt_ms']:.1f}ms")
    print(f"Accuracy: {results['accuracy']:.1%}")
    print("=" * 60)


def main(args: Any) -> Dict:
    """Main function for running the experiment."""
    runner = EnhancedSRTRunner()
    results = runner.run_experiment()
    return results


if __name__ == "__main__":
    parser = create_standard_parser("Run Serial Reaction Time  experiment")
    cli_entrypoint(main, parser)
