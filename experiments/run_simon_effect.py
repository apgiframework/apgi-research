"""
Simon Effect Experiment Runner

This is the AGENT-EDITABLE file for the auto-improvement system.
Modify this file to experiment with different task parameters, stimulus configurations,
and analysis approaches to maximize the simon_effect_ms metric.

Usage:
    uv run run_simon_effect.py

Output:
    Prints summary with simon_effect_ms (primary metric to maximize)

Modification Guidelines:
    - You CAN modify: task parameters, stimulus configurations, spatial cues, etc.
    - You CANNOT modify: prepare_simon_effect.py (stimulus sets are fixed)
    - Goal: Maximize simon_effect_ms (spatial interference effect)
    - Time budget: 10 minutes max per run
"""

import time
from typing import Any, Dict, Optional, Union, cast

import numpy as np

# Standardized APGI imports
from apgi_cli import cli_entrypoint, create_standard_parser

# APGI Integration
from apgi_integration import APGIIntegration, APGIParameters

from .prepare_simon_effect import APGI_PARAMS, TIME_BUDGET, SimonExperiment, TrialType
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

# RT parameters (ms)
CONGRUENT_RT = 450
INCONGRUENT_RT = 480
RT_VARIABILITY = 60

# Accuracy
CONGRUENT_ACC = 0.95
INCONGRUENT_ACC = 0.92

# ---------------------------------------------------------------------------
# Simulated Participant
# ---------------------------------------------------------------------------


class SimulatedParticipant:
    def __init__(self, enable_apgi: bool = True):
        self.reset()

    def reset(self) -> None:
        pass

    def process_trial(self, trial_type: TrialType) -> tuple:
        if trial_type == TrialType.CONGRUENT:
            rt = CONGRUENT_RT + np.random.normal(0, RT_VARIABILITY)
            correct = np.random.random() < CONGRUENT_ACC
        else:
            rt = INCONGRUENT_RT + np.random.normal(0, RT_VARIABILITY)
            correct = np.random.random() < INCONGRUENT_ACC

        return correct, max(250, rt)


# ---------------------------------------------------------------------------
# Enhanced Runner
# ---------------------------------------------------------------------------


class EnhancedSimonRunner:
    def __init__(self, enable_apgi: bool = True):
        # Type declarations first
        self.apgi: Optional[APGIIntegration]
        self.hierarchical: Optional[HierarchicalProcessor]
        self.precision_gap: Optional[PrecisionExpectationState]
        self.neuromodulators: Optional[dict[str, float]]
        self.running_stats: Optional[dict[str, float]]
        self.start_time: Optional[float] = None

        self.experiment = SimonExperiment(num_trials=NUM_TRIALS_CONFIG)
        self.participant = SimulatedParticipant()

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
                    tau_S=cast(
                        float,
                        cast(Union[str, int, float], APGI_PARAMS.get("tau_s", 0.35)),
                    ),
                    beta=cast(
                        float,
                        cast(Union[str, int, float], APGI_PARAMS.get("beta", 1.5)),
                    ),
                    theta_0=cast(
                        float,
                        cast(Union[str, int, float], APGI_PARAMS.get("theta_0", 0.5)),
                    ),
                    alpha=cast(
                        float,
                        cast(Union[str, int, float], APGI_PARAMS.get("alpha", 5.5)),
                    ),
                    gamma_M=cast(
                        float,
                        cast(Union[str, int, float], APGI_PARAMS.get("gamma_M", -0.3)),
                    ),
                    lambda_S=cast(
                        float,
                        cast(Union[str, int, float], APGI_PARAMS.get("lambda_S", 0.1)),
                    ),
                    sigma_S=cast(
                        float,
                        cast(Union[str, int, float], APGI_PARAMS.get("sigma_S", 0.05)),
                    ),
                    sigma_theta=cast(
                        float,
                        cast(
                            Union[str, int, float], APGI_PARAMS.get("sigma_theta", 0.02)
                        ),
                    ),
                    sigma_M=cast(
                        float,
                        cast(Union[str, int, float], APGI_PARAMS.get("sigma_M", 0.03)),
                    ),
                    rho=cast(
                        float,
                        cast(Union[str, int, float], APGI_PARAMS.get("rho", 0.7)),
                    ),
                    theta_survival=cast(
                        float,
                        cast(
                            Union[str, int, float],
                            APGI_PARAMS.get("theta_survival", 0.3),
                        ),
                    ),
                    theta_neutral=cast(
                        float,
                        cast(
                            Union[str, int, float],
                            APGI_PARAMS.get("theta_neutral", 0.7),
                        ),
                    ),
                    beta_cross=cast(
                        float,
                        cast(
                            Union[str, int, float], APGI_PARAMS.get("beta_cross", 0.2)
                        ),
                    ),
                    tau_levels=cast(
                        list[float],
                        APGI_PARAMS.get("tau_levels", [0.1, 0.2, 0.4, 1.0, 5.0]),
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

        correct, rt = self.participant.process_trial(trial.trial_type)

        self.experiment.run_trial(
            trial=trial,
            response=trial.correct_response if correct else "wrong",
            rt_ms=rt,
        )

        # APGI component: Process with APGI if enabled
        if self.apgi:
            assert self.neuromodulators is not None
            assert self.running_stats is not None
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

        return {
            "num_trials": len(self.experiment.trials),
            "completion_time_s": completion_time,
            "d_prime": summary.get("d_prime", 0.0),
            "simon_effect_ms": summary.get("simon_effect_ms", 50.0),
            "congruent_rt_ms": summary.get("congruent_rt_ms", 500.0),
            "incongruent_rt_ms": summary.get("incongruent_rt_ms", 550.0),
            "accuracy": summary.get("accuracy", 0.9),
            **apgi_metrics,
        }


def print_results(results: Dict) -> None:
    print("\n" + "=" * 60)
    print("SIMON EFFECT EXPERIMENT RESULTS")
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
    print(f"Simon Effect: {results['simon_effect_ms']:.1f}ms")
    print(f"Congruent RT: {results['congruent_rt_ms']:.1f}ms")
    print(f"Incongruent RT: {results['incongruent_rt_ms']:.1f}ms")
    print(f"Accuracy: {results['accuracy']:.1%}")
    print("=" * 60)


def main(args: Any) -> Dict:
    """Main function for running the experiment."""
    runner = EnhancedSimonRunner()
    results = runner.run_experiment()
    return results


if __name__ == "__main__":
    parser = create_standard_parser("Run Simon Effect  experiment")
    cli_entrypoint(main, parser)
