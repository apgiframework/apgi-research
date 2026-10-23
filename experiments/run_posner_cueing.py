"""
Posner Cueing Experiment Runner

This is the AGENT-EDITABLE file for the auto-improvement system.
Modify this file to experiment with different task parameters, SOA values,
and analysis approaches to maximize the validity_effect_ms metric.

Usage:
    uv run run_posner_cueing.py

Output:
    Prints summary with validity_effect_ms (primary metric to maximize)

Modification Guidelines:
    - You CAN modify: task parameters, SOA values, cue validity, etc.
    - You CANNOT modify: prepare_posner_cueing.py (stimulus sets are fixed)
    - Goal: Maximize validity_effect_ms (cue validity effect on reaction times)
    - Time budget: 10 minutes max per run
"""

import time
from typing import Any, Dict, Optional, cast

import numpy as np

# Standardized APGI imports
from apgi_cli import cli_entrypoint, create_standard_parser

# APGI Integration
from apgi_integration import APGIIntegration

from .experiment_apgi_integration import APGIParameters
from .prepare_posner_cueing import APGI_PARAMS, TIME_BUDGET, CueType, PosnerExperiment
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

# RT parameters (ms)
VALID_RT = 350
INVALID_RT = 450
NEUTRAL_RT = 400
RT_VARIABILITY = 60

# SOA values
SOA_VALUES_CONFIG = [100, 200, 400, 800]

# ---------------------------------------------------------------------------
# Simulated Participant
# ---------------------------------------------------------------------------


class SimulatedParticipant:
    def __init__(self, enable_apgi: bool = True):
        self.reset()

    def reset(self) -> None:
        pass

    def process_trial(self, cue_type: CueType) -> tuple:
        if cue_type == CueType.VALID:
            rt = VALID_RT + np.random.normal(0, RT_VARIABILITY)
        elif cue_type == CueType.INVALID:
            rt = INVALID_RT + np.random.normal(0, RT_VARIABILITY)
        else:
            rt = NEUTRAL_RT + np.random.normal(0, RT_VARIABILITY)

        correct = np.random.random() < 0.95
        return correct, max(200, rt)


# ---------------------------------------------------------------------------
# Enhanced Runner
# ---------------------------------------------------------------------------


class EnhancedPosnerRunner:
    def __init__(self, enable_apgi: bool = True):
        self.experiment = PosnerExperiment(num_trials=NUM_TRIALS_CONFIG)
        self.participant = SimulatedParticipant()
        self.start_time: Optional[float] = None

        # Initialize 100/100 APGI components
        self.enable_apgi = enable_apgi and APGI_PARAMS.get("enabled", True)
        self.apgi: Optional[APGIIntegration] = None
        self.hierarchical: Optional[HierarchicalProcessor] = None
        self.precision_gap: Optional[PrecisionExpectationState] = None
        self.neuromodulators: Optional[Dict[str, float]] = None
        self.running_stats: Dict[str, float] = {}
        if self.enable_apgi:
            params = APGIParameters(
                tau_S=float(cast(Any, APGI_PARAMS.get("tau_s", 0.35))),
                beta=float(cast(Any, APGI_PARAMS.get("beta", 1.5))),
                theta_0=float(cast(Any, APGI_PARAMS.get("theta_0", 0.5))),
                alpha=float(cast(Any, APGI_PARAMS.get("alpha", 5.5))),
                gamma_M=float(cast(Any, APGI_PARAMS.get("gamma_M", -0.3))),
                lambda_S=float(cast(Any, APGI_PARAMS.get("lambda_S", 0.1))),
                sigma_S=float(cast(Any, APGI_PARAMS.get("sigma_S", 0.05))),
                sigma_theta=float(cast(Any, APGI_PARAMS.get("sigma_theta", 0.02))),
                sigma_M=float(cast(Any, APGI_PARAMS.get("sigma_M", 0.03))),
                rho=float(cast(Any, APGI_PARAMS.get("rho", 0.7))),
                theta_survival=float(cast(Any, APGI_PARAMS.get("theta_survival", 0.3))),
                theta_neutral=float(cast(Any, APGI_PARAMS.get("theta_neutral", 0.7))),
            )
            self.apgi = APGIIntegration(params)

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
                    beta_cross=float(cast(Any, APGI_PARAMS.get("beta_cross", 0.2))),
                    tau_levels=cast(
                        list[float],
                        APGI_PARAMS.get("tau_levels", [0.1, 0.2, 0.4, 1.0, 5.0]),
                    ),
                )
                self.hierarchical = HierarchicalProcessor(ultimate_params)
            else:
                pass  # self.hierarchical already initialized as None

            if APGI_PARAMS.get("precision_gap_enabled", True):
                self.precision_gap = PrecisionExpectationState()
            else:
                pass  # self.precision_gap already initialized as None

            self.neuromodulators = {
                "ACh": float(cast(Any, APGI_PARAMS.get("ACh", 1.0))),
                "NE": float(cast(Any, APGI_PARAMS.get("NE", 1.0))),
                "DA": float(cast(Any, APGI_PARAMS.get("DA", 1.0))),
                "HT5": float(cast(Any, APGI_PARAMS.get("HT5", 1.0))),
            }

            # 100/100: Running statistics for z-score normalization
            self.running_stats = {
                "outcome_mean": 0.5,
                "outcome_var": 0.25,
                "rt_mean": 600.0,
                "rt_var": 36000.0,
            }
        else:
            pass  # Variables already initialized with None/default values above

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

        correct, rt = self.participant.process_trial(trial.cue_type)

        self.experiment.run_trial(
            trial=trial,
            response=trial.target_location if correct else "wrong",
            rt_ms=rt,
        )

        # 100/100: Process with APGI if enabled
        if self.apgi:
            # Compute prediction error from trial outcome
            observed_accuracy = 1.0 if correct else 0.0
            expected_accuracy = 0.5  # Baseline

            # Determine trial type
            trial_type = "neutral"

            # 100/100: Determine precision based on neuromodulators
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

            # 100/100: Update running statistics
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

            # 100/100: Update precision expectation gap (Π vs Π̂)
            if self.precision_gap:
                self.precision_gap.update(
                    precision_ext, precision_int, self.neuromodulators or {}, trial_type
                )
                precision_ext = self.precision_gap.Pi_e_actual
                precision_int = self.precision_gap.Pi_i_actual

            # 100/100: Process with APGI
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

    def _calculate_results(self) -> Dict:
        summary = self.experiment.get_summary()
        completion_time = time.time() - (self.start_time or 0.0)

        results = {
            "num_trials": len(self.experiment.trials),
            "completion_time_s": completion_time,
            "d_prime": summary.get("d_prime", 0.0),
            "accuracy": summary.get("accuracy", 0.0),
            "validity_effect_ms": summary.get("validity_effect_ms", 0.0),
            "valid_rt_ms": summary.get("valid_rt_ms", 0.0),
            "invalid_rt_ms": summary.get("invalid_rt_ms", 0.0),
            "benefit_ms": summary.get("benefit_ms", 0.0),
            "cost_ms": summary.get("cost_ms", 0.0),
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
            if self.precision_gap:
                results["apgi_precision_mismatch"] = (
                    self.precision_gap.precision_mismatch
                )
                results["apgi_anxiety_level"] = self.precision_gap.anxiety_level
            if self.neuromodulators:
                results["apgi_acetylcholine"] = self.neuromodulators.get("ACh", 1.0)
                results["apgi_norepinephrine"] = self.neuromodulators.get("NE", 1.0)
                results["apgi_dopamine"] = self.neuromodulators.get("DA", 1.0)
                results["apgi_serotonin"] = self.neuromodulators.get("HT5", 1.0)
        else:
            results["apgi_enabled"] = False

        return results


def print_results(results: Dict) -> None:
    print("\n" + "=" * 60)
    print("POSNER CUEING EXPERIMENT RESULTS")
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

        # 100/100: Precision expectation gap
        if "apgi_precision_mismatch" in results:
            print(
                f"Precision Mismatch (Π̂-Π): {results['apgi_precision_mismatch']:.3f}"
            )
            print(f"Anxiety Level: {results['apgi_anxiety_level']:.3f}")

        # 100/100: Neuromodulators
        if "apgi_dopamine" in results:
            print("\nNeuromodulator Levels:")
            print(f"  Dopamine (DA): {results['apgi_dopamine']:.2f}")
            print(f"  Serotonin (5-HT): {results['apgi_serotonin']:.2f}")
            print(f"  Acetylcholine (ACh): {results['apgi_acetylcholine']:.2f}")
            print(f"  Norepinephrine (NE): {results['apgi_norepinephrine']:.2f}")

    print(f"Time: {results['completion_time_s']:.2f}s")
    print(f"Validity Effect: {results['validity_effect_ms']:.1f}ms")
    print(f"Valid RT: {results['valid_rt_ms']:.1f}ms")
    print(f"Invalid RT: {results['invalid_rt_ms']:.1f}ms")
    print(f"Benefit: {results['benefit_ms']:.1f}ms")
    print(f"Cost: {results['cost_ms']:.1f}ms")
    print("=" * 60)


def main(args: Any) -> Dict:
    """Main function for running the experiment."""
    runner = EnhancedPosnerRunner()
    results = runner.run_experiment()
    return results


if __name__ == "__main__":
    parser = create_standard_parser("Run Posner Cueing  experiment")
    cli_entrypoint(main, parser)
