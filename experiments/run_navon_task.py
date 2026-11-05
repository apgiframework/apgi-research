"""
Navon Task (Global-Local) Experiment Runner - APGI integration (internal feature checklist; not registered-protocol validation)

This is the AGENT-EDITABLE file for the auto-improvement system.
Modify this file to experiment with different task parameters, stimulus configurations,
and analysis approaches to maximize the global_advantage_ms metric.

Usage:
    uv run run_navon_task.py

Output:
    Prints summary with global_advantage_ms (primary metric to maximize)
    Full APGI metrics including Π vs Π̂ distinction, hierarchical processing,
    neuromodulator mapping (ACh, NE, DA, 5-HT), and domain-specific thresholds.

Modification Guidelines:
    - You CAN modify: task parameters, stimulus configurations, attentional cues, etc.
    - You CANNOT modify: prepare_navon_task.py (stimulus sets are fixed)
    - Goal: Maximize global_advantage_ms (global vs local processing advantage)
    - Time budget: 10 minutes max per run
"""

import time
from typing import Any, Dict, Optional, cast

import numpy as np

# Standardized APGI imports
from apgi_cli import cli_entrypoint, create_standard_parser

# APGI Integration
from apgi_integration import APGIIntegration, APGIParameters, format_apgi_output

from .prepare_navon_task import APGI_PARAMS, NavonExperiment, TargetLevel
from .ultimate_apgi_template import (
    HierarchicalProcessor,
    PrecisionExpectationState,
    UltimateAPGIParameters,
)

# Local TIME_BUDGET for reference (must match prepare file)
TIME_BUDGET = float(
    cast(Any, APGI_PARAMS.get("time_budget", 600))
)  # 10 minutes per experiment

# ---------------------------------------------------------------------------
# MODIFIABLE PARAMETERS
# ---------------------------------------------------------------------------

NUM_TRIALS_CONFIG = 100

# RT parameters (ms)
GLOBAL_RT = 450
LOCAL_RT = 520
RT_VARIABILITY = 70

# Accuracy
GLOBAL_ACC = 0.95
LOCAL_ACC = 0.90

# ---------------------------------------------------------------------------
# Simulated Participant
# ---------------------------------------------------------------------------


class SimulatedParticipant:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        pass

    def process_trial(self, target_level: TargetLevel) -> tuple:
        if target_level == TargetLevel.GLOBAL:
            rt = GLOBAL_RT + np.random.normal(0, RT_VARIABILITY)
            correct = np.random.random() < GLOBAL_ACC
        else:
            rt = LOCAL_RT + np.random.normal(0, RT_VARIABILITY)
            correct = np.random.random() < LOCAL_ACC

        return correct, max(300, rt)


# ---------------------------------------------------------------------------
# Enhanced Runner
# ---------------------------------------------------------------------------


class EnhancedNavonRunner:
    """
    Navon Task Runner with internal APGI feature checklist.

    Integrates hierarchical processing, precision expectation gap (Π vs Π̂),
    neuromodulator mapping, and domain-specific thresholds for global-local attention.
    """

    def __init__(self, enable_apgi: bool = True):
        self.experiment = NavonExperiment(num_trials=NUM_TRIALS_CONFIG)
        self.participant = SimulatedParticipant()
        self.start_time: Optional[float] = None

        # Initialize APGI components
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

            # APGI component: Running statistics for z-score normalization
            self.running_stats = {
                "outcome_mean": 0.5,
                "outcome_var": 0.25,
                "rt_mean": 450.0,
                "rt_var": 20250.0,
            }
        else:
            pass  # Variables already initialized with None/default values above

    def run_experiment(self) -> Dict:
        self.start_time = time.time()
        self.experiment.reset()
        self.participant.reset()

        for trial_num in range(NUM_TRIALS_CONFIG):
            self._run_single_trial(trial_num)

            if time.time() - (self.start_time or 0.0) > TIME_BUDGET:
                break

        return self._calculate_results()

    def _run_single_trial(self, trial_num: int) -> None:
        trial = self.experiment.get_next_trial()
        if trial is None:
            return

        correct, rt = self.participant.process_trial(trial.target_level)

        self.experiment.run_trial(
            trial=trial,
            response=trial.target_letter if correct else "wrong",
            rt_ms=rt,
        )

        # APGI component: Process with APGI if enabled
        if self.apgi:
            # Compute prediction error from RT
            expected_rt = (
                GLOBAL_RT if trial.target_level == TargetLevel.GLOBAL else LOCAL_RT
            )
            observed_rt = rt
            # Trial type based on global vs local processing
            trial_type = (
                "survival" if trial.target_level == TargetLevel.LOCAL else "neutral"
            )

            # APGI component: Determine precision based on attention level and neuromodulators
            # ACh increases attention precision
            ach_boost = (
                self.neuromodulators.get("ACh", 1.0) if self.neuromodulators else 1.0
            )
            # NE increases arousal for local processing (more demanding)
            ne_effect = (
                self.neuromodulators.get("NE", 1.0) if self.neuromodulators else 1.0
            )

            precision_ext = (
                2.0 if trial.target_level == TargetLevel.LOCAL else 1.5
            ) * ach_boost
            precision_int = (1.5 if correct else 1.0) * (1.0 + 0.2 * ne_effect)

            # APGI component: Update running statistics for z-score normalization
            alpha_mu = 0.01
            self.running_stats["rt_mean"] += alpha_mu * (
                observed_rt - self.running_stats["rt_mean"]
            )
            self.running_stats["rt_var"] += 0.005 * (
                (observed_rt - self.running_stats["rt_mean"]) ** 2
                - self.running_stats["rt_var"]
            )
            self.running_stats["rt_var"] = max(100.0, self.running_stats["rt_var"])

            # APGI component: Update precision expectation gap (Π vs Π̂)
            if self.precision_gap:
                self.precision_gap.update(
                    precision_ext, precision_int, self.neuromodulators or {}, trial_type
                )
                precision_ext = self.precision_gap.Pi_e_actual
                precision_int = self.precision_gap.Pi_i_actual

            # APGI component: Process with APGI - computes ignition, surprise, somatic markers
            apgi_state = self.apgi.process_trial(
                observed=observed_rt / 1000.0,  # Normalize to seconds
                predicted=expected_rt / 1000.0,
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
        completion_time = time.time() - (self.start_time or 0.0)

        results = {
            "num_trials": len(self.experiment.trials),
            "completion_time_s": completion_time,
            "global_advantage_ms": summary.get("global_advantage_ms", 0.0),
            "global_rt_ms": summary.get("global_rt_ms", 0.0),
            "local_rt_ms": summary.get("local_rt_ms", 0.0),
            "interference_effect_ms": summary.get("interference_effect_ms", 0.0),
            "accuracy": summary.get("accuracy", 0.0),
        }

        # APGI component: Add APGI metrics if enabled
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

            # APGI component: Precision expectation gap (Π vs Π̂)
            if self.precision_gap:
                results["apgi_precision_mismatch"] = (
                    self.precision_gap.precision_mismatch
                )
                results["apgi_anxiety_level"] = self.precision_gap.anxiety_level
                results["apgi_precision_overestimated"] = (
                    self.precision_gap.precision_overestimated
                )

            # APGI component: Hierarchical processing
            if self.hierarchical:
                hier_summary = self.hierarchical.get_hierarchical_summary()
                results.update({f"apgi_{k}": v for k, v in hier_summary.items()})

            # APGI component: Neuromodulator baselines
            if self.neuromodulators:
                results["apgi_acetylcholine"] = self.neuromodulators.get("ACh", 1.0)
                results["apgi_norepinephrine"] = self.neuromodulators.get("NE", 1.0)
                results["apgi_dopamine"] = self.neuromodulators.get("DA", 1.0)
                results["apgi_serotonin"] = self.neuromodulators.get("HT5", 1.0)

            results["apgi_formatted"] = format_apgi_output(apgi_summary)
        else:
            results["apgi_enabled"] = False

        return results


def print_results(results: Dict) -> None:
    print("\n" + "=" * 60)
    print("NAVON TASK EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Trials: {results['num_trials']}")
    print(f"Time: {results['completion_time_s']:.2f}s")
    print(f"Global Advantage: {results['global_advantage_ms']:.1f}ms")
    print(f"Global RT: {results['global_rt_ms']:.1f}ms")
    print(f"Local RT: {results['local_rt_ms']:.1f}ms")
    print(f"Interference Effect: {results['interference_effect_ms']:.1f}ms")
    print(f"Accuracy: {results['accuracy']:.1%}")

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

    print("=" * 60)


def main(args: Any) -> Dict:
    """Main function for running the experiment."""
    runner = EnhancedNavonRunner()
    results = runner.run_experiment()
    return results


if __name__ == "__main__":
    parser = create_standard_parser("Run Navon Task  experiment")
    cli_entrypoint(main, parser)
