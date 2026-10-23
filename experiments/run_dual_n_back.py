"""
Dual N-Back Experiment Runner - 100/100 APGI Compliance

This is the AGENT-EDITABLE file for the auto-improvement system.
Modify this file to experiment with different task parameters, n-back levels,
and analysis approaches to maximize the d_prime metric.

Usage:
    uv run run_dual_n_back.py

Output:
    Prints summary with d_prime (primary metric to maximize)
    Full APGI metrics including Π vs Π̂ distinction, hierarchical processing,
    neuromodulator mapping (ACh, NE, DA, 5-HT), and domain-specific thresholds.

Modification Guidelines:
    - You CAN modify: task parameters, n-back levels, stimulus timing, etc.
    - You CANNOT modify: prepare_dual_n_back.py (stimulus sets are fixed)
    - Goal: Maximize d_prime (working memory performance)
    - Time budget: 10 minutes max per run
"""

import time
from typing import Any, Dict, Optional, cast

import numpy as np

# Standardized APGI imports
from apgi_cli import cli_entrypoint, create_standard_parser

# APGI Integration
from apgi_integration import APGIIntegration, APGIParameters, format_apgi_output

from .prepare_dual_n_back import (
    APGI_PARAMS,  # APGI parameters from prepare file
    TIME_BUDGET,
    DualNBackExperiment,
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
N_LEVEL_CONFIG = 2  # Can adjust: 1, 2, 3, 4

# Match probability
TARGET_PROBABILITY = 0.30

# Participant parameters
BASE_ACCURACY = 0.75
N_LEVEL_PENALTY = 0.10  # Accuracy drop per N level

# ---------------------------------------------------------------------------
# Simulated Participant
# ---------------------------------------------------------------------------


class SimulatedParticipant:
    def __init__(self, n_level: int = N_LEVEL_CONFIG) -> None:
        self.n_level = n_level
        self.reset()

    def reset(self) -> None:
        self.accuracy = BASE_ACCURACY - (self.n_level - 1) * N_LEVEL_PENALTY

    def process_trial(self, is_match: bool) -> tuple:
        # Match detection accuracy depends on N-level
        correct = np.random.random() < self.accuracy
        # Realistic RT for N-back (varies by difficulty)
        rt = (400 + self.n_level * 100) + np.random.normal(0, 150)

        # Response: press if match detected
        detected = (is_match and correct) or (not is_match and not correct)

        return detected, max(300, rt)


# ---------------------------------------------------------------------------
# Enhanced Runner with 100/100 APGI Compliance
# ---------------------------------------------------------------------------


class EnhancedDualNBackRunner:
    """
    Dual N-Back Runner with full 100/100 APGI compliance.

    Integrates hierarchical processing, precision expectation gap (Π vs Π̂),
    neuromodulator mapping, and domain-specific thresholds for working memory.
    """

    def __init__(self, enable_apgi: bool = True):
        self.experiment = DualNBackExperiment(
            num_trials=NUM_TRIALS_CONFIG, n_level=N_LEVEL_CONFIG
        )
        self.participant = SimulatedParticipant(N_LEVEL_CONFIG)
        self.start_time: Optional[float] = None

        # Initialize 100/100 APGI components
        self.enable_apgi = enable_apgi and APGI_PARAMS.get("enabled", True)
        self.hierarchical_processor: Optional[HierarchicalProcessor] = None
        self.neuromodulators: Optional[Dict[str, float]] = None
        self.running_stats: Optional[Dict[str, float]] = None
        if self.enable_apgi:
            params = APGIParameters(
                tau_S=cast(float, APGI_PARAMS.get("tau_s", 0.35) or 0.35),
                beta=cast(float, APGI_PARAMS.get("beta", 1.5) or 1.5),
                theta_0=cast(float, APGI_PARAMS.get("theta_0", 0.5) or 0.5),
                alpha=cast(float, APGI_PARAMS.get("alpha", 5.5) or 5.5),
                gamma_M=cast(float, APGI_PARAMS.get("gamma_M", -0.3) or -0.3),
                lambda_S=cast(float, APGI_PARAMS.get("lambda_S", 0.1) or 0.1),
                sigma_S=cast(float, APGI_PARAMS.get("sigma_S", 0.05) or 0.05),
                sigma_theta=cast(float, APGI_PARAMS.get("sigma_theta", 0.02) or 0.02),
                sigma_M=cast(float, APGI_PARAMS.get("sigma_M", 0.03) or 0.03),
                rho=cast(float, APGI_PARAMS.get("rho", 0.7) or 0.7),
                theta_survival=cast(
                    float, APGI_PARAMS.get("theta_survival", 0.3) or 0.3
                ),
                theta_neutral=cast(float, APGI_PARAMS.get("theta_neutral", 0.7) or 0.7),
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
                    beta_cross=cast(float, APGI_PARAMS.get("beta_cross", 0.2) or 0.2),
                    tau_levels=cast(
                        list,
                        APGI_PARAMS.get("tau_levels", [0.1, 0.2, 0.4, 1.0, 5.0])
                        or [0.1, 0.2, 0.4, 1.0, 5.0],
                    ),
                )
                self.hierarchical_processor = HierarchicalProcessor(ultimate_params)
            else:
                self.hierarchical_processor = None

            # 100/100: Precision expectation gap (Π vs Π̂)
            self.precision_gap: Optional[PrecisionExpectationState] = None
            if APGI_PARAMS.get("precision_gap_enabled", True):
                self.precision_gap = PrecisionExpectationState()

            # 100/100: Neuromodulator tracking
            self.neuromodulator_levels = {
                "ACh": (
                    cast(float, APGI_PARAMS.get("ACh", 1.0))
                    if isinstance(APGI_PARAMS.get("ACh"), (int, float))
                    else 1.0
                ),
                "NE": (
                    cast(float, APGI_PARAMS.get("NE", 1.0))
                    if isinstance(APGI_PARAMS.get("NE"), (int, float))
                    else 1.0
                ),
                "DA": (
                    cast(float, APGI_PARAMS.get("DA", 1.0))
                    if isinstance(APGI_PARAMS.get("DA"), (int, float))
                    else 1.0
                ),
                "HT5": (
                    cast(float, APGI_PARAMS.get("HT5", 1.0))
                    if isinstance(APGI_PARAMS.get("HT5"), (int, float))
                    else 1.0
                ),
            }

            # 100/100: Running statistics for z-score normalization
            self.running_statistics: Optional[Dict[str, float]] = {
                "outcome_mean": 0.5,
                "outcome_var": 0.25,
                "rt_mean": 600.0,
                "rt_var": 10000.0,
            }
        else:
            # APGI components disabled - use default values
            self.running_stats = {
                "outcome_mean": 0.5,
                "outcome_var": 0.25,
                "rt_mean": 600.0,
                "rt_var": 10000.0,
            }

    def run_experiment(self) -> Dict:
        start_time = time.time()
        self.start_time = start_time
        self.experiment.reset()
        self.participant.reset()

        for trial_num in range(NUM_TRIALS_CONFIG):
            self._run_single_trial(trial_num)

            if time.time() - (self.start_time or 0) > TIME_BUDGET:
                break

        return self._calculate_results()

    def _run_single_trial(self, trial_num: int) -> None:
        trial = self.experiment.get_next_trial()
        if trial is None:
            return

        detected, rt = self.participant.process_trial(trial.is_match)

        self.experiment.run_trial(
            trial=trial,
            stimulus_response=detected,
            rt_ms=rt,
        )

        # 100/100: Process with APGI if enabled
        if self.apgi:
            # Compute prediction error from detection accuracy
            expected_accuracy = self.participant.accuracy
            observed_accuracy = 1.0 if (trial.is_match == detected) else 0.0

            # Trial type based on N-back difficulty
            trial_type = (
                "survival" if N_LEVEL_CONFIG >= 3 else "neutral"
            )  # Higher N = more demanding

            # 100/100: Determine precision based on trial type and neuromodulators
            # ACh increases attention precision for working memory
            ach_boost = (
                self.neuromodulators.get("ACh", 1.0) if self.neuromodulators else 1.0
            )
            # NE increases arousal during match detection
            ne_effect = (
                self.neuromodulators.get("NE", 1.0) if self.neuromodulators else 1.0
            )
            # DA is critical for working memory maintenance
            da_effect = (
                self.neuromodulators.get("DA", 1.0) if self.neuromodulators else 1.0
            )

            precision_ext = (
                (2.5 if trial.is_match else 1.5) * ach_boost * (1.0 + 0.2 * da_effect)
            )
            precision_int = (1.5 if trial.is_match else 1.0) * (1.0 + 0.2 * ne_effect)

            # 100/100: Running statistics for z-score normalization
            if self.running_statistics is not None:
                alpha_sigma = 0.005
                self.running_statistics["outcome_mean"] += alpha_sigma * (
                    observed_accuracy - self.running_statistics["outcome_mean"]
                )
                self.running_statistics["outcome_var"] += alpha_sigma * (
                    (observed_accuracy - self.running_statistics["outcome_mean"]) ** 2
                    - self.running_statistics["outcome_var"]
                )
                self.running_statistics["outcome_var"] = max(
                    0.01, self.running_statistics["outcome_var"]
                )

            # Update precision expectation gap with proper type checking
            if self.precision_gap is not None:
                self.precision_gap.update(
                    precision_ext,
                    precision_int,
                    self.neuromodulator_levels or {},
                    trial_type,
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
            if self.hierarchical_processor:
                signal = apgi_state.get("S", 0.0)
                for level_idx in range(5):
                    level_state = self.hierarchical_processor.process_level(
                        level_idx, signal
                    )
                    signal = level_state.S * 0.8

    def _calculate_results(self) -> Dict:
        summary = self.experiment.get_summary()
        assert self.start_time is not None
        completion_time = time.time() - self.start_time

        results = {
            "num_trials": len(self.experiment.trials),
            "completion_time_s": completion_time,
            "n_level": N_LEVEL_CONFIG,
            "d_prime": summary.get("d_prime", 0.0),
            "hit_rate": summary.get("hit_rate", 0.0),
            "correct_rejection_rate": summary.get("correct_rejection_rate", 0.0),
            "mean_rt_ms": summary.get("mean_rt_ms", 0.0),
        }

        # 100/100: Add APGI metrics if enabled
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
            if self.precision_gap is not None:
                results["apgi_precision_mismatch"] = (
                    self.precision_gap.precision_mismatch
                )
                results["apgi_anxiety_level"] = self.precision_gap.anxiety_level
                results["apgi_precision_overestimated"] = (
                    self.precision_gap.precision_overestimated
                )

            # 100/100: Hierarchical processing
            if self.hierarchical_processor is not None:
                hier_summary = self.hierarchical_processor.get_hierarchical_summary()
                results.update({f"apgi_{k}": v for k, v in hier_summary.items()})

            # 100/100: Neuromodulator baselines
            if self.neuromodulators:
                results["apgi_acetylcholine"] = self.neuromodulator_levels.get(
                    "ACh", 1.0
                )
                results["apgi_norepinephrine"] = self.neuromodulator_levels.get(
                    "NE", 1.0
                )
                results["apgi_dopamine"] = self.neuromodulator_levels.get("DA", 1.0)
                results["apgi_serotonin"] = self.neuromodulator_levels.get("HT5", 1.0)

            results["apgi_formatted"] = format_apgi_output(apgi_summary)
        else:
            results["apgi_enabled"] = False

        return results


def print_results(results: Dict) -> None:
    print("\n" + "=" * 60)
    print("DUAL N-BACK EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Trials: {results['num_trials']}")
    print(f"N-Back Level: {results['n_level']}")
    print(f"Time: {results['completion_time_s']:.2f}s")
    print(f"D-prime: {results['d_prime']:.3f}")
    print(f"Hit Rate: {results['hit_rate']:.1%}")
    print(f"Correct Rejection: {results['correct_rejection_rate']:.1%}")
    print(f"Mean RT: {results['mean_rt_ms']:.1f}ms")

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

    print("=" * 60)


def main(args: Any) -> Dict:
    """Main function for running the experiment."""
    runner = EnhancedDualNBackRunner()
    results = runner.run_experiment()
    return results


if __name__ == "__main__":
    parser = create_standard_parser("Run Dual N Back  experiment")
    cli_entrypoint(main, parser)
