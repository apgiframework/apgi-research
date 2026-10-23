"""
Artificial Grammar Learning Experiment Runner - 100/100 APGI Compliance

This is the AGENT-EDITABLE file for the auto-improvement system.
Modify this file to experiment with different task parameters, grammar complexities,
and analysis approaches to maximize the grammar_accuracy metric.

Usage:
    uv run run_artificial_grammar_learning.py

Output:
    Prints summary with grammar_accuracy (primary metric to maximize)
    Full APGI metrics including Π vs Π̂ distinction, hierarchical processing,
    neuromodulator mapping (ACh, NE, DA, 5-HT), and domain-specific thresholds.

Modification Guidelines:
    - You CAN modify: task parameters, grammar rules, learning rates, etc.
    - You CANNOT modify: prepare_artificial_grammar_learning.py (grammar sets are fixed)
    - You CANNOT modify: ultimate_apgi_template.py (UltimateAPGIParameters are fixed)
    - You CANNOT modify: apgi_integration.py (APGIParameters are fixed)
    - Time budget: 10 minutes max per run
"""

import time
from typing import Any, Dict, cast

import numpy as np

# Standardized APGI imports
# APGI Integration
from apgi_integration import APGIIntegration, APGIParameters, format_apgi_output
from experiments.ultimate_apgi_template import (
    HierarchicalProcessor,
    PrecisionExpectationState,
    UltimateAPGIParameters,
)

from .prepare_artificial_grammar_learning import (
    APGI_PARAMS,
    AGExperiment,
)

# ---------------------------------------------------------------------------
# MODIFIABLE PARAMETERS
# ---------------------------------------------------------------------------

TIME_BUDGET = 600  # noqa: F811

NUM_TRIALS_CONFIG = 100

# Learning parameters
BASE_LEARNING_RATE = 0.05
LEARNING_RATE_VARIABILITY = 0.01

# Grammar parameters
SYMBOL_SET_SIZE = 10
RULE_COMPLEXITY_LEVELS = [1, 2, 3]  # 1=simple, 3=complex

# ---------------------------------------------------------------------------
# Simulated Participant
# ---------------------------------------------------------------------------


class SimulatedParticipant:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.grammar_accuracy = 0.5
        self.learning_rate = BASE_LEARNING_RATE

    def process_trial(self, is_grammatical: bool, response_made: bool) -> tuple:
        # Learning improves with correct responses
        if response_made and is_grammatical:
            self.grammar_accuracy += self.learning_rate * np.random.uniform(0.5, 1.5)
            self.grammar_accuracy = min(0.95, self.grammar_accuracy)

        # Simulate response based on current accuracy
        response_correct = np.random.random() < self.grammar_accuracy
        # Realistic RT: 400-1200ms
        rt = 600 + np.random.normal(0, 150)

        return response_correct, max(300, rt)


# ---------------------------------------------------------------------------
# Enhanced Runner with 100/100 APGI Compliance
# ---------------------------------------------------------------------------


class EnhancedArtificialGrammarRunner:
    """
    Artificial Grammar Learning Runner with full 100/100 APGI compliance.

    Integrates hierarchical processing, precision expectation gap (Π vs Π̂),
    neuromodulator mapping, and domain-specific thresholds for implicit learning.
    """

    apgi: APGIIntegration | None
    hierarchical: HierarchicalProcessor | None
    precision_gap: PrecisionExpectationState | None
    neuromodulators: dict[str, float] | None
    running_stats: dict[str, float] | None

    def __init__(self, enable_apgi: bool = True):
        self.experiment = AGExperiment()
        self.participant = SimulatedParticipant()
        self.start_time: float | None = None

        # Initialize 100/100 APGI components
        self.enable_apgi = enable_apgi and APGI_PARAMS.get("enabled", True)
        if self.enable_apgi:
            params = APGIParameters(
                tau_S=float(cast(float, APGI_PARAMS.get("tau_s", 0.35))),
                beta=float(cast(float, APGI_PARAMS.get("beta", 1.5))),
                theta_0=float(cast(float, APGI_PARAMS.get("theta_0", 0.5))),
                alpha=float(cast(float, APGI_PARAMS.get("alpha", 5.5))),
                gamma_M=float(cast(float, APGI_PARAMS.get("gamma_M", -0.3))),
                lambda_S=float(cast(float, APGI_PARAMS.get("lambda_S", 0.1))),
                sigma_S=float(cast(float, APGI_PARAMS.get("sigma_S", 0.05))),
                sigma_theta=float(cast(float, APGI_PARAMS.get("sigma_theta", 0.02))),
                sigma_M=float(cast(float, APGI_PARAMS.get("sigma_M", 0.03))),
                rho=float(cast(float, APGI_PARAMS.get("rho", 0.7))),
                theta_survival=float(
                    cast(float, APGI_PARAMS.get("theta_survival", 0.3))
                ),
                theta_neutral=float(cast(float, APGI_PARAMS.get("theta_neutral", 0.7))),
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
                    tau_levels=[
                        float(x)
                        for x in cast(
                            list[float],
                            APGI_PARAMS.get("tau_levels", [0.1, 0.2, 0.4, 1.0, 5.0]),
                        )
                    ],
                )
                self.hierarchical = HierarchicalProcessor(ultimate_params)
            else:
                self.hierarchical = None

            # 100/100: Precision expectation gap (Π vs Π̂)
            if APGI_PARAMS.get("precision_gap_enabled", True):
                self.precision_gap = PrecisionExpectationState()
            else:
                self.precision_gap = None

            # 100/100: Neuromodulator tracking
            self.neuromodulators = {
                "ACh": float(cast(float, APGI_PARAMS.get("ACh", 1.0))),
                "NE": float(cast(float, APGI_PARAMS.get("NE", 1.0))),
                "DA": float(cast(float, APGI_PARAMS.get("DA", 1.0))),
                "HT5": float(cast(float, APGI_PARAMS.get("HT5", 1.0))),
            }

            # 100/100: Running statistics for z-score normalization
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

        # Generate study and test phases
        self.experiment.generate_study_phase()
        test_trials = self.experiment.generate_test_phase()

        for trial in test_trials:
            if time.time() - self.start_time > TIME_BUDGET:
                break
            self._run_single_trial(trial)

        return self._calculate_results()

    def _run_single_trial(self, trial: Any) -> None:
        correct, rt = self.participant.process_trial(trial.is_grammatical, True)

        self.experiment.record_response(
            trial=trial,
            response=correct,
            confidence=5 if correct else 3,
            rt_ms=rt,
        )

        # 100/100: Process with APGI if enabled
        if self.apgi:
            # Compute prediction error from accuracy
            observed_accuracy = 1.0 if correct else 0.0
            expected_accuracy = self.participant.grammar_accuracy

            # Trial type: grammatical strings may have higher salience
            trial_type = "neutral"

            # 100/100: Determine precision based on neuromodulators
            # ACh increases attention precision for implicit learning
            ach_boost = (
                self.neuromodulators.get("ACh", 1.0) if self.neuromodulators else 1.0
            )
            # DA is critical for reward prediction in learning
            da_effect = (
                self.neuromodulators.get("DA", 1.0) if self.neuromodulators else 1.0
            )
            # NE affects arousal during novel pattern detection
            ne_effect = (
                self.neuromodulators.get("NE", 1.0) if self.neuromodulators else 1.0
            )

            precision_ext = 1.5 * ach_boost * (1.0 + 0.2 * da_effect)
            precision_int = 1.5 * (1.0 + 0.2 * ne_effect)

            # 100/100: Update running statistics for z-score normalization
            alpha_mu = 0.01
            alpha_sigma = 0.005
            if self.running_stats:
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
            if self.precision_gap and self.neuromodulators:
                self.precision_gap.update(
                    precision_ext, precision_int, self.neuromodulators, trial_type
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

    def _calculate_results(self) -> Dict:
        summary = self.experiment.get_summary()
        assert self.start_time is not None, "start_time should be set by run_experiment"
        completion_time = time.time() - self.start_time

        results = {
            "num_trials": len(self.experiment.test_items),
            "completion_time_s": completion_time,
            "grammar_accuracy": summary.get("accuracy", 0.0),
            "d_prime": summary.get("d_prime", 0.0),
            "grammatical_accuracy": summary.get("grammatical_accuracy", 0.0),
            "non_grammatical_accuracy": summary.get("non_grammatical_accuracy", 0.0),
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


def print_results(results: Dict) -> None:
    print("\n" + "=" * 60)
    print("ARTIFICIAL GRAMMAR LEARNING EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Trials: {results['num_trials']}")
    print(f"Time: {results['completion_time_s']:.2f}s")
    print(f"Grammar Accuracy: {results['grammar_accuracy']:.1%}")
    print(f"D-Prime: {results['d_prime']:.3f}")
    print(f"Grammatical Accuracy: {results['grammatical_accuracy']:.1%}")
    print(f"Non-Grammatical Accuracy: {results['non_grammatical_accuracy']:.1%}")

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
    runner = EnhancedArtificialGrammarRunner()
    results = runner.run_experiment()
    return results


if __name__ == "__main__":
    from apgi_cli import cli_entrypoint, create_standard_parser

    parser = create_standard_parser("Run Artificial Grammar Learning  experiment")
    cli_entrypoint(main, parser)
