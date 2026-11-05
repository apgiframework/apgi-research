"""Probabilistic Category Learning Experiment Runner

This is the AGENT-EDITABLE file for the auto-improvement system.
Modify this file to experiment with different task parameters, category structures,
and analysis approaches to maximize the learning_rate metric.

Usage:
    uv run run_probabilistic_category_learning.py

Output:
    Prints summary with learning_rate (primary metric to maximize)

Modification Guidelines:
    - You CAN modify: task parameters, category structures, learning rates, etc.
    - You CANNOT modify: prepare_probabilistic_category_learning.py (category sets are fixed)
    - Goal: Maximize learning_rate (speed of category acquisition)
    - Time budget: 10 minutes max per run
"""

import time
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely cast a value to float with fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_float_list(
    value: Any, default: Optional[List[float]] = None
) -> Optional[List[float]]:
    """Safely cast a value to list with fallback."""
    try:
        return list(value) if value is not None else default
    except (TypeError, ValueError):
        return default


# Standardized APGI imports
from apgi_cli import cli_entrypoint, create_standard_parser
from apgi_integration import APGIIntegration, APGIParameters

from .prepare_probabilistic_category_learning import (
    APGI_PARAMS,
    TIME_BUDGET,
    PCLExperiment,
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

# Learning parameters
BASE_LEARNING_RATE = 0.10
LEARNING_RATE_VARIABILITY = 0.02

# Category probabilities (sum to 1.0)
CATEGORY_PROBS_CONFIG = [0.3, 0.4, 0.3]

# Feedback parameters
FEEDBACK_PROB = 0.90
NO_FEEDBACK_PROB = 0.10

# ---------------------------------------------------------------------------
# Simulated Participant
# ---------------------------------------------------------------------------


class SimulatedParticipant:
    def __init__(self, enable_apgi: bool = True):
        self.reset()

    def reset(self) -> None:
        self.category_beliefs = np.array(CATEGORY_PROBS_CONFIG)
        self.learning_rate = BASE_LEARNING_RATE + np.random.normal(
            0, LEARNING_RATE_VARIABILITY
        )

    def process_trial(self, correct_category: str, feedback: bool) -> tuple:
        # Update beliefs based on feedback
        if feedback:
            # Simple belief update - map category to index
            if correct_category == "A":
                category_idx = 0
            elif correct_category == "B":
                category_idx = 1
            else:
                category_idx = 0  # Default to A

            likelihood = self.category_beliefs[category_idx]
            self.category_beliefs[category_idx] += likelihood * self.learning_rate
            self.category_beliefs /= np.sum(self.category_beliefs)

        # Make response based on current beliefs
        response_probs = self.category_beliefs / np.sum(self.category_beliefs)
        # Choose a random category based on beliefs
        chosen_idx = np.random.choice(len(self.category_beliefs), p=response_probs)
        chosen_category = (
            ["A", "B", "C"][chosen_idx] if len(self.category_beliefs) == 3 else "A"
        )

        correct = chosen_category == correct_category
        rt = 600 + np.random.normal(0, 100)

        return correct, max(200, rt)


# ---------------------------------------------------------------------------
# Enhanced Runner
# ---------------------------------------------------------------------------


class EnhancedProbabilisticRunner:
    def __init__(self, enable_apgi: bool = True):
        self.experiment = PCLExperiment(num_trials=NUM_TRIALS_CONFIG)
        self.participant = SimulatedParticipant()
        self.start_time: float = 0.0

        # Initialize APGI components
        self.enable_apgi = enable_apgi and APGI_PARAMS.get("enabled", True)
        if self.enable_apgi:
            params = APGIParameters(
                tau_S=safe_float(APGI_PARAMS.get("tau_s", 0.35)),
                beta=safe_float(APGI_PARAMS.get("beta", 1.5)),
                theta_0=safe_float(APGI_PARAMS.get("theta_0", 0.5)),
                alpha=safe_float(APGI_PARAMS.get("alpha", 5.5)),
                gamma_M=safe_float(APGI_PARAMS.get("gamma_M", -0.3)),
                lambda_S=safe_float(APGI_PARAMS.get("lambda_S", 0.1)),
                sigma_S=safe_float(APGI_PARAMS.get("sigma_S", 0.05)),
                sigma_theta=safe_float(APGI_PARAMS.get("sigma_theta", 0.02)),
                sigma_M=safe_float(APGI_PARAMS.get("sigma_M", 0.03)),
                rho=safe_float(APGI_PARAMS.get("rho", 0.7)),
                theta_survival=safe_float(APGI_PARAMS.get("theta_survival", 0.3)),
                theta_neutral=safe_float(APGI_PARAMS.get("theta_neutral", 0.7)),
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
                    beta_cross=safe_float(APGI_PARAMS.get("beta_cross", 0.2)),
                    tau_levels=safe_float_list(
                        APGI_PARAMS.get("tau_levels", [0.1, 0.2, 0.4, 1.0, 5.0])
                    ),
                )
                self.hierarchical = HierarchicalProcessor(ultimate_params)

            # APGI component: Precision expectation gap (Π vs Π̂)
            self.precision_gap: Any = None
            if APGI_PARAMS.get("precision_gap_enabled", True):
                self.precision_gap = PrecisionExpectationState()

            # APGI component: Neuromodulator tracking
            self.neuromodulators: Dict[str, float] = {
                "ACh": safe_float(APGI_PARAMS.get("ACh", 1.0)),
                "NE": safe_float(APGI_PARAMS.get("NE", 1.0)),
                "DA": safe_float(APGI_PARAMS.get("DA", 1.0)),
                "HT5": safe_float(APGI_PARAMS.get("HT5", 1.0)),
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

        feedback = np.random.random() < FEEDBACK_PROB

        correct, rt = self.participant.process_trial(trial.correct_category, feedback)

        self.experiment.run_trial(
            trial=trial,
            response=trial.correct_category if correct else "wrong",
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
            self.ignition_history.append(apgi_state.get("ignition_prob", 0.0))
            self.surprise_history.append(apgi_state.get("surprise", 0.0))
            self.threshold_history.append(apgi_state.get("theta", 0.5))
            self.somatic_marker_history.append(apgi_state.get("M", 0.0))
            self.metabolic_cost_history.append(apgi_state.get("metabolic_cost", 0.0))

    def _calculate_results(self) -> Dict:
        summary = self.experiment.get_summary()
        completion_time = time.time() - self.start_time

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

            # APGI component: Precision expectation gap
            if self.precision_gap:
                apgi_metrics["apgi_precision_mismatch"] = (
                    self.precision_gap.precision_mismatch
                )
                apgi_metrics["apgi_anxiety_level"] = self.precision_gap.anxiety_level

            # APGI component: Neuromodulators
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
                "apgi_enabled": float(cast(Union[str, int, float], self.enable_apgi)),
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
                "learning_rate": summary.get("learning_rate", 0.0),
                "accuracy": summary.get("accuracy", 0.0),
            },
            **apgi_metrics,
        }

    def print_results(self, results: Dict) -> None:
        print("\n" + "=" * 60)
        print("PROBABILISTIC CATEGORY LEARNING EXPERIMENT RESULTS")
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


def print_results(results: Dict) -> None:
    print("\n" + "=" * 60)
    print("PROBABILISTIC CATEGORY LEARNING EXPERIMENT RESULTS")
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
    print(f"Learning Rate: {results['learning_rate']:.3f}")
    print(f"Accuracy: {results['accuracy']:.1%}")
    print("=" * 60)


def main(args: Any) -> Dict:
    """Main function for running the experiment."""
    runner = EnhancedProbabilisticRunner()
    results = runner.run_experiment()
    return results


if __name__ == "__main__":
    parser = create_standard_parser("Run Probabilistic Category Learning  experiment")
    cli_entrypoint(main, parser)
