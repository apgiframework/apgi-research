"""Iowa Gambling Task (IGT) Experiment Runner - APGI integration (internal feature checklist; not registered-protocol validation)

This is the AGENT-EDITABLE file for the auto-improvement system with full APGI integration:
- Hierarchical 5-level processing
- Π vs Π̂ precision expectation gap
- Neuromodulator effects (DA for reward learning)
- Domain-specific thresholds for risk assessment

Modify this file to experiment with task parameters while maintaining APGI compliance.

Usage:
    uv run run_iowa_gambling_task.py

Output:
    Prints summary with net_score (primary metric) and full APGI metrics
"""

import time
from typing import Any, Dict, cast

import numpy as np

# Standardized APGI imports
from apgi_cli import cli_entrypoint, create_standard_parser

# APGI Integration with hierarchical processing and precision gap
from apgi_integration import APGIIntegration

from .experiment_apgi_integration import APGIParameters
from .prepare_iowa_gambling_task import (
    APGI_PARAMS,
    DECK_LABELS,
    TIME_BUDGET,
    IowaGamblingTaskExperiment,
)
from .ultimate_apgi_template import (
    HierarchicalProcessor,
    PrecisionExpectationState,
    UltimateAPGIParameters,
)

# ---------------------------------------------------------------------------
# MODIFIABLE PARAMETERS
# ---------------------------------------------------------------------------

# TIME_BUDGET is imported from prepare_iowa_gambling_task
assert TIME_BUDGET == 600, "TIME_BUDGET must be 600"

NUM_TRIALS_CONFIG = 100

# Learning parameters - adjust these to optimize performance
BASE_LEARNING_RATE = 0.15
LEARNING_RATE_VARIABILITY = 0.05

# Exploration vs exploitation balance
EXPLORATION_PROB = 0.10  # Probability of random exploration
GREEDY_AFTER_LEARNING = True  # Switch to greedy after learning phase
LEARNING_PHASE_TRIALS = 40  # Number of trials for learning phase

# Deck preference decay - how quickly preferences fade
PREFERENCE_DECAY = 0.995

# ---------------------------------------------------------------------------
# Simulated Participant with Learning
# ---------------------------------------------------------------------------


class SimulatedParticipant:
    """
    Simulates a participant learning the IGT deck values.
    Uses reinforcement learning with exploration/exploitation tradeoff.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # Track expected value for each deck (A, B, C, D)
        self.deck_values = np.zeros(4)
        self.deck_counts = np.zeros(4)
        self.total_reward = 0.0
        self.trials_completed = 0

    def choose_deck(self) -> str:
        """
        Choose a deck using epsilon-greedy strategy with learned values.
        """
        self.trials_completed += 1

        # Exploration phase
        if np.random.random() < EXPLORATION_PROB:
            return str(np.random.choice(DECK_LABELS))

        # Greedy choice based on learned values
        if GREEDY_AFTER_LEARNING and self.trials_completed > LEARNING_PHASE_TRIALS:
            # Pure exploitation after learning
            best_idx = np.argmax(self.deck_values)
            return DECK_LABELS[best_idx]

        # Softmax selection during learning
        if np.all(self.deck_values == 0):
            return str(np.random.choice(DECK_LABELS))

        # Temperature decreases over time (more exploitation later)
        temperature = max(0.5, 2.0 - self.trials_completed * 0.02)
        exp_values = np.exp(self.deck_values / temperature)
        probs = exp_values / np.sum(exp_values)
        chosen_idx = np.random.choice(4, p=probs)
        return DECK_LABELS[chosen_idx]

    def update_from_outcome(self, deck_choice: str, outcome: int) -> None:
        """
        Update learned values based on trial outcome.
        """
        deck_idx = DECK_LABELS.index(deck_choice)
        self.deck_counts[deck_idx] += 1

        # Running average of rewards for each deck
        n = self.deck_counts[deck_idx]
        old_value = self.deck_values[deck_idx]
        self.deck_values[deck_idx] = old_value + (outcome - old_value) / n

        self.total_reward += outcome

        # Apply preference decay to encourage exploration of other decks
        self.deck_values *= PREFERENCE_DECAY


# ---------------------------------------------------------------------------
# Enhanced Runner
# ---------------------------------------------------------------------------


class EnhancedIGTRunner:
    """IGT Runner with the internal APGI feature checklist enabled."""

    apgi: APGIIntegration | None
    hierarchical: HierarchicalProcessor | None
    precision_gap: PrecisionExpectationState | None
    neuromodulators: dict[str, float]
    running_stats: dict[str, float]

    def __init__(self, enable_apgi: bool = True):
        self.experiment = IowaGamblingTaskExperiment(num_trials=NUM_TRIALS_CONFIG)
        self.participant = SimulatedParticipant()
        self.start_time: float | None = None

        # Initialize APGI components
        self.enable_apgi = enable_apgi and APGI_PARAMS.get("enabled", True)
        if self.enable_apgi:
            # Core APGI integration
            params = APGIParameters(
                tau_S=float(cast(float, APGI_PARAMS.get("tau_s", 0.4))),
                beta=float(cast(float, APGI_PARAMS.get("beta", 2.0))),
                theta_0=float(cast(float, APGI_PARAMS.get("theta_0", 0.4))),
                alpha=float(cast(float, APGI_PARAMS.get("alpha", 5.0))),
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

            # APGI component: Hierarchical 5-level processing (requires UltimateAPGIParameters)
            if APGI_PARAMS.get("hierarchical_enabled", True):
                tau_levels = APGI_PARAMS.get("tau_levels", [0.1, 0.2, 0.4, 1.0, 5.0])
                if not isinstance(tau_levels, list):
                    tau_levels = [0.1, 0.2, 0.4, 1.0, 5.0]
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
                    tau_levels=[float(x) for x in tau_levels],
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
            neuromodulators_raw = {
                "ACh": APGI_PARAMS.get("ACh", 1.0),
                "NE": APGI_PARAMS.get("NE", 1.0),
                "DA": APGI_PARAMS.get("DA", 1.0),  # Critical for reward learning
                "HT5": APGI_PARAMS.get("HT5", 1.0),
            }
            self.neuromodulators = {
                k: float(cast(float, v)) for k, v in neuromodulators_raw.items()
            }

            # APGI component: Running statistics for z-score normalization
            self.running_stats = {
                "outcome_mean": 0.0,
                "outcome_var": 1.0,
                "rt_mean": 1000.0,
                "rt_var": 100000.0,
            }
        else:
            self.apgi = None
            self.hierarchical = None
            self.precision_gap = None
            self.neuromodulators = {}

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
        # Participant chooses a deck
        deck_choice = self.participant.choose_deck()

        # Run the trial and get outcome
        trial = self.experiment.run_trial(
            deck_choice=deck_choice, reaction_time_ms=np.random.uniform(500, 1500)
        )

        # Update participant's learned values
        self.participant.update_from_outcome(deck_choice, trial.outcome)

        # Process trial with APGI dynamics if enabled
        if self.apgi:
            # Determine trial type based on deck (disadvantageous = survival domain)
            trial_type = "survival" if deck_choice in ["A", "B"] else "neutral"

            # Compute prediction error from expected outcome
            deck_idx = DECK_LABELS.index(deck_choice)
            expected_outcome = self.participant.deck_values[deck_idx]

            # Update running statistics for z-score normalization
            alpha_mu = 0.01
            alpha_sigma = 0.005
            observed_outcome = trial.outcome / 100.0  # Normalize
            if self.running_stats:
                self.running_stats["outcome_mean"] += alpha_mu * (
                    observed_outcome - self.running_stats["outcome_mean"]
                )
                self.running_stats["outcome_var"] += alpha_sigma * (
                    (observed_outcome - self.running_stats["outcome_mean"]) ** 2
                    - self.running_stats["outcome_var"]
                )
                self.running_stats["outcome_var"] = max(
                    0.01, self.running_stats["outcome_var"]
                )

            # APGI component: Determine precision based on trial type and neuromodulators
            # DA (dopamine) increases reward sensitivity
            da_boost = (
                self.neuromodulators.get("DA", 1.0) if self.neuromodulators else 1.0
            )
            # 5-HT (serotonin) increases loss aversion
            ht5_effect = (
                self.neuromodulators.get("HT5", 1.0) if self.neuromodulators else 1.0
            )

            precision_ext = (2.0 if trial_type == "survival" else 1.0) * da_boost
            precision_int = (1.5 if trial.outcome > 0 else 1.0) * ht5_effect

            # APGI component: Update precision expectation gap (Π vs Π̂)
            if self.precision_gap:
                self.precision_gap.update(
                    precision_ext, precision_int, self.neuromodulators, trial_type
                )
                # Use actual precision from gap model
                precision_ext = self.precision_gap.Pi_e_actual
                precision_int = self.precision_gap.Pi_i_actual

            # Process with APGI - computes ignition, surprise, somatic markers
            apgi_state = self.apgi.process_trial(
                observed=observed_outcome,
                predicted=expected_outcome / 100.0 if expected_outcome != 0 else 0,
                trial_type=trial_type,
                precision_ext=precision_ext,
                precision_int=precision_int,
            )

            # APGI component: Process hierarchical levels
            if self.hierarchical:
                signal = apgi_state.get("S", 0.0)
                for level_idx in range(5):
                    level_state = self.hierarchical.process_level(level_idx, signal)
                    signal = level_state.S * 0.8  # Attenuated upward

    def _calculate_results(self) -> Dict:
        summary = self.experiment.get_summary()
        assert self.start_time is not None, "start_time should be set by run_experiment"
        completion_time = time.time() - self.start_time

        results = {
            "num_trials": len(self.experiment.trials),
            "completion_time_s": completion_time,
            "net_score": summary.get("net_score", 0.0),
            "final_money": summary.get("final_money", 0),
            "learning_rate": summary.get("learning_rate", 0.0),
            "advantageous_choices": summary.get("advantageous_choices", 0),
            "disadvantageous_choices": summary.get("disadvantageous_choices", 0),
        }

        # Add APGI metrics if enabled
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
                # Debug: ensure precision gap was updated
                if not hasattr(self.precision_gap, "precision_overestimated"):
                    results["apgi_precision_overestimated"] = False

            # APGI component: Hierarchical processing
            if self.hierarchical:
                hier_summary = self.hierarchical.get_hierarchical_summary()
                results.update({f"apgi_{k}": v for k, v in hier_summary.items()})

            # APGI component: Neuromodulator baselines
            if self.neuromodulators:
                results["apgi_dopamine"] = self.neuromodulators.get("DA", 1.0)
                results["apgi_serotonin"] = self.neuromodulators.get("HT5", 1.0)
                results["apgi_acetylcholine"] = self.neuromodulators.get("ACh", 1.0)
                results["apgi_norepinephrine"] = self.neuromodulators.get("NE", 1.0)

            results["apgi_formatted"] = apgi_summary
        else:
            results["apgi_enabled"] = False

        return results


def print_results(results: Dict) -> None:
    print("\n" + "=" * 60)
    print("IOWA GAMBLING TASK EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Trials: {results['num_trials']}")
    print(f"Time: {results['completion_time_s']:.2f}s")
    print(f"Net Score: {results['net_score']:.1f}")
    print(f"Final Money: ${results['final_money']}")
    print(f"Advantageous Choices: {results['advantageous_choices']}")
    print(f"Disadvantageous Choices: {results['disadvantageous_choices']}")
    print(f"Learning Rate: {results['learning_rate']:.3f}")

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
    runner = EnhancedIGTRunner()
    results = runner.run_experiment()
    return results


if __name__ == "__main__":
    parser = create_standard_parser("Run Iowa Gambling Task  experiment")
    cli_entrypoint(main, parser)
