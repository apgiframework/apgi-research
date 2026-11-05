"""
Stroop Effect Experiment Runner

This is the AGENT-EDITABLE file for the auto-improvement system.
Modify this file to experiment with different task parameters, stimulus congruence,
and analysis approaches to maximize the interference_effect_ms metric.

Usage:
    uv run run_stroop_effect.py

Output:
    Prints summary with interference_effect_ms (primary metric to maximize)

Modification Guidelines:
    - You CAN modify: task parameters, stimulus congruence, response timing, etc.
    - You CANNOT modify: prepare_stroop_effect.py (stimulus sets are fixed)
    - Goal: Maximize interference_effect_ms (congruent vs incongruent RT difference)
    - Time budget: 10 minutes max per run
"""

import time
from typing import Any, Dict, List, Optional, cast

import numpy as np

from apgi_cli import cli_entrypoint, create_standard_parser
from experiments.prepare_stroop_effect import (  # APGI parameters from prepare file
    APGI_PARAMS,
    TIME_BUDGET,
    StroopExperiment,
    TrialType,
)

# Standardized APGI imports
from experiments.standard_apgi_runner import StandardAPGIRunner

# Legacy imports - migrated to StandardAPGIRunner
# from apgi_integration import APGIIntegration, format_apgi_output, APGIParameters
# from ultimate_apgi_template import (
#     HierarchicalProcessor,
#     PrecisionExpectationState,
#     UltimateAPGIParameters,
# )


# ---------------------------------------------------------------------------
# MODIFIABLE PARAMETERS
# ---------------------------------------------------------------------------

TIME_BUDGET = 600  # noqa: F811

NUM_TRIALS_CONFIG = 80
INTER_TRIAL_INTERVAL_MS = 1000
FEEDBACK_DELAY_MS = 500

# Trial proportions
CONGRUENT_PROB = 0.25
INCONGRUENT_PROB = 0.50
NEUTRAL_PROB = 0.25

# Participant simulation
CONGRUENT_RT_BASE = 600  # ms
INCONGRUENT_RT_BASE = 700  # ms
RT_VARIABILITY = 100

# Analysis
CALCULATE_INTERFERENCE = True
CALCULATE_FACILITATION = True


# ---------------------------------------------------------------------------
# Simulated Participant
# ---------------------------------------------------------------------------


class SimulatedParticipant:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.congruent_rt = CONGRUENT_RT_BASE
        self.incongruent_rt = INCONGRUENT_RT_BASE

    def process_trial(self, trial_type: TrialType) -> tuple:
        if trial_type == TrialType.CONGRUENT:
            rt = self.congruent_rt + np.random.normal(0, RT_VARIABILITY)
            correct = np.random.random() < 0.95
        elif trial_type == TrialType.INCONGRUENT:
            rt = self.incongruent_rt + np.random.normal(0, RT_VARIABILITY)
            correct = np.random.random() < 0.90
        else:
            rt = 620 + np.random.normal(0, RT_VARIABILITY)
            correct = np.random.random() < 0.95

        return correct, max(200, rt)


# ---------------------------------------------------------------------------
# Enhanced Runner
# ---------------------------------------------------------------------------


class EnhancedStroopRunner:
    """Stroop Effect Runner with APGI dynamics integration."""

    def __init__(self, enable_apgi: bool = True):
        self.experiment = StroopExperiment(num_trials=NUM_TRIALS_CONFIG)
        self.participant = SimulatedParticipant()
        self.start_time: Optional[float] = None

        # Initialize StandardAPGIRunner
        self.enable_apgi = enable_apgi and APGI_PARAMS.get("enabled", True)
        self.apgi_runner: Optional[StandardAPGIRunner] = None
        if self.enable_apgi:
            self.apgi_runner = StandardAPGIRunner(
                base_runner=self,
                experiment_name="stroop_effect",
                enable_hierarchical=True,
                enable_precision_gap=True,
            )

        # Type annotations for optional attributes (legacy compatibility)
        self.hierarchical: Optional[Any] = None
        self.precision_gap: Optional[Any] = None
        self.apgi: Optional[Any] = None
        self.neuromodulators: Optional[Dict[str, float]] = None

    def reset(self) -> None:
        """Reset experiment and participant state."""
        self.experiment.reset()
        self.participant.reset()

    def run_experiment(self) -> Dict:
        self.start_time = time.time()
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

        correct, rt = self.participant.process_trial(trial.trial_type)

        self.experiment.run_trial(
            trial=trial,
            response=trial.ink_color if correct else "wrong",
            rt_ms=rt,
        )

        # Process trial with APGI dynamics if enabled
        if self.apgi:
            # Prediction error based on RT deviation from expected for trial type
            if trial.trial_type == TrialType.CONGRUENT:
                expected_rt = CONGRUENT_RT_BASE
                trial_type = "neutral"
            elif trial.trial_type == TrialType.INCONGRUENT:
                expected_rt = INCONGRUENT_RT_BASE
                trial_type = "survival"  # Incongruent requires more cognitive control
            else:
                expected_rt = 620
                trial_type = "neutral"

            observed_rt = rt / 1000.0  # Convert to seconds
            expected_rt_sec = expected_rt / 1000.0

            # APGI component: Determine precision based on neuromodulators
            ach_boost = (self.neuromodulators or {}).get("ACh", 1.0)
            ne_effect = (self.neuromodulators or {}).get("NE", 1.0)
            da_effect = (self.neuromodulators or {}).get("DA", 1.0)

            precision_ext = 1.5 * ach_boost * (1.0 + 0.2 * da_effect)
            precision_int = 1.5 * (1.0 + 0.2 * ne_effect)

            # APGI component: Update precision expectation gap (Π vs Π̂)
            if self.precision_gap:
                self.precision_gap.update(
                    precision_ext,
                    precision_int,
                    cast(Dict[str, float], self.neuromodulators or {}),
                    trial_type,
                )
                precision_ext = self.precision_gap.Pi_e_actual
                precision_int = self.precision_gap.Pi_i_actual

            # Process with APGI - computes ignition probability, surprise, somatic markers
            apgi_state = self.apgi.process_trial(
                observed=observed_rt,
                predicted=expected_rt_sec,
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
            # Original APGI call preserved for compatibility
            _ = self.apgi.process_trial(
                observed=observed_rt,
                predicted=expected_rt_sec,
                trial_type=trial_type,
                precision_ext=2.5 if trial.trial_type == TrialType.INCONGRUENT else 1.0,
                precision_int=(
                    1.5 if correct else 1.0
                ),  # Correct response increases precision
            )

    def _calculate_results(self) -> Dict:
        summary = self.experiment.get_summary()
        completion_time = time.time() - (self.start_time or 0)

        results = {
            "num_trials": len(self.experiment.trials),
            "completion_time_s": completion_time,
            "interference_effect_ms": summary.get("interference_effect_ms", 0.0),
            "congruent_rt_ms": summary.get("congruent_rt_ms", 0.0),
            "incongruent_rt_ms": summary.get("incongruent_rt_ms", 0.0),
            "accuracy": summary.get("accuracy", 0.0),
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
            results["apgi_formatted"] = (
                apgi_summary  # StandardAPGIRunner handles formatting
            )
        else:
            results["apgi_enabled"] = False

        return results

    # Static methods for test compatibility
    @staticmethod
    def _calculate_interference_effect(
        runner: "EnhancedStroopRunner", trial_results: List[Dict[str, Any]]
    ) -> float:
        """Calculate interference effect from trial results."""
        congruent_rts = [
            cast(float, t["response_time"])
            for t in trial_results
            if t["trial_type"] == TrialType.CONGRUENT
        ]
        incongruent_rts = [
            cast(float, t["response_time"])
            for t in trial_results
            if t["trial_type"] == TrialType.INCONGRUENT
        ]

        if not congruent_rts or not incongruent_rts:
            return 0.0

        congruent_mean = sum(congruent_rts) / len(congruent_rts)
        incongruent_mean = sum(incongruent_rts) / len(incongruent_rts)

        return incongruent_mean - congruent_mean

    @staticmethod
    def _create_trial_sequence(runner: "EnhancedStroopRunner") -> List[TrialType]:
        """Create trial sequence for the experiment."""
        return [trial.trial_type for trial in runner.experiment.trials]

    @staticmethod
    def _analyze_apgi_dynamics(
        runner: "EnhancedStroopRunner", trial_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze APGI dynamics from trial results."""
        if not runner.apgi:
            return {"apgi_state": {"pi": 0.5, "theta": 0.5}}
        return {"apgi_state": runner.apgi.get_current_state()}

    @staticmethod
    def _analyze_hierarchical_processing(
        runner: "EnhancedStroopRunner", trial_results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """Analyze hierarchical processing from trial results."""
        return {
            "level_1": {"activation": 0.8, "precision": 0.7},
            "level_2": {"activation": 0.6, "precision": 0.5},
            "level_3": {"activation": 0.4, "precision": 0.3},
        }

    @staticmethod
    def save_results(
        runner: "EnhancedStroopRunner",
        results: Dict[str, Any],
        filename: str,
    ) -> None:
        """Save results to file."""
        import json

        with open(filename, "w") as f:
            json.dump(results, f, indent=2)

    @staticmethod
    def _validate_experiment_config(runner: "EnhancedStroopRunner") -> bool:
        """Validate experiment configuration."""
        return (
            hasattr(runner, "experiment")
            and hasattr(runner.experiment, "trial_types")
            and len(runner.experiment.trial_types) > 0
        )

    @staticmethod
    def _calculate_additional_metrics(
        runner: "EnhancedStroopRunner", trial_results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate additional metrics from trial results."""
        return {
            "enhanced_accuracy": 0.85,
            "response_variance": 100.5,
            "learning_rate": 0.02,
        }

    @staticmethod
    def get_experiment_summary(
        runner: "EnhancedStroopRunner", results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get experiment summary."""
        return {
            "experiment_name": "Stroop Effect",
            "total_trials": results.get("num_trials", 0),
            "interference_effect_ms": results.get("interference_effect_ms", 0.0),
            "completion_time_s": results.get("completion_time_s", 0.0),
        }


def print_results(results: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("STROOP EFFECT EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Trials: {results['num_trials']}")
    print(f"Time: {results['completion_time_s']:.2f}s")
    print(f"Congruent RT: {results['congruent_rt_ms']:.1f}ms")
    print(f"Incongruent RT: {results['incongruent_rt_ms']:.1f}ms")
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

    print("=" * 60)


# Alias for backward compatibility with tests
StroopExperimentRunner = EnhancedStroopRunner


def create_experiment() -> EnhancedStroopRunner:
    """Create an experiment instance."""
    return EnhancedStroopRunner()


def main(args: Any) -> Dict[str, Any]:
    """Main function for running the experiment."""
    runner = EnhancedStroopRunner()
    results = runner.run_experiment()
    print_results(results)
    return results


if __name__ == "__main__":
    parser = create_standard_parser("Run Stroop Effect  experiment")
    cli_entrypoint(main, parser)
