"""
Binocular Rivalry Experiment Runner

This is the AGENT-EDITABLE file for the auto-improvement system.
Modify this file to experiment with different task parameters, stimulus types,
and analysis approaches to maximize the rivalry alternation metric.

Usage:
    uv run run_binocular_rivalry.py

Output:
    Prints summary with alternation_rate (primary metric to maximize)

Modification Guidelines:
    - You CAN modify: task parameters, stimulus durations, analysis methods, etc.
    - You CANNOT modify: prepare_binocular_rivalry.py (rivalry configurations are fixed)
    - Goal: Maximize alternation_rate (perceptual switching frequency)
    - Time budget: 10 minutes max per run
"""

import time
from typing import Any, Dict, List, cast

import numpy as np

# Standardized APGI imports
from apgi_cli import cli_entrypoint, create_standard_parser

# APGI Integration
from apgi_integration import APGIIntegration, APGIParameters

# Import fixed configurations from prepare_binocular_rivalry.py
from .prepare_binocular_rivalry import (
    APGI_PARAMS,
    TIME_BUDGET,
    BinocularRivalryGenerator,
    StimulusType,
)
from .ultimate_apgi_template import (
    HierarchicalProcessor,
    PrecisionExpectationState,
    UltimateAPGIParameters,
)

# ---------------------------------------------------------------------------
# MODIFIABLE PARAMETERS - Edit these to experiment with task optimization
# ---------------------------------------------------------------------------

TIME_BUDGET = 600  # noqa: F811

# Task structure parameters
NUM_TRIALS_CONFIG = 60  # Can adjust: 30-120 trials typical
INTER_TRIAL_INTERVAL_MS = 2000  # Delay between trials (ms)

# Stimulus parameters
STIMULUS_INTENSITY = 0.8  # Contrast strength
CONTRAST_BALANCE = 0.5  # Balance between two eyes

# Rivalry dynamics parameters
BASE_ALTERNATION_RATE = 0.3  # Base alternations per second
DOMINANCE_DURATION_VARIABILITY = 2.0  # Variability in dominance durations

# Attention parameters
ATTENTIONAL_FOCUS = 0.7  # How focused the participant is
EYE_MOVEMENT_PROBABILITY = 0.1  # Chance of eye movements affecting rivalry

# Analysis parameters
MIN_PERCEPT_DURATION = 0.2  # Minimum duration to count as percept (seconds)
TRANSITION_DETECTION_THRESHOLD = 0.1  # Threshold for detecting transitions

# ---------------------------------------------------------------------------
# Simulated Perceptual System
# ---------------------------------------------------------------------------


class SimulatedPerceptualSystem:
    """
    Simulates human-like binocular rivalry perception.

    This model uses a simple rivalry dynamics approach with:
    - Stochastic alternation between percepts
    - Realistic dominance duration distributions
    - Attention and eye movement effects
    """

    def __init__(self, enable_apgi: bool = True):
        self.reset()

    def reset(self) -> None:
        """Reset perceptual system state for new experiment."""
        self.trial_count = 0
        self.total_alternations = 0
        self.dominance_durations: List[float] = []

    def generate_rivalry_dynamics(self, trial: Dict) -> List[Dict]:
        """Generate perceptual alternations for a trial."""
        percepts = []
        current_percept = "A"  # Start with percept A
        time_remaining = trial["duration_s"]

        while time_remaining > 0:
            # Generate dominance duration
            base_duration = np.random.exponential(1.0 / BASE_ALTERNATION_RATE)
            duration = max(
                MIN_PERCEPT_DURATION,
                base_duration
                * (1 + np.random.normal(0, DOMINANCE_DURATION_VARIABILITY)),
            )

            # Apply attentional focus effect
            if ATTENTIONAL_FOCUS < 0.5:
                duration *= 1.5  # Less focused = longer dominance periods

            # Apply eye movement effect
            if np.random.random() < EYE_MOVEMENT_PROBABILITY:
                duration *= 0.5  # Eye movement can trigger switch

            # Ensure we don't exceed trial duration
            duration = min(duration, time_remaining)

            # Add percept
            percepts.append(
                {
                    "percept": current_percept,
                    "duration": duration,
                    "start_time": trial["duration_s"] - time_remaining,
                    "end_time": trial["duration_s"] - time_remaining + duration,
                }
            )

            # Switch percept
            current_percept = "B" if current_percept == "A" else "A"
            time_remaining -= duration

        return percepts

    def calculate_alternation_metrics(self, percepts: List[Dict]) -> Dict:
        """Calculate alternation metrics from percept sequence."""
        if len(percepts) < 2:
            return {
                "alternation_count": 0,
                "alternation_rate": 0.0,
                "mean_duration_a": 0.0,
                "mean_duration_b": 0.0,
            }

        # Count alternations
        alternation_count = len(percepts) - 1

        # Calculate alternation rate (per second)
        total_duration = sum(p["duration"] for p in percepts)
        alternation_rate = (
            alternation_count / total_duration if total_duration > 0 else 0.0
        )

        # Calculate mean durations for each percept
        durations_a = [p["duration"] for p in percepts if p["percept"] == "A"]
        durations_b = [p["duration"] for p in percepts if p["percept"] == "B"]

        mean_duration_a = np.mean(durations_a) if durations_a else 0.0
        mean_duration_b = np.mean(durations_b) if durations_b else 0.0

        return {
            "alternation_count": alternation_count,
            "alternation_rate": alternation_rate,
            "mean_duration_a": mean_duration_a,
            "mean_duration_b": mean_duration_b,
        }

    def process_trial(self, trial: Dict) -> Dict:
        """Process a single rivalry trial."""
        # Generate rivalry dynamics
        percepts = self.generate_rivalry_dynamics(trial)

        # Calculate metrics
        metrics = self.calculate_alternation_metrics(percepts)

        # Update statistics
        self.trial_count += 1
        self.total_alternations += metrics["alternation_count"]
        self.dominance_durations.extend([p["duration"] for p in percepts])

        return {
            "percepts": percepts,
            "metrics": metrics,
        }


# ---------------------------------------------------------------------------
# Enhanced Binocular Rivalry Runner
# ---------------------------------------------------------------------------


class EnhancedBinocularRivalryRunner:
    """
    Runs the binocular rivalry experiment with modifiable parameters.

    This is the main experiment orchestrator that coordinates:
    - Rivalry setup and management
    - Simulated perceptual dynamics
    - Data collection and analysis
    - Metrics calculation
    """

    def __init__(self, enable_apgi: bool = True):
        self.generator = BinocularRivalryGenerator()
        self.perceptual_system = SimulatedPerceptualSystem()
        self.start_time: float | None = None

        # Initialize APGI-related variables with proper types
        self.apgi: APGIIntegration | None = None
        self.hierarchical: HierarchicalProcessor | None = None
        self.precision_gap: PrecisionExpectationState | None = None
        self.neuromodulators: dict[str, float] | None = None
        self.running_stats: dict[str, float] | None = None
        self.trials: list[Dict] = []

        # Initialize APGI components
        self.enable_apgi = enable_apgi and APGI_PARAMS.get("enabled", True)
        if self.enable_apgi:
            # Helper function to safely get float values
            def safe_float(key: str, default: float) -> float:
                value = APGI_PARAMS.get(key, default)
                if isinstance(value, (int, float)):
                    return float(value)
                elif value is not None:
                    return float(str(value))
                else:
                    return default

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
                    tau_levels=(
                        [0.1, 0.2, 0.4, 1.0, 5.0]
                        if APGI_PARAMS.get("tau_levels") is None
                        else list(cast(List[float], APGI_PARAMS.get("tau_levels", [])))
                    ),
                )
                self.hierarchical = HierarchicalProcessor(ultimate_params)
            else:
                self.hierarchical = None

            # APGI component: Precision expectation gap (Pi vs Pi_hat)
            if APGI_PARAMS.get("precision_gap_enabled", True):
                self.precision_gap = PrecisionExpectationState()

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
        """
        Execute the full binocular rivalry experiment.

        Returns:
            Dictionary with all experiment results and metrics
        """
        self.start_time = time.time()
        self.generator.reset()
        self.perceptual_system.reset()
        self.trials = []

        # Run all trials
        for trial_num in range(NUM_TRIALS_CONFIG):
            trial = self._run_single_trial(trial_num)
            self.trials.append(trial)

            # Check time budget
            elapsed = time.time() - self.start_time
            if elapsed > TIME_BUDGET:
                print(f"WARNING: Time budget exceeded at trial {trial_num}")
                break

        # Calculate final metrics
        completion_time = time.time() - self.start_time
        results = self._calculate_results(completion_time)

        return results

    def _run_single_trial(self, trial_num: int) -> Dict:
        """Execute a single trial."""
        # Create trial
        trial = self.generator.create_trial(trial_num + 1)

        # Process with perceptual system
        results = self.perceptual_system.process_trial(trial)

        # Update trial with results
        trial["percepts"] = results["percepts"]
        trial["alternation_count"] = results["metrics"]["alternation_count"]
        trial["mean_duration_a"] = results["metrics"]["mean_duration_a"]
        trial["mean_duration_b"] = results["metrics"]["mean_duration_b"]
        trial["timestamp"] = time.time()

        # Simulate inter-trial interval
        if INTER_TRIAL_INTERVAL_MS > 0:
            time.sleep(INTER_TRIAL_INTERVAL_MS / 1000.0)

        return trial

    def _calculate_results(self, completion_time: float) -> Dict:
        """Calculate final experiment results and metrics."""

        # Calculate overall alternation rate (primary metric)
        total_alternations = sum(t["alternation_count"] for t in self.trials)
        total_duration = sum(t["duration_s"] for t in self.trials)
        overall_alternation_rate = (
            total_alternations / total_duration if total_duration > 0 else 0.0
        )

        # Separate by stimulus type
        stimulus_type_results = {}
        for stim_type in StimulusType:
            stim_trials = [t for t in self.trials if t["stimulus_type"] == stim_type]
            if stim_trials:
                stim_alternations = sum(t["alternation_count"] for t in stim_trials)
                stim_duration = sum(t["duration_s"] for t in stim_trials)
                stim_rate = (
                    stim_alternations / stim_duration if stim_duration > 0 else 0.0
                )

                mean_duration_a = np.mean([t["mean_duration_a"] for t in stim_trials])
                mean_duration_b = np.mean([t["mean_duration_b"] for t in stim_trials])

                stimulus_type_results[stim_type.value] = {
                    "alternation_rate": stim_rate,
                    "mean_duration_a": mean_duration_a,
                    "mean_duration_b": mean_duration_b,
                    "num_trials": len(stim_trials),
                }

        # Dominance duration statistics
        all_durations = self.perceptual_system.dominance_durations
        mean_dominance_duration = np.mean(all_durations) if all_durations else 0.0
        dominance_duration_cv = (
            np.std(all_durations) / mean_dominance_duration
            if mean_dominance_duration > 0
            else 0.0
        )

        # Compile results
        results = {
            # Primary output metric
            "alternation_rate": overall_alternation_rate,
            # Timing metrics
            "completion_time_s": completion_time,
            "time_min": completion_time / 60.0,
            # Task metrics
            "num_trials": len(self.trials),
            "total_alternations": total_alternations,
            "total_duration_s": total_duration,
            "mean_dominance_duration_s": mean_dominance_duration,
            "dominance_duration_cv": dominance_duration_cv,
            # Stimulus type performance
            "stimulus_type_performance": stimulus_type_results,
            # Configuration used
            "config": {
                "num_trials": NUM_TRIALS_CONFIG,
                "inter_trial_interval_ms": INTER_TRIAL_INTERVAL_MS,
                "stimulus_intensity": STIMULUS_INTENSITY,
                "base_alternation_rate": BASE_ALTERNATION_RATE,
            },
        }

        return results


# ---------------------------------------------------------------------------
# Main Experiment Execution
# ---------------------------------------------------------------------------


def print_results(results: Dict) -> None:
    """Print formatted experiment results."""
    print("\n" + "=" * 60)
    print("Binocular Rivalry Experiment Results")
    print("=" * 60)

    # Primary metric
    print(f"\nPRIMARY METRIC (alternation_rate): {results['alternation_rate']:.3f}")

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

    print("  (perceptual alternations per second)")

    # Key metrics
    print("\nKey Metrics:")
    print(f"  completion_time_s: {results['completion_time_s']:.1f}")
    print(f"  num_trials:        {results['num_trials']}")
    print(f"  total_alternations: {results['total_alternations']}")
    print(f"  mean_dominance_duration_s: {results['mean_dominance_duration_s']:.2f}")
    print(f"  dominance_duration_cv: {results['dominance_duration_cv']:.3f}")

    # Stimulus type breakdown
    print("\nPerformance by Stimulus Type:")
    for stim_type, perf in results["stimulus_type_performance"].items():
        print(f"  {stim_type}:")
        print(f"    alternation_rate: {perf['alternation_rate']:.3f}")
        print(f"    mean_duration_a: {perf['mean_duration_a']:.2f}s")
        print(f"    mean_duration_b: {perf['mean_duration_b']:.2f}s")

    print("\n" + "=" * 60)


def main(args: Any) -> Dict:
    """Main function for running the experiment."""
    runner = EnhancedBinocularRivalryRunner()
    results = runner.run_experiment()
    return results


if __name__ == "__main__":
    parser = create_standard_parser("Run Binocular Rivalry  experiment")
    cli_entrypoint(main, parser)
