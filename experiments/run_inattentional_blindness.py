"""
Inattentional Blindness Experiment Runner

This is the AGENT-EDITABLE file for the auto-improvement system.
Modify this file to experiment with different task parameters, display sizes,
and analysis approaches to maximize the detection accuracy metric.

Usage:
    uv run run_inattentional_blindness.py

Output:
    Prints summary with accuracy (primary metric to maximize)

Modification Guidelines:
    - You CAN modify: task parameters, display sizes, task demands, etc.
    - You CANNOT modify: prepare_inattentional_blindness.py (blindness configurations are fixed)
    - Goal: Maximize accuracy (correct detection of unexpected stimuli)
    - Time budget: 10 minutes max per run
"""

import time
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np

# Standardized APGI imports
from apgi_cli import cli_entrypoint, create_standard_parser

# APGI Integration
from apgi_integration import APGIIntegration, APGIParameters

# Import fixed configurations from prepare_inattentional_blindness.py
from .prepare_inattentional_blindness import (
    APGI_PARAMS,
    TASK_TYPES,
    TIME_BUDGET,
    UNEXPECTED_OBJECTS,
    IBTrial,
    InattentionalBlindnessGenerator,
    TrialType,
)
from .ultimate_apgi_template import (
    HierarchicalProcessor,
    PrecisionExpectationState,
    UltimateAPGIParameters,
)

# Type aliases for compatibility
InattentionalBlindnessTrial = IBTrial
StimulusType = TrialType  # Using TrialType as StimulusType
TaskType = str  # TaskType is a string in this implementation


# ---------------------------------------------------------------------------
# MODIFIABLE PARAMETERS - Edit these to experiment with task optimization
# ---------------------------------------------------------------------------

TIME_BUDGET = 600  # noqa: F811

# Task structure parameters
NUM_TRIALS_CONFIG = 40  # Can adjust: 20-80 trials typical
INTER_TRIAL_INTERVAL_MS = 1500  # Delay between trials (ms)

# Primary task parameters
PRIMARY_TASK_DIFFICULTY = "medium"  # easy, medium, hard
PRIMARY_TASK_LOAD = 0.7  # Cognitive load of primary task (0-1)

# Unexpected stimulus parameters
UNEXPECTED_STIMULUS_DURATION = 1000  # ms
UNEXPECTED_STIMULUS_SIZE = "medium"  # small, medium, large
UNEXPECTED_STIMULUS_CONTRAST = 0.8  # Visual contrast

# Attention parameters
ATTENTIONAL_FOCUS = 0.6  # How focused on primary task
PERIPHERAL_VISION_QUALITY = 0.7  # Quality of peripheral vision

# Detection parameters
BASE_DETECTION_RATE = 0.5  # Base rate for detecting unexpected stimuli
INATTENTION_EFFECT_STRENGTH = 0.4  # How much inattention affects detection

# Response parameters
RESPONSE_DEADLINE_MS = 2000  # Time to respond to unexpected stimulus
RESPONSE_TIME_VARIABILITY = 300  # ms

# ---------------------------------------------------------------------------
# Simulated Attention System
# ---------------------------------------------------------------------------


class SimulatedAttentionSystem:
    """
    Simulates human-like attention in inattentional blindness tasks.

    This model uses a simple attention approach with:
    - Resource allocation between primary and secondary tasks
    - Inattention effects on stimulus detection
    - Realistic response times and accuracy
    """

    def __init__(self, enable_apgi: bool = True):
        self.reset()

    def reset(self) -> None:
        """Reset attention system state for new experiment."""
        self.trial_count = 0
        self.total_detected = 0
        self.response_times: List[float] = []

    def calculate_detection_probability(
        self, trial: InattentionalBlindnessTrial
    ) -> float:
        """Calculate probability of detecting the unexpected stimulus."""
        if trial.unexpected_object is None:
            return 0.0  # No stimulus to detect

        # Base detection rate
        detection_prob = BASE_DETECTION_RATE

        # Apply inattention effect based on primary task load
        load_effect = PRIMARY_TASK_LOAD * INATTENTION_EFFECT_STRENGTH
        detection_prob -= load_effect

        # Apply attentional focus effect
        focus_effect = ATTENTIONAL_FOCUS * 0.3
        detection_prob += focus_effect

        # Apply peripheral vision quality
        vision_effect = PERIPHERAL_VISION_QUALITY * 0.2
        detection_prob += vision_effect

        # Apply stimulus-specific effects
        if hasattr(trial, "stimulus_size"):
            if trial.stimulus_size == "large":
                detection_prob += 0.2
            elif trial.stimulus_size == "small":
                detection_prob -= 0.1

        if hasattr(trial, "stimulus_contrast"):
            detection_prob *= trial.stimulus_contrast

        # Apply task difficulty effect
        if PRIMARY_TASK_DIFFICULTY == "hard":
            detection_prob -= 0.15
        elif PRIMARY_TASK_DIFFICULTY == "easy":
            detection_prob += 0.1

        # Add noise
        detection_prob += np.random.normal(0, 0.1)

        # Clamp between 0 and 1
        return max(0.0, min(1.0, detection_prob))

    def calculate_response_time(self, detected: bool) -> float:
        """Calculate response time for detected stimulus."""
        if not detected:
            return 0.0

        base_rt: float = 800.0  # Base response time in ms

        # Add load effect
        if PRIMARY_TASK_LOAD > 0.5:
            base_rt *= 1.3  # Slower when primary task is demanding

        # Add variability
        rt = base_rt + np.random.normal(0, RESPONSE_TIME_VARIABILITY)

        return max(300, rt)  # Minimum 300ms

    def process_trial(self, trial: InattentionalBlindnessTrial) -> Dict:
        """Process a single inattentional blindness trial."""
        # Calculate detection probability
        detection_prob = self.calculate_detection_probability(trial)

        # Determine if detected
        detected = np.random.random() < detection_prob

        # Calculate response time
        rt = self.calculate_response_time(detected)

        # Update statistics
        self.trial_count += 1
        if detected:
            self.total_detected += 1
            self.response_times.append(rt)

        return {
            "detected": detected,
            "response_time_ms": rt,
            "detection_probability": detection_prob,
        }


# ---------------------------------------------------------------------------
# Enhanced Inattentional Blindness Runner
# ---------------------------------------------------------------------------


class EnhancedInattentionalBlindnessRunner:
    """
    Runs the inattentional blindness experiment with modifiable parameters.

    This is the main experiment orchestrator that coordinates:
    - Inattentional blindness setup and management
    - Simulated attention performance
    - Data collection and analysis
    - Metrics calculation
    """

    def __init__(self, enable_apgi: bool = True):
        self.generator = InattentionalBlindnessGenerator()
        self.attention_system = SimulatedAttentionSystem()
        self.start_time: Optional[float] = None

        # Initialize APGI components to None (will be set if enabled)
        self.apgi: Optional[APGIIntegration] = None
        self.hierarchical: Optional[HierarchicalProcessor] = None
        self.precision_gap: Optional[PrecisionExpectationState] = None
        self.neuromodulators: Optional[Dict[str, float]] = None
        self.running_stats: Optional[Dict[str, float]] = None
        self.trials: list[InattentionalBlindnessTrial] = []

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
                    beta_cross=float(
                        cast(Union[str, int, float], APGI_PARAMS.get("beta_cross", 0.2))
                    ),
                    tau_levels=cast(
                        Optional[List[float]],
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
                "ACh": float(
                    cast(Union[str, int, float], APGI_PARAMS.get("ACh", 1.0) or 1.0)
                ),
                "NE": float(
                    cast(Union[str, int, float], APGI_PARAMS.get("NE", 1.0) or 1.0)
                ),
                "DA": float(
                    cast(Union[str, int, float], APGI_PARAMS.get("DA", 1.0) or 1.0)
                ),
                "HT5": float(
                    cast(Union[str, int, float], APGI_PARAMS.get("HT5", 1.0) or 1.0)
                ),
            }

            # APGI component: Running statistics for z-score normalization
            self.running_stats = {
                "outcome_mean": 0.5,
                "outcome_var": 0.25,
                "rt_mean": 800.0,
                "rt_var": 40000.0,
            }
        else:
            pass

    def run_experiment(self) -> Dict:
        """
        Execute the full inattentional blindness experiment.

        Returns:
            Dictionary with all experiment results and metrics
        """
        self.start_time = time.time()
        self.generator.reset()
        self.attention_system.reset()
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

    def _run_single_trial(self, trial_num: int) -> InattentionalBlindnessTrial:
        """Execute a single trial."""
        # Create trial
        trial = self.generator.create_trial(trial_num + 1)

        # Process with attention system
        results = self.attention_system.process_trial(trial)

        # Update trial with results
        trial.detected = results["detected"]
        trial.response_time_ms = results["response_time_ms"]
        trial.timestamp = time.time()

        # Simulate inter-trial interval
        if INTER_TRIAL_INTERVAL_MS > 0:
            time.sleep(INTER_TRIAL_INTERVAL_MS / 1000.0)

        return trial

    def _calculate_results(self, completion_time: float) -> Dict:
        """Calculate final experiment results and metrics."""

        # Calculate detection accuracy (primary metric)
        unexpected_trials = [t for t in self.trials if t.unexpected_object is not None]
        detected_count = sum(1 for t in unexpected_trials if t.detected)
        detection_accuracy = (
            detected_count / len(unexpected_trials) if unexpected_trials else 0.0
        )

        # Calculate false alarm rate (for trials without unexpected stimulus)
        no_stimulus_trials = [t for t in self.trials if t.unexpected_object is None]
        false_alarms = sum(
            1 for t in no_stimulus_trials if getattr(t, "detected", False)
        )
        false_alarm_rate = (
            false_alarms / len(no_stimulus_trials) if no_stimulus_trials else 0.0
        )

        # Overall accuracy
        total_correct = detected_count + (len(no_stimulus_trials) - false_alarms)
        total_trials_with_response = len(unexpected_trials) + len(no_stimulus_trials)
        overall_accuracy = (
            total_correct / total_trials_with_response
            if total_trials_with_response > 0
            else 0.0
        )

        # Response time statistics
        detected_rts = [
            t.response_time_ms
            for t in unexpected_trials
            if t.detected and t.response_time_ms > 0
        ]
        mean_response_time = np.mean(detected_rts) if detected_rts else 0.0

        # Separate by unexpected object type (stimulus)
        stimulus_type_results = {}
        for obj_type in UNEXPECTED_OBJECTS:
            type_trials = [
                t for t in unexpected_trials if t.unexpected_object == obj_type
            ]
            if type_trials:
                type_detected = sum(1 for t in type_trials if t.detected)
                type_accuracy = type_detected / len(type_trials)

                stimulus_type_results[obj_type] = {
                    "accuracy": type_accuracy,
                    "num_trials": len(type_trials),
                }

        # Separate by task type
        task_type_results = {}
        for task_type in TASK_TYPES:
            type_trials = [t for t in self.trials if t.task_type == task_type]
            if type_trials:
                type_unexpected = [
                    t for t in type_trials if t.unexpected_object is not None
                ]
                type_detected = sum(1 for t in type_unexpected if t.detected)
                type_accuracy = (
                    type_detected / len(type_unexpected) if type_unexpected else 0.0
                )

                task_type_results[task_type] = {
                    "accuracy": type_accuracy,
                    "num_trials": len(type_trials),
                }

        # Compile results
        results = {
            # Primary output metric
            "accuracy": overall_accuracy,
            # Timing metrics
            "completion_time_s": completion_time,
            "time_min": completion_time / 60.0,
            # Task metrics
            "num_trials": len(self.trials),
            "unexpected_trials": len(unexpected_trials),
            "detected_count": detected_count,
            "detection_accuracy": detection_accuracy,
            "false_alarm_rate": false_alarm_rate,
            "mean_response_time_ms": mean_response_time,
            # Stimulus type performance
            "stimulus_type_performance": stimulus_type_results,
            # Task type performance
            "task_type_performance": task_type_results,
            # Configuration used
            "config": {
                "num_trials": NUM_TRIALS_CONFIG,
                "inter_trial_interval_ms": INTER_TRIAL_INTERVAL_MS,
                "primary_task_difficulty": PRIMARY_TASK_DIFFICULTY,
                "primary_task_load": PRIMARY_TASK_LOAD,
            },
        }

        return results


# ---------------------------------------------------------------------------
# Main Experiment Execution
# ---------------------------------------------------------------------------


def print_results(results: Dict) -> None:
    """Print formatted experiment results."""
    print("\n" + "=" * 60)
    print("Inattentional Blindness Experiment Results")
    print("=" * 60)

    # Primary metric
    print(f"\nPRIMARY METRIC (accuracy): {results['accuracy']:.3f}")

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

    print("  (correct detection - false alarm rate)")

    # Key metrics
    print("\nKey Metrics:")
    print(f"  completion_time_s: {results['completion_time_s']:.1f}")
    print(f"  num_trials:        {results['num_trials']}")
    print(f"  unexpected_trials: {results['unexpected_trials']}")
    print(f"  detected_count:    {results['detected_count']}")
    print(f"  detection_accuracy: {results['detection_accuracy']:.3f}")
    print(f"  false_alarm_rate:  {results['false_alarm_rate']:.3f}")
    print(f"  mean_response_time_ms: {results['mean_response_time_ms']:.1f}")

    # Stimulus type breakdown
    print("\nDetection by Stimulus Type:")
    for stim_type, perf in results["stimulus_type_performance"].items():
        print(f"  {stim_type}: {perf['accuracy']:.3f}")

    # Task type breakdown
    print("\nDetection by Task Type:")
    for task_type, perf in results["task_type_performance"].items():
        print(f"  {task_type}: {perf['accuracy']:.3f}")

    print("\n" + "=" * 60)


def run_standalone() -> Dict:
    """Standalone entry point for inattentional blindness experiment."""
    import gc

    gc.collect()

    # Run experiment
    runner = EnhancedInattentionalBlindnessRunner()
    results = runner.run_experiment()

    # Print results
    print_results(results)

    # Final summary output (for automated parsing)
    print("\n---")
    print(f"accuracy:          {results['accuracy']:.3f}")
    print(f"completion_time_s: {results['completion_time_s']:.1f}")
    print(f"num_trials:        {results['num_trials']}")
    print(f"unexpected_trials: {results['unexpected_trials']}")
    print(f"detected_count:    {results['detected_count']}")
    print(f"detection_accuracy: {results['detection_accuracy']:.3f}")

    # Memory tracking (simplified - using placeholder)
    peak_memory_mb = 0.0
    print(f"peak_vram_mb:      {peak_memory_mb:.1f}")

    # APGI Metrics Output (if enabled)
    if results.get("apgi_enabled"):
        print("\n" + "=" * 40)
        print("APGI METRICS")
        print("=" * 40)
        print(results.get("apgi_formatted", "No APGI metrics available"))
        print("=" * 40)

    return results


def main(args: Any) -> Dict:
    """Main function for running the experiment."""
    runner = EnhancedInattentionalBlindnessRunner()
    results = runner.run_experiment()
    return results


if __name__ == "__main__":
    parser = create_standard_parser("Run Inattentional Blindness  experiment")
    cli_entrypoint(main, parser)
