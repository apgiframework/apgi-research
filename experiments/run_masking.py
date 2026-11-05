"""
Masking Experiment Runner - APGI integration (internal feature checklist; not registered-protocol validation)

This is the AGENT-EDITABLE file for the auto-improvement system.
Modify this file to experiment with different task parameters, mask types,
and analysis approaches to maximize the masking effect metric.

Usage:
    uv run run_masking.py

Output:
    Prints summary with masking_effect_ms (primary metric to maximize)
    Full APGI metrics including Π vs Π̂ distinction, hierarchical processing,
    neuromodulator mapping, and domain-specific thresholds.

Modification Guidelines:
    - You CAN modify: task parameters, SOA values, mask durations, etc.
    - You CANNOT modify: prepare_masking.py (mask configurations are fixed)
    - Goal: Maximize masking_effect_ms (difference between masked and unmasked conditions)
    - Time budget: 10 minutes max per run
"""

import time
from typing import Any, Dict, List, Optional, cast

import numpy as np

# Standardized APGI imports
from apgi_cli import cli_entrypoint, create_standard_parser

# APGI Integration with hierarchical processing and precision gap
from apgi_integration import APGIIntegration, APGIParameters, format_apgi_output

# Import fixed configurations from prepare_masking.py
from .prepare_masking import (
    APGI_PARAMS,
    TIME_BUDGET,
    MaskingGenerator,
    MaskingTrial,
    MaskType,
    TrialType,
)
from .ultimate_apgi_template import (
    HierarchicalProcessor,
    PrecisionExpectationState,
    UltimateAPGIParameters,
)

# Local variables for reference
_TIME_BUDGET_VAL = APGI_PARAMS.get("time_budget", 600)

# ---------------------------------------------------------------------------
# MODIFIABLE PARAMETERS - Edit these to experiment with task optimization
# ---------------------------------------------------------------------------

# Task structure parameters
NUM_TRIALS_CONFIG = 100  # Can adjust: 50-200 trials typical
INTER_TRIAL_INTERVAL_MS = 1000  # Delay between trials (ms)

# SOA (Stimulus Onset Asynchrony) parameters
SOA_VALUES_TO_TEST = [10, 20, 30, 50, 80, 100, 150, 200]  # ms
SOA_DISTRIBUTION = "uniform"  # uniform, logarithmic, focused

# Mask intensity parameters
MASK_INTENSITY_LEVELS = [0.3, 0.5, 0.7, 0.9]  # Relative intensity
USE_VARIABLE_INTENSITY = False  # Whether to vary mask intensity

# Target detection parameters
BASE_DETECTION_RATE = 0.8  # Base detection without masking
MASKING_EFFECT_STRENGTH = 0.6  # How strongly masks affect detection

# Response time simulation (ms)
BASE_RESPONSE_TIME = 400
RESPONSE_TIME_VARIABILITY = 100
MASKING_RT_PENALTY = 150  # Additional RT when masked

# Attention parameters
ATTENTIONAL_FOCUS = 0.7  # How focused the participant is
DISTRACTION_PROBABILITY = 0.1  # Chance of distraction

# ---------------------------------------------------------------------------
# Simulated Participant Model
# ---------------------------------------------------------------------------


class SimulatedParticipant:
    """
    Simulates human-like visual perception in masking experiments.

    This model uses a simple perceptual processing approach with:
    - Base detection rates modified by masking
    - Realistic response times and attention effects
    - SOA-dependent masking functions
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset participant state for new experiment."""
        self.trial_count = 0
        self.total_correct = 0
        self.response_times: List[float] = []
        self.detection_history: List[bool] = []

    def calculate_detection_probability(self, trial: MaskingTrial) -> float:
        """Calculate probability of detecting the target."""
        if trial.trial_type == TrialType.TARGET_ABSENT:
            # False alarm rate for target absent trials
            return 0.2

        # Base detection rate
        detection_prob = BASE_DETECTION_RATE

        # Apply masking effect based on SOA
        if trial.mask_type == MaskType.BACKWARD:
            # Backward masking: stronger effect at shorter SOA
            soa_effect = np.exp(-trial.soa_ms / 50.0)
        elif trial.mask_type == MaskType.FORWARD:
            # Forward masking: different temporal profile
            soa_effect = np.exp(-trial.soa_ms / 80.0) * 0.7
        else:  # METACONTRAST
            # Metacontrast: peak effect at intermediate SOA
            optimal_soa = 30
            soa_effect = np.exp(-((trial.soa_ms - optimal_soa) ** 2) / (2 * 20**2))

        # Apply masking effect
        masking_effect = soa_effect * MASKING_EFFECT_STRENGTH
        detection_prob -= masking_effect

        # Apply mask intensity effect
        if USE_VARIABLE_INTENSITY and hasattr(trial, "mask_intensity"):
            intensity_factor = trial.mask_intensity
            detection_prob *= 1 - intensity_factor * 0.3

        # Apply attentional focus
        detection_prob *= ATTENTIONAL_FOCUS

        # Add some noise
        detection_prob += np.random.normal(0, 0.05)

        # Clamp between chance and perfect
        return float(max(0.1, min(0.95, detection_prob)))

    def calculate_response_time(self, trial: MaskingTrial, detected: bool) -> float:
        """Calculate response time for the trial."""
        base_rt = BASE_RESPONSE_TIME

        # Add masking RT penalty if target was detected but masked
        if detected and trial.trial_type == TrialType.TARGET_PRESENT:
            if trial.mask_type == MaskType.BACKWARD:
                base_rt += MASKING_RT_PENALTY * np.exp(-trial.soa_ms / 50.0)
            elif trial.mask_type == MaskType.FORWARD:
                base_rt += MASKING_RT_PENALTY * np.exp(-trial.soa_ms / 80.0) * 0.7
            else:  # METACONTRAST
                optimal_soa = 30
                base_rt += MASKING_RT_PENALTY * np.exp(
                    -((trial.soa_ms - optimal_soa) ** 2) / (2 * 20**2)
                )

        # Add variability
        rt = base_rt + np.random.normal(0, RESPONSE_TIME_VARIABILITY)

        # Add distraction effect
        if np.random.random() < DISTRACTION_PROBABILITY:
            rt += np.random.uniform(100, 300)

        return max(200, rt)  # Minimum 200ms

    def process_trial(self, trial: MaskingTrial) -> Dict:
        """Process a single masking trial."""
        # Calculate detection probability
        detection_prob = self.calculate_detection_probability(trial)

        # Determine if detected
        detected = np.random.random() < detection_prob

        # Calculate response time
        rt = self.calculate_response_time(trial, detected)

        # Determine response
        if trial.trial_type == TrialType.TARGET_PRESENT:
            response = trial.target if detected else "none"
            correct = detected
        else:  # TARGET_ABSENT
            response = "none" if not detected else trial.target
            correct = not detected

        # Update statistics
        self.trial_count += 1
        if correct:
            self.total_correct += 1
        self.response_times.append(rt)
        self.detection_history.append(detected)

        return {
            "detected": detected,
            "response": response,
            "correct": correct,
            "rt_ms": rt,
            "detection_probability": detection_prob,
        }


# ---------------------------------------------------------------------------
# Enhanced Masking Runner
# ---------------------------------------------------------------------------


class EnhancedMaskingRunner:
    """
    Runs the masking experiment with internal APGI feature checklist.

    This runner integrates:
    - Core APGI dynamical system (ignition, surprise, somatic markers)
    - Hierarchical 5-level processing
    - Π vs Π̂ precision expectation gap
    - Neuromodulator mapping (ACh, NE, DA, 5-HT)
    - Domain-specific thresholds for detection tasks
    """

    def __init__(self, enable_apgi: bool = True):
        self.generator = MaskingGenerator()
        self.participant = SimulatedParticipant()
        self.start_time: Optional[float] = None
        self.trials: List[MaskingTrial] = []

        # Initialize APGI components
        self.enable_apgi = enable_apgi and APGI_PARAMS.get("enabled", True)

        # Helper function to safely get float values from APGI_PARAMS
        def safe_float(key: str, default: float) -> float:
            value = APGI_PARAMS.get(key, default)
            if isinstance(value, (int, float)):
                return float(value)
            elif value is not None:
                return float(str(value))
            else:
                return default

        # Declare APGI component types
        self.apgi: Optional[APGIIntegration] = None
        self.hierarchical: Optional[HierarchicalProcessor] = None
        self.precision_gap: Optional[PrecisionExpectationState] = None
        self.neuromodulators: Optional[Dict[str, float]] = None
        self.running_stats: Optional[Dict[str, float]] = None

        if self.enable_apgi:
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
                    tau_levels=cast(
                        List[float],
                        APGI_PARAMS.get("tau_levels") or [0.1, 0.2, 0.4, 1.0, 5.0],
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
                "ACh": safe_float("ACh", 1.0),
                "NE": safe_float("NE", 1.0),
                "DA": safe_float("DA", 1.0),
                "HT5": safe_float("HT5", 1.0),
            }

            # APGI component: Running statistics for z-score normalization
            self.running_stats = {
                "outcome_mean": 0.5,
                "outcome_var": 0.25,
                "rt_mean": 500.0,
                "rt_var": 25000.0,
            }

    def run_experiment(self) -> Dict:
        """
        Execute the full masking experiment.

        Returns:
            Dictionary with all experiment results and metrics
        """
        self.start_time = time.time()
        self.generator.reset()
        self.participant.reset()
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

    def _run_single_trial(self, trial_num: int) -> MaskingTrial:
        """Execute a single trial with APGI tracking."""
        # Create trial
        trial = self.generator.create_trial(trial_num + 1)

        # Process with participant
        results = self.participant.process_trial(trial)

        # Update trial with results
        trial.response = results["response"]
        trial.correct = results["correct"]
        trial.rt_ms = results["rt_ms"]
        trial.timestamp = time.time()

        # APGI component: Process with APGI if enabled
        if self.apgi:
            # Compute observed and expected values
            observed = 1.0 if trial.correct else 0.0
            expected = results["detection_probability"]

            # Trial type based on masking condition
            trial_type = (
                "survival" if trial.mask_type == MaskType.BACKWARD else "neutral"
            )

            # APGI component: Determine precision based on mask type and neuromodulators
            # ACh increases visual attention precision
            ach_boost = (
                self.neuromodulators.get("ACh", 1.0) if self.neuromodulators else 1.0
            )
            # NE increases arousal for difficult trials
            ne_effect = (
                self.neuromodulators.get("NE", 1.0) if self.neuromodulators else 1.0
            )

            precision_ext = (
                2.0 if trial.mask_type == MaskType.BACKWARD else 1.5
            ) * ach_boost
            precision_int = float(
                (1.5 if trial.correct else 1.0) * (1.0 + 0.2 * ne_effect)
            )

            # APGI component: Update running statistics for z-score normalization
            if self.running_stats:
                alpha_mu = 0.01
                alpha_sigma = 0.005
                self.running_stats["outcome_mean"] += alpha_mu * (
                    observed - self.running_stats["outcome_mean"]
                )
                self.running_stats["outcome_var"] += alpha_sigma * (
                    (observed - self.running_stats["outcome_mean"]) ** 2
                    - self.running_stats["outcome_var"]
                )
                self.running_stats["outcome_var"] = max(
                    0.01, self.running_stats["outcome_var"]
                )

            # APGI component: Update precision expectation gap (Π vs Π̂)
            if self.precision_gap:
                self.precision_gap.update(
                    precision_ext, precision_int, self.neuromodulators or {}
                )
                precision_ext = self.precision_gap.Pi_e_actual
                precision_int = self.precision_gap.Pi_i_actual

            # APGI component: Process with APGI - computes ignition, surprise, somatic markers
            apgi_state = self.apgi.process_trial(
                observed=observed,
                predicted=expected,
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

        # Simulate inter-trial interval
        if INTER_TRIAL_INTERVAL_MS > 0:
            time.sleep(INTER_TRIAL_INTERVAL_MS / 1000.0)

        return trial

    def _calculate_results(self, completion_time: float) -> Dict:
        """Calculate final experiment results and metrics."""

        # Separate trials by mask type
        mask_type_results = {}

        for mask_type in MaskType:
            mask_trials = [t for t in self.trials if t.mask_type == mask_type]
            if mask_trials:
                correct_present = sum(
                    1
                    for t in mask_trials
                    if t.trial_type == TrialType.TARGET_PRESENT and t.correct
                )
                total_present = sum(
                    1 for t in mask_trials if t.trial_type == TrialType.TARGET_PRESENT
                )

                correct_absent = sum(
                    1
                    for t in mask_trials
                    if t.trial_type == TrialType.TARGET_ABSENT and t.correct
                )
                total_absent = sum(
                    1 for t in mask_trials if t.trial_type == TrialType.TARGET_ABSENT
                )

                accuracy = (
                    (correct_present + correct_absent) / (total_present + total_absent)
                    if (total_present + total_absent) > 0
                    else 0.0
                )

                mask_type_results[mask_type.value] = {
                    "accuracy": accuracy,
                    "hit_rate": (
                        correct_present / total_present if total_present > 0 else 0.0
                    ),
                    "correct_rejection": (
                        correct_absent / total_absent if total_absent > 0 else 0.0
                    ),
                }

        # Calculate SOA effects
        soa_groups = {}
        for soa in SOA_VALUES_TO_TEST:
            soa_trials = [t for t in self.trials if abs(t.soa_ms - soa) < 10]
            if soa_trials:
                correct = sum(1 for t in soa_trials if t.correct)
                accuracy = correct / len(soa_trials)
                soa_groups[soa] = accuracy

        # Calculate masking effect (primary metric)
        # Masking effect = performance difference between longest and shortest SOA
        if len(soa_groups) >= 2:
            soas_sorted = sorted(soa_groups.keys())
            long_soa_perf = soa_groups[soas_sorted[-1]]
            short_soa_perf = soa_groups[soas_sorted[0]]
            masking_effect_ms = (
                long_soa_perf - short_soa_perf
            ) * 1000  # Convert to ms scale
        else:
            masking_effect_ms = 0.0

        # Overall statistics
        total_correct = sum(1 for t in self.trials if t.correct)
        overall_accuracy = total_correct / len(self.trials) if self.trials else 0.0

        # Response time statistics
        rts = [t.rt_ms for t in self.trials]
        mean_rt = np.mean(rts) if rts else 0.0

        # Compile results
        results: Dict[str, object] = {
            # Primary output metric
            "masking_effect_ms": masking_effect_ms,
            # Timing metrics
            "completion_time_s": completion_time,
            "time_min": completion_time / 60.0,
            # Task metrics
            "num_trials": len(self.trials),
            "overall_accuracy": overall_accuracy,
            "total_correct": total_correct,
            "total_incorrect": len(self.trials) - total_correct,
            "mean_response_time_ms": mean_rt,
            # Mask type performance
            "mask_type_performance": mask_type_results,
            # SOA performance
            "soa_performance": soa_groups,
            # Configuration used
            "config": {
                "num_trials": NUM_TRIALS_CONFIG,
                "inter_trial_interval_ms": INTER_TRIAL_INTERVAL_MS,
                "soa_values": SOA_VALUES_TO_TEST,
                "mask_intensity_levels": MASK_INTENSITY_LEVELS,
                "time_budget": TIME_BUDGET,
            },
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

            results["apgi_summary_formatted"] = format_apgi_output(apgi_summary)
        else:
            results["apgi_enabled"] = False

        return results


# ---------------------------------------------------------------------------
# Main Experiment Execution
# ---------------------------------------------------------------------------


def print_results(results: Dict) -> None:
    """Print formatted experiment results."""
    print("\n" + "=" * 60)
    print("Masking Experiment Results")
    print("=" * 60)

    # Primary metric
    print(f"\nPRIMARY METRIC (masking_effect_ms): {results['masking_effect_ms']:.1f}")
    print("  (performance difference between longest and shortest SOA, in ms)")

    # Key metrics
    print("\nKey Metrics:")
    print(f"  completion_time_s: {results['completion_time_s']:.1f}")
    print(f"  num_trials:        {results['num_trials']}")
    print(f"  overall_accuracy:  {results['overall_accuracy']:.3f}")
    print(f"  mean_response_time_ms: {results['mean_response_time_ms']:.1f}")

    # Mask type breakdown
    print("\nPerformance by Mask Type:")
    for mask_type, perf in results["mask_type_performance"].items():
        print(f"  {mask_type}:")
        print(f"    accuracy: {perf['accuracy']:.3f}")
        print(f"    hit_rate: {perf['hit_rate']:.3f}")
        print(f"    correct_rejection: {perf['correct_rejection']:.3f}")

    # SOA breakdown
    print("\nPerformance by SOA:")
    for soa, accuracy in sorted(results["soa_performance"].items()):
        print(f"  {soa:3d} ms: {accuracy:.3f}")

    print("\n" + "=" * 60)


def run_standalone() -> Dict:
    """Standalone entry point for masking experiment."""
    import gc

    gc.collect()

    # Run experiment
    runner = EnhancedMaskingRunner()
    results = runner.run_experiment()

    # Print results
    print_results(results)

    # Final summary output (for automated parsing)
    print("\n---")
    print(f"masking_effect_ms: {results['masking_effect_ms']:.1f}")
    print(f"completion_time_s: {results['completion_time_s']:.1f}")
    print(f"num_trials:        {results['num_trials']}")
    print(f"overall_accuracy:  {results['overall_accuracy']:.3f}")
    print(f"total_correct:     {results['total_correct']}")
    print(f"mean_response_time_ms: {results['mean_response_time_ms']:.1f}")

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
    runner = EnhancedMaskingRunner()
    results = runner.run_experiment()
    return results


if __name__ == "__main__":
    parser = create_standard_parser("Run Masking  experiment")
    cli_entrypoint(main, parser)
