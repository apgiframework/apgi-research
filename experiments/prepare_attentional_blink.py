"""
Fixed constants and data preparation for Attentional Blink experiments.

This file is READ-ONLY. Do not modify.
It defines the fixed stimulus timing, RSVP parameters, and evaluation metrics.

Usage:
    python prepare_attentional_blink.py  # Verify stimulus configurations

The Attentional Blink paradigm uses Rapid Serial Visual Presentation (RSVP) where:
- T1: First target stimulus (e.g., white letter among black)
- T2: Second target stimulus appearing 100-800ms after T1
- Lag: SOA between T1 and T2 (typically 2-8 items at 10 items/sec)
- The "blink": Impaired T2 detection at lags 2-3 (200-300ms)
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from apgi_cli import cli_entrypoint, create_standard_parser

# ---------------------------------------------------------------------------
# Fixed Constants (DO NOT MODIFY)
# ---------------------------------------------------------------------------

TIME_BUDGET = 600  # 10 minutes per experiment (in seconds)
NUM_TRIALS = 100  # Standard trial count

# APGI Integration Parameters
# Optimized for attentional blink dynamics
APGI_ENABLED = True

# Core dynamical system parameters
APGI_TAU_S = 0.35  # Surprise decay (200-500ms)
APGI_TAU_THETA = 30.0  # Threshold adaptation (5-60s)
APGI_TAU_M = 1.5  # Somatic marker timescale (1-2s)

# Threshold and sigmoid parameters
APGI_THETA_0 = 0.5  # Baseline ignition threshold (0.1-1.0 AU)
APGI_ALPHA = 5.5  # Sigmoid steepness (3.0-8.0)

# Somatic modulation
APGI_BETA = 1.5  # β_som: Somatic influence (0.5-2.5)
APGI_BETA_M = 1.0  # Marker sensitivity (0.5-2.0)
APGI_M_0 = 0.0  # Reference somatic marker level

# Sensitivities
APGI_GAMMA_M = -0.3  # Metabolic sensitivity (-0.5 to 0.5)
APGI_GAMMA_A = 0.1  # Arousal sensitivity (-0.3 to 0.3)
APGI_LAMBDA_S = 0.1  # Metabolic coupling strength

# Noise strengths
APGI_SIGMA_S = 0.05  # Surprise noise
APGI_SIGMA_THETA = 0.02  # Threshold noise
APGI_SIGMA_M = 0.03  # Somatic marker noise

# Domain-specific thresholds
APGI_THETA_SURVIVAL = 0.3  # Lower threshold for survival-relevant
APGI_THETA_NEUTRAL = 0.7  # Higher threshold for neutral content

# Reset dynamics
APGI_RHO = 0.7  # Reset fraction after ignition (0.3-0.9)

# Hierarchical processing (5-level)
APGI_HIERARCHICAL_ENABLED = True
APGI_BETA_CROSS = 0.2  # Cross-level coupling (0.1-0.5)
APGI_TAU_LEVELS = [0.1, 0.2, 0.4, 1.0, 5.0]  # Level-specific timescales

# Neuromodulator baselines
APGI_ACHT = 1.0  # Acetylcholine
APGI_NE = 1.0  # Norepinephrine
APGI_DA = 1.0  # Dopamine
APGI_HT5 = 1.0  # Serotonin

# Precision expectation gap (Π vs Π̂)
APGI_ENABLE_PRECISION_GAP = True
APGI_PI_E_EXPECTED = 1.0
APGI_PI_I_EXPECTED = 1.0

# Psychiatric profiles
APGI_GAD_PROFILE = False
APGI_MDD_PROFILE = False
APGI_PSYCHOSIS_PROFILE = False  # Sharp ignition for blink detection

# RSVP Stream Parameters
RSVP_RATE_HZ = 10  # Items per second (100ms per item)
ITEM_DURATION_MS = 100  # Duration of each RSVP item
ISI_MS = 0  # Inter-stimulus interval (continuous RSVP)

# Target Configuration
T1_POSITION_MIN = 5  # T1 appears after at least 5 distractors
T1_POSITION_MAX = 12  # T1 appears before at most 12 distractors

# Standard Lags (in items) - SOA = lag * 100ms
STANDARD_LAGS = [1, 2, 3, 4, 5, 6, 7, 8]  # 100-800ms
BLINK_LAGS = [2, 3]  # The classic attentional blink window

# Stimulus Sets
DISTRACTORS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")  # Exclude I and O (confusable)
T1_STIMULI = ["X", "Z"]  # White letters (targets)
T2_STIMULI = ["2", "3", "4", "5", "6", "7", "8", "9"]  # Digits (targets)


class TrialType(Enum):
    """Types of attentional blink trials."""

    BOTH_TARGETS = "both"  # Both T1 and T2 present
    T1_ONLY = "t1_only"  # Only T1 (catch trial)
    T2_ONLY = "t2_only"  # Only T2 (baseline)
    NEITHER = "neither"  # Neither (catch trial)


@dataclass
class ABConfig:
    """Attentional Blink trial configuration."""

    trial_number: int
    trial_type: TrialType
    t1_position: int  # Position in RSVP stream
    lag: int  # Items between T1 and T2
    t1_stimulus: str
    t2_stimulus: str
    stream_items: List[str] = field(default_factory=list)


@dataclass
class ABTrial:
    """Single Attentional Blink trial data."""

    trial_number: int
    config: ABConfig
    t1_response: Optional[str] = None
    t1_correct: bool = False
    t1_rt_ms: float = 0.0
    t2_response: Optional[str] = None
    t2_correct: bool = False
    t2_rt_ms: float = 0.0
    lag: int = 0
    saw_t2: bool = True  # Participant reports seeing T2
    timestamp: float = 0.0


class ABStream:
    """Generates RSVP streams for Attentional Blink experiments."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self) -> None:
        """Reset stream generator."""
        self.trial_count = 0

    def generate_stream(
        self,
        t1_pos: int,
        lag: int,
        t1_stim: str,
        t2_stim: str,
        trial_type: TrialType = TrialType.BOTH_TARGETS,
        stream_length: int = 20,
    ) -> List[str]:
        """
        Generate RSVP stream with embedded targets.

        Args:
            t1_pos: Position of T1 (1-indexed)
            lag: Lag between T1 and T2 (items)
            t1_stim: T1 stimulus
            t2_stim: T2 stimulus
            trial_type: Type of trial
            stream_length: Total items in stream

        Returns:
            List of stimuli in RSVP order
        """
        stream = []
        t2_pos = t1_pos + lag

        for i in range(1, stream_length + 1):
            if (
                trial_type in [TrialType.BOTH_TARGETS, TrialType.T1_ONLY]
                and i == t1_pos
            ):
                stream.append(t1_stim)
            elif (
                trial_type in [TrialType.BOTH_TARGETS, TrialType.T2_ONLY]
                and i == t2_pos
            ):
                stream.append(t2_stim)
            else:
                # Distractor (different from T2 to avoid confusion)
                distractor = self.rng.choice(DISTRACTORS)
                while distractor in T2_STIMULI:
                    distractor = self.rng.choice(DISTRACTORS)
                stream.append(distractor)

        return stream

    def create_trial(self, trial_number: int) -> ABConfig:
        """Create a random trial configuration."""
        trial_type_list = [tt.value for tt in TrialType]
        trial_type_str = self.rng.choice(trial_type_list)
        trial_type = TrialType(trial_type_str)
        t1_pos = self.rng.randint(T1_POSITION_MIN, T1_POSITION_MAX + 1)
        lag = self.rng.choice(STANDARD_LAGS)
        t1_stim = self.rng.choice(T1_STIMULI)
        t2_stim = self.rng.choice(T2_STIMULI)

        stream = self.generate_stream(
            t1_pos=t1_pos,
            lag=lag,
            t1_stim=t1_stim,
            t2_stim=t2_stim,
            trial_type=trial_type,
        )

        return ABConfig(
            trial_number=trial_number,
            trial_type=trial_type,
            t1_position=t1_pos,
            lag=lag,
            t1_stimulus=t1_stim,
            t2_stimulus=t2_stim,
            stream_items=stream,
        )


class ABExperiment:
    """Manages a complete Attentional Blink experiment session."""

    def __init__(self, num_trials: int = NUM_TRIALS, seed: Optional[int] = None):
        self.num_trials = num_trials
        self.stream_gen = ABStream(seed=seed)
        self.trials: List[ABTrial] = []
        self.current_trial_idx = 0
        self.reset()

    def reset(self) -> None:
        """Reset experiment state."""
        self.trials = []
        self.current_trial_idx = 0
        self.stream_gen.reset()

    def get_next_trial(self) -> Optional[ABConfig]:
        """Get the next trial configuration."""
        if self.current_trial_idx >= self.num_trials:
            return None

        config = self.stream_gen.create_trial(self.current_trial_idx + 1)
        return config

    def run_trial(
        self,
        config: ABConfig,
        t1_response: Optional[str],
        t1_rt_ms: float,
        t2_response: Optional[str],
        t2_rt_ms: float,
        saw_t2: bool = True,
    ) -> ABTrial:
        """
        Execute a single trial.

        Args:
            config: Trial configuration
            t1_response: Participant's T1 response
            t1_rt_ms: T1 reaction time
            t2_response: Participant's T2 response
            t2_rt_ms: T2 reaction time
            saw_t2: Whether participant reported seeing T2

        Returns:
            ABTrial with outcome data
        """
        t1_correct = t1_response == config.t1_stimulus if t1_response else False
        t2_correct = t2_response == config.t2_stimulus if t2_response else False

        import time

        trial = ABTrial(
            trial_number=config.trial_number,
            config=config,
            t1_response=t1_response,
            t1_correct=t1_correct,
            t1_rt_ms=t1_rt_ms,
            t2_response=t2_response,
            t2_correct=t2_correct,
            t2_rt_ms=t2_rt_ms,
            lag=config.lag,
            saw_t2=saw_t2,
            timestamp=time.time(),
        )
        self.trials.append(trial)
        self.current_trial_idx += 1
        return trial

    def get_t2_accuracy_by_lag(self) -> Dict[int, float]:
        """
        Calculate T2 accuracy for each lag (T1-correct trials only).

        This is the classic attentional blink curve.
        """
        lag_results: Dict[int, List[bool]] = {lag: [] for lag in STANDARD_LAGS}

        for trial in self.trials:
            # Only include trials where T1 was correct
            if trial.t1_correct and trial.config.trial_type == TrialType.BOTH_TARGETS:
                lag_results[trial.lag].append(trial.t2_correct)

        return {
            lag: float(np.mean(accuracies)) if accuracies else 0.0
            for lag, accuracies in lag_results.items()
        }

    def get_blink_magnitude(self) -> float:
        """
        Calculate attentional blink magnitude.

        Returns:
            Difference between lag 1 and lag 2/3 average T2 accuracy.
            Higher values indicate stronger attentional blink.
        """
        by_lag = self.get_t2_accuracy_by_lag()

        if 1 not in by_lag or not all(lag in by_lag for lag in BLINK_LAGS):
            return 0.0

        lag1_acc = by_lag[1]
        blink_acc = float(np.mean([by_lag[lag] for lag in BLINK_LAGS]))

        return lag1_acc - blink_acc  # Positive = blink present

    def get_t1_accuracy(self) -> float:
        """Calculate overall T1 accuracy."""
        t1_trials = [
            t
            for t in self.trials
            if t.config.trial_type in [TrialType.BOTH_TARGETS, TrialType.T1_ONLY]
        ]
        if not t1_trials:
            return 0.0
        return float(np.mean([t.t1_correct for t in t1_trials]))

    def get_summary(self) -> Dict:
        """Generate experiment summary statistics."""
        if not self.trials:
            return {}

        by_lag = self.get_t2_accuracy_by_lag()

        return {
            "num_trials": len(self.trials),
            "t1_accuracy": self.get_t1_accuracy(),
            "t2_accuracy_by_lag": by_lag,
            "blink_magnitude": self.get_blink_magnitude(),
            "mean_t1_rt_ms": float(
                np.mean([t.t1_rt_ms for t in self.trials if t.t1_rt_ms > 0])
            ),
            "mean_t2_rt_ms": float(
                np.mean([t.t2_rt_ms for t in self.trials if t.t2_rt_ms > 0])
            ),
            "lag_distribution": {
                lag: sum(1 for t in self.trials if t.lag == lag)
                for lag in STANDARD_LAGS
            },
        }

    def save_results(self, filepath: str) -> None:
        """Save trial data to JSON file."""
        data = {
            "trials": [
                {
                    "trial_number": t.trial_number,
                    "trial_type": t.config.trial_type.value,
                    "lag": t.lag,
                    "t1_stimulus": t.config.t1_stimulus,
                    "t2_stimulus": t.config.t2_stimulus,
                    "t1_response": t.t1_response,
                    "t1_correct": t.t1_correct,
                    "t1_rt_ms": t.t1_rt_ms,
                    "t2_response": t.t2_response,
                    "t2_correct": t.t2_correct,
                    "t2_rt_ms": t.t2_rt_ms,
                    "saw_t2": t.saw_t2,
                    "timestamp": t.timestamp,
                }
                for t in self.trials
            ],
            "summary": self.get_summary(),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


def verify_configurations() -> None:
    """Verify and print stimulus configurations."""
    print("=" * 60)
    print("Attentional Blink - Configuration Verification")
    print("=" * 60)

    print(f"\nRSVP Rate: {RSVP_RATE_HZ} Hz ({ITEM_DURATION_MS}ms per item)")
    print(f"Standard Lags (SOA): {[f'{lag} ({lag * 100}ms)' for lag in STANDARD_LAGS]}")
    print(
        f"Blink Window (Lags {BLINK_LAGS}): {BLINK_LAGS[0] * 100}-{BLINK_LAGS[-1] * 100}ms"
    )

    print(f"\nDistractors: {DISTRACTORS}")
    print(f"T1 Stimuli (letters): {T1_STIMULI}")
    print(f"T2 Stimuli (digits): {T2_STIMULI}")

    # Generate sample stream
    stream_gen = ABStream(seed=42)
    sample_config = stream_gen.create_trial(1)

    print("\nSample Trial:")
    print(f"  T1 Position: {sample_config.t1_position}")
    print(f"  Lag: {sample_config.lag} ({sample_config.lag * 100}ms)")
    print(f"  T1 Stimulus: {sample_config.t1_stimulus}")
    print(f"  T2 Stimulus: {sample_config.t2_stimulus}")
    print(f"  Stream: {' '.join(sample_config.stream_items)}")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args=None) -> int:
    """Verify configurations and run a quick sample simulation."""
    verify_configurations()

    # Run a quick simulation
    print("\n\nRunning sample simulation...")
    exp = ABExperiment(num_trials=50, seed=42)

    for _ in range(50):
        config = exp.get_next_trial()
        if config is None:
            break

        # Simulate responses (70% T1 accuracy, blink at lags 2-3)
        t1_correct = np.random.random() < 0.7
        t1_response = config.t1_stimulus if t1_correct else np.random.choice(T1_STIMULI)
        t1_rt = np.random.uniform(400, 800)

        # T2 accuracy depends on lag (blink effect)
        if config.lag in BLINK_LAGS:
            t2_prob = 0.4  # Impaired during blink
        else:
            t2_prob = 0.7  # Normal accuracy

        t2_correct = np.random.random() < t2_prob
        t2_response = config.t2_stimulus if t2_correct else np.random.choice(T2_STIMULI)
        t2_rt = np.random.uniform(400, 800)

        exp.run_trial(
            config=config,
            t1_response=t1_response,
            t1_rt_ms=t1_rt,
            t2_response=t2_response,
            t2_rt_ms=t2_rt,
        )

    summary = exp.get_summary()
    print("\nSample Results:")
    print(f"  T1 Accuracy: {summary['t1_accuracy']:.2%}")
    print("  T2 Accuracy by Lag:")
    for lag, acc in summary["t2_accuracy_by_lag"].items():
        print(f"    Lag {lag} ({lag * 100}ms): {acc:.2%}")
    print(f"  Blink Magnitude: {summary['blink_magnitude']:.3f}")
    return 0


if __name__ == "__main__":
    parser = create_standard_parser("Prepare Attentional Blink experiment")
    cli_entrypoint(main, parser)

# Standardized APGI Parameters Export (READ-ONLY)
# These parameters are used by the AGENT-EDITABLE run file for APGI integration
APGI_PARAMS = {
    # Core identification
    "experiment_name": "attentional_blink",
    "enabled": APGI_ENABLED,
    # Dynamical system timescales
    "tau_s": APGI_TAU_S,
    "tau_theta": APGI_TAU_THETA,
    "tau_M": APGI_TAU_M,
    # Threshold and ignition parameters
    "theta_0": APGI_THETA_0,
    "alpha": APGI_ALPHA,
    "rho": APGI_RHO,
    # Somatic modulation
    "beta": APGI_BETA,
    "beta_M": APGI_BETA_M,
    "M_0": APGI_M_0,
    # Sensitivities
    "gamma_M": APGI_GAMMA_M,
    "gamma_A": APGI_GAMMA_A,
    "lambda_S": APGI_LAMBDA_S,
    # Noise strengths
    "sigma_S": APGI_SIGMA_S,
    "sigma_theta": APGI_SIGMA_THETA,
    "sigma_M": APGI_SIGMA_M,
    # Domain-specific thresholds
    "theta_survival": APGI_THETA_SURVIVAL,
    "theta_neutral": APGI_THETA_NEUTRAL,
    # Hierarchical processing
    "hierarchical_enabled": APGI_HIERARCHICAL_ENABLED,
    "beta_cross": APGI_BETA_CROSS,
    "tau_levels": APGI_TAU_LEVELS,
    # Neuromodulator baselines
    "ACh": APGI_ACHT,
    "NE": APGI_NE,
    "DA": APGI_DA,
    "HT5": APGI_HT5,
    # Precision expectation gap
    "precision_gap_enabled": APGI_ENABLE_PRECISION_GAP,
    "Pi_e_expected": APGI_PI_E_EXPECTED,
    "Pi_i_expected": APGI_PI_I_EXPECTED,
    # Psychiatric profiles
    "GAD_profile": APGI_GAD_PROFILE,
    "MDD_profile": APGI_MDD_PROFILE,
    "psychosis_profile": APGI_PSYCHOSIS_PROFILE,
}
