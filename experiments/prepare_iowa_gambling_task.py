"""
Fixed constants and data preparation for Iowa Gambling Task (IGT) experiments.

This file is READ-ONLY. Do not modify.
It defines the fixed deck configurations, win/loss probabilities, and evaluation metrics.

APGI Integration: Internal APGI checklist with hierarchical processing,
neuromodulator mapping, Π vs Π̂ distinction, and psychiatric profiles.

Usage:
    python prepare_iowa_gambling_task.py  # Verify deck configurations

The IGT uses 4 decks with different reward/penalty schedules:
- Decks A & B: Disadvantageous (high immediate reward, higher long-term loss)
- Decks C & D: Advantageous (lower immediate reward, lower long-term loss)

APGI Framework Integration:
- Somatic markers for decision-making under uncertainty (high β_som)
- Risk-sensitive thresholds (survival domain for disadvantageous decks)
- Neuromodulator effects: DA (reward learning), 5-HT (loss aversion)
- Precision expectation gap for anticipatory anxiety
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, cast

import numpy as np

# ---------------------------------------------------------------------------
# Fixed Constants (DO NOT MODIFY)
# ---------------------------------------------------------------------------

TIME_BUDGET = 600  # 10 minutes per experiment (in seconds)
NUM_TRIALS = 100  # Standard IGT trial count
NUM_DECKS = 4  # Standard IGT: 4 decks (A, B, C, D)

# APGI Integration Parameters
# Optimized for decision-making under uncertainty with somatic markers
APGI_ENABLED = True

# Core dynamical system parameters
APGI_TAU_S = 0.40  # Slower decay for decision integration (350-500ms)
APGI_TAU_THETA = 30.0  # Threshold adaptation (5-60s)
APGI_TAU_M = 1.5  # Somatic marker timescale (1-2s)

# Threshold and sigmoid parameters
APGI_THETA_0 = 0.4  # Lower baseline for risky decisions (0.1-1.0 AU)
APGI_ALPHA = 5.0  # Moderate steepness for decision ignition (3.0-8.0)

# Somatic modulation (HIGH for IGT - vmPFC-insula involvement)
APGI_BETA = 2.0  # β_som: High somatic influence (0.5-2.5)
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

# Domain-specific thresholds (decision-making contexts)
APGI_THETA_SURVIVAL = 0.3  # Lower threshold for risky/disadvantageous decks
APGI_THETA_NEUTRAL = 0.7  # Higher threshold for safe/advantageous decks

# Reset dynamics
APGI_RHO = 0.7  # Reset fraction after ignition (0.3-0.9)

# Hierarchical processing (5-level for decision-making hierarchy)
APGI_HIERARCHICAL_ENABLED = True
APGI_BETA_CROSS = 0.2  # Cross-level coupling (0.1-0.5)
APGI_TAU_LEVELS = [0.1, 0.2, 0.4, 1.0, 5.0]  # Level-specific timescales

# Neuromodulator baselines (decision-making focus)
APGI_ACHT = 1.0  # Acetylcholine - attention to outcomes
APGI_NE = 1.0  # Norepinephrine - arousal, exploration
APGI_DA = 1.0  # Dopamine - reward learning (critical for IGT)
APGI_HT5 = 1.0  # Serotonin - loss aversion, risk assessment

# Precision expectation gap (Π vs Π̂) for anticipatory anxiety
APGI_ENABLE_PRECISION_GAP = True
APGI_PI_E_EXPECTED = 1.2  # Expected exteroceptive precision
APGI_PI_I_EXPECTED = 1.1  # Expected interoceptive precision

# Psychiatric profiles (decision-making deficits)
APGI_GAD_PROFILE = False  # GAD: High precision expectations, risk aversion
APGI_MDD_PROFILE = False  # MDD: Reduced reward sensitivity
APGI_PSYCHOSIS_PROFILE = False  # Psychosis: Aberrant precision

# Deck labels for reference
DECK_LABELS = ["A", "B", "C", "D"]

# ---------------------------------------------------------------------------
# Standard IGT Deck Configurations (Fixed)
# ---------------------------------------------------------------------------
#
# Classic IGT structure (Bechara et al., 1994):
# - Decks A, B: Disadvantageous (high reward but higher penalties)
# - Decks C, D: Advantageous (lower reward but lower penalties)
#
# Win/loss schedules per 10-card blocks:

DECK_CONFIGURATIONS = {
    "A": {
        "label": "A (Disadvantageous)",
        "win_amount": 100,
        "win_frequency": 0.5,  # 5 wins per 10 cards
        "penalty_frequency": 0.5,  # 5 penalties per 10 cards
        "penalty_amounts": [150, 200, 250, 300, 350],  # Avg: 250 per penalty
        "advantageous": False,
    },
    "B": {
        "label": "B (Disadvantageous)",
        "win_amount": 100,
        "win_frequency": 0.9,  # 9 wins per 10 cards
        "penalty_frequency": 0.1,  # 1 penalty per 10 cards
        "penalty_amounts": [1250],  # Single large penalty
        "advantageous": False,
    },
    "C": {
        "label": "C (Advantageous)",
        "win_amount": 50,
        "win_frequency": 0.5,  # 5 wins per 10 cards
        "penalty_frequency": 0.5,  # 5 penalties per 10 cards
        "penalty_amounts": [25, 50, 75, 100, 125],  # Avg: 75 per penalty
        "advantageous": True,
    },
    "D": {
        "label": "D (Advantageous)",
        "win_amount": 50,
        "win_frequency": 0.9,  # 9 wins per 10 cards
        "penalty_frequency": 0.1,  # 1 penalty per 10 cards
        "penalty_amounts": [250],  # Single moderate penalty
        "advantageous": True,
    },
}

# Expected values per 10-card block
EXPECTED_VALUES = {
    "A": (10 * 100) - (5 * 250),  # 1000 - 1250 = -250 (loss)
    "B": (9 * 100) - (1 * 1250),  # 900 - 1250 = -350 (loss)
    "C": (10 * 50) - (5 * 75),  # 500 - 375 = +125 (gain)
    "D": (9 * 50) - (1 * 250),  # 450 - 250 = +200 (gain)
}


@dataclass
class IowaGamblingTaskTrial:
    """Single IGT trial data."""

    trial_number: int
    deck_choice: str  # 'A', 'B', 'C', or 'D'
    outcome: int  # Net outcome (win - penalty)
    win_amount: int
    penalty_amount: int
    reaction_time_ms: float = 0.0
    timestamp: float = 0.0


class IowaGamblingTaskDeck:
    """Represents a single IGT deck with its reward schedule."""

    def __init__(self, deck_id: str, config: Optional[Dict] = None):
        self.deck_id = deck_id
        self.config = config or DECK_CONFIGURATIONS[deck_id]
        self.reset()

    def reset(self) -> None:
        """Reset deck state for new experiment."""
        self.trial_count = 0
        self.card_index = 0
        # Pre-generate penalty schedule (shuffled per block)
        self._generate_penalty_schedule()

    def _generate_penalty_schedule(self) -> None:
        """Generate shuffled penalty schedule for card blocks."""
        freq = self.config["penalty_frequency"]
        penalty_amounts = cast(List[int], self.config["penalty_amounts"])

        # Create penalty sequence for many blocks
        blocks = 20  # Support up to 200 trials
        self.penalty_schedule = []

        for _ in range(blocks):
            # Create block with specified penalty frequency
            block_size = 10
            num_penalties = int(cast(float, freq) * block_size)
            block = [True] * num_penalties + [False] * (block_size - num_penalties)
            np.random.shuffle(block)

            # Assign penalty amounts
            penalty_idx = 0
            for is_penalty in block:
                if is_penalty:
                    amount = penalty_amounts[penalty_idx % len(penalty_amounts)]
                    self.penalty_schedule.append(amount)
                    penalty_idx += 1
                else:
                    self.penalty_schedule.append(0)

    def draw(self) -> Tuple[int, int]:
        """
        Draw a card from this deck.

        Returns:
            (win_amount, penalty_amount)
        """
        win = int(cast(int, self.config["win_amount"]))
        penalty = int(
            self.penalty_schedule[self.card_index % len(self.penalty_schedule)]
        )
        self.card_index += 1
        self.trial_count += 1
        return win, penalty

    def is_advantageous(self) -> bool:
        """Return True if this is an advantageous deck (C or D)."""
        return bool(self.config["advantageous"])


class IowaGamblingTaskExperiment:
    """Manages a complete IGT experiment session."""

    def __init__(self, num_trials: int = NUM_TRIALS):
        self.num_trials = num_trials
        self.decks = {label: IowaGamblingTaskDeck(label) for label in DECK_LABELS}
        self.trials: List[IowaGamblingTaskTrial] = []
        self.total_money = 2000  # Starting amount (typical IGT)
        self.reset()

    def reset(self) -> None:
        """Reset experiment state."""
        self.trials = []
        self.total_money = 2000
        for deck in self.decks.values():
            deck.reset()

    def run_trial(
        self, deck_choice: str, reaction_time_ms: float = 0.0
    ) -> IowaGamblingTaskTrial:
        """
        Execute a single trial.

        Args:
            deck_choice: 'A', 'B', 'C', or 'D'
            reaction_time_ms: Optional reaction time measurement

        Returns:
            IowaGamblingTaskTrial with outcome data
        """
        if deck_choice not in self.decks:
            raise ValueError(f"Invalid deck choice: {deck_choice}")

        deck = self.decks[deck_choice]
        win, penalty = deck.draw()
        net_outcome = win - penalty
        self.total_money += net_outcome

        import time

        trial = IowaGamblingTaskTrial(
            trial_number=len(self.trials) + 1,
            deck_choice=deck_choice,
            outcome=net_outcome,
            win_amount=win,
            penalty_amount=penalty,
            reaction_time_ms=reaction_time_ms,
            timestamp=time.time(),
        )
        self.trials.append(trial)
        return trial

    def get_net_score(self, last_n_trials: int = 20) -> float:
        """
        Calculate net score over last N trials.

        Net score = (advantageous choices - disadvantageous choices)
        Higher is better (more advantageous deck selections).
        """
        if not self.trials:
            return 0.0

        recent_trials = self.trials[-last_n_trials:]
        advantageous = sum(
            1 for t in recent_trials if self.decks[t.deck_choice].is_advantageous()
        )
        disadvantageous = len(recent_trials) - advantageous
        return float(advantageous - disadvantageous)

    def get_learning_rate(self) -> float:
        """
        Calculate learning rate based on choice progression.

        Returns correlation between trial block and advantageous choices.
        """
        if len(self.trials) < 20:
            return 0.0

        # Divide into 5 blocks
        block_size = len(self.trials) // 5
        block_scores = []

        for i in range(5):
            start = i * block_size
            end = start + block_size
            block_trials = self.trials[start:end]
            adv_choices = sum(
                1 for t in block_trials if self.decks[t.deck_choice].is_advantageous()
            )
            block_scores.append(adv_choices / len(block_trials))

        # Calculate trend (slope)
        x = np.arange(5)
        y = np.array(block_scores)
        slope = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0.0
        return float(slope)

    def get_summary(self) -> Dict:
        """Generate experiment summary statistics."""
        if not self.trials:
            return {}

        choices = {"A": 0, "B": 0, "C": 0, "D": 0}
        for t in self.trials:
            choices[t.deck_choice] += 1

        advantageous = sum(
            1 for t in self.trials if self.decks[t.deck_choice].is_advantageous()
        )
        disadvantageous = len(self.trials) - advantageous

        return {
            "num_trials": len(self.trials),
            "advantageous_choices": advantageous,
            "disadvantageous_choices": disadvantageous,
            "choices_by_deck": choices,
            "final_money": self.total_money,
            "net_score": self.get_net_score(),
            "net_score_first_half": self.get_net_score(
                last_n_trials=self.num_trials // 2
            ),
            "learning_rate": self.get_learning_rate(),
            "mean_reaction_time": (
                np.mean([t.reaction_time_ms for t in self.trials])
                if self.trials
                else 0.0
            ),
        }

    def save_results(self, filepath: str) -> None:
        """Save trial data to JSON file."""
        data = {
            "trials": [
                {
                    "trial_number": t.trial_number,
                    "deck_choice": t.deck_choice,
                    "outcome": t.outcome,
                    "win_amount": t.win_amount,
                    "penalty_amount": t.penalty_amount,
                    "reaction_time_ms": t.reaction_time_ms,
                    "timestamp": t.timestamp,
                }
                for t in self.trials
            ],
            "summary": self.get_summary(),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


def evaluate_net_score(
    trials: List[IowaGamblingTaskTrial],
    decks: Dict[str, IowaGamblingTaskDeck],
    last_n: int = 20,
) -> float:
    """
    Standalone evaluation function for net score calculation.

    This is the PRIMARY metric for the auto-improvement system.
    Goal: Maximize net score (advantageous - disadvantageous choices).
    """
    if not trials or len(trials) < last_n:
        return 0.0

    recent = trials[-last_n:]
    advantageous = sum(1 for t in recent if decks[t.deck_choice].is_advantageous())
    disadvantageous = len(recent) - advantageous
    return float(advantageous - disadvantageous)


def verify_deck_configurations() -> None:
    """Verify and print deck configurations."""
    print("=" * 60)
    print("Iowa Gambling Task - Deck Configuration Verification")
    print("=" * 60)

    for deck_id, config in DECK_CONFIGURATIONS.items():
        print(f"\nDeck {config['label']}:")
        print(f"  Win Amount: ${config['win_amount']}")
        print(f"  Win Frequency: {cast(float, config['win_frequency']) * 100:.0f}%")
        print(
            f"  Penalty Frequency: {cast(float, config['penalty_frequency']) * 100:.0f}%"
        )
        print(f"  Penalty Amounts: {config['penalty_amounts']}")
        print(f"  Advantageous: {config['advantageous']}")

        # Calculate expected value per 10 cards
        ev = EXPECTED_VALUES[deck_id]
        print(f"  Expected Value (per 10 cards): ${ev}")

    print("\n" + "=" * 60)
    print("Expected long-term outcomes:")
    print("  Decks A & B: Net loss (disadvantageous)")
    print("  Decks C & D: Net gain (advantageous)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    verify_deck_configurations()

    # Run a quick simulation to verify
    print("\n\nRunning sample simulation (100 trials, random choices)...")
    exp = IowaGamblingTaskExperiment(num_trials=100)

    # Simulate random choices
    for _ in range(100):
        deck = np.random.choice(DECK_LABELS)
        exp.run_trial(deck, reaction_time_ms=np.random.uniform(500, 2000))

    summary = exp.get_summary()
    print("\nSample Results:")
    print(f"  Total trials: {summary['num_trials']}")
    print(f"  Advantageous choices: {summary['advantageous_choices']}")
    print(f"  Disadvantageous choices: {summary['disadvantageous_choices']}")
    print(f"  Net score (last 20): {summary['net_score']:.1f}")
    print(f"  Learning rate: {summary['learning_rate']:.3f}")
    print(f"  Final money: ${summary['final_money']}")

# Standardized APGI Parameters Export (internal checklist)
# These parameters provide full APGI framework integration for the run file
APGI_PARAMS = {
    # Core identification
    "experiment_name": "iowa_gambling_task",
    "enabled": APGI_ENABLED,
    # Dynamical system timescales
    "tau_s": APGI_TAU_S,
    "tau_theta": APGI_TAU_THETA,
    "tau_M": APGI_TAU_M,
    # Threshold and ignition parameters
    "theta_0": APGI_THETA_0,
    "alpha": APGI_ALPHA,
    "rho": APGI_RHO,
    # Somatic modulation (high for IGT decision-making)
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
    # Domain-specific thresholds (risk-sensitive decision-making)
    "theta_survival": APGI_THETA_SURVIVAL,
    "theta_neutral": APGI_THETA_NEUTRAL,
    # Hierarchical processing (5-level for decision hierarchy)
    "hierarchical_enabled": APGI_HIERARCHICAL_ENABLED,
    "beta_cross": APGI_BETA_CROSS,
    "tau_levels": APGI_TAU_LEVELS,
    # Neuromodulator baselines (DA critical for reward learning)
    "ACh": APGI_ACHT,
    "NE": APGI_NE,
    "DA": APGI_DA,
    "HT5": APGI_HT5,
    # Precision expectation gap (Π vs Π̂) for anticipatory anxiety
    "precision_gap_enabled": APGI_ENABLE_PRECISION_GAP,
    "Pi_e_expected": APGI_PI_E_EXPECTED,
    "Pi_i_expected": APGI_PI_I_EXPECTED,
    # Psychiatric profiles (decision-making deficits)
    "GAD_profile": APGI_GAD_PROFILE,
    "MDD_profile": APGI_MDD_PROFILE,
    "psychosis_profile": APGI_PSYCHOSIS_PROFILE,
}
