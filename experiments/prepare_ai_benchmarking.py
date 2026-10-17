"""
Fixed constants and data preparation for AI Benchmarking experiments.

This file is READ-ONLY. Do not modify.
It defines the fixed benchmark configurations, task types, and evaluation metrics.

Usage:
    python prepare_ai_benchmarking.py  # Verify benchmark configurations

AI Benchmarking paradigms:
- Reasoning tasks: Logical deduction and problem solving
- Memory tasks: Information retention and recall
- Attention tasks: Selective focus and monitoring
- Decision making: Choice under uncertainty
- Learning tasks: Pattern recognition and adaptation
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from apgi_cli import cli_entrypoint, create_standard_parser

# ---------------------------------------------------------------------------
# Fixed Constants (DO NOT MODIFY)
# ---------------------------------------------------------------------------

TIME_BUDGET = 600  # 10 minutes per experiment (in seconds)

NUM_TRIALS = 50

# ---------------------------------------------------------------------------
# APGI Integration Parameters (DO NOT MODIFY)
# ---------------------------------------------------------------------------
# These parameters control the Allostatic Precision-Gated Ignition dynamics
# that track surprise accumulation, threshold dynamics, and somatic markers.

APGI_ENABLED = True  # Enable APGI tracking for this experiment
APGI_TAU_S = 0.35  # Surprise decay time constant (200-500 ms)
APGI_BETA = 1.5  # Somatic influence gain (0.5-2.5)
APGI_THETA_0 = 0.5  # Baseline ignition threshold (0.1-1.0)
APGI_ALPHA = 5.5  # Sigmoid steepness for ignition probability (3.0-8.0)


class BenchmarkType(Enum):
    REASONING = "reasoning"
    MEMORY = "memory"
    ATTENTION = "attention"
    DECISION_MAKING = "decision_making"
    LEARNING = "learning"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class AIBenchmarkTrial:
    trial_number: int
    benchmark_type: BenchmarkType
    difficulty: Difficulty
    task_description: str
    correct_answer: str
    model_response: Optional[str] = None
    correct: bool = False
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    tokens_used: int = 0
    timestamp: float = 0.0


class AIBenchmarkGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self) -> None:
        self.trial_count = 0

    def create_trial(self, trial_number: int) -> AIBenchmarkTrial:
        btype_list = [bt.value for bt in BenchmarkType]
        difficulty_list = [d.value for d in Difficulty]
        btype_str = self.rng.choice(btype_list)
        difficulty_str = self.rng.choice(difficulty_list)

        # Convert back to enum
        btype = BenchmarkType(btype_str)
        difficulty = Difficulty(difficulty_str)

        # Simple task descriptions
        tasks = {
            BenchmarkType.REASONING: "What is 2+2?",
            BenchmarkType.MEMORY: "Remember these items: apple, banana, cherry",
            BenchmarkType.ATTENTION: "Find the target in: A B C X D E F",
            BenchmarkType.DECISION_MAKING: "Choose between option A (100 pts) or B (50 pts)",
            BenchmarkType.LEARNING: "Pattern: A-B-A-C. What comes next?",
        }

        answers = {
            BenchmarkType.REASONING: "4",
            BenchmarkType.MEMORY: "apple, banana, cherry",
            BenchmarkType.ATTENTION: "X",
            BenchmarkType.DECISION_MAKING: "A",
            BenchmarkType.LEARNING: "A",
        }

        return AIBenchmarkTrial(
            trial_number=trial_number,
            benchmark_type=btype,
            difficulty=difficulty,
            task_description=tasks[btype],
            correct_answer=answers[btype],
        )


class AIBenchmarkExperiment:
    def __init__(self, num_trials: int = NUM_TRIALS, seed: Optional[int] = None):
        self.num_trials = num_trials
        self.generator = AIBenchmarkGenerator(seed=seed)
        self.trials: List[AIBenchmarkTrial] = []
        self.current_trial_idx = 0
        self.reset()

    def reset(self) -> None:
        self.trials = []
        self.current_trial_idx = 0
        self.generator.reset()

    def get_next_trial(self) -> Optional[AIBenchmarkTrial]:
        if self.current_trial_idx >= self.num_trials:
            return None
        trial = self.generator.create_trial(self.current_trial_idx + 1)
        self.current_trial_idx += 1
        return trial

    def run_trial(
        self,
        trial: AIBenchmarkTrial,
        response: str,
        confidence: float,
        proc_time: float,
        tokens: int,
    ) -> AIBenchmarkTrial:
        trial.model_response = response
        trial.correct = response.lower().strip() == trial.correct_answer.lower().strip()
        trial.confidence = confidence
        trial.processing_time_ms = proc_time
        trial.tokens_used = tokens
        import time

        trial.timestamp = time.time()
        self.trials.append(trial)
        self.current_trial_idx += 1
        return trial

    def get_accuracy_by_type(self, btype: BenchmarkType) -> float:
        trials = [t for t in self.trials if t.benchmark_type == btype]
        if not trials:
            return 0.0
        return float(np.mean([t.correct for t in trials]))

    def get_overall_score(self) -> float:
        """Weighted average across all benchmark types."""
        if not self.trials:
            return 0.0
        return float(np.mean([t.correct for t in self.trials]))

    def get_summary(self) -> Dict:
        if not self.trials:
            return {}
        return {
            "num_trials": len(self.trials),
            "overall_accuracy": self.get_overall_score(),
            "reasoning_accuracy": self.get_accuracy_by_type(BenchmarkType.REASONING),
            "memory_accuracy": self.get_accuracy_by_type(BenchmarkType.MEMORY),
            "attention_accuracy": self.get_accuracy_by_type(BenchmarkType.ATTENTION),
            "decision_accuracy": self.get_accuracy_by_type(
                BenchmarkType.DECISION_MAKING
            ),
            "learning_accuracy": self.get_accuracy_by_type(BenchmarkType.LEARNING),
            "mean_processing_time_ms": float(
                np.mean([t.processing_time_ms for t in self.trials])
            ),
            "total_tokens_used": sum([t.tokens_used for t in self.trials]),
        }

    def save_results(self, filepath: str) -> None:
        data = {
            "trials": [
                {
                    "trial_number": t.trial_number,
                    "benchmark_type": t.benchmark_type.value,
                    "difficulty": t.difficulty.value,
                    "task": t.task_description,
                    "correct_answer": t.correct_answer,
                    "model_response": t.model_response,
                    "correct": t.correct,
                    "confidence": t.confidence,
                    "processing_time_ms": t.processing_time_ms,
                    "tokens_used": t.tokens_used,
                    "timestamp": t.timestamp,
                }
                for t in self.trials
            ],
            "summary": self.get_summary(),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


# Standardized APGI Parameters Export (READ-ONLY)
# These parameters are used by the AGENT-EDITABLE run file for APGI integration
APGI_PARAMS = {
    "experiment_name": "ai_benchmarking",
    "enabled": True,
    "tau_s": 0.35,  # Surprise decay time constant (s)
    "beta": 1.5,  # Somatic influence gain
    "theta_0": 0.5,  # Baseline ignition threshold
    "alpha": 5.5,  # Sigmoid steepness
}


def verify() -> int:
    """Verify configuration and return status."""
    print("AI Benchmarking - Configuration Verification")
    print(f"Benchmark Types: {[b.value for b in BenchmarkType]}")
    print(f"Difficulty Levels: {[d.value for d in Difficulty]}")
    return 0


def main() -> int:
    """Entry point for Ai Benchmarking preparation."""
    return verify()


if __name__ == "__main__":
    parser = create_standard_parser("Prepare Ai Benchmarking experiment")
    cli_entrypoint(main, parser)
