"""
Virtual Navigation Experiment Runner

This is the AGENT-EDITABLE file for the auto-improvement system.
Modify this file to experiment with different task parameters, maze configurations,
and analysis approaches to maximize the path_efficiency metric.

Usage:
    uv run run_virtual_navigation.py

Output:
    Prints summary with path_efficiency (primary metric to maximize)

Modification Guidelines:
    - You CAN modify: task parameters, maze configurations, navigation strategies, etc.
    - You CANNOT modify: prepare_virtual_navigation.py (maze sets are fixed)
    - Goal: Maximize path_efficiency (navigation planning and execution efficiency)
    - Time budget: 10 minutes max per run
"""

import time
from typing import Any, Dict, List, Optional, Tuple

# Standardized APGI imports
from apgi_cli import cli_entrypoint, create_standard_parser
from apgi_integration import APGIIntegration, APGIParameters

from .prepare_virtual_navigation import (
    APGI_PARAMS,
    TIME_BUDGET,
    VirtualNavigationExperiment,
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

NUM_TRIALS_CONFIG = 20

# Navigation parameters
BASE_PATH_EFFICIENCY = 0.75
LEARNING_RATE = 0.02

# ---------------------------------------------------------------------------
# Simulated Participant
# ---------------------------------------------------------------------------


class SimulatedParticipant:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.base_efficiency = BASE_PATH_EFFICIENCY
        self.learning_rate = LEARNING_RATE
        self.trial_count = 0

    def process_trial(
        self, maze_size: int, optimal_length: int
    ) -> List[Tuple[int, int]]:
        # Simulate path with learning
        efficiency = self.base_efficiency + self.trial_count * self.learning_rate
        efficiency = min(0.95, efficiency)

        actual_length = int(optimal_length / efficiency)

        # Generate path (simplified: just add some detours)
        path = []
        x, y = 0, 0
        target_x, target_y = maze_size - 1, maze_size - 1

        # Optimal path
        for _ in range(target_x):
            x += 1
            path.append((x, y))
        for _ in range(target_y):
            y += 1
            path.append((x, y))

        # Add detours for longer paths
        while len(path) < actual_length:
            path.append(path[-1])

        self.trial_count += 1
        return path[:actual_length]


# ---------------------------------------------------------------------------
# Enhanced Runner
# ---------------------------------------------------------------------------


class EnhancedVirtualNavRunner:
    def __init__(self, enable_apgi: bool = True):
        self.experiment = VirtualNavigationExperiment(num_trials=NUM_TRIALS_CONFIG)
        self.participant = SimulatedParticipant()
        self.start_time: Optional[float] = None

        # Type annotations for APGI components
        self.apgi: Optional[APGIIntegration] = None
        self.hierarchical: Optional[HierarchicalProcessor] = None
        self.precision_gap: Optional[PrecisionExpectationState] = None
        self.neuromodulators: Optional[Dict[str, float]] = None
        self.running_stats: Optional[Dict[str, float]] = None

        # Initialize APGI components
        self.enable_apgi = enable_apgi and bool(APGI_PARAMS.get("enabled", True))
        if self.enable_apgi:
            # Type-safe extraction from APGI_PARAMS
            _tau_s: Any = APGI_PARAMS.get("tau_s", 0.35)
            _beta: Any = APGI_PARAMS.get("beta", 1.5)
            _theta_0: Any = APGI_PARAMS.get("theta_0", 0.5)
            _alpha: Any = APGI_PARAMS.get("alpha", 5.5)
            _gamma_M: Any = APGI_PARAMS.get("gamma_M", -0.3)
            _lambda_S: Any = APGI_PARAMS.get("lambda_S", 0.1)
            _sigma_S: Any = APGI_PARAMS.get("sigma_S", 0.05)
            _sigma_theta: Any = APGI_PARAMS.get("sigma_theta", 0.02)
            _sigma_M: Any = APGI_PARAMS.get("sigma_M", 0.03)
            _rho: Any = APGI_PARAMS.get("rho", 0.7)
            _theta_survival: Any = APGI_PARAMS.get("theta_survival", 0.3)
            _theta_neutral: Any = APGI_PARAMS.get("theta_neutral", 0.7)

            params = APGIParameters(
                tau_S=float(_tau_s or 0.35),
                beta=float(_beta or 1.5),
                theta_0=float(_theta_0 or 0.5),
                alpha=float(_alpha or 5.5),
                gamma_M=float(_gamma_M or -0.3),
                lambda_S=float(_lambda_S or 0.1),
                sigma_S=float(_sigma_S or 0.05),
                sigma_theta=float(_sigma_theta or 0.02),
                sigma_M=float(_sigma_M or 0.03),
                rho=float(_rho or 0.7),
                theta_survival=float(_theta_survival or 0.3),
                theta_neutral=float(_theta_neutral or 0.7),
            )
            self.apgi = APGIIntegration(params)

            # APGI component: Hierarchical 5-level processing (requires UltimateAPGIParameters)
            if APGI_PARAMS.get("hierarchical_enabled", True):
                _beta_cross: Any = APGI_PARAMS.get("beta_cross", 0.2)
                _tau_levels_raw: Any = APGI_PARAMS.get(
                    "tau_levels", [0.1, 0.2, 0.4, 1.0, 5.0]
                )
                _tau_levels: List[float] = (
                    _tau_levels_raw
                    if isinstance(_tau_levels_raw, list)
                    else [0.1, 0.2, 0.4, 1.0, 5.0]
                )
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
                    beta_cross=float(_beta_cross or 0.2),
                    tau_levels=_tau_levels,
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
            _ach: Any = APGI_PARAMS.get("ACh", 1.0)
            _ne: Any = APGI_PARAMS.get("NE", 1.0)
            _da: Any = APGI_PARAMS.get("DA", 1.0)
            _ht5: Any = APGI_PARAMS.get("HT5", 1.0)
            self.neuromodulators = {
                "ACh": float(_ach or 1.0),
                "NE": float(_ne or 1.0),
                "DA": float(_da or 1.0),
                "HT5": float(_ht5 or 1.0),
            }

            # APGI component: Running statistics for z-score normalization
            self.running_stats = {
                "outcome_mean": 0.5,
                "outcome_var": 0.25,
                "rt_mean": 500.0,
                "rt_var": 25000.0,
            }
        else:
            # APGI disabled - set all components to None
            pass

    def run_experiment(self) -> Dict:
        self.start_time = time.time()
        self.experiment.reset()
        self.participant.reset()

        for trial_num in range(NUM_TRIALS_CONFIG):
            self._run_single_trial(trial_num)

            if self.start_time and time.time() - self.start_time > TIME_BUDGET:
                break

        return self._calculate_results()

    def _run_single_trial(self, trial_num: int) -> None:
        trial = self.experiment.get_next_trial()
        if trial is None:
            return

        path = self.participant.process_trial(
            trial.maze_size, trial.optimal_path_length
        )

        self.experiment.run_trial(
            trial=trial,
            path=path,
            rt_ms=0,  # Navigation tasks don't have simple RT
        )

        # APGI component: Process with APGI if enabled
        if self.apgi and self.neuromodulators and self.running_stats:
            # Compute prediction error from trial outcome
            observed_accuracy = 1.0  # Navigation doesn't have correct/incorrect
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

            # APGI component: Process hierarchical levels
            if self.hierarchical:
                signal = apgi_state.get("S", 0.0)
                for level_idx in range(5):
                    level_state = self.hierarchical.process_level(level_idx, signal)
                    signal = level_state.S * 0.8

    def _calculate_results(self) -> Dict:
        summary = self.experiment.get_summary()
        completion_time = time.time() - self.start_time if self.start_time else 0.0

        apgi_metrics = {}
        if self.apgi and hasattr(self.apgi, "finalize"):
            apgi_metrics = self.apgi.finalize()

        return {
            **{
                "num_trials": len(self.experiment.trials),
                "completion_time_s": completion_time,
                "d_prime": summary.get("d_prime", 0.0),
                "path_efficiency": summary.get("path_efficiency", 0.0),
                "mean_excess_length": summary.get("mean_excess_length", 0.0),
            },
            **apgi_metrics,
        }


def print_results(results: Dict) -> None:
    print("\n" + "=" * 60)
    print("VIRTUAL NAVIGATION EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Trials: {results['num_trials']}")
    print(f"Time: {results['completion_time_s']:.2f}s")
    print(f"Path Efficiency: {results['path_efficiency']:.2%}")
    print(f"Mean Excess Length: {results['mean_excess_length']:.1f} steps")
    print("=" * 60)


def main(args: Any) -> Dict:
    """Main function for running the experiment."""
    runner = EnhancedVirtualNavRunner()
    results = runner.run_experiment()
    return results


if __name__ == "__main__":
    parser = create_standard_parser("Run Virtual Navigation  experiment")
    cli_entrypoint(main, parser)
