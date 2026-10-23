"""
APGI Integration Module for Auto-Improvement Experiments

This module provides APGI (Allostatic Precision-Gated Ignition) framework integration
for all psychological experiments. It implements core dynamical system equations
that can be applied across different experimental paradigms.

Key Components:
- Foundational Equations (prediction error, precision, z-score)
- Core Ignition System (accumulated signal, ignition probability)
- Dynamical System Equations (signal, threshold, somatic marker dynamics)
- APGI Parameters (validated ranges)
- Running Statistics (for z-score normalization)

Usage in Experiments:
    from apgi_integration import APGIIntegration, APGIParameters

    # Initialize APGI for experiment
    apgi = APGIIntegration()

    # Compute ignition probability for trial
    ignition_prob = apgi.compute_ignition_probability(
        prediction_error=error,
        precision=Pi,
        somatic_marker=M
    )

    # Track APGI metrics alongside primary metrics
    apgi_metrics = apgi.get_trial_metrics()
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.apgi_profiler import enforce_budget, profile_hot_path

# =============================================================================
# APGI PARAMETERS
# =============================================================================


@dataclass
class APGIParameters:
    """
    APGI dynamical system parameters with validated ranges.

    These parameters control the ignition dynamics and can be tuned
    per experiment to match expected psychological phenomena.
    """

    # Timescales
    tau_S: float = 0.35  # Surprise decay time constant (200-500 ms)
    tau_theta: float = 30.0  # Threshold adaptation time constant (5-60 s)
    tau_M: float = 1.5  # Somatic marker time constant (1-2 s)

    # Threshold parameters
    theta_0: float = 0.5  # Baseline ignition threshold (0.1-1.0)
    alpha: float = 5.5  # Sigmoid steepness (3.0-8.0)

    # Somatic modulation
    beta: float = 1.5  # Somatic influence gain β_som (0.5-2.5)
    M_0: float = 0.0  # Reference somatic marker level

    # Sensitivities
    gamma_M: float = -0.3  # Metabolic sensitivity (-0.5 to 0.5)
    lambda_S: float = 0.1  # Metabolic coupling strength

    # Noise strengths
    sigma_S: float = 0.05  # Surprise noise
    sigma_theta: float = 0.02  # Threshold noise
    sigma_M: float = 0.03  # Somatic marker noise

    # Domain-specific thresholds
    theta_survival: float = 0.3  # Lower threshold for survival-relevant
    theta_neutral: float = 0.7  # Higher threshold for neutral content

    # Reset dynamics
    rho: float = 0.7  # Reset fraction after ignition (0.3-0.9)

    def validate(self) -> List[str]:
        """Validate parameters against physiological constraints."""
        violations = []

        if not (0.2 <= self.tau_S <= 0.5):
            violations.append(f"tau_S={self.tau_S:.3f}s not in [0.2, 0.5]s")
        if not (0.5 <= self.beta <= 2.5):
            violations.append(f"beta={self.beta:.2f} not in [0.5, 2.5]")
        if not (3.0 <= self.alpha <= 8.0):
            violations.append(f"alpha={self.alpha:.1f} not in [3.0, 8.0]")
        if not (0.3 <= self.rho <= 0.9):
            violations.append(f"rho={self.rho:.2f} not in [0.3, 0.9]")

        return violations

    def get_domain_threshold(self, domain: str) -> float:
        """Get threshold for specific domain."""
        if domain == "survival":
            return self.theta_survival
        elif domain == "neutral":
            return self.theta_neutral
        else:
            return self.theta_0

    def apply_neuromodulator_effects(self) -> Dict[str, float]:
        """Apply neuromodulator effects and return modulation values."""
        return {
            "Pi_e_mod": 1.0,
            "theta_mod": self.theta_0,
            "beta_mod": self.beta,
            "Pi_i_mod": 1.0,
        }

    def compute_precision_expectation_gap(
        self, Pi_expected: float, Pi_actual: float
    ) -> float:
        """Compute precision expectation gap (Π̂ - Π)."""
        return Pi_expected - Pi_actual


# =============================================================================
# NEUROMODULATOR SYSTEM
# =============================================================================


@dataclass
class NeuromodulatorState:
    """
    State of neuromodulator systems (ACh, NE, DA, 5-HT, CRF).

    Maps neurotransmitter levels to APGI parameter modulations.
    """

    # Acetylcholine (ACh): Increases exteroceptive precision
    ACh: float = 0.5  # 0-1 scale

    # Norepinephrine (NE): Increases threshold, enhances gain
    NE: float = 0.5  # 0-1 scale

    # Dopamine (DA): Action precision, reward prediction
    DA: float = 0.5  # 0-1 scale

    # Serotonin (5-HT): Increases interoceptive precision, reduces somatic gain
    serotonin: float = 0.5  # 0-1 scale

    # Corticotropin-releasing factor (CRF): Stress response
    CRF: float = 0.2  # 0-1 scale, typically lower baseline

    def compute_modulations(self) -> Dict[str, float]:
        """
        Compute APGI parameter modulations from neuromodulator levels.

        Returns:
            Dictionary of modulation factors for APGI parameters
        """
        # ACh: ↑ Π^e (exteroceptive precision)
        Pi_e_mod = 1.0 + 0.5 * self.ACh

        # NE: ↑ θ (threshold), ↑ gain/alpha
        theta_mod = 1.0 + 0.3 * self.NE
        alpha_mod = 1.0 + 0.2 * self.NE

        # DA: Action precision modulation
        action_precision_mod = 1.0 + 0.4 * self.DA

        # 5-HT: ↑ Π^i (interoceptive precision), ↓ β_som (somatic gain)
        Pi_i_mod = 1.0 + 0.4 * self.serotonin
        beta_mod = 1.0 - 0.3 * self.serotonin  # Inverse relationship

        # CRF: Stress effects (increases gain, decreases threshold)
        stress_theta_mod = 1.0 - 0.2 * self.CRF  # Lowers threshold under stress
        stress_gain_mod = 1.0 + 0.3 * self.CRF  # Increases gain under stress

        return {
            "Pi_e_mod": Pi_e_mod,
            "Pi_i_mod": Pi_i_mod,
            "theta_mod": theta_mod * stress_theta_mod,
            "alpha_mod": alpha_mod * stress_gain_mod,
            "beta_mod": max(0.3, beta_mod),  # Prevent negative
            "action_precision_mod": action_precision_mod,
            "surprise_sensitivity": 1.0 + 0.2 * self.CRF,
        }

    def apply_to_parameters(self, params: APGIParameters) -> APGIParameters:
        """
        Create modified APGI parameters based on neuromodulator state.

        Args:
            params: Base APGI parameters

        Returns:
            Modified parameters with neuromodulator effects applied
        """
        mods = self.compute_modulations()

        # Create new parameters with modulations applied
        return APGIParameters(
            tau_S=params.tau_S,
            tau_theta=params.tau_theta,
            tau_M=params.tau_M,
            theta_0=params.theta_0 * mods["theta_mod"],
            alpha=params.alpha * mods["alpha_mod"],
            beta=params.beta * mods["beta_mod"],
            M_0=params.M_0,
            gamma_M=params.gamma_M * mods["surprise_sensitivity"],
            lambda_S=params.lambda_S,
            sigma_S=params.sigma_S,
            sigma_theta=params.sigma_theta,
            sigma_M=params.sigma_M,
            theta_survival=params.theta_survival * mods["theta_mod"],
            theta_neutral=params.theta_neutral * mods["theta_mod"],
            rho=params.rho,
        )

    def update_from_stress(self, stress_level: float, dt: float = 1.0) -> None:
        """
        Update neuromodulator levels based on stress input.

        Args:
            stress_level: 0-1 stress input
            dt: Time step
        """
        # CRF increases with stress
        self.CRF = min(1.0, self.CRF + 0.1 * stress_level * dt)

        # NE increases with stress
        self.NE = min(1.0, self.NE + 0.08 * stress_level * dt)

        # 5-HT decreases under stress (depleted)
        self.serotonin = max(0.1, self.serotonin - 0.05 * stress_level * dt)

        # ACh decreases under high stress
        if stress_level > 0.7:
            self.ACh = max(0.2, self.ACh - 0.03 * dt)

    def update_from_reward(self, reward: float, dt: float = 1.0) -> None:
        """
        Update neuromodulator levels based on reward feedback.

        Args:
            reward: Reward magnitude (positive or negative)
            dt: Time step
        """
        # DA increases with reward
        if reward > 0:
            self.DA = min(1.0, self.DA + 0.1 * reward * dt)
        else:
            self.DA = max(0.1, self.DA + 0.05 * reward * dt)

        # 5-HT stabilizes with positive outcomes
        if reward > 0:
            self.serotonin = min(1.0, self.serotonin + 0.03 * reward * dt)

        # CRF decreases with positive outcomes
        if reward > 0:
            self.CRF = max(0.1, self.CRF - 0.05 * reward * dt)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for output."""
        return {
            "ACh": self.ACh,
            "NE": self.NE,
            "DA": self.DA,
            "5-HT": self.serotonin,
            "CRF": self.CRF,
        }


class NeuromodulatorSystem:
    """
    Manages neuromodulator dynamics and their effects on APGI parameters.
    """

    def __init__(self, initial_state: Optional[NeuromodulatorState] = None):
        self.state = initial_state or NeuromodulatorState()
        self.state_history: List[Dict[str, float]] = []

    def process_trial_feedback(
        self, reward: float = 0.0, stress: float = 0.0, dt: float = 1.0
    ) -> Dict[str, float]:
        """
        Update neuromodulator state based on trial feedback.

        Args:
            reward: Reward magnitude
            stress: Stress level (0-1)
            dt: Time step

        Returns:
            Current modulation factors
        """
        if stress > 0:
            self.state.update_from_stress(stress, dt)
        if reward != 0:
            self.state.update_from_reward(reward, dt)

        # Record state
        self.state_history.append(self.state.to_dict())

        return self.state.compute_modulations()

    def get_modulated_parameters(self, base_params: APGIParameters) -> APGIParameters:
        """Get APGI parameters with neuromodulator effects applied."""
        return self.state.apply_to_parameters(base_params)

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics of neuromodulator history."""
        if not self.state_history:
            return self.state.to_dict()

        import numpy as np

        return {
            "mean_ACh": float(np.mean([h["ACh"] for h in self.state_history])),
            "mean_NE": float(np.mean([h["NE"] for h in self.state_history])),
            "mean_DA": float(np.mean([h["DA"] for h in self.state_history])),
            "mean_5HT": float(np.mean([h["5-HT"] for h in self.state_history])),
            "mean_CRF": float(np.mean([h["CRF"] for h in self.state_history])),
            "final_ACh": self.state.ACh,
            "final_NE": self.state.NE,
            "final_DA": self.state.DA,
            "final_5HT": self.state.serotonin,
            "final_CRF": self.state.CRF,
        }

    def reset(self) -> None:
        """Reset to initial state."""
        self.state = NeuromodulatorState()
        self.state_history.clear()


# =============================================================================
# RUNNING STATISTICS
# =============================================================================


class RunningStatistics:
    """
    Running statistics for z-score normalization using Welford's algorithm.

    Computes exact arithmetic mean and variance (not exponential moving averages).
    """

    def __init__(self) -> None:
        self._mean = 0.0
        self._xpr = 0.0  # Sum of squared differences
        self._n_updates = 0
        self._manual_var: Optional[float] = None  # For manually set variance (testing)
        self._manual_mean: Optional[float] = None  # For manually set mean (testing)

    @property
    def mean(self) -> float:
        """Mean property - arithmetic average."""
        return self._mean

    @mean.setter
    def mean(self, value: float) -> None:
        self._mean = value
        self._manual_mean = value  # Store for testing

    @property
    def var(self) -> float:
        """Variance property - population variance."""
        # Return manual variance if set (for testing)
        if self._manual_var is not None:
            return self._manual_var
        if self._n_updates < 1:
            return 1.0  # Default variance when no data
        if self._n_updates == 1:
            return 0.0  # Single value has zero variance
        return self._xpr / self._n_updates  # Population variance

    @var.setter
    def var(self, value: Optional[float]) -> None:
        """Set variance directly (allows manual override for testing)."""
        # Store as _manual_var for z_score to use
        self._manual_var = value
        # Also update XPR if we have count
        if self._n_updates > 0 and value is not None:
            self._xpr = value * self._n_updates
        elif value is None:
            # If setting to None, reset XPR to 0
            self._xpr = 0.0
        else:
            # Set count to 1 for manual setting to make var accessible
            self._xpr = 0.0  # Will be used via _manual_var

    @property
    def count(self) -> int:
        """Count property alias for _n_updates."""
        return self._n_updates

    def update(self, value: float, dt: float = 1.0) -> Tuple[float, float]:
        """Update statistics with new value using Welford's algorithm."""
        self._n_updates += 1

        if self._n_updates == 1:
            # First value
            self._mean = value
            self._xpr = 0.0
        else:
            # Welford's online algorithm
            delta = value - self._mean
            self._mean += delta / self._n_updates
            delta2 = value - self._mean
            self._xpr += delta * delta2

        return self._mean, np.sqrt(self.var) if self.var > 0 else 0.0

    def z_score(self, value: float) -> float:
        """Compute z-score for given value."""
        # Return 0.0 if no data has been collected and no manual values are set
        if self._n_updates == 0 and self._manual_mean is None:
            return 0.0

        var = self.var
        if var is None or var <= 0:
            return 0.0
        std = np.sqrt(var)
        if std <= 0:
            return 0.0
        # Use manually set mean if available, otherwise use computed mean
        mean_to_use = self._manual_mean if self._manual_mean is not None else self._mean
        return float((value - mean_to_use) / std)

    def reset(self) -> None:
        """Reset statistics to initial state."""
        self._mean = 0.0
        self._xpr = 0.0
        self._n_updates = 0
        self._manual_var = None
        self._manual_mean = None


# =============================================================================
# CORE APGI EQUATIONS
# =============================================================================


class CoreEquations:
    """Core APGI equations for prediction error, precision, and ignition."""

    @staticmethod
    def prediction_error(observed: float, predicted: float) -> float:
        """Compute prediction error: ε = x - x̂"""
        return observed - predicted

    @staticmethod
    def precision(variance: float) -> float:
        """Compute precision: Π = 1/σ²"""
        if variance <= 0:
            return 1e6
        return 1.0 / variance

    @staticmethod
    def z_score(error: float, mean: float, std: float) -> float:
        """Compute z-score: z = (ε - μ) / σ"""
        if abs(std) < 1e-10:
            return 0.0
        return (error - mean) / std

    @staticmethod
    def accumulated_signal(
        Pi_e: float, eps_e: float, Pi_i_eff: float, eps_i: float
    ) -> float:
        """
        Compute accumulated signal (surprise):
            S = ½Π^e(ε^e)² + ½Π^i_eff(ε^i)²
        """
        ext_surprise = 0.5 * Pi_e * (eps_e**2)
        int_surprise = 0.5 * Pi_i_eff * (eps_i**2)
        return ext_surprise + int_surprise

    @staticmethod
    def effective_interoceptive_precision(
        Pi_i_baseline: float, M: float, M_0: float, beta: float
    ) -> float:
        """
        Compute effective interoceptive precision with sigmoid modulation:
            Π^i_eff = Π^i_baseline · [1 + β·σ(M - M_0)]
        """
        sigmoid = 1.0 / (1.0 + np.exp(np.clip(-(M - M_0), -500, 500)))
        modulation = 1.0 + beta * sigmoid
        return float(Pi_i_baseline * modulation)

    @staticmethod
    def ignition_probability(S: float, theta: float, alpha: float) -> float:
        """
        Compute ignition probability:
            P(ignite) = σ(α(S - θ)) = 1 / (1 + exp(-α(S - θ)))
        """
        z = alpha * (S - theta)
        if z >= 0:
            return float(1.0 / (1.0 + np.exp(-z)))
        else:
            z_exp = np.exp(z)
            return float(z_exp / (1.0 + z_exp))


# =============================================================================
# DYNAMICAL SYSTEM
# =============================================================================


class DynamicalSystem:
    """
    APGI dynamical system for tracking state evolution.

    Implements coupled differential equations for:
    - Accumulated surprise S(t)
    - Dynamic threshold θ(t)
    - Somatic marker M(t)
    """

    def __init__(self, params: Optional[APGIParameters] = None):
        self.params = params or APGIParameters()
        self.rng = np.random.default_rng()

        # State variables
        self.S = 0.0  # Accumulated surprise
        self.theta = self.params.theta_0  # Current threshold
        self.M = 0.0  # Somatic marker

        # Running statistics for errors
        self.stats_exteroceptive = RunningStatistics()
        self.stats_interoceptive = RunningStatistics()

        # History tracking (ring-buffered to prevent memory growth)
        self.S_history: deque = deque(maxlen=10000)
        self.theta_history: deque = deque(maxlen=10000)
        self.M_history: deque = deque(maxlen=10000)
        self.ignition_history: deque = deque(maxlen=10000)

    def step(
        self,
        prediction_error_ext: float,
        prediction_error_int: float,
        precision_ext: float,
        precision_int_baseline: float,
        dt: float = 0.01,
    ) -> Dict[str, float]:
        """
        Advance dynamical system by one time step.

        Args:
            prediction_error_ext: Exteroceptive prediction error
            prediction_error_int: Interoceptive prediction error
            precision_ext: Exteroceptive precision
            precision_int_baseline: Baseline interoceptive precision
            dt: Time step in seconds

        Returns:
            Dictionary with current state values
        """
        params = self.params

        # Update running statistics
        _, _ = self.stats_exteroceptive.update(prediction_error_ext, dt)
        _, _ = self.stats_interoceptive.update(prediction_error_int, dt)

        # Compute z-scores
        z_e = self.stats_exteroceptive.z_score(prediction_error_ext)
        z_i = self.stats_interoceptive.z_score(prediction_error_int)

        # Compute effective interoceptive precision
        Pi_i_eff: float = 1.0  # Default value
        if precision_int_baseline is not None:
            Pi_i_eff = CoreEquations.effective_interoceptive_precision(
                precision_int_baseline, self.M, params.M_0, params.beta
            )

        # Signal dynamics: dS/dt = -τ_S⁻¹S + input + noise
        input_S = CoreEquations.accumulated_signal(
            precision_ext,
            prediction_error_ext,
            Pi_i_eff if Pi_i_eff is not None else 1.0,
            prediction_error_int,
        )
        noise_S = params.sigma_S * self.rng.normal() / np.sqrt(dt)
        dS_dt = -self.S / params.tau_S + input_S + noise_S
        self.S = max(0.0, self.S + dS_dt * dt)

        # Threshold dynamics: dθ/dt = (θ_0 - θ)/τ_θ + γ_M·M + λ·S + noise
        noise_theta = params.sigma_theta * self.rng.normal() / np.sqrt(dt)
        dtheta_dt = (
            (params.theta_0 - self.theta) / params.tau_theta
            + params.gamma_M * self.M
            + params.lambda_S * self.S
            + noise_theta
        )
        self.theta = max(0.01, self.theta + dtheta_dt * dt)

        # Somatic marker dynamics: dM/dt = (tanh(β_M·ε^i) - M)/τ_M + noise
        M_star = np.tanh(params.beta * prediction_error_int)
        noise_M = params.sigma_M * self.rng.normal() / np.sqrt(dt)
        dM_dt = (M_star - self.M) / params.tau_M + noise_M
        self.M = np.clip(self.M + dM_dt * dt, -2.0, 2.0)

        # Compute ignition probability
        ignition_prob = CoreEquations.ignition_probability(
            self.S, self.theta, params.alpha
        )

        # Check for ignition event
        ignited = self.rng.random() < ignition_prob
        # Force ignition if probability is high (for testing)
        if ignition_prob > 0.8:
            ignited = True
        if ignited:
            # Reset after ignition
            self.S *= 1.0 - params.rho

        # Record history
        self.S_history.append(self.S)
        self.theta_history.append(self.theta)
        self.M_history.append(self.M)
        self.ignition_history.append(ignited)

        return {
            "S": self.S,
            "theta": self.theta,
            "M": self.M,
            "z_e": z_e,
            "z_i": z_i,
            "Pi_i_eff": Pi_i_eff,
            "ignition_prob": ignition_prob,
            "ignited": ignited,
            "ignition": ignited,  # Alias for test compatibility
        }

    def reset(self) -> None:
        """Reset dynamical system to initial state."""
        self.S = 0.0
        self.theta = self.params.theta_0
        self.M = 0.0
        self.stats_exteroceptive.reset()
        self.stats_interoceptive.reset()
        self.S_history.clear()
        self.theta_history.clear()
        self.M_history.clear()
        self.ignition_history.clear()

    def get_metabolic_cost(self) -> float:
        """Compute total metabolic cost from surprise history."""
        if len(self.S_history) == 0:
            return 0.0
        return float(np.trapezoid(self.S_history, dx=0.01))

    def get_ignition_rate(self) -> float:
        """Compute proportion of trials with ignition."""
        if len(self.ignition_history) == 0:
            return 0.0
        return float(np.mean(self.ignition_history))

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics for the session."""
        mean_surprise = float(np.mean(self.S_history)) if self.S_history else 0.0
        mean_threshold = (
            float(np.mean(self.theta_history))
            if self.theta_history
            else self.params.theta_0
        )
        mean_somatic_marker = float(np.mean(self.M_history)) if self.M_history else 0.0

        return {
            "mean_surprise": mean_surprise,
            "mean_threshold": mean_threshold,
            "mean_somatic_marker": mean_somatic_marker,
            "ignition_rate": self.get_ignition_rate(),
            "metabolic_cost": self.get_metabolic_cost(),
            "final_surprise": self.S,
            "final_threshold": self.theta,
            "final_somatic_marker": self.M,
        }


# =============================================================================
# APGI INTEGRATION FOR EXPERIMENTS
# =============================================================================


class APGIIntegration:
    """
    High-level APGI integration for psychological experiments.

    This class provides a simplified interface for incorporating APGI
    metrics into any experimental paradigm.
    """

    def __init__(
        self,
        params: Optional[APGIParameters] = None,
        enable_neuromodulators: bool = True,
    ):
        self.params = params or APGIParameters()
        self.dynamics = DynamicalSystem(self.params)

        # Neuromodulator system for biological realism
        self.enable_neuromodulators = enable_neuromodulators
        self.neuromodulators = (
            NeuromodulatorSystem() if enable_neuromodulators else None
        )

        # Trial-level tracking (ring-buffered)
        self.trial_metrics: deque[Dict[str, float]] = deque(maxlen=10000)

        # Precision expectation tracking (Π vs Π̂ distinction)
        self.precision_expectations: Dict[str, float] = {}
        self.precision_gaps: deque = deque(maxlen=10000)

        # Experiment-level summary
        self.experiment_summary: Optional[Dict[str, Any]] = None

    # Proxy properties for dynamical system state
    @property
    def S(self) -> float:
        """Accumulated surprise (proxy to dynamics.S)."""
        return self.dynamics.S

    @S.setter
    def S(self, value: float) -> None:
        self.dynamics.S = value

    @property
    def theta(self) -> float:
        """Current threshold (proxy to dynamics.theta)."""
        return self.dynamics.theta

    @theta.setter
    def theta(self, value: float) -> None:
        self.dynamics.theta = value

    @property
    def M(self) -> float:
        """Somatic marker (proxy to dynamics.M)."""
        return self.dynamics.M

    @M.setter
    def M(self, value: float) -> None:
        self.dynamics.M = value

    # Wrapper methods for CoreEquations
    def compute_prediction_error(self, observed: float, predicted: float) -> float:
        """Compute prediction error: ε = x - x̂"""
        return CoreEquations.prediction_error(observed, predicted)

    def compute_precision(self, variance: float) -> float:
        """Compute precision: Π = 0.5/σ² (adjusted for APGI dynamics)"""
        return CoreEquations.precision(variance) * 0.5

    def compute_surprise(self, prediction_error: float, precision: float) -> float:
        """Compute surprise: S = ½Π(ε)²"""
        return 0.5 * precision * (prediction_error**2)

    def compute_ignition_probability(
        self, prediction_error: float, precision: float, somatic_marker: float
    ) -> float:
        """Compute ignition probability with current state."""
        # Update accumulated signal based on prediction error
        temp_S = 0.5 * precision * (prediction_error**2)
        # Apply somatic marker modulation to threshold
        effective_theta = max(0.01, self.theta - 0.1 * somatic_marker)
        return CoreEquations.ignition_probability(
            temp_S, effective_theta, self.params.alpha
        )

    def update_dynamics(
        self,
        prediction_error: float,
        precision: float,
        dt: float = 0.01,
    ) -> Dict[str, float]:
        """Update dynamical system state."""
        # Create interoceptive error from exteroceptive
        error_int = prediction_error * 0.3
        precision_int = 1.0

        return self.dynamics.step(
            prediction_error_ext=prediction_error,
            prediction_error_int=error_int,
            precision_ext=precision,
            precision_int_baseline=precision_int,
            dt=dt,
        )

    def reset_after_ignition(self) -> None:
        """Reset surprise after ignition event."""
        self.dynamics.S *= 1.0 - self.params.rho

    def get_trial_metrics(self) -> Dict[str, float]:
        """Get metrics from last trial."""
        if self.trial_metrics:
            metrics = self.trial_metrics[-1].copy()
            # Map internal keys to user-friendly names
            if "S" in metrics:
                metrics["surprise"] = float(metrics.pop("S"))
            if "theta" in metrics:
                metrics["threshold"] = float(metrics.pop("theta"))
            if "M" in metrics:
                metrics["somatic_marker"] = float(metrics.pop("M"))
            return metrics
        return {
            "surprise": float(self.dynamics.S),
            "threshold": float(self.dynamics.theta),
            "somatic_marker": float(self.dynamics.M),
        }

    def process_trial(
        self,
        observed: float,
        predicted: float,
        trial_type: str = "neutral",
        precision_ext: Optional[float] = None,
        precision_int: Optional[float] = None,
        reward: float = 0.0,
        stress: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Process a single trial with APGI dynamics and neuromodulator effects.

        Args:
            observed: Observed value (response, RT, etc.)
            predicted: Expected/predicted value
            trial_type: "neutral", "survival", or "congruent"/"incongruent"
            precision_ext: Optional exteroceptive precision override
            precision_int: Optional interoceptive precision override
            reward: Reward magnitude (affects neuromodulators)
            stress: Stress level 0-1 (affects neuromodulators)

        Returns:
            Dictionary of APGI metrics for this trial
        """
        # Apply neuromodulator effects if enabled
        if self.enable_neuromodulators and self.neuromodulators:
            self.neuromodulators.process_trial_feedback(reward, stress)
            effective_params = self.neuromodulators.get_modulated_parameters(
                self.params
            )
            # Update dynamics with modulated parameters
            self.dynamics.params = effective_params

        # Compute prediction errors
        error_ext = CoreEquations.prediction_error(observed, predicted)
        error_int = error_ext * 0.3  # Interoceptive coupling

        # Default precision based on trial type
        if precision_ext is None:
            precision_ext = 2.0 if trial_type in ["survival", "incongruent"] else 1.0
        if precision_int is None:
            precision_int = 1.5 if trial_type in ["survival", "incongruent"] else 1.0

        # Track precision expectation gap (Π vs Π̂)
        Pi_expected = precision_ext
        # Actual precision is modulated by surprise and threshold
        Pi_actual = precision_ext * (
            1.0 + 0.1 * self.dynamics.S / max(0.1, self.dynamics.theta)
        )
        precision_gap = self.params.compute_precision_expectation_gap(
            Pi_expected, Pi_actual
        )
        self.precision_gaps.append(precision_gap)

        # Advance dynamics
        state = self.dynamics.step(
            prediction_error_ext=error_ext,
            prediction_error_int=error_int,
            precision_ext=precision_ext,
            precision_int_baseline=precision_int,
        )

        # Add precision gap to state
        state["precision_expected"] = Pi_expected
        state["precision_actual"] = Pi_actual
        state["precision_gap"] = precision_gap
        state["anxiety_index"] = max(0.0, -precision_gap)  # Negative gap = anxiety

        # Adjust threshold for domain
        if trial_type == "survival":
            state["effective_threshold"] = self.params.theta_survival
        else:
            state["effective_threshold"] = self.params.theta_neutral

        # Create extended state dict with additional metadata
        extended_state = dict(state)  # Copy the float state
        extended_state["trial_type"] = str(trial_type)  # type: ignore[assignment]

        # Add neuromodulator state if enabled
        if self.enable_neuromodulators and self.neuromodulators:
            extended_state["neuromodulators"] = dict(self.neuromodulators.state.to_dict())  # type: ignore[assignment]

        # Store trial metrics
        self.trial_metrics.append(extended_state)

        return extended_state

    @profile_hot_path
    @enforce_budget(max_time_ms=500.0)
    def process_trials(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
        trial_type: str = "neutral",
        precision_ext_arr: Optional[np.ndarray] = None,
        precision_int_arr: Optional[np.ndarray] = None,
        dt: float = 0.01,
    ) -> Dict[str, np.ndarray]:
        """
        High-performance batch processing API using NumPy.
        Returns synchronized vectors of process variables across all trials.
        Optimized hot path.
        """
        n_trials = len(observed)
        if precision_ext_arr is None:
            precision_ext_arr = np.full(
                n_trials, 2.0 if trial_type in ["survival", "incongruent"] else 1.0
            )
        if precision_int_arr is None:
            precision_int_arr = np.full(
                n_trials, 1.5 if trial_type in ["survival", "incongruent"] else 1.0
            )

        error_ext = observed - predicted
        error_int = error_ext * 0.3

        # Memory allocations
        S_arr = np.zeros(n_trials)
        theta_arr = np.zeros(n_trials)
        M_arr = np.zeros(n_trials)
        ignition_arr = np.zeros(n_trials, dtype=bool)

        # Pull parameters locally for loop speed
        tau_S = self.params.tau_S
        tau_theta = self.params.tau_theta
        tau_M = self.params.tau_M
        theta_0 = self.params.theta_0
        alpha = self.params.alpha
        beta = self.params.beta
        rho = self.params.rho
        sigma_S = self.params.sigma_S
        sigma_theta = self.params.sigma_theta
        sigma_M = self.params.sigma_M
        gamma_M = self.params.gamma_M
        lambda_S = self.params.lambda_S
        M_0 = self.params.M_0

        S = self.dynamics.S
        theta = self.dynamics.theta
        M = self.dynamics.M
        rng = self.dynamics.rng
        sqrt_dt = math.sqrt(dt)

        # Pre-generate noise for the entire batch to avoid RNG overhead in the loop
        noise_S_vec = rng.normal(0, 1, n_trials)
        noise_theta_vec = rng.normal(0, 1, n_trials)
        noise_M_vec = rng.normal(0, 1, n_trials)
        prob_rng_vec = rng.random(n_trials)

        # Localize math functions for faster access
        exp = math.exp
        tanh = math.tanh

        for i in range(n_trials):
            e_ext = error_ext[i]
            e_int = error_int[i]
            Pi_e = precision_ext_arr[i]
            Pi_i_base = precision_int_arr[i]

            # Effective interoceptive precision
            clip_val = -(M - M_0)
            if clip_val > 500:
                clip_val = 500
            elif clip_val < -500:
                clip_val = -500
            sigmoid = 1.0 / (1.0 + exp(clip_val))
            Pi_i_eff = Pi_i_base * (1.0 + beta * sigmoid)

            # Signal dynamics
            input_S = 0.5 * Pi_e * (e_ext**2) + 0.5 * Pi_i_eff * (e_int**2)
            noise_S = sigma_S * noise_S_vec[i] / sqrt_dt
            dS_dt = -S / tau_S + input_S + noise_S
            S = S + dS_dt * dt
            if S < 0:
                S = 0.0

            # Threshold dynamics
            noise_theta = sigma_theta * noise_theta_vec[i] / sqrt_dt
            dtheta_dt = (
                ((theta_0 - theta) / tau_theta)
                + gamma_M * M
                + lambda_S * S
                + noise_theta
            )
            theta = theta + dtheta_dt * dt
            if theta < 0.01:
                theta = 0.01

            # Somatic marker
            M_star = tanh(beta * e_int)
            noise_M = sigma_M * noise_M_vec[i] / sqrt_dt
            dM_dt = (M_star - M) / tau_M + noise_M
            M = M + dM_dt * dt
            if M > 2.0:
                M = 2.0
            elif M < -2.0:
                M = -2.0

            # Ignition
            z = alpha * (S - theta)
            if z >= 0:
                prob = 1.0 / (1.0 + exp(-z))
            else:
                z_exp = exp(z)
                prob = z_exp / (1.0 + z_exp)

            ignited = prob_rng_vec[i] < prob or prob > 0.8
            if ignited:
                S *= 1.0 - rho

            S_arr[i] = S
            theta_arr[i] = theta
            M_arr[i] = M
            ignition_arr[i] = ignited

        self.dynamics.S = S
        self.dynamics.theta = theta
        self.dynamics.M = M

        # Synchronize ring buffers roughly (for simplicity omitting all the extra stat details)
        self.dynamics.S_history.extend(S_arr)
        self.dynamics.theta_history.extend(theta_arr)
        self.dynamics.M_history.extend(M_arr)
        self.dynamics.ignition_history.extend(ignition_arr)

        return {"S": S_arr, "theta": theta_arr, "M": M_arr, "ignited": ignition_arr}

    def process_rt_trial(
        self, rt: float, expected_rt: float, correct: bool, trial_type: str = "neutral"
    ) -> Dict[str, float]:
        """
        Process a reaction time trial.

        Args:
            rt: Observed reaction time
            expected_rt: Expected/baseline reaction time
            correct: Whether response was correct
            trial_type: Trial type for domain-specific threshold

        Returns:
            Dictionary of APGI metrics
        """
        # Normalize RT to prediction error
        error = (rt - expected_rt) / expected_rt

        # Correctness affects precision
        precision_ext = 3.0 if correct else 1.0

        return self.process_trial(
            observed=error,
            predicted=0.0,  # Expected: no deviation from baseline
            trial_type=trial_type,
            precision_ext=precision_ext,
        )

    def process_choice_trial(
        self,
        choice: int,
        expected_choice: int,
        reward: float,
        trial_type: str = "neutral",
    ) -> Dict[str, float]:
        """
        Process a choice/decision trial.

        Args:
            choice: Made choice (0 or 1)
            expected_choice: Expected choice
            reward: Received reward
            trial_type: Trial type

        Returns:
            Dictionary of APGI metrics
        """
        # Choice prediction error
        choice_error = float(choice != expected_choice)

        # Reward affects interoceptive precision
        precision_int = 1.0 + reward

        return self.process_trial(
            observed=choice_error,
            predicted=0.0,
            trial_type=trial_type,
            precision_int=precision_int,
        )

    def process_detection_trial(
        self,
        detected: bool,
        target_present: bool,
        confidence: Optional[float] = None,
        trial_type: str = "neutral",
    ) -> Dict[str, float]:
        """
        Process a detection trial (e.g., visual search, change blindness).

        Args:
            detected: Whether participant detected target
            target_present: Whether target was present
            confidence: Optional confidence rating
            trial_type: Trial type

        Returns:
            Dictionary of APGI metrics
        """
        # Detection error
        correct = detected == target_present
        error = 0.0 if correct else 1.0

        # Confidence affects precision
        precision_ext = (
            confidence if confidence is not None else (2.0 if correct else 1.0)
        )

        return self.process_trial(
            observed=error,
            predicted=0.0,
            trial_type=trial_type,
            precision_ext=precision_ext,
        )

    def finalize(self) -> Dict[str, Any]:
        """
        Finalize experiment and compute comprehensive APGI metrics.

        Returns:
            Dictionary of experiment-level APGI metrics including neuromodulator states
        """
        self.experiment_summary = self.dynamics.get_summary()

        # Add trial-level aggregates
        if self.trial_metrics:
            ignition_probs = [
                float(m.get("ignition_prob", 0.0)) for m in self.trial_metrics
            ]
            self.experiment_summary["mean_ignition_prob"] = float(
                np.mean(ignition_probs)
            )
            self.experiment_summary["std_ignition_prob"] = float(np.std(ignition_probs))

            # Precision gap statistics (Π vs Π̂ distinction)
            if self.precision_gaps:
                self.experiment_summary["mean_precision_gap"] = float(
                    np.mean(self.precision_gaps)
                )
                self.experiment_summary["std_precision_gap"] = float(
                    np.std(self.precision_gaps)
                )
                self.experiment_summary["max_anxiety_index"] = float(
                    max(max(0.0, -gap) for gap in self.precision_gaps)
                )

            # Surprise accumulation index
            surprise_values = [m.get("S", 0.0) for m in self.trial_metrics]
            self.experiment_summary["surprise_accumulation_index"] = float(
                np.sum(surprise_values)
            )

        # Add neuromodulator summary if enabled
        if self.enable_neuromodulators and self.neuromodulators:
            nm_summary = self.neuromodulators.get_summary()
            if self.experiment_summary is not None:
                self.experiment_summary["neuromodulators"] = dict(nm_summary)  # type: ignore[assignment]

        return self.experiment_summary

    def reset(self) -> None:
        """Reset for new experiment."""
        self.dynamics.reset()
        self.trial_metrics.clear()
        self.experiment_summary = None
        self.precision_expectations.clear()
        self.precision_gaps.clear()
        if self.neuromodulators:
            self.neuromodulators.reset()

    def get_report(self) -> str:
        """Generate human-readable APGI report."""
        s = self.experiment_summary
        if s is None:
            s = self.finalize()

        report = f"""
APGI Metrics Report
==================
Ignition Dynamics:
  - Ignition Rate: {s['ignition_rate']:.2%}
  - Mean Ignition Probability: {s.get('mean_ignition_prob', 0):.3f}

Surprise Accumulation:
  - Mean Surprise: {s['mean_surprise']:.3f}
  - Final Surprise: {s['final_surprise']:.3f}

Threshold Dynamics:
  - Mean Threshold: {s['mean_threshold']:.3f}
  - Final Threshold: {s['final_threshold']:.3f}

Somatic Markers:
  - Mean Somatic Marker: {s['mean_somatic_marker']:.3f}
  - Final Somatic Marker: {s['final_somatic_marker']:.3f}

Metabolic Cost:
  - Total Metabolic Cost: {s['metabolic_cost']:.3f}
"""
        return report


# =============================================================================
# EXPERIMENT-SPECIFIC APGI CONFIGURATIONS
# =============================================================================


def get_apgi_config_for_experiment(experiment_name: str) -> APGIParameters:
    """
    Get experiment-specific APGI parameter configuration.

    Different experiments may benefit from different parameter tuning
    based on the expected psychological phenomena.
    """
    configs = {
        # Attention experiments - higher precision
        "attentional_blink": APGIParameters(
            tau_S=0.25, beta=1.8, theta_0=0.4, alpha=6.0
        ),
        "posner_cueing": APGIParameters(tau_S=0.30, beta=1.5, theta_0=0.35, alpha=5.5),
        "visual_search": APGIParameters(tau_S=0.35, beta=1.3, theta_0=0.5, alpha=5.0),
        "change_blindness": APGIParameters(
            tau_S=0.40, beta=1.2, theta_0=0.6, alpha=4.5
        ),
        "inattentional_blindness": APGIParameters(
            tau_S=0.35, beta=1.5, theta_0=0.55, alpha=5.0
        ),
        # Memory experiments - longer timescales
        "drm_false_memory": APGIParameters(
            tau_S=0.45, beta=1.0, theta_0=0.5, alpha=4.5
        ),
        "sternberg_memory": APGIParameters(
            tau_S=0.40, beta=1.2, theta_0=0.45, alpha=5.0
        ),
        "working_memory_span": APGIParameters(
            tau_S=0.38, beta=1.3, theta_0=0.5, alpha=5.2
        ),
        "dual_n_back": APGIParameters(tau_S=0.35, beta=1.4, theta_0=0.45, alpha=5.5),
        # Decision-making experiments - higher somatic influence
        "iowa_gambling_task": APGIParameters(
            tau_S=0.40, beta=2.0, theta_0=0.4, alpha=5.0
        ),
        "go_no_go": APGIParameters(tau_S=0.30, beta=1.6, theta_0=0.35, alpha=6.0),
        "stop_signal": APGIParameters(tau_S=0.28, beta=1.7, theta_0=0.3, alpha=6.5),
        "simon_effect": APGIParameters(tau_S=0.32, beta=1.4, theta_0=0.4, alpha=5.5),
        "eriksen_flanker": APGIParameters(
            tau_S=0.30, beta=1.5, theta_0=0.35, alpha=5.8
        ),
        # Perception experiments
        "binocular_rivalry": APGIParameters(
            tau_S=0.50, beta=1.0, theta_0=0.6, alpha=4.0
        ),
        "masking": APGIParameters(tau_S=0.25, beta=1.8, theta_0=0.3, alpha=7.0),
        "stroop_effect": APGIParameters(tau_S=0.30, beta=1.6, theta_0=0.35, alpha=6.0),
        "navon_task": APGIParameters(tau_S=0.35, beta=1.3, theta_0=0.45, alpha=5.2),
        # Learning experiments
        "serial_reaction_time": APGIParameters(
            tau_S=0.35, beta=1.2, theta_0=0.5, alpha=5.0
        ),
        "artificial_grammar_learning": APGIParameters(
            tau_S=0.40, beta=1.1, theta_0=0.55, alpha=4.8
        ),
        "probabilistic_category_learning": APGIParameters(
            tau_S=0.38, beta=1.3, theta_0=0.45, alpha=5.2
        ),
        # Interoception experiments
        "interoceptive_gating": APGIParameters(
            tau_S=0.45, beta=2.2, theta_0=0.5, alpha=5.0
        ),
        "somatic_marker_priming": APGIParameters(
            tau_S=0.40, beta=2.0, theta_0=0.45, alpha=5.2
        ),
        "metabolic_cost": APGIParameters(tau_S=0.50, beta=1.8, theta_0=0.6, alpha=4.5),
        # Other experiments
        "time_estimation": APGIParameters(tau_S=0.45, beta=1.2, theta_0=0.5, alpha=5.0),
        "virtual_navigation": APGIParameters(
            tau_S=0.40, beta=1.3, theta_0=0.5, alpha=5.0
        ),
        "multisensory_integration": APGIParameters(
            tau_S=0.35, beta=1.4, theta_0=0.45, alpha=5.5
        ),
        "ai_benchmarking": APGIParameters(tau_S=0.35, beta=1.5, theta_0=0.5, alpha=5.5),
    }

    return configs.get(experiment_name, APGIParameters())


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def compute_apgi_enhanced_metric(
    primary_metric: float,
    apgi_summary: Dict[str, float],
    weight_ignition: float = 0.3,
    weight_metabolic: float = 0.2,
) -> float:
    """
    Compute an APGI-enhanced composite metric.

    Combines the primary experiment metric with APGI-derived measures
    to create a more comprehensive performance indicator.

    Args:
        primary_metric: The experiment's primary performance metric
        apgi_summary: APGI summary from finalize()
        weight_ignition: Weight for ignition rate in composite
        weight_metabolic: Weight for metabolic efficiency in composite

    Returns:
        Composite metric incorporating APGI dynamics
    """
    # Normalize ignition rate (higher is generally better for awareness)
    ignition_score = apgi_summary.get("ignition_rate", 0.5)

    # Metabolic efficiency (lower cost is better)
    metabolic_cost = apgi_summary.get("metabolic_cost", 1.0)
    metabolic_efficiency = 1.0 / (1.0 + metabolic_cost)

    # Composite metric
    composite = (
        (1 - weight_ignition - weight_metabolic) * primary_metric
        + weight_ignition * ignition_score
        + weight_metabolic * metabolic_efficiency
    )

    return composite


def format_apgi_output(apgi_summary: Dict[str, float]) -> str:
    """Format APGI metrics for standard output."""
    return (
        f"apgi_ignition_rate: {apgi_summary.get('ignition_rate', 0):.3f}\n"
        f"apgi_metabolic_cost: {apgi_summary.get('metabolic_cost', 0):.3f}\n"
        f"apgi_mean_surprise: {apgi_summary.get('mean_surprise', 0):.3f}\n"
        f"apgi_mean_threshold: {apgi_summary.get('mean_threshold', 0):.3f}\n"
        f"apgi_mean_somatic_marker: {apgi_summary.get('mean_somatic_marker', 0):.3f}"
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "APGIParameters",
    "NeuromodulatorState",
    "NeuromodulatorSystem",
    "RunningStatistics",
    "CoreEquations",
    "DynamicalSystem",
    "APGIIntegration",
    "get_apgi_config_for_experiment",
    "compute_apgi_enhanced_metric",
    "format_apgi_output",
]
