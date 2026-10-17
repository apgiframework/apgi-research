"""
APGI Implementation
============================

Structured implementation of the APGI (Allostatic Precision-Gated Ignition)
dynamical system based on ultimate_apgi_template.py.

This template provides:
- Core dynamical components (generative models, signal processing, threshold dynamics)
- Stability enforcement and bounded dynamics
- Empirical mapping functions for EEG/behavioral proxies
- Full runner system integrating all components

Usage:
    from utils.apgi_implementation_template import APGIModel, CONFIG

    model = APGIModel()
    for t in range(1000):
        x = np.sin(t * 0.01) + np.random.randn() * 0.1
        out = model.step(x)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

CONFIG = {
    "dt": 0.01,
    "tau_theta": 20.0,
    "theta0": 0.5,
    "alpha": 5.0,
    "tau_S": 0.35,
    "tau_M": 1.5,
    "beta": 1.5,
    "beta_M": 1.0,
    "M_0": 0.0,
    "gamma_M": -0.3,
    "lambda_S": 0.1,
    "sigma_S": 0.05,
    "sigma_theta": 0.02,
    "sigma_M": 0.03,
    "rho": 0.7,
    "alpha_mu": 0.01,
    "alpha_sigma": 0.005,
}

# =============================================================================
# CORE: GENERATIVE MODEL (LEARNING)
# =============================================================================


class GenerativeModel:
    """
    Learns predictions x̂ via gradient descent on prediction error.

    Update rule: x̂_{t+1} = x̂_t + η * ε_t
    where ε_t = x(t) - x̂(t) is the prediction error.
    """

    def __init__(self, lr: float = 0.05):
        self.x_hat: float = 0.0
        self.lr: float = lr

    def predict(self) -> float:
        """Return current prediction."""
        return self.x_hat

    def update(self, x: float) -> float:
        """
        Update prediction based on observation.

        Args:
            x: Observed value

        Returns:
            epsilon: Prediction error
        """
        epsilon = x - self.x_hat
        self.x_hat += self.lr * epsilon
        return epsilon


# =============================================================================
# CORE: PREPROCESSING (RUNNING STATISTICS)
# =============================================================================


class RunningStatsEMA:
    """
    Exponential moving average for online mean and variance estimation.

    Updates:
        μ_{t+1} = μ_t + α_μ * (x - μ_t)
        σ²_{t+1} = σ²_t + α_σ * ((x - μ_t)² - σ²_t)
    """

    def __init__(self, alpha_mu: float = 0.01, alpha_sigma: float = 0.005):
        self.mu: float = 0.0
        self.var: float = 1.0
        self.alpha_mu: float = alpha_mu
        self.alpha_sigma: float = alpha_sigma

    def update(self, x: float) -> None:
        """Update running statistics with new observation."""
        self.mu += self.alpha_mu * (x - self.mu)
        self.var += self.alpha_sigma * ((x - self.mu) ** 2 - self.var)
        # Ensure positive variance
        self.var = max(self.var, 1e-8)

    def z(self, x: float) -> float:
        """Compute z-score: z = (x - μ) / σ"""
        return float((x - self.mu) / (np.sqrt(self.var) + 1e-8))


# =============================================================================
# CORE: PRECISION
# =============================================================================


def compute_precision(var: float) -> float:
    """
    Compute precision from variance: Π = 1/σ²
    """
    return float(1.0 / (var + 1e-8))


def effective_interoceptive_precision(
    pi_i: float, beta: float, M: float, M0: float = 0.0
) -> float:
    """
    Compute effective interoceptive precision with somatic modulation.

    Π^i_eff = Π^i_baseline · [1 + β·σ(M - M₀)]

    where σ is the sigmoid function.
    """
    sigmoid_M = float(1.0 / (1.0 + np.exp(-(M - M0))))
    return float(pi_i * (1.0 + beta * sigmoid_M))


# =============================================================================
# CORE: SIGNAL
# =============================================================================


def compute_signal(z_e: float, z_i: float, pi_e: float, pi_i_eff: float) -> float:
    """
    Compute accumulated signal S (energy-based formulation).

    S = ½Π^e(z^e)² + ½Π^i_eff(z^i)²

    This represents the weighted sum of squared prediction errors
    (exteroceptive and interoceptive).
    """
    return float(0.5 * pi_e * (z_e**2) + 0.5 * pi_i_eff * (z_i**2))


# =============================================================================
# CORE: THRESHOLD
# =============================================================================


def compute_information_value(z_e: float, z_i: float) -> float:
    """
    Free-energy proxy: expected surprise reduction.

    V_info = ½((z^e)² + (z^i)²)

    Higher values indicate more informative signals.
    """
    return float(0.5 * (z_e**2 + z_i**2))


def update_threshold(
    theta: float,
    theta0: float,
    S: float,
    V_info: float,
    dt: float,
    tau_theta: float,
    gamma_M: float = 0.0,
) -> float:
    """
    Stable threshold dynamics.

    dθ/dt = (θ0 - θ)/τ_θ - γ_M·S + η·(C - V_info)

    where:
        - C is metabolic cost proxy (0.1 * S)
        - η is learning rate (0.05)
        - V_info is information value

    Args:
        theta: Current threshold
        theta0: Baseline threshold
        S: Current signal
        V_info: Information value
        dt: Time step
        tau_theta: Threshold adaptation timescale
        gamma_M: Metabolic sensitivity

    Returns:
        Updated threshold
    """
    C = 0.1 * S  # metabolic proxy
    eta = 0.05

    dtheta = (theta0 - theta) / tau_theta - gamma_M * S + eta * (C - V_info)

    return theta + dtheta * dt


# =============================================================================
# CORE: IGNITION
# =============================================================================


def ignition_probability(S: float, theta: float, alpha: float = 5.0) -> float:
    """
    Compute ignition probability via sigmoid.

    P(ignite) = σ(α(S - θ)) = 1 / (1 + exp(-α(S - θ)))

    Args:
        S: Accumulated signal
        theta: Current threshold
        alpha: Sigmoid steepness

    Returns:
        Probability of ignition (0-1)
    """
    return float(1.0 / (1.0 + np.exp(-alpha * (S - theta))))


def ignite(S: float, theta: float) -> bool:
    """
    Determine if ignition occurs.

    Ignition occurs when S > θ (signal exceeds threshold).
    """
    return S > theta


# =============================================================================
# CORE: STABILITY
# =============================================================================


def clip(x: float, lo: float, hi: float) -> float:
    """Clip value to [lo, hi] range."""
    return max(lo, min(hi, x))


def enforce_stability(state: Dict[str, float]) -> Dict[str, float]:
    """
    Hard constraints for bounded dynamics.

    Enforces:
        - S ∈ [0, 10]
        - θ ∈ [0.1, 5.0]
        - Π_e ∈ [0.01, 10]
        - Π_i ∈ [0.01, 10]
    """
    state["S"] = clip(state["S"], 0.0, 10.0)
    state["theta"] = clip(state["theta"], 0.1, 5.0)
    state["Pi_e"] = clip(state["Pi_e"], 0.01, 10.0)
    state["Pi_i"] = clip(state["Pi_i"], 0.01, 10.0)
    return state


# =============================================================================
# EMPIRICAL: MAPPING
# =============================================================================


def map_to_p3b_latency(S: float) -> float:
    """
    Map signal S to P3b latency (ERP component).

    P3b latency ~ inverse signal strength
    Range: ~300ms (strong signal) to ~350ms (weak signal)
    """
    return float(300.0 - 50.0 * np.tanh(S))


def map_to_hep_amplitude(z_i: float, pi_i: float) -> float:
    """
    Map interoceptive precision and error to HEP amplitude.

    Heartbeat-evoked potential proxy.
    HEP ~ Π^i · |z^i|
    """
    return pi_i * abs(z_i)


def map_to_reaction_time(S: float, theta: float) -> float:
    """
    Map signal and threshold to reaction time.

    Decision latency model:
        RT = 800 / (1 + exp(S - θ))

    Faster responses when signal strongly exceeds threshold.
    """
    margin = S - theta
    return float(800.0 / (1.0 + np.exp(margin)))


# =============================================================================
# HIERARCHICAL 5-LEVEL PROCESSING
# =============================================================================


@dataclass
class HierarchicalLevel:
    """
    Single hierarchical level state (100/100 APGI Standard).

    Levels:
        1: Fast sensory (50-100ms)
        2: Feature integration (100-200ms)
        3: Pattern recognition (200-500ms)
        4: Semantic processing (500ms-2s)
        5: Executive control (2-10s)
    """

    S: float = 0.0  # Accumulated surprise
    theta: float = 0.5  # Threshold
    M: float = 0.0  # Somatic marker
    A: float = 0.5  # Arousal
    Pi_e: float = 1.0  # Exteroceptive precision
    Pi_i: float = 1.0  # Interoceptive precision
    ignition_prob: float = 0.0
    broadcast: bool = False
    tau: float = 0.1  # Level-specific timescale


class HierarchicalProcessor:
    """
    5-level hierarchical processor with cross-level coupling.

    Cross-level coupling: Π_{ℓ-1} ← Π_{ℓ-1} · (1 + β_cross · B_ℓ)
    where B_ℓ is the broadcast signal from level ℓ.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = config or CONFIG
        self.beta_cross: float = self.cfg.get("beta_cross", 0.2)

        # Initialize 5 hierarchical levels with timescales
        # From ultimate_apgi_template.py: [0.1, 0.2, 0.4, 1.0, 5.0]
        default_taus = [0.1, 0.2, 0.4, 1.0, 5.0]
        self.levels: List[HierarchicalLevel] = [
            HierarchicalLevel(tau=default_taus[i]) for i in range(5)
        ]

        # Level names for reference
        self.level_names = [
            "sensory",
            "feature_integration",
            "pattern_recognition",
            "semantic",
            "executive",
        ]

    def process_level(
        self,
        level_idx: int,
        input_signal: float,
        z_e: float,
        z_i: float,
        dt: float = 0.01,
    ) -> HierarchicalLevel:
        """
        Process a single hierarchical level.

        Args:
            level_idx: Index of level (0-4)
            input_signal: Input signal to this level
            z_e: Exteroceptive z-score
            z_i: Interoceptive z-score
            dt: Time step

        Returns:
            Updated level state
        """
        level = self.levels[level_idx]

        # Update accumulated signal with decay
        # dS/dt = -S/τ + input
        level.S += dt * (-level.S / level.tau + input_signal)

        # Compute precision for this level
        level.Pi_e = max(0.01, min(10.0, level.Pi_e))
        level.Pi_i = max(0.01, min(10.0, level.Pi_i))

        # Compute ignition probability for this level
        level.ignition_prob = float(
            1.0 / (1.0 + np.exp(-5.0 * (level.S - level.theta)))
        )

        # Determine if broadcast occurs
        level.broadcast = level.S > level.theta

        # Update threshold dynamics
        dtheta = (0.5 - level.theta) / 30.0 - 0.01 * level.S
        level.theta += dt * dtheta
        level.theta = max(0.1, min(5.0, level.theta))

        return level

    def apply_cross_level_coupling(self) -> None:
        """
        Apply cross-level precision coupling.

        Higher levels modulate lower-level precision:
            Π_{ℓ-1} ← Π_{ℓ-1} · (1 + β_cross · B_ℓ)
        """
        # Process from top (level 4) to bottom (level 1)
        for i in range(4, 0, -1):
            if self.levels[i].broadcast:
                # Higher level broadcasts -> increase lower level precision
                boost = 1.0 + self.beta_cross
                self.levels[i - 1].Pi_e *= boost
                self.levels[i - 1].Pi_i *= boost

                # Clip to stability bounds
                self.levels[i - 1].Pi_e = min(10.0, self.levels[i - 1].Pi_e)
                self.levels[i - 1].Pi_i = min(10.0, self.levels[i - 1].Pi_i)

    def process_all_levels(
        self,
        base_signal: float,
        z_e: float,
        z_i: float,
        dt: float = 0.01,
    ) -> List[HierarchicalLevel]:
        """
        Process all 5 hierarchical levels.

        Args:
            base_signal: Base input signal (from level 0 computation)
            z_e: Exteroceptive z-score
            z_i: Interoceptive z-score
            dt: Time step

        Returns:
            List of updated level states
        """
        # Process each level with decaying signal strength
        for i in range(5):
            # Signal attenuates at higher levels
            attenuation = 0.8**i
            level_input = base_signal * attenuation

            self.process_level(i, level_input, z_e, z_i, dt)

        # Apply cross-level coupling
        self.apply_cross_level_coupling()

        return self.levels

    def get_aggregate_signal(self) -> float:
        """
        Compute aggregate signal across all levels.

        Weighted sum favoring higher levels (executive control).
        """
        weights = [0.1, 0.15, 0.2, 0.25, 0.3]  # Higher levels weighted more
        total = sum(w * level.S for w, level in zip(weights, self.levels))
        return float(total)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of hierarchical processing."""
        return {
            f"level_{i}_{name}": {
                "S": level.S,
                "theta": level.theta,
                "ignition_prob": level.ignition_prob,
                "broadcast": level.broadcast,
                "Pi_e": level.Pi_e,
                "Pi_i": level.Pi_i,
            }
            for i, (name, level) in enumerate(zip(self.level_names, self.levels))
        }

    def reset(self) -> None:
        """Reset all levels to initial state."""
        default_taus = [0.1, 0.2, 0.4, 1.0, 5.0]
        self.levels = [HierarchicalLevel(tau=default_taus[i]) for i in range(5)]


# =============================================================================
# FULL SYSTEM: APGI MODEL (with hierarchical processing)
# =============================================================================


class APGIModel:
    """
    Full APGI dynamical system integrating all components.

    This class implements the complete APGI processing pipeline:
    1. Generative prediction and learning
    2. Running statistics (z-scores)
    3. Precision computation (exteroceptive + interoceptive)
    4. Signal accumulation (energy formulation)
    5. Information value computation
    6. Threshold dynamics
    7. Ignition detection
    8. Stability enforcement
    9. Empirical mapping (P3b, HEP, RT)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize APGI model.

        Args:
            config: Optional configuration dict (uses CONFIG default if None)
        """
        self.cfg = config or CONFIG

        # Generative model for exteroceptive predictions
        self.gen = GenerativeModel(lr=0.05)

        # Running statistics for z-score normalization
        self.stats_e = RunningStatsEMA(
            alpha_mu=self.cfg["alpha_mu"],
            alpha_sigma=self.cfg["alpha_sigma"],
        )
        self.stats_i = RunningStatsEMA(
            alpha_mu=self.cfg["alpha_mu"],
            alpha_sigma=self.cfg["alpha_sigma"],
        )

        # State variables
        self.theta: float = self.cfg["theta0"]
        self.S: float = 0.0
        self.M: float = 0.0

        # Parameters
        self.beta: float = self.cfg["beta"]
        self.beta_M: float = self.cfg["beta_M"]
        self.M_0: float = self.cfg["M_0"]
        self.gamma_M: float = self.cfg["gamma_M"]

        # Hierarchical 5-level processor
        self.hierarchical = HierarchicalProcessor(config=self.cfg)

        # History for tracking
        self.history: List[Dict[str, Any]] = []

    def step(self, x: float) -> Dict[str, Any]:
        """
        Process a single observation through the APGI pipeline.

        Args:
            x: Observed input value

        Returns:
            Dictionary containing all state variables and outputs
        """
        # 1. Prediction + learning
        x_hat = self.gen.predict()
        eps = self.gen.update(x)

        # 2. Interoceptive proxy (scaled exteroceptive error)
        eps_i = 0.3 * eps

        # 3. Update running statistics
        self.stats_e.update(eps)
        self.stats_i.update(eps_i)

        # Compute z-scores
        z_e = self.stats_e.z(eps)
        z_i = self.stats_i.z(eps_i)

        # 4. Compute precision
        pi_e = compute_precision(self.stats_e.var)
        pi_i = compute_precision(self.stats_i.var)
        pi_i_eff = effective_interoceptive_precision(pi_i, self.beta, self.M, self.M_0)

        # 5. Compute signal S (accumulated surprise)
        self.S = compute_signal(z_e, z_i, pi_e, pi_i_eff)

        # 6. Hierarchical 5-level processing
        self.hierarchical.process_all_levels(
            base_signal=self.S,
            z_e=z_e,
            z_i=z_i,
            dt=self.cfg["dt"],
        )

        # Get aggregate signal from hierarchy (weighted across levels)
        S_hierarchical = self.hierarchical.get_aggregate_signal()

        # 7. Compute information value (free-energy proxy)
        V = compute_information_value(z_e, z_i)

        # 8. Update threshold (using hierarchical aggregate)
        self.theta = update_threshold(
            self.theta,
            self.cfg["theta0"],
            S_hierarchical,
            V,
            self.cfg["dt"],
            self.cfg["tau_theta"],
            self.gamma_M,
        )

        # 9. Ignition detection (using hierarchical aggregate)
        p = ignition_probability(S_hierarchical, self.theta, self.cfg["alpha"])
        ignited = ignite(S_hierarchical, self.theta)

        # 10. Stability enforcement
        state = {
            "S": self.S,
            "theta": self.theta,
            "Pi_e": pi_e,
            "Pi_i": pi_i,
        }
        state = enforce_stability(state)
        self.S = state["S"]
        self.theta = state["theta"]

        # 10. Empirical mapping
        p3b = map_to_p3b_latency(self.S)
        hep = map_to_hep_amplitude(z_i, pi_i)
        rt = map_to_reaction_time(self.S, self.theta)

        # Get hierarchical summary
        hierarchical_summary = self.hierarchical.get_summary()

        # Compile output
        output = {
            **state,
            "z_e": z_e,
            "z_i": z_i,
            "x_hat": x_hat,
            "epsilon": eps,
            "ignition_prob": p,
            "ignited": ignited,
            "p3b_latency_ms": p3b,
            "hep_amplitude": hep,
            "reaction_time_ms": rt,
            "information_value": V,
            "pi_i_eff": pi_i_eff,
            "S_hierarchical": S_hierarchical,
            **hierarchical_summary,
        }

        self.history.append(output)
        return output

    def run(self, inputs: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run APGI on a sequence of inputs.

        Args:
            inputs: Array of input values

        Returns:
            List of output dictionaries (one per time step)
        """
        results = []
        for x in inputs:
            out = self.step(x)
            results.append(out)
        return results

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics from history.

        Returns:
            Dictionary with aggregate statistics
        """
        if not self.history:
            return {}

        S_values = [h["S"] for h in self.history]
        S_hier_values = [h.get("S_hierarchical", 0.0) for h in self.history]
        theta_values = [h["theta"] for h in self.history]
        ignition_count = sum(1 for h in self.history if h["ignited"])

        return {
            "mean_S": np.mean(S_values),
            "max_S": np.max(S_values),
            "mean_S_hierarchical": np.mean(S_hier_values),
            "max_S_hierarchical": np.max(S_hier_values),
            "mean_theta": np.mean(theta_values),
            "ignition_rate": ignition_count / len(self.history),
            "num_steps": len(self.history),
            "hierarchical_state": self.hierarchical.get_summary(),
        }

    def reset(self) -> None:
        """Reset model to initial state."""
        self.gen = GenerativeModel(lr=0.05)
        self.stats_e = RunningStatsEMA(
            alpha_mu=self.cfg["alpha_mu"],
            alpha_sigma=self.cfg["alpha_sigma"],
        )
        self.stats_i = RunningStatsEMA(
            alpha_mu=self.cfg["alpha_mu"],
            alpha_sigma=self.cfg["alpha_sigma"],
        )
        self.theta = self.cfg["theta0"]
        self.S = 0.0
        self.M = 0.0
        self.hierarchical.reset()
        self.history = []


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Create model
    model = APGIModel()

    # Generate synthetic input (sinusoid + noise)
    np.random.seed(42)
    t = np.arange(1000) * 0.01
    inputs = np.sin(t * 2 * np.pi) + np.random.randn(len(t)) * 0.1

    # Run model
    print("Running APGI model on synthetic data...")
    results = model.run(inputs)

    # Get summary
    summary = model.get_summary()

    print("\n" + "=" * 60)
    print("APGI MODEL SUMMARY")
    print("=" * 60)
    print(f"Steps processed: {summary['num_steps']}")
    print(f"Mean signal S: {summary['mean_S']:.3f}")
    print(f"Max signal S: {summary['max_S']:.3f}")
    print(f"Mean threshold θ: {summary['mean_theta']:.3f}")
    print(f"Ignition rate: {summary['ignition_rate']:.2%}")
    print("=" * 60)

    # Show final step details
    final = results[-1]
    print("\nFinal step state:")
    print(f"  Signal S: {final['S']:.3f}")
    print(f"  Threshold θ: {final['theta']:.3f}")
    print(f"  Ignition probability: {final['ignition_prob']:.3f}")
    print(f"  Ignited: {final['ignited']}")
    print(f"  P3b latency: {final['p3b_latency_ms']:.1f} ms")
    print(f"  HEP amplitude: {final['hep_amplitude']:.3f}")
    print(f"  Reaction time: {final['reaction_time_ms']:.1f} ms")
    print(f"  Precision (extero): {final['Pi_e']:.3f}")
    print(f"  Precision (intero): {final['Pi_i']:.3f}")
