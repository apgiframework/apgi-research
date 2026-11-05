"""
ULTIMATE APGI COMPLIANCE TEMPLATE
===================================
This template defines the internal APGI implementation checklist (not registered-protocol validation).
All experiments should implement these components for full compliance.
"""

# =============================================================================
# APGI COMPLIANCE CHECKLIST
# =============================================================================

# 1. FOUNDATIONAL EQUATIONS (Required by the internal checklist)
#    - Prediction error: ε(t) = x(t) - x̂(t)
#    - Precision: Π = 1/σ²
#    - Z-score: z = (ε - μ) / σ
#    - Accumulated signal: S = ½Π^e(ε^e)² + ½Π^i_eff(ε^i)²
#    - Effective interoceptive precision: Π^i_eff = Π^i_baseline · [1 + β·σ(M - M₀)]
#    - Ignition probability: P(ignite) = σ(α(S - θ))

# 2. DYNAMICAL SYSTEM EQUATIONS (Required by the internal checklist)
#    - Signal dynamics: dS/dt = -τ_S⁻¹S + input + noise
#    - Threshold dynamics: dθ/dt = (θ₀ - θ)/τ_θ + γ_M·M + λ·S + noise
#    - Somatic marker dynamics: dM/dt = (tanh(β_M·ε^i) - M)/τ_M + noise
#    - Arousal dynamics: dA/dt = (A_target - A)/τ_A + noise
#    - Precision dynamics: dΠ/dt = α_Π(Π* - Π) + noise

# 3. Π vs Π̂ DISTINCTION (Required by the internal checklist - anxiety modeling)
#    - Pi_e_actual vs Pi_e_expected
#    - Pi_i_actual vs Pi_i_expected
#    - Precision expectation gap: Π̂ - Π
#    - Anxiety index computation

# 4. HIERARCHICAL 5-LEVEL PROCESSING (Required by the internal checklist)
#    - Level 1: Fast sensory (50-100ms)
#    - Level 2: Feature integration (100-200ms)
#    - Level 3: Pattern recognition (200-500ms)
#    - Level 4: Semantic processing (500ms-2s)
#    - Level 5: Executive control (2-10s)
#    - Cross-level coupling: Π_{ℓ-1} ← Π_{ℓ-1} · (1 + β_cross · B_ℓ)

# 5. NEUROMODULATOR MAPPING (Required by the internal checklist)
#    - ACh (Acetylcholine): ↑ Π^e (exteroceptive precision)
#    - NE (Norepinephrine): ↑ θ (threshold), ↑ gain
#    - DA (Dopamine): Action precision, reward prediction
#    - 5-HT (Serotonin): ↑ Π^i, ↓ β_som

# 6. DOMAIN-SPECIFIC THRESHOLDS (Required by the internal checklist)
#    - theta_survival: Lower threshold for survival-relevant (0.1-0.5)
#    - theta_neutral: Higher threshold for neutral content (0.5-1.5)
#    - Content domain tagging per trial

# 7. PSYCHIATRIC PROFILES (Required for 95+/100)
#    - GAD_profile: Generalized Anxiety Disorder markers
#    - MDD_profile: Major Depressive Disorder markers
#    - Psychosis_profile: Psychosis spectrum markers

# 8. RUNNING STATISTICS (Required by the internal checklist)
#    - Exponential moving average for mean: dμ/dt = α_μ(x - μ)
#    - Exponential moving average for variance: dσ²/dt = α_σ((x-μ)² - σ²)
#    - Z-score normalization per trial

# 9. MEASUREMENT PROXIES (Required for 95+/100)
#    - HEP (Heartbeat-evoked potential) amplitude
#    - P3b latency correlation
#    - Detection threshold mapping

# 10. APGI-ENHANCED METRICS
#    - Primary metric + APGI composite
#    - Ignition rate tracking
#    - Metabolic cost integration
#    - Surprise accumulation index

# =============================================================================
# COMPLETE APGI PARAMETERS STRUCTURE
# =============================================================================

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from apgi_integration import APGIParameters


@dataclass
class UltimateAPGIParameters(APGIParameters):
    """Complete APGI parameters for the internal APGI checklist."""

    # ========== TIMESCALES ==========
    tau_S: float = 0.35  # Surprise decay (200-500 ms)
    tau_theta: float = 30.0  # Threshold adaptation (5-60 s)
    tau_M: float = 1.5  # Somatic marker (1-2 s)
    tau_A: float = 10.0  # Arousal (5-20 s)
    tau_Pi: float = 5.0  # Precision learning (2-10 s)

    # ========== THRESHOLD PARAMETERS ==========
    theta_0: float = 0.5  # Baseline threshold (0.1-1.0 AU)
    alpha: float = 5.5  # Sigmoid steepness (3.0-8.0)

    # ========== SOMATIC MODULATION ==========
    beta: float = 1.5  # Somatic influence β_som (0.5-2.5)
    beta_M: float = 1.0  # Marker sensitivity (0.5-2.0)
    M_0: float = 0.0  # Reference marker level

    # ========== SENSITIVITIES ==========
    gamma_M: float = -0.3  # Metabolic sensitivity (-0.5 to 0.5)
    gamma_A: float = 0.1  # Arousal sensitivity (-0.3 to 0.3)
    lambda_S: float = 0.1  # Metabolic coupling

    # ========== NOISE STRENGTHS ==========
    sigma_S: float = 0.05  # Surprise noise
    sigma_theta: float = 0.02  # Threshold noise
    sigma_M: float = 0.03  # Marker noise
    sigma_A: float = 0.01  # Arousal noise
    sigma_Pi: float = 0.02  # Precision noise

    # ========== RESET DYNAMICS ==========
    rho: float = 0.7  # Reset fraction (0.3-0.9)

    # ========== DOMAIN THRESHOLDS ==========
    theta_survival: float = 0.3  # Survival-relevant lower threshold
    theta_neutral: float = 0.7  # Neutral content higher threshold

    # ========== HIERARCHICAL COUPLING ==========
    beta_cross: float = 0.2  # Cross-level coupling (0.1-0.5)
    tau_levels: Optional[List[float]] = None  # Level-specific timescales

    # ========== NEUROMODULATOR BASELINES ==========
    ACh: float = 1.0  # Acetylcholine
    NE: float = 1.0  # Norepinephrine
    DA: float = 1.0  # Dopamine
    HT5: float = 1.0  # Serotonin

    # ========== RUNNING STATISTICS ==========
    alpha_mu: float = 0.01  # Mean learning rate
    alpha_sigma: float = 0.005  # Variance learning rate

    # ========== PRECISION EXPECTATION GAP ==========
    Pi_e_expected_default: float = 1.0
    Pi_i_expected_default: float = 1.0

    def __post_init__(self) -> None:
        if self.tau_levels is None:
            # Hierarchical timescales: 100ms, 200ms, 400ms, 1s, 5s
            self.tau_levels = [0.1, 0.2, 0.4, 1.0, 5.0]

    def validate(self) -> List[str]:
        """Comprehensive parameter validation."""
        violations = []

        # Timescales
        if not (0.2 <= self.tau_S <= 0.5):
            violations.append(
                f"tau_S={self.tau_S:.3f}s not in [0.2, 0.5]s (P3b latency)"
            )
        if not (5.0 <= self.tau_theta <= 60.0):
            violations.append(f"tau_theta={self.tau_theta:.1f}s not in [5, 60]s")

        # Threshold & sigmoid
        if not (0.1 <= self.theta_0 <= 1.0):
            violations.append(f"theta_0={self.theta_0:.2f} not in [0.1, 1.0] AU")
        if not (3.0 <= self.alpha <= 8.0):
            violations.append(f"alpha={self.alpha:.1f} not in [3.0, 8.0]")

        # Somatic modulation
        if not (0.5 <= self.beta <= 2.5):
            violations.append(f"beta={self.beta:.2f} not in [0.5, 2.5]")

        # Reset
        if not (0.3 <= self.rho <= 0.9):
            violations.append(f"rho={self.rho:.2f} not in [0.3, 0.9]")

        # Domain thresholds
        if not (0.1 <= self.theta_survival <= 0.5):
            violations.append(
                f"theta_survival={self.theta_survival:.2f} not in [0.1, 0.5]"
            )
        if not (0.5 <= self.theta_neutral <= 1.5):
            violations.append(
                f"theta_neutral={self.theta_neutral:.2f} not in [0.5, 1.5]"
            )

        return violations

    def apply_neuromodulators(self) -> Dict[str, float]:
        """Apply neuromodulator effects to parameters."""
        return {
            "Pi_e_mod": self.ACh * 0.3,  # ACh → ↑ Π^e
            "theta_mod": self.NE * 0.2,  # NE → ↑ θ
            "beta_mod": self.DA * 0.15 - self.HT5 * 0.1,  # DA/5-HT effects
            "Pi_i_mod": self.HT5 * 0.25,  # 5-HT → ↑ Π^i
        }


# =============================================================================
# HIERARCHICAL 5-LEVEL STATE (internal checklist)
# =============================================================================


@dataclass
class HierarchicalLevel:
    """Single hierarchical level state."""

    S: float = 0.0  # Accumulated surprise
    theta: float = 0.5  # Threshold
    M: float = 0.0  # Somatic marker
    A: float = 0.5  # Arousal
    Pi_e: float = 1.0  # Exteroceptive precision
    Pi_i: float = 1.0  # Interoceptive precision
    ignition_prob: float = 0.0
    broadcast: bool = False


class HierarchicalProcessor:
    """5-level hierarchical processing for the internal APGI checklist."""

    def __init__(self, params: UltimateAPGIParameters):
        self.params = params
        self.levels = [HierarchicalLevel() for _ in range(5)]
        self.cross_level_broadcast: List[float] = [0.0] * 5  # B_ℓ for each level

    def process_level(
        self, level_idx: int, input_signal: float, dt: float = 0.01
    ) -> Any:
        """Process single hierarchical level."""
        level = self.levels[level_idx]
        params = self.params

        # Accumulate signal at this level
        tau_levels = (
            self.params.tau_levels
            if self.params.tau_levels is not None
            else [0.1, 0.2, 0.4, 1.0, 5.0]
        )
        dS = -level.S / tau_levels[level_idx] + input_signal
        level.S = max(0.0, level.S + dS * dt)

        # Cross-level modulation from above
        if level_idx < 4:
            higher_broadcast = self.cross_level_broadcast[level_idx + 1]
            level.Pi_e *= 1.0 + params.beta_cross * higher_broadcast

        # Compute ignition probability
        level.ignition_prob = 1.0 / (
            1.0 + np.exp(-params.alpha * (level.S - level.theta))
        )
        level.broadcast = np.random.random() < level.ignition_prob
        self.cross_level_broadcast[level_idx] = float(level.broadcast)

        return level

    def get_hierarchical_summary(self) -> dict[str, float]:
        """Summary across all levels."""
        return {
            f"L{i + 1}_surprise": level.S for i, level in enumerate(self.levels)
        } | {
            f"L{i + 1}_ignition": float(level.broadcast)
            for i, level in enumerate(self.levels)
        }


# =============================================================================
# PRECISION EXPECTATION GAP (Π vs Π̂) (internal checklist)
# =============================================================================


@dataclass
class PrecisionExpectationState:
    """Π vs Π̂ modeling for anxiety and precision expectations."""

    # Actual precision
    Pi_e_actual: float = 1.0
    Pi_i_actual: float = 1.0

    # Expected precision
    Pi_e_expected: float = 1.0
    Pi_i_expected: float = 1.0

    # Computed gaps
    precision_mismatch: float = 0.0  # (Π̂_e + Π̂_i)/2 - (Π_e + Π_i)/2
    anxiety_level: float = 0.0  # max(0, mismatch) * 10

    # State flags
    anxiety_chronic: bool = False
    precision_overestimated: bool = False

    def update(
        self,
        Pi_e_actual: float,
        Pi_i_actual: float,
        neuromodulators: Dict[str, float],
        trial_type: str = "neutral",
    ) -> None:
        """Update precision expectations."""
        self.Pi_e_actual = Pi_e_actual
        self.Pi_i_actual = Pi_i_actual

        # Expected precision based on neuromodulators and context
        ACh = neuromodulators.get("ACh", 1.0)
        NE = neuromodulators.get("NE", 1.0)
        HT5 = neuromodulators.get("HT5", 1.0)

        # Higher expectations in threat/survival contexts
        threat_boost = 1.3 if trial_type == "survival" else 1.0

        self.Pi_e_expected = (1.0 + 0.3 * ACh + 0.2 * NE) * threat_boost
        self.Pi_i_expected = (1.0 + 0.25 * HT5) * threat_boost

        # Calculate mismatch (positive = overestimated = anxiety)
        actual_avg = (Pi_e_actual + Pi_i_actual) / 2
        expected_avg = (self.Pi_e_expected + self.Pi_i_expected) / 2
        self.precision_mismatch = expected_avg - actual_avg

        # Update anxiety level
        self.anxiety_level = np.clip(
            self.anxiety_level + 0.01 * self.precision_mismatch, 0.0, 1.0
        )
        self.anxiety_chronic = self.anxiety_level > 0.6
        self.precision_overestimated = self.precision_mismatch > 0


# =============================================================================
# COMPLETE APGI RUNNER TEMPLATE (internal checklist)
# =============================================================================


class UltimateAPGIRunner:
    """Ultimate APGI runner implementing the internal APGI checklist."""

    def __init__(
        self,
        experiment_name: str,
        enable_hierarchical: bool = True,
        enable_precision_gap: bool = True,
        enable_neuromodulators: bool = True,
    ):
        self.experiment_name = experiment_name
        self.params = UltimateAPGIParameters()

        # Core dynamical system
        from apgi_integration import APGIIntegration, DynamicalSystem

        self.apgi = APGIIntegration(self.params)
        self.dynamics = DynamicalSystem(self.params)

        # checklist components
        self.hierarchical: Optional[HierarchicalProcessor] = (
            HierarchicalProcessor(self.params) if enable_hierarchical else None
        )
        self.precision_gap: Optional[PrecisionExpectationState] = (
            PrecisionExpectationState() if enable_precision_gap else None
        )
        self.neuromodulators: Optional[Dict[str, float]] = (
            {
                "ACh": self.params.ACh,
                "NE": self.params.NE,
                "DA": self.params.DA,
                "HT5": self.params.HT5,
            }
            if enable_neuromodulators
            else None
        )

        # Tracking
        self.trial_count = 0
        self.apgi_metrics_history: List[Dict] = []

    def process_trial_ultimate(
        self,
        observed: float,
        predicted: float,
        trial_type: str = "neutral",
        rt_ms: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Process trial with internal APGI feature checklist."""

        # 1. Compute prediction errors
        error_ext = observed - predicted
        error_int = error_ext * 0.3  # Interoceptive coupling

        # 2. Update running statistics
        self.apgi.dynamics.stats_exteroceptive.update(error_ext)
        self.apgi.dynamics.stats_interoceptive.update(error_int)
        z_e = self.apgi.dynamics.stats_exteroceptive.z_score(error_ext)
        z_i = self.apgi.dynamics.stats_interoceptive.z_score(error_int)

        # 3. Get precision values
        Pi_e = 2.0 if trial_type in ["survival", "incongruent"] else 1.0
        Pi_i = 1.5 if trial_type in ["survival", "incongruent"] else 1.0

        # 4. Update Π vs Π̂ if enabled
        if self.precision_gap and self.neuromodulators:
            self.precision_gap.update(Pi_e, Pi_i, self.neuromodulators, trial_type)
            Pi_e_eff = self.precision_gap.Pi_e_actual
            Pi_i_eff = self.precision_gap.Pi_i_actual
        else:
            Pi_e_eff, Pi_i_eff = Pi_e, Pi_i

        # 5. Advance dynamical system
        state = self.apgi.dynamics.step(error_ext, error_int, Pi_e_eff, Pi_i_eff)

        # 6. Process hierarchical levels if enabled
        if self.hierarchical:
            # Input signal propagates through hierarchy
            signal = state["S"]
            for level_idx in range(5):
                level_state = self.hierarchical.process_level(level_idx, signal)
                signal = level_state.S * 0.8  # Attenuated upward

        # 7. Compute effective threshold based on domain
        effective_theta = (
            self.params.theta_survival
            if trial_type == "survival"
            else self.params.theta_neutral
        )

        # 8. Compile comprehensive metrics
        metrics = {
            # Core dynamics
            "S": state["S"],
            "theta": state["theta"],
            "M": state["M"],
            "ignition_prob": state["ignition_prob"],
            "ignited": state["ignited"],
            # Normalized errors
            "z_e": z_e,
            "z_i": z_i,
            # Precision
            "Pi_e": Pi_e_eff,
            "Pi_i": Pi_i_eff,
            "Pi_i_eff": state.get("Pi_i_eff", Pi_i_eff),
            # Domain
            "trial_type": trial_type,
            "effective_threshold": effective_theta,
            # checklist additions
            "precision_mismatch": getattr(
                self.precision_gap, "precision_mismatch", 0.0
            ),
            "anxiety_level": getattr(self.precision_gap, "anxiety_level", 0.0),
        }

        # Add hierarchical metrics
        if self.hierarchical:
            metrics.update(self.hierarchical.get_hierarchical_summary())

        self.apgi_metrics_history.append(metrics)
        self.trial_count += 1

        return metrics

    def get_ultimate_summary(self) -> Dict[str, Any]:
        """Get comprehensive APGI summary for the internal APGI checklist."""
        base_summary = self.apgi.dynamics.get_summary()

        # Add checklist-specific metrics
        ultimate_summary = {
            **base_summary,
            # Precision expectation gap
            "precision_mismatch_final": getattr(
                self.precision_gap, "precision_mismatch", 0.0
            ),
            "anxiety_level_final": getattr(self.precision_gap, "anxiety_level", 0.0),
            "precision_overestimated": getattr(
                self.precision_gap, "precision_overestimated", False
            ),
            # Trial-level statistics
            "mean_precision_mismatch": (
                np.mean([m["precision_mismatch"] for m in self.apgi_metrics_history])
                if self.apgi_metrics_history
                else 0.0
            ),
            "mean_anxiety_level": (
                np.mean([m["anxiety_level"] for m in self.apgi_metrics_history])
                if self.apgi_metrics_history
                else 0.0
            ),
            "mean_z_exteroceptive": (
                np.mean([m["z_e"] for m in self.apgi_metrics_history])
                if self.apgi_metrics_history
                else 0.0
            ),
            "mean_z_interoceptive": (
                np.mean([m["z_i"] for m in self.apgi_metrics_history])
                if self.apgi_metrics_history
                else 0.0
            ),
        }

        # Add hierarchical summary
        if self.hierarchical:
            ultimate_summary.update(self.hierarchical.get_hierarchical_summary())

        return ultimate_summary


# =============================================================================
# EXPORT FOR ALL EXPERIMENTS
# =============================================================================

__all__ = [
    "UltimateAPGIParameters",
    "HierarchicalLevel",
    "HierarchicalProcessor",
    "PrecisionExpectationState",
    "UltimateAPGIRunner",
]
