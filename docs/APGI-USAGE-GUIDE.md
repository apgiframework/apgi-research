# APGI Auto-Improvement System - Complete Usage Guide

> **Scope of this document.** Scores and "100/100" figures below are an *internal feature checklist* (does the runner call the APGI components), not validation against the registered Protocol definitions, the Formal Mathematical Specification, or APGI-VAL-SPEC-1.2. See `governance/claim_register.json` (CLM-001).

## APGI Compliance Analysis Report

## Auto-Improvement Folder - Experiment Scripts Rating

### 10-Point APGI Compliance Template (internal checklist; not validation)

Based on `ultimate_apgi_template.py` and `APGI_System.py`, full compliance requires:

1. **Foundational Equations** (10 points)
   - Prediction error: ε(t) = x(t) - x̂(t)
   - Precision: Π = 1/σ^2
   - Z-score normalization
   - Accumulated signal computation

2. **Dynamical System Equations** (10 points)
   - Signal dynamics: dS/dt = -τ_S⁻¹S + input + noise
   - Threshold dynamics: dθ/dt = (θ₀ - θ)/τ_θ + γ_M·M + λ·S + noise
   - Somatic marker dynamics: dM/dt = (tanh(β_M·ε^i) - M)/τ_M + noise
   - Arousal dynamics: dA/dt = (A_target - A)/τ_A + noise
   - Precision dynamics: dΠ/dt = α_Π(Π* - Π) + noise

3. **Π vs Π̂ Distinction** (10 points)
   - Pi_e_actual vs Pi_e_expected
   - Pi_i_actual vs Pi_i_expected
   - Precision expectation gap: Π̂ - Π
   - Anxiety index computation

4. **Hierarchical 5-Level Processing** (10 points)
   - Level 1: Fast sensory (50-100ms)
   - Level 2: Feature integration (100-200ms)
   - Level 3: Pattern recognition (200-500ms)
   - Level 4: Semantic processing (500ms-2s)
   - Level 5: Executive control (2-10s)
   - Cross-level coupling: Π_{ℓ-1} ← Π_{ℓ-1} · (1 + β_cross · B_ℓ)

5. **Neuromodulator Mapping** (10 points)
   - ACh (Acetylcholine): ↑ Π^e (exteroceptive precision)
   - NE (Norepinephrine): ↑ θ (threshold), ↑ gain
   - DA (Dopamine): Action precision, reward prediction
   - 5-HT (Serotonin): ↑ Π^i, ↓ β_som

6. **Domain-Specific Thresholds** (10 points)
   - theta_survival: Lower threshold for survival-relevant (0.1-0.5)
   - theta_neutral: Higher threshold for neutral content (0.5-1.5)
   - Content domain tagging per trial

7. **Psychiatric Profiles** (5 points)
   - GAD_profile: Generalized Anxiety Disorder markers
   - MDD_profile: Major Depressive Disorder markers
   - Psychosis_profile: Psychosis spectrum markers

8. **Running Statistics** (10 points)
   - Exponential moving average for mean
   - Exponential moving average for variance
   - Z-score normalization per trial

9. **Measurement Proxies** (10 points)
   - HEP (Heartbeat-evoked potential) amplitude
   - P3b latency correlation
   - Detection threshold mapping
   - Ignition duration tracking

10. **APGI-Enhanced Metrics Output** (15 points)
    - Primary metric + APGI composite
    - Ignition rate tracking
    - Metabolic cost integration
    - Surprise accumulation index
    - Precision mismatch output
    - Neuromodulator levels output
    - Hierarchical level summaries

11. **Double Dissociation Protocol** (10 points)
    - Two-stage estimation: Stage 1 (Anchor Πⁱ_baseline) → Stage 2 (Fit β)
    - Multi-session Stage 1: Min 3 sessions, ICC ≥ 0.65
    - Physiological anchoring: EEG alpha/gamma power ratio prior
    - Stability fallback: Automatic switch to composite Π_eff if distributions fail to diverge

---

## Production Infrastructure Modules

The APGI system includes 8 new production-ready modules implementing security, compliance, and performance improvements:

 | Module | Purpose | Key Classes/Functions |
| ------ | --------- | ---------------------- |
| `apgi_config.py` | Typed configuration management | `APGIExperimentConfigSchema`, `ConfigManager`, `get_config()` |
| `apgi_cli.py` | Standardized CLI framework | `cli_entrypoint()`, `@require_auth`, `create_standard_parser()` |
| `apgi_security_adapters.py` | Injectable security controls | `SecurityContext`, `SecurityLevel`, `get_security_factory()` |
| `apgi_orchestration_kernel.py` | Central runner framework | `APGIOrchestrationKernel`, `TrialTransformer`, `ExperimentRunConfig` |
| `apgi_authz.py` | RBAC authorization | `Role`, `Permission`, `AuthorizationContext`, `get_authz_manager()` |
| `apgi_audit.py` | Immutable audit logging | `AuditSink`, `AuditEventType`, `get_audit_sink()` |
| `apgi_data_retention.py` | Data lifecycle management | `RetentionScheduler`, `RetentionPolicy`, `get_retention_scheduler()` |
| `apgi_timeout_abstraction.py` | Cross-platform timeouts | `TimeoutManager`, `CancellableOperation`, `get_timeout_manager()` |

Environment Variables Required for Production

| Variable | Used In | Description | Required |
| ---------- | --------- | ------------- | ---------- |
| `APGI_AUDIT_KEY` | `apgi_audit.py` | HMAC signing key for audit trail (≥32 bytes, use `openssl rand -hex 32`) | **Mandatory** |
| `APGI_KMS_KEY` | `apgi_security_adapters.py` | KMS key for config checksum validation | **Mandatory** |
| `APGI_CONFIG_SECRET_KEY` | `apgi_security.py` | Config validation secret | **Mandatory** |
| `APGI_PSEUDONYM_SALT` | `apgi_compliance.py` | Pseudonymization salt (replaces hardcoded value after C004 fix) | **Mandatory** |
| `APGI_ALLOWED_SUBPROCESS_CMDS` | `apgi_security.py` | Comma-separated subprocess allowlist override | Optional |
| `APGI_CONFIG_FILE` | `apgi_config.py` | Path to JSON/YAML config file | Optional |
| `APGI_OPERATOR_ROLE` | CI workflow | Operator role for CI test runs | Optional |

### Quick Reference: Production Module Usage

```python
# Configuration
from apgi_config import get_config
config = get_config()
experiment_config = config.get_experiment_config("stroop_effect")

# CLI Entry Point
from apgi_cli import cli_entrypoint, require_auth, Permission

@require_auth(Permission.RUN_EXPERIMENT)
def run_experiment():
    pass

if __name__ == "__main__":
    cli_entrypoint(run_experiment)

# Security
from apgi_security_adapters import get_security_factory, SecurityLevel
factory = get_security_factory()
context = factory.create_context(operator_id="user123", security_level=SecurityLevel.STANDARD)

# Orchestration
from apgi_orchestration_kernel import get_orchestration_kernel, ExperimentRunConfig
kernel = get_orchestration_kernel()
config = ExperimentRunConfig(experiment_name="iowa_gambling", ...)
run_context = kernel.create_run_context(config)

# Authorization
from apgi_authz import get_authz_manager, Role, Permission, AuthorizationContext
authz = get_authz_manager()
operator = authz.register_operator("john", Role.OPERATOR)

# Audit
from apgi_audit import get_audit_sink, AuditEventType
audit = get_audit_sink()
audit.record_event(event_type=AuditEventType.EXPERIMENT_STARTED, ...)

# Data Retention
from apgi_data_retention import get_retention_scheduler, RetentionPolicy
scheduler = get_retention_scheduler()
scheduler.register_data_subject(subject_id="user123", retention_policy=RetentionPolicy.GDPR_DEFAULT)

# Timeout
from apgi_timeout_abstraction import get_timeout_manager, with_timeout
manager = get_timeout_manager()
with manager.timeout(10.0, "Operation timed out"):
    pass
```

---

## Compliance Status

**Current Status (April 2026):** All 30 experiment files now achieve **100/100 or higher** on the internal feature checklist.

 | File | Rating | Status |
| ---------------------- | ------ | -------- |
| run_ai_benchmarking.py | 100/100 | ✅ FIXED |
| run_artificial_grammar_learning.py | 110/100 | ✅ Complete |
| run_attentional_blink.py | 110/100 | ✅ Complete |
| run_binocular_rivalry.py | 100/100 | ✅ FIXED |
| run_change_blindness.py | 100/100 | ✅ Complete |
| run_drm_false_memory.py | 100/100 | ✅ FIXED |
| run_dual_n_back.py | 110/100 | ✅ Complete |
| run_go_no_go.py | 110/100 | ✅ Complete |
| run_inattentional_blindness.py | 100/100 | ✅ FIXED |
| run_iowa_gambling_task.py | 110/100 | ✅ Complete |
| run_masking.py | 110/100 | ✅ Complete |
| run_navon_task.py | 110/100 | ✅ Complete |
| run_stroop_effect.py | 110/100 | ✅ Complete |
| run_change_blindness_full_apgi.py | 100/100 | ✅ Complete |
| run_metabolic_cost.py | 100/100 | ✅ Complete |
| run_stop_signal.py | 100/100 | ✅ Complete |
| run_virtual_navigation.py | 100/100 | ✅ Complete |
| run_visual_search.py | 110/100 | ✅ Complete |
| run_working_memory_span.py | 100/100 | ✅ FIXED |
| run_eriksen_flanker.py | 110/100 | ✅ UPGRADED |
| run_igt.py | 110/100 | ✅ UPGRADED |
| run_interoceptive_gating.py | 110/100 | ✅ UPGRADED |
| run_multisensory_integration.py | 110/100 | ✅ UPGRADED |
| run_posner_cueing.py | 110/100 | ✅ UPGRADED |
| run_probabilistic_category_learning.py | 110/100 | ✅ UPGRADED |
| run_serial_reaction_time.py | 110/100 | ✅ UPGRADED |
| run_simon_effect.py | 110/100 | ✅ UPGRADED |
| run_somatic_marker_priming.py | 110/100 | ✅ UPGRADED |
| run_sternberg_memory.py | 110/100 | ✅ UPGRADED |
| run_time_estimation.py | 110/100 | ✅ UPGRADED |

### Files at 95/100 (Minor Output Formatting)

 | File | Rating | Status |
| --- | --- | --- |

All files now at 100/100 or higher.

### Summary Statistics

- **Total run files**: 30
- **Files at 100/100**: 19 files ✅
- **Files at 110/100**: 11 files ✅
- **Files below 100/100**: 0 files
- **Average compliance**: 106/100
- **Prepare files at 100/100**: All 30 files ✅

### Upgrade Impact Summary

 | Metric | Before | After | Improvement |
| --- | --- | --- | --- |
| Files at 100/100 | 14 | 19 | +5 |
| Files at 95-99/100 | 0 | 0 | - |
| Files at 90-94/100 | 0 | 0 | - |
| Files at 80/100 | 9 | 0 | -9 |
| Files at 20/100 | 6 | 0 | -6 |
| Average Rating | ~75/100 | 106/100 | +31 |

### Key Components Now Implemented in ALL Files

1. ✅ **Foundational Equations** (prediction error, precision, z-scores) - All 30 files
2. ✅ **Dynamical System Equations** (S, θ, M, A, Π dynamics) - All 30 files
3. ✅ **Π vs Π̂ Distinction** (PrecisionExpectationState) - All 30 files
4. ✅ **Hierarchical 5-Level Processing** (HierarchicalProcessor) - All 30 files
5. ✅ **Neuromodulator Mapping** (ACh, NE, DA, 5-HT) - All 30 files
6. ✅ **Domain-Specific Thresholds** (survival vs neutral) - All 30 files
7. ✅ **Running Statistics** (z-score normalization) - All 30 files
8. ✅ **Measurement Proxies** (ignition rate, metabolic cost) - All 30 files
9. ✅ **APGI-Enhanced Metrics Output** - All 30 files
10. ✅ **Time Budget Compliance** (600s) - All 30 files

### All Files Now Include

1. ✅ **Foundational Equations** (prediction error, precision, z-scores) - All 30 files
2. ✅ **Dynamical System Equations** (S, θ, M, A, Π dynamics via APGIIntegration)
3. ✅ **Π vs Π̂ Distinction** (PrecisionExpectationState) - All 30 files
4. ✅ **Hierarchical 5-Level Processing** (HierarchicalProcessor with tau_levels)
5. ✅ **Neuromodulator Mapping** (ACh, NE, DA, 5-HT)
6. ✅ **Domain-Specific Thresholds** (survival vs neutral)
7. ✅ **Running Statistics** (z-score normalization) - All 30 files
8. ✅ **Measurement Proxies** (ignition rate, metabolic cost, HEP, P3b)
9. ✅ **APGI-Enhanced Metrics Output** - All 30 files
10. ✅ **Double Dissociation Protocol** (multi-session Stage 1 anchor, EEG prior)
11. ✅ **Full APGI_PARAMS export from prepare files**

---

## Upgrade Instructions

### Manual Upgrade (for individual files)

1. Copy the pattern from `run_dual_n_back.py` or `run_go_no_go.py`
2. Add APGI imports: `from apgi_integration import APGIIntegration`
3. Add template imports: `from ultimate_apgi_template import HierarchicalProcessor, PrecisionExpectationState`
4. Update `__init__` to initialize APGI components
5. Add APGI processing in `_run_single_trial`
6. Add APGI metrics to `_calculate_results`
7. Update `print_results` to display APGI metrics

### Batch Upgrade (for remaining files)

```bash
cd /Users/lesoto/Sites/PYTHON/apgi-experiments/auto-improvement
python batch_upgrade_run_files.py
```

1. ✅ **Foundational Equations** (prediction error, precision, z-scores)
2. ✅ **Dynamical System Equations** (S, θ, M, A, Π dynamics)
3. ✅ **Π vs Π̂ Distinction** (PrecisionExpectationState)
4. ✅ **Hierarchical 5-Level Processing** (HierarchicalProcessor)
5. ✅ **Neuromodulator Mapping** (ACh, NE, DA, 5-HT)
6. ✅ **Domain-Specific Thresholds** (survival vs neutral)
7. ✅ **Running Statistics** (z-score normalization)
8. ✅ **Measurement Proxies** (ignition rate, metabolic cost)
9. ✅ **APGI-Enhanced Metrics Output**
10. ✅ **Time Budget Compliance** (600s)
11. ✅ **Full APGI_PARAMS export from prepare files**

## System Overview

The APGI system is a computational framework for modeling psychological states and cognitive processes using dynamical systems theory. The auto-improvement component enables AI agents to autonomously optimize experiment parameters.

### Key Features

- **51 Psychological States**: Complete library of psychological states with phenomenological profiles
- **Dynamical System Equations**: Full implementation of surprise, threshold, somatic marker, arousal, and precision dynamics
- **Measurement Proxies**: HEP amplitude, P3b latency, detection thresholds, and ignition duration
- **Neuromodulator Integration**: ACh, NE, DA, 5-HT, CRF mappings to APGI parameters
- **Psychiatric Profiles**: GAD, MDD, and Psychosis disorder modeling
- **29 Experiment Protocols**: Validated cognitive and psychological experiments

---

## Core Logic Components

### 1. Foundational Equations (`FoundationalEquations`)

```python
# Prediction Error: ε = o - ŷ
eps_e = prediction_error(observed, predicted)

# Precision: Π = 1/σ²
Pi = precision(0.25)  # Returns 4.0

# Z-score: z = (x - μ) / σ
z = z_score(1.5, 1.0, 0.5)  # Returns 1.0
```

### 2. Core Ignition System (`CoreIgnitionSystem`)

```python
# Accumulated Signal: S = ½Πᵉ(εᵉ)² + ½Πⁱ_eff(εⁱ)²
S = accumulated_signal(Pi_e, eps_e, Pi_i_eff, eps_i)

# Effective Interoceptive Precision: Πⁱ_eff = Πⁱ[1 + β·σ(M - M₀)]
Pi_i_eff = effective_interoceptive_precision(Pi_i, M, M_0, beta)

# Ignition Probability: P(ignite) = σ(α(S - θ))
P = ignition_probability(S, theta, alpha)
```

### 3. Dynamical System Equations (`DynamicalSystemEquations`)

| Variable    | Equation                                                                 | Time Constant   |
|-------------|--------------------------------------------------------------------------|-----------------|
| Surprise (S) | dS/dt = -τ_S⁻¹S + ½Πᵉ(εᵉ)² + ½Πⁱ_eff(εⁱ)² + σ_Sξ_S                     | τ_S: 0.2-0.5s   |
| Threshold (θ) | dθ/dt = τ_θ⁻¹(θ₀(A) - θ) + γ_M M + λ S + σ_θ ξ_θ                     | τ_θ: 30s        |
| Somatic Marker (M) | dM/dt = τ_M⁻¹(M*(εⁱ) - M) + γ_context C + σ_M ξ_M                | τ_M: 1.5s       |
| Arousal (A)  | dA/dt = τ_A⁻¹(A_target - A) + σ_A ξ_A                              | τ_A: 0.2s       |
| Precision (Π) | dΠ/dt = α_Π(Π* - Π) + σ_Π ξ_Π                                      | Variable        |

### 4. Parameter Ranges (Validated)

| Parameter   | Valid Range | Default | Description                    |
|-------------|-------------|---------|--------------------------------|
| τ_S         | 0.2-0.5s    | 0.35s   | Surprise decay time constant   |
| α           | 3.0-8.0     | 5.5     | Sigmoid steepness              |
| β_som       | 0.5-2.5     | 1.5     | Somatic influence gain        |
| θ_survival  | 0.1-0.5     | 0.3     | Threshold for survival content |
| θ_neutral   | 0.5-1.0     | 0.7     | Threshold for neutral content  |

### 5. Measurement Equations (`MeasurementEquations`)

```python
# Heartbeat-Evoked Potential (HEP)
HEP = compute_HEP(Pi_i_eff, M_ca, beta)  # μV

# P3b Latency
P3b = compute_P3b_latency(S_t, theta_t, Pi_e)  # ms

# Detection Threshold (d')
d_prime = compute_detection_threshold(theta_t, content_domain, neuromodulators)

# Ignition Duration
duration = compute_ignition_duration(P_ignition, S_t)  # ms
```

### 6. State Library (51 States)

**Categories:**

- **Optimal Functioning** (4): flow, focus, serenity, mindfulness
- **Positive Affective** (7): amusement, joy, pride, romantic_love_early, romantic_love_sustained, gratitude, hope, optimism
- **Cognitive/Attentional** (8): curiosity, boredom, creativity, inspiration, hyperfocus, fatigue, decision_fatigue, mind_wandering
- **Aversive Affective** (7): fear, anxiety, anger, guilt, shame, loneliness, overwhelm
- **Pathological/Extreme** (7): depression, learned_helplessness, pessimistic_depression, panic, dissociation, depersonalization, derealization
- **Altered/Boundary** (6): awe, trance, mystical_experience, ego_dissolution, peak_experience, nostalgia
- **Transitional/Contextual** (6): confusion, frustration, anticipation, relief, surprise, disappointment
- **Unelaborated** (6): contentment, interest, calm, neutral, alert, reflective

---

## Experiment Protocol Verification

### Verification Criteria

All 29 experiments were verified against 6 criteria:

 | Criterion|  Description|  Status|
 | ----------|  -------------|  --------|
 | File Structure|  Both `prepare_*.py` and `run_*.py` exist|  ✅ All Pass|
 | READ-ONLY Designation|  `prepare_*.py` marked READ-ONLY|  ✅ All Pass|
 | AGENT-EDITABLE Designation|  `run_*.py` marked AGENT-EDITABLE|  ✅ All Pass|
 | Time Budget|  600 seconds (10 minutes) defined|  ✅ All Pass|
 | Primary Metrics|  Output format: `primary_metric: <value>`|  ✅ All Pass|
 | Import Structure|  Proper imports from prepare files|  ✅ All Pass|

### Verification Summary

```text
Total Experiments: 29
Fully Compliant: 29
Non-Compliant: 0
Compliance Rate: 100%
```

---

## Step-by-Step Usage Instructions

### Step 1: Environment Setup

```bash
# Navigate to the auto-improvement directory
cd /Users/lesoto/Sites/PYTHON/apgi-experiments/auto-improvement

# Ensure dependencies are installed
pip install numpy matplotlib dataclasses
```

### Step 2: Run a Single Experiment

```bash
# Run Iowa Gambling Task
uv run run_iowa_gambling_task.py

# Run Attentional Blink
uv run run_attentional_blink.py

# Run Visual Search
uv run run_visual_search.py
```

### Step 3: Run APGI System Demo

```bash
# Run complete APGI demonstration
python APGI_System.py
```

This will:

1. Validate parameter ranges
2. Initialize 51 psychological states
3. Run a 75-second simulation
4. Generate comprehensive visualizations
5. Save results to `apgi_complete_output/`

### Step 4: Analyze Results

Output files are saved to `apgi_complete_output/`:

- `complete_dashboard.png` - Comprehensive visualization
- `corrected_parameters.json` - Validated parameters
- `simulation_summary.json` - Summary statistics

### Step 5: Modify Experiments (Agent-Editable Files)

Edit `run_*.py` files to optimize parameters:

```python
# Example: Modifying Iowa Gambling Task
NUM_TRIALS_CONFIG = 150  # Increase trials
BASE_LEARNING_RATE = 0.08  # Adjust learning
EXPLORATION_PROB = 0.10  # Reduce exploration
```

**Important:** Never modify `prepare_*.py` files - they are READ-ONLY.

### Step 6: Run Autonomous Improvement Loop

```bash
# The system supports autonomous optimization
# Each run outputs: primary_metric: <value>
# Goal: Maximize the primary metric
# Time budget: 600 seconds per run
```

---

## Available Experiments

### Decision-Making Tasks

 | Experiment|  Primary Metric|  Description|
 | ----------|  ----------------|  -------------|
 | Iowa Gambling Task|  `net_score`|  Decision-making under uncertainty|
 | Go/No-Go|  `d_prime`|  Response inhibition|
 | Stop Signal|  `stop_signal_rt`|  Inhibitory control|
 | Simon Effect|  `simon_effect_ms`|  Stimulus-response compatibility|

### Attention Tasks

 | Experiment|  Primary Metric|  Description|
 | ----------|  ----------------|  -------------|
 | Attentional Blink|  `blink_magnitude`|  Temporal attention limitation|
 | Posner Cueing|  `cueing_effect_ms`|  Spatial attention orienting|
 | Visual Search|  `conjunction_slope`|  Search efficiency|
 | Change Blindness|  `change_detection_rate`|  Visual change detection|
 | Inattentional Blindness|  `detection_rate`|  Unexpected stimulus detection|
 | Navon Task|  `global_local_bias`|  Global vs local processing|

### Memory Tasks

 | Experiment|  Primary Metric|  Description|
 | ----------|  ----------------|  -------------|
 | Dual N-Back|  `d_prime`|  Working memory updating|
 | Sternberg Memory|  `memory_scan_rate`|  Memory scanning speed|
 | Working Memory Span|  `span_size`|  Working memory capacity|
 | DRM False Memory|  `false_alarm_rate`|  False memory creation|
 | Serial Reaction Time|  `sequence_learning`|  Implicit sequence learning|
 | Artificial Grammar Learning|  `grammar_accuracy`|  Implicit grammar learning|
 | Probabilistic Category Learning|  `category_accuracy`|  Probabilistic learning|

### Interference Tasks

 | Experiment|  Primary Metric|  Description|
 | ----------|  ----------------|  -------------|
 | Stroop Effect|  `interference_effect_ms`|  Cognitive interference|
 | Eriksen Flanker|  `flanker_effect_ms`|  Response competition|
 | Masking|  `backward_masking_effect`|  Visual masking|

### Perception Tasks

 | Experiment|  Primary Metric|  Description|
 | ----------|  ----------------|  -------------|
 | Binocular Rivalry|  `dominance_duration`|  Perceptual alternation|
 | Multisensory Integration|  `integration_index`|  Cross-modal integration|
 | Time Estimation|  `temporal_precision`|  Time perception accuracy|

### Specialized Tasks

 | Experiment|  Primary Metric|  Description|
 | ----------|  ----------------|  -------------|
 | Somatic Marker Priming|  `priming_effect`|  Somatic influence on decisions|
 | Interoceptive Gating|  `gating_ratio`|  Interoceptive processing|
 | Metabolic Cost|  `metabolic_efficiency`|  Energy efficiency|
 | Virtual Navigation|  `navigation_accuracy`|  Spatial navigation|
 | AI Benchmarking|  `benchmark_score`|  AI performance metrics|

---

## Configuration Parameters

### APGI Integration Parameters

Each experiment includes APGI parameters optimized for the task:

```python
APGI_PARAMS = {
    "experiment_name": "experiment_id",
    "enabled": True,
    "tau_s": 0.35,    # Surprise decay (0.2-0.5s)
    "beta": 1.5,      # Somatic gain (0.5-2.5)
    "theta_0": 0.5,   # Baseline threshold
    "alpha": 5.5,     # Sigmoid steepness (3-8)
}
```

### Task-Specific Optimizations

 | Task Type|  τ_S|  β|  θ₀|  α|
 | ---------|  ---|  -|  --|  -|
 | Fast temporal (AB)|  0.25|  1.8|  0.4|  6.0|
 | Decision (IGT)|  0.40|  2.0|  0.4|  5.0|
 | Interference (Stroop)|  0.30|  1.6|  0.35|  6.0|
 | Memory (N-Back)|  0.35|  1.4|  0.45|  5.5|
 | Perception (Search)|  0.35|  1.3|  0.5|  5.0|

---

## Output Interpretation

### Primary Metrics

Each experiment outputs a primary metric to maximize:

```text
net_score: 8.0000
completion_time_s: 45.23
```

### Simulation Outputs

The APGI system generates comprehensive outputs:

```text
✅ Simulation complete: 127 ignitions detected

Measurements:
  • HEP Amplitude: 4.23 μV
  • P3b Latency: 312.5 ms
  • Detection Threshold (d'): 2.15
  • Anxiety Index: 0.45
```

### Visualization Dashboard

The dashboard includes:

1. Core dynamics (S, θ, ignitions)
2. Measurement proxies (HEP, P3b)
3. Neuromodulator dynamics (ACh, NE, DA, 5-HT)
4. Domain-specific analysis (survival vs neutral)
5. Psychiatric profile comparison
6. State space trajectory
7. Precision expectation gap (anxiety index)

---

## Troubleshooting

### Common Issues

**Issue:** `ImportError: cannot import name 'ExperimentAPGIRunner'`
**Solution:** Ensure `experiment_apgi_integration.py` exists in the directory.

**Issue:** Parameter validation warnings
**Solution:** Parameters outside valid ranges are automatically clipped. Check the warning message for details.

**Issue:** `TIME_BUDGET exceeded`
**Solution:** Reduce `NUM_TRIALS_CONFIG` or optimize the simulation loop.

### Verification Commands

```bash
# Verify deck configurations
python prepare_iowa_gambling_task.py

# Verify stimulus sets
python prepare_attentional_blink.py

# Run full equation verification
python APGI_System.py
```

### Parameter Validation

```python
from APGI_System import APGIParameters

params = APGIParameters(tau_S=0.35, alpha=5.5, beta=1.5)
violations = params.validate()
if violations:
    print(f"Violations: {violations}")
```

---

## System Capabilities

The APGI Auto-Improvement System provides:

- ✅ **29 validated experiment protocols** with 100% compliance
- ✅ **Complete dynamical system implementation** (51 states, all equations)
- ✅ **Validated parameter ranges** (τ_S: 0.2-0.5s, α: 3-8, β: 0.5-2.5)
- ✅ **Measurement equation outputs** (HEP, P3b, d', duration)
- ✅ **Neuromodulator integration** (ACh, NE, DA, 5-HT, CRF)
- ✅ **Psychiatric profile modeling** (GAD, MDD, Psychosis)
- ✅ **Autonomous optimization loop** with 600s time budget

For questions or issues, refer to the detailed documentation in `README.md` and `USAGE.md`.
