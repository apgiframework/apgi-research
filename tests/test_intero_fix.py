#!/usr/bin/env python3
"""Final validation of interoceptive gating fix."""

import sys
from pathlib import Path

if "pytest" in sys.modules:
    import pytest

    pytest.skip(
        "manual validation script; run directly with python",
        allow_module_level=True,
    )

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments.run_interoceptive_gating as igo
from experiments.run_interoceptive_gating import EnhancedInteroceptiveGatingRunner

print("=" * 70)
print("INTEROCEPTIVE GATING - FINAL VALIDATION")
print("=" * 70)

# Test with standard trial count
igo.NUM_TRIALS_CONFIG = 100

runner = EnhancedInteroceptiveGatingRunner(enable_apgi=True)
results = runner.run_experiment()

print("\nTrial Configuration:")
print(f"  Total trials: {results['num_trials']}")
print(f"  Completion time: {results['completion_time_s']:.2f}s")

print("\nSIGNAL DETECTION METRICS")
print("-" * 70)
print(f"  Hit Rate:                {results['hit_rate']:>8.1%}  (expected: 0.3-0.5)")
print(
    f"  False Alarm Rate:        {results['false_alarm_rate']:>8.1%}  (expected: 0.05-0.15)"
)
print(f"  D-Prime:                 {results['d_prime']:>8.3f}  (expected: 0.5-1.5)")
print(
    f"  Gating Effect:           {results['gating_effect']:>8.3f}  (expected: 0.1-0.4)"
)
print(
    f"  Gating Threshold:        {results['gating_threshold']:>8.3f}  (expected: 0.3-0.6)"
)
print(f"  Primary Metric:          {results['primary_metric']:>8.1%}")
print(f"  Mean RT:                 {results['mean_rt_ms']:>8.1f}ms")

print("\nAPGI DYNAMICS METRICS")
print("-" * 70)
if results.get("apgi_enabled"):
    print(f"  Mean Surprise:           {results.get('apgi_mean_surprise', 0.0):>8.3f}")
    print(
        f"  Mean Somatic Marker:     {results.get('apgi_mean_somatic_marker', 0.0):>8.3f}"
    )
    print(f"  Mean Threshold:          {results.get('apgi_mean_threshold', 0.0):>8.3f}")
    print(f"  Metabolic Cost:          {results.get('apgi_metabolic_cost', 0.0):>8.3f}")
else:
    print("  APGI disabled")

print("\nVALIDATION STATUS")
print("-" * 70)

issues = []

# Check hit rate
if results["hit_rate"] > 0:
    issues.append(f"✅ Hit rate is {results['hit_rate']:.1%} (non-zero)")
else:
    issues.append("❌ Hit rate is ZERO!")

# Check d_prime
if results["d_prime"] > 0:
    issues.append(f"✅ D-prime is {results['d_prime']:.3f} (positive)")
else:
    issues.append("❌ D-prime is ZERO!")

# Check gating threshold
if results["gating_threshold"] > 0:
    issues.append(
        f"✅ Gating threshold is {results['gating_threshold']:.3f} (non-zero)"
    )
else:
    issues.append("❌ Gating threshold is ZERO!")

# Check gating effect
if abs(results["gating_effect"]) > 0.01:
    issues.append(f"✅ Gating effect is {results['gating_effect']:.3f}")
else:
    issues.append(f"⚠️  Gating effect is {results['gating_effect']:.3f} (very small)")

for issue in issues:
    print(f"  {issue}")

print("\n" + "=" * 70)
