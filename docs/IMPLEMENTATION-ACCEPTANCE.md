# Implementation and acceptance register

This register preserves the supplied task statuses. It is not a scientific
validation report. A source review, passing component test, or scheduling label
cannot close a browser, hardware, study, or pilot acceptance requirement.

## Research CI repair (XC-CI-04.1–04.4)

The gate job installs `requirements-ci-gates.txt`, including pytest-cov, and
uses `pytest-gates.ini` to scope coverage to the gate scripts. Each matrix job
runs its gate and the mutation/regression suite. JUnit and coverage XML are
retained as separate artifacts for each gate. Run locally:

```sh
python -m pip install -r requirements-ci-gates.txt
python scripts/ci_gates.py
python -m pytest -c pytest-gates.ini --junitxml=gate-tests.xml
```

The local gate commands pass and the 23 mutation/regression tests pass. This
establishes protocol-gate behavior only. GitHub run results and required branch
checks must also be inspected before declaring XC-CI-04 complete. Required check
names are `Research Gates (trial-sequencing)`, `Research Gates (protocol-locking)`,
`Research Gates (timing-contracts)`, and `Research Gates (parallel-form-balance)`.

## Reference model draft

`../mathematical-contracts.yaml` is a JSON-compatible YAML draft for S1-REG-04
and S1-REG-04.1–04.9. It proposes equations, input normalization, units, update
order, reset/refractory semantics, integration, observations and pi_i_eff. It
records differences from the existing simulator and explicit unresolved choices.
It is not qualified or registered and does not authorize collection. The supplied
S1-REG-04 rows remain Not Started in the source register; draft preparation is
implementation progress, not scientific approval.

## Required decisions and evidence

- S1-REG-04: approved shared reference model and registered version, required for
  S2-PR5-01 and S2-PR5-02. No model decision is inferred from Python simulations.
- S3-DEC-01: masking decision, required for S3-BAT-01.3 and S3-BAT-02.
- S1-L1-09: check-in decision, required for S3-BAT-07.
- Protocol 1: resolve 30 versus 50 QUEST top-up trials before acquisition lock.
- S3-PIL-06: pilot authorization, IRB disposition and observed data, required for
  completion, timing failure, comprehension, reliability and retention figures.
- Item bank: revise question stems for five-point agreement and version scoring;
  relabeling the existing four bespoke options is not a valid conversion.
- Reliability: prospectively register omega >= 0.80, ICC(2,1) >= 0.70 at 2–3
  weeks, and self-report/behavior correlation >= 0.15, with exclusions and
  estimators fixed before collection. These remain targets, not observations.

## Acceptance evidence still required

| Workstream | Required acceptance run |
| --- | --- |
| Browser battery | Real browser detection, reversal and approved masking sessions; calibration; frame-locked SOAs with achieved timing; confidence/PAS responses; visibility and fullscreen changes; matched forms. Physical onset remains separate from animation callbacks. |
| Item bank | Five-point agreement renderer, fixed block order and seeded within-block shuffling; diagnostic/sensitive-content and free-text exclusions; frozen reliability plan. |
| Study integration | Registered protocol, synthetic participant and four-purpose consent including one decline; shared-model trials/signals; persisted uploads and checksums; exposure-gated scoring; conformance and lineage; deletion with zero residual rows; every repository commit and tolerance-checked regeneration. |
| Clients | Persisted enrollment form assignments consumed by clients; browser quality payload sent to API; screening outcomes persisted; interrupted checkpoint resume tested on every supported client. |
| Pilot reporting | Observed denominator-based completion, timing failure, comprehension, reliability, exclusion and retention estimates, with uncertainty and dataset provenance. No synthetic results labeled as pilot results. |

## Supplied task rows

Statuses below are the supplied baseline, not completion claims for this change.

| Task | Supplied status | Deliverable |
| --- | --- | --- |
| S1-L1-05 | Blocked | Use 5-point agreement scale, not sliders |
| S1-L1-06 | Blocked | Randomise within blocks; fix block order |
| S1-L1-10.1 | In Progress | Strip printed parameter labels from options ('Low ε', 'High ε') |
| S1-L1-15 | Not Started | Exclude diagnostic instruments (STAI, BDI-II, PHQ-9 and similar) |
| S1-L1-16 | Not Started | Ban sensitive content and free-text fields from the item bank |
| S1-L1-17 | Not Started | Set reliability targets before collection |
| S2-ING-02 | In Progress | Reliable session journal |
| S2-ING-02.3 | In Progress | Resume sessions from journaled checkpoints |
| S2-PR5-01 | Not Started | End-to-end synthetic research test |
| S2-PR5-01.1 | Not Started | Integration step 1: Retrieve a registered protocol version |
| S2-PR5-01.2 | Not Started | Integration step 2: Create a synthetic study and participant |
| S2-PR5-01.3 | Not Started | Integration step 3: Register test consents for all four purposes incl. one declined |
| S2-PR5-01.4 | Not Started | Integration step 4: Start an experiment session |
| S2-PR5-01.5 | Not Started | Integration step 5: Execute deterministic behavioural trials |
| S2-PR5-01.6 | Not Started | Integration step 6: Produce simulated physiological signals |
| S2-PR5-01.7 | Not Started | Integration step 7: Upload trial events and signal chunks |
| S2-PR5-01.8 | Not Started | Integration step 8: Verify database records and checksums |
| S2-PR5-01.9 | Not Started | Integration step 9: Execute model scoring through the exposure gate |
| S2-PR5-01.10 | Not Started | Integration step 10: Run registered-method conformance checks |
| S2-PR5-01.11 | Not Started | Integration step 11: Verify data lineage |
| S2-PR5-01.12 | Not Started | Integration step 12: Request test deletion; confirm zero residual rows |
| S2-PR5-01.13 | Not Started | Integration step 13: Produce integration report naming every repository commit |
| S2-PR5-01.14 | Not Started | Drive integration test from shared ODE generator (doubles as recovery check) |
| S2-PR5-02 | Not Started | Acceptance: same result regenerated within tolerance with full provenance |
| S3-BAT-01.1 | Not Started | Build near-threshold visual detection task (3–4 min) |
| S3-BAT-01.1.1 | Not Started | Calibrate contrast individually |
| S3-BAT-01.1.2 | Not Started | Collect present/absent response then confidence |
| S3-BAT-01.1.3 | Not Started | Compute d′ = z(H) − z(FA) and c = −½[z(H) + z(FA)] |
| S3-BAT-01.1.4 | Not Started | Model psychometric functions for performance and confidence |
| S3-BAT-01.1.5 | Not Started | Estimate hierarchical meta-d′ separately for present and absent |
| S3-BAT-01.2 | Not Started | Build reversal-learning task (4 min) |
| S3-BAT-01.2.1 | Not Started | Two options, probabilistic rewards, unannounced reversals |
| S3-BAT-01.2.2 | Not Started | Estimate reward sensitivity, ± prediction-error learning, perseveration |
| S3-BAT-01.2.3 | Not Started | Report precision_weighted_learning_composite only (α·Π product) |
| S3-BAT-01.3 | Not Started | Build backward-masking task (3–4 min) |
| S3-BAT-01.3.1 | Not Started | Lock every SOA to a whole number of frames |
| S3-BAT-01.3.2 | Not Started | Log achieved SOA per trial |
| S3-BAT-01.3.3 | Not Started | Collect per-trial 4-point Perceptual Awareness Scale |
| S3-BAT-01.3.4 | Not Started | Down-tier sessions with dropped frames |
| S3-BAT-01.3.5 | Not Started | Output accuracy by SOA, RT distribution, confidence, drift, reliability |
| S3-BAT-01.4 | Not Started | Build calibration and quality checks (1–2 min) |
| S3-BAT-01.4.1 | Not Started | Measure FPS, display timing and screen size |
| S3-BAT-01.4.2 | Not Started | Card-against-screen pixels-per-mm calibration |
| S3-BAT-01.4.3 | Not Started | Blind-spot viewing-distance estimate (Li et al., 2020) |
| S3-BAT-01.4.4 | Not Started | Embed attention checks |
| S3-BAT-01.5 | Not Started | Log per-trial timing record |
| S3-BAT-01.5.1 | Not Started | Use requestAnimationFrame and performance.now() |
| S3-BAT-01.5.2 | Not Started | Enforce fullscreen, preload stimuli, schedule trials locally |
| S3-BAT-01.5.3 | Not Started | Detect page visibility changes |
| S3-BAT-01.5.4 | Not Started | Store callback time, refresh estimate and photodiode onset separately |
| S3-BAT-01.6 | In Progress | Assign device quality tiers from versioned tier_config |
| S3-BAT-01.6.1 | In Progress | Tier A: stable frames, fullscreen, desktop/laptop, no tab loss → primary norms |
| S3-BAT-01.6.2 | In Progress | Tier B: moderate timing noise → sensitivity analyses |
| S3-BAT-01.6.3 | In Progress | Tier C: major jitter, mobile, tab switch, failed checks → exclude from norms |
| S3-BAT-01.6.4 | In Progress | Keep per-session timing-error distribution, not only the mean |
| S3-BAT-01.6.5 | In Progress | Store tier_config_version on every session |
| S3-BAT-01.8 | Not Started | Report contrast thresholds as device-relative with tier/display covariates |
| S3-BAT-02 | Not Started | Use backward masking instead of attentional blink |
| S3-BAT-03 | In Progress | Build three matched parallel forms per task rotated by session index |
| S3-BAT-04 | In Progress | Counterbalance parallel-form order at enrolment |
| S3-BAT-05 | Not Started | Complete construct-to-task bridge rows before tasks enter battery |
| S3-BAT-05.1 | Not Started | Bridge: Access threshold θₜ ↔ contrast at 50% detection; criterion c |
| S3-BAT-05.2 | Not Started | Bridge: Exteroceptive precision Πₑ ↔ slope; meta-d′/d′ |
| S3-BAT-05.3 | Not Started | Bridge: Precision updating α·Π ↔ reversal composite; perseveration |
| S3-BAT-05.4 | Not Started | Bridge: Ignition ↔ accuracy-by-SOA; SOA-50; threshold-SOA visibility distributio |
| S3-BAT-05.5 | Not Started | Bridge: Interoceptive precision Πᵢ ↔ self-report only (not testable v0.1) |
| S3-BAT-05.6 | Not Started | Bridge: Somatic valuation β ↔ not measured v0.1 |
| S3-BAT-06 | Not Started | Build order 1: detection and reversal tasks first, with logging and tiers |
| S3-BAT-07 | Not Started | Build order 2: minimal Layer 1 check-in (sleep, stress, confidence, visibility, stakes) |
| S3-SCR-01 | In Progress | Effort, fraud and validity screening |
| S3-SCR-01.1 | In Progress | Flag impossibly fast responses (< ≈ 150 ms) |
| S3-SCR-01.2 | In Progress | Flag invariant responding (same key, zero confidence variance) |
| S3-SCR-01.3 | In Progress | Flag straight-lining and zig-zagging in questionnaires |
| S3-SCR-01.4 | In Progress | Exclude sessions with ≥ 2 failed attention checks from norms |
| S3-SCR-01.5 | In Progress | Flag duplicates via device fingerprint + email hash (flag, not block) |
| S3-SCR-01.6 | In Progress | Count one norm-contributing session per participant per battery version |
| S3-SCR-01.7 | In Progress | Screen bots at the form layer |
| S3-SCR-01.8 | In Progress | Report exclusion rates in the dataset card |
| S3-PIL-06.1 | Not Started | Measure completion rate |
| S3-PIL-06.2 | Not Started | Measure timing-failure rate |
| S3-PIL-06.3 | Not Started | Measure comprehension |
| S3-PIL-06.4 | Not Started | Measure reliability |
| S3-PIL-08 | Not Started | Replace attrition and usable-record estimates with pilot figures |
