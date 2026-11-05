# Changelog

All notable changes to `apgi-research` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

Work against the open-items register and APGI-VAL-SPEC-1.2 gap audit. Nothing here is a
scientific or release approval; see `docs/TASK-DISPOSITION.md` for per-task status and residuals.

### Changed (behaviour)
- `ProtocolSpec.from_file` / `load_all_protocols` verify `protocols/MANIFEST.json` before parsing
  and raise `ProtocolLockError` on any altered, missing or unlisted protocol file (REP-001).
- `classify_prediction(True, True)` now raises `VerdictConflictError` instead of returning
  CONFIRMED; non-boolean flags are rejected (VAL-001).
- Deletion no longer reports success without evidence: `delete_experiment_data` needs a path,
  `destroy_kms_key` needs a callback, and the scheduler deletes from a per-subject store
  inventory and verifies zero residual files (DAT-003, CON-002).
- CLI identity: `APGI_OPERATOR_ROLE`/`APGI_OPERATOR_ID` no longer grant privilege. Use
  `APGI_OPERATOR_TOKEN`, or opt in with `APGI_TRUSTED_LOCAL=1` (never in production).
  `create_token_pair` binds tokens to the supplied operator. `APGI_ENV=production` requires
  `APGI_JWT_SECRET`. Added `AuthorizationManager.emergency_lock/unlock` (API-002, IMP-002, OPS-001).
- Tier assignment quarantines sessions with missing/non-finite timing measurements to Tier C;
  psychometric fits reject NaN, negative, fractional or unconverged data (DAT-001, DAT-002).
- `APGI_System` draws all randomness from one recorded run seed (`set_run_seed`/`get_run_seed`) (REP-002).
- Unsupported "100/100 APGI compliance" wording rescoped to "internal feature checklist" (SCI-007);
  retired non-canonical acronym expansions removed from the changelog (SCI-006).
- CI: security scanners no longer end in `|| true`; `scripts/security_gate.py` fails on
  unresolved high/critical findings or missing reports (waivers expire).

### Added
- `battery/`: `screening`, `calibration`, `staircase` (3-down/1-up, QUEST), `metad` (MLE meta-d',
  present/absent split), `reversal_task`, `masking`, `acquisition`, `bridge` + `construct_bridge.json`,
  Williams-design `assign_order_at_enrolment`.
- `item_bank/scoring.py` (versioned scoring, omega, alpha, ICC(2,1), target evaluation).
- `utils/`: `session_journal` (checksummed append-only journal, checkpoint resume),
  `apgi_consent`, `apgi_deviation`, `apgi_provenance`, `results_io` (GUI import/export formats).
- CI gates `construct-bridge` and release-only `acceptance-coverage` (fails on an empty item bank
  or missing/unbalanced three-form stimulus bank).
- `governance/` records and `scripts/governance.py` (check, impact, release-manifest + SBOM, certify).

---

## [1.1.0] - 2026-10-17

Brings the research platform into compliance with APGI protocol specification 1.1.0, aligned
with `apgi-framework` 1.1.0.

- `protocols/` (new): all eight protocol JSONs (Protocol 0–7), the Draft 2020-12 schema,
  `CHANGELOG.md`, `MANIFEST.json` and `VALIDATION.json`, byte-identical to apgi-framework 1.1.0.
- `scripts/validate_protocols.py` and `tests/test_validate_protocols.py`, ported from
  apgi-framework with logic unchanged (reformatted with black for this repo's CI). Run with `make protocols-check`; CI runs it before the unit tests. `jsonschema`
  joins the `[dev]` extras and `requirements.txt`.
- `utils/apgi_protocol_spec.py` (new): loads the registered protocols (version, study status,
  cohort sizes, per-prediction status and thresholds, the reduced empirical model) and adds
  `Verdict` / `classify_prediction`, the v1.1 rule that a failure to confirm is INDETERMINATE, not
  FALSIFIED. Exposes the registered θ₀, γ_sig and β_SM ranges; κ stays `null` (uncalibrated) and
  has no legacy default.
- `check_parameter_defaults()` compares the `APGI_System.APGIParameters` defaults with the
  registered ranges. θ₀ = 0.5 and the sigmoid steepness `alpha` = 5.5 (γ_sig) are inside; the
  somatic gain `beta` = 1.5 (β_SM) is outside the registered [0.3, 0.7]. The model is unchanged;
  the deviation is reported so the default is not used for a confirmatory fit.
- Version 1.1.0 (was 0.1.0). `utils/apgi_version.py` now reads `pyproject.toml` first, so a stale
  installed `autoresearch.egg-info` no longer reports an old version.
- Project name unified as *Allostatic Precision-Gated Ignition*. Five earlier, non-canonical expansions
  of the acronym were retired everywhere; they are recoverable from version control history only.
- `CITATION.cff` (new) and a BibTeX entry in `README.md`, both 1.1.0. README explains that the 29
  paradigm experiments are not the registered Protocol 0–7 studies.
- `tests/test_protocol_spec_v1_1.py` (new): release, κ, cohort, verdict, parameter-range and
  version-agreement checks.

## [0.1.0] - 2026-03-22

Initial release.
