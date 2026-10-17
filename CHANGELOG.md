# Changelog

All notable changes to `apgi-research` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- Project name unified as *Allostatic Precision-Gated Ignition*. It was previously also given as
  "Autonomous Program Generation and Improvement", "Autonomous Psychological General
  Intelligence", "Autonomous Personal Growth Intelligence", "Active Perception and Generative
  Inference" and "Active Predictive Global Ignition".
- `CITATION.cff` (new) and a BibTeX entry in `README.md`, both 1.1.0. README explains that the 29
  paradigm experiments are not the registered Protocol 0–7 studies.
- `tests/test_protocol_spec_v1_1.py` (new): release, κ, cohort, verdict, parameter-range and
  version-agreement checks.

## [0.1.0] - 2026-03-22

Initial release.
