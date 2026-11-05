"""Build the task disposition register from the supplied task CSV.

Usage: python scripts/build_task_register.py TASKS.csv

Writes governance/task_register.json and docs/TASK-DISPOSITION.md. The CSV is
read-only input; dispositions come from RULES below (first matching rule wins),
so every supplied task gets exactly one explicit, evidence-linked disposition.
A task ID matching no rule is an error (nothing is silently skipped).

Dispositions
  IMPLEMENTED         repo deliverable exists and is verified by tests in this checkout
  PARTIAL             repo-side part implemented; named residual remains
  EXTERNAL_BLOCKED    needs another repository, service, data or approval; nothing implementable here
  DECISION_REQUIRED   needs a recorded decision by a named authority first
"""

import csv
import json
import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent

I, P, E, D = "IMPLEMENTED", "PARTIAL", "EXTERNAL_BLOCKED", "DECISION_REQUIRED"
BROWSER = "Browser build lives in apgiframework.com (not in this checkout); needs a real-browser acceptance run."
# (regex over task id, disposition, evidence files, residual)
RULES = [
 (r"S1-L1-05$", P, ["item_bank/lint.py", "item_bank/scoring.py", "tests/test_item_bank.py", "tests/test_item_scoring.py"], "candidates.json is intentionally empty; reviewed five-point stems and a scoring version are required (acceptance-coverage gate fails until then)."),
 (r"S1-L1-06$", I, ["item_bank/render.py", "tests/test_item_bank.py"], "Mechanism verified; applies to the bank once populated."),
 (r"S1-L1-10\.1$", I, ["item_bank/lint.py", "tests/test_item_bank.py"], "Mechanism verified; applies to the bank once populated."),
 (r"S1-L1-15$", I, ["item_bank/exclusion_record.json", "item_bank/lint.py"], "Exclusion list needs clinical-governance sign-off."),
 (r"S1-L1-16$", I, ["item_bank/lint.py"], "Banned-topic list needs content-review sign-off."),
 (r"S1-L1-17$", I, ["item_bank/reliability_targets.json", "item_bank/reliability_targets.sha256", "item_bank/scoring.py", "scripts/ci_gates.py"], "Targets frozen and hash-gated; observed reliability needs pilot data and a registered amendment path."),
 (r"S2-ING-02(\.3)?$", P, ["utils/session_journal.py", "tests/test_session_journal.py"], "Journal + checkpoint-resume library done; API persistence and resume on every supported client are in other repositories."),
 (r"S2-PR5-01\.1$", P, ["utils/apgi_protocol_spec.py"], "Lock-verified local protocol load done; retrieval from the API registry is external."),
 (r"S2-PR5-01\.3$", P, ["utils/apgi_consent.py", "tests/test_governance_modules.py"], "Four-purpose consent ledger with decline handling done; API-side registration is external."),
 (r"S2-PR5-01\.(7|8)$", P, ["utils/apgi_provenance.py", "battery/acquisition.py"], "Checksum/integrity helpers done; upload and database verification need apgi-api."),
 (r"S2-PR5-01\.12$", P, ["utils/apgi_data_retention.py", "tests/test_apgi_data_retention_comprehensive.py"], "Inventory-based deletion with zero-residual verification done locally; database/backups external."),
 (r"S2-PR5-0[12](\.\d+)?$", E, ["mathematical-contracts.yaml"], "Requires approved shared reference model (S1-REG-04), apgi-api and synthetic-generator access."),
 (r"S3-BAT-01\.1\.2$", P, ["battery/sdt.py", "tests/test_battery.py"], "Server-side handling of present/absent + confidence trial vectors done (dprime_by_presence). " + BROWSER),
 (r"S3-BAT-01\.1\.3$", I, ["battery/sdt.py", "tests/test_battery.py"], "None for the formulae; task data come from the browser."),
 (r"S3-BAT-01\.1\.4$", I, ["battery/psychometric.py", "tests/test_battery.py"], "Inputs now validated; real fits need browser data."),
 (r"S3-BAT-01\.1\.5$", P, ["battery/metad.py", "tests/test_battery_extended.py"], "Per-session MLE meta-d' with present/absent split done; the registered hierarchical Bayesian fit needs a sampler (JAGS/Stan) and pooled data."),
 (r"S3-BAT-01\.1\.1$", P, ["battery/staircase.py", "tests/test_battery_extended.py"], "3-down/1-up and QUEST reference implementations done; client port needed. " + BROWSER),
 (r"S3-BAT-01\.2\.1$", P, ["battery/reversal_task.py", "tests/test_battery_extended.py"], "Seeded schedule generator done. " + BROWSER),
 (r"S3-BAT-01\.2\.[23]$", I, ["battery/reversal.py", "tests/test_battery.py"], "Estimators verified on synthetic data only."),
 (r"S3-BAT-01\.3\.[1-5]$", P, ["battery/masking.py", "tests/test_battery_extended.py"], "Decision-agnostic frame-lock / achieved-SOA / PAS / down-tier / summary helpers done; task gated on S3-DEC-01. " + BROWSER),
 (r"S3-BAT-01\.4(\.\d)?$", P, ["battery/calibration.py", "tests/test_battery_extended.py"], "Calibration maths and attention-check scoring done. " + BROWSER),
 (r"S3-BAT-01\.5\.4$", I, ["battery/timing.py", "tests/test_battery.py"], "Record keeps callback, refresh and photodiode fields separate."),
 (r"S3-BAT-01\.5(\.\d)?$", P, ["battery/timing.py", "battery/tiers.py"], "Server-side record and visibility counting done. " + BROWSER),
 (r"S3-BAT-01\.6\.[1-4]$", I, ["battery/tiers.py", "tests/test_battery.py"], "Missing measurements now quarantine to Tier C rather than defaulting to A."),
 (r"S3-BAT-01\.6(\.5)?$", P, ["battery/tiers.py"], "Version constant defined; storing it on every persisted session is an API/client change."),
 (r"S3-BAT-01\.8$", I, ["battery/psychometric.py", "tests/test_battery.py"], "None."),
 (r"S3-BAT-01\.[123]$", P, ["battery/"], "Analysis-side done. " + BROWSER),
 (r"S3-BAT-02$", D, ["battery/masking.py"], "Blocked on decision S3-DEC-01 (masking vs attentional blink)."),
 (r"S3-BAT-03$", P, ["battery/forms.py", "scripts/ci_gates.py"], "Matched-split and rotation done; real stimuli absent so acceptance-coverage gate fails by design."),
 (r"S3-BAT-04$", P, ["battery/forms.py", "tests/test_battery_extended.py"], "Williams-design assignment done; persisting the order at enrolment is an API change."),
 (r"S3-BAT-05(\.[1-6])?$", I, ["battery/construct_bridge.json", "battery/bridge.py", "scripts/ci_gates.py"], "Rows drafted and gated; scientific-authority review pending (status draft)."),
 (r"S3-BAT-06$", P, ["battery/"], "Reference analysis for detection and reversal done. " + BROWSER),
 (r"S3-BAT-07$", D, [], "Blocked on S1-L1-09 check-in design decision."),
 (r"S3-SCR-01(\.[1-8])?$", I, ["battery/screening.py", "tests/test_battery_extended.py"], "Library verified; server/form-layer wiring and fingerprint capture are in other repositories."),
 (r"S3-PIL-", E, [], "Needs pilot authorization, IRB disposition and observed participant data; no synthetic figures may stand in."),
 (r"XC-CI-04(\.[1-4])?$", P, [".github/workflows/ci.yml", "scripts/ci_gates.py", "pytest-gates.ini"], "Gates and mutation tests pass locally; a hosted run and required-check branch protection must be inspected on GitHub."),
 (r"S1-REG-04$", D, ["mathematical-contracts.yaml"], "Draft only; scientific authority must approve the reference model."),
 (r"S3-DEC-01$", D, [], "Decision for the scientific authority."),
 (r"S1-L1-09$", D, [], "Decision for the scientific authority."),
 (r"S3-PIL-06$", E, [], "Needs pilot authorization and data."),
 (r"APGI-PRO-001$", E, ["utils/apgi_protocol_spec.py"], "Needs approved v1.2/1.1.0 identity and per-definition implementation map."),
 (r"APGI-PRO-002$", P, ["battery/psychometric.py", "battery/tiers.py", "utils/apgi_protocol_spec.py", "tests/test_battery.py"], "Atomic rejection added for protocol lock, psychometric inputs and tiers; other entry points not audited."),
 (r"APGI-PRO-003$", P, ["utils/session_journal.py", "tests/test_session_journal.py"], "Ordering/idempotency/torn-write recovery verified for the journal; interface applicability profile pending."),
 (r"APGI-PRO-004$", P, ["governance/validation_profile.json"], "Draft profile written; approval pending."),
 (r"APGI-SCI-001$", D, ["governance/discrepancies.json"], "DISC-001 recorded; code deliberately not changed until the scientific authority selects the canonical equation and golden vectors."),
 (r"APGI-SCI-00[234]$", D, ["governance/discrepancies.json"], "Needs approved formal specification, hierarchy ranges, notation appendix and fixtures."),
 (r"APGI-SCI-005$", P, ["governance/parameter_provenance.json", "utils/apgi_deviation.py", "tests/test_governance_modules.py"], "Every default is inventoried and flagged; classification (derived/calibrated/assumed + source) needs the scientific authority."),
 (r"APGI-SCI-006$", I, ["CHANGELOG.md", "scripts/governance.py"], "Non-canonical expansions removed; scan enforced by `governance.py check`."),
 (r"APGI-SCI-007$", P, ["governance/claim_register.json", "scripts/governance.py"], "Starter claim register and unsupported-claim scan; full statement-to-claim map needs scientific review."),
 (r"APGI-REP-001$", P, ["utils/apgi_protocol_spec.py", "tests/test_protocol_spec_v1_1.py"], "Runtime lock verification done; OpenTimestamps proof reference is external."),
 (r"APGI-REP-002$", P, ["APGI_System.py", "tests/test_run_seed.py"], "Single recorded seed through every stochastic path of APGI_System; other modules and cross-platform run comparison remain."),
 (r"APGI-REP-00[35]$", E, [], "Needs approved assumptions/units inventory and independent calculation set."),
 (r"APGI-REP-004$", P, ["utils/apgi_deviation.py", "tests/test_governance_modules.py"], "Deviation detection and exploratory labelling done for parameters/analyses; export-time enforcement not wired into every runner."),
 (r"APGI-IMP-001$", P, ["scripts/governance.py"], "Release-set manifest and SBOM generator done; dependency lock and reproducible-build demonstration pending."),
 (r"APGI-IMP-002$", P, ["utils/apgi_authz.py", "apgi_cli.py", "tests/test_authz_hardening.py"], "Fail-closed signing key and verified CLI identity done; production safety profile and secret scanning pending."),
 (r"APGI-IMP-003$", E, [], "Needs fault-induction runs under the declared operating profile."),
 (r"APGI-DAT-001$", P, ["battery/psychometric.py", "tests/test_battery.py"], "Non-finite/unconverged results rejected; approved golden pipeline set pending."),
 (r"APGI-DAT-002$", P, ["battery/tiers.py", "battery/screening.py", "battery/acquisition.py"], "Missing/non-finite measurements quarantined; end-to-end quarantine report workflow pending."),
 (r"APGI-DAT-003$", P, ["utils/apgi_data_retention.py", "utils/apgi_compliance.py"], "Fail-closed deletion with inventory done; per-class lifecycle and restore record need approval."),
 (r"APGI-DAT-004$", P, ["utils/apgi_provenance.py", "tests/test_governance_modules.py"], "Provenance record + checksum verification done; every dataset still needs inventorying."),
 (r"APGI-API-00[134]$", E, [], "Needs deployed interfaces / approved N/A decision / consumer-version records."),
 (r"APGI-API-002$", P, ["utils/apgi_authz.py", "apgi_cli.py", "tests/test_authz_hardening.py"], "Token binds to supplied operator; CLI no longer trusts env role; object-level protection unproven."),
 (r"APGI-INT-00[1234]$", E, [], "Needs apgi-api, clients and approved model checked out."),
 (r"APGI-REL-001$", E, ["scripts/governance.py"], "Certification refuses until evidence and approvals exist; no release candidate certified."),
 (r"APGI-REL-002$", P, ["utils/apgi_authz.py", "docs/Operational-Playbooks.md"], "emergency_lock now exists and playbook matches; deployment/rollback package pending."),
 (r"APGI-REL-003$", P, ["scripts/governance.py"], "`governance.py impact` maps changed files to requirements and regression tests; recorded approvals pending."),
 (r"APGI-REL-004$", P, ["scripts/governance.py"], "Compatibility record generated for this checkout only; multi-component tested matrix pending."),
 (r"APGI-REL-005$", E, [], "Needs certified release digests before any archival deposit."),
 (r"APGI-CON-001$", P, ["utils/apgi_consent.py", "tests/test_governance_modules.py"], "Consent ledger with scope enforcement and zero persistence on decline done; binding into every ingest/export path of the other components pending."),
 (r"APGI-CON-002$", P, ["utils/apgi_data_retention.py", "tests/test_apgi_data_retention_comprehensive.py"], "Inventory deletion + residual proof + KMS fail-closed done; backup policy [CONFIRM] pending."),
 (r"APGI-ACQ-001$", P, ["battery/acquisition.py", "tests/test_governance_modules.py"], "Checksum/drift/dropped-sample detection done; protocol-approved tolerances and hardware telemetry pending."),
 (r"APGI-VAL-001$", P, ["utils/apgi_protocol_spec.py", "tests/test_protocol_spec_v1_1.py"], "Conflicting/non-boolean evidence now raises; invoking pinned registered analyses with lineage pending."),
 (r"APGI-CLI-001$", E, ["item_bank/scoring.py"], "Needs populated instruments, independent scoring fixtures and an approved accessibility level."),
 (r"AUDIT-GOV-00[1-5]$", P, ["governance/"], "Draft records created (profile, claims, approvals, exceptions); approval and certification require the named authorities."),
 (r"AUDIT-CI-001$", I, [".github/workflows/ci.yml", "scripts/security_gate.py", "tests/test_security_gate.py"], "`|| true` removed; gate fails on unresolved high/critical or missing reports; needs a hosted run."),
 (r"AUDIT-CI-002$", I, ["scripts/ci_gates.py", "tests/test_item_bank.py"], "Release-only gate fails on zero items / missing or unbalanced three-form bank."),
 (r"AUDIT-GUI-001$", P, ["utils/results_io.py", "tests/test_results_io.py", "GUI_auto_improve_experiments.py"], "Import now loads data; CSV/HTML/PDF saves match their extension (headless-tested). Saved-settings application and XPR additional-constraint submission are not yet changed; native GUI run not performed."),
 (r"AUDIT-OPS-001$", I, ["utils/apgi_authz.py", "docs/Operational-Playbooks.md", "tests/test_authz_hardening.py"], "emergency_lock/unlock implemented and playbook corrected."),
]


def disposition(task_id):
    for pat, disp, evidence, residual in RULES:
        if re.match(pat, task_id):
            return disp, evidence, residual
    raise KeyError(f"no disposition rule for {task_id}")


def main(argv):
    rows = list(csv.DictReader(open(argv[1], encoding="utf-8-sig")))
    out = []
    for r in rows:
        disp, evidence, residual = disposition(r["Task ID"])
        missing = [e for e in evidence if not (ROOT / e).exists()]
        if missing:
            raise FileNotFoundError(f"{r['Task ID']}: evidence missing {missing}")
        out.append({"task_id": r["Task ID"], "task": r["Task Name"], "supplied_status": r["Current status"],
                    "disposition": disp, "evidence": evidence, "residual": residual})
    counts = Counter(o["disposition"] for o in out)
    (ROOT / "governance" / "task_register.json").write_text(
        json.dumps({"source_rows": len(rows), "counts": counts, "tasks": out}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Task disposition register", "",
             "Generated by `scripts/build_task_register.py` from the supplied task CSV. A disposition says what is",
             "verified **in this checkout**; it does not change the supplied status and is not a scientific",
             "or release approval. `IMPLEMENTED` = repo deliverable verified by tests; `PARTIAL` = named residual",
             "remains; `EXTERNAL_BLOCKED` = needs another repository, service, data or approval;",
             "`DECISION_REQUIRED` = needs a recorded decision first.", "",
             f"**{len(rows)} tasks:** " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), "",
             "| Task | Name | Supplied status | Disposition | Evidence | Residual |", "|---|---|---|---|---|---|"]
    for o in out:
        ev = "<br>".join(f"`{e}`" for e in o["evidence"]) or "—"
        lines.append(f"| {o['task_id']} | {o['task'][:70].replace('|','/')} | {o['supplied_status']} | {o['disposition']} | {ev} | {o['residual'].replace('|','/')} |")
    (ROOT / "docs" / "TASK-DISPOSITION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(dict(counts))


if __name__ == "__main__":
    main(sys.argv)
