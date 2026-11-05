import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("governance", ROOT / "scripts/governance.py")
gov = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gov)


def test_shipped_records_and_text_are_clean():
    assert gov.record_errors() == []
    assert gov.scan_text_claims() == []


def test_unsupported_claim_and_noncanonical_expansion_detected(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("Runner reaches 100/100 APGI compliance.\n")
    (tmp_path / "b.md").write_text("Active Predictive Global Ignition\n")
    monkeypatch.setattr(gov, "ROOT", tmp_path)
    problems = gov.scan_text_claims()
    assert any("unsupported" in p for p in problems) and any("non-canonical" in p for p in problems)


def test_unsupported_claim_allowed_when_scoped(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("100/100 APGI on the internal checklist.\n")
    monkeypatch.setattr(gov, "ROOT", tmp_path)
    assert gov.scan_text_claims() == []


def test_exception_rules(tmp_path):
    for n in ("approvals.json", "claim_register.json", "requirements_map.json"):
        (tmp_path / n).write_text((ROOT / "governance" / n).read_text())
    ex = {"non_waivable": ["APGI-SCI-001"], "exceptions": [
        {"requirement": "APGI-SCI-001", "impact": "i", "compensating_controls": "c", "owner": "o",
         "expires": "2000-01-01", "remediation_plan": "r", "approver": "a"},
        {"requirement": "APGI-DAT-001", "impact": "i"}]}
    (tmp_path / "exceptions.json").write_text(json.dumps(ex))
    errs = gov.record_errors(tmp_path)
    assert any("non-waivable" in e for e in errs) and any("expired" in e for e in errs)
    assert any("missing owner" in e for e in errs)


def test_approval_by_unappointed_role_flagged(tmp_path):
    for n in ("claim_register.json", "requirements_map.json", "exceptions.json"):
        (tmp_path / n).write_text((ROOT / "governance" / n).read_text())
    ap = json.loads((ROOT / "governance/approvals.json").read_text())
    ap["approvals"].append({"role": "Release authority"})
    (tmp_path / "approvals.json").write_text(json.dumps(ap))
    assert any("unappointed" in e for e in gov.record_errors(tmp_path))


def test_impact_selects_tests_and_reports_unmapped():
    out = gov.impact(["utils/apgi_consent.py", "battery/tiers.py", "README.md.bak"])
    assert "APGI-CON-001" in out["requirements"] and "tests/test_governance_modules.py" in out["regression_tests"]
    assert "APGI-DAT-002" in out["requirements"] and out["unmapped_changes"] == ["README.md.bak"]


def test_release_manifest_identity():
    m = gov.build_release_manifest()
    assert m["certified"] is False and len(m["protocol_files"]) >= 10
    assert m["version"] and set(m["schema_hashes"]) == {"protocol.schema.json"}
    sbom = gov.build_sbom()
    assert sbom["bomFormat"] == "CycloneDX" and sbom["components"]


def test_certification_refused_without_evidence_and_approvals(tmp_path):
    r = tmp_path / "r.json"
    r.write_text(json.dumps({"checks": {n: "PASS" for n in gov.MANDATORY_CHECKS}, "release_digests": {"x": "y"}}))
    refusals = gov.certify(r)
    assert any("Not appointed" in x for x in refusals)  # roles unfilled -> never certifies
    r.write_text(json.dumps({"checks": {"Release identity": "BLOCKED"}}))
    refusals = gov.certify(r)
    assert any("Release identity: BLOCKED" in x for x in refusals) and any("no release digests" in x for x in refusals)
    assert len([x for x in refusals if "MISSING" in x]) == len(gov.MANDATORY_CHECKS) - 1


def test_parameter_provenance_covers_every_default():
    import dataclasses
    import sys

    sys.path.insert(0, str(ROOT))
    import APGI_System as A

    rec = json.loads((ROOT / "governance/parameter_provenance.json").read_text())["fields"]
    names = {f.name for f in dataclasses.fields(A.APGIParameters) if f.default is not dataclasses.MISSING}
    assert names == set(rec), "APGIParameters defaults and parameter_provenance.json differ"
    assert rec["beta"]["confirmatory_use"] == "blocked_outside_registered_range"


def test_task_disposition_register_is_complete():
    reg = json.loads((ROOT / "governance/task_register.json").read_text())
    assert reg["source_rows"] == 142 == len(reg["tasks"])
    assert {t["disposition"] for t in reg["tasks"]} <= {"IMPLEMENTED", "PARTIAL", "EXTERNAL_BLOCKED", "DECISION_REQUIRED"}
    assert all(t["residual"] for t in reg["tasks"])
