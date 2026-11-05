import datetime
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("security_gate", ROOT / "scripts/security_gate.py")
sg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sg)

HIGH = {"results": [{"test_id": "B602", "filename": "a.py", "line_number": 3, "issue_severity": "HIGH"}]}
LOW = {"results": [{"test_id": "B101", "filename": "a.py", "line_number": 3, "issue_severity": "LOW"}]}
SEM = {"results": [{"check_id": "r1", "path": "a.py", "start": {"line": 1}, "extra": {"severity": "ERROR"}}]}
SAFE = {"vulnerabilities": [{"vulnerability_id": "1", "package_name": "x"}]}


def test_clean_reports_pass():
    assert sg.evaluate({"bandit": LOW, "semgrep": {"results": []}, "safety": {"vulnerabilities": []}}, {}) == []


def test_high_findings_fail_each_tool():
    for tool, rep in (("bandit", HIGH), ("semgrep", SEM), ("safety", SAFE)):
        assert sg.evaluate({tool: rep}, {}), tool


def test_missing_report_fails():
    assert "did not run" in sg.evaluate({"bandit": None}, {})[0]


def test_waiver_suppresses_until_expiry(tmp_path):
    w = tmp_path / "w.json"
    w.write_text(json.dumps({"waivers": [{"id": "B602", "owner": "o", "reason": "r", "expires": "2099-01-01"}]}))
    active, errs = sg.load_waivers(w)
    assert not errs and sg.evaluate({"bandit": HIGH}, active) == []
    w.write_text(json.dumps({"waivers": [{"id": "B602", "owner": "o", "reason": "r", "expires": "2000-01-01"}]}))
    active, errs = sg.load_waivers(w, today=datetime.date(2026, 1, 1))
    assert errs and sg.evaluate({"bandit": HIGH}, active)
    w.write_text(json.dumps({"waivers": [{"id": "B602"}]}))
    assert sg.load_waivers(w)[1]


def test_main_exit_codes(tmp_path):
    b = tmp_path / "b.json"
    b.write_text(json.dumps(HIGH))
    s = tmp_path / "s.json"
    s.write_text(json.dumps({"results": []}))
    f = tmp_path / "f.json"
    f.write_text(json.dumps({"vulnerabilities": []}))
    args = ["--bandit", str(b), "--semgrep", str(s), "--safety", str(f), "--waivers", str(tmp_path / "none.json")]
    assert sg.main(args) == 1
    b.write_text(json.dumps(LOW))
    assert sg.main(args) == 0
