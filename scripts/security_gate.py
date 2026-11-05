"""Fail the build on unresolved high/critical security findings (AUDIT-CI-001).

Reads the JSON written by bandit, semgrep and safety and exits 1 when any
finding at or above the threshold is not covered by an unexpired waiver in
``security-waivers.json`` (each waiver needs id, owner, reason and an ISO
``expires`` date). A missing report is a failure: a scanner that did not run
must never read as a clean result.

Usage: python scripts/security_gate.py [--bandit F] [--semgrep F] [--safety F]
"""

import argparse
import datetime
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
WAIVERS = ROOT / "security-waivers.json"


def load_waivers(path=WAIVERS, today=None):
    today = today or datetime.date.today()
    if not pathlib.Path(path).is_file():
        return {}, []
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    active, errors = {}, []
    for w in data.get("waivers", []):
        missing = [k for k in ("id", "owner", "reason", "expires") if not w.get(k)]
        if missing:
            errors.append(f"waiver {w.get('id', '?')}: missing {missing}")
            continue
        if datetime.date.fromisoformat(w["expires"]) < today:
            errors.append(f"waiver {w['id']} expired {w['expires']}")
            continue
        active[w["id"]] = w
    return active, errors


def bandit_findings(report):
    return [
        (f"bandit:{r['test_id']}:{r['filename']}:{r['line_number']}", r["issue_severity"])
        for r in report.get("results", [])
        if r.get("issue_severity", "").upper() in ("HIGH", "CRITICAL")
    ]


def semgrep_findings(report):
    out = []
    for r in report.get("results", []):
        sev = str(r.get("extra", {}).get("severity", "")).upper()
        if sev in ("ERROR", "CRITICAL", "HIGH"):
            out.append((f"semgrep:{r['check_id']}:{r['path']}:{r['start']['line']}", sev))
    return out


def safety_findings(report):
    vulns = report.get("vulnerabilities", []) if isinstance(report, dict) else []
    return [(f"safety:{v.get('vulnerability_id') or v.get('CVE')}:{v.get('package_name')}", "HIGH") for v in vulns]


def evaluate(reports, waivers):
    """reports: {tool: parsed_json or None}. Returns list of failure strings."""
    parsers = {"bandit": bandit_findings, "semgrep": semgrep_findings, "safety": safety_findings}
    failures = []
    for tool, report in reports.items():
        if report is None:
            failures.append(f"{tool}: report missing - scanner did not run")
            continue
        for fid, sev in parsers[tool](report):
            if fid not in waivers and fid.split(":")[1] not in waivers:
                failures.append(f"{fid} [{sev}]")
    return failures


def _read(path):
    p = pathlib.Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bandit", default="bandit-results.json")
    ap.add_argument("--semgrep", default="semgrep-results.json")
    ap.add_argument("--safety", default="safety-results.json")
    ap.add_argument("--waivers", default=str(WAIVERS))
    args = ap.parse_args(argv)
    waivers, werrors = load_waivers(args.waivers)
    failures = werrors + evaluate(
        {"bandit": _read(args.bandit), "semgrep": _read(args.semgrep), "safety": _read(args.safety)},
        waivers,
    )
    for f in failures:
        print(f"FAIL {f}", file=sys.stderr)
    print(f"{'FAIL' if failures else 'OK  '} security gate ({len(failures)} unresolved)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
