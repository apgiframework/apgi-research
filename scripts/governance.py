"""Governance tooling for APGI-VAL-SPEC-1.2 evidence (AUDIT-GOV-*, APGI-REL-*).

  check              validate governance records; scan for unsupported claims and
                     non-canonical APGI expansions (SCI-006/007)
  impact FILE...     map changed files to requirements and select regression tests (REL-003)
  release-manifest   write governance/release_set.json (identity, protocol lock,
                     schema hashes, dependency inventory) and governance/sbom.cdx.json (IMP-001, REL-004)
  certify RESULTS    refuse unless every one of the twelve mandatory checks passes with
                     approvals present; never writes a certificate otherwise (REL-001)

Exit status is 1 whenever a check fails or certification is refused.
"""

import argparse
import datetime
import hashlib
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
GOV = ROOT / "governance"

# Canonical expansion only (spec SCI-006). Prior non-canonical expansions are
# matched by their distinctive word pairs so this file never spells them out.
CANONICAL = "Allostatic Precision-Gated Ignition"
NON_CANONICAL = [r"Autonomous Program Generation", r"Autonomous Psychological General", r"Autonomous Personal Growth",
                 r"Active Perception and Generative", r"Active Predictive Global Ignition",
                 r"Adaptive Precision[- ]Gated"]
UNSUPPORTED = re.compile(r"100/100\s+APGI|perfect(ly)?\s+APGI[- ]compli|fully\s+validated\s+APGI", re.I)
SCAN_SUFFIX = {".py", ".md", ".cff", ".toml", ".json", ".yaml", ".yml"}
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "test_reports", "governance"}
SKIP_FILES = {"governance.py", "test_governance_tooling.py"}

MANDATORY_CHECKS = ["Release identity", "Protocol lock", "Scientific conformance", "Reproducibility and deviations",
                    "Data and API validation", "Security", "Cross-application integration", "Release package",
                    "Consent and deletion", "Acquisition and validation decisions", "Client conformance", "Approvals"]


def _files():
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix in SCAN_SUFFIX and not (set(p.relative_to(ROOT).parts) & SKIP_DIRS) \
                and p.name not in SKIP_FILES and not p.name.startswith("task_register"):
            yield p


def scan_text_claims():
    problems = []
    for p in _files():
        text = p.read_text(encoding="utf-8", errors="ignore")
        for pat in NON_CANONICAL:
            if re.search(pat, text, re.I):
                problems.append(f"{p.relative_to(ROOT)}: non-canonical APGI expansion /{pat}/")
        if UNSUPPORTED.search(text) and "internal" not in text.lower()[:20000] and "checklist" not in text.lower():
            problems.append(f"{p.relative_to(ROOT)}: unsupported compliance claim without internal-checklist scope")
    return problems


def record_errors(root=GOV, today=None):
    today = today or datetime.date.today()
    errors = []
    load = lambda n: json.loads((root / n).read_text(encoding="utf-8"))
    ap = load("approvals.json")
    for role, v in ap["roles"].items():
        if v["holder"] is None and v["status"] != "Not appointed":
            errors.append(f"approvals: unfilled role {role} must read 'Not appointed'")
    for a in ap["approvals"]:
        if ap["roles"].get(a.get("role"), {}).get("holder") is None:
            errors.append(f"approvals: approval by unappointed role {a.get('role')}")
    ex = load("exceptions.json")
    for e in ex["exceptions"]:
        for k in ("requirement", "impact", "compensating_controls", "owner", "expires", "remediation_plan", "approver"):
            if not e.get(k):
                errors.append(f"exceptions: {e.get('requirement', '?')} missing {k}")
        if e.get("requirement") in ex["non_waivable"]:
            errors.append(f"exceptions: {e['requirement']} is non-waivable")
        if e.get("expires") and datetime.date.fromisoformat(e["expires"]) < today:
            errors.append(f"exceptions: {e['requirement']} expired {e['expires']}")
    cl = load("claim_register.json")
    ids = [c["id"] for c in cl["claims"]]
    if len(set(ids)) != len(ids) or not all(re.fullmatch(r"CLM-\d{3}", i) for i in ids):
        errors.append("claim_register: ids must be unique CLM-nnn")
    for c in cl["claims"]:
        if c.get("epistemic_status") not in ("implementation_fact", "simulation_only", "registered_prediction", "not_claimed"):
            errors.append(f"claim_register: {c['id']} has invalid epistemic_status")
    rm = load("requirements_map.json")["requirements"]
    for rid, r in rm.items():
        for path in r["paths"] + r["tests"]:
            if not (ROOT / path).exists():
                errors.append(f"requirements_map: {rid}: missing path {path}")
    return errors


def impact(changed, req_map=None):
    rm = req_map or json.loads((GOV / "requirements_map.json").read_text(encoding="utf-8"))["requirements"]
    hit, tests = {}, set()
    for f in changed:
        for rid, r in rm.items():
            if any(f == p or (p.endswith("/") and f.startswith(p)) for p in r["paths"]):
                hit.setdefault(rid, []).append(f)
                tests.update(r["tests"])
    return {"changed": sorted(changed), "requirements": hit, "regression_tests": sorted(tests),
            "unmapped_changes": sorted(f for f in changed if not any(f in v for v in hit.values()))}


def _sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_release_manifest():
    import tomllib
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "protocols" / "MANIFEST.json").read_text(encoding="utf-8"))
    dirty = bool(_git("status", "--porcelain"))
    return {
        "component": "apgi-research", "version": meta["project"]["version"],
        "git_commit": _git("rev-parse", "HEAD"), "working_tree_dirty": dirty,
        "protocol_release": lock["release"], "protocol_manifest_sha256": _sha(ROOT / "protocols" / "MANIFEST.json"),
        "protocol_files": {f["path"]: f["sha256"] for f in lock["files"]},
        "schema_hashes": {p.name: _sha(p) for p in sorted((ROOT / "protocols" / "schemas").glob("*.json"))},
        "reference_contract_sha256": _sha(ROOT / "mathematical-contracts.yaml"),
        "dependency_declarations": {"pyproject_sha256": _sha(ROOT / "pyproject.toml"),
                                    "requirements_sha256": _sha(ROOT / "requirements.txt")},
        "dependency_lock_present": any((ROOT / n).exists() for n in ("requirements.lock", "uv.lock", "poetry.lock")),
        "note": "Identity record for this checkout. It is not a certificate; a dirty tree cannot be certified.",
        "certified": False,
    }


def build_sbom():
    from importlib import metadata
    comps = sorted(({"type": "library", "name": d.metadata["Name"], "version": d.version}
                    for d in metadata.distributions() if d.metadata["Name"]), key=lambda c: c["name"].lower())
    return {"bomFormat": "CycloneDX", "specVersion": "1.5", "version": 1,
            "metadata": {"component": {"type": "application", "name": "apgi-research"}},
            "components": comps}


def certify(results_path):
    """results: {"checks": {name: "PASS"|"FAIL"|"BLOCKED"}, "release_digests": {...}}"""
    res = json.loads(pathlib.Path(results_path).read_text(encoding="utf-8"))
    refusals = [f"{n}: {res.get('checks', {}).get(n, 'MISSING')}" for n in MANDATORY_CHECKS
                if res.get("checks", {}).get(n) != "PASS"]
    ap = json.loads((GOV / "approvals.json").read_text(encoding="utf-8"))
    for role in ("Release authority", "Scientific authority", "Specification owner"):
        if ap["roles"][role]["holder"] is None:
            refusals.append(f"approval: {role} Not appointed")
    if not res.get("release_digests"):
        refusals.append("no release digests supplied")
    return refusals


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    pi = sub.add_parser("impact"); pi.add_argument("files", nargs="+")
    sub.add_parser("release-manifest")
    pc = sub.add_parser("certify"); pc.add_argument("results")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        problems = record_errors() + scan_text_claims()
        for p in problems:
            print(f"FAIL {p}", file=sys.stderr)
        print(f"{'FAIL' if problems else 'OK  '} governance check ({len(problems)} problems)")
        return 1 if problems else 0
    if a.cmd == "impact":
        print(json.dumps(impact(a.files), indent=2))
        return 0
    if a.cmd == "release-manifest":
        (GOV / "release_set.json").write_text(json.dumps(build_release_manifest(), indent=2) + "\n", encoding="utf-8")
        (GOV / "sbom.cdx.json").write_text(json.dumps(build_sbom(), indent=1) + "\n", encoding="utf-8")
        print("wrote governance/release_set.json and governance/sbom.cdx.json")
        return 0
    refusals = certify(a.results)
    for r in refusals:
        print(f"REFUSED {r}", file=sys.stderr)
    print("CERTIFICATION REFUSED" if refusals else "all mandatory checks pass; certificate may be issued by the Release authority")
    return 1 if refusals else 0


if __name__ == "__main__":
    sys.exit(main())
