"""Required CI gates for apgi-research (tracker XC-CI-04.1-.4).

Each gate returns a list of error strings and is a pure function of the parsed
protocol documents, so a mutation test can prove it actually fails:

  trial-sequencing      trial counts compose exactly (blocks, catch/signal splits)
  protocol-locking      MANIFEST hashes lock every file; lock metadata is coherent
  timing-contracts      windows are ordered and fall on whole samples
  parallel-form-balance declared counterbalancing can be realised with equal cells
  item-bank             item bank passes the content linter; reliability targets stay frozen

Usage: python scripts/ci_gates.py [--gate NAME ...]   (exit 1 on any failure)
"""

import argparse
import importlib.util
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "validate_protocols", ROOT / "scripts" / "validate_protocols.py"
)
_vp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vp)

PROTOCOLS = ROOT / "protocols"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def load_documents(folder=PROTOCOLS):
    return {
        p.name: _vp.read_json(p)
        for p in sorted(pathlib.Path(folder).glob("protocol_*.json"))
    }


# ---------------------------------------------------------------------------
# Gate: trial sequencing
# ---------------------------------------------------------------------------


def trial_sequencing_errors(documents):
    errors = []
    for name, d in documents.items():
        b = d["blocks"]
        n_blocks, per_block, total = (
            b.get("experimental_blocks"),
            b.get("trials_per_block"),
            b.get("total_experimental_trials"),
        )
        if None not in (n_blocks, per_block, total) and n_blocks * per_block != total:
            errors.append(f"{name}: {n_blocks} x {per_block} != total {total}")
        if (b.get("practice_trials") or 0) < 0:
            errors.append(f"{name}: negative practice_trials")

        # Per-protocol composition invariants: parts must tile the block/total.
        if name.startswith("protocol_1_"):
            parts = (
                b["main_signal_trials_per_session"] + b["main_catch_trials_per_session"]
            )
            if parts != per_block:
                errors.append(
                    f"{name}: signal+catch {parts} != trials_per_block {per_block}"
                )
        if name.startswith("protocol_4_"):
            levels = len(b["contrast_multipliers_c50"])
            catch = round(per_block * b["catch_fraction"])
            if (per_block - catch) % levels:
                errors.append(
                    f"{name}: signal trials not divisible across {levels} contrasts"
                )
        if name.startswith("protocol_5_"):
            if (
                b["visual_trials_per_session"] + b["heartbeat_trials_per_session"]
                != per_block
            ):
                errors.append(f"{name}: visual+heartbeat != trials_per_block")
            if (
                b["tms_visual_trials"] + b["no_tms_visual_trials"]
                != b["visual_trials_per_session"]
            ):
                errors.append(f"{name}: tms+no_tms != visual_trials_per_session")
        if name.startswith("protocol_6_"):
            kinds = (
                b["catch_trials"]
                + b["suprathreshold_trials"]
                + b["near_threshold_trials"]
            )
            if kinds != total:
                errors.append(f"{name}: trial kinds {kinds} != total {total}")
            near = (len(b["absolute_michelson_contrasts"]) - 2) * b[
                "near_threshold_per_intermediate_contrast"
            ]
            if near != b["near_threshold_trials"]:
                errors.append(
                    f"{name}: near-threshold grid {near} != near_threshold_trials"
                )
            if sum(b["attention_conditions"].values()) != n_blocks:
                errors.append(
                    f"{name}: attention-condition blocks != experimental_blocks"
                )
    return errors


# ---------------------------------------------------------------------------
# Gate: protocol locking
# ---------------------------------------------------------------------------


def protocol_locking_errors(documents, folder=PROTOCOLS):
    errors = list(_vp.manifest_errors(pathlib.Path(folder)))
    for name, d in documents.items():
        if d.get("version") != _vp.RELEASE:
            errors.append(
                f"{name}: version {d.get('version')} != release {_vp.RELEASE}"
            )
        for key in ("input_sha256", "input_schema_sha256"):
            if not SHA256_RE.match(str(d.get("provenance", {}).get(key, ""))):
                errors.append(f"{name}: provenance.{key} missing or not a SHA-256")
        status = d.get("study_status")
        if status != "pre_collection":
            # Past pre-collection the registration and ethics lock must be real.
            if not d["readiness"].get("registration_verified"):
                errors.append(
                    f"{name}: study_status {status} without verified registration"
                )
            if d["ethics"].get("approval_status") != "approved":
                errors.append(f"{name}: study_status {status} without approved ethics")
        if (
            d["ethics"].get("enrollment_authorized")
            and d["ethics"].get("approval_status") != "approved"
        ):
            errors.append(f"{name}: enrollment authorised without approved ethics")
    return errors


# ---------------------------------------------------------------------------
# Gate: timing contracts
# ---------------------------------------------------------------------------


def _walk(node, path=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield path + (k,), v
            yield from _walk(v, path + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, path + (i,))


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def timing_contract_errors(documents):
    errors = []
    for name, d in documents.items():
        for path, value in _walk(d):
            key = path[-1] if path else ""
            if not isinstance(key, str):
                continue
            where = f"{name}:{'.'.join(map(str, path))}"
            # SOAs are signed (negative = before the reference event); every
            # other scalar *_ms is a duration or exclusion span and cannot be.
            if (
                key.endswith("_ms")
                and "soa" not in key
                and _is_num(value)
                and value < 0
            ):
                errors.append(f"{where}: negative duration {value}")
            if (key.endswith("_ms") or key.endswith("_hz")) and (
                isinstance(value, list)
                and len(value) == 2
                and all(_is_num(v) for v in value)
            ):
                if not value[0] < value[1]:
                    errors.append(f"{where}: window {value} is not increasing")

        eeg = d.get("eeg_analysis_pipeline")
        if not eeg:
            continue
        rate = eeg.get("sample_rate_hz")
        if not _is_num(rate) or rate <= 0:
            errors.append(f"{name}: eeg sample_rate_hz missing")
            continue
        samples_per_ms = rate / 1000.0
        for key, value in eeg.items():
            if key.endswith("_ms") and isinstance(value, list):
                for v in value:
                    if (
                        _is_num(v)
                        and abs(v * samples_per_ms - round(v * samples_per_ms)) > 1e-9
                    ):
                        errors.append(
                            f"{name}:{key}: {v} ms is not a whole sample at {rate} Hz"
                        )
        base, epoch = eeg.get("stimulus_baseline_ms"), eeg.get("stimulus_epoch_ms")
        if base and epoch and not (epoch[0] <= base[0] and base[1] <= epoch[1]):
            errors.append(f"{name}: stimulus_baseline_ms outside stimulus_epoch_ms")
        win, hep = eeg.get("hep_window_ms"), eeg.get("hep_epoch_ms")
        if win and hep and not (hep[0] <= win[0] and win[1] <= hep[1]):
            errors.append(f"{name}: hep_window_ms outside hep_epoch_ms")
    return errors


# ---------------------------------------------------------------------------
# Gate: parallel-form / condition-order balance
# ---------------------------------------------------------------------------


def williams_square(n):
    """Rows of a Williams design: every condition once per row and per column;
    for even n every ordered pair of neighbours occurs exactly once."""
    seq, lo, hi = [], 0, n - 1
    for i in range(n):
        seq.append(lo if i % 2 == 0 else hi)
        lo, hi = (lo + 1, hi) if i % 2 == 0 else (lo, hi - 1)
    return [[(v + r) % n for v in seq] for r in range(n)]


def parallel_form_balance_errors(documents):
    """Any protocol that declares counterbalancing over a session/form list
    must be able to fill every Williams row equally within each site."""
    errors = []
    for name, d in documents.items():
        ts = d.get("trial_structure", {})
        sessions = ts.get("sessions")
        if not ts.get("counterbalancing") or not isinstance(sessions, list):
            continue
        n = len(sessions)
        if n % 2:
            errors.append(
                f"{name}: odd number of conditions ({n}) cannot be pair-balanced"
            )
        square = williams_square(n)
        for col in range(n):
            if sorted(row[col] for row in square) != list(range(n)):
                errors.append(f"{name}: Williams design unbalanced at position {col}")
        pairs = {(r[i], r[i + 1]) for r in square for i in range(n - 1)}
        if n % 2 == 0 and len(pairs) != n * (n - 1):
            errors.append(f"{name}: Williams design has carry-over imbalance")
        sites = d["participants"].get("sites", 1)
        # Enrolment targets must tile the order cells; analyzable_target is a
        # post-attrition estimate and cannot be balanced by design.
        for key in ("n_target", "n_minimum"):
            total = d["participants"].get(key)
            if total is not None and (total % sites or (total // sites) % n):
                errors.append(
                    f"{name}: participants.{key}={total} does not split into equal "
                    f"order cells ({n} orders x {sites} sites)"
                )
    return errors


def item_bank_errors(documents=None, bank=None, folder=None):
    """Lint the item bank and verify the frozen reliability targets (S1-L1-05..17)."""
    import hashlib
    import json
    import sys

    sys.path.insert(0, str(ROOT))
    from item_bank import lint

    folder = pathlib.Path(folder) if folder else ROOT / "item_bank"
    errors = list(
        lint.lint_bank(
            bank if bank is not None else lint.load_bank(folder / "candidates.json")
        )
    )
    targets_path = folder / "reliability_targets.json"
    lock = (folder / "reliability_targets.sha256").read_text(encoding="utf-8").strip()
    if hashlib.sha256(targets_path.read_bytes()).hexdigest() != lock:
        errors.append("reliability_targets.json changed without updating its lock hash")
    t = json.loads(targets_path.read_text(encoding="utf-8"))
    checks = [
        ("omega >= 0.80", t["internal_consistency"]["minimum_per_subscale"] >= 0.80),
        ("ICC(2,1) >= 0.70", t["test_retest"]["minimum"] >= 0.70),
        ("retest interval 2-3 weeks", t["test_retest"]["interval_days"] == [14, 21]),
        (
            "6-month stability reported",
            t["long_term_stability"]["interval_months"] == 6,
        ),
        (
            "EFA/CFA on independent samples",
            t["factor_structure"]["samples"] == "independent",
        ),
        (
            "self-report vs behaviour r >= 0.15",
            t["self_report_vs_behaviour"]["minimum_abs"] >= 0.15,
        ),
        ("status frozen", t.get("status") == "frozen"),
    ]
    errors.extend(
        f"reliability targets: {name} not met" for name, ok in checks if not ok
    )
    return errors


GATES = {
    "trial-sequencing": trial_sequencing_errors,
    "protocol-locking": protocol_locking_errors,
    "timing-contracts": timing_contract_errors,
    "parallel-form-balance": parallel_form_balance_errors,
    "item-bank": item_bank_errors,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gate", action="append", choices=sorted(GATES))
    args = parser.parse_args(argv)
    documents = load_documents()
    failed = False
    for name in args.gate or sorted(GATES):
        errors = GATES[name](documents)
        print(f"{'FAIL' if errors else 'OK  '} {name}")
        for e in errors:
            print(f"     {e}", file=sys.stderr)
        failed = failed or bool(errors)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
