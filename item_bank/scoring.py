"""Versioned scoring and reliability estimators for the self-report bank
(S1-L1-05, S1-L1-17). The estimators evaluate observed data against the frozen
targets in reliability_targets.json; they never supply observed values."""

import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np

MIN_COMPLETE_FRACTION = 0.8


def score_subscales(bank: Mapping[str, Any], responses: Mapping[str, int]) -> dict[str, Any]:
    """Mean score (1-5) per subscale with reverse keying. A subscale with less
    than 80% of its items answered is None (not imputed). Result carries the
    bank's scoring_version so a score is never separated from its rules."""
    version = bank.get("scoring_version")
    if not version:
        raise ValueError("bank has no scoring_version")
    by_sub: dict[str, list[float]] = {}
    total: dict[str, int] = {}
    for item in bank["items"]:
        sub = item["subscale"]
        total[sub] = total.get(sub, 0) + 1
        r = responses.get(item["item_id"])
        if r is None:
            continue
        if r not in (1, 2, 3, 4, 5) or isinstance(r, bool):
            raise ValueError(f"{item['item_id']}: response must be an integer 1-5")
        by_sub.setdefault(sub, []).append(6 - r if item.get("reverse_keyed") else r)
    unknown = set(responses) - {i["item_id"] for i in bank["items"]}
    if unknown:
        raise ValueError(f"responses for unknown items: {sorted(unknown)}")
    scores = {
        sub: (sum(v) / len(v) if len(v) / total[sub] >= MIN_COMPLETE_FRACTION else None)
        for sub, v in ((s, by_sub.get(s, [])) for s in total)
    }
    return {"scoring_version": version, "scores": scores,
            "n_answered": {s: len(by_sub.get(s, [])) for s in total}}


def cronbach_alpha(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2 or x.shape[0] < 3:
        raise ValueError("need an (n_respondents >= 3, n_items >= 2) matrix")
    k = x.shape[1]
    return float(k / (k - 1) * (1 - x.var(axis=0, ddof=1).sum() / x.sum(axis=1).var(ddof=1)))


def mcdonald_omega(x: np.ndarray, iters: int = 200) -> float:
    """Omega total from a one-factor model fit by iterated principal axis."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] < 3 or x.shape[0] < 10:
        raise ValueError("need an (n_respondents >= 10, n_items >= 3) matrix")
    r = np.corrcoef(x, rowvar=False)
    h = 1 - 1 / np.diag(np.linalg.inv(r))  # initial communalities (SMC)
    for _ in range(iters):
        rr = r.copy()
        np.fill_diagonal(rr, h)
        w, v = np.linalg.eigh(rr)
        lam = v[:, -1] * math.sqrt(max(w[-1], 0))
        if lam.sum() < 0:
            lam = -lam
        new_h = lam ** 2
        if np.max(np.abs(new_h - h)) < 1e-8:
            h = new_h
            break
        h = new_h
    psi = np.clip(1 - h, 1e-6, None)
    return float(lam.sum() ** 2 / (lam.sum() ** 2 + psi.sum()))


def icc_2_1(x: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single measure.
    Rows are subjects, columns are occasions."""
    x = np.asarray(x, dtype=float)
    n, k = x.shape
    if n < 3 or k < 2:
        raise ValueError("need >= 3 subjects and >= 2 occasions")
    gm = x.mean()
    msr = k * ((x.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(axis=0) - gm) ** 2).sum() / (k - 1)
    sse = ((x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def evaluate_against_targets(targets: Mapping[str, Any], observed: Mapping[str, float]) -> dict[str, Any]:
    """Compare *observed* estimates with the frozen targets. Missing observed
    values are reported as 'not_observed', never as pass."""
    spec = {
        "omega_per_subscale": targets["internal_consistency"]["minimum_per_subscale"],
        "icc_2_1": targets["test_retest"]["minimum"],
        "self_report_behaviour_abs_r": targets["self_report_vs_behaviour"]["minimum_abs"],
    }
    out = {}
    for key, minimum in spec.items():
        val: Optional[float] = observed.get(key)
        out[key] = {"minimum": minimum, "observed": val,
                    "result": "not_observed" if val is None else ("pass" if abs(val) >= minimum else "fail")}
    return {"targets_version": targets["targets_version"], "results": out}
