"""Meta-d' by maximum likelihood (Maniscalco & Lau, 2012), with response-
specific variants for present vs absent reports (Maniscalco & Lau, 2014)
(S3-BAT-01.1.5).

This is the per-session, non-hierarchical estimator. The registered analysis
asks for a *hierarchical* Bayesian fit (HMeta-d, Fleming 2017), which needs a
sampler (JAGS/Stan) and a multi-participant dataset; `fit_meta_d` supplies the
per-session likelihood fit and the data structure that estimator consumes, and
must not be reported as the hierarchical result."""

from statistics import NormalDist
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize

_N = NormalDist()
_PHI = np.vectorize(_N.cdf)


def count_table(
    present: Sequence[bool], said_present: Sequence[bool], confidence: Sequence[int],
    n_levels: int,
) -> np.ndarray:
    """counts[stimulus(0 absent,1 present), response(0 absent,1 present), rating-1]."""
    t = np.zeros((2, 2, n_levels))
    for p, r, c in zip(present, said_present, confidence):
        if not 1 <= int(c) <= n_levels:
            raise ValueError(f"confidence {c} outside 1..{n_levels}")
        t[int(bool(p)), int(bool(r)), int(c) - 1] += 1
    return t


def _type1(counts: np.ndarray) -> tuple[float, float]:
    tot = counts.sum(axis=2)  # [stim, resp]
    h = (tot[1, 1] + 0.5) / (tot[1].sum() + 1.0)
    f = (tot[0, 1] + 0.5) / (tot[0].sum() + 1.0)
    return _N.inv_cdf(h) - _N.inv_cdf(f), -0.5 * (_N.inv_cdf(h) + _N.inv_cdf(f))


def _cdf(x: float) -> float:
    return _N.cdf(x) if np.isfinite(x) else (1.0 if x > 0 else 0.0)


def _cell_probs(meta_d: float, c1: float, inc_s2: np.ndarray, inc_s1: np.ndarray):
    """P(rating k, response | stimulus) for both stimuli; shape [stim, resp, K]."""
    k = len(inc_s2) + 1
    t2 = np.concatenate([[c1], c1 + np.cumsum(inc_s2), [np.inf]])
    t1 = np.concatenate([[c1], c1 - np.cumsum(inc_s1), [-np.inf]])
    out = np.zeros((2, 2, k))
    for s, mu in ((0, -meta_d / 2), (1, meta_d / 2)):
        for j in range(k):
            out[s, 1, j] = _cdf(mu - t2[j]) - _cdf(mu - t2[j + 1])
            out[s, 0, j] = _cdf(t1[j] - mu) - _cdf(t1[j + 1] - mu)
    return out


def fit_meta_d(
    counts: np.ndarray, response: Optional[str] = None, seed: int = 0, restarts: int = 4
) -> dict[str, float]:
    """MLE meta-d'. `response` None fits all trials; "present"/"absent" fits only
    the type-2 data conditional on that report (response-specific meta-d')."""
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 3 or counts.shape[:2] != (2, 2):
        raise ValueError("counts must have shape (2, 2, n_levels)")
    k = counts.shape[2]
    if k < 2:
        raise ValueError("need >= 2 confidence levels")
    if counts.sum(axis=(1, 2)).min() < 10:
        raise ValueError("need >= 10 trials per stimulus class")
    adj = counts + 1.0 / (2 * 2 * k)
    d1, c1 = _type1(counts)
    ratio = c1 / d1 if abs(d1) > 1e-6 else 0.0
    sel = {None: (0, 1), "present": (1,), "absent": (0,)}.get(response)
    if sel is None:
        raise ValueError("response must be None, 'present' or 'absent'")
    rng = np.random.default_rng(seed)

    def nll(theta: np.ndarray) -> float:
        meta_d = theta[0]
        inc2 = np.exp(theta[1:k])
        inc1 = np.exp(theta[k:2 * k - 1])
        p = np.clip(_cell_probs(meta_d, ratio * meta_d, inc2, inc1), 1e-12, 1.0)
        total = 0.0
        for r in sel:
            for s in (0, 1):
                total -= float((adj[s, r] * np.log(p[s, r])).sum())
        return total

    best = None
    for i in range(restarts):
        x0 = np.concatenate([[d1 * rng.uniform(0.6, 1.2)], rng.normal(-0.5, 0.3, 2 * (k - 1))])
        res = minimize(nll, x0, method="L-BFGS-B",
                       bounds=[(-6, 6)] + [(-4, 2)] * (2 * (k - 1)))
        if best is None or res.fun < best.fun:
            best = res
    meta_d = float(best.x[0])
    if not (np.isfinite(meta_d) and np.isfinite(best.fun)):
        raise ArithmeticError("meta-d' fit did not converge")
    return {"d_prime": d1, "criterion_c": c1, "meta_d_prime": meta_d,
            "m_ratio": meta_d / d1 if abs(d1) > 1e-6 else float("nan"),
            "neg_log_likelihood": float(best.fun), "response": response or "all",
            "estimator": "mle_non_hierarchical"}


def meta_d_by_report(counts: np.ndarray, seed: int = 0) -> dict[str, dict[str, float]]:
    """Separate present and absent fits (the registered split)."""
    return {"present": fit_meta_d(counts, "present", seed), "absent": fit_meta_d(counts, "absent", seed)}
