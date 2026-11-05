import json

import numpy as np
import pytest

from item_bank import AGREEMENT_5, lint_bank, score_subscales
from item_bank.scoring import (cronbach_alpha, evaluate_against_targets, icc_2_1,
                               mcdonald_omega)

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


def mk(i, sub, rev=False):
    d = {"item_id": f"I{i}", "block": "A", "subscale": sub, "text": "I notice small changes around me.",
         "response_type": "agreement_5", "options": list(AGREEMENT_5)}
    if rev:
        d["reverse_keyed"] = True
    return d


BANK = {"scoring_version": "score-1.0.0", "block_order": ["A"],
        "items": [mk(1, "s"), mk(2, "s"), mk(3, "s", rev=True), mk(4, "s"), mk(5, "s")]}


def test_bank_lints_and_requires_scoring_version():
    assert lint_bank(BANK) == []
    bad = {k: v for k, v in BANK.items() if k != "scoring_version"}
    assert any("scoring_version" in e for e in lint_bank(bad))
    b2 = json.loads(json.dumps(BANK))
    b2["items"][0]["reverse_keyed"] = "yes"
    assert any("reverse_keyed" in e for e in lint_bank(b2))


def test_reverse_keying_missingness_and_version():
    out = score_subscales(BANK, {"I1": 5, "I2": 5, "I3": 1, "I4": 5, "I5": 5})
    assert out["scores"]["s"] == 5.0 and out["scoring_version"] == "score-1.0.0"
    assert score_subscales(BANK, {"I1": 5, "I2": 5, "I3": 1, "I4": 5})["scores"]["s"] == 5.0  # 4/5 = 80%
    assert score_subscales(BANK, {"I1": 5, "I2": 5, "I3": 1})["scores"]["s"] is None
    with pytest.raises(ValueError):
        score_subscales(BANK, {"I1": 6})
    with pytest.raises(ValueError):
        score_subscales(BANK, {"X": 3})
    with pytest.raises(ValueError):
        score_subscales({"items": []}, {})


def _sim(n=400, k=6, loading=0.8, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(n, 1))
    return loading * f + np.sqrt(1 - loading**2) * rng.normal(size=(n, k))


def test_reliability_estimators_recover_known_values():
    x = _sim()
    true_omega = (6 * 0.8) ** 2 / ((6 * 0.8) ** 2 + 6 * (1 - 0.64))
    assert mcdonald_omega(x) == pytest.approx(true_omega, abs=0.03)
    assert cronbach_alpha(x) == pytest.approx(true_omega, abs=0.04)
    rng = np.random.default_rng(1)
    true = rng.normal(size=(300, 1))
    two = true + rng.normal(scale=0.7, size=(300, 2))
    expect = 1 / (1 + 0.49)
    assert icc_2_1(two) == pytest.approx(expect, abs=0.05)
    with pytest.raises(ValueError):
        icc_2_1(np.ones((2, 2)))


def test_targets_evaluation_never_passes_unobserved():
    t = json.loads((ROOT / "item_bank/reliability_targets.json").read_text())
    r = evaluate_against_targets(t, {"omega_per_subscale": 0.85, "icc_2_1": 0.5})
    assert r["results"]["omega_per_subscale"]["result"] == "pass"
    assert r["results"]["icc_2_1"]["result"] == "fail"
    assert r["results"]["self_report_behaviour_abs_r"]["result"] == "not_observed"
