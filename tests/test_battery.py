"""Battery analysis and design rules: SDT, psychometric fits, reversal model,
tiers, timing record, forms."""

import math

import numpy as np
import pytest

from battery import forms, psychometric, reversal, sdt, tiers
from battery.timing import TrialTimingRecord


class TestSDT:
    def test_known_values(self):
        # 84/100 hits, 16/100 false alarms -> d' ~ 1.99, c ~ 0
        out = sdt.dprime_c(84, 16, 16, 84)
        assert out["d_prime"] == pytest.approx(1.98, abs=0.05)
        assert out["criterion_c"] == pytest.approx(0.0, abs=1e-9)

    def test_liberal_bias_gives_negative_c(self):
        assert sdt.dprime_c(95, 5, 40, 60)["criterion_c"] < 0

    def test_extremes_are_finite(self):
        out = sdt.dprime_c(50, 0, 0, 50)
        assert math.isfinite(out["d_prime"]) and out["d_prime"] > 0

    def test_validation(self):
        with pytest.raises(ValueError):
            sdt.dprime_c(0, 0, 5, 5)
        with pytest.raises(ValueError):
            sdt.dprime_c(-1, 5, 5, 5)

    def test_trial_vectors_and_confidence(self):
        present = [True] * 4 + [False] * 4
        said = [True, True, True, False, True, False, False, False]
        conf = [4, 3, 5, 2, 1, 2, 2, 3]
        assert sdt.counts_from_trials(present, said) == (3, 1, 1, 3)
        out = sdt.dprime_by_presence(present, said, conf)
        assert out["mean_confidence_present"] == pytest.approx(3.5)
        assert out["mean_confidence_absent"] == pytest.approx(2.0)


class TestPsychometric:
    def test_recovers_threshold(self):
        true_mu, true_sigma = math.log10(0.05), 0.15
        c = np.logspace(-2, 0, 8)
        p = 1 / (1 + np.exp(-(np.log10(c) - true_mu) / true_sigma))
        p = 0.0 + (1 - 0.0 - 0.02) * p
        n = np.full(8, 200)
        k = np.round(p * n).astype(int)
        fit = psychometric.fit_psychometric(c, k, n)
        assert fit["threshold"] == pytest.approx(0.05, rel=0.08)
        assert fit["slope"] == pytest.approx(1 / true_sigma, rel=0.25)

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            psychometric.fit_psychometric([0.0, 0.1], [1, 2], [5, 5])
        with pytest.raises(ValueError):
            psychometric.fit_psychometric([0.1, 0.2], [6, 2], [5, 5])

    def test_confidence_function(self):
        c = np.logspace(-2, 0, 7)
        y = 1 + 4 / (1 + np.exp(-(np.log10(c) + 1.3) / 0.2))
        out = psychometric.fit_confidence_function(c, y)
        assert out["midpoint"] == pytest.approx(10**-1.3, rel=0.1)
        with pytest.raises(ValueError):
            psychometric.fit_confidence_function(c, [3] * 7)

    def test_device_relative_report(self):
        dev = {"refresh_hz": 60, "gamma": 2.2, "bit_depth": 8, "screen_px": [1440, 900]}
        out = psychometric.device_relative_threshold(
            0.04, dev, "A", "tier_config-1.0.0"
        )
        assert (
            out["scale"] == "device_relative"
            and out["comparable_across_devices"] is False
        )
        assert out["threshold_in_display_steps"] == pytest.approx(0.04 * 255)
        assert out["covariates"]["tier"] == "A"
        with pytest.raises(ValueError):
            psychometric.device_relative_threshold(0.04, {"refresh_hz": 60}, "A")
        with pytest.raises(ValueError):
            psychometric.device_relative_threshold(0.04, dev, "Z")


class TestReversal:
    @staticmethod
    def simulate(seed=1, n=400, a_pos=0.5, a_neg=0.2, beta=4.0, kappa=0.0):
        rng = np.random.default_rng(seed)
        q = np.zeros(2)
        prev = -1
        good = 0
        ch, rw = [], []
        for t in range(n):
            if t and t % 40 == 0:
                good = 1 - good
            logits = beta * q
            if prev >= 0:
                logits[prev] += kappa
            p = np.exp(logits - logits.max())
            p /= p.sum()
            c = int(rng.choice(2, p=p))
            r = float(rng.random() < (0.8 if c == good else 0.2))
            d = r - q[c]
            q[c] += (a_pos if d >= 0 else a_neg) * d
            ch.append(c)
            rw.append(r)
            prev = c
        return ch, rw

    def test_recovers_direction_of_parameters(self):
        ch, rw = self.simulate()
        fit = reversal.fit_reversal(ch, rw)
        assert fit["alpha_positive"] > fit["alpha_negative"]
        assert fit["reward_sensitivity"] > 1.5

    def test_perseveration_detected(self):
        lo = reversal.fit_reversal(*self.simulate(seed=3, kappa=0.0))
        hi = reversal.fit_reversal(*self.simulate(seed=3, kappa=1.5))
        assert hi["perseveration"] > lo["perseveration"]

    def test_deterministic(self):
        ch, rw = self.simulate(n=120)
        assert reversal.fit_reversal(ch, rw, seed=5) == reversal.fit_reversal(
            ch, rw, seed=5
        )

    def test_validation(self):
        with pytest.raises(ValueError):
            reversal.fit_reversal([0, 1], [1, 0])
        with pytest.raises(ValueError):
            reversal.fit_reversal([2] * 30, [1.0] * 30)

    def test_composite_exposes_only_the_product(self):
        out = reversal.precision_weighted_learning_composite(0.4, 2.5)
        assert out == {"precision_weighted_learning_composite": 1.0}
        with pytest.raises(ValueError):
            reversal.precision_weighted_learning_composite(-1, 1)


class TestTiers:
    def test_tiers(self):
        assert tiers.assign_tier(
            {"timing_errors_ms": [1, 2, 3], "fullscreen": True}
        ) == ("A", [])
        assert (
            tiers.assign_tier({"timing_errors_ms": [12] * 5, "fullscreen": True})[0]
            == "B"
        )
        assert tiers.assign_tier({"is_mobile": True, "fullscreen": True})[0] == "C"
        assert (
            tiers.assign_tier({"visibility_changes": 1, "fullscreen": True})[0] == "C"
        )
        assert (
            tiers.assign_tier({"failed_attention_checks": 2, "fullscreen": True})[0]
            == "C"
        )

    def test_version_is_stamped_and_matches_api(self):
        assert tiers.TIER_CONFIG["version"] == "tier_config-1.0.0"

    def test_distribution_keeps_quantiles(self):
        d = tiers.timing_error_distribution([1, 2, 3, 4, 100])
        assert d["max"] == 100 and d["median"] == 3 and d["n"] == 5


class TestTiming:
    def test_fields_stay_separate(self):
        r = TrialTimingRecord(0, 100.0, 104.0, 16.7, photodiode_onset_ms=108.0)
        assert r.callback_error_ms == 4.0 and r.photodiode_error_ms == 8.0
        assert r.timing_provenance == "photodiode"
        d = r.to_dict()
        assert {
            "callback_time_ms",
            "refresh_estimate_ms",
            "photodiode_onset_ms",
        } <= d.keys()

    def test_no_photodiode(self):
        r = TrialTimingRecord(1, 50.0, 52.0, 16.7)
        assert r.photodiode_error_ms is None and r.timing_provenance == "raf_callback"
        with pytest.raises(ValueError):
            TrialTimingRecord(1, 0, 0, 0)


class TestForms:
    def stimuli(self, n=30):
        return [{"id": f"s{i}", "difficulty": (i * 7) % 23 + i / 100} for i in range(n)]

    def test_equal_size_and_matched(self):
        out = forms.split_matched_forms(self.stimuli())
        rep = forms.balance_report(out)
        assert rep["sizes"] == [10, 10, 10]
        assert rep["max_mean_difference"] < 0.6
        flat = [s["id"] for f in out for s in f]
        assert sorted(flat) == sorted(
            s["id"] for s in self.stimuli()
        )  # every stimulus used once

    def test_deterministic_and_validation(self):
        assert forms.split_matched_forms(self.stimuli()) == forms.split_matched_forms(
            self.stimuli()
        )
        with pytest.raises(ValueError):
            forms.split_matched_forms(self.stimuli(31))
        with pytest.raises(ValueError):
            forms.split_matched_forms(self.stimuli(), n_forms=1)

    def test_session_rotation(self):
        order = ["B", "C", "A"]
        assert [forms.form_for_session(order, i) for i in range(1, 5)] == [
            "B",
            "C",
            "A",
            "B",
        ]
        with pytest.raises(ValueError):
            forms.form_for_session(order, 0)
        with pytest.raises(ValueError):
            forms.form_for_session([], 1)
