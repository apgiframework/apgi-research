"""Screening, calibration, staircase/QUEST, meta-d', reversal schedule, masking,
form counterbalancing and the construct bridge."""

import math

import numpy as np
import pytest

from battery import (bridge, calibration, forms, masking, metad, reversal_task,
                     screening, staircase)


class TestScreening:
    def test_fast_responses(self):
        assert screening.flag_fast_responses([100] * 5 + [400] * 5)["flag"]
        assert not screening.flag_fast_responses([400] * 20)["flag"]
        assert screening.flag_fast_responses([])["reason"] == "no_rts"

    def test_invariant(self):
        assert screening.flag_invariant_responding(["f"] * 30, [3] * 30)["flag"]
        assert not screening.flag_invariant_responding(["f"] * 30, [1, 2, 3] * 10)["flag"]
        assert not screening.flag_invariant_responding(["f", "j"] * 15, [3] * 30)["flag"]

    def test_straightline_and_zigzag(self):
        assert screening.flag_straightlining([3] * 10)["straightline"]
        zz = screening.flag_straightlining([1, 5] * 6)
        assert zz["zigzag"] and zz["flag"]
        assert not screening.flag_straightlining([1, 3, 2, 5, 4, 2, 1, 3, 5, 2])["flag"]
        assert screening.flag_straightlining([3, 3])["reason"] == "too_few_items"

    def test_attention_exclusion(self):
        assert not screening.exclude_from_norms(1) and screening.exclude_from_norms(2)
        with pytest.raises(ValueError):
            screening.exclude_from_norms(-1)

    def test_duplicates_flag_not_block(self):
        s = [{"session_id": "a", "device_fingerprint": "d", "email": "X@y.com "},
             {"session_id": "b", "device_fingerprint": "d", "email": "x@y.com"},
             {"session_id": "c", "device_fingerprint": "e", "email": "x@y.com"}]
        out = screening.flag_duplicates(s)
        assert out["a"]["duplicate"] and out["b"]["duplicate"] and not out["c"]["duplicate"]
        assert set(out) == {"a", "b", "c"}  # nothing removed

    def test_one_norm_session_per_participant_per_version(self):
        s = [{"session_id": "s2", "participant_id": "p", "battery_version": "1", "started_at": 2, "norm_eligible": True},
             {"session_id": "s1", "participant_id": "p", "battery_version": "1", "started_at": 1, "norm_eligible": True},
             {"session_id": "s3", "participant_id": "p", "battery_version": "2", "started_at": 3, "norm_eligible": True},
             {"session_id": "s0", "participant_id": "q", "battery_version": "1", "started_at": 0, "norm_eligible": False}]
        out = screening.select_norm_sessions(s)
        assert sorted(out["norm"]) == ["s1", "s3"] and sorted(out["not_norm"]) == ["s0", "s2"]

    def test_bot_form_screen(self):
        assert screening.screen_form_submission({"honeypot": "x", "total_seconds": 60})["flag"]
        assert "form_too_fast" in screening.screen_form_submission({"total_seconds": 1})["reasons"]
        assert not screening.screen_form_submission({"total_seconds": 60, "item_seconds": [3] * 10})["flag"]

    def test_session_and_report(self):
        bad = screening.screen_session({"session_id": "a", "rts_ms": [90] * 10, "keys": ["f"] * 30,
                                        "confidence": [3] * 30, "failed_attention_checks": 2})
        good = screening.screen_session({"session_id": "b", "rts_ms": [400] * 10, "keys": ["f", "j"] * 15})
        assert not bad["norm_eligible"] and good["norm_eligible"]
        rep = screening.exclusion_report([bad, good])
        assert rep["n_sessions"] == 2 and rep["rates"]["fast_responses"]["rate"] == 0.5


class TestCalibration:
    def test_display_timing(self):
        ts = [i * 16.667 for i in range(60)]
        out = calibration.measure_display_timing(ts)
        assert out["fps"] == pytest.approx(60, rel=0.01) and out["dropped_frame_fraction"] == 0
        ts2 = ts[:30] + [t + 16.667 for t in ts[30:]]
        assert calibration.measure_display_timing(ts2)["dropped_frame_fraction"] > 0
        with pytest.raises(ValueError):
            calibration.measure_display_timing(ts[:5])

    def test_scale_and_distance(self):
        ppmm = calibration.pixels_per_mm(342.4)
        assert ppmm == pytest.approx(4.0)
        true_d = 600.0
        offset_px = true_d * math.tan(math.radians(13.5)) * ppmm
        out = calibration.viewing_distance_mm([offset_px] * 3, ppmm)
        assert out["viewing_distance_mm"] == pytest.approx(true_d, rel=1e-6)
        with pytest.raises(ValueError):
            calibration.viewing_distance_mm([100, 100], ppmm)
        with pytest.raises(ValueError):
            calibration.pixels_per_mm(-1)

    def test_attention_checks(self):
        assert calibration.score_attention_checks([1, 2, 3], [1, 2, 3])["failed"] == 0
        assert calibration.score_attention_checks([1, 2, 3], [0, 0, 3])["excluded_from_norms"]


class TestAdaptive:
    def test_staircase_converges_near_truth(self):
        rng = np.random.default_rng(0)
        true_log = -1.4
        sc = staircase.Staircase(start_log10=-0.5)
        for _ in range(400):
            if sc.done:
                break
            p = 0.5 + 0.5 / (1 + math.exp(-(math.log10(sc.contrast) - true_log) / 0.1))
            sc.update(bool(rng.random() < p))
        assert sc.done
        assert abs(math.log10(sc.threshold()) - true_log) < 0.25

    def test_staircase_bounds(self):
        sc = staircase.Staircase(start_log10=0.0, hi=0.0)
        for _ in range(20):
            sc.update(False)
        assert sc.level <= 0.0
        with pytest.raises(ValueError):
            staircase.Staircase(start_log10=1.0)

    def test_quest_posterior_narrows(self):
        rng = np.random.default_rng(1)
        q = staircase.Quest()
        true_log = -1.2
        sd0 = q.estimate()["posterior_sd"]
        for _ in range(80):
            x = q.next_level()
            p = 0.5 + 0.48 * (1 - math.exp(-(10 ** (3.5 * (x - true_log)))))
            q.update(x, bool(rng.random() < p))
        est = q.estimate()
        assert est["posterior_sd"] < sd0 and abs(est["log10_threshold"] - true_log) < 0.3


class TestMetaD:
    @staticmethod
    def simulate(meta_ratio, n=4000, d=1.5, seed=0, k=4):
        rng = np.random.default_rng(seed)
        present = rng.random(n) < 0.5
        x = rng.normal(np.where(present, d / 2, -d / 2), 1.0)
        said = x > 0
        md = d * meta_ratio
        # type-2 evidence from a noisier copy: ratio sets its reliability
        x2 = rng.normal(np.where(present, md / 2, -md / 2), 1.0) if meta_ratio < 1 else x
        evid = np.abs(np.where(meta_ratio < 1, x2 * np.where(said, 1, -1), x))
        cuts = np.quantile(evid, np.linspace(0, 1, k + 1)[1:-1])
        conf = 1 + np.searchsorted(cuts, evid)
        return metad.count_table(present, said, conf, k)

    def test_ideal_observer_ratio_near_one(self):
        out = metad.fit_meta_d(self.simulate(1.0))
        assert out["m_ratio"] == pytest.approx(1.0, abs=0.25)

    def test_noisier_metacognition_lowers_ratio(self):
        hi = metad.fit_meta_d(self.simulate(1.0))["m_ratio"]
        lo = metad.fit_meta_d(self.simulate(0.3, seed=2))["m_ratio"]
        assert lo < hi

    def test_present_absent_split_and_validation(self):
        out = metad.meta_d_by_report(self.simulate(1.0))
        assert out["present"]["response"] == "present" and out["absent"]["response"] == "absent"
        assert all(math.isfinite(v["meta_d_prime"]) for v in out.values())
        with pytest.raises(ValueError):
            metad.fit_meta_d(np.zeros((2, 2, 4)))
        with pytest.raises(ValueError):
            metad.fit_meta_d(self.simulate(1.0), response="x")
        with pytest.raises(ValueError):
            metad.count_table([True], [True], [9], 4)


class TestReversalSchedule:
    def test_deterministic_and_reversing(self):
        a, b = reversal_task.make_schedule(7), reversal_task.make_schedule(7)
        assert a == b and a != reversal_task.make_schedule(8)
        assert a["reversal_trials"] and a["announced"] is False
        g = np.array(a["good_option"])
        assert all(g[t] != g[t - 1] for t in a["reversal_trials"])

    def test_reward_rate_matches_p_good(self):
        s = reversal_task.make_schedule(3, n_trials=2000, run_min=20, run_max=40)
        good = np.array(s["good_option"])
        rw = np.array(s["reward"])
        rate = rw[np.arange(len(good)), good].mean()
        assert rate == pytest.approx(0.8, abs=0.03)

    def test_validation(self):
        with pytest.raises(ValueError):
            reversal_task.make_schedule(0, p_good=0.5)
        with pytest.raises(ValueError):
            reversal_task.make_schedule(0, run_min=30, run_max=10)


class TestMasking:
    def test_frame_lock(self):
        out = masking.lock_soa_to_frames(50, 16.667)
        assert out["frames"] == 3 and out["locked_soa_ms"] == pytest.approx(50.0, abs=0.01)
        assert masking.lock_soa_to_frames(3, 16.667)["frames"] == 1
        with pytest.raises(ValueError):
            masking.lock_soa_to_frames(0, 16.7)

    def test_achieved_soa_and_dropped(self):
        ok = masking.achieved_soa(100, 150.2, 3, 16.667)
        bad = masking.achieved_soa(100, 167, 3, 16.667)
        assert not ok["dropped"] and bad["dropped"] and bad["frame_error"] == 1

    def test_pas(self):
        assert masking.validate_pas(3) == 3
        with pytest.raises(ValueError):
            masking.validate_pas(5)

    def test_downtier(self):
        t = [{"dropped": True}] * 3 + [{"dropped": False}] * 7
        assert masking.downtier_for_dropped_frames(t, "A")[0] == "B"
        assert masking.downtier_for_dropped_frames(t, "C")[0] == "C"
        assert masking.downtier_for_dropped_frames([{"dropped": False}] * 10, "A") == ("A", [])
        assert masking.downtier_for_dropped_frames([], "A")[0] == "C"

    def test_summary(self):
        rng = np.random.default_rng(0)
        trials = []
        for i in range(240):
            soa = [1, 2, 3, 4][i % 4]
            trials.append({"trial_index": i, "soa_frames": soa, "correct": rng.random() < 0.4 + 0.15 * soa,
                           "rt_ms": 400 + rng.normal(0, 50), "confidence": soa})
        out = masking.summarize(trials)
        acc = out["accuracy_by_soa"]
        assert acc[4]["accuracy"] > acc[1]["accuracy"] and out["rt_ms"]["n"] == 240
        assert math.isfinite(out["split_half_reliability_within_soa"])
        with pytest.raises(ValueError):
            masking.summarize([])


class TestFormsCounterbalance:
    def test_williams_balance(self):
        rows = forms.williams_orders(3)
        assert len(rows) == 6 and all(sorted(r) == [0, 1, 2] for r in rows)
        for pos in range(3):  # every form in every position equally often
            assert sorted(r[pos] for r in rows) == [0, 0, 1, 1, 2, 2]
        pairs = [(r[i], r[i + 1]) for r in rows for i in range(2)]
        assert all(pairs.count(p) == 2 for p in {(a, b) for a in range(3) for b in range(3) if a != b})

    def test_enrolment_cycles_and_is_stable(self):
        a = [tuple(forms.assign_order_at_enrolment(i)) for i in range(12)]
        assert len(set(a[:6])) == 6 and a[:6] == a[6:]
        assert forms.assign_order_at_enrolment(4) == forms.assign_order_at_enrolment(4)
        with pytest.raises(ValueError):
            forms.assign_order_at_enrolment(-1)

    def test_even_n(self):
        assert len(forms.williams_orders(4)) == 4


class TestBridge:
    def test_shipped_bridge_is_complete(self):
        b = bridge.load_bridge()
        assert bridge.bridge_errors(b) == []
        ids = {r["row_id"] for r in b["rows"]}
        assert ids == {f"S3-BAT-05.{i}" for i in range(1, 7)}

    def test_gating(self):
        b = bridge.load_bridge()
        cleared = bridge.tasks_cleared_for_battery(
            b, ["near_threshold_detection", "reversal_learning", "backward_masking", "unknown_task"])
        assert cleared == {"near_threshold_detection": True, "reversal_learning": True,
                           "backward_masking": False, "unknown_task": False}

    def test_incomplete_row_fails(self):
        errs = bridge.bridge_errors({"rows": [{"row_id": "x", "testable_v0_1": True}]})
        assert any("missing" in e for e in errs)
        errs = bridge.bridge_errors({"rows": [{"row_id": "y", "construct": "c", "symbol": "s", "task": "t",
            "observables": [], "estimator": None, "testable_v0_1": True, "reporting_scale": "r", "limitations": "l"}]})
        assert any("testable row" in e for e in errs)
