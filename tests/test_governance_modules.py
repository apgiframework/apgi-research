"""Consent enforcement, deviation labelling, provenance and acquisition integrity."""

import math

import pytest

from battery import acquisition
from utils.apgi_consent import ConsentError, ConsentLedger, Purpose, requires_consent
from utils.apgi_deviation import detect_deviations, label_output
from utils.apgi_provenance import make_record, verify_record, write_sidecar

ALL = {p.value: True for p in Purpose}


class TestConsent:
    def test_missing_decline_withdraw_version_all_block(self):
        led = ConsentLedger("v2")
        with pytest.raises(ConsentError, match="no consent"):
            led.require("p", Purpose.CORE_RESEARCH, "ingest")
        led.record("p", "v2", {**ALL, "data_sharing": False})
        led.require("p", Purpose.CORE_RESEARCH, "ingest")
        with pytest.raises(ConsentError, match="declined"):
            led.require("p", Purpose.DATA_SHARING, "export")
        led.record("q", "v1", ALL)
        with pytest.raises(ConsentError, match="version"):
            led.require("q", Purpose.CORE_RESEARCH, "ingest")
        led.withdraw("p")
        with pytest.raises(ConsentError, match="withdrawn"):
            led.require("p", Purpose.CORE_RESEARCH, "ingest")
        assert len(led.denials) == 4

    def test_decision_for_every_purpose_required(self):
        with pytest.raises(ValueError):
            ConsentLedger("v1").record("p", "v1", {"core_research": True})
        with pytest.raises(ValueError):
            ConsentLedger("v1").record("p", "v1", {**ALL, "recontact": "yes"})

    def test_decorator_blocks_before_side_effects(self):
        led = ConsentLedger("v1")
        led.record("p", "v1", {**ALL, "physiological_signals": False})
        stored = []

        @requires_consent(led, Purpose.PHYSIOLOGICAL_SIGNALS)
        def ingest_signal(participant_id, chunk):
            stored.append(chunk)

        with pytest.raises(ConsentError):
            ingest_signal("p", [1, 2])
        assert stored == []  # zero persistence for the declined purpose

    def test_fingerprint_changes_with_state(self):
        led = ConsentLedger("v1")
        led.record("p", "v1", ALL)
        a = led.fingerprint("p")
        led.withdraw("p")
        assert led.fingerprint("p") != a


class TestDeviation:
    def test_in_range_is_confirmatory_eligible(self):
        out = label_output({"theta_0": 0.5, "gamma_sig": 5.5, "beta_SM": 0.5})
        assert out["status"] == "confirmatory_eligible" and out["protocol_lock"]

    def test_out_of_range_and_unregistered_are_exploratory(self):
        devs = detect_deviations({"theta_0": 0.5, "gamma_sig": 5.5, "beta_SM": 1.5, "kappa": 100},
                                 ["a", "b"], ["a"])
        kinds = {d["kind"] for d in devs}
        assert {"parameter_outside_registered_range", "unregistered_parameter", "unregistered_analysis"} <= kinds
        assert label_output({"theta_0": 0.5, "gamma_sig": 5.5, "beta_SM": 1.5})["status"] == "exploratory"

    def test_missing_registered_parameter(self):
        assert any(d["kind"] == "missing_registered_parameter" for d in detect_deviations({"theta_0": 0.5}))


class TestProvenance:
    def test_record_verify_and_tamper(self, tmp_path):
        f = tmp_path / "d.csv"
        f.write_text("a,b\n1,2\n")
        rec = make_record(f, "ds1", "synthetic", "1.0.0", "CC0-1.0",
                          [{"step": "generate", "software_version": "1.1.0"}])
        assert verify_record(rec, f) == [] and write_sidecar(rec, f).exists()
        f.write_text("a,b\n1,3\n")
        assert "checksum mismatch" in verify_record(rec, f)
        bad = dict(rec, license="")
        assert any("license" in e for e in verify_record(bad, f))
        assert any("software_version" in e for e in verify_record(dict(rec, transformations=[{"step": "x"}]), f))
        with pytest.raises(ValueError):
            make_record(f, "", "s", "v", "l")


class TestAcquisition:
    def _rec(self, n=1000, hz=250.0):
        ts = [i * 1000.0 / hz for i in range(n)]
        return [math.sin(i / 10) for i in range(n)], ts

    def test_clean_recording_passes(self):
        s, t = self._rec()
        h = acquisition.recording_header(s, t, 250.0)
        assert acquisition.check_recording(s, t, h)["ok"]

    def test_injected_faults_detected(self):
        s, t = self._rec()
        h = acquisition.recording_header(s, t, 250.0)
        corrupt = list(s)
        corrupt[10] = 99.0
        assert "checksum_or_length_mismatch" in acquisition.check_recording(corrupt, t, h)["problems"]
        s2 = s[:100] + s[110:]
        t2 = t[:100] + t[110:]
        res = acquisition.check_recording(s2, t2, acquisition.recording_header(s2, t2, 250.0))
        assert "dropped_samples" in res["problems"] and res["dropped_samples"] == 10
        t3 = [x * 1.001 for x in t]
        res = acquisition.check_recording(s, t3, acquisition.recording_header(s, t3, 250.0))
        assert "clock_drift" in res["problems"]
        s4 = list(s)
        s4[5] = float("nan")
        assert "nonfinite_samples" in acquisition.check_recording(s4, t, acquisition.recording_header(s4, t, 250.0))["problems"]
        with pytest.raises(ValueError):
            acquisition.recording_header([1.0], [0.0], 250.0)
