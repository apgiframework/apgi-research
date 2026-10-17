"""Protocol specification 1.1.0: loader fields, verdicts and release metadata."""

import re
from pathlib import Path

import pytest

from APGI_System import APGIParameters
from utils.apgi_protocol_spec import (
    SPECIFICATION_VERSION,
    Verdict,
    check_parameter_defaults,
    classify_prediction,
    load_all_protocols,
    load_protocol,
    registered_ranges,
)
from utils.apgi_version import get_version

ROOT = Path(__file__).resolve().parent.parent
FRAMEWORK_PROTOCOLS = ROOT.parent / "apgi-framework" / "protocols"


class TestSpecificationRelease:
    def test_every_protocol_is_release_1_1_0(self):
        specs = load_all_protocols()
        assert sorted(specs) == [f"APGI-P0{n}" for n in range(8)]
        for spec in specs.values():
            assert spec.version == SPECIFICATION_VERSION
            assert spec.is_current_specification
            assert spec.study_status == "pre_collection"

    def test_kappa_is_uncalibrated(self):
        for spec in load_all_protocols().values():
            kappa = spec.apgi_parameters["kappa"]
            assert kappa["value"] is None
            assert kappa["status"] == "uncalibrated"

    def test_registered_ranges(self):
        assert registered_ranges() == {
            "theta_0": (0.25, 0.85),
            "gamma_sig": (2, 7.5),
            "beta_SM": (0.3, 0.7),
        }

    @pytest.mark.parametrize(
        "protocol, expected",
        [
            (1, (62, 59, 48)),
            ("P3", (40, 36, 36)),
            ("APGI-P04", (64, 55, 48)),
            (5, (52, 44, 36)),
        ],
    )
    def test_cohort_sizes(self, protocol, expected):
        people = load_protocol(protocol).participants
        assert (
            people["n_target"],
            people["n_analyzable_target"],
            people["n_minimum"],
        ) == expected

    def test_sub_prediction_fields(self):
        spec = load_protocol(1)
        pred = spec.get_prediction("Pred 1.A")
        assert pred is not None and pred.is_confirmatory
        assert "not evidence of falsification" in pred.indeterminate_rule
        assert pred.confirming_thresholds["alpha"] == pytest.approx(0.0125)
        assert not spec.get_prediction("Pred 1.E").is_confirmatory
        assert spec.get_prediction("Pred 9.Z") is None

    def test_unknown_protocol(self):
        with pytest.raises(KeyError):
            load_protocol(9)

    def test_missing_folder(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_all_protocols(tmp_path / "absent")

    @pytest.mark.skipif(
        not FRAMEWORK_PROTOCOLS.is_dir(), reason="apgi-framework not checked out"
    )
    def test_byte_identical_to_apgi_framework(self):
        ours = sorted(
            p.relative_to(ROOT / "protocols") for p in (ROOT / "protocols").rglob("*")
        )
        theirs = sorted(
            p.relative_to(FRAMEWORK_PROTOCOLS) for p in FRAMEWORK_PROTOCOLS.rglob("*")
        )
        assert ours == theirs
        for rel in ours:
            if (ROOT / "protocols" / rel).is_file():
                assert (ROOT / "protocols" / rel).read_bytes() == (
                    FRAMEWORK_PROTOCOLS / rel
                ).read_bytes()


class TestVerdicts:
    @pytest.mark.parametrize(
        "confirmed, falsified, expected",
        [
            (True, False, Verdict.CONFIRMED),
            (True, True, Verdict.CONFIRMED),
            (False, True, Verdict.FALSIFIED),
            (False, False, Verdict.INDETERMINATE),
        ],
    )
    def test_failure_to_confirm_is_not_falsification(
        self, confirmed, falsified, expected
    ):
        assert classify_prediction(confirmed, falsified) is expected


class TestParameterDefaults:
    def test_model_defaults_against_registry(self):
        checks = {c.quantity: c for c in check_parameter_defaults()}
        assert checks["theta_0"].inside
        assert checks["gamma_sig"].inside and checks["gamma_sig"].model_field == "alpha"
        # The legacy somatic gain default (1.5) is outside the registered beta_SM range.
        assert not checks["beta_SM"].inside
        assert checks["beta_SM"].registered == (0.3, 0.7)

    def test_custom_parameters(self):
        checks = check_parameter_defaults(APGIParameters(beta=0.6))
        assert all(c.inside for c in checks)


class TestReleaseMetadata:
    def test_version_is_1_1_0(self):
        assert get_version() == "1.1.0"

    def test_versions_agree(self):
        version = get_version()
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert re.search(r'^version = "(.+)"$', pyproject, re.M).group(1) == version
        cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        assert re.search(r'^version: "(.+)"$', cff, re.M).group(1) == version
        assert f"## [{version}]" in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        assert f"version={{{version}}}" in (ROOT / "README.md").read_text(
            encoding="utf-8"
        )

    def test_project_name_is_unified(self):
        legacy = re.compile(
            r"Autonomous Program Generation|Autonomous Psychological General|"
            r"Autonomous Personal Growth|Active Perception and Generative|"
            r"Active Predictive Global Ignition"
        )
        for path in [
            *ROOT.glob("*.py"),
            *ROOT.glob("*.md"),
            *(ROOT / "docs").glob("*.md"),
            *(ROOT / "experiments").glob("*.py"),
        ]:
            if path.name != "CHANGELOG.md":  # records the old names
                assert not legacy.search(path.read_text(encoding="utf-8")), path
