"""Each required CI gate passes on the release and fails on a targeted mutation."""

import copy
import importlib.util
import pathlib
import shutil

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("ci_gates", ROOT / "scripts/ci_gates.py")
gates = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gates)


@pytest.fixture
def docs():
    return gates.load_documents()


def mutated(docs, filename_prefix, edit):
    out = copy.deepcopy(docs)
    name = next(n for n in out if n.startswith(filename_prefix))
    edit(out[name])
    return out


@pytest.mark.parametrize("gate", sorted(gates.GATES))
def test_release_passes_every_gate(docs, gate):
    assert gates.GATES[gate](docs) == []


def test_main_exit_codes(capsys):
    assert gates.main([]) == 0
    assert gates.main(["--gate", "timing-contracts"]) == 0


class TestTrialSequencing:
    def test_block_arithmetic(self, docs):
        bad = mutated(
            docs, "protocol_2_", lambda d: d["blocks"].update(trials_per_block=99)
        )
        assert any("!= total" in e for e in gates.trial_sequencing_errors(bad))

    def test_signal_catch_split(self, docs):
        bad = mutated(
            docs,
            "protocol_1_",
            lambda d: d["blocks"].update(main_catch_trials_per_session=100),
        )
        assert any("signal+catch" in e for e in gates.trial_sequencing_errors(bad))

    def test_protocol6_trial_kinds(self, docs):
        bad = mutated(docs, "protocol_6_", lambda d: d["blocks"].update(catch_trials=1))
        assert any("trial kinds" in e for e in gates.trial_sequencing_errors(bad))

    def test_protocol4_contrast_tiling(self, docs):
        bad = mutated(
            docs, "protocol_4_", lambda d: d["blocks"].update(catch_fraction=0.3)
        )
        assert any("divisible" in e for e in gates.trial_sequencing_errors(bad))


class TestProtocolLocking:
    def test_hash_mismatch_detected(self, tmp_path):
        folder = tmp_path / "protocols"
        shutil.copytree(gates.PROTOCOLS, folder)
        target = next(folder.glob("protocol_3_*.json"))
        target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")
        errors = gates.protocol_locking_errors(gates.load_documents(folder), folder)
        assert any("SHA256 mismatch" in e for e in errors)

    def test_collecting_without_registration_or_ethics(self, docs):
        bad = mutated(
            docs, "protocol_1_", lambda d: d.update(study_status="collecting")
        )
        errors = gates.protocol_locking_errors(bad)
        assert any("verified registration" in e for e in errors)
        assert any("approved ethics" in e for e in errors)

    def test_enrollment_needs_ethics(self, docs):
        bad = mutated(
            docs,
            "protocol_1_",
            lambda d: d["ethics"].update(enrollment_authorized=True),
        )
        assert any(
            "enrollment authorised" in e for e in gates.protocol_locking_errors(bad)
        )

    def test_bad_provenance_hash(self, docs):
        bad = mutated(
            docs, "protocol_1_", lambda d: d["provenance"].update(input_sha256="xyz")
        )
        assert any("input_sha256" in e for e in gates.protocol_locking_errors(bad))


class TestTimingContracts:
    def test_window_must_increase(self, docs):
        bad = mutated(
            docs,
            "protocol_1_",
            lambda d: d["eeg_analysis_pipeline"].update(hep_window_ms=[400, 250]),
        )
        assert any("not increasing" in e for e in gates.timing_contract_errors(bad))

    def test_fractional_sample_boundary(self, docs):
        def edit(d):
            d["eeg_analysis_pipeline"].update(
                sample_rate_hz=500, hep_window_ms=[251, 400]
            )

        errors = gates.timing_contract_errors(mutated(docs, "protocol_1_", edit))
        assert any("whole sample" in e for e in errors)

    def test_window_nesting(self, docs):
        bad = mutated(
            docs,
            "protocol_1_",
            lambda d: d["eeg_analysis_pipeline"].update(hep_window_ms=[250, 900]),
        )
        assert any(
            "outside hep_epoch_ms" in e for e in gates.timing_contract_errors(bad)
        )

    def test_negative_duration_but_signed_soa_allowed(self, docs):
        bad = mutated(
            docs,
            "protocol_1_",
            lambda d: d["eeg_analysis_pipeline"].update(
                respiratory_extrema_exclusion_ms=-5
            ),
        )
        assert any("negative duration" in e for e in gates.timing_contract_errors(bad))
        assert (
            gates.timing_contract_errors(docs) == []
        )  # tms_soa_ms == -100 is legitimate


class TestParallelFormBalance:
    @pytest.mark.parametrize("n", [2, 4, 6, 8])
    def test_williams_square_is_balanced(self, n):
        sq = gates.williams_square(n)
        for col in range(n):
            assert sorted(r[col] for r in sq) == list(range(n))
        pairs = [(r[i], r[i + 1]) for r in sq for i in range(n - 1)]
        assert len(set(pairs)) == len(pairs) == n * (n - 1)

    def test_unbalanced_enrolment_target(self, docs):
        bad = mutated(
            docs, "protocol_4_", lambda d: d["participants"].update(n_target=63)
        )
        assert any(
            "equal order cells" in e for e in gates.parallel_form_balance_errors(bad)
        )

    def test_odd_condition_count(self, docs):
        bad = mutated(
            docs,
            "protocol_4_",
            lambda d: d["trial_structure"].update(sessions=["a", "b", "c"]),
        )
        assert any("odd number" in e for e in gates.parallel_form_balance_errors(bad))
