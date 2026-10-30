"""Item bank linter, renderer and frozen reliability targets."""

import copy
import importlib.util
import json
import pathlib
import shutil

import pytest

from item_bank import AGREEMENT_5, lint_bank, load_bank, render_order
from item_bank.lint import lint_item

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("ci_gates", ROOT / "scripts/ci_gates.py")
gates = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gates)


def item(
    item_id="I1", block="A", text="I notice small changes in my surroundings.", **kw
):
    base = {
        "item_id": item_id,
        "block": block,
        "subscale": "awareness",
        "text": text,
        "response_type": "agreement_5",
        "options": list(AGREEMENT_5),
    }
    base.update(kw)
    return base


def bank(*items, order=("A", "B")):
    return {"schema_version": "1.0.0", "block_order": list(order), "items": list(items)}


def test_shipped_bank_is_clean():
    assert lint_bank(load_bank()) == []


def test_valid_item_passes():
    assert lint_item(item()) == []


class TestFivePointAgreement:
    def test_slider_rejected(self):
        errs = lint_item(item(response_type="slider"))
        assert any("response_type" in e for e in errs)

    def test_changed_options_rejected(self):
        errs = lint_item(item(options=["No", "Yes"]))
        assert any("5-point agreement" in e for e in errs)

    def test_slider_field_rejected_as_unregistered(self):
        assert any("unregistered" in e for e in lint_item(item(slider_min=0)))


class TestParameterLabels:
    @pytest.mark.parametrize("label", ["Low ε", "High ε", "high θ", "Low Π"])
    def test_label_in_options_rejected(self, label):
        opts = list(AGREEMENT_5)
        opts[0] = f"Strongly disagree ({label})"
        errs = lint_item(item(options=opts))
        assert any("parameter label" in e for e in errs)

    def test_label_in_text_rejected(self):
        assert any(
            "parameter label" in e
            for e in lint_item(item(text="Rate your High ε today"))
        )

    def test_ordinary_text_not_flagged(self):
        assert lint_item(item(text="I feel low on energy by evening.")) == []


class TestExclusions:
    @pytest.mark.parametrize(
        "name", ["STAI", "BDI-II", "PHQ-9", "GAD-7", "Beck Depression Inventory"]
    )
    def test_diagnostic_instrument_rejected(self, name):
        errs = lint_item(item(source=f"adapted from {name}"))
        assert any("diagnostic instrument" in e for e in errs)

    @pytest.mark.parametrize(
        "text,topic",
        [
            ("I have thought about suicide", "suicidality"),
            ("I engage in self-harm when stressed", "self-harm"),
            ("I restrict food because of anorexia", "disordered eating"),
            ("I drink alcohol to cope", "substance use"),
            ("I take an antidepressant daily", "psychiatric medication"),
        ],
    )
    def test_sensitive_content_rejected(self, text, topic):
        assert any(topic in e for e in lint_item(item(text=text)))

    def test_free_text_field_rejected(self):
        assert any("unregistered" in e for e in lint_item(item(free_text=True)))


class TestBankStructure:
    def test_duplicate_ids_and_block_mismatch(self):
        errs = lint_bank(bank(item("I1"), item("I1", block="B"), order=("A",)))
        assert any("duplicate item_id" in e for e in errs)
        assert any("does not match" in e for e in errs)


class TestRenderer:
    def make(self):
        items = [item(f"A{i}", "A") for i in range(8)] + [
            item(f"B{i}", "B") for i in range(8)
        ]
        return bank(*items)

    def test_block_order_fixed_and_items_stay_in_block(self):
        b = self.make()
        for seed in ("s1", "s2", "s3"):
            order = render_order(b, seed)
            assert [i[0] for i in order] == ["A"] * 8 + ["B"] * 8

    def test_randomised_within_block_and_deterministic(self):
        b = self.make()
        assert render_order(b, "s1") == render_order(b, "s1")
        orders = {tuple(render_order(b, f"s{i}")) for i in range(20)}
        assert len(orders) > 1  # not a fixed order
        assert sorted(render_order(b, "s1")) == sorted(i["item_id"] for i in b["items"])

    def test_block_order_follows_bank(self):
        b = self.make()
        b["block_order"] = ["B", "A"]
        assert render_order(b, "x")[0].startswith("B")


class TestGate:
    def test_release_passes(self):
        assert gates.item_bank_errors() == []

    def test_gate_flags_bad_bank(self):
        bad = bank(item(text="I have thought about suicide"))
        assert gates.item_bank_errors(bank=bad)

    def test_targets_cannot_change_without_lock(self, tmp_path):
        folder = tmp_path / "item_bank"
        shutil.copytree(ROOT / "item_bank", folder)
        path = folder / "reliability_targets.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["internal_consistency"]["minimum_per_subscale"] = 0.5
        path.write_text(json.dumps(data), encoding="utf-8")
        errs = gates.item_bank_errors(folder=folder)
        assert any("lock hash" in e for e in errs)
        assert any("omega" in e for e in errs)

    def test_registered_targets_values(self):
        t = json.loads(
            (ROOT / "item_bank/reliability_targets.json").read_text(encoding="utf-8")
        )
        assert t["internal_consistency"]["minimum_per_subscale"] == 0.80
        assert t["test_retest"]["minimum"] == 0.70 and t["test_retest"][
            "interval_days"
        ] == [14, 21]
        assert t["self_report_vs_behaviour"]["minimum_abs"] == 0.15


def test_deepcopy_independence():
    b = bank(item())
    assert copy.deepcopy(b) == b
