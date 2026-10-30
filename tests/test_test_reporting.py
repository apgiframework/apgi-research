"""A successful teardown must not erase a test failure from the summary."""

from types import SimpleNamespace

from tests import conftest as reporting
import pytest


@pytest.mark.parametrize(
    "outcomes,expected",
    [
        ([("setup", "passed"), ("call", "failed"), ("teardown", "passed")], "failed"),
        ([("setup", "failed"), ("teardown", "passed")], "failed"),
        ([("call", "passed"), ("teardown", "failed")], "failed"),
        ([("setup", "skipped"), ("teardown", "passed")], "skipped"),
        ([("setup", "passed"), ("call", "passed"), ("teardown", "passed")], "passed"),
    ],
)
def test_summary_preserves_phase_outcomes(monkeypatch, outcomes, expected):
    results = {}
    with monkeypatch.context() as patcher:
        patcher.setattr(reporting, "test_results", results)
        for phase, outcome in outcomes:
            reporting.pytest_runtest_logreport(
                SimpleNamespace(
                    nodeid="synthetic",
                    when=phase,
                    failed=outcome == "failed",
                    skipped=outcome == "skipped",
                    longrepr="example failure",
                )
            )
        assert results["synthetic"]["status"] == expected
