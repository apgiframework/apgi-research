"""A successful teardown must not erase a test failure from the summary."""

from types import SimpleNamespace

import pytest

from tests import conftest as reporting


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


def test_teardown_without_recorded_start_is_tolerated(monkeypatch):
    """A test skipped at setup reaches teardown with no start_time recorded."""
    results = {}
    with monkeypatch.context() as patcher:
        patcher.setattr(reporting, "test_results", results)
        reporting.pytest_runtest_logreport(
            SimpleNamespace(
                nodeid="synthetic",
                when="setup",
                failed=False,
                skipped=True,
                longrepr=None,
            )
        )
        reporting.pytest_runtest_teardown(SimpleNamespace(nodeid="synthetic"), None)
        assert results["synthetic"]["status"] == "skipped"
        assert "duration" not in results["synthetic"]
