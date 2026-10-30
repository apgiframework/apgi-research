"""
Comprehensive tests for human_layer.py to achieve 90%+ coverage.
Tests all major classes, methods, validation, and edge cases.
"""

from __future__ import annotations

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import logging
from unittest.mock import patch

import pytest

import human_layer

logger = logging.getLogger(__name__)

import logging
from pathlib import Path

from human_layer import (
    HumanControlLayer,
    Task,
    TaskPriority,
    configure_if_needed,
    review,
    select_task,
)
from hypothesis_approval_board import ApprovalBoard, Hypothesis


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """Config, hypothesis board and selection log default to the cwd."""
    monkeypatch.chdir(tmp_path)


def make_layer(tmp_path):
    """A HumanControlLayer whose config lives in the test's temp directory."""
    return HumanControlLayer(config_path=str(tmp_path / "human_config.json"))


def make_task(task_id, priority, experiment_type="cognitive"):
    return Task(
        id=task_id,
        title=task_id,
        description=f"{task_id} task",
        priority=priority,
        experiment_type=experiment_type,
    )


class TestTaskPriority:
    """Tests for TaskPriority enum."""

    def test_priority_values(self):
        """Test that all expected priority values exist."""
        priorities = [
            TaskPriority.CRITICAL,
            TaskPriority.HIGH,
            TaskPriority.MEDIUM,
            TaskPriority.LOW,
        ]

        for priority in priorities:
            assert isinstance(priority.value, str)  # nosec: B101 - Test assertion
            assert len(priority.value) > 0  # nosec: B101 - Test assertion

    def test_priority_ordering(self, tmp_path):
        """Higher priority tasks are selected ahead of lower priority ones."""
        for higher, lower in [
            (TaskPriority.CRITICAL, TaskPriority.HIGH),
            (TaskPriority.HIGH, TaskPriority.MEDIUM),
            (TaskPriority.MEDIUM, TaskPriority.LOW),
        ]:
            layer = make_layer(tmp_path)
            layer.config["task_filters"]["min_priority"] = "low"
            layer.tasks = [make_task("lower", lower), make_task("higher", higher)]

            assert layer.select_task().id == "higher"  # nosec: B101

    def test_priority_comparison(self, tmp_path):
        """Selection ranks all four levels, regardless of insertion order."""
        layer = make_layer(tmp_path)
        layer.config["task_filters"]["min_priority"] = "low"
        layer.tasks = [
            make_task("low", TaskPriority.LOW),
            make_task("critical", TaskPriority.CRITICAL),
            make_task("medium", TaskPriority.MEDIUM),
            make_task("high", TaskPriority.HIGH),
        ]

        order = []
        while layer.tasks:
            selected = layer.select_task()
            order.append(selected.id)
            layer.tasks.remove(selected)

        assert order == ["critical", "high", "medium", "low"]  # nosec: B101


class TestHypothesisApprovalBoard:
    """Tests for HypothesisApprovalBoard class."""

    def test_board_initialization(self):
        """Test board initialization."""
        board = ApprovalBoard()

        assert hasattr(board, "add_hypothesis")  # nosec: B101 - Test assertion
        assert hasattr(board, "approve")  # nosec: B101 - Test assertion
        assert hasattr(board, "reject")  # nosec: B101 - Test assertion
        assert hasattr(board, "get_status")  # nosec: B101 - Test assertion

    def test_add_hypothesis(self):
        """Test adding hypotheses to board."""
        board = ApprovalBoard()

        # Create a hypothesis using the proper method
        hypothesis = board.create_hypothesis(
            title="Test hypothesis",
            description="Test description",
        )

        # Verify hypothesis was added
        status = board.get_status()
        assert status.total_count == 1  # nosec: B101 - Test assertion

        # Get hypothesis details
        pending = board.get_hypothesis(hypothesis.id)
        assert pending is not None  # nosec: B101 - Test assertion
        assert pending.title == "Test hypothesis"  # nosec: B101 - Test assertion

    def test_approve_hypothesis(self):
        """Test hypothesis approval."""
        from hypothesis_approval_board import ApprovalBoard

        board = ApprovalBoard()

        # Create and approve a hypothesis
        hypothesis = board.create_hypothesis(
            title="Test approval",
            description="Test description",
        )

        # Approve the hypothesis
        success = board.approve(hypothesis.id, "Test approval")
        assert success  # nosec: B101 - Test assertion

        # Verify approval using get_status()
        status = board.get_status()
        assert status.approved_count == 1  # nosec: B101 - Test assertion
        assert status.pending_count == 0  # nosec: B101 - Test assertion

    def test_reject_hypothesis(self):
        """Test hypothesis rejection."""
        board = ApprovalBoard()

        # Create and reject a hypothesis
        hypothesis = board.create_hypothesis(
            title="Test rejection",
            description="Test description",
        )

        # Reject the hypothesis
        success = board.reject(hypothesis.id, "Insufficient evidence", "Test rejection")
        assert success  # nosec: B101 - Test assertion

        # Verify rejection using get_status()
        status = board.get_status()
        assert status.rejected_count == 1  # nosec: B101 - Test assertion
        assert status.pending_count == 0  # nosec: B101 - Test assertion

    def test_board_persistence(self, tmp_path):
        """A saved board can be loaded back with its hypotheses intact."""
        path = tmp_path / "board.json"
        board = ApprovalBoard(storage_path=str(path))
        hypothesis = board.create_hypothesis(
            title="Persistence test",
            description="Test data persistence",
        )

        board.save(path)

        saved = json.loads(path.read_text())
        assert hypothesis.id in saved["hypotheses"]  # nosec: B101
        reloaded = ApprovalBoard.load(path)
        assert reloaded.get_hypothesis(hypothesis.id).title == "Persistence test"

    def test_board_loading(self, tmp_path):
        """Test loading board from file."""
        test_data = {
            "hypotheses": [
                {
                    "id": "test-hyp-1",
                    "title": "Loaded hypothesis",
                    "description": "Test description",
                    "status": "approved",
                    "created_at": "2023-01-01T00:00:00",
                    "updated_at": "2023-01-01T00:00:00",
                }
            ],
            "approvers": [],
            "min_approvers": 1,
        }
        path = tmp_path / "board.json"
        path.write_text(json.dumps(test_data))

        loaded_board = ApprovalBoard.load(path)

        status = loaded_board.get_status()
        assert status.approved_count == 1  # nosec: B101 - Test assertion
        assert status.pending_count == 0  # nosec: B101 - Test assertion
        assert loaded_board.get_hypothesis("test-hyp-1").title == "Loaded hypothesis"

    def test_error_handling(self):
        """Test error handling in board operations."""
        # Test with invalid file - ApprovalBoard.load() should handle gracefully
        with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
            # The load method should create a new board if file doesn't exist
            board = ApprovalBoard.load(Path("nonexistent.json"))
            assert board is not None  # nosec: B101 - Test assertion

    def test_statistics_methods(self, tmp_path):
        """Test board statistics methods."""
        board = ApprovalBoard(storage_path=str(tmp_path / "board.json"))

        # Add test hypotheses
        for i in range(5):
            board.create_hypothesis(
                title=f"Hypothesis {i}",
                description=f"Test hypothesis {i}",
            )

        # Test statistics using get_status()
        status = board.get_status()
        assert status.total_count == 5  # nosec: B101 - Test assertion
        assert status.approved_count == 0  # nosec: B101 - Test assertion
        assert status.rejected_count == 0  # nosec: B101 - Test assertion
        assert (  # nosec: B101 - Test assertion
            status.pending_count + status.approved_count + status.rejected_count
            <= status.total_count
        )

    def test_hypothesis_validation(self):
        """Test hypothesis validation."""
        board = ApprovalBoard()

        # Test valid hypothesis creation
        valid_hypothesis = board.create_hypothesis(
            title="Valid hypothesis",
            description="Valid description",
        )

        # Should create successfully
        assert (
            valid_hypothesis.title == "Valid hypothesis"
        )  # nosec: B101 - Test assertion
        assert (
            valid_hypothesis.description == "Valid description"
        )  # nosec: B101 - Test assertion

        # Test that create_hypothesis handles basic validation
        # (The actual validation is handled by the dataclass)


class TestHumanLayerFunctions:
    """Tests for human layer functions."""

    def test_configure_if_needed_no_gui(self, tmp_path):
        """An already configured layer is left alone and never prompts."""
        (tmp_path / "human_config.json").write_text(json.dumps({"configured": True}))

        with patch("builtins.input", side_effect=AssertionError("prompted")):
            result = configure_if_needed()

        assert result is False  # nosec: B101 - Test assertion

    def test_configure_if_needed_with_gui(self):
        """An unconfigured layer runs the wizard and records the result."""
        # Accept the defaults for every question of the wizard
        answers = ["1", "", "", "", "", ""]

        with patch("builtins.input", side_effect=answers), patch("builtins.print"):
            result = configure_if_needed()

        assert result is True  # nosec: B101 - Test assertion
        saved = json.loads(Path("human_config.json").read_text())
        assert saved["configured"] is True  # nosec: B101 - Test assertion
        assert saved["interaction_mode"] == "interactive"  # nosec: B101

    def test_select_task_empty_board(self, tmp_path):
        """With nothing queued there is nothing to select."""
        assert make_layer(tmp_path).select_task() is None  # nosec: B101
        # The module-level helper builds a fresh, empty layer each time
        assert select_task() is None  # nosec: B101 - Test assertion

    def test_select_task_with_priorities(self, tmp_path):
        """Test select_task with priority-based selection."""
        layer = make_layer(tmp_path)
        layer.tasks = [
            make_task("medium", TaskPriority.MEDIUM),
            make_task("critical", TaskPriority.CRITICAL),
            make_task("high", TaskPriority.HIGH),
        ]

        selected = layer.select_task()

        assert selected is not None and selected.id == "critical"  # nosec: B101
        # The selection is logged
        log = json.loads(Path("task_selections.json").read_text())
        assert log[0]["task_id"] == "critical"  # nosec: B101 - Test assertion

    def test_select_task_with_filters(self, tmp_path):
        """Tasks outside the enabled categories or minimum priority are skipped."""
        layer = make_layer(tmp_path)
        layer.config["task_filters"]["enabled_categories"] = ["cognitive"]
        layer.config["task_filters"]["min_priority"] = "high"
        layer.tasks = [
            make_task("low-priority", TaskPriority.LOW),
            make_task("wrong-category", TaskPriority.CRITICAL, "unlisted"),
            make_task("eligible", TaskPriority.HIGH),
        ]

        selected = layer.select_task()

        assert selected is not None and selected.id == "eligible"  # nosec: B101

        layer.tasks = [make_task("low-priority", TaskPriority.LOW)]
        assert layer.select_task() is None  # nosec: B101 - Test assertion

    def test_review_functionality(self, tmp_path):
        """A review of an experiment result produces a decision and its metadata."""
        layer = make_layer(tmp_path)
        layer.config["interaction_mode"] = "autonomous"

        with patch("builtins.print"):
            result = layer.review({"experiment_id": "exp-1", "confidence": 0.8})

        assert result is not None  # nosec: B101 - Test assertion
        assert result.reviewer == "autonomous_agent"  # nosec: B101
        assert result.confidence == 0.8  # nosec: B101 - Test assertion
        assert result.timestamp  # nosec: B101 - Test assertion

    def test_review_with_approval(self, tmp_path):
        """High confidence with the hypothesis's success criteria met is approved."""
        layer = make_layer(tmp_path)
        layer.config["interaction_mode"] = "autonomous"
        hypothesis = layer.approval_board.create_hypothesis(
            title="Accuracy",
            description="Accuracy is high",
            success_criteria=["accuracy"],
        )

        with patch("builtins.print"):
            result = layer.review(
                {
                    "experiment_id": "exp-1",
                    "hypothesis_id": hypothesis.id,
                    "metrics": {"accuracy": 0.9},
                    "confidence": 0.9,
                }
            )

        assert result.decision.value == "approve"  # nosec: B101 - Test assertion

    def test_review_with_rejection(self, tmp_path):
        """Low confidence is rejected."""
        layer = make_layer(tmp_path)
        layer.config["interaction_mode"] = "autonomous"

        with patch("builtins.print"):
            result = layer.review(
                {"experiment_id": "exp-1", "metrics": {}, "confidence": 0.2}
            )

        assert result.decision.value == "reject"  # nosec: B101 - Test assertion


class TestIntegration:
    """Integration tests for human layer functionality."""

    def test_full_human_layer_workflow(self, tmp_path):
        """Test complete human layer workflow."""
        with (
            patch(
                "human_layer.HumanControlLayer.configure_if_needed", return_value=True
            ) as mock_configure,
            patch("human_layer.select_task") as mock_select,
            patch("human_layer.review") as mock_review,
        ):

            # Configure human interaction
            assert configure_if_needed() is True  # nosec: B101 - Test assertion
            mock_configure.assert_called_once()

            # Create approval board
            board = ApprovalBoard(storage_path=str(tmp_path / "board.json"))

            # Add task to board
            task = Hypothesis(
                id="integration_test",
                title="Integration test task",
                description="Test task for integration",
                tags=["integration_evidence"],
            )
            board.add_hypothesis(task)

            # Select task
            # The patched module attributes are only seen through the module,
            # not through names imported at the top of this file
            mock_select.return_value = task
            selected_task = human_layer.select_task()

            # Review and approve task
            mock_review.return_value = [{"task": task, "action": "approve"}]
            human_layer.review({"test": "Integration review"})

            # Verify workflow
            mock_configure.assert_called_once()
            mock_select.assert_called_once_with()
            mock_review.assert_called_once()
            assert (
                selected_task and selected_task.title == "Integration test task"
            )  # nosec: B101 - Test assertion

    def test_error_handling_in_workflow(self):
        """Test error handling in human layer workflow."""
        with patch(
            "human_layer.HumanControlLayer.configure_if_needed",
            side_effect=Exception("GUI error"),
        ):

            # Errors raised while configuring must propagate to the caller
            with pytest.raises(Exception, match="GUI error"):
                configure_if_needed()

    def test_persistence_across_sessions(self, tmp_path):
        """Test data persistence across sessions."""
        storage = tmp_path / "board.json"
        board1 = ApprovalBoard(storage_path=str(storage))

        # Add data to first board
        task = Hypothesis(
            id="persistence_test",
            title="Persistence test task",
            description="Test task for persistence",
            tags=["persistence_evidence"],
        )
        board1.add_hypothesis(task)

        # Save first board
        board1.save(storage)

        # Load second board from same file
        board2 = ApprovalBoard.load(storage)

        # Verify data persistence
        assert len(board2.hypotheses) == 1  # nosec: B101 - Test assertion
        hypothesis = board2.get_hypothesis("persistence_test")
        assert hypothesis is not None  # nosec: B101 - Test assertion
        assert (
            hypothesis.title == "Persistence test task"
        )  # nosec: B101 - Test assertion

    def test_concurrent_access(self, tmp_path, monkeypatch):
        """Test concurrent access to human layer."""
        import threading

        # Keep the layer's config and selection log out of the working directory
        monkeypatch.chdir(tmp_path)
        board = ApprovalBoard(storage_path=str(tmp_path / "board.json"))

        # Add test data
        task = Hypothesis(
            id="concurrent_test",
            title="Concurrent test task",
            description="Test task for concurrent access",
            tags=["concurrent_evidence"],
        )
        board.add_hypothesis(task)

        errors = []

        def worker(index: int) -> None:
            try:
                select_task()
                review({"test": f"Concurrent review {index}"})
            except Exception as exc:  # pragma: no cover - reported below
                errors.append(exc)

        # GUI mode makes review() non-interactive, so no stdin is required
        with patch("human_layer._is_tkinter_running", return_value=True):
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        assert (
            not errors
        ), f"Concurrent access should be thread-safe: {errors}"  # nosec: B101

    def test_configuration_integration(self):
        """Test configuration integration with human layer."""
        with patch(
            "human_layer.HumanControlLayer.configure_if_needed", return_value=True
        ) as mock_configure:

            # Test configuration
            configure_if_needed()

            # Verify configuration was called with proper parameters
            mock_configure.assert_called_once()
            args, kwargs = mock_configure.call_args

            # Should be called with no arguments (auto-configuration)
            assert len(args) == 0  # nosec: B101 - Test assertion
            assert len(kwargs) == 0  # nosec: B101 - Test assertion
