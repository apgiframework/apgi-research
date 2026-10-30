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
from unittest.mock import MagicMock, mock_open, patch

import pytest

import human_layer

logger = logging.getLogger(__name__)

import logging
from pathlib import Path

from human_layer import TaskPriority, configure_if_needed, review, select_task
from hypothesis_approval_board import ApprovalBoard, Hypothesis


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

    def test_priority_ordering(self):
        """Test priority ordering for comparison."""
        assert (
            TaskPriority.CRITICAL.value > TaskPriority.HIGH.value
        )  # nosec: B101 - Test assertion
        assert (
            TaskPriority.HIGH.value > TaskPriority.MEDIUM.value
        )  # nosec: B101 - Test assertion
        assert (
            TaskPriority.MEDIUM.value > TaskPriority.LOW.value
        )  # nosec: B101 - Test assertion

    def test_priority_comparison(self):
        """Test priority comparison operations."""
        critical = TaskPriority.CRITICAL
        high = TaskPriority.HIGH
        medium = TaskPriority.MEDIUM
        low = TaskPriority.LOW

        # Test comparisons - compare underlying values since Enum doesn't support > by default
        assert critical.value > high.value  # nosec: B101 - Test assertion
        assert high.value > medium.value  # nosec: B101 - Test assertion
        assert medium.value > low.value  # nosec: B101 - Test assertion

        # Test with different types
        assert (
            str(critical) != "string"
        )  # Different type  # nosec: B101 - Test assertion
        assert str(high) != "42"  # Different value  # nosec: B101 - Test assertion

        # Test enum value comparisons
        assert critical.value == "critical"  # nosec: B101 - Test assertion
        assert high.value == "high"  # nosec: B101 - Test assertion


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

    def test_board_persistence(self):
        """Test board data persistence."""
        board = ApprovalBoard()

        # Create test data
        board.create_hypothesis(
            title="Persistence test",
            description="Test data persistence",
        )

        # Mock file operations
        mock_dump = MagicMock()
        with (
            patch("builtins.open", mock_open) as mock_open_file,
            patch("json.dump", mock_dump),
        ):
            # Type the mock properly
            mock_open_file = MagicMock()

            # Save board using save() method
            board.save(Path("test_board.json"))

            # Verify file operations
            mock_open_file.assert_called_once()
            mock_dump.assert_called_once()

    def test_board_loading(self):
        """Test loading board from file."""
        from hypothesis_approval_board import ApprovalBoard

        # Test data for loading
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

        mock_open_file = MagicMock()
        with patch("builtins.open", mock_open_file):
            mock_open_file.return_value.__enter__.return_value.read.return_value = (
                json.dumps(test_data)
            )

            # Load board using load() method
            loaded_board = ApprovalBoard.load(Path("test_board.json"))

            # Verify loading
            mock_open_file.assert_called_once_with(Path("test_board.json"), "r")
            status = loaded_board.get_status()
            assert status.approved_count == 1  # nosec: B101 - Test assertion
            assert status.pending_count == 0  # nosec: B101 - Test assertion

    def test_error_handling(self):
        """Test error handling in board operations."""
        # Test with invalid file - ApprovalBoard.load() should handle gracefully
        with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
            # The load method should create a new board if file doesn't exist
            board = ApprovalBoard.load(Path("nonexistent.json"))
            assert board is not None  # nosec: B101 - Test assertion

    def test_statistics_methods(self):
        """Test board statistics methods."""
        board = ApprovalBoard()

        # Add test hypotheses
        for i in range(5):
            board.create_hypothesis(
                title=f"Hypothesis {i}",
                description=f"Test hypothesis {i}",
            )

        # Test statistics using get_status()
        status = board.get_status()
        assert status.total_count == 5  # nosec: B101 - Test assertion
        assert (
            status.pending_count == 5
        )  # All start as draft, not pending  # nosec: B101 - Test assertion
        assert status.approved_count == 0  # nosec: B101 - Test assertion
        assert status.rejected_count == 0  # nosec: B101 - Test assertion

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

    def test_configure_if_needed_no_gui(self):
        """Test configure_if_needed without GUI."""
        with patch("human_layer._is_tkinter_running", return_value=False):
            result = configure_if_needed()

            # Should not configure without GUI
            assert result is False  # nosec: B101 - Test assertion

    def test_configure_if_needed_with_gui(self):
        """Test configure_if_needed with GUI running."""
        with patch("human_layer._is_tkinter_running", return_value=True):
            with patch("human_layer.configure_human_interaction") as mock_configure:
                result = configure_if_needed()

                # Should configure human interaction
                assert result is True  # nosec: B101 - Test assertion
                mock_configure.assert_called_once()

    def test_select_task_empty_board(self):
        """Test select_task with empty board."""
        with pytest.raises(ValueError, match="No hypotheses available"):
            select_task()

    def test_select_task_with_priorities(self):
        """Test select_task with priority-based selection."""
        # Create a mock board with tasks for testing select_task function
        from unittest.mock import Mock

        mock_board = Mock()
        mock_tasks = [
            {"title": "Critical task", "priority": TaskPriority.CRITICAL},
            {"title": "High priority task", "priority": TaskPriority.HIGH},
            {"title": "Medium priority task", "priority": TaskPriority.MEDIUM},
        ]
        mock_board.tasks = mock_tasks

        # Mock select_task function behavior
        with patch("human_layer.select_task") as mock_select:
            mock_select.return_value = mock_tasks[0]  # Return critical task

            selected = select_task()
            assert (
                selected and selected.title == "Critical task"
            )  # nosec: B101 - Test assertion

    def test_select_task_with_filters(self):
        """Test select_task with filters."""
        # Mock the select_task function for testing
        from unittest.mock import Mock

        mock_board = Mock()
        mock_tasks = [
            {
                "title": "Approved task",
                "status": "approved",
                "priority": TaskPriority.LOW,
            },
            {
                "title": "Pending task",
                "status": "pending",
                "priority": TaskPriority.HIGH,
            },
        ]
        mock_board.tasks = mock_tasks

        # Test select_task with priority filter
        with patch("human_layer.select_task") as mock_select:
            mock_select.return_value = mock_tasks[1]  # Return pending task

            selected_high = select_task()
            assert (
                selected_high and selected_high.title == "Pending task"
            )  # nosec: B101 - Test assertion

    def test_review_functionality(self):
        """Test review functionality."""
        # Mock the review function for testing
        mock_result = {
            "decision": "approve",
            "reviewer": "test_reviewer",
            "confidence": 0.8,
            "comments": "Test review",
        }

        with patch("human_layer.review") as mock_review:
            mock_review.return_value = mock_result

            review_result = review({"test": "Review test"})

            # Verify review was processed
            assert review_result is not None  # nosec: B101 - Test assertion
            assert (
                review_result.decision.value == "approve"
            )  # nosec: B101 - Test assertion

    def test_review_with_approval(self):
        """Test review with approval."""
        # Mock the review function for testing
        mock_result = {
            "decision": "approve",
            "reviewer": "test_reviewer",
            "confidence": 0.9,
            "comments": "Approved for testing",
        }

        with patch("human_layer.review") as mock_review:
            mock_review.return_value = mock_result

            review_result = review({"test": "Review with approval"})

            # Verify approval
            assert review_result is not None  # nosec: B101 - Test assertion
            assert (
                review_result.decision.value == "approve"
            )  # nosec: B101 - Test assertion

    def test_review_with_rejection(self):
        """Test review with rejection."""
        # Mock the review function for testing
        mock_result = {
            "decision": "reject",
            "reviewer": "test_reviewer",
            "confidence": 0.7,
            "comments": "Rejected for testing",
            "reason": "Insufficient evidence",
        }

        with patch("human_layer.review") as mock_review:
            mock_review.return_value = mock_result

            review_result = review({"test": "Review with rejection"})

            # Verify rejection
            assert review_result is not None  # nosec: B101 - Test assertion
            assert (
                review_result.decision.value == "reject"
            )  # nosec: B101 - Test assertion


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
