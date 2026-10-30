"""
Comprehensive tests for human_layer.py - human control layer for APGI system.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from human_layer import (
    HumanControlLayer,
    ReviewDecision,
    ReviewResult,
    Task,
    TaskPriority,
    _is_tkinter_running,
    configure_if_needed,
    review,
    select_task,
)


class TestTaskPriority:
    """Tests for TaskPriority enum."""

    def test_priority_values(self):
        """Test TaskPriority enum values."""
        assert TaskPriority.CRITICAL.value == "critical"  # nosec: B101 - Test assertion
        assert TaskPriority.HIGH.value == "high"  # nosec: B101 - Test assertion
        assert TaskPriority.MEDIUM.value == "medium"  # nosec: B101 - Test assertion
        assert TaskPriority.LOW.value == "low"  # nosec: B101 - Test assertion


class TestReviewDecision:
    """Tests for ReviewDecision enum."""

    def test_decision_values(self):
        """Test ReviewDecision enum values."""
        assert ReviewDecision.APPROVE.value == "approve"  # nosec: B101 - Test assertion
        assert ReviewDecision.MODIFY.value == "modify"  # nosec: B101 - Test assertion
        assert ReviewDecision.REJECT.value == "reject"  # nosec: B101 - Test assertion


class TestTask:
    """Tests for Task dataclass."""

    def test_task_creation(self):
        """Test Task creation with all fields."""
        task = Task(
            id="task1",
            title="Test Task",
            description="A test task",
            priority=TaskPriority.HIGH,
            experiment_type="cognitive",
            estimated_duration=30,
            dependencies=["task0"],
            metadata={"key": "value"},
        )
        assert task.id == "task1"  # nosec: B101 - Test assertion
        assert task.title == "Test Task"  # nosec: B101 - Test assertion
        assert task.priority == TaskPriority.HIGH  # nosec: B101 - Test assertion
        assert task.experiment_type == "cognitive"  # nosec: B101 - Test assertion
        assert task.estimated_duration == 30  # nosec: B101 - Test assertion
        assert task.dependencies == ["task0"]  # nosec: B101 - Test assertion
        assert task.metadata == {"key": "value"}  # nosec: B101 - Test assertion

    def test_task_minimal(self):
        """Test Task creation with minimal fields."""
        task = Task(
            id="task1",
            title="Test Task",
            description="A test task",
            priority=TaskPriority.MEDIUM,
            experiment_type="performance",
        )
        assert task.id == "task1"  # nosec: B101 - Test assertion
        assert task.estimated_duration is None  # nosec: B101 - Test assertion
        assert task.dependencies is None  # nosec: B101 - Test assertion
        assert task.metadata is None  # nosec: B101 - Test assertion


class TestReviewResult:
    """Tests for ReviewResult dataclass."""

    def test_review_result_creation(self):
        """Test ReviewResult creation."""
        result = ReviewResult(
            decision=ReviewDecision.APPROVE,
            comments="Looks good",
            reviewer="human",
            timestamp="2024-01-01T00:00:00",
            confidence=0.9,
            modifications=["fix1", "fix2"],
        )
        assert result.decision == ReviewDecision.APPROVE  # nosec: B101 - Test assertion
        assert result.comments == "Looks good"  # nosec: B101 - Test assertion
        assert result.reviewer == "human"  # nosec: B101 - Test assertion
        assert result.confidence == 0.9  # nosec: B101 - Test assertion
        assert result.modifications == ["fix1", "fix2"]  # nosec: B101 - Test assertion

    def test_review_result_minimal(self):
        """Test ReviewResult without modifications."""
        result = ReviewResult(
            decision=ReviewDecision.REJECT,
            comments="Not good",
            reviewer="system",
            timestamp="2024-01-01T00:00:00",
            confidence=0.3,
        )
        assert result.decision == ReviewDecision.REJECT  # nosec: B101 - Test assertion
        assert result.modifications is None  # nosec: B101 - Test assertion


class TestIsTkinterRunning:
    """Tests for _is_tkinter_running function."""

    def test_tkinter_not_available(self):
        """Test when tkinter is not available."""
        import builtins
        import sys

        # Remove tkinter from sys.modules to simulate ImportError
        tkinter_modules = [
            k for k in sys.modules.keys() if k in ("tkinter", "_tkinter")
        ]
        saved_modules = {
            k: sys.modules.pop(k) for k in tkinter_modules if k in sys.modules
        }

        # Create a custom import function that raises ImportError for tkinter
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name in ("tkinter", "_tkinter"):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = mock_import
        try:
            result = _is_tkinter_running()
            assert result is False  # nosec: B101 - Test assertion
        finally:
            builtins.__import__ = original_import
            # Restore modules
            sys.modules.update(saved_modules)

    def test_tkinter_no_default_root(self):
        """Test when tkinter has no default root."""
        import sys

        mock_tk = MagicMock()
        mock_tk._default_root = None
        mock_tkinter = MagicMock()
        mock_tkinter._default_root = None

        with patch.dict(sys.modules, {"tkinter": mock_tk, "_tkinter": mock_tkinter}):
            result = _is_tkinter_running()
            assert result is False  # nosec: B101 - Test assertion

    def test_tkinter_with_root_exists(self):
        """Test when tkinter has existing root."""
        import sys

        mock_root = MagicMock()
        mock_root.winfo_exists.return_value = True
        mock_tk = MagicMock()
        mock_tk._default_root = mock_root

        with patch.dict(sys.modules, {"tkinter": mock_tk}):
            result = _is_tkinter_running()
            assert result is True  # nosec: B101 - Test assertion

    def test_tkinter_with_root_not_exists(self):
        """Test when tkinter root doesn't exist."""
        import sys

        mock_root = MagicMock()
        mock_root.winfo_exists.side_effect = Exception("No display")
        mock_tk = MagicMock()
        mock_tk._default_root = mock_root

        with patch.dict(sys.modules, {"tkinter": mock_tk}):
            result = _is_tkinter_running()
            assert result is False  # nosec: B101 - Test assertion


class TestHumanControlLayer:
    """Tests for HumanControlLayer class."""

    def setup_method(self):
        """Setup test fixtures."""
        # Clean up any existing config files
        config_path = Path("human_config.json")
        if config_path.exists():
            config_path.unlink()
        task_log_path = Path("task_selections.json")
        if task_log_path.exists():
            task_log_path.unlink()
        review_log_path = Path("human_reviews.json")
        if review_log_path.exists():
            review_log_path.unlink()

    def teardown_method(self):
        """Cleanup test fixtures."""
        config_path = Path("human_config.json")
        if config_path.exists():
            config_path.unlink()
        task_log_path = Path("task_selections.json")
        if task_log_path.exists():
            task_log_path.unlink()
        review_log_path = Path("human_reviews.json")
        if review_log_path.exists():
            review_log_path.unlink()

    def test_initialization(self):
        """Test HumanControlLayer initialization."""
        layer = HumanControlLayer()
        assert layer.config_path == Path(
            "human_config.json"
        )  # nosec: B101 - Test assertion
        assert layer.tasks == []  # nosec: B101 - Test assertion
        assert layer.config is not None  # nosec: B101 - Test assertion
        assert "interaction_mode" in layer.config  # nosec: B101 - Test assertion

    def test_initialization_with_custom_path(self):
        """Test HumanControlLayer with custom config path."""
        layer = HumanControlLayer(config_path="custom_config.json")
        assert layer.config_path == Path(
            "custom_config.json"
        )  # nosec: B101 - Test assertion

    def test_load_config_default(self):
        """Test loading default configuration."""
        layer = HumanControlLayer()
        assert (
            layer.config["interaction_mode"] == "interactive"
        )  # nosec: B101 - Test assertion
        assert layer.config["review_threshold"] == 0.7  # nosec: B101 - Test assertion
        assert "task_filters" in layer.config  # nosec: B101 - Test assertion
        assert "notification_settings" in layer.config  # nosec: B101 - Test assertion

    def test_load_config_from_file(self):
        """Test loading configuration from file."""
        # Create a test config file
        config_data = {
            "interaction_mode": "autonomous",
            "review_threshold": 0.8,
            "configured": True,
        }
        config_path = Path("human_config.json")
        config_path.write_text(json.dumps(config_data))

        layer = HumanControlLayer()
        assert (
            layer.config["interaction_mode"] == "autonomous"
        )  # nosec: B101 - Test assertion
        assert layer.config["review_threshold"] == 0.8  # nosec: B101 - Test assertion

    def test_load_config_invalid_json(self):
        """Test loading invalid JSON config file."""
        config_path = Path("human_config.json")
        config_path.write_text("invalid json {{{")

        layer = HumanControlLayer()
        # Should fall back to default config
        assert (
            layer.config["interaction_mode"] == "interactive"
        )  # nosec: B101 - Test assertion

    def test_save_config(self):
        """Test saving configuration."""
        layer = HumanControlLayer()
        layer.config["test_key"] = "test_value"
        layer._save_config()

        config_path = Path("human_config.json")
        assert config_path.exists()  # nosec: B101 - Test assertion
        loaded_data = json.loads(config_path.read_text())
        assert loaded_data["test_key"] == "test_value"  # nosec: B101 - Test assertion

    def test_configure_if_needed_already_configured(self):
        """Test configure_if_needed when already configured."""
        layer = HumanControlLayer()
        layer.config["configured"] = True
        result = layer.configure_if_needed()
        assert result is False  # nosec: B101 - Test assertion

    def test_configure_if_needed_keyboard_interrupt(self):
        """Test configure_if_needed with keyboard interrupt."""
        layer = HumanControlLayer()
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            result = layer.configure_if_needed()
            assert result is False  # nosec: B101 - Test assertion

    def test_configure_if_needed_invalid_choice(self, tmp_path):
        """An invalid mode choice is rejected and re-prompted, not accepted."""
        layer = HumanControlLayer(config_path=str(tmp_path / "human_config.json"))
        # The wizard re-prompts until it gets a valid choice, so a constant
        # invalid answer would loop forever; cancel after one rejection.
        with patch(
            "builtins.input", side_effect=["invalid", KeyboardInterrupt()]
        ) as mock_input:
            result = layer.configure_if_needed()
            assert result is False  # nosec: B101 - Test assertion
            assert mock_input.call_count == 2  # nosec: B101 - Test assertion
            assert not layer.config.get("configured")  # nosec: B101

    def test_configure_if_needed_invalid_threshold(self):
        """Test configure_if_needed with invalid threshold."""
        layer = HumanControlLayer()
        with patch("builtins.input", side_effect=["1", "invalid"]):
            result = layer.configure_if_needed()
            assert result is False  # nosec: B101 - Test assertion

    def test_configure_if_needed_threshold_out_of_range(self):
        """Test configure_if_needed with threshold out of range."""
        layer = HumanControlLayer()
        with patch("builtins.input", side_effect=["1", "2.0"]):
            result = layer.configure_if_needed()
            assert result is False  # nosec: B101 - Test assertion

    def test_add_task(self):
        """Test adding a task to the queue."""
        layer = HumanControlLayer()
        task = Task(
            id="task1",
            title="Test Task",
            description="Test",
            priority=TaskPriority.HIGH,
            experiment_type="cognitive",
        )
        layer.add_task(task)
        assert len(layer.tasks) == 1  # nosec: B101 - Test assertion
        assert layer.tasks[0] == task  # nosec: B101 - Test assertion

    def test_select_task_no_tasks(self):
        """Test select_task with no tasks available."""
        layer = HumanControlLayer()
        result = layer.select_task()
        assert result is None  # nosec: B101 - Test assertion

    def test_select_task_with_tasks(self):
        """Test select_task with available tasks."""
        layer = HumanControlLayer()
        task = Task(
            id="task1",
            title="Test Task",
            description="Test",
            priority=TaskPriority.HIGH,
            experiment_type="cognitive",
        )
        layer.add_task(task)
        result = layer.select_task()
        assert result is not None  # nosec: B101 - Test assertion
        assert result.id == "task1"  # nosec: B101 - Test assertion

    def test_select_task_with_filters(self):
        """Test select_task with category and priority filters."""
        layer = HumanControlLayer()
        layer.config["task_filters"]["enabled_categories"] = ["performance"]
        layer.config["task_filters"]["min_priority"] = "high"

        task1 = Task(
            id="task1",
            title="Cognitive Task",
            description="Test",
            priority=TaskPriority.HIGH,
            experiment_type="cognitive",
        )
        task2 = Task(
            id="task2",
            title="Performance Task",
            description="Test",
            priority=TaskPriority.HIGH,
            experiment_type="performance",
        )
        layer.add_task(task1)
        layer.add_task(task2)

        result = layer.select_task()
        assert result is not None  # nosec: B101 - Test assertion
        assert result.experiment_type == "performance"  # nosec: B101 - Test assertion

    def test_select_task_priority_sorting(self):
        """Test select_task sorts by priority."""
        layer = HumanControlLayer()
        task1 = Task(
            id="task1",
            title="Low Priority",
            description="Test",
            priority=TaskPriority.LOW,
            experiment_type="cognitive",
        )
        task2 = Task(
            id="task2",
            title="High Priority",
            description="Test",
            priority=TaskPriority.HIGH,
            experiment_type="cognitive",
        )
        layer.add_task(task1)
        layer.add_task(task2)

        result = layer.select_task()
        assert result is not None  # nosec: B101 - Test assertion
        assert (
            result.id == "task2"
        )  # High priority selected first  # nosec: B101 - Test assertion

    def test_select_task_queue_limit(self):
        """Test select_task respects max queue size."""
        layer = HumanControlLayer()
        layer.config["task_filters"]["max_queue_size"] = 2

        for i in range(5):
            task = Task(
                id=f"task{i}",
                title=f"Task {i}",
                description="Test",
                priority=TaskPriority.HIGH,
                experiment_type="cognitive",
            )
            layer.add_task(task)

        result = layer.select_task()
        assert result is not None  # nosec: B101 - Test assertion

    def test_select_task_logging(self):
        """Test select_task logs selection to file."""
        layer = HumanControlLayer()
        task = Task(
            id="task1",
            title="Test Task",
            description="Test",
            priority=TaskPriority.HIGH,
            experiment_type="cognitive",
        )
        layer.add_task(task)
        layer.select_task()

        log_path = Path("task_selections.json")
        assert log_path.exists()  # nosec: B101 - Test assertion
        logs = json.loads(log_path.read_text())
        assert len(logs) > 0  # nosec: B101 - Test assertion
        assert logs[0]["task_id"] == "task1"  # nosec: B101 - Test assertion

    def test_review_autonomous_mode(self):
        """Test review in autonomous mode."""
        layer = HumanControlLayer()
        layer.config["interaction_mode"] = "autonomous"
        layer.config["review_threshold"] = 0.7

        result_data = {
            "experiment_id": "exp1",
            "metrics": {"accuracy": 0.9},
            "outcomes": {"success": True},
            "analysis": "Good results",
            "confidence": 0.8,
        }

        with patch("builtins.print"):
            review_result = layer.review(result_data)

        assert (
            review_result.decision == ReviewDecision.APPROVE
        )  # nosec: B101 - Test assertion
        assert (
            review_result.reviewer == "autonomous_agent"
        )  # nosec: B101 - Test assertion

    def test_review_gui_mode(self):
        """Test review in GUI mode (auto-approve)."""
        layer = HumanControlLayer()
        layer.config["review_threshold"] = 0.7

        result_data = {
            "experiment_id": "exp1",
            "metrics": {"accuracy": 0.9},
            "outcomes": {"success": True},
            "analysis": "Good results",
            "confidence": 0.8,
        }

        with patch("human_layer._is_tkinter_running", return_value=True):
            with patch("builtins.print"):
                review_result = layer.review(result_data)

        assert (
            review_result.decision == ReviewDecision.APPROVE
        )  # nosec: B101 - Test assertion
        assert (
            review_result.reviewer == "gui_autonomous_agent"
        )  # nosec: B101 - Test assertion

    def test_review_low_confidence(self):
        """Test review with low confidence (should reject)."""
        layer = HumanControlLayer()
        layer.config["interaction_mode"] = "autonomous"
        layer.config["review_threshold"] = 0.7

        result_data = {
            "experiment_id": "exp1",
            "metrics": {"accuracy": 0.5},
            "outcomes": {"success": False},
            "analysis": "Poor results",
            "confidence": 0.3,
        }

        with patch("builtins.print"):
            review_result = layer.review(result_data)

        assert (
            review_result.decision == ReviewDecision.REJECT
        )  # nosec: B101 - Test assertion

    def test_review_with_hypothesis(self):
        """Test review with associated hypothesis."""
        layer = HumanControlLayer()
        layer.config["interaction_mode"] = "autonomous"

        result_data = {
            "experiment_id": "exp1",
            "metrics": {"accuracy": 0.9},
            "outcomes": {"success": True},
            "analysis": "Good results",
            "confidence": 0.8,
            "hypothesis_id": "hyp1",
        }

        # Mock the hypothesis
        mock_hypothesis = MagicMock()
        mock_hypothesis.title = "Test Hypothesis"
        mock_hypothesis.predicted_outcome = "High accuracy"
        mock_hypothesis.success_criteria = ["accuracy"]

        with (
            patch.object(
                layer.approval_board, "get_hypothesis", return_value=mock_hypothesis
            ),
            patch("builtins.print"),
        ):
            review_result = layer.review(result_data)

        assert (
            review_result.decision == ReviewDecision.APPROVE
        )  # nosec: B101 - Test assertion

    def test_review_keyboard_interrupt(self):
        """Test review with keyboard interrupt."""
        layer = HumanControlLayer()
        layer.config["interaction_mode"] = "interactive"

        result_data = {
            "experiment_id": "exp1",
            "metrics": {"accuracy": 0.9},
            "outcomes": {"success": True},
            "analysis": "Good results",
            "confidence": 0.8,
        }

        with patch("builtins.input", side_effect=KeyboardInterrupt):
            with patch("builtins.print"):
                review_result = layer.review(result_data)

        assert (
            review_result.decision == ReviewDecision.REJECT
        )  # nosec: B101 - Test assertion
        assert review_result.reviewer == "interrupted"  # nosec: B101 - Test assertion

    def test_review_custom_decision(self):
        """Test review with custom decision."""
        layer = HumanControlLayer()
        layer.config["interaction_mode"] = "interactive"

        result_data = {
            "experiment_id": "exp1",
            "metrics": {"accuracy": 0.9},
            "outcomes": {"success": True},
            "analysis": "Good results",
            "confidence": 0.8,
        }

        with patch("builtins.input", side_effect=["4", "MODIFY", "Custom comment"]):
            with patch("builtins.print"):
                review_result = layer.review(result_data)

        assert (
            review_result.decision == ReviewDecision.MODIFY
        )  # nosec: B101 - Test assertion
        assert (
            review_result.comments == "Custom comment"
        )  # nosec: B101 - Test assertion

    def test_review_invalid_custom_decision(self):
        """Test review with invalid custom decision."""
        layer = HumanControlLayer()
        layer.config["interaction_mode"] = "interactive"

        result_data = {
            "experiment_id": "exp1",
            "metrics": {"accuracy": 0.9},
            "outcomes": {"success": True},
            "analysis": "Good results",
            "confidence": 0.8,
        }

        with patch("builtins.input", side_effect=["4", "INVALID", "1"]):
            with patch("builtins.print"):
                review_result = layer.review(result_data)

        assert (
            review_result.decision == ReviewDecision.APPROVE
        )  # nosec: B101 - Test assertion

    def test_review_modifications_gui_mode(self):
        """Test review modifications in GUI mode."""
        layer = HumanControlLayer()
        layer.config["review_threshold"] = 0.7

        result_data = {
            "experiment_id": "exp1",
            "metrics": {"accuracy": 0.5},
            "outcomes": {"success": False},
            "analysis": "Needs improvement",
            "confidence": 0.8,
        }

        with patch("human_layer._is_tkinter_running", return_value=True):
            with patch("builtins.print"):
                review_result = layer.review(result_data)

        assert (
            review_result.decision == ReviewDecision.MODIFY
        )  # nosec: B101 - Test assertion
        assert review_result.modifications is not None  # nosec: B101 - Test assertion
        assert (
            "GUI mode" in review_result.modifications[0]
        )  # nosec: B101 - Test assertion

    def test_review_modifications_console_mode(self):
        """Test review modifications in console mode."""
        layer = HumanControlLayer()
        layer.config["interaction_mode"] = "interactive"
        layer.config["review_threshold"] = 0.7

        result_data = {
            "experiment_id": "exp1",
            "metrics": {"accuracy": 0.5},
            "outcomes": {"success": False},
            "analysis": "Needs improvement",
            "confidence": 0.8,
        }

        with patch("human_layer._is_tkinter_running", return_value=False):
            with patch("builtins.input", side_effect=["2", "fix1", "fix2", ""]):
                with patch("builtins.print"):
                    review_result = layer.review(result_data)

        assert (
            review_result.decision == ReviewDecision.MODIFY
        )  # nosec: B101 - Test assertion
        assert review_result.modifications == [
            "fix1",
            "fix2",
        ]  # nosec: B101 - Test assertion

    def test_evaluate_success_criteria_empty(self):
        """Test _evaluate_success_criteria with empty criteria."""
        layer = HumanControlLayer()
        metrics = {"accuracy": 0.9}
        result = layer._evaluate_success_criteria(metrics, [])
        assert result is True  # nosec: B101 - Test assertion

    def test_evaluate_success_criteria_missing_metric(self):
        """Test _evaluate_success_criteria with missing metric."""
        layer = HumanControlLayer()
        metrics = {"accuracy": 0.9}
        result = layer._evaluate_success_criteria(metrics, ["latency"])
        assert result is False  # nosec: B101 - Test assertion

    def test_evaluate_success_criteria_improvement(self):
        """Test _evaluate_success_criteria with improvement metric."""
        layer = HumanControlLayer()
        metrics = {"improvement": 0.1}
        result = layer._evaluate_success_criteria(metrics, ["improvement"])
        assert result is True  # nosec: B101 - Test assertion

    def test_evaluate_success_criteria_improvement_negative(self):
        """Test _evaluate_success_criteria with negative improvement."""
        layer = HumanControlLayer()
        metrics = {"improvement": -0.1}
        result = layer._evaluate_success_criteria(metrics, ["improvement"])
        assert result is False  # nosec: B101 - Test assertion

    def test_evaluate_success_criteria_accuracy(self):
        """Test _evaluate_success_criteria with accuracy metric."""
        layer = HumanControlLayer()
        metrics = {"accuracy": 0.9}
        result = layer._evaluate_success_criteria(metrics, ["accuracy"])
        assert result is True  # nosec: B101 - Test assertion

    def test_evaluate_success_criteria_accuracy_low(self):
        """Test _evaluate_success_criteria with low accuracy."""
        layer = HumanControlLayer()
        metrics = {"accuracy": 0.7}
        result = layer._evaluate_success_criteria(metrics, ["accuracy"])
        assert result is False  # nosec: B101 - Test assertion

    def test_evaluate_success_criteria_latency(self):
        """Test _evaluate_success_criteria with latency metric."""
        layer = HumanControlLayer()
        metrics = {"latency": 300}
        result = layer._evaluate_success_criteria(metrics, ["latency"])
        assert result is True  # nosec: B101 - Test assertion

    def test_evaluate_success_criteria_latency_high(self):
        """Test _evaluate_success_criteria with high latency."""
        layer = HumanControlLayer()
        metrics = {"latency": 600}
        result = layer._evaluate_success_criteria(metrics, ["latency"])
        assert result is False  # nosec: B101 - Test assertion

    def test_evaluate_success_criteria_non_numeric(self):
        """Test _evaluate_success_criteria with non-numeric metric."""
        layer = HumanControlLayer()
        metrics = {"status": "success"}
        result = layer._evaluate_success_criteria(metrics, ["status"])
        assert result is True  # nosec: B101 - Test assertion

    def test_evaluate_success_criteria_non_numeric_empty(self):
        """Test _evaluate_success_criteria with empty non-numeric metric."""
        layer = HumanControlLayer()
        metrics = {"status": ""}
        result = layer._evaluate_success_criteria(metrics, ["status"])
        assert result is False  # nosec: B101 - Test assertion

    def test_log_review(self):
        """Test _log_review method."""
        layer = HumanControlLayer()
        review_result = ReviewResult(
            decision=ReviewDecision.APPROVE,
            comments="Good",
            reviewer="human",
            timestamp="2024-01-01T00:00:00",
            confidence=0.9,
        )
        experiment_result = {"experiment_id": "exp1"}
        layer._log_review(review_result, experiment_result, True)

        log_path = Path("human_reviews.json")
        assert log_path.exists()  # nosec: B101 - Test assertion
        logs = json.loads(log_path.read_text())
        assert len(logs) > 0  # nosec: B101 - Test assertion
        assert logs[0]["decision"] == "approve"  # nosec: B101 - Test assertion

    def test_get_pending_reviews(self):
        """Test get_pending_reviews method."""
        layer = HumanControlLayer()
        pending = layer.get_pending_reviews()
        assert isinstance(pending, list)  # nosec: B101 - Test assertion
        assert (
            len(pending) == 3
        )  # Default mock returns 3 items  # nosec: B101 - Test assertion

    def test_get_configuration_summary(self):
        """Test get_configuration_summary method."""
        layer = HumanControlLayer()
        summary = layer.get_configuration_summary()
        assert "configured" in summary  # nosec: B101 - Test assertion
        assert "interaction_mode" in summary  # nosec: B101 - Test assertion
        assert "review_threshold" in summary  # nosec: B101 - Test assertion
        assert "task_filters" in summary  # nosec: B101 - Test assertion
        assert "notification_settings" in summary  # nosec: B101 - Test assertion
        assert "pending_tasks" in summary  # nosec: B101 - Test assertion

    def test_get_last_review_summary_no_log(self):
        """Test _get_last_review_summary when no log exists."""
        layer = HumanControlLayer()
        summary = layer._get_last_review_summary()
        assert summary is None  # nosec: B101 - Test assertion

    def test_get_last_review_summary_with_log(self):
        """Test _get_last_review_summary with existing log."""
        layer = HumanControlLayer()
        # Create a log file
        log_data = [
            {
                "review_timestamp": "2024-01-01T00:00:00",
                "decision": "approve",
                "reviewer": "human",
            }
        ]
        log_path = Path("human_reviews.json")
        log_path.write_text(json.dumps(log_data))

        summary = layer._get_last_review_summary()
        assert summary is not None  # nosec: B101 - Test assertion
        assert summary["decision"] == "approve"  # nosec: B101 - Test assertion


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def setup_method(self):
        """Setup test fixtures."""
        config_path = Path("human_config.json")
        if config_path.exists():
            config_path.unlink()

    def teardown_method(self):
        """Cleanup test fixtures."""
        config_path = Path("human_config.json")
        if config_path.exists():
            config_path.unlink()

    def test_configure_if_needed_function(self):
        """Test configure_if_needed convenience function."""
        with patch("human_layer.HumanControlLayer") as mock_layer:
            mock_instance = MagicMock()
            mock_instance.configure_if_needed.return_value = True
            mock_layer.return_value = mock_instance
            result = configure_if_needed()
            assert result is True  # nosec: B101 - Test assertion

    def test_select_task_function(self):
        """Test select_task convenience function."""
        with patch("human_layer.HumanControlLayer") as mock_layer:
            mock_instance = MagicMock()
            mock_task = Task(
                id="task1",
                title="Test",
                description="Test",
                priority=TaskPriority.HIGH,
                experiment_type="cognitive",
            )
            mock_instance.select_task.return_value = mock_task
            mock_layer.return_value = mock_instance
            result = select_task()
            assert result == mock_task  # nosec: B101 - Test assertion

    def test_review_function(self):
        """Test review convenience function."""
        with patch("human_layer.HumanControlLayer") as mock_layer:
            mock_instance = MagicMock()
            mock_result = ReviewResult(
                decision=ReviewDecision.APPROVE,
                comments="Good",
                reviewer="human",
                timestamp="2024-01-01T00:00:00",
                confidence=0.9,
            )
            mock_instance.review.return_value = mock_result
            mock_layer.return_value = mock_instance
            result_data = {"experiment_id": "exp1"}
            result = review(result_data)
            assert result == mock_result  # nosec: B101 - Test assertion


class TestHumanControlLayerEdgeCases:
    """Tests for edge cases in HumanControlLayer."""

    def setup_method(self):
        """Setup test fixtures."""
        config_path = Path("human_config.json")
        if config_path.exists():
            config_path.unlink()
        task_log_path = Path("task_selections.json")
        if task_log_path.exists():
            task_log_path.unlink()
        review_log_path = Path("human_reviews.json")
        if review_log_path.exists():
            review_log_path.unlink()

    def teardown_method(self):
        """Cleanup test fixtures."""
        config_path = Path("human_config.json")
        if config_path.exists():
            config_path.unlink()
        task_log_path = Path("task_selections.json")
        if task_log_path.exists():
            task_log_path.unlink()
        review_log_path = Path("human_reviews.json")
        if review_log_path.exists():
            review_log_path.unlink()

    def test_select_task_all_filtered_out(self):
        """Test select_task when all tasks are filtered out."""
        layer = HumanControlLayer()
        layer.config["task_filters"]["enabled_categories"] = ["performance"]

        task = Task(
            id="task1",
            title="Cognitive Task",
            description="Test",
            priority=TaskPriority.HIGH,
            experiment_type="cognitive",
        )
        layer.add_task(task)

        result = layer.select_task()
        assert result is None  # nosec: B101 - Test assertion

    def test_select_task_invalid_priority(self):
        """Test select_task with invalid priority in config."""
        layer = HumanControlLayer()
        layer.config["task_filters"]["min_priority"] = "invalid"

        task = Task(
            id="task1",
            title="Test",
            description="Test",
            priority=TaskPriority.HIGH,
            experiment_type="cognitive",
        )
        layer.add_task(task)

        result = layer.select_task()
        # Should still work with default priority value
        assert result is not None  # nosec: B101 - Test assertion

    def test_review_with_missing_fields(self):
        """Test review with missing result fields."""
        layer = HumanControlLayer()
        layer.config["interaction_mode"] = "autonomous"

        result_data: dict[str, object] = {}  # Empty result

        with patch("builtins.print"):
            review_result = layer.review(result_data)

        # Should handle missing fields gracefully
        assert (
            review_result.decision == ReviewDecision.REJECT
        )  # nosec: B101 - Test assertion

    def test_review_log_save_failure(self):
        """Test review log save failure."""
        layer = HumanControlLayer()
        review_result = ReviewResult(
            decision=ReviewDecision.APPROVE,
            comments="Good",
            reviewer="human",
            timestamp="2024-01-01T00:00:00",
            confidence=0.9,
        )
        experiment_result = {"experiment_id": "exp1"}

        # Make log file unwritable
        log_path = Path("human_reviews.json")
        log_path.write_text("test")
        log_path.chmod(0o000)

        try:
            layer._log_review(review_result, experiment_result, True)
            # Should not raise, just log error
        finally:
            log_path.chmod(0o644)
            if log_path.exists():
                log_path.unlink()

    def test_task_selection_log_save_failure(self):
        """Test task selection log save failure."""
        layer = HumanControlLayer()
        task = Task(
            id="task1",
            title="Test",
            description="Test",
            priority=TaskPriority.HIGH,
            experiment_type="cognitive",
        )
        layer.add_task(task)

        # Make log file unwritable
        log_path = Path("task_selections.json")
        log_path.write_text("test")
        log_path.chmod(0o000)

        try:
            layer.select_task()
            # Should not raise, just log warning
        finally:
            log_path.chmod(0o644)
            if log_path.exists():
                log_path.unlink()

    def test_config_save_failure(self):
        """Test config save failure."""
        layer = HumanControlLayer()
        layer.config["test"] = "value"

        # Make config file unwritable
        config_path = Path("human_config.json")
        config_path.write_text("test")
        config_path.chmod(0o000)

        try:
            layer._save_config()
            # Should not raise, just log error
        finally:
            config_path.chmod(0o644)
            if config_path.exists():
                config_path.unlink()
