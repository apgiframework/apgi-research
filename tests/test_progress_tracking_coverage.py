# mypy: ignore-errors
"""
Enhanced tests for progress_tracking.py - Covering missing lines to reach 100%.

This module tests the previously uncovered functionality:
- Auto-save functionality
- Progress and trial callbacks
- APGI metrics integration
- Trial management
- Estimated time calculation
- Summary report export
- Error handling paths
"""

from __future__ import annotations

import time
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest

from progress_tracking import (
    Checkpoint,
    ExperimentProgress,
    ProgressMonitor,
    ProgressReport,
    ProgressStatus,
    ProgressTracker,
    TaskProgress,
    TrialResult,
    create_progress_tracker,
    load_progress_tracker,
    monitor_experiment_progress,
)


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """Trackers default to writing ./progress, so run every test in a temp dir."""
    monkeypatch.chdir(tmp_path)


def make_tracker(tmp_path, **kwargs):
    """A tracker that writes into the test's temp directory."""
    kwargs.setdefault("experiment_name", "test-exp")
    kwargs.setdefault("participant_id", "p1")
    kwargs.setdefault("total_trials", 10)
    return ProgressTracker(output_dir=str(tmp_path), **kwargs)


def make_trial(number, **kwargs):
    return TrialResult(trial_number=number, timestamp=time.time(), **kwargs)


class TestTaskProgressPostInit:
    """Test TaskProgress __post_init__ behavior."""

    def test_post_init_sets_start_time_in_progress(self):
        """Test that start_time is set when status is IN_PROGRESS."""
        task = TaskProgress(task_id="test", status=ProgressStatus.IN_PROGRESS)
        assert task.start_time is not None  # nosec: B101 - Test assertion
        assert task.start_time > 0  # nosec: B101 - Test assertion

    def test_post_init_no_start_time_pending(self):
        """Test that start_time is not set when status is PENDING."""
        task = TaskProgress(task_id="test", status=ProgressStatus.PENDING)
        assert task.start_time is None  # nosec: B101 - Test assertion

    def test_post_init_existing_start_time_preserved(self):
        """Test that existing start_time is preserved."""
        start = time.time() - 100
        task = TaskProgress(
            task_id="test", status=ProgressStatus.IN_PROGRESS, start_time=start
        )
        assert task.start_time == start  # nosec: B101 - Test assertion


class TestTaskProgressDuration:
    """Test TaskProgress duration calculations."""

    def test_duration_with_end_time(self):
        """Test duration calculation with explicit end_time."""
        start = time.time() - 10
        end = time.time()
        task = TaskProgress(task_id="test", start_time=start, end_time=end)
        assert task.duration_seconds is not None  # nosec: B101 - Test assertion
        assert (
            abs(task.duration_seconds - 10) < 1
        )  # Allow 1 second tolerance  # nosec: B101 - Test assertion

    def test_duration_no_start_time(self):
        """Test duration returns None when no start_time."""
        task = TaskProgress(task_id="test")
        assert task.duration_seconds is None  # nosec: B101 - Test assertion


class TestProgressReportCalculate:
    """Test ProgressReport.calculate method variations."""

    def test_calculate_with_failed_tasks(self):
        """Test calculation including failed tasks."""
        tasks = [
            TaskProgress(
                task_id="1", status=ProgressStatus.COMPLETED, progress_percent=100
            ),
            TaskProgress(
                task_id="2", status=ProgressStatus.FAILED, progress_percent=50
            ),
            TaskProgress(
                task_id="3", status=ProgressStatus.CANCELLED, progress_percent=25
            ),
        ]
        report = ProgressReport.calculate(tasks)
        assert report.failed_tasks == 1  # nosec: B101 - Test assertion
        assert report.completed_tasks == 1  # nosec: B101 - Test assertion

    def test_calculate_empty_tasks(self):
        """Test calculation with empty task list."""
        report = ProgressReport.calculate([])
        assert report.total_tasks == 0  # nosec: B101 - Test assertion
        assert report.overall_progress == 0.0  # nosec: B101 - Test assertion

    def test_calculate_only_pending(self):
        """Test calculation with only pending tasks."""
        tasks = [
            TaskProgress(
                task_id="1", status=ProgressStatus.PENDING, progress_percent=0
            ),
            TaskProgress(
                task_id="2", status=ProgressStatus.PENDING, progress_percent=0
            ),
        ]
        report = ProgressReport.calculate(tasks)
        assert report.pending_tasks == 2  # nosec: B101 - Test assertion
        assert report.overall_progress == 0.0  # nosec: B101 - Test assertion


class TestProgressTrackerAdvanced:
    """Test ProgressTracker advanced functionality."""

    def test_get_report_empty(self):
        """Test getting report with no tasks."""
        tracker = ProgressTracker()
        report = tracker.get_report()
        assert report.total_tasks == 0  # nosec: B101 - Test assertion
        assert report.overall_progress == 0.0  # nosec: B101 - Test assertion

    def test_get_report_with_failed_tasks(self):
        """Test report with failed tasks."""
        tracker = ProgressTracker()
        tracker.add_task("task-1")
        tracker.fail_task("task-1")
        report = tracker.get_report()
        assert report.failed_tasks == 1  # nosec: B101 - Test assertion

    def test_get_report_with_cancelled_tasks(self):
        """Test report with cancelled tasks."""
        tracker = ProgressTracker()
        tracker.add_task("task-1")
        task = tracker.tasks["task-1"]
        task.status = ProgressStatus.CANCELLED
        _ = tracker.get_report()
        # Cancelled tasks are not in any specific count

    def test_get_elapsed_time_not_started(self):
        """A tracker that has only just been created has almost no elapsed time."""
        tracker = ProgressTracker()
        elapsed = tracker.get_elapsed_time()
        assert 0.0 <= elapsed < 1.0  # nosec: B101 - Test assertion

    def test_get_elapsed_time_with_start_time(self):
        """Elapsed time is measured from the experiment's start time."""
        tracker = ProgressTracker()
        tracker.progress.start_time = time.time() - 5
        elapsed = tracker.get_elapsed_time()
        assert 5.0 <= elapsed < 6.0  # nosec: B101 - Test assertion

    def test_get_elapsed_time_stops_at_end_time(self):
        """Once the experiment has ended, elapsed time stops growing."""
        tracker = ProgressTracker()
        tracker.progress.start_time = 100.0
        tracker.progress.end_time = 130.0
        assert tracker.get_elapsed_time() == 30.0  # nosec: B101

    def test_get_estimated_remaining_time_no_tasks(self):
        """Test estimated time with no tasks."""
        tracker = ProgressTracker()
        remaining = tracker.get_estimated_remaining_time()
        assert remaining is None  # nosec: B101 - Test assertion

    def test_get_estimated_remaining_time_zero_progress(self):
        """With no completed trials there is nothing to extrapolate from."""
        tracker = ProgressTracker(total_trials=10)
        tracker.add_task("task-1")
        assert tracker.get_estimated_remaining_time() is None  # nosec: B101

    def test_get_estimated_remaining_time_with_progress(self):
        """Remaining time extrapolates the average time per completed trial."""
        tracker = ProgressTracker(total_trials=10)
        tracker.progress.start_time = time.time() - 10
        tracker.progress.completed_trials = 5

        remaining = tracker.get_estimated_remaining_time()

        # 5 trials took ~10s, so the other 5 should take ~10s
        assert remaining == pytest.approx(10.0, abs=0.5)  # nosec: B101

    def test_get_progress_percentage(self):
        """Progress is completed trials over total trials."""
        tracker = ProgressTracker(total_trials=4)
        assert tracker.get_progress_percentage() == 0.0  # nosec: B101
        tracker.progress.completed_trials = 1
        assert tracker.get_progress_percentage() == 25.0  # nosec: B101

    def test_get_progress_percentage_without_total(self):
        """An unknown trial total reports zero rather than dividing by zero."""
        assert ProgressTracker(total_trials=0).get_progress_percentage() == 0.0


class TestProgressTrackerCallbacks:
    """Test ProgressTracker callback functionality."""

    def test_add_progress_callback(self, tmp_path):
        """Progress callbacks receive the experiment progress after each trial."""
        tracker = make_tracker(tmp_path)
        received = []
        tracker.add_progress_callback(received.append)

        tracker.complete_trial(make_trial(1))

        assert received == [tracker.progress]  # nosec: B101 - Test assertion

    def test_add_trial_callback(self, tmp_path):
        """Trial callbacks receive each completed trial."""
        tracker = make_tracker(tmp_path)
        received = []
        tracker.add_trial_callback(received.append)

        tracker.start_trial(1)
        trial = make_trial(1, response_time=0.4, accuracy=1.0)
        tracker.complete_trial(trial)

        assert received == [trial]  # nosec: B101 - Test assertion

    def test_callback_error_handling(self, tmp_path):
        """A failing callback is logged and does not break the tracker."""
        tracker = make_tracker(tmp_path)

        def bad_callback(_):
            raise Exception("Callback error")

        tracker.add_trial_callback(bad_callback)
        tracker.add_progress_callback(bad_callback)

        tracker.complete_trial(make_trial(1))  # Should not raise

        assert tracker.progress.completed_trials == 1  # nosec: B101
        assert any(  # nosec: B101 - Test assertion
            "Trial callback error" in entry for entry in tracker.progress.error_log
        )
        assert any(  # nosec: B101 - Test assertion
            "Progress callback error" in entry for entry in tracker.progress.error_log
        )


class TestProgressTrackerAPGI:
    """Test ProgressTracker APGI integration."""

    def test_update_apgi_metrics(self, tmp_path):
        """APGI metrics come from the integration's finalize() summary."""
        tracker = make_tracker(tmp_path)
        integration = MagicMock()
        integration.finalize.return_value = {
            "tau_S": 0.35,
            "tau_theta": 30.0,
            "alpha": 5.5,
        }

        tracker.update_apgi_metrics(integration)

        assert tracker.progress.apgi_summary == {  # nosec: B101
            "tau_S": 0.35,
            "tau_theta": 30.0,
            "alpha": 5.5,
        }
        assert tracker.get_apgi_statistics()["alpha"] == 5.5  # nosec: B101

    def test_update_apgi_metrics_with_errors(self, tmp_path):
        """A failing integration is logged rather than raised."""
        tracker = make_tracker(tmp_path)
        integration = MagicMock()
        integration.finalize.side_effect = RuntimeError("no data")

        tracker.update_apgi_metrics(integration)  # Should not raise

        assert tracker.progress.apgi_summary is None  # nosec: B101
        assert any(  # nosec: B101 - Test assertion
            "Failed to update APGI metrics" in entry
            for entry in tracker.progress.error_log
        )

    def test_get_apgi_statistics(self):
        """Test getting APGI statistics."""
        tracker = ProgressTracker()
        stats = tracker.get_apgi_statistics()
        assert isinstance(stats, dict)  # nosec: B101 - Test assertion
        assert stats == {}  # nosec: B101 - Test assertion


class TestProgressTrackerTrials:
    """Test ProgressTracker trial management."""

    def test_start_trial(self, tmp_path):
        """Starting a trial records which trial is current."""
        tracker = make_tracker(tmp_path)
        tracker.start_trial(3)
        assert tracker.progress.current_trial == 3  # nosec: B101

    def test_complete_trial(self, tmp_path):
        """Completing a trial stores it and advances the completed count."""
        tracker = make_tracker(tmp_path)
        trial = make_trial(1, response_time=0.5, accuracy=1.0)

        tracker.complete_trial(trial)

        assert tracker.progress.trials == [trial]  # nosec: B101
        assert tracker.progress.completed_trials == 1  # nosec: B101

    def test_complete_trial_with_metadata(self, tmp_path):
        """Trial metadata is preserved."""
        tracker = make_tracker(tmp_path)
        tracker.complete_trial(make_trial(1, metadata={"note": "success"}))

        assert (  # nosec: B101 - Test assertion
            tracker.progress.trials[0].metadata.get("note") == "success"
        )

    def test_get_trial_statistics(self, tmp_path):
        """Statistics summarise response times and accuracy."""
        tracker = make_tracker(tmp_path)
        for i, (rt, acc) in enumerate([(0.2, 0.8), (0.4, 0.9), (0.6, 1.0)], start=1):
            tracker.complete_trial(make_trial(i, response_time=rt, accuracy=acc))

        stats = tracker.get_trial_statistics()

        assert stats["completed_trials"] == 3  # nosec: B101
        assert stats["error_trials"] == 0  # nosec: B101 - Test assertion
        assert stats["avg_response_time"] == pytest.approx(0.4)  # nosec: B101
        assert stats["min_response_time"] == 0.2  # nosec: B101
        assert stats["max_response_time"] == 0.6  # nosec: B101
        assert stats["avg_accuracy"] == pytest.approx(0.9)  # nosec: B101

    def test_get_trial_statistics_with_failures(self, tmp_path):
        """Trials that recorded an error are counted separately."""
        tracker = make_tracker(tmp_path)
        tracker.complete_trial(make_trial(1, response_time=0.5, accuracy=1.0))
        tracker.complete_trial(make_trial(2, error="timeout"))

        stats = tracker.get_trial_statistics()

        assert stats["completed_trials"] == 2  # nosec: B101
        assert stats["error_trials"] == 1  # nosec: B101 - Test assertion
        # The errored trial has no timing, so it does not skew the averages
        assert stats["avg_response_time"] == 0.5  # nosec: B101

    def test_get_trial_statistics_empty(self, tmp_path):
        """No completed trials means no statistics."""
        assert make_tracker(tmp_path).get_trial_statistics() == {}  # nosec: B101


class TestProgressTrackerExperimentLifecycle:
    """Test experiment lifecycle management."""

    def test_complete_experiment(self, tmp_path):
        """Completing an experiment records status and end time."""
        tracker = make_tracker(tmp_path)
        tracker.add_task("task-1")
        tracker.complete_task("task-1")

        tracker.complete_experiment()

        assert tracker.progress.status == "completed"  # nosec: B101
        assert tracker.progress.end_time is not None  # nosec: B101

    def test_complete_experiment_with_summary(self, tmp_path):
        """An experiment can be closed with a non-default status."""
        tracker = make_tracker(tmp_path)

        tracker.complete_experiment(status="failed")

        assert tracker.progress.status == "failed"  # nosec: B101
        assert tracker.export_summary()["experiment_info"]["status"] == "failed"

    def test_pause_and_resume_experiment(self, tmp_path):
        """Pausing and resuming toggles the experiment status."""
        tracker = make_tracker(tmp_path)

        tracker.pause_experiment()
        assert tracker.progress.status == "paused"  # nosec: B101

        tracker.resume_experiment()
        assert tracker.progress.status == "running"  # nosec: B101

    def test_resume_experiment_not_paused(self, tmp_path):
        """Resuming an experiment that is not paused changes nothing."""
        tracker = make_tracker(tmp_path)
        tracker.resume_experiment()  # Should handle gracefully
        assert tracker.progress.status == "running"  # nosec: B101

        tracker.complete_experiment()
        tracker.resume_experiment()
        assert tracker.progress.status == "completed"  # nosec: B101


class TestProgressTrackerAutoSave:
    """Test ProgressTracker auto-save functionality."""

    def test_auto_save_on_task_update(self, tmp_path):
        """Test auto-save triggered on task updates."""
        tracker = ProgressTracker(
            experiment_name="test",
            participant_id="p1",
            total_trials=10,
            output_dir=str(tmp_path),
        )
        tracker.add_task("task-1")
        tracker.update_task("task-1", progress_percent=50)
        # Auto-save is triggered internally on updates

    def test_save_progress_called(self, tmp_path):
        """Test that save progress is called."""
        tracker = ProgressTracker(
            experiment_name="test",
            participant_id="p1",
            total_trials=10,
            output_dir=str(tmp_path),
        )
        tracker.add_task("task-1")
        # Manually trigger save
        tracker._save_progress()

    def test_auto_save_waits_for_the_interval(self, tmp_path):
        """Auto-save only writes once the save interval has elapsed."""
        tracker = make_tracker(tmp_path)
        saved = tmp_path / "test-exp_p1_progress.json"

        tracker.log_error("first")  # Interval not yet elapsed
        assert not saved.exists()  # nosec: B101 - Test assertion

        tracker._last_save = time.time() - tracker._save_interval - 1
        tracker.log_error("second")
        assert saved.exists()  # nosec: B101 - Test assertion

    def test_auto_save_can_be_disabled(self, tmp_path):
        """Disabling auto-save suppresses the periodic write."""
        tracker = make_tracker(tmp_path)
        tracker._auto_save_enabled = False
        tracker._last_save = time.time() - tracker._save_interval - 1

        tracker.log_error("ignored")

        assert not (tmp_path / "test-exp_p1_progress.json").exists()  # nosec: B101


class TestProgressTrackerSaveAndLoad:
    """Test ProgressTracker save and load functionality."""

    def test_save_progress(self, tmp_path):
        """Test saving progress."""
        tracker = ProgressTracker(
            experiment_name="test",
            participant_id="p1",
            total_trials=10,
            output_dir=str(tmp_path),
        )
        tracker.add_task("task-1", progress_percent=50)

        tracker._save_progress()

        # Check file was created
        progress_files = list(tmp_path.glob("*_progress.json"))
        assert len(progress_files) > 0  # nosec: B101 - Test assertion

    def test_save_progress_failure(self, tmp_path):
        """Test save progress failure."""
        tracker = ProgressTracker(
            experiment_name="test",
            participant_id="p1",
            total_trials=10,
            output_dir=str(tmp_path),
        )

        with patch("builtins.open", side_effect=IOError("Write error")):
            # Should not raise, just log error
            tracker._save_progress()

    def test_load_progress(self, tmp_path):
        """Saved progress can be loaded back into a new tracker."""
        tracker1 = make_tracker(tmp_path, experiment_name="test-exp")
        tracker1.start_trial(1)
        tracker1.complete_trial(make_trial(1, response_time=0.5, accuracy=1.0))
        tracker1.pause_experiment()  # Pausing saves to disk

        tracker2 = make_tracker(tmp_path, experiment_name="test-exp")
        result = tracker2.load_progress()

        assert result is True  # nosec: B101 - Test assertion
        assert tracker2.progress.status == "paused"  # nosec: B101
        assert tracker2.progress.completed_trials == 1  # nosec: B101
        assert len(tracker2.progress.trials) == 1  # nosec: B101

    def test_load_progress_no_file(self, tmp_path):
        """Test loading progress when no file exists."""
        tracker = ProgressTracker(
            experiment_name="nonexistent",
            participant_id="p1",
            total_trials=10,
            output_dir=str(tmp_path),
        )

        result = tracker.load_progress()
        assert result is False  # nosec: B101 - Test assertion

    def test_load_progress_invalid_json(self, tmp_path):
        """A corrupt progress file is reported as not loaded, not raised."""
        tracker = make_tracker(tmp_path, experiment_name="test-exp")
        (tmp_path / "test-exp_p1_progress.json").write_text("{not valid json")

        result = tracker.load_progress()
        assert result is False  # nosec: B101 - Test assertion


class TestProgressTrackerExportSummary:
    """Test export summary functionality."""

    def test_export_summary(self, tmp_path):
        """The exported summary describes the experiment and its progress."""
        tracker = make_tracker(tmp_path, total_trials=4)
        tracker.complete_trial(make_trial(1, response_time=0.5, accuracy=1.0))
        tracker.log_error("something odd")

        summary = tracker.export_summary()

        assert summary["experiment_info"]["name"] == "test-exp"  # nosec: B101
        assert summary["experiment_info"]["participant_id"] == "p1"  # nosec: B101
        assert summary["progress"]["completed_trials"] == 1  # nosec: B101
        assert summary["progress"]["progress_percentage"] == 25.0  # nosec: B101
        assert summary["trial_statistics"]["completed_trials"] == 1  # nosec: B101
        assert summary["errors"]["total_errors"] == 1  # nosec: B101

    def test_save_summary_report(self, tmp_path):
        """Saving the report writes the summary and the progress file."""
        tracker = make_tracker(tmp_path)
        tracker.add_task("task-1")

        path = tracker.save_summary_report()

        assert path == tmp_path / "test-exp_p1_summary.json"  # nosec: B101
        assert path.exists()  # nosec: B101 - Test assertion
        assert (tmp_path / "test-exp_p1_progress.json").exists()  # nosec: B101


class TestProgressTrackerErrorHandling:
    """Test ProgressTracker error handling."""

    def test_log_error(self):
        """Errors are appended to the log with a timestamp."""
        tracker = ProgressTracker()
        tracker.log_error("Test error message")

        assert len(tracker.progress.error_log) == 1  # nosec: B101
        assert tracker.progress.error_log[0].startswith("[")  # nosec: B101
        assert tracker.progress.error_log[0].endswith(  # nosec: B101
            "Test error message"
        )

    def test_update_nonexistent_task_creates(self):
        """Test that updating nonexistent task creates it."""
        tracker = ProgressTracker()
        tracker.update_task(
            "new-task", progress_percent=75, message="Created and updated"
        )

        assert "new-task" in tracker.tasks  # nosec: B101 - Test assertion
        assert (
            tracker.tasks["new-task"].progress_percent == 75
        )  # nosec: B101 - Test assertion
        assert (
            tracker.tasks["new-task"].message == "Created and updated"
        )  # nosec: B101 - Test assertion


class TestExperimentProgress:
    """Test ExperimentProgress dataclass."""

    def test_experiment_progress_creation(self):
        """Test creating ExperimentProgress."""
        progress = ExperimentProgress(
            experiment_name="Test Experiment",
            participant_id="p1",
            start_time=time.time(),
            current_trial=0,
            total_trials=10,
            completed_trials=0,
            trials=[],
        )
        assert progress.experiment_name == "Test Experiment"  # nosec: B101
        assert progress.participant_id == "p1"  # nosec: B101 - Test assertion
        assert progress.status == "running"  # nosec: B101 - Test assertion
        assert progress.error_log == []  # nosec: B101 - Test assertion

    def test_experiment_progress_with_tasks(self):
        """ExperimentProgress holds the trials completed so far."""
        trials = [make_trial(1), make_trial(2)]
        progress = ExperimentProgress(
            experiment_name="Test",
            participant_id="p1",
            start_time=time.time(),
            current_trial=2,
            total_trials=10,
            completed_trials=2,
            trials=trials,
        )
        assert len(progress.trials) == 2  # nosec: B101 - Test assertion
        assert progress.completed_trials == 2  # nosec: B101 - Test assertion


class TestProgressMonitor:
    """Test ProgressMonitor class."""

    def test_init(self):
        """Test ProgressMonitor initialization."""
        monitor = ProgressMonitor()
        assert monitor.active_experiments == {}  # nosec: B101 - Test assertion

    def test_register_experiment(self, tmp_path):
        """Experiments are registered under their name and participant."""
        monitor = ProgressMonitor()
        tracker = make_tracker(tmp_path, experiment_name="exp-1")
        monitor.register_experiment(tracker)

        assert "exp-1_p1" in monitor.active_experiments  # nosec: B101
        assert monitor.active_experiments["exp-1_p1"] is tracker  # nosec: B101

    def test_get_overall_status(self, tmp_path):
        """Overall status counts experiments by state and lists each one."""
        monitor = ProgressMonitor()

        tracker1 = make_tracker(tmp_path, experiment_name="exp-1", total_trials=2)
        tracker1.complete_trial(make_trial(1))
        tracker1.complete_experiment()

        tracker2 = make_tracker(tmp_path, experiment_name="exp-2")

        monitor.register_experiment(tracker1)
        monitor.register_experiment(tracker2)

        status = monitor.get_overall_status()
        assert status["total_experiments"] == 2  # nosec: B101
        assert status["completed_experiments"] == 1  # nosec: B101
        assert status["running_experiments"] == 1  # nosec: B101
        assert status["failed_experiments"] == 0  # nosec: B101
        assert status["experiments"]["exp-1_p1"]["progress_percentage"] == 50.0
        assert "exp-2_p1" in status["experiments"]  # nosec: B101

    def test_get_experiment_tracker(self, tmp_path):
        """Test getting experiment tracker."""
        monitor = ProgressMonitor()
        tracker = make_tracker(tmp_path, experiment_name="exp-1")
        monitor.register_experiment(tracker)

        result = monitor.get_experiment_tracker("exp-1", "p1")
        assert result is tracker  # nosec: B101 - Test assertion

    def test_get_experiment_tracker_nonexistent(self):
        """Test getting nonexistent experiment tracker."""
        monitor = ProgressMonitor()
        result = monitor.get_experiment_tracker("nonexistent", "p1")
        assert result is None  # nosec: B101 - Test assertion


class TestTrialResult:
    """Test TrialResult dataclass."""

    def test_trial_result_creation(self):
        """Optional fields default to None."""
        trial = TrialResult(trial_number=1, timestamp=123.0)
        assert trial.trial_number == 1  # nosec: B101 - Test assertion
        assert trial.timestamp == 123.0  # nosec: B101 - Test assertion
        assert trial.response_time is None  # nosec: B101 - Test assertion
        assert trial.accuracy is None  # nosec: B101 - Test assertion
        assert trial.error is None  # nosec: B101 - Test assertion

    def test_trial_result_completed(self):
        """A fully populated TrialResult keeps every field."""
        trial = TrialResult(
            trial_number=1,
            timestamp=123.0,
            response_time=0.45,
            accuracy=1.0,
            apgi_metrics={"theta": 0.7},
            metadata={"block": 2},
        )
        assert trial.response_time == 0.45  # nosec: B101 - Test assertion
        assert trial.accuracy == 1.0  # nosec: B101 - Test assertion
        assert trial.apgi_metrics["theta"] == 0.7  # nosec: B101
        assert trial.metadata["block"] == 2  # nosec: B101 - Test assertion


class TestCheckpointEdgeCases:
    """Test Checkpoint edge cases."""

    def test_checkpoint_to_dict(self):
        """A Checkpoint serialises to a plain dict."""
        checkpoint = Checkpoint(
            checkpoint_id="cp-1",
            timestamp=time.time(),
            task_states={"task1": {"progress": 50}},
            metadata={"version": "1.0"},
        )
        data = asdict(checkpoint)
        assert data["checkpoint_id"] == "cp-1"  # nosec: B101 - Test assertion
        assert "task_states" in data  # nosec: B101 - Test assertion

    def test_checkpoint_from_dict(self):
        """A Checkpoint round-trips through its dict form."""
        data = {
            "checkpoint_id": "cp-1",
            "timestamp": time.time(),
            "task_states": {"task1": {"progress": 50}},
            "metadata": {"version": "1.0"},
        }
        checkpoint = Checkpoint(**data)
        assert checkpoint.checkpoint_id == "cp-1"  # nosec: B101 - Test assertion
        assert (
            checkpoint.task_states["task1"]["progress"] == 50
        )  # nosec: B101 - Test assertion
        assert asdict(checkpoint) == data  # nosec: B101 - Test assertion


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_progress_tracker(self, tmp_path):
        """Test create_progress_tracker function."""
        tracker = create_progress_tracker(
            experiment_name="test-exp",
            participant_id="p1",
            total_trials=10,
            output_dir=str(tmp_path),
        )
        assert tracker.experiment_name == "test-exp"  # nosec: B101
        assert tracker.progress.total_trials == 10  # nosec: B101

    def test_load_progress_tracker_exists(self, tmp_path):
        """Test load_progress_tracker when file exists."""
        # First create and save a tracker
        tracker1 = create_progress_tracker(
            experiment_name="test-exp",
            participant_id="p1",
            total_trials=10,
            output_dir=str(tmp_path),
        )
        tracker1._save_progress()

        # Now load it
        tracker2 = load_progress_tracker(
            experiment_name="test-exp",
            participant_id="p1",
            total_trials=10,
            output_dir=str(tmp_path),
        )
        assert tracker2 is not None  # nosec: B101 - Test assertion

    def test_load_progress_tracker_not_exists(self, tmp_path):
        """Test load_progress_tracker when file doesn't exist."""
        tracker = load_progress_tracker(
            experiment_name="nonexistent",
            participant_id="p1",
            total_trials=10,
            output_dir=str(tmp_path),
        )
        assert tracker is None  # nosec: B101 - Test assertion

    def test_monitor_experiment_progress(self, tmp_path):
        """The callback receives the saved progress of a running experiment."""
        # The monitor looks in the default ./progress directory, which the
        # autouse fixture has made the (empty) temp working directory.
        tracker = create_progress_tracker(
            experiment_name="test-exp", participant_id="p1", total_trials=10
        )
        tracker.complete_trial(make_trial(1))
        tracker._save_progress()

        callback_calls = []
        monitor_experiment_progress(
            experiment_name="test-exp",
            participant_id="p1",
            update_callback=callback_calls.append,
        )

        assert len(callback_calls) == 1  # nosec: B101 - Test assertion
        assert callback_calls[0].experiment_name == "test-exp"  # nosec: B101
        assert callback_calls[0].completed_trials == 1  # nosec: B101

    def test_monitor_experiment_progress_unknown_experiment(self):
        """Nothing is reported for an experiment that has no saved progress."""
        callback_calls = []
        monitor_experiment_progress(
            experiment_name="missing",
            participant_id="p1",
            update_callback=callback_calls.append,
        )
        assert callback_calls == []  # nosec: B101 - Test assertion


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
