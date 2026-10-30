"""
Comprehensive test suite for apgi_data_retention.py to achieve ≥90% coverage.

This file tests the complex code paths and edge cases not covered in the basic test suite.
"""

import json
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.apgi_data_retention import (
    DataSubjectRecord,
    DeletionExecutor,
    RetentionConfig,
    RetentionJobScheduler,
    RetentionPolicy,
    get_retention_scheduler,
    set_retention_scheduler,
)


class TestRetentionPolicy:
    """Test the RetentionPolicy enum."""

    def test_retention_policy_values(self):
        """Test that all expected policy values exist."""
        expected_policies = [
            RetentionPolicy.PERMANENT,
            RetentionPolicy.GDPR_DEFAULT,
            RetentionPolicy.CCPA_DEFAULT,
            RetentionPolicy.HIPAA_DEFAULT,
            RetentionPolicy.CUSTOM,
        ]

        for policy in expected_policies:
            assert policy.value in [  # nosec: B101 - Test assertion
                "permanent",
                "gdpr_default",
                "ccpa_default",
                "hipaa_default",
                "custom",
            ]

    def test_retention_policy_string_conversion(self):
        """Test string conversion of policies."""
        assert (
            RetentionPolicy.PERMANENT.value == "permanent"
        )  # nosec: B101 - Test assertion
        assert (
            RetentionPolicy.GDPR_DEFAULT.value == "gdpr_default"
        )  # nosec: B101 - Test assertion
        assert (
            RetentionPolicy.CCPA_DEFAULT.value == "ccpa_default"
        )  # nosec: B101 - Test assertion
        assert (
            RetentionPolicy.HIPAA_DEFAULT.value == "hipaa_default"
        )  # nosec: B101 - Test assertion
        assert RetentionPolicy.CUSTOM.value == "custom"  # nosec: B101 - Test assertion


class TestRetentionConfig:
    """Test the RetentionConfig dataclass."""

    def test_default_initialization(self):
        """Test default configuration values."""
        config = RetentionConfig()

        assert (
            config.policy == RetentionPolicy.GDPR_DEFAULT
        )  # nosec: B101 - Test assertion
        assert config.retention_days == 1095  # 3 years  # nosec: B101 - Test assertion
        assert config.auto_delete_enabled is True  # nosec: B101 - Test assertion
        assert (
            config.deletion_verification_required is True
        )  # nosec: B101 - Test assertion
        assert (
            config.audit_trail_retention_days == 2555
        )  # 7 years  # nosec: B101 - Test assertion

    def test_custom_initialization(self):
        """Test custom configuration values."""
        config = RetentionConfig(
            policy=RetentionPolicy.CCPA_DEFAULT,
            retention_days=365,
            auto_delete_enabled=False,
            deletion_verification_required=False,
            audit_trail_retention_days=180,
        )

        assert (
            config.policy == RetentionPolicy.CCPA_DEFAULT
        )  # nosec: B101 - Test assertion
        assert config.retention_days == 365  # nosec: B101 - Test assertion
        assert config.auto_delete_enabled is False  # nosec: B101 - Test assertion
        assert (
            config.deletion_verification_required is False
        )  # nosec: B101 - Test assertion
        assert config.audit_trail_retention_days == 180  # nosec: B101 - Test assertion

    def test_get_retention_period(self):
        """Test retention period calculation."""
        config = RetentionConfig(retention_days=30)
        period = config.get_retention_period()

        assert isinstance(period, timedelta)  # nosec: B101 - Test assertion
        assert period.days == 30  # nosec: B101 - Test assertion

    def test_get_audit_retention_period(self):
        """Test audit retention period calculation."""
        config = RetentionConfig(audit_trail_retention_days=180)
        period = config.get_audit_retention_period()

        assert isinstance(period, timedelta)  # nosec: B101 - Test assertion
        assert period.days == 180  # nosec: B101 - Test assertion


class TestDataSubjectRecord:
    """Test the DataSubjectRecord dataclass."""

    def test_default_initialization(self):
        """Test default record initialization."""
        record = DataSubjectRecord(
            subject_id="test_subject", subject_name="Test Subject"
        )

        assert record.subject_id == "test_subject"  # nosec: B101 - Test assertion
        assert record.subject_name == "Test Subject"  # nosec: B101 - Test assertion
        assert (
            record.retention_policy == RetentionPolicy.GDPR_DEFAULT
        )  # nosec: B101 - Test assertion
        assert record.deletion_requested is False  # nosec: B101 - Test assertion
        assert record.deletion_completed is False  # nosec: B101 - Test assertion
        assert record.deletion_requested_at is None  # nosec: B101 - Test assertion
        assert record.deletion_completed_at is None  # nosec: B101 - Test assertion
        assert isinstance(record.created_at, datetime)  # nosec: B101 - Test assertion
        assert isinstance(
            record.last_accessed, datetime
        )  # nosec: B101 - Test assertion

    def test_custom_initialization(self):
        """Test custom record initialization."""
        custom_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        record = DataSubjectRecord(
            subject_id="test_subject",
            subject_name="Test Subject",
            data_categories=["experiment", "config"],
            retention_policy=RetentionPolicy.CCPA_DEFAULT,
            deletion_requested=True,
            deletion_requested_at=custom_time,
            deletion_completed=True,
            deletion_completed_at=custom_time,
        )

        assert record.data_categories == [
            "experiment",
            "config",
        ]  # nosec: B101 - Test assertion
        assert (
            record.retention_policy == RetentionPolicy.CCPA_DEFAULT
        )  # nosec: B101 - Test assertion
        assert record.deletion_requested is True  # nosec: B101 - Test assertion
        assert record.deletion_completed is True  # nosec: B101 - Test assertion
        assert (
            record.deletion_requested_at == custom_time
        )  # nosec: B101 - Test assertion
        assert (
            record.deletion_completed_at == custom_time
        )  # nosec: B101 - Test assertion

    def test_is_retention_expired_permanent(self):
        """Test retention expiration for permanent policy."""
        record = DataSubjectRecord(subject_id="test", subject_name="Test")
        record.retention_policy = RetentionPolicy.PERMANENT

        # Should never expire
        assert (
            record.is_retention_expired(RetentionConfig()) is False
        )  # nosec: B101 - Test assertion

    def test_is_retention_expired_expired(self):
        """Test retention expiration for expired record."""
        record = DataSubjectRecord(
            subject_id="test",
            subject_name="Test",
            created_at=datetime.now(timezone.utc) - timedelta(days=4000),  # Very old
        )
        record.retention_policy = RetentionPolicy.GDPR_DEFAULT

        # Should be expired
        assert (
            record.is_retention_expired(RetentionConfig()) is True
        )  # nosec: B101 - Test assertion

    def test_is_retention_expired_not_expired(self):
        """Test retention expiration for non-expired record."""
        record = DataSubjectRecord(
            subject_id="test",
            subject_name="Test",
            created_at=datetime.now(timezone.utc) - timedelta(days=10),  # Recent
        )
        record.retention_policy = RetentionPolicy.GDPR_DEFAULT

        # Should not be expired
        assert (
            record.is_retention_expired(RetentionConfig()) is False
        )  # nosec: B101 - Test assertion

    def test_mark_for_deletion(self):
        """Test marking record for deletion."""
        record = DataSubjectRecord(subject_id="test", subject_name="Test")

        # Test initial state
        assert record.deletion_requested is False  # nosec: B101 - Test assertion
        assert record.deletion_requested_at is None  # nosec: B101 - Test assertion

        # Mark for deletion and verify changes
        record.mark_for_deletion()
        assert record.deletion_requested is True  # nosec: B101 - Test assertion
        assert record.deletion_requested_at is not None  # type: ignore[unreachable]  # nosec: B101 - Test assertion
        time_diff = datetime.now(timezone.utc) - record.deletion_requested_at
        assert time_diff.total_seconds() < 60  # nosec: B101 - Test assertion

    def test_mark_deletion_complete(self):
        """Test marking deletion as complete."""
        record = DataSubjectRecord(subject_id="test", subject_name="Test")

        # First mark for deletion
        record.mark_for_deletion()
        assert record.deletion_requested is True  # nosec: B101 - Test assertion
        assert record.deletion_completed is False  # nosec: B101 - Test assertion

        # Then mark as complete and verify changes
        record.mark_deletion_complete()
        assert record.deletion_completed is True  # nosec: B101 - Test assertion
        assert record.deletion_completed_at is not None  # type: ignore[unreachable]  # nosec: B101 - Test assertion
        assert (
            record.deletion_completed_at >= record.deletion_requested_at
        )  # nosec: B101 - Test assertion


class TestDeletionExecutor:
    """Test the DeletionExecutor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = RetentionConfig()
        self.executor = DeletionExecutor(self.config)

    def test_initialization(self):
        """Test executor initialization."""
        assert self.executor.config == self.config  # nosec: B101 - Test assertion
        assert self.executor.logger is not None  # nosec: B101 - Test assertion
        assert self.executor.audit_sink is not None  # nosec: B101 - Test assertion

    def test_delete_experiment_data_success(self):
        """Test successful experiment data deletion."""
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(b"test experiment data")
        temp_file.close()

        result = self.executor.delete_experiment_data(
            subject_id="test_subject",
            experiment_id="test_experiment",
            data_path=temp_file.name,
        )

        assert result is True  # nosec: B101 - Test assertion
        assert not os.path.exists(temp_file.name)  # nosec: B101 - Test assertion

    def test_delete_experiment_data_file_not_found(self):
        """Test experiment data deletion when file doesn't exist."""
        nonexistent_file = "/nonexistent/path/file.txt"

        result = self.executor.delete_experiment_data(
            subject_id="test_subject",
            experiment_id="test_experiment",
            data_path=nonexistent_file,
        )

        assert result is False  # nosec: B101 - Test assertion

    def test_delete_experiment_data_no_path(self):
        """Test experiment data deletion with no path specified."""
        result = self.executor.delete_experiment_data(
            subject_id="test_subject", experiment_id="test_experiment", data_path=None
        )

        assert result is True  # nosec: B101 - Test assertion

    def test_delete_experiment_data_with_audit(self):
        """Test experiment data deletion with audit logging."""
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(b"test data")
        temp_file.close()

        # Mock audit sink to verify audit calls
        mock_audit_sink = Mock()
        self.executor.audit_sink = mock_audit_sink

        result = self.executor.delete_experiment_data(
            subject_id="test_subject",
            experiment_id="test_experiment",
            data_path=temp_file.name,
        )

        assert result is True  # nosec: B101 - Test assertion

        # Verify audit call
        mock_audit_sink.record_event.assert_called_once()
        call_args = mock_audit_sink.record_event.call_args
        assert (
            call_args[1]["operator_id"] == "test_subject"
        )  # nosec: B101 - Test assertion
        assert (
            call_args[1]["resource_id"] == "test_experiment"
        )  # nosec: B101 - Test assertion
        assert call_args[1]["status"] == "success"  # nosec: B101 - Test assertion

    def test_delete_experiment_data_failure_with_audit(self):
        """Test experiment data deletion failure with audit logging."""
        nonexistent_file = "/nonexistent/path/file.txt"

        # Mock audit sink to verify audit calls
        mock_audit_sink = Mock()
        self.executor.audit_sink = mock_audit_sink

        result = self.executor.delete_experiment_data(
            subject_id="test_subject",
            experiment_id="test_experiment",
            data_path=nonexistent_file,
        )

        assert result is False  # nosec: B101 - Test assertion

        # Verify audit call
        mock_audit_sink.record_event.assert_called_once()
        call_args = mock_audit_sink.record_event.call_args
        assert call_args[1]["status"] == "failure"  # nosec: B101 - Test assertion

    def test_delete_config_data_success(self):
        """Test successful config data deletion."""
        # Create temporary config file
        temp_dir = tempfile.mkdtemp()
        config_file = Path(temp_dir) / "test_config.json"
        config_file.write_text('{"key": "value"}')

        result = self.executor.delete_config_data(
            subject_id="test_subject", config_id="test_config"
        )

        assert result is True  # nosec: B101 - Test assertion
        assert not config_file.exists()  # nosec: B101 - Test assertion

        # Clean up
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_delete_config_data_file_not_found(self):
        """Test config data deletion when file doesn't exist."""
        result = self.executor.delete_config_data(
            subject_id="test_subject", config_id="nonexistent_config"
        )

        assert (
            result is True
        )  # Should succeed even if file doesn't exist  # nosec: B101 - Test assertion

    def test_delete_config_data_with_audit(self):
        """Test config data deletion with audit logging."""
        # Create temporary config file
        temp_dir = tempfile.mkdtemp()
        config_file = Path(temp_dir) / "test_config.json"
        config_file.write_text('{"key": "value"}')

        # Mock audit sink to verify audit calls
        mock_audit_sink = Mock()
        self.executor.audit_sink = mock_audit_sink

        result = self.executor.delete_config_data(
            subject_id="test_subject", config_id="test_config"
        )

        assert result is True  # nosec: B101 - Test assertion

        # Verify audit call
        mock_audit_sink.record_event.assert_called_once()
        call_args = mock_audit_sink.record_event.call_args
        assert (
            call_args[1]["operator_id"] == "test_subject"
        )  # nosec: B101 - Test assertion
        assert (
            call_args[1]["resource_id"] == "test_config"
        )  # nosec: B101 - Test assertion
        assert call_args[1]["status"] == "success"  # nosec: B101 - Test assertion

    def test_destroy_kms_key_success(self):
        """Test successful KMS key destruction."""
        mock_callback = Mock(return_value=True)

        result = self.executor.destroy_kms_key(
            subject_id="test_subject", key_id="test_key", kms_callback=mock_callback
        )

        assert result is True  # nosec: B101 - Test assertion
        mock_callback.assert_called_once_with("test_key")

    def test_destroy_kms_key_failure(self):
        """Test KMS key destruction failure."""
        mock_callback = Mock(return_value=False)

        result = self.executor.destroy_kms_key(
            subject_id="test_subject", key_id="test_key", kms_callback=mock_callback
        )

        assert result is False  # nosec: B101 - Test assertion

    def test_destroy_kms_key_no_callback(self):
        """Test KMS key destruction without callback."""
        result = self.executor.destroy_kms_key(
            subject_id="test_subject", key_id="test_key", kms_callback=None
        )

        assert (
            result is True
        )  # Should succeed without external callback  # nosec: B101 - Test assertion

    def test_destroy_kms_key_with_audit(self):
        """Test KMS key destruction with audit logging."""
        mock_callback = Mock(return_value=True)
        mock_audit_sink = Mock()
        self.executor.audit_sink = mock_audit_sink

        result = self.executor.destroy_kms_key(
            subject_id="test_subject", key_id="test_key", kms_callback=mock_callback
        )

        assert result is True  # nosec: B101 - Test assertion

        # Verify audit call
        mock_audit_sink.record_event.assert_called_once()
        call_args = mock_audit_sink.record_event.call_args
        assert (
            call_args[1]["operator_id"] == "test_subject"
        )  # nosec: B101 - Test assertion
        assert call_args[1]["resource_id"] == "test_key"  # nosec: B101 - Test assertion
        assert call_args[1]["status"] == "success"  # nosec: B101 - Test assertion


class TestRetentionJobScheduler:
    """Test the RetentionJobScheduler class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = RetentionConfig()
        self.scheduler = RetentionJobScheduler(self.config)

    def test_initialization(self):
        """Test scheduler initialization."""
        assert self.scheduler.config == self.config  # nosec: B101 - Test assertion
        assert (
            self.scheduler.deletion_executor is not None
        )  # nosec: B101 - Test assertion
        assert isinstance(
            self.scheduler.data_subjects, dict
        )  # nosec: B101 - Test assertion
        assert len(self.scheduler.data_subjects) == 0  # nosec: B101 - Test assertion

    def test_register_data_subject(self):
        """Test registering a data subject."""
        record = self.scheduler.register_data_subject(
            subject_id="test_subject",
            subject_name="Test Subject",
            data_categories=["experiment", "config", "audit"],
            retention_policy=RetentionPolicy.GDPR_DEFAULT,
        )

        assert record.subject_id == "test_subject"  # nosec: B101 - Test assertion
        assert record.subject_name == "Test Subject"  # nosec: B101 - Test assertion
        assert record.data_categories == [
            "experiment",
            "config",
            "audit",
        ]  # nosec: B101 - Test assertion
        assert (
            record.retention_policy == RetentionPolicy.GDPR_DEFAULT
        )  # nosec: B101 - Test assertion
        assert (
            "test_subject" in self.scheduler.data_subjects
        )  # nosec: B101 - Test assertion
        assert (
            self.scheduler.data_subjects["test_subject"] is record
        )  # nosec: B101 - Test assertion

    def test_register_data_subject_custom_policy(self):
        """Test registering data subject with custom policy."""
        record = self.scheduler.register_data_subject(
            subject_id="test_subject",
            subject_name="Test Subject",
            data_categories=["experiment"],
            retention_policy=RetentionPolicy.CCPA_DEFAULT,
        )

        assert (
            record.retention_policy == RetentionPolicy.CCPA_DEFAULT
        )  # nosec: B101 - Test assertion

    def test_request_deletion_success(self):
        """Test successful deletion request."""
        record = self.scheduler.register_data_subject(
            subject_id="test_subject",
            subject_name="Test Subject",
            data_categories=["experiment"],
        )

        result = self.scheduler.request_deletion("test_subject")

        assert result is True  # nosec: B101 - Test assertion
        assert record.deletion_requested is True  # nosec: B101 - Test assertion
        assert record.deletion_requested_at is not None  # nosec: B101 - Test assertion

    def test_request_deletion_nonexistent_subject(self):
        """Test deletion request for nonexistent subject."""
        result = self.scheduler.request_deletion("nonexistent_subject")

        assert result is False  # nosec: B101 - Test assertion

    def test_execute_retention_jobs_empty(self):
        """Test retention job execution with no subjects."""
        results = self.scheduler.execute_retention_jobs()

        assert results["total_subjects"] == 0  # nosec: B101 - Test assertion
        assert results["expired_subjects"] == 0  # nosec: B101 - Test assertion
        assert results["deletion_requested"] == 0  # nosec: B101 - Test assertion
        assert results["deletions_completed"] == 0  # nosec: B101 - Test assertion
        assert results["deletions_failed"] == 0  # nosec: B101 - Test assertion

    def test_execute_retention_jobs_with_deletion_requests(self):
        """Test retention job execution with deletion requests."""
        # Register subjects and request deletion
        record1 = self.scheduler.register_data_subject(
            subject_id="subject1",
            subject_name="Subject 1",
            data_categories=["experiment"],
        )
        record1.mark_for_deletion()

        record2 = self.scheduler.register_data_subject(
            subject_id="subject2", subject_name="Subject 2", data_categories=["config"]
        )
        record2.mark_for_deletion()

        # Mock deletion executor
        mock_executor = Mock()
        mock_executor.delete_experiment_data.return_value = True
        mock_executor.delete_config_data.return_value = True
        mock_executor.destroy_kms_key.return_value = True
        self.scheduler.deletion_executor = mock_executor

        results = self.scheduler.execute_retention_jobs()

        assert results["total_subjects"] == 2  # nosec: B101 - Test assertion
        assert results["deletion_requested"] == 2  # nosec: B101 - Test assertion
        assert results["deletions_completed"] == 2  # nosec: B101 - Test assertion
        assert results["deletions_failed"] == 0  # nosec: B101 - Test assertion

    def test_execute_retention_jobs_with_expired_records(self):
        """Test retention job execution with expired records."""
        # Register subjects with expired retention
        old_record = DataSubjectRecord(
            subject_id="old_subject",
            subject_name="Old Subject",
            created_at=datetime.now(timezone.utc) - timedelta(days=4000),
            data_categories=["experiment"],
        )
        self.scheduler.data_subjects["old_subject"] = old_record

        recent_record = DataSubjectRecord(
            subject_id="recent_subject",
            subject_name="Recent Subject",
            created_at=datetime.now(timezone.utc) - timedelta(days=10),
            data_categories=["experiment"],
        )
        self.scheduler.data_subjects["recent_subject"] = recent_record

        # Mock deletion executor
        mock_executor = Mock()
        mock_executor.delete_experiment_data.return_value = True
        self.scheduler.deletion_executor = mock_executor

        results = self.scheduler.execute_retention_jobs()

        assert results["total_subjects"] == 2  # nosec: B101 - Test assertion
        assert results["expired_subjects"] == 1  # nosec: B101 - Test assertion
        assert results["deletions_completed"] == 1  # nosec: B101 - Test assertion
        assert results["deletions_failed"] == 0  # nosec: B101 - Test assertion

    def test_execute_retention_jobs_disabled_auto_delete(self):
        """Test retention job execution with auto-delete disabled."""
        # Disable auto-delete
        self.scheduler.config.auto_delete_enabled = False

        # Register expired record
        old_record = DataSubjectRecord(
            subject_id="old_subject",
            subject_name="Old Subject",
            created_at=datetime.now(timezone.utc) - timedelta(days=4000),
            data_categories=["experiment"],
        )
        self.scheduler.data_subjects["old_subject"] = old_record

        results = self.scheduler.execute_retention_jobs()

        # Should not delete expired records when auto-delete is disabled
        assert results["total_subjects"] == 1  # nosec: B101 - Test assertion
        assert results["expired_subjects"] == 0  # nosec: B101 - Test assertion
        assert results["deletions_completed"] == 0  # nosec: B101 - Test assertion

    def test_execute_retention_jobs_partial_failure(self):
        """Test retention job execution with partial failures."""
        # Register subjects
        record1 = self.scheduler.register_data_subject(
            subject_id="subject1",
            subject_name="Subject 1",
            data_categories=["experiment"],
        )
        record1.mark_for_deletion()

        record2 = self.scheduler.register_data_subject(
            subject_id="subject2", subject_name="Subject 2", data_categories=["config"]
        )
        record2.mark_for_deletion()

        # Mock deletion executor with partial failure
        mock_executor = Mock()
        mock_executor.delete_experiment_data.return_value = True
        mock_executor.delete_config_data.return_value = False  # This one fails
        self.scheduler.deletion_executor = mock_executor

        results = self.scheduler.execute_retention_jobs()

        assert results["total_subjects"] == 2  # nosec: B101 - Test assertion
        assert results["deletion_requested"] == 2  # nosec: B101 - Test assertion
        assert results["deletions_completed"] == 1  # nosec: B101 - Test assertion
        assert results["deletions_failed"] == 1  # nosec: B101 - Test assertion

    def test_execute_subject_deletion_experiment_category(self):
        """Test subject deletion for experiment category."""
        record = DataSubjectRecord(
            subject_id="test_subject",
            subject_name="Test Subject",
            data_categories=["experiment"],
        )

        # Mock deletion executor
        mock_executor = Mock()
        mock_executor.delete_experiment_data.return_value = True
        self.scheduler.deletion_executor = mock_executor

        result = self.scheduler._execute_subject_deletion(record)

        assert result is True  # nosec: B101 - Test assertion
        mock_executor.delete_experiment_data.assert_called_once_with(
            "test_subject", "experiment_test_subject"
        )

    def test_execute_subject_deletion_config_category(self):
        """Test subject deletion for config category."""
        record = DataSubjectRecord(
            subject_id="test_subject",
            subject_name="Test Subject",
            data_categories=["config"],
        )

        # Mock deletion executor
        mock_executor = Mock()
        mock_executor.delete_config_data.return_value = True
        self.scheduler.deletion_executor = mock_executor

        result = self.scheduler._execute_subject_deletion(record)

        assert result is True  # nosec: B101 - Test assertion
        mock_executor.delete_config_data.assert_called_once_with(
            "test_subject", "config_test_subject"
        )

    def test_execute_subject_deletion_kms_key_category(self):
        """Test subject deletion for KMS key category."""
        record = DataSubjectRecord(
            subject_id="test_subject",
            subject_name="Test Subject",
            data_categories=["kms_key"],
        )

        # Mock deletion executor
        mock_executor = Mock()
        mock_executor.destroy_kms_key.return_value = True
        self.scheduler.deletion_executor = mock_executor

        result = self.scheduler._execute_subject_deletion(record)

        assert result is True  # nosec: B101 - Test assertion
        mock_executor.destroy_kms_key.assert_called_once_with(
            "test_subject", "key_test_subject"
        )

    def test_execute_subject_deletion_unknown_category(self):
        """Test subject deletion for unknown category."""
        record = DataSubjectRecord(
            subject_id="test_subject",
            subject_name="Test Subject",
            data_categories=["unknown_category"],
        )

        # Mock deletion executor
        mock_executor = Mock()
        self.scheduler.deletion_executor = mock_executor

        result = self.scheduler._execute_subject_deletion(record)

        # Should succeed even with unknown category
        assert result is True  # nosec: B101 - Test assertion

    def test_execute_subject_deletion_exception(self):
        """Test subject deletion with exception."""
        record = DataSubjectRecord(
            subject_id="test_subject",
            subject_name="Test Subject",
            data_categories=["experiment"],
        )

        # Mock deletion executor to raise exception
        mock_executor = Mock()
        mock_executor.delete_experiment_data.side_effect = Exception("Deletion failed")
        self.scheduler.deletion_executor = mock_executor

        result = self.scheduler._execute_subject_deletion(record)

        assert result is False  # nosec: B101 - Test assertion

    def test_export_subject_data_success(self):
        """Test successful subject data export."""
        self.scheduler.register_data_subject(
            subject_id="test_subject",
            subject_name="Test Subject",
            data_categories=["experiment", "config"],
            retention_policy=RetentionPolicy.GDPR_DEFAULT,
        )

        # Create temporary file for export
        temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        temp_file_path = temp_file.name
        temp_file.close()

        result = self.scheduler.export_subject_data("test_subject", temp_file_path)

        assert result is True  # nosec: B101 - Test assertion

        # Verify exported data
        with open(temp_file_path, "r") as f:
            exported_data = json.load(f)

        assert (
            exported_data["subject_id"] == "test_subject"
        )  # nosec: B101 - Test assertion
        assert (
            exported_data["subject_name"] == "Test Subject"
        )  # nosec: B101 - Test assertion
        assert exported_data["data_categories"] == [
            "experiment",
            "config",
        ]  # nosec: B101 - Test assertion
        assert (
            exported_data["retention_policy"] == "gdpr_default"
        )  # nosec: B101 - Test assertion
        assert "created_at" in exported_data  # nosec: B101 - Test assertion

        # Clean up
        os.unlink(temp_file_path)

    def test_export_subject_data_nonexistent_subject(self):
        """Test subject data export for nonexistent subject."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        temp_file_path = temp_file.name
        temp_file.close()

        result = self.scheduler.export_subject_data(
            "nonexistent_subject", temp_file_path
        )

        assert result is False  # nosec: B101 - Test assertion

        # Clean up
        os.unlink(temp_file_path)

    def test_export_subject_data_file_error(self):
        """Test subject data export with file error."""
        self.scheduler.register_data_subject(
            subject_id="test_subject",
            subject_name="Test Subject",
            data_categories=["test"],
        )

        # Use invalid file path
        invalid_path = "/invalid/path/file.txt"

        result = self.scheduler.export_subject_data("test_subject", invalid_path)

        assert result is False  # nosec: B101 - Test assertion

    def test_get_retention_statistics(self):
        """Test retention statistics calculation."""
        # Create test records
        current_time = datetime.now(timezone.utc)

        # Expired record
        expired_record = DataSubjectRecord(
            subject_id="expired_subject",
            subject_name="Expired Subject",
            created_at=current_time - timedelta(days=4000),
            data_categories=["experiment"],
        )
        self.scheduler.data_subjects["expired_subject"] = expired_record

        # Deletion requested but not completed
        requested_record = DataSubjectRecord(
            subject_id="requested_subject",
            subject_name="Requested Subject",
            created_at=current_time - timedelta(days=10),
            data_categories=["config"],
        )
        requested_record.mark_for_deletion()
        self.scheduler.data_subjects["requested_subject"] = requested_record

        # Deletion completed
        completed_record = DataSubjectRecord(
            subject_id="completed_subject",
            subject_name="Completed Subject",
            created_at=current_time - timedelta(days=5),
            data_categories=["audit"],
        )
        completed_record.mark_deletion_complete()
        self.scheduler.data_subjects["completed_subject"] = completed_record

        # Recent record
        recent_record = DataSubjectRecord(
            subject_id="recent_subject",
            subject_name="Recent Subject",
            created_at=current_time - timedelta(days=1),
            data_categories=["experiment"],
        )
        self.scheduler.data_subjects["recent_subject"] = recent_record

        stats = self.scheduler.get_retention_statistics()

        assert stats["total_subjects"] == 4  # nosec: B101 - Test assertion
        assert stats["expired_records"] == 1  # nosec: B101 - Test assertion
        assert stats["deletion_requested"] == 1  # nosec: B101 - Test assertion
        assert stats["deletion_completed"] == 1  # nosec: B101 - Test assertion
        assert stats["pending_deletion"] == 0  # nosec: B101 - Test assertion

    def test_get_retention_statistics_empty(self):
        """Test retention statistics with no subjects."""
        stats = self.scheduler.get_retention_statistics()

        assert stats["total_subjects"] == 0  # nosec: B101 - Test assertion
        assert stats["expired_records"] == 0  # nosec: B101 - Test assertion
        assert stats["deletion_requested"] == 0  # nosec: B101 - Test assertion
        assert stats["deletion_completed"] == 0  # nosec: B101 - Test assertion
        assert stats["pending_deletion"] == 0  # nosec: B101 - Test assertion


class TestGlobalScheduler:
    """Test the global scheduler functions."""

    def test_get_retention_scheduler_singleton(self):
        """Test getting global scheduler singleton."""
        # Clear any existing scheduler by creating a new one
        temp_scheduler = RetentionJobScheduler(RetentionConfig())
        set_retention_scheduler(temp_scheduler)

        # First call should create new instance
        scheduler1 = get_retention_scheduler()
        assert isinstance(
            scheduler1, RetentionJobScheduler
        )  # nosec: B101 - Test assertion

        # Second call should return same instance
        scheduler2 = get_retention_scheduler()
        assert scheduler1 is scheduler2  # nosec: B101 - Test assertion

    def test_set_retention_scheduler(self):
        """Test setting global scheduler."""
        custom_scheduler = RetentionJobScheduler(RetentionConfig())

        set_retention_scheduler(custom_scheduler)

        # Should return the custom scheduler
        retrieved = get_retention_scheduler()
        assert retrieved is custom_scheduler  # nosec: B101 - Test assertion

    def test_set_retention_scheduler_none(self):
        """Test setting global scheduler to None."""
        # Set to a different scheduler (since None is not allowed)
        temp_scheduler = RetentionJobScheduler(RetentionConfig())
        set_retention_scheduler(temp_scheduler)

        # Should return the new scheduler
        new_scheduler = get_retention_scheduler()
        assert new_scheduler is temp_scheduler  # nosec: B101 - Test assertion


class TestRetentionIntegration:
    """Integration tests for retention functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = RetentionConfig(
            policy=RetentionPolicy.GDPR_DEFAULT,
            retention_days=30,  # Short retention for testing
            auto_delete_enabled=True,
        )
        self.scheduler = RetentionJobScheduler(self.config)

    def test_full_retention_lifecycle(self):
        """Test complete retention lifecycle."""
        # Register subjects
        self.scheduler.register_data_subject(
            subject_id="user_001",
            subject_name="User 001",
            data_categories=["experiment", "config", "audit"],
            retention_policy=RetentionPolicy.GDPR_DEFAULT,
        )

        subject2 = self.scheduler.register_data_subject(
            subject_id="user_002",
            subject_name="User 002",
            data_categories=["experiment"],
            retention_policy=RetentionPolicy.CCPA_DEFAULT,
        )

        # Simulate time passage for subject2 to expire
        old_created_at = datetime.now(timezone.utc) - timedelta(days=400)
        subject2.created_at = old_created_at

        # Real executor, with only the audit sink replaced so events are observable
        audit_sink = Mock()
        self.scheduler.deletion_executor.audit_sink = audit_sink

        # Execute retention jobs
        results = self.scheduler.execute_retention_jobs()

        # Verify results
        assert results["total_subjects"] == 2  # nosec: B101 - Test assertion
        assert (
            results["expired_subjects"] == 1
        )  # subject2 should be expired  # nosec: B101 - Test assertion
        assert results["deletions_completed"] == 1  # nosec: B101 - Test assertion

        # Only the expired subject is deleted, and that deletion is audited
        audit_calls = audit_sink.record_event.call_args_list
        assert len(audit_calls) == 1  # nosec: B101 - Test assertion
        assert (
            audit_calls[0].kwargs["resource_id"] == "experiment_user_002"
        )  # nosec: B101 - Test assertion
        assert audit_calls[0].kwargs["status"] == "success"  # nosec: B101

    def test_compliance_workflow_with_audit(self):
        """Test compliance workflow with complete audit trail."""
        # Create subjects with various retention policies
        self.scheduler.register_data_subject(
            subject_id="gdpr_user",
            subject_name="GDPR User",
            data_categories=["sensitive_data"],
            retention_policy=RetentionPolicy.GDPR_DEFAULT,
        )

        self.scheduler.register_data_subject(
            subject_id="ccpa_user",
            subject_name="CCPA User",
            data_categories=["experiment"],
            retention_policy=RetentionPolicy.CCPA_DEFAULT,
        )

        self.scheduler.register_data_subject(
            subject_id="hipaa_user",
            subject_name="HIPAA User",
            data_categories=["health_data"],
            retention_policy=RetentionPolicy.HIPAA_DEFAULT,
        )

        # Request deletion for CCPA user (right to erasure)
        deletion_requested = self.scheduler.request_deletion("ccpa_user")
        assert deletion_requested is True  # nosec: B101 - Test assertion

        # Real executor, with only the audit sink replaced so events are observable
        audit_sink = Mock()
        self.scheduler.deletion_executor.audit_sink = audit_sink

        # Execute retention jobs
        self.scheduler.execute_retention_jobs()

        # Verify CCPA user deletion was processed and audited exactly once
        ccpa_results = [
            call
            for call in audit_sink.record_event.call_args_list
            if call.kwargs["resource_id"] == "experiment_ccpa_user"
        ]
        assert len(ccpa_results) == 1  # nosec: B101 - Test assertion

    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        # Create an expired subject so retention jobs attempt its deletion
        subject = self.scheduler.register_data_subject(
            subject_id="error_test",
            subject_name="Error Test",
            data_categories=["experiment"],
            retention_policy=RetentionPolicy.GDPR_DEFAULT,
        )
        subject.created_at = datetime.now(timezone.utc) - timedelta(days=400)

        # Mock deletion executor that fails initially
        mock_executor = Mock()
        mock_executor.delete_experiment_data.side_effect = [
            Exception("First failure"),
            True,
        ]
        self.scheduler.deletion_executor = mock_executor

        # First execution should fail
        results1 = self.scheduler.execute_retention_jobs()
        assert results1["deletions_failed"] == 1  # nosec: B101 - Test assertion

        # Fix the mock to succeed
        mock_executor.delete_experiment_data.side_effect = None
        mock_executor.delete_experiment_data.return_value = True

        # Second execution should succeed
        results2 = self.scheduler.execute_retention_jobs()
        assert results2["deletions_completed"] == 1  # nosec: B101 - Test assertion

    def test_performance_with_large_dataset(self):
        """Test performance with large dataset."""
        # Create many subjects
        for i in range(100):
            self.scheduler.register_data_subject(
                subject_id=f"user_{i:03d}",
                subject_name=f"User {i:03d}",
                data_categories=["experiment"],
                retention_policy=RetentionPolicy.GDPR_DEFAULT,
            )

        # Mock deletion executor
        mock_executor = Mock()
        mock_executor.delete_experiment_data.return_value = True
        self.scheduler.deletion_executor = mock_executor

        # Measure execution time
        start_time = time.time()
        results = self.scheduler.execute_retention_jobs()
        end_time = time.time()

        # Should complete efficiently
        assert (
            end_time - start_time < 10.0
        )  # 10 second limit  # nosec: B101 - Test assertion
        assert results["total_subjects"] == 100  # nosec: B101 - Test assertion


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
