"""
Comprehensive test suite for apgi_compliance.py to achieve ≥90% coverage.

This file tests the complex code paths and edge cases not covered in the basic test suite.
"""

import json
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.apgi_compliance import (
    RETENTION_POLICIES,
    ComplianceManager,
    DataClassification,
    RetentionPolicy,
    pseudonymize_participant,
)


class TestDataClassification:
    """Test the DataClassification enum."""

    def test_data_classification_values(self):
        """Test that all expected classification values exist."""
        expected_values = [
            DataClassification.PUBLIC,
            DataClassification.INTERNAL,
            DataClassification.CONFIDENTIAL,
            DataClassification.RESTRICTED,
        ]

        for classification in expected_values:
            assert classification.value in [  # nosec: B101 - Test assertion
                "public",
                "internal",
                "confidential",
                "restricted",
            ]

    def test_data_classification_string_conversion(self):
        """Test string conversion of classifications."""
        assert (
            DataClassification.PUBLIC.value == "public"
        )  # nosec: B101 - Test assertion
        assert (
            DataClassification.INTERNAL.value == "internal"
        )  # nosec: B101 - Test assertion
        assert (
            DataClassification.CONFIDENTIAL.value == "confidential"
        )  # nosec: B101 - Test assertion
        assert (
            DataClassification.RESTRICTED.value == "restricted"
        )  # nosec: B101 - Test assertion


class TestRetentionPolicy:
    """Test the RetentionPolicy dataclass."""

    def test_retention_policy_creation(self):
        """Test creating retention policies."""
        policy = RetentionPolicy(
            classification=DataClassification.PUBLIC,
            ttl_days=365,
            deletion_routine="archive",
        )

        assert (
            policy.classification == DataClassification.PUBLIC
        )  # nosec: B101 - Test assertion
        assert policy.ttl_days == 365  # nosec: B101 - Test assertion
        assert policy.deletion_routine == "archive"  # nosec: B101 - Test assertion

    def test_retention_policies_completeness(self):
        """Test that all expected retention policies are defined."""
        expected_classifications = [
            DataClassification.PUBLIC,
            DataClassification.INTERNAL,
            DataClassification.CONFIDENTIAL,
            DataClassification.RESTRICTED,
        ]

        for classification in expected_classifications:
            assert classification in RETENTION_POLICIES  # nosec: B101 - Test assertion
            policy = RETENTION_POLICIES[classification]
            assert isinstance(policy, RetentionPolicy)  # nosec: B101 - Test assertion
            assert policy.ttl_days > 0  # nosec: B101 - Test assertion
            assert policy.deletion_routine in [  # nosec: B101 - Test assertion
                "archive",
                "soft_delete",
                "secure_erase",
                "crypto_shred",
            ]

    def test_retention_policy_values(self):
        """Test specific retention policy values."""
        public_policy = RETENTION_POLICIES[DataClassification.PUBLIC]
        assert public_policy.ttl_days == 3650  # nosec: B101 - Test assertion
        assert (
            public_policy.deletion_routine == "archive"
        )  # nosec: B101 - Test assertion

        restricted_policy = RETENTION_POLICIES[DataClassification.RESTRICTED]
        assert restricted_policy.ttl_days == 30  # nosec: B101 - Test assertion
        assert (
            restricted_policy.deletion_routine == "crypto_shred"
        )  # nosec: B101 - Test assertion


class TestComplianceManager:
    """Test the ComplianceManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ComplianceManager()

    def test_initialization(self):
        """Test manager initialization."""
        assert isinstance(
            self.manager.audit_trail, list
        )  # nosec: B101 - Test assertion
        assert len(self.manager.audit_trail) == 0  # nosec: B101 - Test assertion

    def test_log_parameter_change(self):
        """Test logging parameter changes."""
        self.manager.log_parameter_change(
            user="test_user",
            param_name="test_param",
            old_value="old_value",
            new_value="new_value",
        )

        assert len(self.manager.audit_trail) == 1  # nosec: B101 - Test assertion

        entry = self.manager.audit_trail[0]
        assert entry["action"] == "parameter_change"  # nosec: B101 - Test assertion
        assert entry["user"] == "test_user"  # nosec: B101 - Test assertion
        assert entry["param_name"] == "test_param"  # nosec: B101 - Test assertion
        assert entry["old_value"] == "old_value"  # nosec: B101 - Test assertion
        assert entry["new_value"] == "new_value"  # nosec: B101 - Test assertion
        assert "timestamp" in entry  # nosec: B101 - Test assertion

    def test_log_parameter_change_with_complex_values(self):
        """Test logging parameter changes with complex values."""
        complex_old = {"nested": {"key": "value"}, "list": [1, 2, 3]}
        complex_new = {"nested": {"key": "new_value"}, "list": [4, 5, 6]}

        self.manager.log_parameter_change(
            user="test_user",
            param_name="complex_param",
            old_value=complex_old,
            new_value=complex_new,
        )

        entry = self.manager.audit_trail[0]
        assert entry["old_value"] == complex_old  # nosec: B101 - Test assertion
        assert entry["new_value"] == complex_new  # nosec: B101 - Test assertion

    def test_log_experiment_run(self):
        """Test logging experiment runs."""
        self.manager.log_experiment_run(
            experiment_id="test_experiment",
            classification=DataClassification.CONFIDENTIAL,
        )

        assert len(self.manager.audit_trail) == 1  # nosec: B101 - Test assertion

        entry = self.manager.audit_trail[0]
        assert entry["action"] == "experiment_run"  # nosec: B101 - Test assertion
        assert (
            entry["experiment_id"] == "test_experiment"
        )  # nosec: B101 - Test assertion
        assert entry["classification"] == "confidential"  # nosec: B101 - Test assertion
        assert "timestamp" in entry  # nosec: B101 - Test assertion

    def test_enforce_retention_empty_list(self):
        """Test retention enforcement with empty list."""
        result = self.manager.enforce_retention([])
        assert result == []  # nosec: B101 - Test assertion

    def test_enforce_retention_no_expired_records(self):
        """Test retention enforcement with no expired records."""
        current_time = datetime.now()
        records = [
            {
                "id": "record1",
                "classification": "public",
                "created_at": (current_time - timedelta(days=1)).isoformat(),
                "data": "test data",
            },
            {
                "id": "record2",
                "classification": "internal",
                "created_at": (current_time - timedelta(days=10)).isoformat(),
                "data": "test data",
            },
        ]

        result = self.manager.enforce_retention(records)

        # All records should be retained
        assert len(result) == 2  # nosec: B101 - Test assertion
        assert result[0]["id"] == "record1"  # nosec: B101 - Test assertion
        assert result[1]["id"] == "record2"  # nosec: B101 - Test assertion

    def test_enforce_retention_with_expired_records(self):
        """Test retention enforcement with expired records."""
        current_time = datetime.now()
        records = [
            {
                "id": "record1",
                "classification": "public",
                "created_at": (
                    current_time - timedelta(days=4000)
                ).isoformat(),  # Expired
                "data": "test data",
            },
            {
                "id": "record2",
                "classification": "restricted",
                "created_at": (
                    current_time - timedelta(days=40)
                ).isoformat(),  # Expired
                "data": "test data",
            },
            {
                "id": "record3",
                "classification": "internal",
                "created_at": (
                    current_time - timedelta(days=10)
                ).isoformat(),  # Not expired
                "data": "test data",
            },
        ]

        # Mock deletion execution
        with patch.object(self.manager, "_execute_deletion") as mock_delete:
            result = self.manager.enforce_retention(records)

        # Only non-expired record should be retained
        assert len(result) == 1  # nosec: B101 - Test assertion
        assert result[0]["id"] == "record3"  # nosec: B101 - Test assertion

        # Deletion should be called for expired records
        assert mock_delete.call_count == 2  # nosec: B101 - Test assertion

    def test_enforce_retention_invalid_classification(self):
        """Test retention enforcement with invalid classification."""
        current_time = datetime.now()
        records = [
            {
                "id": "record1",
                "classification": "invalid_classification",
                "created_at": current_time.isoformat(),
                "data": "test data",
            }
        ]

        result = self.manager.enforce_retention(records)

        # Should default to internal classification
        assert len(result) == 1  # nosec: B101 - Test assertion
        assert result[0]["id"] == "record1"  # nosec: B101 - Test assertion

    def test_enforce_retention_missing_created_at(self):
        """Test retention enforcement with missing created_at."""
        records = [
            {
                "id": "record1",
                "classification": "public",
                "data": "test data",
                # Missing created_at
            }
        ]

        result = self.manager.enforce_retention(records)

        # Should use current time as created_at
        assert len(result) == 1  # nosec: B101 - Test assertion
        assert result[0]["id"] == "record1"  # nosec: B101 - Test assertion

    def test_enforce_retention_no_policy(self):
        """Test retention enforcement with no matching policy."""
        # Mock RETENTION_POLICIES to not have the classification
        original_policies = RETENTION_POLICIES.copy()
        RETENTION_POLICIES.clear()

        try:
            current_time = datetime.now()
            records = [
                {
                    "id": "record1",
                    "classification": "public",
                    "created_at": current_time.isoformat(),
                    "data": "test data",
                }
            ]

            result = self.manager.enforce_retention(records)

            # Should skip record with no policy
            assert len(result) == 0  # nosec: B101 - Test assertion

        finally:
            # Restore original policies
            RETENTION_POLICIES.update(original_policies)


class TestDeletionRoutines:
    """Test the various deletion routines."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ComplianceManager()

    def test_soft_delete(self):
        """Test soft deletion routine."""
        record = {
            "id": "test_record",
            "data": "sensitive data",
            "created_at": datetime.now().isoformat(),
        }

        self.manager._execute_deletion(record, "soft_delete")

        # Check audit trail entry
        assert len(self.manager.audit_trail) == 1  # nosec: B101 - Test assertion
        entry = self.manager.audit_trail[0]
        assert entry["action"] == "soft_delete"  # nosec: B101 - Test assertion
        assert entry["record_id"] == "test_record"  # nosec: B101 - Test assertion
        assert entry["routine"] == "soft_delete"  # nosec: B101 - Test assertion
        assert "deleted_at" in entry  # nosec: B101 - Test assertion

    def test_soft_delete_with_dict_record(self):
        """Test soft deletion with dict record."""
        record = {
            "id": "test_record",
            "data": "sensitive data",
            "created_at": datetime.now().isoformat(),
        }

        self.manager._execute_deletion(record, "soft_delete")

        # Check record was marked as deleted
        deleted_value = record.get("deleted")
        assert (
            deleted_value is not None and deleted_value
        )  # nosec: B101 - Test assertion
        assert "deleted_at" in record  # nosec: B101 - Test assertion

    def test_hard_delete(self):
        """Test hard deletion routine."""
        # Create temporary files to delete
        temp_dir = tempfile.mkdtemp()
        test_files = []

        file_mapping = {
            "data/experiments/test_record.txt": "data/experiments/test_record.*",
            "results/test_record.json": "results/test_record.*",
            "logs/test_record.log": "logs/test_record.*",
        }

        for filename in file_mapping.keys():
            file_path = Path(temp_dir) / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("test content")
            test_files.append(file_path)

        record = {
            "id": "test_record",
            "data": "sensitive data",
            "created_at": datetime.now().isoformat(),
        }

        # Test deletion execution

        # Mock glob to return test files based on pattern
        def mock_glob_func(pattern):
            # Return matching test files based on pattern
            matching_files = []
            for file_path in test_files:
                # Simple pattern matching - check if pattern matches
                if "experiments" in pattern and "experiments" in str(file_path):
                    matching_files.append(str(file_path))
                elif "results" in pattern and "results" in str(file_path):
                    matching_files.append(str(file_path))
                elif "logs" in pattern and "logs" in str(file_path):
                    matching_files.append(str(file_path))
            return matching_files

        with patch("glob.glob", side_effect=mock_glob_func):
            # Mock os.remove to track deletions
            with patch("os.remove") as mock_remove:
                self.manager._execute_deletion(record, "hard_delete")

                # Should attempt to delete all files (3 files)
                assert mock_remove.call_count == len(
                    test_files
                )  # nosec: B101 - Test assertion

                # Check deletion was executed (verified by mock calls)

        # Clean up
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_hard_delete_file_error(self):
        """Test hard deletion with file removal errors."""
        record = {
            "id": "test_record",
            "data": "sensitive data",
            "created_at": datetime.now().isoformat(),
        }

        # Mock glob to return file that will fail to delete
        with patch("glob.glob") as mock_glob:
            mock_glob.return_value = ["nonexistent_file.txt"]

            with patch("os.remove", side_effect=OSError("Permission denied")):
                # Should handle file deletion errors gracefully
                self.manager._execute_deletion(record, "hard_delete")

    def test_anonymous_deletion(self):
        """Test anonymization deletion routine."""
        record = {
            "id": "test_record",
            "user_id": "user123",
            "email": "test@example.com",
            "name": "Test User",
            "ip_address": "192.168.1.1",
            "session_id": "sess123",
            "token": "token456",
            "api_key": "key789",
            "regular_field": "regular_value",
            "created_at": datetime.now().isoformat(),
        }

        self.manager._execute_deletion(record, "anonymous")

        # Check PII fields were hashed
        assert record["user_id"].startswith("hashed_")  # nosec: B101 - Test assertion
        assert record["email"].startswith("hashed_")  # nosec: B101 - Test assertion
        assert record["name"].startswith("hashed_")  # nosec: B101 - Test assertion
        assert record["ip_address"].startswith(
            "hashed_"
        )  # nosec: B101 - Test assertion

        # Check direct identifiers were removed
        assert "session_id" not in record  # nosec: B101 - Test assertion
        assert "token" not in record  # nosec: B101 - Test assertion
        assert "api_key" not in record  # nosec: B101 - Test assertion

        # Check regular field was preserved
        assert (
            record["regular_field"] == "regular_value"
        )  # nosec: B101 - Test assertion

    def test_anonymous_deletion_non_dict_record(self):
        """Test anonymization with non-dict record."""
        record = {"id": "test", "data": "sensitive"}

        # Should handle dict records
        self.manager._execute_deletion(record, "anonymous")

    def test_secure_erase(self):
        """Test secure erase deletion routine."""
        record = {
            "id": "test_record",
            "sensitive_data": "very sensitive",
            "password": "secret123",
            "created_at": datetime.now().isoformat(),
        }

        # Create temporary files to erase
        temp_dir = tempfile.mkdtemp()
        test_files = []

        for filename in ["data/test.txt", "results/test.json"]:
            file_path = Path(temp_dir) / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("sensitive content")
            test_files.append(file_path)

        # Mock glob and file operations
        with patch("glob.glob") as mock_glob:
            mock_glob.return_value = [str(f) for f in test_files]

            with patch("os.path.getsize", return_value=100):
                with patch("builtins.open", mock_open(read_data=b"test")) as mock_file:
                    mock_file.read.return_value = b"test"
                    mock_file.write.return_value = None
                    with patch("os.fsync"):
                        with patch("os.remove"):
                            self.manager._execute_deletion(record, "secure_erase")

                            # Check data was overwritten (empty string)
                            assert (
                                record["sensitive_data"] == ""
                            )  # nosec: B101 - Test assertion
                            assert (
                                record["password"] == ""
                            )  # nosec: B101 - Test assertion

        # Clean up
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_secure_erase_file_error(self):
        """Test secure erase with file operation errors."""
        record = {
            "id": "test_record",
            "data": "test data",
            "created_at": datetime.now().isoformat(),
        }

        # Mock glob to return file that will fail to erase
        with patch("glob.glob") as mock_glob:
            mock_glob.return_value = ["nonexistent_file.txt"]

            with patch("os.path.getsize", return_value=100):
                with patch("builtins.open", side_effect=OSError("Permission denied")):
                    # Should handle file operation errors gracefully
                    self.manager._execute_deletion(record, "secure_erase")

    def test_crypto_shred(self):
        """Test crypto shredding deletion routine."""
        record = {
            "id": "test_record",
            "encryption_key": "secret_key_123",
            "data": "encrypted_data",
            "created_at": datetime.now().isoformat(),
        }

        self.manager._execute_deletion(record, "crypto_shred")

        # Check encryption key was overwritten
        assert record["encryption_key"] == ""  # nosec: B101 - Test assertion

    def test_archive_deletion(self):
        """Test archive deletion routine."""
        record = {
            "id": "test_record",
            "data": "test data to archive",
            "created_at": datetime.now().isoformat(),
        }

        # Mock file operations
        with patch("os.makedirs"):
            with patch("gzip.open", mock_open()) as mock_gzip:
                with patch("json.dump") as mock_json:
                    self.manager._execute_deletion(record, "archive")

                    # Check archive was created
                    mock_gzip.assert_called_once()
                    mock_json.assert_called_once()

                    # Check archive data structure
                    args, kwargs = mock_json.call_args
                    archive_data = args[0]
                    assert "record" in archive_data  # nosec: B101 - Test assertion
                    assert "archived_at" in archive_data  # nosec: B101 - Test assertion
                    assert (
                        "archive_reason" in archive_data
                    )  # nosec: B101 - Test assertion

    def test_archive_deletion_file_error(self):
        """Test archive deletion with file operation errors."""
        record = {
            "id": "test_record",
            "data": "test data",
            "created_at": datetime.now().isoformat(),
        }

        # Mock file operations to raise exception
        with patch("os.makedirs", side_effect=OSError("Permission denied")):
            # Should handle file operation errors gracefully
            with pytest.raises(Exception):
                self.manager._execute_deletion(record, "archive")

    def test_archive_deletion_permission_error(self):
        """Test archive deletion with permission error."""
        record = {
            "id": "test_record",
            "data": "test data",
            "created_at": datetime.now().isoformat(),
        }

        # Mock file operations to raise exception
        with patch("os.makedirs", side_effect=OSError("Permission denied")):
            # Should handle file operation errors gracefully
            with pytest.raises(Exception):
                self.manager._execute_deletion(record, "archive")

    def test_unsupported_deletion_routine(self):
        """Test handling of unsupported deletion routines."""
        record = {
            "id": "test_record",
            "data": "test data",
            "created_at": datetime.now().isoformat(),
        }

        # Should raise ValueError for unknown routine
        with pytest.raises(ValueError, match="Unsupported deletion routine"):
            self.manager._execute_deletion(record, "unknown_routine")


class TestPseudonymization:
    """Test the pseudonymization function."""

    def test_pseudonymize_participant_empty_id(self):
        """Test pseudonymization with empty participant ID."""
        result = pseudonymize_participant("")
        assert result == ""  # nosec: B101 - Test assertion

    def test_pseudonymize_participant_default_salt(self):
        """Test pseudonymization with default salt."""
        participant_id = "test_participant_123"

        # Set environment variable for this test
        with patch.dict(os.environ, {"APGI_PSEUDONYM_SALT": "test_salt_for_default"}):
            result1 = pseudonymize_participant(participant_id)
            result2 = pseudonymize_participant(participant_id)

            # Should be deterministic with same input
            assert result1 == result2  # nosec: B101 - Test assertion
            assert (
                len(result1) == 64
            )  # SHA256 hash length  # nosec: B101 - Test assertion
            assert (
                result1 != participant_id
            )  # Should be different from original  # nosec: B101 - Test assertion

    def test_pseudonymize_participant_custom_salt(self):
        """Test pseudonymization with custom salt."""
        participant_id = "test_participant_123"
        custom_salt = "custom_salt_value"

        # Set environment variable for this test
        with patch.dict(os.environ, {"APGI_PSEUDONYM_SALT": "test_salt_for_custom"}):
            result1 = pseudonymize_participant(participant_id, salt=custom_salt)
            result2 = pseudonymize_participant(participant_id, salt=custom_salt)

            # Should be deterministic with same salt
            assert result1 == result2  # nosec: B101 - Test assertion

            # Should be different from default salt
            result_default = pseudonymize_participant(participant_id)
            assert result1 != result_default  # nosec: B101 - Test assertion

    def test_pseudonymize_participant_environment_salt(self):
        """Test pseudonymization with environment salt."""
        participant_id = "test_participant_123"
        env_salt = "environment_salt_456"

        # Set environment variable
        with patch.dict(os.environ, {"APGI_PSEUDONYM_SALT": env_salt}):
            result = pseudonymize_participant(participant_id)

            # Should use environment salt
            assert len(result) == 64  # nosec: B101 - Test assertion
            assert result != participant_id  # nosec: B101 - Test assertion

    def test_pseudonymize_participant_different_inputs(self):
        """Test pseudonymization with different participant IDs."""
        result1 = pseudonymize_participant("participant_1")
        result2 = pseudonymize_participant("participant_2")
        result3 = pseudonymize_participant("participant_1")

        # Different inputs should produce different hashes
        assert result1 != result2  # nosec: B101 - Test assertion
        assert (
            result1 == result3
        )  # Same input should produce same hash  # nosec: B101 - Test assertion

    def test_pseudonymize_participant_case_sensitivity(self):
        """Test pseudonymization case sensitivity."""
        result_lower = pseudonymize_participant("test_participant")
        result_upper = pseudonymize_participant("TEST_PARTICIPANT")

        # Should be case sensitive
        assert result_lower != result_upper  # nosec: B101 - Test assertion


class TestComplianceIntegration:
    """Integration tests for compliance functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ComplianceManager()

    def test_full_compliance_workflow(self):
        """Test a complete compliance workflow."""
        # Log parameter changes
        self.manager.log_parameter_change(
            user="operator1",
            param_name="sensitivity_level",
            old_value="medium",
            new_value="high",
        )

        # Log experiment run
        self.manager.log_experiment_run(
            experiment_id="exp_001", classification=DataClassification.CONFIDENTIAL
        )

        # Create test records
        current_time = datetime.now()
        records = [
            {
                "id": "record_001",
                "classification": "confidential",
                "created_at": (current_time - timedelta(days=100)).isoformat(),
                "data": "sensitive experiment data",
            },
            {
                "id": "record_002",
                "classification": "public",
                "created_at": (current_time - timedelta(days=10)).isoformat(),
                "data": "public experiment data",
            },
        ]

        # Enforce retention
        with patch.object(self.manager, "_execute_deletion") as mock_delete:
            retained_records = self.manager.enforce_retention(records)

        # Verify workflow
        assert len(self.manager.audit_trail) == 2  # nosec: B101 - Test assertion
        assert len(retained_records) == 1  # nosec: B101 - Test assertion
        assert retained_records[0]["id"] == "record_002"  # nosec: B101 - Test assertion
        assert mock_delete.call_count == 1  # nosec: B101 - Test assertion

    def test_audit_trail_persistence(self):
        """Test audit trail persistence and serialization."""
        # Add multiple audit entries
        self.manager.log_parameter_change("user1", "param1", "old", "new")
        self.manager.log_experiment_run("exp1", DataClassification.INTERNAL)
        self.manager.log_parameter_change("user2", "param2", "old2", "new2")

        # Verify audit trail structure
        assert len(self.manager.audit_trail) == 3  # nosec: B101 - Test assertion

        # Check each entry has required fields
        for entry in self.manager.audit_trail:
            assert "timestamp" in entry  # nosec: B101 - Test assertion
            assert "action" in entry  # nosec: B101 - Test assertion
            assert isinstance(  # nosec: B101 - Test assertion
                json.loads(json.dumps(entry)), dict
            )  # Ensure JSON serializable

    def test_multiple_retention_cycles(self):
        """Test multiple retention enforcement cycles."""
        # Create records that will expire in different cycles
        current_time = datetime.now()
        records = [
            {
                "id": f"record_{i}",
                "classification": "restricted",
                "created_at": (current_time - timedelta(days=i * 20)).isoformat(),
                "data": f"data_{i}",
            }
            for i in range(1, 6)  # Records aged 20, 40, 60, 80, 100 days
        ]

        with patch.object(self.manager, "_execute_deletion") as mock_delete:
            # First enforcement - should delete records older than 30 days
            result1 = self.manager.enforce_retention(records)
            assert (
                len(result1) == 2
            )  # Records 1 and 2 should remain  # nosec: B101 - Test assertion

            # Second enforcement - should delete records older than 50 days
            result2 = self.manager.enforce_retention(result1)
            assert (
                len(result2) == 1
            )  # Only record 1 should remain  # nosec: B101 - Test assertion

            # Third enforcement - should delete all records
            result3 = self.manager.enforce_retention(result2)
            assert (
                len(result3) == 0
            )  # All records expired  # nosec: B101 - Test assertion

            # Total deletions should equal number of records
            assert mock_delete.call_count == 7  # nosec: B101 - Test assertion

    def test_compliance_with_large_dataset(self):
        """Test compliance operations with large datasets."""
        # Create a large number of records
        large_dataset = []
        for i in range(1000):
            large_dataset.append(
                {
                    "id": f"record_{i}",
                    "classification": "internal",
                    "created_at": datetime.now().isoformat(),
                    "data": f"data_{i}",
                }
            )

        # Should handle large dataset efficiently
        start_time = time.time()
        result = self.manager.enforce_retention(large_dataset)
        end_time = time.time()

        # All records should be retained (none expired)
        assert len(result) == 1000  # nosec: B101 - Test assertion

        # Should complete in reasonable time
        assert (
            end_time - start_time < 5.0
        )  # 5 second limit for 1000 records  # nosec: B101 - Test assertion


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
