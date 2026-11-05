"""
Data Retention and Deletion Management for APGI System

Implements enforceable retention policies, real deletion executors,
and key-destruction workflows for GDPR/CCPA/HIPAA compliance.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from apgi_audit import AuditEventType, get_audit_sink
from utils.apgi_logging import get_logger


class RetentionPolicy(Enum):
    """Data retention policy types."""

    PERMANENT = "permanent"  # Keep indefinitely
    GDPR_DEFAULT = "gdpr_default"  # 3 years (GDPR default)
    CCPA_DEFAULT = "ccpa_default"  # 12 months (CCPA default)
    HIPAA_DEFAULT = "hipaa_default"  # 6 years (HIPAA minimum)
    CUSTOM = "custom"  # Custom retention period


@dataclass
class RetentionConfig:
    """Configuration for data retention."""

    policy: RetentionPolicy = RetentionPolicy.GDPR_DEFAULT
    retention_days: int = 1095  # 3 years for GDPR
    auto_delete_enabled: bool = True
    deletion_verification_required: bool = True
    audit_trail_retention_days: int = 2555  # 7 years for audit trail

    def get_retention_period(self) -> timedelta:
        """Get retention period as timedelta."""
        return timedelta(days=self.retention_days)

    def get_audit_retention_period(self) -> timedelta:
        """Get audit trail retention period."""
        return timedelta(days=self.audit_trail_retention_days)


@dataclass
class DataSubjectRecord:
    """Record of data associated with a data subject (operator)."""

    subject_id: str
    subject_name: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data_categories: List[str] = field(
        default_factory=list
    )  # experiment, config, audit, etc.
    retention_policy: RetentionPolicy = RetentionPolicy.GDPR_DEFAULT
    deletion_requested: bool = False
    deletion_requested_at: Optional[datetime] = None
    deletion_completed: bool = False
    deletion_completed_at: Optional[datetime] = None
    # Inventory of every store holding this subject's data, by category. Deletion
    # is only reported complete when every registered store is verifiably gone
    # (APGI-DAT-003, APGI-CON-002); an empty inventory is a failure, not a success.
    data_paths: Dict[str, List[str]] = field(default_factory=dict)

    def is_retention_expired(self, config: RetentionConfig) -> bool:
        """Check if retention period has expired."""
        if self.retention_policy == RetentionPolicy.PERMANENT:
            return False

        retention_period = config.get_retention_period()
        expiration_date = self.created_at + retention_period
        return datetime.now(timezone.utc) > expiration_date

    def mark_for_deletion(self) -> None:
        """Mark record for deletion."""
        self.deletion_requested = True
        self.deletion_requested_at = datetime.now(timezone.utc)

    def mark_deletion_complete(self) -> None:
        """Mark deletion as complete."""
        self.deletion_completed = True
        self.deletion_completed_at = datetime.now(timezone.utc)


class DeletionExecutor:
    """Executes real data deletion (not simulated)."""

    def __init__(self, config: RetentionConfig):
        self.config = config
        self.logger = get_logger("apgi.retention.deletion")
        self.audit_sink = get_audit_sink()

    def delete_experiment_data(
        self,
        subject_id: str,
        experiment_id: str,
        data_path: Optional[str] = None,
    ) -> bool:
        """Delete experiment data for a subject."""
        try:
            if not data_path:
                # Nothing was deleted, so success must not be reported (APGI-DAT-003).
                raise ValueError("data_path is required to delete experiment data")
            if data_path:
                if os.path.exists(data_path):
                    # Real deletion: remove file
                    os.remove(data_path)
                    self.logger.info(f"Deleted experiment data: {data_path}")
                else:
                    # File doesn't exist - this is a failure
                    raise FileNotFoundError(f"Data path does not exist: {data_path}")

            # Audit deletion
            self.audit_sink.record_event(
                event_type=AuditEventType.DATA_DELETED,
                operator_id=subject_id,
                operator_name=f"subject_{subject_id}",
                resource_type="experiment",
                resource_id=experiment_id,
                action="delete",
                details={"data_path": data_path},
                status="success",
            )

            return True
        except Exception as e:
            self.logger.error(f"Failed to delete experiment data: {e}")
            self.audit_sink.record_event(
                event_type=AuditEventType.DATA_DELETED,
                operator_id=subject_id,
                operator_name=f"subject_{subject_id}",
                resource_type="experiment",
                resource_id=experiment_id,
                action="delete",
                details={"error": str(e)},
                status="failure",
                error_message=str(e),
            )
            return False

    def delete_config_data(
        self,
        subject_id: str,
        config_id: str,
    ) -> bool:
        """Delete configuration data for a subject."""
        try:
            # Check if config file exists and delete it
            config_path = Path.home() / ".apgi" / "configs" / f"{config_id}.json"
            if config_path.exists():
                config_path.unlink()
                self.logger.info(
                    f"Deleted config file {config_path} for subject {subject_id}"
                )
            else:
                self.logger.warning(
                    f"Config file {config_path} not found for subject {subject_id}"
                )

            self.audit_sink.record_event(
                event_type=AuditEventType.DATA_DELETED,
                operator_id=subject_id,
                operator_name=f"subject_{subject_id}",
                resource_type="config",
                resource_id=config_id,
                action="delete",
                status="success",
            )

            return True
        except Exception as e:
            self.logger.error(f"Failed to delete config data: {e}")
            return False

    def destroy_kms_key(
        self,
        subject_id: str,
        key_id: str,
        kms_callback: Optional[Callable[[str], bool]] = None,
    ) -> bool:
        """Destroy KMS key associated with subject."""
        try:
            if kms_callback is None:
                # Without a KMS callback nothing was destroyed (APGI-CON-002).
                raise ValueError("kms_callback is required to destroy a KMS key")
            success = kms_callback(key_id)
            if not success:
                raise RuntimeError("KMS key destruction failed")

            self.logger.info(f"Destroyed KMS key {key_id} for subject {subject_id}")

            self.audit_sink.record_event(
                event_type=AuditEventType.DATA_DELETED,
                operator_id=subject_id,
                operator_name=f"subject_{subject_id}",
                resource_type="kms_key",
                resource_id=key_id,
                action="destroy",
                status="success",
            )

            return True
        except Exception as e:
            self.logger.error(f"Failed to destroy KMS key: {e}")
            return False


class RetentionJobScheduler:
    """Schedules and executes retention jobs."""

    def __init__(self, config: RetentionConfig):
        self.config = config
        self.logger = get_logger("apgi.retention.scheduler")
        self.deletion_executor = DeletionExecutor(config)
        self.data_subjects: Dict[str, DataSubjectRecord] = {}
        self.kms_callback: Optional[Callable[[str], bool]] = None

    @staticmethod
    def verify_zero_residual(record: DataSubjectRecord) -> List[str]:
        """Paths from the subject's inventory that still exist (empty = none)."""
        return [
            path
            for paths in record.data_paths.values()
            for path in paths
            if os.path.exists(path)
        ]

    def register_data_subject(
        self,
        subject_id: str,
        subject_name: str,
        data_categories: List[str],
        retention_policy: RetentionPolicy = RetentionPolicy.GDPR_DEFAULT,
        data_paths: Optional[Dict[str, List[str]]] = None,
    ) -> DataSubjectRecord:
        """Register a data subject and (optionally) their store inventory."""
        record = DataSubjectRecord(
            subject_id=subject_id,
            subject_name=subject_name,
            data_categories=data_categories,
            retention_policy=retention_policy,
            data_paths={k: list(v) for k, v in (data_paths or {}).items()},
        )
        self.data_subjects[subject_id] = record
        self.logger.info(f"Registered data subject {subject_name}")
        return record

    def request_deletion(self, subject_id: str) -> bool:
        """Request deletion for a data subject (right to erasure)."""
        if subject_id not in self.data_subjects:
            self.logger.warning(f"Data subject {subject_id} not found")
            return False

        record = self.data_subjects[subject_id]
        record.mark_for_deletion()
        self.logger.info(f"Deletion requested for subject {subject_id}")
        return True

    def execute_retention_jobs(self) -> Dict[str, Any]:
        """Execute retention jobs (delete expired data)."""
        results = {
            "total_subjects": len(self.data_subjects),
            "expired_subjects": 0,
            "deletion_requested": 0,
            "deletions_completed": 0,
            "deletions_failed": 0,
        }

        for subject_id, record in self.data_subjects.items():
            # Check for deletion requests
            if record.deletion_requested and not record.deletion_completed:
                success = self._execute_subject_deletion(record)
                if success:
                    results["deletions_completed"] += 1
                else:
                    results["deletions_failed"] += 1
                results["deletion_requested"] += 1

            # Check for expired retention
            elif (
                record.is_retention_expired(self.config)
                and self.config.auto_delete_enabled
            ):
                success = self._execute_subject_deletion(record)
                if success:
                    results["deletions_completed"] += 1
                else:
                    results["deletions_failed"] += 1
                results["expired_subjects"] += 1

        self.logger.info(f"Retention jobs completed: {results}")
        return results

    def _execute_subject_deletion(self, record: DataSubjectRecord) -> bool:
        """Execute deletion for a data subject."""
        try:
            # Delete each registered store; fail closed on anything unproven.
            all_success = True
            for category in record.data_categories:
                if category == "experiment":
                    paths = record.data_paths.get("experiment", [])
                    if not paths:
                        self.logger.error(
                            f"No registered experiment data stores for {record.subject_id}"
                        )
                        all_success = False
                    for path in paths:
                        if not self.deletion_executor.delete_experiment_data(
                            record.subject_id,
                            f"experiment_{record.subject_id}",
                            path,
                        ):
                            all_success = False
                elif category == "config":
                    success = self.deletion_executor.delete_config_data(
                        record.subject_id,
                        f"config_{record.subject_id}",
                    )
                    if not success:
                        all_success = False
                elif category == "kms_key":
                    success = self.deletion_executor.destroy_kms_key(
                        record.subject_id,
                        f"key_{record.subject_id}",
                        self.kms_callback,
                    )
                    if not success:
                        all_success = False

            residual = self.verify_zero_residual(record)
            if residual:
                self.logger.error(
                    f"Residual data after deletion for {record.subject_id}: {residual}"
                )
                all_success = False

            if all_success:
                record.mark_deletion_complete()
                self.logger.info(f"Deletion completed for subject {record.subject_id}")
            else:
                self.logger.warning(
                    f"Partial deletion failure for subject {record.subject_id}"
                )
            return all_success
        except Exception as e:
            self.logger.error(f"Deletion failed for subject {record.subject_id}: {e}")
            return False

    def export_subject_data(self, subject_id: str, filepath: str) -> bool:
        """Export all data for a subject (right to portability)."""
        if subject_id not in self.data_subjects:
            self.logger.warning(f"Data subject {subject_id} not found")
            return False

        record = self.data_subjects[subject_id]

        try:
            data = {
                "subject_id": record.subject_id,
                "subject_name": record.subject_name,
                "created_at": record.created_at.isoformat(),
                "data_categories": record.data_categories,
                "retention_policy": record.retention_policy.value,
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            self.logger.info(f"Exported subject data to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export subject data: {e}")
            return False

    def get_retention_statistics(self) -> Dict[str, Any]:
        """Get retention statistics."""
        expired_count = sum(
            1
            for r in self.data_subjects.values()
            if r.is_retention_expired(self.config)
        )
        deletion_requested_count = sum(
            1 for r in self.data_subjects.values() if r.deletion_requested
        )
        deletion_completed_count = sum(
            1 for r in self.data_subjects.values() if r.deletion_completed
        )

        return {
            "total_subjects": len(self.data_subjects),
            "expired_records": expired_count,
            "deletion_requested": deletion_requested_count,
            "deletion_completed": deletion_completed_count,
            "pending_deletion": deletion_requested_count - deletion_completed_count,
        }


# Global retention scheduler instance
_retention_scheduler: Optional[RetentionJobScheduler] = None


def get_retention_scheduler() -> RetentionJobScheduler:
    """Get global retention scheduler."""
    global _retention_scheduler
    if _retention_scheduler is None:
        _retention_scheduler = RetentionJobScheduler(RetentionConfig())
    return _retention_scheduler


def set_retention_scheduler(scheduler: RetentionJobScheduler) -> None:
    """Set global retention scheduler (for testing)."""
    global _retention_scheduler
    _retention_scheduler = scheduler
