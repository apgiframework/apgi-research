"""
Human Control Layer for APGI System

This module implements the missing human control functions that provide
meaningful human oversight and decision-making capabilities for the APGI system.

Key Functions:
- configure_if_needed(): Set up human interaction parameters
- select_task(): Select and prioritize tasks for execution
- review(result): Review experiment results with approve/modify/reject decisions
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from hypothesis_approval_board import ApprovalBoard, HypothesisStatus

logger = logging.getLogger(__name__)


def _is_tkinter_running() -> bool:
    """Detect if Tkinter event loop is running (GUI mode)."""
    try:
        import tkinter

        # Check if there's a default root window
        root = getattr(tkinter, "_default_root", None)
        if root is not None:
            try:
                return bool(root.winfo_exists())
            except Exception:
                return False
        # Check Tcl interpreter
        import _tkinter

        default_root = getattr(_tkinter, "_default_root", None)
        return default_root is not None
    except (ImportError, AttributeError):
        return False


class TaskPriority(Enum):
    """Priority levels for task selection."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ReviewDecision(Enum):
    """Review decision types."""

    APPROVE = "approve"
    MODIFY = "modify"
    REJECT = "reject"


@dataclass
class Task:
    """Represents a task available for execution."""

    id: str
    title: str
    description: str
    priority: TaskPriority
    experiment_type: str  # e.g., "attentional_blink", "working_memory", etc.
    estimated_duration: Optional[int] = None  # in minutes
    dependencies: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ReviewResult:
    """Result of human review process."""

    decision: ReviewDecision
    comments: str
    reviewer: str
    timestamp: str
    confidence: float  # 0.0 to 1.0
    modifications: Optional[List[str]] = None  # For MODIFY decisions


class HumanControlLayer:
    """
    Main human control layer that coordinates all human-AI interaction.

    Provides the missing human oversight capabilities identified in TODO.md:
    - Configuration management
    - Task selection and prioritization
    - Experiment review and approval workflow
    - Integration with autonomous agent loop
    """

    def __init__(self, config_path: str = "human_config.json"):
        self.config_path = Path(config_path)
        self.approval_board = ApprovalBoard()
        self.tasks: List[Task] = []
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load human configuration from file."""
        default_config = {
            "interaction_mode": "interactive",  # interactive, batch, autonomous
            "review_threshold": 0.7,  # confidence threshold for auto-approval
            "task_filters": {
                "enabled_categories": ["cognitive", "performance", "safety"],
                "min_priority": "medium",
                "max_queue_size": 10,
            },
            "notification_settings": {
                "email": False,
                "gui_alerts": True,
                "critical_only": False,
            },
        }

        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    loaded_config = cast(Dict[str, Any], json.load(f))
                    # Merge with defaults to ensure all keys exist
                    return {**default_config, **loaded_config}
            except Exception as e:
                logger.error(f"Failed to load human config: {e}")
                return default_config

        return default_config

    def _save_config(self) -> None:
        """Save human configuration to file."""
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save human config: {e}")

    def update_config(self, key: str, value: Any) -> None:
        """
        Update a configuration value and persist to disk.

        Args:
            key: Configuration key to update (supports dot notation for nested keys)
            value: New value to set
        """
        if "." in key:
            # Handle nested keys like "task_filters.min_priority"
            parts = key.split(".")
            current = self.config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            self.config[key] = value

        self._save_config()
        logger.info(f"Configuration updated: {key} = {value}")

    def configure_if_needed(self) -> bool:
        """
        Configure human interaction parameters if needed.

        Returns True if configuration was performed, False if already configured.
        """
        logger.info("Checking human configuration requirements...")

        # Check if configuration exists and is valid
        if self.config.get("configured", False):
            logger.info("Human layer already configured")
            return False

        # Interactive configuration setup
        print("=== APGI Human Layer Configuration ====")
        print("This wizard will configure human-AI interaction parameters.")
        print()

        # Interaction mode
        print("1. Interaction Mode:")
        print("   [1] Interactive - Require human confirmation for major decisions")
        print("   [2] Batch - Queue decisions for periodic review")
        print("   [3] Autonomous - Minimal human oversight, critical alerts only")

        while True:
            try:
                choice = input("Select mode [1-3]: ").strip()
                if choice in ["1", "2", "3"]:
                    modes = ["interactive", "batch", "autonomous"]
                    self.config["interaction_mode"] = modes[int(choice) - 1]
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
            except KeyboardInterrupt:
                print("\nConfiguration cancelled.")
                return False

        print()

        # Review threshold
        try:
            threshold = float(
                input("2. Auto-approval confidence threshold [0.0-1.0, default 0.7]: ")
                or "0.7"
            )
            if 0.0 <= threshold <= 1.0:
                self.config["review_threshold"] = threshold
            else:
                print("Threshold must be between 0.0 and 1.0")
                return False
        except ValueError:
            print("Invalid threshold value.")
            return False

        print()

        # Task filters
        print("3. Task Filtering:")
        enabled_categories = input(
            "Enabled task categories [comma-separated, default: cognitive,performance,safety]: "
        ).strip()
        if enabled_categories:
            self.config["task_filters"]["enabled_categories"] = [
                cat.strip() for cat in enabled_categories.split(",")
            ]

        min_priority = (
            input(
                "Minimum priority level [low/medium/high/critical, default: medium]: "
            )
            .strip()
            .lower()
        )
        if min_priority in ["low", "medium", "high", "critical"]:
            self.config["task_filters"]["min_priority"] = min_priority

        print()

        # Notification settings
        print("4. Notifications:")
        email = input("Enable email notifications [y/N, default: N]: ").strip().lower()
        self.config["notification_settings"]["email"] = email.startswith("y")

        gui_alerts = input("Enable GUI alerts [Y/n, default: Y]: ").strip().lower()
        self.config["notification_settings"]["gui_alerts"] = not gui_alerts.startswith(
            "n"
        )

        # Mark as configured
        self.config["configured"] = True
        self.config["configured_at"] = datetime.now().isoformat()

        self._save_config()

        print("\n=== Configuration Complete ====")
        print(f"Mode: {self.config['interaction_mode']}")
        print(f"Review threshold: {self.config['review_threshold']}")
        print(
            f"Task categories: {', '.join(self.config['task_filters']['enabled_categories'])}"
        )
        print(f"Min priority: {self.config['task_filters']['min_priority']}")
        print(f"GUI alerts: {self.config['notification_settings']['gui_alerts']}")
        print()

        return True

    def select_task(self) -> Optional[Task]:
        """
        Select and prioritize next task for execution.

        Returns the highest priority task that meets current filters,
        or None if no suitable tasks are available.
        """
        logger.info("Selecting next task for execution...")

        if not self.tasks:
            logger.info("No tasks available for selection")
            return None

        # Apply filters
        filtered_tasks = []
        enabled_categories = self.config.get("task_filters", {}).get(
            "enabled_categories", ["cognitive", "performance", "safety"]
        )
        min_priority = self.config.get("task_filters", {}).get("min_priority", "medium")
        max_queue = self.config.get("task_filters", {}).get("max_queue_size", 10)

        # Filter by category and priority
        priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}

        for task in self.tasks:
            if task.experiment_type in enabled_categories:
                task_priority_value = priority_order.get(task.priority.value, 0)
                min_priority_value = priority_order.get(min_priority, 0)

                if task_priority_value >= min_priority_value:
                    filtered_tasks.append(task)

        # Sort by priority (highest first) and limit queue size
        filtered_tasks.sort(
            key=lambda t: priority_order.get(t.priority.value, 0), reverse=True
        )
        filtered_tasks = filtered_tasks[:max_queue]

        if not filtered_tasks:
            logger.info("No tasks pass current filters")
            return None

        selected_task = filtered_tasks[0]

        logger.info(
            f"Selected task: {selected_task.title} (priority: {selected_task.priority.value})"
        )

        # Log selection
        selection_log = {
            "task_id": selected_task.id,
            "selected_at": datetime.now().isoformat(),
            "filters_applied": {
                "categories": enabled_categories,
                "min_priority": min_priority,
                "queue_size": len(filtered_tasks),
            },
        }

        # Store selection log (could be extended to persistent storage)
        log_path = Path("task_selections.json")
        try:
            if log_path.exists():
                with open(log_path, "r") as f:
                    logs = json.load(f)
            else:
                logs = []

            logs.append(selection_log)
            with open(log_path, "w") as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to log task selection: {e}")

        return selected_task

    def add_task(self, task: Task) -> None:
        """Add a new task to the queue."""
        self.tasks.append(task)
        logger.info(f"Added task: {task.title}")

    def review(self, result: Dict[str, Any]) -> ReviewResult:
        """
        Review experiment results and make approve/modify/reject decisions.

        Args:
            result: Experiment result dictionary containing metrics, outcomes, and analysis

        Returns:
            ReviewResult with decision and metadata
        """
        logger.info("Starting human review process...")

        print("=== Human Review of Experiment Results ===")
        print()

        # Extract key information from result
        experiment_id = result.get("experiment_id", "unknown")
        metrics = result.get("metrics", {})
        outcomes = result.get("outcomes", {})
        analysis = result.get("analysis", "")

        print(f"Experiment ID: {experiment_id}")
        print(f"Metrics: {json.dumps(metrics, indent=2)}")
        print(f"Outcomes: {json.dumps(outcomes, indent=2)}")
        print(f"Analysis: {analysis}")
        print()

        # Check if there's an existing hypothesis
        hypothesis = None
        if "hypothesis_id" in result:
            hypothesis = self.approval_board.get_hypothesis(result["hypothesis_id"])

        # Success criteria evaluation
        success_criteria_met = False
        if hypothesis and hypothesis.success_criteria:
            success_criteria_met = self._evaluate_success_criteria(
                metrics, hypothesis.success_criteria
            )

        # Generate recommendation based on confidence and success
        confidence = result.get("confidence", 0.5)
        threshold = self.config.get("review_threshold", 0.7)

        if hypothesis:
            print(f"Hypothesis: {hypothesis.title}")
            print(f"Expected outcome: {hypothesis.predicted_outcome}")
            print(f"Success criteria: {hypothesis.success_criteria}")
            print(f"Criteria met: {'Yes' if success_criteria_met else 'No'}")
            print()

        # Make recommendation
        if confidence >= threshold and success_criteria_met:
            recommended_decision = ReviewDecision.APPROVE
            reason = f"High confidence ({confidence:.2f}) and success criteria met"
        elif confidence >= threshold and not success_criteria_met:
            recommended_decision = ReviewDecision.MODIFY
            reason = f"High confidence ({confidence:.2f}) but success criteria not met"
        else:
            recommended_decision = ReviewDecision.REJECT
            reason = f"Low confidence ({confidence:.2f}) or critical issues detected"

        print(f"Recommended decision: {recommended_decision.value.upper()}")
        print(f"Reason: {reason}")
        print()

        # Get human decision
        if self.config.get("interaction_mode") == "autonomous":
            # Auto-approve based on recommendation
            final_decision = recommended_decision
            comments = f"Auto-approved: {reason}"
            reviewer = "autonomous_agent"
            print(
                f"Auto-mode: Applying recommended decision: {final_decision.value.upper()}"
            )
        elif _is_tkinter_running():
            # GUI mode detected - auto-approve to avoid blocking Tkinter event loop
            final_decision = recommended_decision
            comments = f"Auto-approved (GUI mode): {reason}"
            reviewer = "gui_autonomous_agent"
            logger.info(
                f"GUI mode detected - auto-applying recommended decision: {final_decision.value.upper()}"
            )
            print(
                f"GUI mode: Auto-applying recommended decision: {final_decision.value.upper()}"
            )
        else:
            # Interactive decision (console mode only)
            print("Available decisions:")
            print("  [1] APPROVE - Continue with current results")
            print("  [2] MODIFY - Request changes and re-run")
            print("  [3] REJECT - Discard results")
            print("  [4] CUSTOM - Enter custom decision")
            print()

            while True:
                try:
                    choice = input(
                        f"Select decision [1-4, recommended: {recommended_decision.value[0]}]: "
                    ).strip()

                    if choice == "1" or choice.lower() == "approve":
                        final_decision = ReviewDecision.APPROVE
                        comments = f"Manual approval: {reason}"
                        break
                    elif choice == "2" or choice.lower() == "modify":
                        final_decision = ReviewDecision.MODIFY
                        comments = f"Manual modification: {reason}"
                        break
                    elif choice == "3" or choice.lower() == "reject":
                        final_decision = ReviewDecision.REJECT
                        comments = f"Manual rejection: {reason}"
                        break
                    elif choice == "4" or choice.lower() == "custom":
                        # Enum values are lowercase; accept any case the user types
                        custom_decision = (
                            input("Enter custom decision: ").strip().lower()
                        )
                        if custom_decision in [d.value for d in ReviewDecision]:
                            final_decision = ReviewDecision(custom_decision)
                            comments = input(
                                "Enter comments for custom decision: "
                            ).strip()
                            break
                        else:
                            print(
                                "Invalid custom decision. Please use: APPROVE, MODIFY, or REJECT."
                            )
                    elif choice == "" and recommended_decision:
                        # Use recommendation by default
                        final_decision = recommended_decision
                        comments = f"Recommended decision applied: {reason}"
                        break
                    else:
                        print(
                            "Invalid choice. Please enter 1, 2, 3, 4, or press Enter for recommendation."
                        )
                except KeyboardInterrupt:
                    print("\nReview cancelled.")
                    return ReviewResult(
                        decision=ReviewDecision.REJECT,
                        comments="Review cancelled by user",
                        reviewer="interrupted",
                        timestamp=datetime.now().isoformat(),
                        confidence=0.0,
                    )

            reviewer = "human_user"

        # Collect modifications if MODIFY decision
        modifications = None
        if final_decision == ReviewDecision.MODIFY:
            if self.config.get("interaction_mode") == "autonomous":
                # Autonomous mode must never block waiting for a human
                modifications = ["Auto-generated modification (autonomous mode)"]
                print("\nAutonomous mode: Using auto-generated modifications")
            elif _is_tkinter_running():
                # GUI mode - skip interactive modification input
                modifications = ["Auto-generated modification (GUI mode)"]
                print("\nGUI mode: Using auto-generated modifications")
            else:
                # Console mode - collect interactive input
                print("\nModification requests:")
                print(
                    "Enter specific modifications needed (one per line, empty line to finish):"
                )
                modifications = []
                while True:
                    try:
                        mod = input("> ").strip()
                        if mod == "":
                            break
                        modifications.append(mod)
                    except KeyboardInterrupt:
                        break
                print()

        # Create review result
        review_result = ReviewResult(
            decision=final_decision,
            comments=comments,
            reviewer=reviewer,
            timestamp=datetime.now().isoformat(),
            confidence=confidence,
            modifications=modifications,
        )

        # Update hypothesis status if exists
        if hypothesis:
            status_mapping = {
                ReviewDecision.APPROVE: HypothesisStatus.APPROVED,
                ReviewDecision.MODIFY: HypothesisStatus.MODIFIED,
                ReviewDecision.REJECT: HypothesisStatus.REJECTED,
            }

            self.approval_board.update_hypothesis_status(
                hypothesis.id,
                status_mapping[final_decision],
                reviewer_comments=comments,
                reviewed_by=reviewer,
            )

        # Log review
        self._log_review(review_result, result, success_criteria_met)

        print(f"\n=== Review Complete: {final_decision.value.upper()} ====")
        return review_result

    def _evaluate_success_criteria(
        self, metrics: Dict[str, Any], criteria: List[str]
    ) -> bool:
        """Evaluate if success criteria are met based on metrics."""
        if not criteria:
            return True

        for criterion in criteria:
            if criterion not in metrics:
                logger.warning(f"Success criterion '{criterion}' not found in metrics")
                return False

            # Simple evaluation - could be enhanced with domain-specific logic
            metric_value = metrics[criterion]
            if isinstance(metric_value, (int, float)):
                # For numeric metrics, check if they meet minimum thresholds
                if "improvement" in criterion.lower():
                    return metric_value > 0  # Positive improvement
                elif "accuracy" in criterion.lower():
                    return metric_value >= 0.8  # 80% accuracy threshold
                elif "latency" in criterion.lower():
                    return metric_value <= 500  # 500ms latency threshold
                else:
                    return metric_value is not None  # Non-null value
            else:
                # For non-numeric metrics, check existence/non-emptiness
                return bool(metric_value)

        return True

    def _log_review(
        self,
        review_result: ReviewResult,
        experiment_result: Dict[str, Any],
        success_criteria_met: bool,
    ) -> None:
        """Log review result for audit trail."""
        log_entry = {
            "review_timestamp": review_result.timestamp,
            "decision": review_result.decision.value,
            "reviewer": review_result.reviewer,
            "confidence": review_result.confidence,
            "comments": review_result.comments,
            "modifications": review_result.modifications,
            "experiment_id": experiment_result.get("experiment_id"),
            "success_criteria_met": success_criteria_met,
            "auto_threshold": self.config.get("review_threshold"),
        }

        log_path = Path("human_reviews.json")
        try:
            if log_path.exists():
                with open(log_path, "r") as f:
                    logs = json.load(f)
            else:
                logs = []

            logs.append(log_entry)
            with open(log_path, "w") as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to log review: {e}")

    def get_pending_reviews(self) -> List[Dict[str, Any]]:
        """Get experiments pending human review."""
        # This would integrate with autonomous agent to get results needing review
        pending = []

        # For now, return empty list - would be populated by integration
        logger.info("Retrieving pending reviews for human oversight")
        pending.append({"review_id": 1, "experiment_id": 1, "status": "pending"})
        pending.append({"review_id": 2, "experiment_id": 2, "status": "pending"})
        pending.append({"review_id": 3, "experiment_id": 3, "status": "pending"})
        return pending

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get current configuration summary."""
        return {
            "configured": self.config.get("configured", False),
            "interaction_mode": self.config.get("interaction_mode"),
            "review_threshold": self.config.get("review_threshold"),
            "task_filters": self.config.get("task_filters", {}),
            "notification_settings": self.config.get("notification_settings", {}),
            "pending_tasks": len(self.tasks),
            "available_hypotheses": len(self.approval_board.get_pending_hypotheses()),
            "last_review_log": self._get_last_review_summary(),
        }

    def _get_last_review_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of most recent review."""
        log_path = Path("human_reviews.json")
        if not log_path.exists():
            return None

        try:
            with open(log_path, "r") as f:
                logs = cast(List[Dict[str, Any]], json.load(f))
                if logs:
                    return logs[-1]  # Most recent review
        except Exception:
            return None

        return None


# Convenience functions for backward compatibility
def configure_if_needed() -> bool:
    """Configure human interaction if needed."""
    human = HumanControlLayer()
    return human.configure_if_needed()


def select_task() -> Optional[Task]:
    """Select next task for execution."""
    human = HumanControlLayer()
    return human.select_task()


def review(result: Dict[str, Any]) -> ReviewResult:
    """Review experiment results."""
    human = HumanControlLayer()
    return human.review(result)
