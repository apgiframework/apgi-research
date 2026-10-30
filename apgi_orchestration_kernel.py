"""
APGI Orchestration Kernel

Central runner framework that consolidates trial extraction and reduces
per-script duplication. Provides the single integration path for all experiments.

This kernel replaces the need for 29 separate runner implementations with
a unified, typed adapter pattern.
"""

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar

from apgi_audit import AuditEventType, get_audit_sink
from apgi_integration import APGIIntegration, APGIParameters
from apgi_security_adapters import SecurityLevel, get_security_factory
from utils.apgi_authz import get_authz_manager
from utils.apgi_config import APGIExperimentConfigSchema
from utils.apgi_errors import APGIIntegrationError, APGIRuntimeError, APGITimeoutError
from utils.apgi_logging import APGIContextLogger, get_logger

T = TypeVar("T")  # Generic trial data type


class TrialExtractor(Protocol):
    """Protocol for extracting trial data from experiment results."""

    def extract_trial_data(self, trial_result: Any) -> Dict[str, Any]:
        """Extract APGI-relevant trial data from experiment result."""
        ...


class TrialTransformer(ABC):
    """Base class for experiment-specific trial transformations."""

    @abstractmethod
    def transform_trial(self, trial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform trial data to APGI format."""
        pass

    @abstractmethod
    def extract_prediction_error(self, trial_data: Dict[str, Any]) -> float:
        """Extract prediction error from trial data."""
        pass

    @abstractmethod
    def extract_precision(self, trial_data: Dict[str, Any]) -> float:
        """Extract precision estimate from trial data."""
        pass


@dataclass
class TrialMetrics:
    """Standardized trial metrics across all experiments."""

    trial_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trial_number: int = 0
    timestamp: float = field(default_factory=time.time)

    # Core APGI metrics
    prediction_error: float = 0.0
    precision: float = 1.0
    somatic_marker: float = 0.0
    ignition_probability: float = 0.0

    # Experiment-specific metrics
    experiment_metrics: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    operator_id: str = ""
    experiment_name: str = ""


@dataclass
class ExperimentRunConfig:
    """Configuration for a single experiment run."""

    experiment_name: str
    operator_id: str
    operator_name: str
    apgi_config: APGIExperimentConfigSchema
    timeout_seconds: int = 600
    enable_hierarchical: bool = True
    enable_precision_gap: bool = True
    security_level: str = "standard"

    # Optional callbacks
    trial_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    completion_callback: Optional[Callable[[Dict[str, Any]], None]] = None


class APGIOrchestrationKernel:
    """
    Central APGI orchestration kernel for all experiments.

    Provides:
    - Unified trial processing pipeline
    - Standardized metrics collection
    - Authorization and audit integration
    - Security context management
    - Timeout and error handling
    """

    def __init__(self) -> None:
        self.logger = get_logger("apgi.kernel")
        self.authz_manager = get_authz_manager()
        self.audit_sink = get_audit_sink()
        self.security_factory = get_security_factory()

        # Run tracking
        self.active_runs: Dict[str, Dict[str, Any]] = {}
        self.completed_runs: List[Dict[str, Any]] = []

    def create_run_context(
        self,
        config: ExperimentRunConfig,
    ) -> Dict[str, Any]:
        """Create a new experiment run context."""
        run_id = str(uuid.uuid4())

        # Create security context
        security_context = self.security_factory.create_context(
            operator_id=config.operator_id,
            role="operator",
            security_level=SecurityLevel[config.security_level.upper()],
        )

        # Create logger with context
        context_logger = APGIContextLogger(
            get_logger(f"apgi.run.{run_id}"),
            correlation_id=run_id,
        )

        # Initialize APGI integration
        apgi = APGIIntegration(
            APGIParameters(
                tau_S=config.apgi_config.tau_S,
                beta=config.apgi_config.beta,
                theta_0=config.apgi_config.theta_0,
                alpha=config.apgi_config.alpha,
            )
        )

        run_context = {
            "run_id": run_id,
            "config": config,
            "security_context": security_context,
            "logger": context_logger,
            "apgi": apgi,
            "start_time": time.time(),
            "trial_count": 0,
            "trial_metrics": [],
            "status": "running",
        }

        self.active_runs[run_id] = run_context

        # Audit: experiment started
        self.audit_sink.record_event(
            event_type=AuditEventType.EXPERIMENT_STARTED,
            operator_id=config.operator_id,
            operator_name=config.operator_name,
            resource_type="experiment",
            resource_id=config.experiment_name,
            action="start",
            details={
                "run_id": run_id,
                "timeout_seconds": config.timeout_seconds,
            },
        )

        context_logger.info(
            f"Created run context {run_id} for {config.experiment_name}"
        )

        return run_context

    def process_trial(
        self,
        run_context: Dict[str, Any],
        trial_data: Dict[str, Any],
        trial_transformer: TrialTransformer,
    ) -> TrialMetrics:
        """Process a single trial through APGI pipeline."""
        config = run_context["config"]
        logger = run_context["logger"]
        apgi = run_context["apgi"]

        # Check timeout
        elapsed = time.time() - run_context["start_time"]
        if elapsed > config.timeout_seconds:
            raise APGITimeoutError(
                f"Experiment exceeded timeout of {config.timeout_seconds}s",
                context={"run_id": run_context["run_id"], "elapsed": elapsed},
            )

        # Transform trial data
        try:
            transformed_data = trial_transformer.transform_trial(trial_data)
            prediction_error = trial_transformer.extract_prediction_error(
                transformed_data
            )
            precision = trial_transformer.extract_precision(transformed_data)
        except Exception as e:
            raise APGIIntegrationError(
                f"Failed to transform trial data: {e}",
                context={"trial_number": run_context["trial_count"]},
            )

        # Process through APGI
        try:
            ignition_prob = apgi.compute_ignition_probability(
                prediction_error=prediction_error,
                precision=precision,
                somatic_marker=0.0,
            )
        except Exception as e:
            raise APGIRuntimeError(
                f"APGI processing failed: {e}",
                context={"trial_number": run_context["trial_count"]},
            )

        # Create trial metrics
        trial_metrics = TrialMetrics(
            trial_number=run_context["trial_count"] + 1,
            prediction_error=prediction_error,
            precision=precision,
            ignition_probability=ignition_prob,
            operator_id=config.operator_id,
            experiment_name=config.experiment_name,
            experiment_metrics=transformed_data,
        )

        # Store metrics
        run_context["trial_metrics"].append(trial_metrics)
        run_context["trial_count"] += 1

        # Call trial callback if provided
        if config.trial_callback:
            try:
                config.trial_callback(trial_metrics.__dict__)
            except Exception as e:
                logger.warning(f"Trial callback failed: {e}")

        logger.debug(f"Processed trial {trial_metrics.trial_number}")
        return trial_metrics

    def finalize_run(
        self,
        run_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Finalize experiment run and collect results."""
        config = run_context["config"]
        logger = run_context["logger"]
        apgi = run_context["apgi"]

        # Get APGI summary
        apgi_summary = apgi.finalize()

        # Compute run statistics
        elapsed = time.time() - run_context["start_time"]

        results = {
            "run_id": run_context["run_id"],
            "experiment_name": config.experiment_name,
            "operator_id": config.operator_id,
            "status": "completed",
            "trial_count": run_context["trial_count"],
            "elapsed_seconds": elapsed,
            "apgi_summary": apgi_summary,
            "trial_metrics": [m.__dict__ for m in run_context["trial_metrics"]],
        }

        # Call completion callback if provided
        if config.completion_callback:
            try:
                config.completion_callback(results)
            except Exception as e:
                logger.warning(f"Completion callback failed: {e}")

        # Audit: experiment completed
        self.audit_sink.record_event(
            event_type=AuditEventType.EXPERIMENT_COMPLETED,
            operator_id=config.operator_id,
            operator_name=config.operator_name,
            resource_type="experiment",
            resource_id=config.experiment_name,
            action="complete",
            details={
                "run_id": run_context["run_id"],
                "trial_count": run_context["trial_count"],
                "elapsed_seconds": elapsed,
            },
        )

        # Move to completed runs
        self.active_runs.pop(run_context["run_id"], None)
        self.completed_runs.append(results)

        logger.info(f"Finalized run {run_context['run_id']}")

        return results

    def get_run_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a run."""
        if run_id in self.active_runs:
            run = self.active_runs[run_id]
            return {
                "run_id": run_id,
                "status": "running",
                "trial_count": run["trial_count"],
                "elapsed_seconds": time.time() - run["start_time"],
            }

        for completed_run in self.completed_runs:
            if completed_run["run_id"] == run_id:
                return {
                    "run_id": run_id,
                    "status": "completed",
                    "trial_count": completed_run["trial_count"],
                    "elapsed_seconds": completed_run["elapsed_seconds"],
                }

        return None

    def get_metrics(self) -> Dict[str, Any]:
        """Get kernel metrics."""
        return {
            "active_runs": len(self.active_runs),
            "completed_runs": len(self.completed_runs),
            "total_trials_processed": sum(
                r["trial_count"] for r in self.completed_runs
            ),
            "security_metrics": self.security_factory.get_metrics(),
            "audit_events": len(self.audit_sink.events),
        }


# Global kernel instance
_kernel: Optional[APGIOrchestrationKernel] = None


def get_orchestration_kernel() -> APGIOrchestrationKernel:
    """Get global orchestration kernel."""
    global _kernel
    if _kernel is None:
        _kernel = APGIOrchestrationKernel()
    return _kernel


def set_orchestration_kernel(kernel: Optional[APGIOrchestrationKernel]) -> None:
    """Set global orchestration kernel (for testing)."""
    global _kernel
    _kernel = kernel
