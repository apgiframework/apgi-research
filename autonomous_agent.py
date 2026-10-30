"""
Autonomous Agent Controller for APGI Auto-Improvement System

This module implements the infinite autonomous optimization loop described in USAGE.md.
It provides AI-driven parameter optimization, Git-based learning, and performance tracking.

Key Features:
- Infinite autonomous optimization loop
- Git-based performance tracking and rollback
- AI agent parameter modification strategies
- Performance-driven decision logic
- Multi-experiment optimization coordination
- Async/await Git operations for non-blocking execution
- Rate limiting for autonomous operations
- Request timeouts with retry logic

Usage:
    python autonomous_agent.py --experiment masking --iterations 100
    python autonomous_agent.py --all-experiments --overnight

Classes:
    AutonomousAgent: Main autonomous agent controller
    GitPerformanceTracker: Git-based performance tracking
    ParameterOptimizer: AI-driven optimization algorithms
    ExperimentResult: Data structure for experiment results
    TimeoutError: Custom timeout exception
    RateLimiter: Rate limiting for autonomous operations
    RequestRetryHandler: Request timeout and retry logic
"""

import argparse
import ast
import asyncio
import importlib
import json
import logging
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import git
import numpy as np

if TYPE_CHECKING:
    from typing import Any

# Import APGI components (unused for now, available for future integration)
from human_layer import HumanControlLayer
from memory_store import MemoryStore, update_memory_from_report
from xpr_agent_engine import XPRAgentEngineEnhanced, register_xpr_skills

# from apgi_integration import APGIIntegration, APGIParameters, format_apgi_output
# from experiment_apgi_integration import ExperimentAPGIRunner, get_experiment_apgi_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("autonomous_agent.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class APGITimeoutError(Exception):
    """Custom timeout exception."""

    pass


def timeout_handler(signum: int, frame: Any) -> None:
    """Signal handler for timeout."""
    raise APGITimeoutError("Experiment execution timed out")


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""

    commit_hash: str
    experiment_name: str
    primary_metric: float
    apgi_metrics: Dict[str, float]
    apgi_enhanced_metric: Optional[float]
    completion_time_s: float
    timestamp: str
    parameter_modifications: Dict[str, Any]
    status: str  # "success", "crash", "timeout"


@dataclass
class ExperimentPlan:
    """APGI generated Hypothesis and Execution representation."""

    hypothesis: str
    success_metrics: Dict[
        str, str
    ]  # Metric name mapped to target condition (e.g. "> 5%")
    constraints: List[str]  # e.g. "Do not modify file X structure"
    steps: List[str]  # Sequence of instructions for agent execution


@dataclass
class ExecutionReport:
    """APGI outcome abstract reporting and analysis."""

    experiment_name: str
    summary: str
    metric_deltas: Dict[str, float]
    root_causes: List[str]
    suggested_fixes: List[str]
    confidence_score: float  # Scale of 0-1, used by guardrails


@dataclass
class OptimizationStrategy:
    """AI agent optimization strategy."""

    name: str
    description: str
    parameter_ranges: Dict[str, Tuple[Any, ...]]
    mutation_strength: float
    exploration_rate: float
    learning_rate: float


class GitPerformanceTracker:
    """Git-based performance tracking and optimization."""

    repo: Optional[git.Repo]
    async_git: Optional["AsyncGitOperations"]

    def __init__(self, repo_path: str = ".", agent: Optional[Any] = None):
        self.repo_path = Path(repo_path)
        self.agent = agent

        # Handle missing git repository gracefully
        try:
            self.repo = git.Repo(self.repo_path)
        except git.InvalidGitRepositoryError:
            logger.warning(
                f"No git repository at {self.repo_path}, initializing new repo"
            )
            try:
                self.repo = git.Repo.init(self.repo_path)
            except Exception as e:
                logger.error(f"Failed to initialize git repository: {e}")
                self.repo = None
        except Exception as e:
            logger.warning(
                f"Could not open git repository: {e}, operating without git tracking"
            )
            self.repo = None

        self.results_file = self.repo_path / "optimization_results.json"
        self.best_results = self._load_best_results()

        # Initialize async Git operations for non-blocking execution (only if repo exists)
        self.rate_limiter = RateLimiter(max_requests=20, time_window=60)
        if self.repo is not None:
            self.async_git = AsyncGitOperations(self.repo_path, self.rate_limiter)
        else:
            self.async_git = None

    def _load_best_results(self) -> Dict[str, ExperimentResult]:
        """Load best results from previous runs."""
        if self.results_file.exists():
            try:
                with open(self.results_file, "r") as f:
                    data = json.load(f)
                return {
                    exp_name: ExperimentResult(**result_data)
                    for exp_name, result_data in data.items()
                }
            except Exception as e:
                logger.warning(f"Could not load results file: {e}")
        return {}

    def save_results(self, results: Dict[str, ExperimentResult]) -> None:
        """Save results to file."""
        try:
            with open(self.results_file, "w") as f:
                json.dump(
                    {exp_name: asdict(result) for exp_name, result in results.items()},
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"Could not save results: {e}")

    def commit_experiment(self, modifications: Dict[str, Any]) -> str:
        """Commit experiment modifications and return commit hash."""
        if self.repo is None:
            logger.warning("Git repository not available, skipping commit")
            return "no_repo"

        try:
            # Only stage specific experiment files, not everything
            files_to_stage = [
                "run_*.py",  # Only run files
                "*.md",  # Documentation files
                "*.txt",  # Text files
                "*.json",  # JSON configuration files
            ]

            # Use regular git operations for compatibility
            repo = git.Repo(self.repo_path)
            # Stage files matching patterns
            for pattern in files_to_stage:
                try:
                    repo.index.add(pattern)
                except (git.exc.GitCommandError, FileNotFoundError):
                    # If pattern doesn't match any files or git command fails, continue
                    pass

            # Create commit message
            mod_desc = ", ".join([f"{k}={v}" for k, v in modifications.items()])
            commit_msg = (
                f"Experiment: {mod_desc}"
                if mod_desc
                else "Experiment: Parameter update"
            )

            # Commit changes
            if repo.is_dirty(untracked_files=True):
                commit = repo.index.commit(commit_msg)
                return commit.hexsha
            else:
                # No changes to commit
                return "no_changes"

        except Exception as e:
            logger.error(f"Could not commit experiment: {e}")
            return "error"

    def rollback_experiment(self, commit_hash: Optional[str] = None) -> bool:
        """Rollback to previous commit."""
        if self.repo is None:
            logger.warning("Git repository not available, skipping rollback")
            return False

        try:
            target = commit_hash if commit_hash else "HEAD~1"
            repo = git.Repo(self.repo_path)
            repo.git.reset("--hard", target)
            return True
        except Exception as e:
            logger.error(f"Could not rollback experiment: {e}")
            return False

    def is_improvement(self, experiment_name: str, new_metric: float) -> bool:
        """Check if new metric is an improvement over best known."""
        if experiment_name not in self.best_results:
            return True

        best_metric = self.best_results[experiment_name].primary_metric
        # Default direction logic if no agent available
        if self.agent:
            direction = self.agent._get_metric_direction(experiment_name)
        else:
            # Default: higher is better for most metrics
            direction = "higher"

        if direction == "higher":
            return new_metric > best_metric
        else:  # "lower"
            return new_metric < best_metric

    def get_best_metric(self, experiment_name: str) -> Optional[float]:
        """Get best known metric for experiment."""
        if experiment_name in self.best_results:
            return self.best_results[experiment_name].primary_metric
        return None


class ParameterOptimizer:
    """AI-driven parameter optimization algorithms."""

    def __init__(self) -> None:
        self.strategies = self._initialize_strategies()
        self.current_strategy = None

    def _initialize_strategies(self) -> Dict[str, OptimizationStrategy]:
        """Initialize optimization strategies for different experiment types."""
        return {
            "attentional_blink": OptimizationStrategy(
                name="attention_optimization",
                description="Optimize for attentional processing speed",
                parameter_ranges={
                    "SOA_VALUES_TO_TEST": ([10, 20, 30, 50, 80, 100, 150, 200], "list"),
                    "BASE_DETECTION_RATE": (0.5, 0.9, "float"),
                    "MASKING_EFFECT_STRENGTH": (0.3, 0.8, "float"),
                },
                mutation_strength=0.2,
                exploration_rate=0.15,
                learning_rate=0.1,
            ),
            "iowa_gambling_task": OptimizationStrategy(
                name="decision_making_optimization",
                description="Optimize for decision-making under uncertainty",
                parameter_ranges={
                    "BASE_LEARNING_RATE": (0.05, 0.3, "float"),
                    "EXPLORATION_PROB": (0.05, 0.2, "float"),
                    "PREFERENCE_DECAY": (0.98, 0.999, "float"),
                    "LEARNING_PHASE_TRIALS": (20, 60, "int"),
                },
                mutation_strength=0.15,
                exploration_rate=0.1,
                learning_rate=0.08,
            ),
            "stroop_effect": OptimizationStrategy(
                name="cognitive_control_optimization",
                description="Optimize for cognitive interference processing",
                parameter_ranges={
                    "CONGRUENT_RT_BASE": (500, 700, "int"),
                    "INCONGRUENT_RT_BASE": (650, 850, "int"),
                    "RT_VARIABILITY": (50, 150, "int"),
                    "CALCULATE_INTERFERENCE": (True, False, "bool"),
                },
                mutation_strength=0.1,
                exploration_rate=0.05,
                learning_rate=0.05,
            ),
            "double_dissociation": OptimizationStrategy(
                name="precision_gain_optimization",
                description="Optimize interoceptive precision and somatic gain decoupling",
                parameter_ranges={
                    "PI_I_BASELINE": (0.1, 15.0, "float"),
                    "BETA_SOM": (0.3, 2.5, "float"),
                    "ALPHA_IG": (3.0, 10.0, "float"),
                    "TAU_S": (0.2, 0.5, "float"),
                },
                mutation_strength=0.1,
                exploration_rate=0.1,
                learning_rate=0.05,
            ),
            "default": OptimizationStrategy(
                name="general_optimization",
                description="General purpose optimization strategy",
                parameter_ranges={
                    "NUM_TRIALS_CONFIG": (20, 200, "int"),
                    "INTER_TRIAL_INTERVAL_MS": (500, 2000, "int"),
                    "BASE_DETECTION_RATE": (0.3, 0.9, "float"),
                },
                mutation_strength=0.2,
                exploration_rate=0.1,
                learning_rate=0.1,
            ),
        }

    def get_strategy(self, experiment_name: str) -> OptimizationStrategy:
        """Get optimization strategy for experiment."""
        strategy = self.strategies.get(experiment_name)
        if strategy is None:
            return self.strategies["default"]
        return strategy

    def suggest_modifications(
        self,
        experiment_name: str,
        current_params: Dict[str, Any],
        performance_history: List[float],
    ) -> Dict[str, Any]:
        """Suggest parameter modifications based on performance history."""
        strategy = self.get_strategy(experiment_name)

        modifications: Dict[str, Any] = {}

        # Analyze performance trend
        if len(performance_history) >= 3:
            hist_len = len(performance_history)
            recent_vals = [
                performance_history[i] for i in range(max(0, hist_len - 3), hist_len)
            ]
            prev_vals = [
                performance_history[i]
                for i in range(max(0, hist_len - 6), max(0, hist_len - 3))
            ]

            recent_trend = (
                np.mean(recent_vals) - np.mean(prev_vals) if len(prev_vals) > 0 else 0
            )

            # Adjust exploration rate based on performance
            if recent_trend > 0:  # Improving
                exploration_rate = strategy.exploration_rate * 0.8  # Exploit more
            else:  # Not improving
                exploration_rate = strategy.exploration_rate * 1.2  # Explore more
        else:
            exploration_rate = strategy.exploration_rate

        # Suggest modifications for each parameter
        for param_name, param_config in strategy.parameter_ranges.items():
            # Handle different parameter configurations
            if len(param_config) == 3:
                min_val, max_val, param_type = param_config
            elif len(param_config) == 2:
                min_val, max_val = param_config
                param_type = "float"  # Default type
            else:
                continue  # Skip invalid configurations
            if np.random.random() < exploration_rate:
                if param_type == "float":
                    # Compute default value, handling list types gracefully
                    if isinstance(min_val, (int, float)) and isinstance(
                        max_val, (int, float)
                    ):
                        default_val = (min_val + max_val) / 2
                    else:
                        continue  # Skip parameters with invalid ranges
                    current_val = current_params.get(param_name, default_val)
                    mutation = np.random.normal(
                        0, strategy.mutation_strength * (max_val - min_val)
                    )
                    new_val = np.clip(current_val + mutation, min_val, max_val)
                    modifications[param_name] = float(new_val)

                elif param_type == "int":
                    # Compute default value, handling list types gracefully
                    if isinstance(min_val, (int, float)) and isinstance(
                        max_val, (int, float)
                    ):
                        default_val = int((min_val + max_val) / 2)
                    else:
                        continue  # Skip parameters with invalid ranges
                    current_val = current_params.get(param_name, default_val)
                    mutation = int(
                        np.random.normal(
                            0, strategy.mutation_strength * (max_val - min_val)
                        )
                    )
                    new_val = np.clip(
                        current_val + mutation, int(min_val), int(max_val)
                    )
                    modifications[param_name] = int(new_val)

                elif param_type == "bool":
                    # Flip boolean with some probability
                    if np.random.random() < 0.3:
                        current_val = current_params.get(param_name, True)
                        modifications[param_name] = not current_val

                elif param_type == "list":
                    # For list parameters, suggest different subset or ordering
                    current_list = current_params.get(param_name, min_val)
                    if isinstance(current_list, list) and len(current_list) > 1:
                        # Suggest removing or adding elements
                        if np.random.random() < 0.5 and len(current_list) > 2:
                            # Remove element
                            idx = np.random.randint(0, len(current_list))
                            new_list = current_list[:idx] + current_list[idx + 1 :]
                            modifications[param_name] = new_list
                        else:
                            # Shuffle order
                            modifications[param_name] = list(
                                np.random.permutation(current_list)
                            )

        return modifications


class RateLimiter:
    """Rate limiter for autonomous operations to prevent overwhelming resources."""

    def __init__(self, max_requests: int = 10, time_window: float = 60.0):
        """Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in time_window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: List[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        """Try to acquire a rate limit slot. Returns True if allowed, False otherwise."""
        with self._lock:
            now = time.time()
            # Remove expired requests
            self.requests = [
                req_time
                for req_time in self.requests
                if now - req_time < self.time_window
            ]

            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False

    def wait_time(self) -> float:
        """Calculate time to wait before next request is allowed."""
        with self._lock:
            if len(self.requests) < self.max_requests:
                return 0.0
            oldest = min(self.requests)
            return max(0.0, self.time_window - (time.time() - oldest))


class RequestRetryHandler:
    """Handler for requests with timeout and retry logic."""

    def __init__(
        self,
        max_retries: int = 3,
        timeout: float = 30.0,
        backoff_base: float = 1.0,
        max_backoff: float = 60.0,
    ):
        """Initialize retry handler.

        Args:
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            backoff_base: Base for exponential backoff
            max_backoff: Maximum backoff time in seconds
        """
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_base = backoff_base
        self.max_backoff = max_backoff

    def execute_with_retry(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Execute a function with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of function execution

        Raises:
            TimeoutError: If all retries are exhausted
        """
        last_error: Optional[
            Union[APGITimeoutError, subprocess.TimeoutExpired, Exception]
        ] = None
        for attempt in range(self.max_retries + 1):
            try:
                # Set timeout if function supports it
                if "timeout" in kwargs or hasattr(func, "__code__"):
                    kwargs.setdefault("timeout", self.timeout)
                return func(*args, **kwargs)
            except (APGITimeoutError, subprocess.TimeoutExpired) as e:
                last_error = e
                if attempt < self.max_retries:
                    backoff = min(self.backoff_base * (2**attempt), self.max_backoff)
                    logger.warning(
                        f"[APGI AGENT] Request timed out, retrying in {backoff:.1f}s..."
                    )
                    time.sleep(backoff)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    backoff = min(self.backoff_base * (2**attempt), self.max_backoff)
                    logger.warning(
                        f"[APGI AGENT] Request failed ({e}), retrying in {backoff:.1f}s..."
                    )
                    time.sleep(backoff)

        raise APGITimeoutError(
            f"[APGI AGENT] Request failed after {self.max_retries + 1} attempts: {last_error}"
        )


# Subprocess whitelist for safe command execution
SAFE_COMMAND_PREFIXES = (
    "git",
    "python",
    "python3",
    "pytest",
    "flake8",
    "black",
    "mypy",
)


def validate_subprocess_command(command: Union[str, List[str]]) -> bool:
    """Validate that a subprocess command is in the allowed whitelist.

    Args:
        command: Command string or list to validate

    Returns:
        True if command is allowed, False otherwise
    """
    if isinstance(command, str):
        # Split command and take first part
        cmd_parts = command.split()
        if not cmd_parts:
            return False
        executable = cmd_parts[0]
    else:  # isinstance(command, list)
        if not command:
            return False
        executable = command[0]

    # Remove path components if present
    executable_name = Path(executable).name

    return executable_name.startswith(SAFE_COMMAND_PREFIXES) or executable_name in [
        "git",
        "python",
        "python3",
    ]


def safe_subprocess_run(
    command: Union[str, List[str]],
    **kwargs: Any,
) -> subprocess.CompletedProcess:
    """Execute subprocess command with validation and timeout.

    Args:
        command: Command to execute
        **kwargs: Additional arguments for subprocess.run

    Returns:
        CompletedProcess result

    Raises:
        ValueError: If command is not in whitelist
        subprocess.TimeoutExpired: If command times out
    """
    if not validate_subprocess_command(command):
        raise ValueError(f"[APGI AGENT] Command not in whitelist: {command}")

    # Set default timeout
    kwargs.setdefault("timeout", 60)
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)

    return subprocess.run(command, **kwargs)


class AsyncGitOperations:
    """Async Git operations handler for non-blocking execution."""

    RETRYABLE_ERRORS = {
        "Connection refused",
        "Connection timed out",
        "Could not resolve host",
        "RPC failed",
        "remote: Internal error",
        "error: failed to push",
        "error: cannot lock",
    }

    MAX_RETRIES = 3
    RETRY_DELAY = 2.0

    def __init__(self, repo_path: Path, rate_limiter: Optional[RateLimiter] = None):
        """Initialize async Git operations.

        Args:
            repo_path: Path to the Git repository
            rate_limiter: Optional rate limiter for throttling operations
        """
        self.repo_path = repo_path
        self.rate_limiter = rate_limiter or RateLimiter(max_requests=20, time_window=60)
        self._executor = None

    def _is_retryable_error(self, error_msg: str) -> bool:
        """Check if an error is transient and worth retrying."""
        return any(err in error_msg for err in self.RETRYABLE_ERRORS)

    async def _run_git_command_with_retry(
        self, args: List[str], timeout: float = 30.0
    ) -> Tuple[int, str, str]:
        """Run a Git command asynchronously with retry logic.

        Args:
            args: Git command arguments (without 'git')
            timeout: Timeout in seconds

        Returns:
            Tuple of (returncode, stdout, stderr)
        """
        last_error = None

        for attempt in range(self.MAX_RETRIES):
            try:
                # Apply rate limiting
                while not self.rate_limiter.acquire():
                    wait_time = self.rate_limiter.wait_time()
                    logger.debug(
                        f"[APGI AGENT] Rate limiting Git operations, waiting {wait_time:.1f}s..."
                    )
                    await asyncio.sleep(wait_time)

                # Validate command
                command = ["git"] + args
                if not validate_subprocess_command(command):
                    raise ValueError(
                        f"[APGI AGENT] Git command not in whitelist: {args}"
                    )

                # Run command in executor to avoid blocking
                loop = asyncio.get_running_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: safe_subprocess_run(command, cwd=str(self.repo_path)),
                    ),
                    timeout=timeout,
                )

                # Check if we should retry based on error
                if result.returncode != 0:
                    error_msg = result.stderr
                    if (
                        self._is_retryable_error(error_msg)
                        and attempt < self.MAX_RETRIES - 1
                    ):
                        logger.warning(
                            f"[APGI AGENT] Retryable error on attempt {attempt + 1}/{self.MAX_RETRIES}: {error_msg}"
                        )
                        await asyncio.sleep(self.RETRY_DELAY * (attempt + 1))
                        continue

                return result.returncode, result.stdout, result.stderr

            except asyncio.TimeoutError:
                last_error = "[APGI AGENT] Git command timed out"
                logger.warning(
                    f"[APGI AGENT] Timeout on attempt {attempt + 1}/{self.MAX_RETRIES}"
                )
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_DELAY * (attempt + 1))
            except Exception as e:
                last_error = str(e)
                logger.error(f"[APGI AGENT] Git command failed: {e}")
                if attempt < self.MAX_RETRIES - 1 and self._is_retryable_error(str(e)):
                    await asyncio.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    break

        return -1, "", last_error or "[APGI AGENT] Git command failed after all retries"

    async def async_add(self, files: List[str], timeout: float = 30.0) -> bool:
        """Stage files asynchronously.

        Args:
            files: List of file paths to stage
            timeout: Timeout in seconds

        Returns:
            True if successful, False otherwise
        """
        for pattern in files:
            returncode, _, stderr = await self._run_git_command_with_retry(
                ["add", pattern], timeout=timeout
            )
            if returncode != 0:
                logger.warning(f"[APGI AGENT] Failed to stage {pattern}: {stderr}")
        return True

    async def async_commit(self, message: str, timeout: float = 30.0) -> Optional[str]:
        """Create commit asynchronously and return commit hash.

        Args:
            message: Commit message
            timeout: Timeout in seconds

        Returns:
            Commit hash if successful, None otherwise
        """
        returncode, stdout, stderr = await self._run_git_command_with_retry(
            ["commit", "-m", message], timeout=timeout
        )
        if returncode != 0:
            logger.warning(f"[APGI AGENT] Failed to commit: {stderr}")
            return None

        # Get commit hash
        returncode, stdout, _ = await self._run_git_command_with_retry(
            ["rev-parse", "HEAD"], timeout=timeout
        )
        if returncode == 0:
            return stdout.strip()
        return None

    async def async_reset(
        self, target: str = "HEAD~1", hard: bool = True, timeout: float = 30.0
    ) -> bool:
        """Reset repository asynchronously.

        Args:
            target: Target commit/ref to reset to
            timeout: Timeout in seconds
            hard: Whether to use --hard flag

        Returns:
            True if successful, False otherwise
        """
        cmd = ["reset"]
        if hard:
            cmd.append("--hard")
        cmd.append(target)

        returncode, _, stderr = await self._run_git_command_with_retry(
            cmd, timeout=timeout
        )
        if returncode != 0:
            logger.warning(f"[APGI AGENT] Failed to reset: {stderr}")
            return False
        return True


class AutonomousAgent:
    """Main autonomous agent controller for APGI optimization.

    This class orchestrates the autonomous optimization loop across multiple experiments,
    handling parameter optimization, experiment execution, and performance tracking.

    Attributes:
        research_dir (Path): Directory containing experiment files
        experiment_modules (Dict[str, Dict]): Loaded experiment modules
        git_tracker (GitPerformanceTracker): Git-based performance tracker
        optimizer (ParameterOptimizer): AI-driven parameter optimizer
        running_experiments (Set[str]): Currently running experiments
        stop_all (bool): Flag to stop all experiments

    Methods:
        load_experiments(): Load all available experiment modules
        run_experiment(): Run a single experiment with modifications
        optimize_experiment(): Run optimization loop for an experiment
        run_overnight_optimization(): Run overnight optimization
        _get_current_parameters(): Extract current parameters from run file
        _apply_modifications(): Apply parameter modifications with validation
        _extract_primary_metric(): Extract primary metric from results
        _get_metric_direction(): Get metric direction (higher/lower is better)
    """

    def __init__(
        self,
        repo_path: str = ".",
        param_change_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        """Initialize the autonomous agent.

        Args:
            repo_path (str): Path to the repository (default: ".")
            param_change_callback (Optional[Callable]): Called after each parameter
                modification with (experiment_name, modifications_dict).
        """
        self.repo_path = Path(repo_path)
        self.git_tracker = GitPerformanceTracker(repo_path, agent=self)
        self.experiment_modules = self._load_experiment_modules()
        self.running = False
        self.memory_store = MemoryStore()
        self.agent_engine = XPRAgentEngineEnhanced()
        self.optimizer = ParameterOptimizer()
        self.human_control = HumanControlLayer()
        self.checkpoint_file = Path(repo_path) / ".autonomous_agent_checkpoint.json"
        self.last_checkpoint_time = 0.0
        self.checkpoint_interval_s = 60  # Save checkpoint every 60 seconds
        self.param_change_callback = param_change_callback

        # Phase 5: Async Git Operations for non-blocking auto-loop
        self.async_git = AsyncGitOperations(self.repo_path)
        self._session_branch: Optional[str] = None

    def _save_checkpoint(
        self,
        experiment_name: str,
        iteration: int,
        current_params: dict,
        performance_history: list,
    ) -> None:
        """Save checkpoint for resuming after crash."""
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "experiment_name": experiment_name,
            "iteration": iteration,
            "current_params": current_params,
            "performance_history": performance_history,
            "best_results": {
                k: asdict(v) for k, v in self.git_tracker.best_results.items()
            },
        }
        try:
            with open(self.checkpoint_file, "w") as f:
                json.dump(checkpoint, f, indent=2)
            self.last_checkpoint_time = time.time()
            logger.debug(f"[APGI AGENT] Checkpoint saved at iteration {iteration}")
        except Exception as e:
            logger.warning(f"[APGI AGENT] Failed to save checkpoint: {e}")

    def _load_checkpoint(self) -> dict | None:
        """Load checkpoint if it exists."""
        if not self.checkpoint_file.exists():
            return None
        try:
            with open(self.checkpoint_file, "r") as f:
                checkpoint = json.load(f)
            logger.info(
                f"[APGI AGENT] Loaded checkpoint from {checkpoint.get('timestamp', 'unknown')}"
            )
            return cast(Optional[Dict[str, Any]], checkpoint)
        except Exception as e:
            logger.warning(f"[APGI AGENT] Failed to load checkpoint: {e}")
            return None

    def _clear_checkpoint(self) -> None:
        """Clear checkpoint after successful completion."""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                logger.debug("[APGI AGENT] Checkpoint cleared")
        except Exception as e:
            logger.warning(f"[APGI AGENT] Failed to clear checkpoint: {e}")

    # ... rest of the code remains the same ...
    def _load_experiment_modules(self) -> Dict[str, Any]:
        """Dynamically load experiment modules."""
        modules = {}

        # Find all prepare_*.py files in the experiments subdirectory
        prepare_files = list((self.repo_path / "experiments").glob("prepare_*.py"))

        for prepare_file in prepare_files:
            experiment_name = prepare_file.stem.replace("prepare_", "")

            try:
                # Validate module names before importing
                prepare_module_name = prepare_file.stem
                run_module_name = prepare_file.stem.replace("prepare_", "run_")

                # Strict validation of module names
                allowed_prefixes = ["prepare_", "run_"]
                if not any(
                    prepare_module_name.startswith(prefix)
                    for prefix in allowed_prefixes
                ):
                    raise ValueError(f"Invalid module name: {prepare_module_name}")
                if not run_module_name.startswith("run_"):
                    raise ValueError(f"Invalid run module name: {run_module_name}")

                # Only allow alphanumeric characters and underscores
                if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", prepare_module_name):
                    raise ValueError(
                        f"Invalid characters in module name: {prepare_module_name}"
                    )
                if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", run_module_name):
                    raise ValueError(
                        f"Invalid characters in run module name: {run_module_name}"
                    )

                # Use importlib.import_module for secure loading
                # Modules are in the experiments package, so prefix with "experiments."
                importlib.invalidate_caches()  # Clear stale module cache
                prepare_module = importlib.import_module(
                    f"experiments.{prepare_module_name}"
                )
                importlib.invalidate_caches()  # Clear stale module cache
                run_module = importlib.import_module(f"experiments.{run_module_name}")

                modules[experiment_name] = {
                    "prepare": prepare_module,
                    "run": run_module,
                    "prepare_file": str(prepare_file),
                    "run_file": str(prepare_file.parent / f"{run_module_name}.py"),
                }

                logger.info(f"[APGI AGENT] Loaded experiment: {experiment_name}")

            except Exception as e:
                logger.warning(
                    f"[APGI AGENT] Could not load experiment {experiment_name}: {e}"
                )

        return modules

    def run_experiment(
        self,
        experiment_name: str,
        modifications: Optional[Dict[str, Any]] = None,
        timeout_seconds: int = 1800,
        max_retries: int = 1,
    ) -> ExperimentResult:
        """Run a single experiment with optional modifications, timeout, and self-healing retry."""
        if experiment_name not in self.experiment_modules:
            raise ValueError(f"[APGI AGENT] Unknown experiment: {experiment_name}")

        modules = self.experiment_modules[experiment_name]

        # Apply modifications if provided
        if modifications:
            self._apply_modifications(modules["run_file"], modifications)

        # Commit modifications
        commit_hash = self.git_tracker.commit_experiment(modifications or {})

        for attempt in range(1 + max_retries):
            # Run experiment with timeout
            start_time = time.time()
            old_signal_handler = None
            signals_set = False
            _timer: Optional[threading.Timer] = None
            if sys.platform != "win32":
                # SIGALRM is Unix-only; use it for low-overhead timeout.
                try:
                    old_signal_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(timeout_seconds)
                    signals_set = True
                except (ValueError, AttributeError):
                    pass
            else:
                # Windows fallback: threading.Timer fires timeout_handler from a
                # daemon thread which raises in the main thread via signal.raise_signal
                # if available (Python 3.8+), otherwise logs a warning.
                def _win_timeout() -> None:
                    try:
                        signal.raise_signal(signal.SIGTERM)
                    except Exception:
                        logging.getLogger(__name__).warning(
                            "Experiment timed out but could not raise signal on Windows."
                        )

                _timer = threading.Timer(timeout_seconds, _win_timeout)
                _timer.daemon = True
                _timer.start()
                signals_set = True

            try:
                # Import and run experiment
                run_module = modules["run"]

                # Look for main runner class
                runner_class = None
                for attr_name in dir(run_module):
                    if "Runner" in attr_name:
                        attr = getattr(run_module, attr_name)
                        if hasattr(attr, "run_experiment"):
                            runner_class = attr
                            break

                if runner_class is None:
                    raise ValueError(
                        f"[APGI AGENT] No runner class found in {run_module}"
                    )

                # Initialize and run experiment
                runner = runner_class()
                results = runner.run_experiment()

                completion_time = time.time() - start_time

                # Extract primary metric
                primary_metric = self._extract_primary_metric(results, experiment_name)

                # Extract APGI metrics if available
                apgi_metrics = results.get("apgi_metrics", {})
                apgi_enhanced_metric = results.get("apgi_enhanced_metric")

                return ExperimentResult(
                    commit_hash=commit_hash,
                    experiment_name=experiment_name,
                    primary_metric=primary_metric,
                    apgi_metrics=apgi_metrics,
                    apgi_enhanced_metric=apgi_enhanced_metric,
                    completion_time_s=completion_time,
                    timestamp=datetime.now().isoformat(),
                    parameter_modifications=modifications or {},
                    status="success",
                )

            except APGITimeoutError:
                completion_time = time.time() - start_time
                logger.error(
                    f"[APGI AGENT] Experiment {experiment_name} timed out after {timeout_seconds} seconds"
                )
                return ExperimentResult(
                    commit_hash=commit_hash,
                    experiment_name=experiment_name,
                    primary_metric=0.0,
                    apgi_metrics={},
                    apgi_enhanced_metric=None,
                    completion_time_s=completion_time,
                    timestamp=datetime.now().isoformat(),
                    parameter_modifications=modifications or {},
                    status="timeout",
                )
            except Exception as e:
                completion_time = time.time() - start_time
                logger.error(
                    f"[APGI AGENT] Experiment {experiment_name} crashed (attempt {attempt + 1}): {str(e)}"
                )

                # Phase 2 Component: Agent Harness & Skill Chaining (Self-Healing)
                logger.info(
                    "[APGI AGENT] Executing APGI Agent Engine self-healing skill chain..."
                )
                healed = False
                try:
                    healing_chain = self.agent_engine.xpr_skill_chain(
                        initial_input={
                            "error": str(e),
                            "experiment": experiment_name,
                            "file": modules["run_file"],
                        },
                        skills_list=[
                            "xpr_job_debug",
                            "xpr_issue_fix",
                            "xpr_issue_report",
                        ],
                    )
                    if healing_chain and len(healing_chain) > 0:
                        logger.info(
                            f"[APGI AGENT] Self-Healing Report: {healing_chain[-1].result}"
                        )
                        # Check if the fix was actually applied (patch_source_code succeeded)
                        for sr in healing_chain:
                            if sr.skill_type == "issue_fix" and sr.success:
                                healed = True
                                break
                except Exception as e_engine:
                    logger.error(
                        f"[APGI AGENT] Agent Engine failed during recovery: {e_engine}"
                    )

                # If healed and we have retries left, reload module and retry
                if healed and attempt < max_retries:
                    logger.info(
                        f"[APGI AGENT] Self-healing applied fix. Retrying experiment (attempt {attempt + 2})..."
                    )
                    try:
                        importlib.invalidate_caches()
                        run_module_name = f"run_{experiment_name}"
                        modules["run"] = importlib.reload(
                            importlib.import_module(run_module_name)
                        )
                    except Exception as reload_err:
                        logger.warning(
                            f"[APGI AGENT] Failed to reload module after fix: {reload_err}"
                        )
                    continue  # Retry the experiment

                return ExperimentResult(
                    commit_hash=commit_hash,
                    experiment_name=experiment_name,
                    primary_metric=0.0,
                    apgi_metrics={},
                    apgi_enhanced_metric=None,
                    completion_time_s=completion_time,
                    timestamp=datetime.now().isoformat(),
                    parameter_modifications=modifications or {},
                    status="crash",
                )
            finally:
                # Clean up timeout mechanism
                if signals_set:
                    if sys.platform != "win32":
                        try:
                            signal.alarm(0)
                            if old_signal_handler is not None:
                                signal.signal(signal.SIGALRM, old_signal_handler)
                        except (ValueError, AttributeError):
                            pass
                    elif _timer is not None:
                        _timer.cancel()

        # Should not reach here, but safety fallback
        return ExperimentResult(
            commit_hash=commit_hash,
            experiment_name=experiment_name,
            primary_metric=0.0,
            apgi_metrics={},
            apgi_enhanced_metric=None,
            completion_time_s=0.0,
            timestamp=datetime.now().isoformat(),
            parameter_modifications=modifications or {},
            status="crash",
        )

    def _apply_modifications(
        self, run_file: str, modifications: Dict[str, Any]
    ) -> None:
        """Apply parameter modifications to run file with validation."""
        # Parameter validation whitelist - only allow known safe parameters
        ALLOWED_PARAMETERS = {
            # Numeric parameters
            "BASE_DETECTION_RATE",
            "TARGET_DETECTION_RATE",
            "DETECTION_THRESHOLD",
            "STIMULUS_DURATION",
            "INTER_STIMULUS_INTERVAL",
            "MASK_DURATION",
            "SOA_DURATION",
            "CUE_DURATION",
            "TARGET_DURATION",
            "RESPONSE_WINDOW",
            "TIMEOUT_DURATION",
            "MAX_RESPONSE_TIME",
            "SAMPLE_RATE",
            "TRIALS_PER_BLOCK",
            "NUM_BLOCKS",
            "NUM_TRIALS",
            "ACCURACY_THRESHOLD",
            "PERFORMANCE_THRESHOLD",
            "CONFIDENCE_LEVEL",
            "SIGNAL_TO_NOISE_RATIO",
            "CONTRAST_LEVEL",
            "INTENSITY_LEVEL",
            "FREQUENCY",
            "AMPLITUDE",
            "PHASE",
            "DELAY",
            "OFFSET",
            "LEARNING_RATE",
            "MOMENTUM",
            "WEIGHT_DECAY",
            "BATCH_SIZE",
            "HIDDEN_UNITS",
            "NUM_LAYERS",
            "DROPOUT_RATE",
            "REGULARIZATION",
            # Boolean parameters
            "USE_ADAPTIVE_STIMULUS",
            "USE_FEEDBACK",
            "USE_PRACTICE_TRIALS",
            "USE_RANDOMIZATION",
            "USE_CUEING",
            "USE_MASKING",
            "USE_TIMEOUT",
            "VERBOSE",
            "DEBUG",
            "SAVE_RESULTS",
            "PLOT_RESULTS",
            # String parameters (limited)
            "EXPERIMENT_MODE",
            "RESPONSE_TYPE",
            "STIMULUS_TYPE",
            "OUTPUT_FORMAT",
            "LOG_LEVEL",
            # List parameters (limited)
            "STIMULUS_LOCATIONS",
            "RESPONSE_KEYS",
            "CONDITIONS",
            "BLOCKS",
        }

        # Validate all parameters against whitelist
        valid_modifications = {}
        for param_name, param_value in modifications.items():
            if param_name in ALLOWED_PARAMETERS:
                valid_modifications[param_name] = param_value
            else:
                logger.warning(
                    f"[APGI AGENT] Skipping unauthorized parameter: {param_name}"
                )
        modifications = valid_modifications

        if not modifications:
            logger.warning("[APGI AGENT] No valid parameters to modify")
            return

        # Notify GUI of which parameters are being changed
        if self.param_change_callback is not None:
            experiment_name = Path(run_file).stem.replace("run_", "")
            self.param_change_callback(experiment_name, dict(modifications))

        with open(run_file, "r") as f:
            content = f.read()

        # -----------------------------------------------------------
        # Strategy 1: Try LLM-generated code patch (preferred)
        # -----------------------------------------------------------
        if hasattr(self.agent_engine, "llm_integration"):
            llm = self.agent_engine.llm_integration
            patch_result = llm.generate_code_patch(
                file_path=run_file,
                file_content=content,
                modifications=modifications,
            )
            if patch_result.success and patch_result.result:
                logger.info(
                    "[APGI AGENT] Applied LLM-generated code patch "
                    f"(confidence {patch_result.confidence:.2f})"
                )
                with open(run_file, "w") as f:
                    f.write(patch_result.result)
                return
            else:
                logger.info(
                    f"[APGI AGENT] LLM patch unavailable ({patch_result.error}), "
                    "falling back to regex"
                )

        # -----------------------------------------------------------
        # Strategy 2: Regex-based fallback
        # -----------------------------------------------------------
        for param_name, new_value in modifications.items():
            # Find the parameter definition and replace it
            # Use raw string to avoid ambiguous backreferences
            # Match parameter assignment and preserve only the assignment prefix
            pattern = rf"({param_name}\s*=\s*).*?(?=\n|#|$)"  # Non-greedy, stop at newline/comment
            replacement = rf"\g<1>{repr(new_value)}"  # Use named group to avoid backreference issues

            content = re.sub(pattern, replacement, content)

        # Write back
        with open(run_file, "w") as f:
            f.write(content)

    def _extract_primary_metric(
        self, results: Dict[str, Any], experiment_name: str
    ) -> float:
        """Extract primary metric from results."""
        # Known primary metrics for different experiments
        primary_metrics = {
            "attentional_blink": "blink_magnitude",
            "iowa_gambling_task": "net_score",
            "stroop_effect": "interference_effect_ms",
            "visual_search": "conjunction_present_slope",
            "change_blindness": "detection_rate",
            "masking": "masking_effect_ms",
            "binocular_rivalry": "alternation_rate",
            "posner_cueing": "cue_effect_ms",
            "simon_effect": "simon_effect_ms",
            "flanker_task": "flanker_effect_ms",
            "stop_signal": "stop_signal_reaction_time",
            "dual_n_back": "accuracy",
            "working_memory_span": "span_score",
            "time_estimation": "estimation_error_ms",
            "probabilistic_category_learning": "accuracy",
            "artificial_grammar_learning": "grammar_accuracy",
            # Add more as needed
        }

        primary_key = primary_metrics.get(experiment_name)
        if primary_key and primary_key in results:
            return float(results[primary_key])

        # Fallback: look for common metric names
        for key in ["primary_metric", "accuracy", "score", "effect", "performance"]:
            if key in results:
                return float(results[key])

        # Last resort: first numeric value
        for value in results.values():
            if isinstance(value, (int, float)):
                return float(value)

        return 0.0

    def _get_metric_direction(self, experiment_name: str) -> str:
        """Get metric direction: 'higher' or 'lower' is better."""
        # Metrics where higher values are better
        higher_is_better = {
            "iowa_gambling_task",
            "change_blindness",
            "dual_n_back",
            "working_memory_span",
            "probabilistic_category_learning",
            "artificial_grammar_learning",
            "accuracy",
            "score",
            "performance",
            "detection_rate",
            "alternation_rate",
            "span_score",
            "grammar_accuracy",
        }

        # Metrics where lower values are better (reaction times, errors)
        lower_is_better = {
            "attentional_blink",
            "stroop_effect",
            "visual_search",
            "masking",
            "posner_cueing",
            "simon_effect",
            "flanker_task",
            "stop_signal",
            "time_estimation",
            "reaction_time",
            "response_time",
            "error",
            "error_rate",
            "interference",
            "cue_effect",
            "simon_effect",
            "flanker_effect",
            "stop_signal_reaction_time",
            "estimation_error",
            "masking_effect",
        }

        if experiment_name in higher_is_better:
            return "higher"
        elif experiment_name in lower_is_better:
            return "lower"
        else:
            # Default to higher is better
            return "higher"

    def _create_session_branch(self, experiment_name: str) -> None:
        """Create a Git branch for this optimization session (USAGE.md requirement)."""
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        branch_name = f"{experiment_name}/{tag}"
        try:
            if self.git_tracker.repo is not None:
                self.git_tracker.repo.git.checkout("-b", branch_name)
                self._session_branch = branch_name
                logger.info(f"[APGI AGENT] Created session branch: {branch_name}")
            else:
                logger.warning(
                    "[APGI AGENT] No git repository available for branch creation"
                )
        except Exception as e:
            logger.warning(f"[APGI AGENT] Could not create session branch: {e}")
            self._session_branch = None

    def optimize_experiment(
        self, experiment_name: str, iterations: int = 10, resume: bool = True
    ) -> List[ExperimentResult]:
        """Run optimization loop for a single experiment with checkpointing."""
        logger.info(
            f"[APGI AGENT] Starting optimization for {experiment_name} ({iterations} iterations)"
        )

        # Phase 5 / USAGE.md: Create a session branch (git checkout -b <exp>/<tag>)
        self._create_session_branch(experiment_name)

        results = []
        performance_history: List[float] = []
        start_iteration = 0

        # Try to resume from checkpoint if enabled
        if resume:
            checkpoint = self._load_checkpoint()
            if checkpoint and checkpoint.get("experiment_name") == experiment_name:
                start_iteration = checkpoint.get("iteration", 0) + 1
                performance_history = checkpoint.get("performance_history", [])
                # Restore best results
                for exp_name, result_data in checkpoint.get("best_results", {}).items():
                    self.git_tracker.best_results[exp_name] = ExperimentResult(
                        **result_data
                    )
                logger.info(
                    f"[APGI AGENT] Resuming from iteration {start_iteration + 1}"
                )

        for iteration in range(start_iteration, iterations):
            logger.info(f"[APGI AGENT] Iteration {iteration + 1}/{iterations}")

            # Get current parameters
            current_params = self._get_current_parameters(experiment_name)

            # Phase 3 Step 1: Agent Plan Generation (/exp-plan)
            if iteration == 0:
                modifications = {}  # First run with baseline
            else:
                logger.info(
                    f"[APGI AGENT] Agent generating plan for iteration {iteration}..."
                )

                # Step 4: Activate Reading from Cognitive Memory
                past_memories = self.memory_store.retrieve_memories(
                    experiment_name=experiment_name
                )
                memory_context = [
                    f"[{m.pattern_type}] {m.content}" for m in past_memories[-10:]
                ]

                plan_result = self.agent_engine.plan_experiment(
                    task=f"Optimize {experiment_name}. Past insights: {memory_context}",
                    current_params=current_params,
                )

                # Extract JSON from LLM output
                try:
                    if plan_result.result and isinstance(plan_result.result, dict):
                        # Result is already a dict, use it directly
                        plan_data = plan_result.result
                    elif plan_result.result:
                        # Try to parse as JSON string
                        result_str = str(plan_result.result).strip()
                        # Handle single quotes (common Python dict representation)
                        result_str = result_str.replace("'", '"')
                        plan_data = json.loads(result_str)
                    else:
                        plan_data = {}
                    modifications = plan_data.get("modifications", {})
                except Exception as e:
                    logger.warning(
                        f"[APGI AGENT] Failed to parse Agent Engine JSON output: {e}. "
                        f"Result type: {type(plan_result.result)}. Fallback to empty."
                    )
                    modifications = {}

            # Phase 3 Step 2: Execution Engine
            result = self.run_experiment(experiment_name, modifications)
            results.append(result)

            delta = 0.0
            if performance_history:
                delta = result.primary_metric - performance_history[-1]

            performance_history.append(result.primary_metric)

            # Phase 1: Compile ExecutionReport and update Persistent Memory
            report = ExecutionReport(
                experiment_name=experiment_name,
                summary=f"Iteration {iteration}: Metric {result.primary_metric:.4f} (Delta: {delta:+.4f})",
                metric_deltas={experiment_name: delta},
                root_causes=(
                    ["Execution failure occurred"] if result.status != "success" else []
                ),
                suggested_fixes=(
                    ["Adjust bound limits"]
                    if result.status != "success"
                    else ["Valid parameters found"]
                ),
                confidence_score=0.9 if result.status == "success" else 0.1,
            )
            # Get the generate method if available
            llm_gen_fn = None
            if (
                hasattr(self.agent_engine, "llm_integration")
                and self.agent_engine.llm_integration
            ):
                llm_gen_fn = self.agent_engine.llm_integration.generate_text

            update_memory_from_report(
                asdict(report),
                self.memory_store,
                llm_call_fn=llm_gen_fn,
            )

            # Phase 3 Step 3: Analyze & Report (enriched context)
            # Register XPR skills if not already registered
            if "xpr_issue_report" not in self.agent_engine.skills:
                register_xpr_skills(self.agent_engine)

            analysis_res = self.agent_engine.execute_skill(
                "xpr_issue_report",
                {
                    "error": str(
                        getattr(result, "error", "No error information available")
                    ),
                    "experiment": experiment_name,
                    "metrics": {"primary_metric": result.primary_metric},
                },
            )
            confidence = getattr(
                analysis_res,
                "confidence",
                (
                    getattr(analysis_res.metadata, "get", lambda x, y=0: 0)(
                        "confidence", 0.0
                    )
                    if hasattr(analysis_res, "metadata")
                    else 0.0
                ),
            )

            # Phase 3 Step 4: Human Review Checkpoint Integration
            if confidence > 0.5:
                logger.info(
                    "High confidence improvement detected - checking for human review requirements"
                )

                # Check if human review is needed based on configuration
                config = self.human_control.get_configuration_summary()

                # If human interaction mode is not autonomous, pause for review
                if (
                    config.get("configured", False)
                    or config.get("interaction_mode") != "autonomous"
                ):
                    logger.info("Human review required - pausing for human oversight")

                    # Prepare result for human review
                    review_data = {
                        "experiment_id": f"{experiment_name}_iter_{iteration}",
                        "hypothesis_id": f"auto_opt_{experiment_name}",
                        "metrics": result.metrics if hasattr(result, "metrics") else {},
                        "outcomes": (
                            result.outcomes if hasattr(result, "outcomes") else {}
                        ),
                        "analysis": (
                            analysis_res.analysis
                            if hasattr(analysis_res, "analysis")
                            else ""
                        ),
                        "confidence": confidence,
                    }

                    # Request human review
                    review_result = self.human_control.review(review_data)

                    # Apply human decision
                    if review_result.decision.value == "approve":
                        logger.info("Human approved - committing improvements")
                        if result.status == "success":
                            self.git_tracker.best_results[experiment_name] = result
                            self.git_tracker.save_results(self.git_tracker.best_results)

                        # Commit with human approval
                        try:
                            loop = asyncio.new_event_loop()
                            loop.run_until_complete(
                                self.async_git.async_commit(
                                    f"[APGI AGENT] Human approved: {experiment_name} iter {iteration}"
                                )
                            )
                            loop.close()
                        except Exception as git_err:
                            logger.warning(
                                f"Failed to commit human approval: {git_err}"
                            )

                    elif review_result.decision.value == "modify":
                        logger.info(
                            "Human requested modifications - generating parameter changes"
                        )
                        # Apply modifications and restart optimization
                        if review_result.modifications:
                            mod_params = self.optimizer.suggest_modifications(
                                experiment_name,
                                {"modifications": review_result.modifications},
                                performance_history,
                            )
                            modifications.update(mod_params)

                        # Continue to next iteration with modifications
                        continue

                    elif review_result.decision.value == "reject":
                        logger.info("Human rejected improvements - rolling back")
                        # Rollback to previous best parameters
                        if experiment_name in self.git_tracker.best_results:
                            rollback_params = self.git_tracker.best_results[
                                experiment_name
                            ].parameter_modifications
                            modifications.update(rollback_params)

                        # Continue with rollback parameters
                        continue

                # If autonomous mode or no human review needed, continue with normal flow
                if confidence > 0.5:
                    logger.info(
                        f"[APGI AGENT] Improvement verified by Agent! New best: {result.primary_metric:.4f}"
                    )
                    if result.status == "success":
                        self.git_tracker.best_results[experiment_name] = result
                        self.git_tracker.save_results(self.git_tracker.best_results)

                    # Phase 5: Use async Git for high-confidence commits
                    try:
                        loop = asyncio.new_event_loop()
                        loop.run_until_complete(
                            self.async_git.async_add(["run_*.py", "*.json"])
                        )
                        loop.run_until_complete(
                            self.async_git.async_commit(
                                f"[APGI AGENT] {experiment_name} iter {iteration}: "
                                f"metric {result.primary_metric:.4f} (Δ{delta:+.4f})"
                            )
                        )
                        loop.close()
                        logger.info(
                            "[APGI AGENT] AsyncGitOperations: committed improvement."
                        )
                    except Exception as git_err:
                        logger.warning(
                            f"Async git commit failed (non-fatal): {git_err}"
                        )
            else:
                logger.info(
                    f"No improvement (confidence {confidence:.2f}). Rolling back..."
                )
                self.git_tracker.rollback_experiment()

            # Phase 3 Step 4: Guardrails Evaluation
            if confidence < 0.2:
                logger.warning(
                    f"Guardrail triggered: Agent confidence {confidence} is too low. Escalating to human..."
                )
                break

            # Phase 3 Step 4b: Metric Regression Detection
            if len(performance_history) >= 3:
                recent_3 = performance_history[-3:]
                direction = self._get_metric_direction(experiment_name)
                if direction == "higher":
                    regressing = all(
                        recent_3[i] <= recent_3[i - 1] for i in range(1, len(recent_3))
                    )
                else:
                    regressing = all(
                        recent_3[i] >= recent_3[i - 1] for i in range(1, len(recent_3))
                    )
                if regressing:
                    logger.warning(
                        f"[APGI AGENT] Metric regression detected over last 3 iterations "
                        f"({recent_3}). Escalating to human review."
                    )
                    # Rollback to best known state
                    self.git_tracker.rollback_experiment()
                    break

            # Phase 3 Step 4c: Safety Violation Checks
            if result.primary_metric is not None:
                import math as _math

                if _math.isnan(result.primary_metric) or _math.isinf(
                    result.primary_metric
                ):
                    logger.error(
                        f"[APGI AGENT] Safety violation: metric is {result.primary_metric}. Halting."
                    )
                    self.git_tracker.rollback_experiment()
                    break
                # Check for extreme outlier (>10x best known metric)
                best_metric = self.git_tracker.get_best_metric(experiment_name)
                if best_metric is not None and best_metric != 0.0:
                    ratio = abs(result.primary_metric / best_metric)
                    if ratio > 10.0 or ratio < 0.01:
                        logger.warning(
                            f"[APGI AGENT] Safety violation: metric {result.primary_metric} "
                            f"is an extreme outlier vs best {best_metric}. Halting."
                        )
                        self.git_tracker.rollback_experiment()
                        break

            # Save checkpoint periodically
            if time.time() - self.last_checkpoint_time > self.checkpoint_interval_s:
                self._save_checkpoint(
                    experiment_name, iteration, current_params, performance_history
                )

            # Brief pause to avoid overwhelming the system
            time.sleep(1)

        # Clear checkpoint on successful completion
        self._clear_checkpoint()
        return results

    def get_current_parameters(self, experiment_name: str) -> Dict[str, Any]:
        """Public wrapper for _get_current_parameters."""
        return self._get_current_parameters(experiment_name)

    def apply_modifications(self, run_file: str, modifications: Dict[str, Any]) -> None:
        """Public wrapper for _apply_modifications."""
        return self._apply_modifications(run_file, modifications)

    def extract_primary_metric(
        self, results: Dict[str, Any], experiment_name: str = "unknown"
    ) -> float:
        """Public wrapper for _extract_primary_metric."""
        return self._extract_primary_metric(results, experiment_name)

    def get_metric_direction(self, experiment_name: str) -> str:
        """Public wrapper for _get_metric_direction."""
        return self._get_metric_direction(experiment_name)

    def _get_current_parameters(self, experiment_name: str) -> Dict[str, Any]:
        """Get current parameter values from run file."""
        if experiment_name not in self.experiment_modules:
            return {}

        run_file = self.experiment_modules[experiment_name]["run_file"]

        try:
            # Parse the Python file to extract parameter values
            with open(run_file, "r") as f:
                content = f.read()

            # Commented-out assignments are not current values
            content = "\n".join(
                line
                for line in content.splitlines()
                if not line.lstrip().startswith("#")
            )

            parameters: Dict[str, Any] = {}

            # Common parameter patterns to extract
            numeric_patterns = [
                r"([A-Z_][A-Z0-9_]*)\s*=\s*(-?[0-9]*\.?[0-9]+)\b",
            ]
            bool_patterns = [
                r"([A-Z_][A-Z0-9_]*)\s*=\s*(True|False)\b",
            ]
            string_patterns = [
                r'([A-Z_][A-Z0-9_]*)\s*=\s*["\']([^"\']*)["\']',
            ]
            list_patterns = [
                r"([A-Z_][A-Z0-9_]*)\s*=\s*\[([^\]]*)\]",
            ]

            all_patterns = (
                numeric_patterns + bool_patterns + string_patterns + list_patterns
            )

            for pattern in all_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if isinstance(match, tuple) and len(match) >= 2:
                        param_name = str(match[0])
                        param_value = str(match[1])

                        # Skip if this looks like a reserved word
                        if param_name in [
                            "def",
                            "class",
                            "import",
                            "from",
                            "if",
                            "for",
                            "while",
                        ]:
                            continue

                        try:
                            if param_value in ["True", "False"]:
                                parameters[param_name] = param_value == "True"
                            elif re.match(r"^-?[0-9]*\.?[0-9]+$", param_value):
                                if "." in param_value:
                                    parameters[param_name] = float(param_value)
                                else:
                                    parameters[param_name] = int(param_value)
                            else:
                                # Try to evaluate safely for lists/strings or keep as string
                                try:
                                    parameters[param_name] = ast.literal_eval(
                                        param_value
                                    )
                                except (ValueError, SyntaxError):
                                    parameters[param_name] = param_value
                        except (ValueError, SyntaxError):
                            parameters[param_name] = param_value

            return parameters

        except Exception as e:
            logger.warning(f"Failed to extract parameters from {run_file}: {e}")
            return {}

    def run_overnight_optimization(self, max_hours: float = 8.0) -> None:
        """Run overnight optimization across all experiments."""
        logger.info(f"Starting overnight optimization ({max_hours} hours)")

        start_time = time.time()
        end_time = start_time + (max_hours * 3600)

        experiment_names = list(self.experiment_modules.keys())

        while time.time() < end_time:
            # Pick random experiment to optimize
            experiment_name = np.random.choice(experiment_names)

            # Run more iterations of optimization for overnight mode (increased from 3 to 10)
            try:
                self.optimize_experiment(experiment_name, iterations=10)
            except Exception as e:
                logger.error(f"Error optimizing {experiment_name}: {e}")

            # Check if we have time for another round
            if time.time() + 600 > end_time:  # Need at least 10 minutes
                break

        logger.info("Overnight optimization completed")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Autonomous APGI Agent")
    parser.add_argument(
        "--experiment", type=str, help="Specific experiment to optimize"
    )
    parser.add_argument(
        "--all-experiments", action="store_true", help="Optimize all experiments"
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Iterations per experiment"
    )
    parser.add_argument(
        "--overnight", action="store_true", help="Run overnight optimization"
    )
    parser.add_argument(
        "--hours", type=float, default=8.0, help="Hours for overnight mode"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Non-interactive mode for CI/CD (uses defaults)",
    )

    args = parser.parse_args()

    # In auto mode, default to all-experiments if nothing else specified
    if args.auto and not (args.experiment or args.all_experiments or args.overnight):
        args.all_experiments = True
        logger.info("Auto mode: defaulting to --all-experiments")

    # Validate arguments
    if not (args.experiment or args.all_experiments or args.overnight):
        parser.error(
            "Must specify --experiment, --all-experiments, --overnight, or --auto"
        )

    agent = AutonomousAgent()

    if args.overnight:
        agent.run_overnight_optimization(args.hours)
    elif args.all_experiments:
        for experiment_name in agent.experiment_modules.keys():
            try:
                agent.optimize_experiment(experiment_name, args.iterations)
            except Exception as e:
                logger.error(f"Failed to optimize {experiment_name}: {e}")
    elif args.experiment:
        agent.optimize_experiment(args.experiment, args.iterations)


if __name__ == "__main__":
    main()
