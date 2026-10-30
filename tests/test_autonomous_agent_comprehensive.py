"""
Comprehensive test suite for autonomous_agent.py to achieve ≥90% coverage.

This file tests the complex code paths and edge cases not covered in the basic test suite.
"""

import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import git
import pytest

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autonomous_agent import (
    APGITimeoutError,
    AsyncGitOperations,
    AutonomousAgent,
    ExperimentResult,
    GitPerformanceTracker,
    OptimizationStrategy,
    ParameterOptimizer,
    RateLimiter,
    RequestRetryHandler,
    safe_subprocess_run,
    timeout_handler,
    validate_subprocess_command,
)


class TestUtilityFunctions:
    """Test utility functions and helper classes."""

    def test_timeout_handler(self):
        """Test the timeout signal handler."""
        with pytest.raises(APGITimeoutError, match="Experiment execution timed out"):
            timeout_handler(signal.SIGALRM, None)

    def test_validate_subprocess_command_valid(self):
        """Test command validation with allowed commands."""
        # Test allowed commands
        allowed_commands = [
            ["git", "status"],
            ["python", "script.py"],
            ["pytest", "test_file.py"],
        ]

        for cmd in allowed_commands:
            assert (
                validate_subprocess_command(cmd) is True
            )  # nosec: B101 - Test assertion

    def test_validate_subprocess_command_invalid(self):
        """Test command validation with disallowed commands."""
        # Test disallowed commands
        disallowed_commands = [
            ["rm", "-rf", "/"],
            ["sudo", "rm", "-rf", "/"],
            ["chmod", "777", "/etc/passwd"],
            ["mv", "/etc/passwd", "/tmp"],
            ["cp", "/etc/shadow", "/tmp"],
            # Only the whitelisted tooling may run, not general shell utilities
            ["ls", "-la"],
            ["echo", "test"],
        ]

        for cmd in disallowed_commands:
            assert (
                validate_subprocess_command(cmd) is False
            )  # nosec: B101 - Test assertion

    def test_safe_subprocess_run(self):
        """Test safe subprocess execution."""
        # Test successful command
        result = safe_subprocess_run([sys.executable, "-c", "print('test')"])
        assert result.returncode == 0  # nosec: B101 - Test assertion
        assert "test" in result.stdout  # nosec: B101 - Test assertion

    def test_safe_subprocess_run_rejects_unlisted_command(self):
        """Commands outside the whitelist are refused before they run."""
        with pytest.raises(ValueError, match="not in whitelist"):
            safe_subprocess_run(["echo", "test"])

    def test_safe_subprocess_run_with_kwargs(self):
        """Test safe subprocess with additional arguments."""
        result = safe_subprocess_run(
            [sys.executable, "-c", "import os; print(os.getcwd())"],
            cwd=tempfile.gettempdir(),
        )
        assert result.returncode == 0  # nosec: B101 - Test assertion
        # Default timeout is set by the function


class TestRateLimiter:
    """Test the RateLimiter class."""

    def test_rate_limiter_basic(self):
        """Test basic rate limiting functionality."""
        limiter = RateLimiter(max_requests=3, time_window=1.0)

        # Should allow first few requests
        assert limiter.acquire() is True  # nosec: B101 - Test assertion
        assert limiter.acquire() is True  # nosec: B101 - Test assertion
        assert limiter.acquire() is True  # nosec: B101 - Test assertion

        # Should reject when limit exceeded
        assert limiter.acquire() is False  # nosec: B101 - Test assertion

    def test_rate_limiter_time_window(self):
        """Test rate limiter time window reset."""
        limiter = RateLimiter(max_requests=2, time_window=0.1)

        # Use up limit
        assert limiter.acquire() is True  # nosec: B101 - Test assertion
        assert limiter.acquire() is True  # nosec: B101 - Test assertion
        assert limiter.acquire() is False  # nosec: B101 - Test assertion

        # Wait for time window to pass
        time.sleep(0.2)

        # Should allow requests again
        assert limiter.acquire() is True  # nosec: B101 - Test assertion

    def test_rate_limiter_wait_time(self):
        """Test wait time calculation."""
        limiter = RateLimiter(max_requests=2, time_window=1.0)

        # Use up limit
        limiter.acquire()
        limiter.acquire()

        # Should calculate wait time
        wait_time = limiter.wait_time()
        assert 0 < wait_time <= 1.0  # nosec: B101 - Test assertion

    def test_rate_limiter_concurrent_access(self):
        """Test rate limiter with concurrent access."""
        import threading

        limiter = RateLimiter(max_requests=5, time_window=1.0)
        results = []

        def worker():
            results.append(limiter.acquire())

        threads = [threading.Thread(target=worker) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should have some successes and some failures
        assert sum(results) <= 5  # nosec: B101 - Test assertion


class TestRequestRetryHandler:
    """Test the RequestRetryHandler class."""

    def test_retry_handler_initialization(self):
        """Test retry handler initialization."""
        handler = RequestRetryHandler(
            max_retries=5, timeout=60.0, backoff_base=2.0, max_backoff=120.0
        )

        assert handler.max_retries == 5  # nosec: B101 - Test assertion
        assert handler.timeout == 60.0  # nosec: B101 - Test assertion
        assert handler.backoff_base == 2.0  # nosec: B101 - Test assertion
        assert handler.max_backoff == 120.0  # nosec: B101 - Test assertion

    @patch("time.sleep")
    def test_retry_handler_success_on_first_try(self, mock_sleep):
        """Test retry handler when function succeeds on first try."""
        handler = RequestRetryHandler()
        mock_func = Mock(return_value="success")

        result = handler.execute_with_retry(mock_func, arg1="test")

        assert result == "success"  # nosec: B101 - Test assertion
        mock_func.assert_called_once_with(arg1="test")
        mock_sleep.assert_not_called()

    @patch("time.sleep")
    def test_retry_handler_success_after_retries(self, mock_sleep):
        """Test retry handler when function succeeds after retries."""
        handler = RequestRetryHandler(max_retries=3)
        mock_func = Mock(side_effect=[Exception("fail"), Exception("fail"), "success"])

        result = handler.execute_with_retry(mock_func)

        assert result == "success"  # nosec: B101 - Test assertion
        assert mock_func.call_count == 3  # nosec: B101 - Test assertion
        assert mock_sleep.call_count == 2  # nosec: B101 - Test assertion

    @patch("time.sleep")
    def test_retry_handler_exhausted_retries(self, mock_sleep):
        """Test retry handler when retries are exhausted."""
        handler = RequestRetryHandler(max_retries=2)
        mock_func = Mock(side_effect=Exception("always fails"))

        with pytest.raises(Exception, match="always fails"):
            handler.execute_with_retry(mock_func)

        assert (
            mock_func.call_count == 3
        )  # Initial try + 2 retries  # nosec: B101 - Test assertion
        assert mock_sleep.call_count == 2  # nosec: B101 - Test assertion

    @patch("time.sleep")
    def test_retry_handler_backoff_calculation(self, mock_sleep):
        """Test exponential backoff calculation."""
        handler = RequestRetryHandler(max_retries=3, backoff_base=2.0)
        mock_func = Mock(side_effect=[Exception("fail")] * 4)

        with pytest.raises(Exception):
            handler.execute_with_retry(mock_func)

        # Backoff is backoff_base * 2**attempt: 2.0 * (1, 2, 4) seconds
        expected_calls = [call(2.0), call(4.0), call(8.0)]
        mock_sleep.assert_has_calls(expected_calls)


class TestAsyncGitOperations:
    """Test the AsyncGitOperations class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)

        # Initialize git repo
        self.repo = git.Repo.init(self.repo_path)
        self.repo.config_writer().set_value("user", "name", "Test User").release()
        self.repo.config_writer().set_value(
            "user", "email", "test@example.com"
        ).release()

        # Create initial commit
        (self.repo_path / "test.txt").write_text("Initial content")
        self.repo.index.add(["test.txt"])
        self.repo.index.commit("Initial commit")

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_async_git_initialization(self):
        """Test AsyncGitOperations initialization."""
        rate_limiter = RateLimiter()
        async_git = AsyncGitOperations(self.repo_path, rate_limiter)

        assert async_git.repo_path == self.repo_path  # nosec: B101 - Test assertion
        assert async_git.rate_limiter == rate_limiter  # nosec: B101 - Test assertion

    @pytest.mark.asyncio
    async def test_async_git_command_success(self):
        """Test successful async git command."""
        rate_limiter = RateLimiter(max_requests=100, time_window=1.0)
        async_git = AsyncGitOperations(self.repo_path, rate_limiter)

        returncode, stdout, stderr = await async_git._run_git_command_with_retry(
            ["status"]
        )

        assert returncode == 0  # nosec: B101 - Test assertion
        assert (
            "On branch" in stdout or "nothing to commit" in stdout
        )  # nosec: B101 - Test assertion

    @pytest.mark.asyncio
    async def test_async_git_command_timeout(self):
        """Test async git command timeout."""
        rate_limiter = RateLimiter(max_requests=100, time_window=1.0)
        async_git = AsyncGitOperations(self.repo_path, rate_limiter)

        # Use very short timeout
        returncode, stdout, stderr = await async_git._run_git_command_with_retry(
            ["log"], timeout=0.001
        )

        assert returncode == -1  # nosec: B101 - Test assertion
        assert "timed out" in stderr  # nosec: B101 - Test assertion

    @pytest.mark.asyncio
    async def test_async_git_command_invalid(self):
        """Test async git command with invalid command."""
        rate_limiter = RateLimiter(max_requests=100, time_window=1.0)
        async_git = AsyncGitOperations(self.repo_path, rate_limiter)

        returncode, stdout, stderr = await async_git._run_git_command_with_retry(
            ["invalid_command"]
        )

        assert returncode != 0  # nosec: B101 - Test assertion
        assert (
            "not a git command" in stderr.lower() or "unknown" in stderr.lower()
        )  # nosec: B101 - Test assertion

    @pytest.mark.asyncio
    async def test_async_add_files(self):
        """Test async git add operation."""
        rate_limiter = RateLimiter(max_requests=100, time_window=1.0)
        async_git = AsyncGitOperations(self.repo_path, rate_limiter)

        # Create a new file
        (self.repo_path / "new_file.txt").write_text("New content")

        # Add the file
        result = await async_git.async_add(["new_file.txt"])

        assert result is True  # nosec: B101 - Test assertion

    @pytest.mark.asyncio
    async def test_async_commit(self):
        """Test async git commit operation."""
        rate_limiter = RateLimiter(max_requests=100, time_window=1.0)
        async_git = AsyncGitOperations(self.repo_path, rate_limiter)

        # Create and add a file
        (self.repo_path / "commit_test.txt").write_text("Commit test")
        await async_git.async_add(["commit_test.txt"])

        # Commit
        commit_hash = await async_git.async_commit("Test commit")

        assert commit_hash is not None  # nosec: B101 - Test assertion
        assert (
            len(commit_hash) == 40
        )  # Git commit hash length  # nosec: B101 - Test assertion

    @pytest.mark.asyncio
    async def test_async_commit_no_changes(self):
        """Test async commit with no changes."""
        rate_limiter = RateLimiter(max_requests=100, time_window=1.0)
        async_git = AsyncGitOperations(self.repo_path, rate_limiter)

        # Try to commit without any staged changes
        commit_hash = await async_git.async_commit("No changes commit")

        assert commit_hash is None  # nosec: B101 - Test assertion

    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self):
        """Test rate limiting integration with async git operations."""
        # Create rate limiter with very low limit
        rate_limiter = RateLimiter(max_requests=1, time_window=0.1)
        async_git = AsyncGitOperations(self.repo_path, rate_limiter)

        # First request should succeed
        start_time = time.time()
        result1 = await async_git._run_git_command_with_retry(["status"])
        assert result1[0] == 0  # nosec: B101 - Test assertion

        # Second request should be rate limited and wait for the window that
        # opened with the first request (part of it passes during that command)
        result2 = await async_git._run_git_command_with_retry(["status"])
        end_time = time.time()

        assert result2[0] == 0  # nosec: B101 - Test assertion
        assert (
            end_time - start_time >= 0.1
        )  # Should have waited  # nosec: B101 - Test assertion


class TestParameterOptimizerComprehensive:
    """Comprehensive tests for ParameterOptimizer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = ParameterOptimizer()

    def test_all_strategy_initialization(self):
        """Test that all expected strategies are initialized."""
        expected_strategies = [
            "attentional_blink",
            "iowa_gambling_task",
            "stroop_effect",
            "double_dissociation",
        ]

        for strategy_name in expected_strategies:
            assert (
                strategy_name in self.optimizer.strategies
            )  # nosec: B101 - Test assertion
            strategy = self.optimizer.strategies[strategy_name]
            assert hasattr(strategy, "name")  # nosec: B101 - Test assertion
            assert hasattr(strategy, "parameter_ranges")  # nosec: B101 - Test assertion
            assert hasattr(
                strategy, "mutation_strength"
            )  # nosec: B101 - Test assertion

    def test_parameter_modification_float(self):
        """Test float parameter modification."""
        current_params = {"BASE_DETECTION_RATE": 0.5}
        modifications = self.optimizer.suggest_modifications(
            "attentional_blink", current_params, []
        )

        if "BASE_DETECTION_RATE" in modifications:
            value = modifications["BASE_DETECTION_RATE"]
            assert isinstance(value, float)  # nosec: B101 - Test assertion
            assert (
                0.1 <= value <= 0.9
            )  # Should be within valid range  # nosec: B101 - Test assertion

    def test_parameter_modification_int(self):
        """Test integer parameter modification."""
        current_params = {"NUM_TRIALS": 100}
        modifications = self.optimizer.suggest_modifications(
            "attentional_blink", current_params, []
        )

        if "NUM_TRIALS" in modifications:
            value = modifications["NUM_TRIALS"]
            assert isinstance(value, int)  # nosec: B101 - Test assertion
            assert (
                20 <= value <= 200
            )  # Should be within valid range  # nosec: B101 - Test assertion

    def test_parameter_modification_bool(self):
        """Test boolean parameter modification."""
        current_params = {"USE_FEEDBACK": True}
        modifications = self.optimizer.suggest_modifications(
            "attentional_blink", current_params, []
        )

        if "USE_FEEDBACK" in modifications:
            assert isinstance(
                modifications["USE_FEEDBACK"], bool
            )  # nosec: B101 - Test assertion

    def test_parameter_modification_list(self):
        """Test list parameter modification."""
        current_params = {"SOA_VALUES": [10, 20, 30, 50, 80, 100, 150, 200]}
        modifications = self.optimizer.suggest_modifications(
            "attentional_blink", current_params, []
        )

        if "SOA_VALUES" in modifications:
            value = modifications["SOA_VALUES"]
            assert isinstance(value, list)  # nosec: B101 - Test assertion
            assert len(value) > 0  # nosec: B101 - Test assertion

    def test_strategy_specific_parameters(self):
        """Test that strategies have their specific parameters."""
        # Test attentional blink strategy
        strategy = self.optimizer.strategies["attentional_blink"]
        expected_params = [
            "SOA_VALUES_TO_TEST",
            "BASE_DETECTION_RATE",
            "MASKING_EFFECT_STRENGTH",
        ]

        for param in expected_params:
            assert param in strategy.parameter_ranges  # nosec: B101 - Test assertion

        # Test Iowa gambling task strategy
        strategy = self.optimizer.strategies["iowa_gambling_task"]
        expected_params = ["BASE_LEARNING_RATE", "EXPLORATION_PROB", "PREFERENCE_DECAY"]

        for param in expected_params:
            assert param in strategy.parameter_ranges  # nosec: B101 - Test assertion

    def test_mutation_strength_application(self):
        """Test that mutation strength affects modification magnitude."""
        # Create a custom strategy with high mutation strength
        high_mutation_strategy = OptimizationStrategy(
            name="high_mutation",
            description="High mutation strength test",
            parameter_ranges={"TEST_PARAM": (0.0, 1.0, "float")},
            mutation_strength=0.8,  # High mutation
            exploration_rate=0.5,
            learning_rate=0.1,
        )

        self.optimizer.strategies["test"] = high_mutation_strategy

        current_params = {"TEST_PARAM": 0.5}
        modifications = self.optimizer.suggest_modifications("test", current_params, [])

        if "TEST_PARAM" in modifications:
            # With high mutation strength, should move significantly from 0.5
            value = modifications["TEST_PARAM"]
            assert (
                abs(value - 0.5) > 0.1
            )  # Should be significantly different  # nosec: B101 - Test assertion

    def test_exploration_rate_effect(self):
        """Test that exploration rate affects modification frequency."""
        # Test with low exploration rate
        low_explore_strategy = OptimizationStrategy(
            name="low_explore",
            description="Low exploration test",
            parameter_ranges={"TEST_PARAM": (0.0, 1.0, "float")},
            mutation_strength=0.2,
            exploration_rate=0.01,  # Very low exploration
            learning_rate=0.1,
        )

        self.optimizer.strategies["test"] = low_explore_strategy

        current_params = {"TEST_PARAM": 0.5}

        # Run multiple times to see exploration effect
        modification_count = 0
        for _ in range(100):
            modifications = self.optimizer.suggest_modifications(
                "test", current_params, []
            )
            if "TEST_PARAM" in modifications:
                modification_count += 1

        # With low exploration rate, should modify less frequently
        assert (
            modification_count < 20
        )  # Should be much less than 100  # nosec: B101 - Test assertion


class TestGitPerformanceTrackerComprehensive:
    """Comprehensive tests for GitPerformanceTracker."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)

        # Initialize git repo
        self.repo = git.Repo.init(self.repo_path)
        self.repo.config_writer().set_value("user", "name", "Test User").release()
        self.repo.config_writer().set_value(
            "user", "email", "test@example.com"
        ).release()

        # Create initial commit
        (self.repo_path / "test.txt").write_text("Initial content")
        self.repo.index.add(["test.txt"])
        self.repo.index.commit("Initial commit")

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_best_results_corrupted_file(self):
        """Test loading best results from corrupted JSON file."""
        tracker = GitPerformanceTracker(str(self.repo_path))

        # Create corrupted JSON file
        tracker.results_file.write_text("invalid json content")

        # Should handle corruption gracefully
        results = tracker._load_best_results()
        assert results == {}  # nosec: B101 - Test assertion

    def test_save_results_permission_error(self):
        """Test saving results with permission error."""
        tracker = GitPerformanceTracker(str(self.repo_path))

        # Make results file read-only
        tracker.results_file.touch()
        tracker.results_file.chmod(0o444)

        # Should handle permission error gracefully
        result = ExperimentResult(
            commit_hash="abc123",
            experiment_name="test",
            primary_metric=0.5,
            apgi_metrics={},
            apgi_enhanced_metric=None,
            completion_time_s=1.0,
            timestamp="2026-01-01T00:00:00",
            parameter_modifications={},
            status="success",
        )

        # Should not raise exception
        tracker.save_results({"test": result})

    def test_commit_experiment_no_files(self):
        """Test committing when no files match patterns."""
        tracker = GitPerformanceTracker(str(self.repo_path))

        modifications = {"TEST_PARAM": 0.5}
        commit_hash = tracker.commit_experiment(modifications)

        # Should return "no_changes" or "error" but not crash
        assert commit_hash in ["no_changes", "error"]  # nosec: B101 - Test assertion

    def test_commit_experiment_git_error(self):
        """Test committing when git command fails."""
        tracker = GitPerformanceTracker(str(self.repo_path))

        # Per-pattern staging failures are tolerated by design; a failing commit
        # is the error path. commit_experiment opens its own git.Repo.
        with patch("autonomous_agent.git.Repo") as mock_repo_class:
            mock_repo = mock_repo_class.return_value
            mock_repo.is_dirty.return_value = True
            mock_repo.index.commit.side_effect = git.exc.GitCommandError(
                "commit", "git error"
            )
            modifications = {"TEST_PARAM": 0.5}
            commit_hash = tracker.commit_experiment(modifications)

        assert commit_hash == "error"  # nosec: B101 - Test assertion

    def test_commit_experiment_tolerates_staging_errors(self):
        """A pattern that cannot be staged does not abort the commit."""
        tracker = GitPerformanceTracker(str(self.repo_path))

        with patch("autonomous_agent.git.Repo") as mock_repo_class:
            mock_repo = mock_repo_class.return_value
            mock_repo.index.add.side_effect = git.exc.GitCommandError("add", "none")
            mock_repo.is_dirty.return_value = False

            assert tracker.commit_experiment({}) == "no_changes"  # nosec: B101

    def test_rollback_experiment_success(self):
        """Test successful rollback."""
        tracker = GitPerformanceTracker(str(self.repo_path))

        # Create a new commit first
        (self.repo_path / "rollback_test.txt").write_text("Test content")
        self.repo.index.add(["rollback_test.txt"])
        new_commit = self.repo.index.commit("Test commit")

        # Rollback
        result = tracker.rollback_experiment()

        assert result is True  # nosec: B101 - Test assertion
        assert (
            self.repo.head.commit.hexsha != new_commit.hexsha
        )  # nosec: B101 - Test assertion

    def test_rollback_experiment_failure(self):
        """Test rollback failure."""
        tracker = GitPerformanceTracker(str(self.repo_path))

        # rollback_experiment opens its own git.Repo, so fail the reset there
        with patch("autonomous_agent.git.Repo") as mock_repo_class:
            mock_repo_class.return_value.git.reset.side_effect = Exception("git error")
            result = tracker.rollback_experiment()

        assert result is False  # nosec: B101 - Test assertion

    def test_is_improvement_with_agent_direction(self):
        """Test improvement check with agent metric direction."""
        tracker = GitPerformanceTracker(str(self.repo_path))

        # Mock agent with specific direction
        mock_agent = Mock()
        mock_agent._get_metric_direction.return_value = "higher"
        tracker.agent = mock_agent

        # Add existing result
        existing_result = ExperimentResult(
            commit_hash="abc123",
            experiment_name="test",
            primary_metric=0.8,
            apgi_metrics={},
            apgi_enhanced_metric=None,
            completion_time_s=1.0,
            timestamp="2026-01-01T00:00:00",
            parameter_modifications={},
            status="success",
        )
        tracker.best_results["test"] = existing_result

        # Test improvements
        assert (
            tracker.is_improvement("test", 0.9) is True
        )  # Higher is better  # nosec: B101 - Test assertion
        assert (
            tracker.is_improvement("test", 0.7) is False
        )  # Lower is not better  # nosec: B101 - Test assertion

    def test_is_improvement_lower_better(self):
        """Test improvement check when lower is better."""
        tracker = GitPerformanceTracker(str(self.repo_path))

        # Mock agent with lower direction
        mock_agent = Mock()
        mock_agent._get_metric_direction.return_value = "lower"
        tracker.agent = mock_agent

        # Add existing result
        existing_result = ExperimentResult(
            commit_hash="abc123",
            experiment_name="test",
            primary_metric=100.0,
            apgi_metrics={},
            apgi_enhanced_metric=None,
            completion_time_s=1.0,
            timestamp="2026-01-01T00:00:00",
            parameter_modifications={},
            status="success",
        )
        tracker.best_results["test"] = existing_result

        # Test improvements
        assert (
            tracker.is_improvement("test", 90.0) is True
        )  # Lower is better  # nosec: B101 - Test assertion
        assert (
            tracker.is_improvement("test", 110.0) is False
        )  # Higher is not better  # nosec: B101 - Test assertion


class TestAutonomousAgentComprehensive:
    """Comprehensive tests for AutonomousAgent."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)

        # Initialize git repo
        self.repo = git.Repo.init(self.repo_path)
        self.repo.config_writer().set_value("user", "name", "Test User").release()
        self.repo.config_writer().set_value(
            "user", "email", "test@example.com"
        ).release()

        # Create initial commit
        (self.repo_path / "test.txt").write_text("Initial content")
        self.repo.index.add(["test.txt"])
        self.repo.index.commit("Initial commit")

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization_with_components(self):
        """Test agent initialization with all components."""
        agent = AutonomousAgent(str(self.repo_path))

        assert agent.git_tracker is not None  # nosec: B101 - Test assertion
        assert agent.optimizer is not None  # nosec: B101 - Test assertion
        assert agent.memory_store is not None  # nosec: B101 - Test assertion
        assert agent.agent_engine is not None  # nosec: B101 - Test assertion
        assert agent.human_control is not None  # nosec: B101 - Test assertion
        assert agent.async_git is not None  # nosec: B101 - Test assertion

    def test_load_experiment_modules_invalid_names(self):
        """Test loading experiment modules with invalid names."""
        agent = AutonomousAgent(str(self.repo_path))

        # Create experiments directory
        experiments_dir = self.repo_path / "experiments"
        experiments_dir.mkdir(exist_ok=True)

        # Create files with invalid names
        (experiments_dir / "prepare_invalid.py").write_text("# Invalid name")
        (experiments_dir / "run_invalid.py").write_text("# Invalid name")

        # Should skip invalid modules
        modules = agent._load_experiment_modules()
        assert "invalid" not in modules  # nosec: B101 - Test assertion

    def test_load_experiment_modules_import_error(self):
        """Test loading experiment modules with import errors."""
        agent = AutonomousAgent(str(self.repo_path))

        # Create experiments directory
        experiments_dir = self.repo_path / "experiments"
        experiments_dir.mkdir(exist_ok=True)

        # Create file with syntax error
        (experiments_dir / "prepare_syntax_error.py").write_text(
            "invalid python syntax"
        )

        # Should handle import error gracefully
        modules = agent._load_experiment_modules()
        assert "syntax_error" not in modules  # nosec: B101 - Test assertion

    def test_save_checkpoint(self):
        """Test checkpoint saving."""
        agent = AutonomousAgent(str(self.repo_path))

        agent._save_checkpoint("test_exp", 5, {}, [0.1, 0.2, 0.3])

        assert agent.checkpoint_file.exists()  # nosec: B101 - Test assertion

        # Verify checkpoint content
        with open(agent.checkpoint_file, "r") as f:
            saved_data = json.load(f)

        assert (
            saved_data["experiment_name"] == "test_exp"
        )  # nosec: B101 - Test assertion
        assert saved_data["iteration"] == 5  # nosec: B101 - Test assertion

    def test_load_checkpoint(self):
        """Test checkpoint loading."""
        agent = AutonomousAgent(str(self.repo_path))

        # Create test checkpoint
        checkpoint_data = {
            "experiment_name": "test_exp",
            "iteration": 3,
            "performance_history": [0.1, 0.2],
            "best_results": {},
        }

        agent.checkpoint_file.write_text(json.dumps(checkpoint_data))

        loaded = agent._load_checkpoint()

        assert loaded is not None  # nosec: B101 - Test assertion
        assert loaded["experiment_name"] == "test_exp"  # nosec: B101 - Test assertion
        assert loaded["iteration"] == 3  # nosec: B101 - Test assertion

    def test_load_checkpoint_corrupted(self):
        """Test loading corrupted checkpoint."""
        agent = AutonomousAgent(str(self.repo_path))

        # Create corrupted checkpoint
        agent.checkpoint_file.write_text("invalid json")

        loaded = agent._load_checkpoint()

        assert loaded is None  # nosec: B101 - Test assertion

    def test_clear_checkpoint(self):
        """Test checkpoint clearing."""
        agent = AutonomousAgent(str(self.repo_path))

        # Create checkpoint
        agent.checkpoint_file.write_text("{}")
        assert agent.checkpoint_file.exists()  # nosec: B101 - Test assertion

        # Clear checkpoint
        agent._clear_checkpoint()
        assert not agent.checkpoint_file.exists()  # nosec: B101 - Test assertion

    def test_apply_modifications_llm_patch(self):
        """Test applying modifications with LLM patch."""
        agent = AutonomousAgent(str(self.repo_path))

        # Mock LLM integration
        mock_llm = Mock()
        mock_patch_result = Mock()
        mock_patch_result.success = True
        mock_patch_result.result = "MODIFIED_CODE"
        mock_patch_result.confidence = 0.9
        mock_llm.generate_code_patch.return_value = mock_patch_result

        agent.agent_engine.llm_integration = mock_llm

        # Create test file
        test_file = self.repo_path / "test.py"
        test_file.write_text("ORIGINAL_CODE")

        # Apply modifications (only allow-listed parameters are applied)
        modifications = {"LEARNING_RATE": 0.5}
        agent._apply_modifications(str(test_file), modifications)

        # Should use LLM patch
        assert test_file.read_text() == "MODIFIED_CODE"  # nosec: B101 - Test assertion
        mock_llm.generate_code_patch.assert_called_once()

    def test_apply_modifications_llm_patch_fallback(self):
        """Test applying modifications with LLM patch fallback to regex."""
        agent = AutonomousAgent(str(self.repo_path))

        # Mock LLM integration to fail
        mock_llm = Mock()
        mock_patch_result = Mock()
        mock_patch_result.success = False
        mock_patch_result.error = "LLM unavailable"
        mock_llm.generate_code_patch.return_value = mock_patch_result

        agent.agent_engine.llm_integration = mock_llm

        # Create test file
        test_file = self.repo_path / "test.py"
        test_file.write_text("LEARNING_RATE = 0.5\nOTHER_PARAM = 1.0")

        # Apply modifications
        modifications = {"LEARNING_RATE": 0.75}
        agent._apply_modifications(str(test_file), modifications)

        # Should fall back to regex
        content = test_file.read_text()
        assert "LEARNING_RATE = 0.75" in content  # nosec: B101 - Test assertion
        assert "OTHER_PARAM = 1.0" in content  # nosec: B101 - Test assertion

    def test_apply_modifications_parameter_validation(self):
        """Test parameter validation in modifications."""
        agent = AutonomousAgent(str(self.repo_path))

        # Create test file
        test_file = self.repo_path / "test.py"
        test_file.write_text("LEARNING_RATE = 0.5\nINVALID_PARAM = 1.0")

        # Apply modifications with parameters that are not on the allow-list
        modifications = {
            "LEARNING_RATE": 0.75,
            "INVALID_PARAM": 2.0,
            "DANGEROUS_COMMAND": "rm -rf /",
        }
        agent._apply_modifications(str(test_file), modifications)

        # Only allow-listed parameters are applied; the rest are skipped
        content = test_file.read_text()
        assert "LEARNING_RATE = 0.75" in content  # nosec: B101 - Test assertion
        assert (
            "INVALID_PARAM = 1.0" in content
        )  # Left untouched  # nosec: B101 - Test assertion
        assert "INVALID_PARAM = 2.0" not in content  # nosec: B101
        assert (
            "DANGEROUS_COMMAND" not in content
        )  # Should be skipped  # nosec: B101 - Test assertion

    def test_extract_primary_metric_comprehensive(self):
        """Test primary metric extraction with various result formats."""
        agent = AutonomousAgent(str(self.repo_path))

        # Test known experiment metrics
        results = {"blink_magnitude": 0.5}
        metric = agent._extract_primary_metric(results, "attentional_blink")
        assert metric == 0.5  # nosec: B101 - Test assertion

        results = {"net_score": 100}
        metric = agent._extract_primary_metric(results, "iowa_gambling_task")
        assert metric == 100  # nosec: B101 - Test assertion

        # Test fallback patterns
        results = {"accuracy": 0.85}
        metric = agent._extract_primary_metric(results, "unknown_experiment")
        assert metric == 0.85  # nosec: B101 - Test assertion

        results = {"score": 0.9}
        metric = agent._extract_primary_metric(results, "unknown_experiment")
        assert metric == 0.9  # nosec: B101 - Test assertion

        # Test first numeric value fallback
        results = {"text": "abc", "numeric_value": 42, "other": 2.0}
        metric = agent._extract_primary_metric(results, "unknown_experiment")
        assert metric == 42.0  # nosec: B101 - Test assertion

        # Test no numeric values
        results = {"text": "abc", "other": "def"}
        metric = agent._extract_primary_metric(results, "unknown_experiment")
        assert metric == 0.0  # nosec: B101 - Test assertion

    def test_get_metric_direction_comprehensive(self):
        """Test metric direction for all known experiments."""
        agent = AutonomousAgent(str(self.repo_path))

        # Test higher is better experiments
        higher_better = [
            "iowa_gambling_task",
            "change_blindness",
            "dual_n_back",
            "working_memory_span",
            "probabilistic_category_learning",
        ]

        for exp in higher_better:
            direction = agent._get_metric_direction(exp)
            assert direction == "higher"  # nosec: B101 - Test assertion

        # Test lower is better experiments
        lower_better = [
            "attentional_blink",
            "stroop_effect",
            "visual_search",
            "masking",
            "posner_cueing",
        ]

        for exp in lower_better:
            direction = agent._get_metric_direction(exp)
            assert direction == "lower"  # nosec: B101 - Test assertion

    def test_create_session_branch(self):
        """Test creating session branch."""
        agent = AutonomousAgent(str(self.repo_path))

        agent._create_session_branch("test_experiment")

        # Should have created a branch
        assert agent._session_branch is not None  # nosec: B101 - Test assertion
        assert (
            "test_experiment" in agent._session_branch
        )  # nosec: B101 - Test assertion

    def test_create_session_branch_failure(self):
        """Test session branch creation failure."""
        agent = AutonomousAgent(str(self.repo_path))

        # GitPython objects reject attribute patching, so swap the whole repo
        mock_repo = MagicMock()
        mock_repo.git.checkout.side_effect = Exception("git error")
        with patch.object(agent.git_tracker, "repo", mock_repo):
            agent._create_session_branch("test_experiment")

        # Should handle error gracefully
        assert agent._session_branch is None  # nosec: B101 - Test assertion

    def test_get_current_parameters_edge_cases(self):
        """Test parameter extraction with edge cases."""
        agent = AutonomousAgent(str(self.repo_path))

        # Create test file with various parameter types
        test_file = self.repo_path / "test_params.py"
        test_file.write_text("""
# Numeric parameters
FLOAT_PARAM = 0.5
INT_PARAM = 100
NEGATIVE_PARAM = -10

# Boolean parameters
BOOL_TRUE = True
BOOL_FALSE = False

# String parameters
STRING_PARAM = "test_value"

# List parameters
LIST_PARAM = [1, 2, 3, 4, 5]

# Commented out parameters (should be ignored)
# COMMENTED_PARAM = 999

# Function definitions (should be ignored)
def some_function():
    return None

# Class definitions (should be ignored)
class SomeClass:
    pass
""")

        agent.experiment_modules = {"test": {"run_file": str(test_file)}}

        params = agent._get_current_parameters("test")

        # Should extract valid parameters
        assert "FLOAT_PARAM" in params  # nosec: B101 - Test assertion
        assert "INT_PARAM" in params  # nosec: B101 - Test assertion
        assert "NEGATIVE_PARAM" in params  # nosec: B101 - Test assertion
        assert "BOOL_TRUE" in params  # nosec: B101 - Test assertion
        assert "BOOL_FALSE" in params  # nosec: B101 - Test assertion
        assert "STRING_PARAM" in params  # nosec: B101 - Test assertion
        assert "LIST_PARAM" in params  # nosec: B101 - Test assertion

        # Should not extract invalid ones
        assert "COMMENTED_PARAM" not in params  # nosec: B101 - Test assertion
        assert "some_function" not in params  # nosec: B101 - Test assertion
        assert "SomeClass" not in params  # nosec: B101 - Test assertion

    @patch("signal.signal")
    @patch("signal.alarm")
    def test_run_experiment_timeout_handling(self, mock_alarm, mock_signal):
        """Test timeout handling in experiment execution."""
        agent = AutonomousAgent(str(self.repo_path))

        # The experiment runs past its time limit
        mock_runner = Mock()
        mock_runner.run_experiment.side_effect = APGITimeoutError("Timeout")
        mock_run_module = Mock()
        mock_run_module.MockRunner = Mock(return_value=mock_runner)

        agent.experiment_modules = {
            "test": {
                "run": mock_run_module,
                "run_file": str(self.repo_path / "test.py"),
            }
        }

        result = agent.run_experiment("test", timeout_seconds=1)

        assert result.status == "timeout"  # nosec: B101 - Test assertion
        assert result.primary_metric == 0.0  # nosec: B101 - Test assertion

    def test_run_experiment_crash_with_self_healing(self):
        """Test experiment crash with self-healing."""
        agent = AutonomousAgent(str(self.repo_path))

        # Mock experiment modules that crash
        mock_runner = Mock()
        mock_runner.run_experiment.side_effect = Exception("Crash")
        mock_run_module = Mock()
        mock_run_module.MockRunner = Mock(return_value=mock_runner)

        agent.experiment_modules = {
            "test": {
                "run": mock_run_module,
                "run_file": str(self.repo_path / "test.py"),
            }
        }

        # Mock self-healing to succeed
        mock_healing_result = Mock()
        mock_healing_result.success = True
        mock_healing_result.result = "Fixed code"
        setattr(
            agent.agent_engine,
            "xpr_skill_chain",
            Mock(return_value=[mock_healing_result]),
        )

        result = agent.run_experiment("test", max_retries=1)

        assert (  # nosec: B101 - Test assertion
            result.status == "crash"
        )  # Should still be crash if healing fails to reload

    def test_run_experiment_success_with_callbacks(self):
        """Test successful experiment with callbacks."""
        agent = AutonomousAgent(str(self.repo_path))

        # Mock experiment modules
        mock_runner = Mock()
        mock_runner.run_experiment.return_value = {
            "accuracy": 0.85,
            "primary_metric": 0.85,
            "completion_time": 1.0,
        }
        mock_run_module = Mock()
        mock_run_module.MockRunner = Mock(return_value=mock_runner)

        agent.experiment_modules = {
            "test": {
                "run": mock_run_module,
                "run_file": str(self.repo_path / "test.py"),
            }
        }

        result = agent.run_experiment("test")

        assert result.status == "success"  # nosec: B101 - Test assertion
        assert result.primary_metric == 0.85  # nosec: B101 - Test assertion

    def test_optimize_experiment_checkpoint_resume(self):
        """Test optimization with checkpoint resume."""
        agent = AutonomousAgent(str(self.repo_path))

        # No human reviewer is available: skip the interactive review checkpoint
        agent.human_control.config["interaction_mode"] = "autonomous"
        agent.human_control.config["configured"] = False

        # Create checkpoint
        checkpoint_data = {
            "experiment_name": "test",
            "iteration": 2,
            "performance_history": [0.1, 0.2, 0.3],
            "best_results": {},
        }
        agent.checkpoint_file.write_text(json.dumps(checkpoint_data))

        # Mock experiment modules
        mock_runner = Mock()
        mock_runner.run_experiment.return_value = {"primary_metric": 0.4}
        mock_run_module = Mock()
        mock_run_module.MockRunner = Mock(return_value=mock_runner)

        agent.experiment_modules = {
            "test": {"run": mock_run_module, "run_file": "test.py"}
        }

        # Mock optimization methods
        setattr(agent, "_create_session_branch", Mock())
        setattr(agent, "_get_current_parameters", Mock(return_value={}))
        setattr(agent, "_apply_modifications", Mock())
        setattr(agent, "_extract_primary_metric", Mock(return_value=0.4))
        setattr(agent.git_tracker, "is_improvement", Mock(return_value=True))
        setattr(agent.git_tracker, "save_results", Mock())
        setattr(agent, "_clear_checkpoint", Mock())

        results = agent.optimize_experiment("test", iterations=4, resume=True)

        # Should resume from iteration 3 (2 + 1)
        assert (
            len(results) == 1
        )  # Only iteration 3 should run  # nosec: B101 - Test assertion
        clear_checkpoint_mock = getattr(agent, "_clear_checkpoint")
        clear_checkpoint_mock.assert_called_once()

    def test_optimize_experiment_guardrail_trigger(self):
        """Test optimization with guardrail trigger."""
        agent = AutonomousAgent(str(self.repo_path))

        # No human reviewer is available: skip the interactive review checkpoint
        agent.human_control.config["interaction_mode"] = "autonomous"
        agent.human_control.config["configured"] = False

        # Mock experiment modules
        mock_runner = Mock()
        mock_runner.run_experiment.return_value = {"primary_metric": float("nan")}
        mock_run_module = Mock()
        mock_run_module.MockRunner = Mock(return_value=mock_runner)

        agent.experiment_modules = {
            "test": {"run": mock_run_module, "run_file": "test.py"}
        }

        # Mock optimization methods
        setattr(agent, "_create_session_branch", Mock())
        setattr(agent, "_get_current_parameters", Mock(return_value={}))
        setattr(agent, "_apply_modifications", Mock())
        setattr(agent, "_extract_primary_metric", Mock(return_value=float("nan")))

        results = agent.optimize_experiment("test", iterations=2)

        # Should stop on safety violation
        assert len(results) == 1  # nosec: B101 - Test assertion

    def test_optimize_experiment_metric_regression(self):
        """Test optimization with metric regression detection."""
        agent = AutonomousAgent(str(self.repo_path))

        # No human reviewer is available: skip the interactive review checkpoint
        agent.human_control.config["interaction_mode"] = "autonomous"
        agent.human_control.config["configured"] = False

        # Mock experiment modules with decreasing performance
        mock_runner = Mock()
        mock_runner.run_experiment.return_value = {"primary_metric": 0.1}
        mock_run_module = Mock()
        mock_run_module.MockRunner = Mock(return_value=mock_runner)

        agent.experiment_modules = {
            "test": {"run": mock_run_module, "run_file": "test.py"}
        }

        # Mock optimization methods
        setattr(agent, "_create_session_branch", Mock())
        setattr(agent, "_get_current_parameters", Mock(return_value={}))
        setattr(agent, "_apply_modifications", Mock())
        setattr(agent, "_extract_primary_metric", Mock(return_value=0.1))
        setattr(agent, "_get_metric_direction", Mock(return_value="higher"))
        setattr(agent.git_tracker, "rollback_experiment", Mock())

        results = agent.optimize_experiment("test", iterations=4)

        # Should detect regression and stop
        assert (
            len(results) >= 3
        )  # At least 3 iterations to detect regression  # nosec: B101 - Test assertion

    def test_public_wrapper_methods(self):
        """Test public wrapper methods."""
        agent = AutonomousAgent(str(self.repo_path))

        # Test wrappers exist and call private methods
        assert hasattr(agent, "get_current_parameters")  # nosec: B101 - Test assertion
        assert hasattr(agent, "apply_modifications")  # nosec: B101 - Test assertion
        assert hasattr(agent, "extract_primary_metric")  # nosec: B101 - Test assertion
        assert hasattr(agent, "get_metric_direction")  # nosec: B101 - Test assertion

        # Test they call the private methods
        setattr(agent, "_get_current_parameters", Mock(return_value={"test": 0.5}))
        result = agent.get_current_parameters("test")
        assert result == {"test": 0.5}  # nosec: B101 - Test assertion

        setattr(agent, "_apply_modifications", Mock())
        agent.apply_modifications("file.py", {})

        setattr(agent, "_extract_primary_metric", Mock(return_value=0.8))
        metric_result: float = agent.extract_primary_metric({"test": "data"}, "test")
        assert metric_result == 0.8  # nosec: B101 - Test assertion

        setattr(agent, "_get_metric_direction", Mock(return_value="higher"))
        direction_result: str = agent.get_metric_direction("test")
        assert direction_result == "higher"  # nosec: B101 - Test assertion


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
