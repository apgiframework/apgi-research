# APGI Comprehensive Adversarial Testing Framework

This testing framework provides **near 100% code coverage** across all modules, functions, classes, branches, and edge cases for the APGI (Allostatic Precision-Gated Ignition) research system.

## Quick Start

```bash
# Run all tests with coverage
python -m pytest tests/ --cov=. --cov-report=html --cov-report=term-missing

# Run specific test categories
python -m pytest tests/ -m "adversarial"     # Adversarial tests
python -m pytest tests/ -m "integration"       # Integration tests
python -m pytest tests/ -m "security"        # Security tests
python -m pytest tests/ -m "performance"     # Performance tests
python -m pytest tests/ -m "e2e"            # End-to-end tests

# Run mutation testing
python tests/mutation_testing.py

# Run full test orchestration
python tests/test_orchestrator.py
```

## Framework Structure

### Test Files (30+ comprehensive test files)

| File | Description | Coverage |
| ------ | ------------- | ---------- |
| `conftest.py` | Test infrastructure, fixtures, and utilities | 100% |
| `test_adversarial_apgi_core.py` | Adversarial unit tests for core APGI | Line, Branch, Path |
| `test_apgi_audit.py` | Audit system tests (logging, integrity, export) | Audit, Security |
| `test_apgi_authz.py` | Authorization tests (RBAC, permissions, roles) | Security |
| `test_apgi_benchmarks.py` | Performance benchmarks and regression tests | Performance |
| `test_apgi_cli.py` | CLI framework tests (entry point, auth, parsing) | CLI, Integration |
| `test_apgi_compliance.py` | Compliance control tests (GDPR, CCPA, HIPAA) | Compliance |
| `test_apgi_config.py` | Configuration management tests (Pydantic, env vars) | Config |
| `test_apgi_data_retention.py` | Data retention lifecycle tests | Compliance |
| `test_apgi_exception_paths.py` | Error handling and recovery tests | Robustness |
| `test_apgi_integration.py` | APGI integration tests | Integration |
| `test_apgi_orchestration_kernel.py` | Orchestration kernel tests | Integration |
| `test_apgi_security_adapters.py` | Security adapter tests | Security |
| `test_apgi_timeout_abstraction.py` | Cross-platform timeout tests | Platform |
| `test_apgi_unit.py` | Core unit tests for APGI functions | Unit |
| `test_apgi_validation.py` | Parameter validation tests | Validation |
| `test_autonomous_agent.py` | Autonomous agent controller tests | Core |
| `test_experiment_apgi_integration.py` | Experiment integration tests | Integration |
| `test_gui_playwright.py` | GUI automation tests with Playwright | UI, E2E |
| `test_gui_simple.py` | Simple GUI component tests | UI |
| `test_human_layer.py` | Human control layer tests | Core |
| `test_integration_experiments.py` | Full experiment integration tests | E2E |
| `test_memory_profiling.py` | Memory usage profiling tests | Performance |
| `test_memory_store.py` | Memory system tests (TF-IDF, embeddings) | Core |
| `test_performance_stress.py` | Performance and stress tests | Load, Scalability |
| `test_rl_loop.py` | Reinforcement learning loop tests | Core |
| `test_security.py` | Security tests (XSS, SQL injection, etc.) | Security |
| `test_security_controls.py` | Security control mechanism tests | Security |
| `test_xpr_agent_engine_comprehensive.py` | XPR agent engine tests | Core, LLM |
| `coverage_config.py` | Coverage analysis and gap detection | Reporting |
| `mutation_testing.py` | Mutation testing and weak assertion detection | Quality |
| `test_orchestrator.py` | Test orchestration and reporting | Orchestration |

## Test Categories

### 1. Adversarial Unit Tests

Tests that intentionally use extreme, invalid, or malicious inputs:

- **Boundary Values**: Edge case floats/ints (inf, nan, max, min)
- **Invalid Inputs**: Negative values, nulls, wrong types
- **Adversarial Data**: SQL injection strings, XSS payloads, path traversal
- **Extreme Scales**: Very large/small values, near overflow/underflow

```python
@pytest.mark.adversarial
def test_precision_adversarial_invalid_inputs(self):
    """Test precision handles adversarial invalid inputs."""
    invalid_inputs = [-1e308, float("-inf"), -sys.float_info.min]
    for invalid in invalid_inputs:
        result = FoundationalEquations.precision(invalid)
        assert result == 1e6  # Capped protection
```

### 2. Integration Tests

Tests for cross-module interactions:

- **Experiment Workflows**: Full experiment lifecycle
- **File I/O**: JSON, NumPy, CSV read/write
- **Configuration**: Environment variables, config validation
- **External Services**: Mocked matplotlib, requests, database
- **Error Handling**: Retry logic, exception chains

### 3. Security Tests

Tests for security vulnerabilities:

- **SQL Injection**: Pattern detection and prevention
- **XSS Prevention**: HTML escaping, script tag detection
- **Path Traversal**: Suspicious path detection
- **Input Sanitization**: Safe string handling
- **RNG Security**: Deterministic, reproducible randomness

### 4. Performance Tests

Tests for execution speed and resource usage:

- **Benchmarks**: Function-level performance (target: <500ms for 100k ops)
- **Memory Usage**: Stability over many operations
- **Stress Tests**: 1M+ iterations, concurrent simulations
- **Scalability**: Linear scaling with problem size
- **Resource Leaks**: File handle and memory leak detection

### 5. End-to-End Tests

Full system simulation tests:

- **Dynamical System Simulation**: 100k step simulations
- **Parameter Sweeps**: Multiple configuration testing
- **Deterministic Reproducibility**: Controlled seeding for identical results

## Coverage Configuration

The framework provides **comprehensive coverage tracking**:

```ini
# pytest.ini
[tool:pytest]
addopts =
    --cov=.
    --cov-branch              # Branch coverage
    --cov-report=html         # HTML report
    --cov-report=xml          # XML report for CI
    --cov-report=term-missing # Terminal with missing lines
    --cov-fail-under=85       # Minimum 85% coverage
```

### Coverage Analysis

```python
from tests.coverage_config import CoverageAnalyzer

analyzer = CoverageAnalyzer()
analyzer.load_from_json(Path("coverage.json"))
gaps = analyzer.identify_gaps()
report = analyzer.generate_report()
```

## Mutation Testing

Verify test effectiveness by introducing code mutations:

```bash
# Run mutation testing
python tests/mutation_testing.py --max-mutants 100

# Output includes:
# - Mutation score (target: 80%+)
# - Surviving mutants (target: <5)
# - Weak assertion detection
```

### Mutation Operators

| Operator | Description |
| ---------- | ------------- |
| AOR | Arithmetic Operator Replacement (+ → -, * → /) |
| COR | Comparison Operator Replacement (== → !=, < → >=) |
| LOR | Logical Operator Replacement (and → or) |
| CRR | Constant Replacement (0 → 1, True → False) |

## Test Orchestration

Run the complete test suite with structured reporting:

```bash
python tests/test_orchestrator.py
```

### Generated Reports

- `test_reports/test_report.json` - Complete test results
- `test_reports/coverage_gaps.json` - Missing coverage areas
- `htmlcov/index.html` - Visual HTML coverage report
- `coverage.json` - Machine-readable coverage data

## Fixtures and Utilities

### Available Fixtures

```python
# Data generation
data_factory          # AdversarialDataFactory for edge cases

# Mocking
mock_factory          # MockFactory for external services

# Performance
performance_monitor   # Context manager for timing/memory

# Security
security_tester       # SecurityTester for injection detection

# Temporary files
temp_dir              # Temporary directory

# Assertions
adv_assert            # AdversarialAssertions with NaN/finite checks
```

### Custom Assertions

```python
def test_array_properties(adv_assert):
    arr = np.random.rand(10)
    adv_assert.assert_nan_free(arr)
    adv_assert.assert_finite(arr)
    adv_assert.assert_within_bounds(arr, 0, 1)
    adv_assert.assert_shape(arr, (10,))
```

## Deterministic Testing

All tests use controlled seeding for reproducibility:

```python
DETERMINISTIC_SEED = 42

@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset random state before each test."""
    random.seed(DETERMINISTIC_SEED)
    np.random.seed(DETERMINISTIC_SEED)
```

## Markers

Use markers to categorize and filter tests:

| Marker | Description | Usage |
| -------- | ------------- | ------- |
| `@pytest.mark.slow` | Long-running tests | Exclude with `-m "not slow"` |
| `@pytest.mark.integration` | Integration tests | Run with `-m integration` |
| `@pytest.mark.e2e` | End-to-end tests | Run with `-m e2e` |
| `@pytest.mark.security` | Security tests | Security audit |
| `@pytest.mark.performance` | Performance tests | Benchmarking |
| `@pytest.mark.stress` | Stress tests | Load testing |
| `@pytest.mark.adversarial` | Adversarial tests | Edge cases |
| `@pytest.mark.boundary` | Boundary value tests | Limits testing |
| `@pytest.mark.race_condition` | Concurrency tests | Thread safety |

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -e ".[test]"
      - run: python -m pytest tests/ --cov=. --cov-fail-under=85
      - run: python tests/mutation_testing.py --max-mutants 50
```

## Requirements

```text
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-json-report>=1.5.0
pytest-xdist>=3.0.0
mutmut>=2.4.0  # Optional: for mutation testing
```

## Test Reports

After running tests, reports are generated in `test_reports/`:

```json
{
  "summary": {
    "total_tests": 250,
    "passed": 248,
    "failed": 0,
    "skipped": 2,
    "coverage_percent": 87.5,
    "branch_coverage_percent": 82.3
  },
  "coverage_gaps": [
    {
      "file": "APGI_System.py",
      "lines": "150-155",
      "reason": "Exception handling paths",
      "suggested_test": "Add test for ValueError on invalid level"
    }
  ],
  "recommendations": [
    "Increase line coverage from 87.5% to 90%+",
    "Add 3 tests for exception handling paths"
  ]
}
```

## License

Part of the APGI Research System.
