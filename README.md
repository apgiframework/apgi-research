# APGI Research Platform

An autonomous research system designed to optimize psychological experiments through AI-driven parameter tuning and continuous learning. The system implements a complete Autonomous Learning Algorithm (ALGORITH) with human-in-the-loop oversight and production-grade security, compliance, and observability.

## Architecture

The APGI Research Application implements a comprehensive autonomous learning algorithm with multiple integrated layers:

### Core System Components

#### 1. Human Control Layer (`human_layer.py`)

- `HumanControlLayer` - Interactive configuration and approval workflow
- Task prioritization with `TaskPriority` enum (CRITICAL, HIGH, MEDIUM, LOW)
- Review & Decision system with `ReviewDecision` enum (APPROVE, MODIFY, REJECT)
- Three interaction modes: interactive, batch, autonomous
- Integration with `hypothesis_approval_board.py` for hypothesis management

#### 2. Agent Harness (`autonomous_agent.py`)

- `AutonomousAgent` - Main controller orchestrating optimization loops
- `GitPerformanceTracker` - Version control-based performance tracking with async operations
- `ParameterOptimizer` - AI-driven parameter modification with experiment-specific strategies
- `AsyncGitOperations` - Non-blocking git commits for continuous optimization
- Checkpoint system for crash recovery

#### 3. Agent Execution Engine (`xpr_agent_engine.py`)

- `XPRAgentEngine` - Core engine with skill registration and execution
- `XPRAgentEngineEnhanced` - Advanced engine with LLM integration via litellm
- `LLMIntegration` - Multi-provider LLM support (OpenAI, Anthropic, local models)
- Self-healing skill chain: `xpr_job_debug` → `xpr_issue_fix` → `xpr_issue_report`
- LLM-generated code patches with syntax validation

#### 4. Memory System (`memory_store.py`)

- `MemoryStore` - Indexed knowledge base with TF-IDF semantic search
- `MemoryEntry` - Structured storage for patterns, strategies, and failures
- `VectorEmbedding` - Neural-style embeddings for semantic similarity
- Hybrid search combining TF-IDF and vector similarity
- `update_memory_from_report()` - Automatic extraction of lessons learned

#### 5. Experiment GUI (`GUI_auto_improve_experiments.py`)

- `ExperimentRunnerGUI` - Modern CustomTkinter interface
- Real-time experiment execution and visualization
- Hypothesis approval board integration
- Results visualization with matplotlib
- Guardrail dashboard with live metrics
- 1400×900 pixel default window, responsive layout

#### 6. APGI Core (`APGI_System.py`)

- Complete dynamical system implementation
- 51 psychological states with Π vs Π̂ distinction
- Neuromodulator mapping (ACh, NE, DA, 5-HT)
- Measurement equations (HEP, P3b, detection thresholds)
- Hierarchical 5-level processing
- Psychiatric profile modeling (GAD, MDD, Psychosis)

### Production Infrastructure Modules

#### 7. Config Management (`apgi_config.py`)

- `APGIExperimentConfigSchema` - Pydantic schema for experiment config
- `APGISecurityConfigSchema` - Security configuration validation
- `APGIMetricsConfigSchema` - Metrics and monitoring config
- `ConfigManager` - Singleton configuration manager with:
  - Environment variable loading (APGI_* prefix)
  - JSON/YAML config file support
  - LRU caching for performance
  - Source tracking for debugging

#### 8. CLI Framework (`apgi_cli.py`)

- Standardized CLI entry point with `cli_entrypoint()`
- `@require_auth` decorator for authorization enforcement
- `create_standard_parser()` for consistent argument handling
- Exit codes: 0 (success), 77 (auth denied), 78 (config error), 130 (interrupted)
- Legacy compatibility via `standardized_main()`

#### 9. Security Adapters (`apgi_security_adapters.py`)

- `SecurityContext` - Per-context security configuration
- `SecurityLevel` enum: PERMISSIVE, STANDARD, STRICT
- Deny-by-default subprocess allowlist
- Explicit pickle/serialization controls (JSON, msgpack, protobuf)
- Security metrics and telemetry
- KMS-backed secret support

#### 10. Orchestration Kernel (`apgi_orchestration_kernel.py`)

- `APGIOrchestrationKernel` - Central runner framework
- `TrialTransformer` - Abstract base for experiment-specific transformations
- `ExperimentRunConfig` - Typed configuration for runs
- `TrialMetrics` - Standardized metrics across all experiments
- Unified trial processing pipeline

#### 11. Authorization (`apgi_authz.py`)

- Role-based access control (RBAC)
- `Role` enum: ADMIN, OPERATOR, RESEARCHER, REVIEWER, AUDITOR
- `Permission` enum for granular access control
- `AuthorizationContext` for permission checks
- `OperatorIdentity` for user tracking
- Consent tracking for GDPR compliance

#### 12. Audit System (`apgi_audit.py`)

- `AuditSink` - Immutable audit logging with HMAC signatures
- `AuditEventType` enum covering all system events
- Chain verification for integrity
- Export to JSON for compliance reviews
- Event filtering and querying
- Tamper-evident hash chain

#### 13. Data Retention (`apgi_data_retention.py`)

- `RetentionScheduler` - Automated data lifecycle management
- `RetentionPolicy` enum: GDPR_DEFAULT, HIPAA_MINIMUM, RESEARCH_EXTENDED
- Right to erasure (GDPR Article 17) implementation
- Right to data portability export
- Real deletion executors (not simulated)
- Deletion verification with audit trail

#### 14. Timeout Abstraction (`apgi_timeout_abstraction.py`)

- Cross-platform timeout management (Windows, macOS, Linux)
- `TimeoutManager` with context manager and decorator support
- `CancellableOperation` for cooperative cancellation
- `SignalTimeout` for Unix signal-based timeouts
- `ThreadTimeout` for cross-platform thread-based timeouts

#### 15. Profiling (`apgi_profiler.py`)

- `APGILoggingProfiler` - Structured logging profiler
- `@profiled` decorator for automatic profiling
- Function, method, and line-level profiling
- `ProfilingLevel` enum: NONE, FUNCTION, METHOD, LINE
- Metrics export to JSON/CSV

#### 16. Error Handling (`apgi_errors.py`)

- Comprehensive error taxonomy with `APGIBaseError`
- `APGIValidationError` - Parameter validation failures
- `APGITimeoutError` - Timeout violations
- `APGIAuthorizationError` - Permission denied
- `APGIAuditError` - Audit system failures
- `APGIComplianceError` - Compliance violations
- `APGIIntegrationError` - Integration failures
- Exception chaining with `__cause__`

#### 17. Structured Logging (`apgi_logging.py`)

- `APGIContextLogger` - Contextual logging with operator tracking
- `APGIFormatter` - JSON/formatted output
- Context variables: operator_id, experiment_id, trial_id
- Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Integration with audit sink

#### 18. Metrics (`apgi_metrics.py`)

- `MetricsCollector` - Centralized metrics collection
- Prometheus-compatible export format
- Custom metric types: Counter, Gauge, Histogram
- Metric tagging with labels
- Time-series data export

#### 19. Integration (`apgi_integration.py`)

- `APGIIntegration` - Main integration class for experiments
- `APGIParameters` - Parameter validation and storage
- `PrecisionExpectationState` - Π vs Π̂ distinction
- `HierarchicalProcessor` - 5-level hierarchical processing
- Neuromodulator mappings to precision parameters

#### 20. Validation (`apgi_validation.py`)

- `APGIParameterValidator` - Parameter range validation
- `ExperimentValidator` - Experiment structure validation
- Input sanitization for security
- Schema validation with Pydantic

#### 21. Double Dissociation (`apgi_double_dissociation.py`)

- Two-stage estimation protocol
- Stage 1: Anchor Πⁱ_baseline (min 3 sessions, ICC ≥ 0.65)
- Stage 2: Fit β parameter
- Physiological anchoring with EEG alpha/gamma power
- Stability fallback to composite Π_eff

#### 22. Compliance (`apgi_compliance.py`)

- `ComplianceManager` - Central compliance checking
- GDPR, CCPA, HIPAA control mapping
- Data classification (PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED)
- Policy enforcement validation
- Compliance report generation

### Key Features

- **Human-in-the-Loop Control**: Interactive approval workflow with APPROVE/MODIFY/REJECT decisions
- **Self-Healing Experiments**: Automatic error detection, debugging, and retry with skill chaining
- **Skill Chaining System**: Modular agent skills (debug → fix → report) for autonomous recovery
- **Cognitive Memory**: TF-IDF and vector embeddings for pattern recognition across iterations
- **Infinite Autonomous Optimization**: Continuous parameter optimization with async git operations
- **Comprehensive Guardrails**: Confidence thresholds, metric regression detection, safety violations
- **Git-Based Learning**: Version control tracking with automatic rollback on failures
- **Multi-Experiment Support**: 29+ psychological experiments with 100/100 APGI compliance
- **LLM Integration**: Multi-provider support (OpenAI, Anthropic, local) via litellm
- **Real-Time Monitoring**: GUI with live experiment execution and embedded matplotlib visualization
- **Crash Recovery**: Checkpoint system for resuming after interruptions

### Security & Compliance Features

- **Deny-by-Default Security**: Subprocess allowlist, pickle/serialization controls
- **Role-Based Access Control**: 5 roles (ADMIN, OPERATOR, RESEARCHER, REVIEWER, AUDITOR)
- **Immutable Audit Logging**: HMAC-signed audit trail with chain verification
- **GDPR/CCPA/HIPAA Compliance**: Data retention, right to erasure, data portability
- **Cross-Platform Timeouts**: Windows, macOS, Linux timeout abstraction
- **Structured Logging**: JSON logging with operator/experiment context
- **Security Metrics**: Real-time security telemetry and deny metrics

## APGI Protocol Specification 1.1.0

The experimental protocols in [`protocols/`](protocols/) follow APGI protocol specification
1.1.0, byte-identical to [apgi-framework](https://github.com/apgiframework/apgi-framework)
1.1.0. Validate them with `make protocols-check`; see [`protocols/CHANGELOG.md`](protocols/CHANGELOG.md)
for what changed from 1.0. Under 1.1.0 a failure to confirm a prediction is not falsification
(verdicts are CONFIRMED, FALSIFIED or INDETERMINATE), and κ (ATP/bit) is uncalibrated.

`utils/apgi_protocol_spec.py` loads the registered protocols and provides
`classify_prediction()` for the three-way verdict. `check_parameter_defaults()` compares the
`APGI_System.APGIParameters` defaults with the registered θ₀, γ_sig and β_SM ranges. The
default somatic gain `beta = 1.5` lies outside the registered β_SM range [0.3, 0.7], so it
must not be used for a confirmatory fit.

The 29 experiments below are paradigm simulations for autonomous parameter tuning. They are
not the registered Protocol 0–7 studies, and their "APGI compliance" score is this platform's
internal implementation check, not a protocol verdict.

## Supported Experiments

The application supports 29 psychological experiments, all achieving **100/100 APGI compliance**:

| Category | Experiment | Primary Metric |
| -------- | ---------- | ------------- |
| **Decision-Making** | Iowa Gambling Task | `net_score` |
| | Go/No-Go | `d_prime` |
| | Stop Signal | `ssrt_ms` |
| | Simon Effect | `simon_effect_ms` |
| **Attention** | Attentional Blink | `blink_magnitude` |
| | Posner Cueing | `cueing_effect_ms` |
| | Visual Search | `conjunction_slope` |
| | Change Blindness | `change_detection_rate` |
| | Inattentional Blindness | `detection_rate` |
| | Navon Task | `global_local_bias` |
| **Memory** | Dual N-Back | `d_prime` |
| | Sternberg Memory | `memory_scan_rate` |
| | Working Memory Span | `span_size` |
| | DRM False Memory | `false_alarm_rate` |
| | Serial Reaction Time | `sequence_learning` |
| | Artificial Grammar Learning | `grammar_accuracy` |
| **Interference** | Stroop Effect | `interference_effect_ms` |
| | Eriksen Flanker | `flanker_effect_ms` |
| | Masking | `backward_masking_effect` |
| **Perception** | Binocular Rivalry | `dominance_duration` |
| | Multisensory Integration | `integration_index` |
| | Time Estimation | `temporal_precision` |
| **Specialized** | Somatic Marker Priming | `priming_effect` |
| | Interoceptive Gating | `gating_ratio` |
| | Metabolic Cost | `metabolic_efficiency` |
| | Virtual Navigation | `navigation_accuracy` |
| | Probabilistic Category Learning | `category_accuracy` |
| | AI Benchmarking | `benchmark_score` |

## Usage

### Environment Setup

```bash
# Set required environment variables
export APGI_OPERATOR_ID="user123"
export APGI_OPERATOR_NAME="Researcher Name"
export APGI_OPERATOR_ROLE="operator"  # admin, operator, researcher, reviewer, auditor

# Optional security settings
export APGI_ENABLE_PROFILING=false
export APGI_KMS_KEY="your-kms-key"
export APGI_AUDIT_KEY="your-audit-key"

# Load from config file
export APGI_CONFIG_FILE="config/apgi.json"
```

### Command Line Interface

#### Standard CLI Entry Point (Recommended)

```bash
# Run experiment with standardized CLI
python run_iowa_gambling_task.py --trials 100 --verbose
python run_stroop_effect.py --operator user123 --role operator

# Run autonomous optimization
python autonomous_agent.py --experiment masking --iterations 100
python autonomous_agent.py --all-experiments --iterations 50
python autonomous_agent.py --overnight
```

#### Legacy Entry Points

```bash
# Direct experiment execution (for backward compatibility)
python run_attentional_blink.py
python run_visual_search.py
python run_change_blindness.py

# Data preparation
python prepare.py

# Training script
python train.py
```

### GUI Interface

```bash
# Launch the graphical interface
python GUI_auto_improve_experiments.py

# With debug mode
python GUI_auto_improve_experiments.py --debug
```

The GUI provides:

- Real-time experiment execution with CustomTkinter interface
- Guardrail dashboard with live metrics (confidence, regression, escalation count)
- Hypothesis approval board integration
- Sequential and parallel run modes
- Results visualization with embedded matplotlib
- 1400×900 pixel default window, responsive layout

### API Integration

#### Core Components

```python
from autonomous_agent import AutonomousAgent
from human_layer import HumanControlLayer, ReviewDecision
from memory_store import MemoryStore
from xpr_agent_engine import XPRAgentEngineEnhanced

# Initialize components
agent = AutonomousAgent()
human = HumanControlLayer()
memory = MemoryStore()
engine = XPRAgentEngineEnhanced()

# Configure human interaction layer
human.configure_if_needed()

# Run optimization with human oversight
results = agent.optimize_experiment("attentional_blink", iterations=100)

# Run single experiment with modifications
result = agent.run_experiment(
    "masking",
    modifications={"BASE_DETECTION_RATE": 0.5},
    timeout_seconds=1800,
    max_retries=1
)

# Review and update memory
if review_result.decision == ReviewDecision.APPROVE:
    memory.add_memory_with_embedding(
        experiment_name="masking",
        pattern_type="success_pattern",
        content="BASE_DETECTION_RATE=0.5 improved performance by 15%",
        context={"metric_delta": 0.15}
    )
```

#### Security & Authorization

```python
from apgi_security_adapters import get_security_factory, SecurityLevel
from apgi_authz import get_authz_manager, Role, Permission, AuthorizationContext
from apgi_audit import get_audit_sink, AuditEventType
from apgi_config import get_config

# Security context
factory = get_security_factory()
context = factory.create_context(
    operator_id="user123",
    security_level=SecurityLevel.STANDARD,
)
secure_popen = factory.get_secure_popen(context)

# Authorization
authz = get_authz_manager()
operator = authz.register_operator("john", Role.OPERATOR)
auth_context = AuthorizationContext(
    operator=operator,
    resource_type="experiment",
    resource_id="iowa_gambling",
    action=Permission.RUN_EXPERIMENT,
)
if authz.authorize_action(auth_context):
    print("Permission granted")

# Audit logging
audit = get_audit_sink()
audit.record_event(
    event_type=AuditEventType.EXPERIMENT_STARTED,
    operator_id="user123",
    operator_name="John Doe",
    resource_type="experiment",
    resource_id="iowa_gambling",
    action="start",
)

# Configuration
config = get_config()
experiment_config = config.get_experiment_config("stroop_effect")
security_config = config.get_security_config()
```

#### Orchestration Kernel

```python
from apgi_orchestration_kernel import (
    get_orchestration_kernel,
    ExperimentRunConfig,
    TrialTransformer,
)
from apgi_config import get_config

class MyTrialTransformer(TrialTransformer):
    def transform_trial(self, trial_data):
        return trial_data
    
    def extract_prediction_error(self, trial_data):
        return trial_data["error"]
    
    def extract_precision(self, trial_data):
        return trial_data["precision"]

# Run experiment through kernel
kernel = get_orchestration_kernel()
config = get_config()
apgi_config = config.get_experiment_config("iowa_gambling")

run_config = ExperimentRunConfig(
    experiment_name="iowa_gambling",
    operator_id="user123",
    operator_name="John Doe",
    apgi_config=apgi_config,
)

run_context = kernel.create_run_context(run_config)
for trial_data in trials:
    metrics = kernel.process_trial(run_context, trial_data, MyTrialTransformer())
results = kernel.finalize_run(run_context)
```

## System Configuration

### Environment Variables

| Variable | Description | Required | Default |
| -------- | ----------- | -------- | ------- |
| `APGI_OPERATOR_ID` | Unique operator identifier | No | auto-generated |
| `APGI_OPERATOR_NAME` | Human-readable operator name | No | "cli_user" |
| `APGI_OPERATOR_ROLE` | Role (admin, operator, researcher, reviewer, auditor) | No | "operator" |
| `APGI_CONFIG_FILE` | Path to config file | No | auto-search |
| `APGI_ENABLE_PROFILING` | Enable performance profiling | No | false |
| `APGI_PROFILING_LEVEL` | Profiling detail level | No | "function" |
| `APGI_AUDIT_KEY` | HMAC signing key for audit trail | **Yes** | none |
| `APGI_KMS_KEY` | KMS key for config checksum validation | **Yes** | none |
| `APGI_CONFIG_SECRET_KEY` | Config validation secret | **Yes** | none |
| `APGI_PSEUDONYM_SALT` | Pseudonymization salt | **Yes** | none |

### Dependencies

1. **Core Requirements** (Python 3.10+)

   ```bash
   # Using uv (recommended)
   uv sync
   
   # Using pip
   pip install -r requirements.txt
   ```

2. **Development Dependencies**

   ```bash
   pip install -r requirements-test.txt
   ```

### Config File Format

```json
{
  "experiment_iowa_gambling_tau_s": 0.35,
  "experiment_iowa_gambling_beta": 1.5,
  "security_audit_enabled": true,
  "security_authz_enabled": true,
  "metrics_profiling_enabled": false
}
```

### Performance Tuning

- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Memory Optimization**: Ring-buffered deques for bounded history
- **Parallel Processing**: Multi-threaded experiment execution
- **Timeout Management**: Cross-platform configurable timeouts
- **Caching**: LRU cache for config access, TTL for static assets

## Development

### Testing

The system includes 30+ comprehensive test files:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tests --cov-report=html

# Run specific component tests
pytest tests/test_autonomous_agent.py
pytest tests/test_xpr_agent_engine_comprehensive.py
pytest tests/test_human_layer.py
pytest tests/test_memory_store.py
pytest tests/test_rl_loop.py

# Run integration tests
pytest tests/test_integration_experiments.py

# Run performance and stress tests
pytest tests/test_performance_stress.py

# Run security tests
pytest tests/test_security.py

# GUI tests with Playwright
pytest tests/test_gui_playwright.py
```

### Test Coverage Areas

- **Unit Tests**: Core components (autonomous_agent, xpr_agent_engine, human_layer, memory_store)
- **Integration Tests**: Full RL loop, experiment execution, end-to-end workflows
- **XPR Engine Tests**: Skill chaining, LLM integration, self-healing mechanisms
- **Human Layer Tests**: Review workflow, approval decisions, configuration
- **Memory Tests**: Semantic search, vector embeddings, pattern extraction
- **Performance Tests**: Stress testing, timeout handling, async operations
- **Security Tests**: Input validation, module loading, sandboxed execution
- **GUI Tests**: Playwright-based UI automation and visualization

### Code Quality

```bash
# Linting
flake8 autonomous_agent.py GUI-auto_improve_experiments.py

# Formatting
black *.py

# Type checking
mypy autonomous_agent.py
```

### Documentation

```bash
# Generate coverage report
pytest --cov=tests --cov-report=html
```

## Troubleshooting

### Common Issues

1. **Module Import Errors**

   - Clear Python cache: `python -B -m pytest`
   - Check PYTHONPATH configuration
   - Verify experiment file structure

2. **Git Repository Issues**

   - Initialize git repo: `git init`
   - Check permissions on .git directory
   - Verify remote repository configuration

3. **Performance Issues**

   - Monitor memory usage with `htop`
   - Check GPU availability: `nvidia-smi`
   - Optimize experiment parameters

4. **GUI Issues**

   - Check CustomTkinter installation
   - Verify display server configuration
   - Test with minimal example

### Debug Mode

```bash
# Enable debug logging
python autonomous_agent.py --experiment masking --debug

# GUI debug mode
python GUI-auto_improve_experiments.py --debug
```

## Contributing

### Development Workflow

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure code quality standards
5. Submit pull request

### Code Standards

- **PEP 8** compliance
- **Type hints** for all public APIs
- **Docstrings** for all classes and methods
- **Tests** with >70% coverage
- **Security** review for all changes

## Citation

If you use this platform in your research, please cite:

```bibtex
@software{pesochin_apgi_research_2026,
  title={APGI Research: Allostatic Precision-Gated Ignition — Autonomous Research Platform},
  author={Pesochin, Dimitry},
  year={2026},
  version={1.1.0},
  doi={10.5281/zenodo.21632264},
  url={https://github.com/apgiframework/apgi-research},
  orcid={https://orcid.org/0009-0005-0328-2896}
}
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

### Core Frameworks

- **ALGORITHM** - Autonomous learning algorithm with human-in-the-loop
- **APGI** - Allostatic Precision-Gated Ignition framework
- **CustomTkinter** - Modern GUI framework (5.0+)
- **Pydantic** - Data validation and settings management
- **litellm** - Multi-provider LLM API integration

### Infrastructure & Security

- **GitPython** - Git integration for version control learning
- **sentence-transformers** - Semantic embeddings for memory system
- **PyTorch** - Deep learning backend
- **NumPy** - Numerical computing foundation
- **Matplotlib** - Visualization and plotting

### Production Infrastructure

- **apgi_config** - Typed configuration management
- **apgi_cli** - Standardized CLI framework
- **apgi_security_adapters** - Injectable security controls
- **apgi_orchestration_kernel** - Central experiment runner
- **apgi_authz** - Role-based access control
- **apgi_audit** - Immutable audit logging
- **apgi_data_retention** - Data lifecycle management
- **apgi_timeout_abstraction** - Cross-platform timeouts

### Testing & Quality

- **pytest** - Testing framework
- **Playwright** - Browser automation for GUI testing
- **flake8** - Code linting
- **mypy** - Static type checking
- **black** - Code formatting
- **mutation_testing** - Test effectiveness verification

## Quick Start

```bash
# Install dependencies
pip install -e .

# Set required environment variables
export APGI_AUDIT_KEY="$(openssl rand -hex 32)"
export APGI_KMS_KEY="$(openssl rand -hex 32)"
export APGI_CONFIG_SECRET_KEY="$(openssl rand -hex 32)"
export APGI_PSEUDONYM_SALT="$(openssl rand -hex 16)"

# Run experiments
python GUI_auto_improve_experiments.py

# Or use CLI
python apgi_cli.py --help
```

## Requirements

- Python 3.10+
- See `requirements-test.txt` for development dependencies
- See `pyproject.toml` for runtime dependencies

## Support

For issues, questions, or contributions:

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check README.md and inline docs
- **Community**: Join discussions in GitHub Discussions

- `docs/APGI-USAGE-GUIDE.md` - Complete usage guide
- `docs/ALGORITHM.md` - Algorithm documentation
- `docs/APGI-COMPLIANCE.md` - Compliance framework
- `TODO.md` - Current tasks and known issues

## Version

See `pyproject.toml` for current version (accessed via `apgi_version.get_version()`)
