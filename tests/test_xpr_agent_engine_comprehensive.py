"""
================================================================================
COMPREHENSIVE TESTS FOR XPR AGENT ENGINE
================================================================================

This module provides extensive testing for the XPR Agent Engine including:
- XPRAgentEngine core functionality
- LLMIntegration provider management
- EnhancedXPRAgentEngine features
- XPRAgentEngineEnhanced advanced capabilities
- Error handling and edge cases
- Concurrent execution testing

Addresses critical testing gap identified in testing-report.md:
"xpr_agent_engine.py (4 tests for 1,198 lines)"

Target: 40+ test functions to achieve comprehensive coverage
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from xpr_agent_engine import (
    EnhancedXPRAgentEngine,
    ExecutionReport,
    LLMIntegration,
    SkillResult,
    XPRAgentEngine,
    XPRAgentEngineEnhanced,
    XPRSkillResult,
    XPRSkillType,
    register_xpr_skills,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def engine() -> XPRAgentEngine:
    """Provide a fresh XPRAgentEngine instance."""
    return XPRAgentEngine()


@pytest.fixture
def enhanced_engine() -> XPRAgentEngineEnhanced:
    """Provide a fresh XPRAgentEngineEnhanced instance."""
    return XPRAgentEngineEnhanced()


@pytest.fixture
def llm_integration() -> LLMIntegration:
    """Provide a fresh LLMIntegration instance."""
    return LLMIntegration()


@pytest.fixture
def sample_skill_func():
    """Provide a sample skill function."""

    def skill(x: int) -> int:
        return x * 2

    return skill


@pytest.fixture
def sample_failing_skill():
    """Provide a skill that raises an exception."""

    def skill(x: int) -> int:
        raise ValueError("Skill failed")

    return skill


@pytest.fixture
def sample_execution_report() -> ExecutionReport:
    """Provide a sample execution report."""
    return ExecutionReport(
        experiment_name="test_experiment",
        success=True,
        execution_time=1.5,
        result={"accuracy": 0.95},
        confidence=0.8,
    )


# =============================================================================
# SKILL RESULT AND EXECUTION REPORT TESTS
# =============================================================================


class TestSkillResult:
    """Test SkillResult dataclass."""

    def test_skill_result_creation(self) -> None:
        """Test creating a SkillResult with all fields."""
        result = SkillResult(
            success=True,
            skill_type="test_skill",
            result={"key": "value"},
            error=None,
            execution_time=1.5,
            confidence=0.9,
            metadata={"version": "1.0"},
        )
        if not result.success:
            raise AssertionError("Result should be successful")
        if result.skill_type != "test_skill":
            raise AssertionError(
                f"Expected skill_type 'test_skill', got: {result.skill_type}"
            )
        if result.result != {"key": "value"}:
            raise AssertionError(
                f"Expected result {{'key': 'value'}}, got: {result.result}"
            )
        if result.error is not None:
            raise AssertionError(f"Expected error to be None, got: {result.error}")
        if result.execution_time != 1.5:
            raise AssertionError(
                f"Expected execution_time 1.5, got: {result.execution_time}"
            )
        if result.confidence != 0.9:
            raise AssertionError(f"Expected confidence 0.9, got: {result.confidence}")
        if result.metadata != {"version": "1.0"}:
            raise AssertionError(
                f"Expected metadata {{'version': '1.0'}}, got: {result.metadata}"
            )

    def test_skill_result_defaults(self) -> None:
        """Test SkillResult with default values."""
        result = SkillResult(success=True, skill_type="test")
        if result.result is not None:
            raise AssertionError(f"Expected result to be None, got: {result.result}")
        if result.error is not None:
            raise AssertionError(f"Expected error to be None, got: {result.error}")
        if result.execution_time != 0.0:
            raise AssertionError(
                f"Expected execution_time 0.0, got: {result.execution_time}"
            )
        if result.confidence != 0.0:
            raise AssertionError(f"Expected confidence 0.0, got: {result.confidence}")
        if result.metadata is not None:
            raise AssertionError(
                f"Expected metadata to be None, got: {result.metadata}"
            )

    def test_skill_result_failure(self) -> None:
        """Test SkillResult for failed execution."""
        result = SkillResult(
            success=False,
            skill_type="failing_skill",
            error="Something went wrong",
            confidence=0.0,
        )
        if result.success:
            raise AssertionError("Result should be unsuccessful")
        if result.error != "Something went wrong":
            raise AssertionError(
                f"Expected error 'Something went wrong', got: {result.error}"
            )


class TestExecutionReport:
    """Test ExecutionReport dataclass."""

    def test_execution_report_creation(self) -> None:
        """Test creating an ExecutionReport."""
        report = ExecutionReport(
            experiment_name="exp_001",
            success=True,
            execution_time=2.5,
            result={"metric": 0.95},
            confidence=0.85,
        )
        if report.experiment_name != "exp_001":
            raise AssertionError(
                f"Expected experiment_name 'exp_001', got: {report.experiment_name}"
            )
        if not report.success:
            raise AssertionError("Expected success to be True")
        if report.execution_time != 2.5:
            raise AssertionError(
                f"Expected execution_time 2.5, got: {report.execution_time}"
            )

    def test_execution_report_post_init(self) -> None:
        """Test __post_init__ initializes metadata."""
        report = ExecutionReport(
            experiment_name="test",
            success=True,
            execution_time=1.0,
        )
        if report.metadata != {}:
            raise AssertionError(
                f"Expected empty metadata dict, got: {report.metadata}"
            )

    def test_execution_report_with_metadata(self) -> None:
        """Test ExecutionReport with custom metadata."""
        report = ExecutionReport(
            experiment_name="test",
            success=True,
            execution_time=1.0,
            metadata={"git_commit": "abc123", "timestamp": "2024-01-01"},
        )
        if report.metadata:
            if report.metadata["git_commit"] != "abc123":
                raise AssertionError(
                    f"Expected git_commit 'abc123', got: {report.metadata['git_commit']}"
                )


# =============================================================================
# XPR AGENT ENGINE CORE TESTS
# =============================================================================


class TestXPRAgentEngine:
    """Test XPRAgentEngine core functionality."""

    def test_engine_initialization(self, engine: XPRAgentEngine) -> None:
        """Test engine initializes with empty state."""
        if engine.skills != {}:
            raise AssertionError(f"Expected empty skills dict, got: {engine.skills}")
        if engine.execution_history != []:
            raise AssertionError(
                f"Expected empty execution_history, got: {engine.execution_history}"
            )
        if engine.current_plan is not None:
            raise AssertionError(
                f"Expected current_plan to be None, got: {engine.current_plan}"
            )

    def test_register_skill(
        self, engine: XPRAgentEngine, sample_skill_func: Callable[[int], int]
    ) -> None:
        """Test registering a skill."""
        engine.register_skill("double", sample_skill_func)
        if "double" not in engine.skills:
            raise AssertionError("Expected 'double' in engine.skills")
        if engine.skills["double"] != sample_skill_func:
            raise AssertionError(
                "Expected engine.skills['double'] to equal sample_skill_func"
            )

    def test_register_multiple_skills(self, engine: XPRAgentEngine) -> None:
        """Test registering multiple skills."""
        engine.register_skill("skill1", lambda x: x)
        engine.register_skill("skill2", lambda x: x + 1)
        if len(engine.skills) != 2:
            raise AssertionError(f"Expected 2 skills, got: {len(engine.skills)}")

    def test_execute_skill_success(
        self, engine: XPRAgentEngine, sample_skill_func: Callable[[int], int]
    ) -> None:
        """Test executing a registered skill successfully."""
        engine.register_skill("double", sample_skill_func)
        result = engine.execute_skill("double", 5)

        if not result.success:
            raise AssertionError("Result should be successful")
        if result.skill_type != "double":
            raise AssertionError(
                f"Expected skill_type 'double', got: {result.skill_type}"
            )
        if result.result != 10:
            raise AssertionError(f"Expected result 10, got: {result.result}")
        if result.error is not None:
            raise AssertionError(f"Expected error to be None, got: {result.error}")
        if result.confidence != 0.8:
            raise AssertionError(f"Expected confidence 0.8, got: {result.confidence}")

    def test_execute_skill_not_found(self, engine: XPRAgentEngine) -> None:
        """Test executing a non-existent skill."""
        result = engine.execute_skill("nonexistent")

        if result.success:
            raise AssertionError("Result should be unsuccessful")
        if result.error != "Skill 'nonexistent' not found":
            raise AssertionError(
                f"Expected error 'Skill 'nonexistent' not found', got: {result.error}"
            )
        if result.confidence != 0.0:
            raise AssertionError(f"Expected confidence 0.0, got: {result.confidence}")

    def test_execute_skill_failure(
        self, engine: XPRAgentEngine, sample_failing_skill: Callable[[int], int]
    ) -> None:
        """Test executing a skill that raises an exception."""
        engine.register_skill("failing", sample_failing_skill)
        result = engine.execute_skill("failing", 5)

        if result.success:
            raise AssertionError("Result should be unsuccessful")
        if result.error is None or "Skill failed" not in result.error:
            raise AssertionError(
                f"Expected error with 'Skill failed', got: {result.error}"
            )
        if result.confidence != 0.0:
            raise AssertionError(f"Expected confidence 0.0, got: {result.confidence}")

    def test_execute_skill_with_kwargs(self, engine: XPRAgentEngine) -> None:
        """Test executing skill with keyword arguments."""

        def skill_with_kwargs(x: int, multiplier: int = 2) -> int:
            return x * multiplier

        engine.register_skill("multiply", skill_with_kwargs)
        result = engine.execute_skill("multiply", 5, multiplier=3)

        if not result.success:
            raise AssertionError("Result should be successful")
        if result.result != 15:
            raise AssertionError(f"Expected result 15, got: {result.result}")

    def test_get_performance_summary_empty(self, engine: XPRAgentEngine) -> None:
        """Test performance summary with no history."""
        summary = engine.get_performance_summary()

        if summary["total_experiments"] != 0:
            raise AssertionError(
                f"Expected total_experiments 0, got: {summary['total_experiments']}"
            )
        if summary["success_rate"] != 0.0:
            raise AssertionError(
                f"Expected success_rate 0.0, got: {summary['success_rate']}"
            )
        if summary["avg_execution_time"] != 0.0:
            raise AssertionError(
                f"Expected avg_execution_time 0.0, got: {summary['avg_execution_time']}"
            )

    def test_get_performance_summary_with_data(
        self, engine: XPRAgentEngine, sample_execution_report: ExecutionReport
    ) -> None:
        """Test performance summary with execution history."""
        engine.add_execution_report(sample_execution_report)

        summary = engine.get_performance_summary()

        if summary["total_experiments"] != 1:
            raise AssertionError(
                f"Expected total_experiments 1, got: {summary['total_experiments']}"
            )
        if summary["success_rate"] != 1.0:
            raise AssertionError(
                f"Expected success_rate 1.0, got: {summary['success_rate']}"
            )
        if summary["avg_execution_time"] != 1.5:
            raise AssertionError(
                f"Expected avg_execution_time 1.5, got: {summary['avg_execution_time']}"
            )

    def test_add_execution_report(self, engine: XPRAgentEngine) -> None:
        """Test adding execution report to history."""
        report = ExecutionReport(
            experiment_name="test_exp",
            success=True,
            execution_time=1.0,
        )
        engine.add_execution_report(report)

        if len(engine.execution_history) != 1:
            raise AssertionError(
                f"Expected 1 execution in history, got: {len(engine.execution_history)}"
            )
        if engine.execution_history[0].experiment_name != "test_exp":
            raise AssertionError(
                f"Expected experiment_name 'test_exp', got: {engine.execution_history[0].experiment_name}"
            )

    def test_add_multiple_execution_reports(self, engine: XPRAgentEngine) -> None:
        """Test adding multiple execution reports."""
        for i in range(5):
            report = ExecutionReport(
                experiment_name=f"exp_{i}",
                success=i % 2 == 0,
                execution_time=float(i),
            )
            engine.add_execution_report(report)

        if len(engine.execution_history) != 5:
            raise AssertionError(
                f"Expected 5 executions in history, got: {len(engine.execution_history)}"
            )
        summary = engine.get_performance_summary()
        if summary["success_rate"] != 0.6:
            raise AssertionError(
                f"Expected success_rate 0.6, got: {summary['success_rate']}"
            )  # 3 out of 5 successful

    def test_set_and_get_current_plan(self, engine: XPRAgentEngine) -> None:
        """Test setting and getting current plan."""
        plan = {"hypothesis": "Test hypothesis", "steps": ["step1", "step2"]}
        engine.set_current_plan(plan)

        if engine.get_current_plan() != plan:
            raise AssertionError("Expected current_plan to equal plan")

    def test_get_current_plan_none(self, engine: XPRAgentEngine) -> None:
        """Test getting plan when none is set."""
        if engine.get_current_plan() is not None:
            raise AssertionError("Expected current_plan to be None")

    def test_plan_experiment_with_lr(self, engine: XPRAgentEngine) -> None:
        """Test experiment planning with learning rate parameter."""
        current_params = {"lr": 0.01, "epochs": 10}
        result = engine.plan_experiment("optimize_learning", current_params)

        if not result.success:
            raise AssertionError("Result should be successful")
        if "modifications" not in str(result.result):
            raise AssertionError("Expected 'modifications' in result.result")
        # lr gets multiplied by 0.9 (9% decrease)
        if isinstance(result.result, dict):
            if (
                not pytest.approx(result.result["modifications"]["lr"], rel=1e-3)
                == 0.009
            ):
                raise AssertionError(
                    f"Expected lr approximately 0.009, got: {result.result['modifications']['lr']}"
                )

    def test_plan_experiment_with_epochs(self, engine: XPRAgentEngine) -> None:
        """Test experiment planning with epochs parameter."""
        current_params = {"epochs": 100}
        result = engine.plan_experiment("optimize_training", current_params)

        if not result.success:
            raise AssertionError("Result should be successful")
        if isinstance(result.result, dict):
            if result.result["modifications"]["epochs"] != 110:
                raise AssertionError(
                    f"Expected epochs 110, got: {result.result['modifications']['epochs']}"
                )  # 100 * 1.1

    def test_plan_experiment_empty_params(self, engine: XPRAgentEngine) -> None:
        """Test experiment planning with empty parameters."""
        result = engine.plan_experiment("test_task", {})

        if not result.success:
            raise AssertionError("Result should be successful")
        if isinstance(result.result, dict):
            if result.result["modifications"] != {}:
                raise AssertionError("Expected empty modifications dict")


# =============================================================================
# LLM INTEGRATION TESTS
# =============================================================================


class TestLLMIntegration:
    """Test LLMIntegration functionality."""

    def test_llm_integration_initialization(
        self, llm_integration: LLMIntegration
    ) -> None:
        """Test LLMIntegration initialization."""
        if llm_integration.preferred_provider != "openai":
            raise AssertionError(
                f"Expected preferred_provider 'openai', got: {llm_integration.preferred_provider}"
            )
        if llm_integration._initialized != {}:
            raise AssertionError(
                f"Expected empty _initialized dict, got: {llm_integration._initialized}"
            )

    def test_llm_integration_custom_provider(self) -> None:
        """Test LLMIntegration with custom provider."""
        integration = LLMIntegration(preferred_provider="local")
        if integration.preferred_provider != "local":
            raise AssertionError(
                f"Expected preferred_provider 'local', got: {integration.preferred_provider}"
            )

    def test_get_provider_config_existing(
        self, llm_integration: LLMIntegration
    ) -> None:
        """Test getting config for existing provider."""
        config = llm_integration._get_provider_config("openai")
        if config is None:
            raise AssertionError("Expected config to not be None")
        if "model" not in config:
            raise AssertionError("Expected 'model' in config")

    def test_get_provider_config_nonexistent(
        self, llm_integration: LLMIntegration
    ) -> None:
        """Test getting config for non-existent provider."""
        config = llm_integration._get_provider_config("nonexistent")
        if config != {}:
            raise AssertionError(f"Expected empty config dict, got: {config}")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_initialize_client_openai_success(
        self, llm_integration: LLMIntegration
    ) -> None:
        """Test initializing OpenAI client with API key."""
        result = llm_integration._initialize_client("openai")
        if result is not True:
            raise AssertionError("Expected result to be True")
        if not llm_integration._initialized.get("openai"):
            raise AssertionError("Expected openai to be initialized")

    def test_initialize_client_openai_no_key(
        self, llm_integration: LLMIntegration
    ) -> None:
        """Test initializing OpenAI client without API key."""
        # Without an API key, a provider is usable only through litellm, so
        # pin the "litellm unavailable" case this test is about
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("xpr_agent_engine.litellm", None),
        ):
            result = llm_integration._initialize_client("openai")
            if result:
                raise AssertionError("Expected result to be False")

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
    def test_initialize_client_anthropic_raises_valueerror(
        self, llm_integration: LLMIntegration
    ) -> None:
        """Test that Anthropic raises ValueError as it's not in LLM_PROVIDERS."""
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            llm_integration._initialize_client("anthropic")

    def test_initialize_client_local(self, llm_integration: LLMIntegration) -> None:
        """Test initializing local provider."""
        result = llm_integration._initialize_client("local")
        if result is not True:
            raise AssertionError("Expected result to be True")

    def test_initialize_client_unsupported_raises_valueerror(
        self, llm_integration: LLMIntegration
    ) -> None:
        """Test that unsupported provider raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            llm_integration._initialize_client("unsupported_provider")

    def test_get_client_initializes_when_needed(
        self, llm_integration: LLMIntegration
    ) -> None:
        """Test get_client initializes client if not already done."""
        with patch.object(
            llm_integration, "_initialize_client", return_value=True
        ) as mock_init:
            result = llm_integration.get_client("openai")
            mock_init.assert_called_once_with("openai")
            if result is not True:
                raise AssertionError("Expected result to be True")

    def test_get_client_already_initialized(
        self, llm_integration: LLMIntegration
    ) -> None:
        """Test get_client returns True when already initialized."""
        llm_integration._initialized["openai"] = True
        result = llm_integration.get_client("openai")
        if result is not True:
            raise AssertionError("Expected result to be True")

    def test_generate_text_provider_not_initialized(
        self, llm_integration: LLMIntegration
    ) -> None:
        """Test generate_text when provider not initialized."""
        with patch.object(llm_integration, "_initialize_client", return_value=False):
            result = llm_integration.generate_text("test prompt", provider="openai")
            if result != "":
                raise AssertionError("Expected result to be empty string")

    def test_generate_text_no_litellm(self, llm_integration: LLMIntegration) -> None:
        """Test generate_text when litellm is not available."""
        with patch("xpr_agent_engine.litellm", None):
            with patch.object(llm_integration, "_initialized", {"openai": True}):
                result = llm_integration.generate_text("test", provider="openai")
                if result != "":
                    raise AssertionError("Expected result to be empty string")


# =============================================================================
# ENHANCED XPR AGENT ENGINE TESTS
# =============================================================================


class TestEnhancedXPRAgentEngine:
    """Test EnhancedXPRAgentEngine functionality."""

    def test_enhanced_engine_initialization(self) -> None:
        """Test EnhancedXPRAgentEngine initialization."""
        engine = EnhancedXPRAgentEngine()
        if engine.llm_integration is None:
            raise AssertionError("Expected llm_integration to not be None")
        if not isinstance(engine.llm_integration, LLMIntegration):
            raise AssertionError(
                "Expected llm_integration to be LLMIntegration instance"
            )

    def test_enhanced_engine_skills_registered(self) -> None:
        """Test that enhanced skills are registered on init."""
        engine = EnhancedXPRAgentEngine()
        if "plan_experiment" not in engine.skills:
            raise AssertionError("Expected 'plan_experiment' in engine.skills")
        if "llm_plan_generation" not in engine.skills:
            raise AssertionError("Expected 'llm_plan_generation' in engine.skills")


# =============================================================================
# XPR AGENT ENGINE ENHANCED TESTS
# =============================================================================


class TestXPRAgentEngineEnhanced:
    """Test XPRAgentEngineEnhanced advanced features."""

    def test_enhanced_engine_initialization(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test XPRAgentEngineEnhanced initialization."""
        if enhanced_engine.llm_integration is None:
            raise AssertionError("Expected llm_integration to not be None")
        if enhanced_engine.llm_providers != {}:
            raise AssertionError(
                f"Expected empty llm_providers dict, got: {enhanced_engine.llm_providers}"
            )
        if enhanced_engine.performance_history != {}:
            raise AssertionError(
                f"Expected empty performance_history dict, got: {enhanced_engine.performance_history}"
            )
        if enhanced_engine.skill_dependencies != {}:
            raise AssertionError(
                f"Expected empty skill_dependencies dict, got: {enhanced_engine.skill_dependencies}"
            )
        if enhanced_engine.optimization_strategies != {}:
            raise AssertionError(
                f"Expected empty optimization_strategies dict, got: {enhanced_engine.optimization_strategies}"
            )

    def test_register_llm_provider(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test registering an LLM provider."""
        config = {"model": "gpt-4", "api_key": "test"}
        enhanced_engine.register_llm_provider("custom", config)

        if "custom" not in enhanced_engine.llm_providers:
            raise AssertionError("Expected 'custom' in llm_providers")
        if enhanced_engine.llm_providers["custom"] != config:
            raise AssertionError("Expected llm_providers['custom'] to equal config")

    def test_set_optimization_strategy(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test setting optimization strategy."""
        strategy = {"learning_rate": 0.01, "batch_size": 32}
        enhanced_engine.set_optimization_strategy("training", strategy)

        if "training" not in enhanced_engine.optimization_strategies:
            raise AssertionError("Expected 'training' in optimization_strategies")
        if enhanced_engine.optimization_strategies["training"] != strategy:
            raise AssertionError(
                "Expected optimization_strategies['training'] to equal strategy"
            )

    def test_analyze_performance_trend_no_history(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test analyzing trend with no history."""
        result = enhanced_engine.analyze_performance_trend("unknown_type")

        if result["trend"] != 0.0:
            raise AssertionError(f"Expected trend 0.0, got: {result['trend']}")
        if result["volatility"] != 0.0:
            raise AssertionError(
                f"Expected volatility 0.0, got: {result['volatility']}"
            )

    def test_analyze_performance_trend_with_history(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test analyzing trend with performance history."""
        enhanced_engine.performance_history["test_type"] = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = enhanced_engine.analyze_performance_trend("test_type", window_size=3)

        if result["trend"] <= 0:
            raise AssertionError(
                f"Expected trend > 0, got: {result['trend']}"
            )  # Upward trend
        if result["volatility"] < 0:
            raise AssertionError(
                f"Expected volatility >= 0, got: {result['volatility']}"
            )

    def test_analyze_performance_trend_with_window_larger_than_data(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test analyzing trend when window is larger than data."""
        enhanced_engine.performance_history["test"] = [0.5, 0.6]
        result = enhanced_engine.analyze_performance_trend("test", window_size=10)

        if result["trend"] != 0.55:  # Average of [0.5, 0.6]
            raise AssertionError(f"Expected trend 0.55, got: {result['trend']}")

    def test_xpr_plan_experiment_success(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test XPR plan experiment generation."""
        current_params = {"lr": 0.01, "epochs": 100}
        result = enhanced_engine.xpr_plan_experiment("optimize", current_params)

        if not isinstance(result, XPRSkillResult):
            raise AssertionError("Result should be XPRSkillResult instance")
        if not result.success:
            raise AssertionError("Result should be successful")
        if result.skill_type != XPRSkillType.PLAN_GENERATION.value:
            raise AssertionError(
                f"Expected skill_type 'plan_generation', got: {result.skill_type}"
            )
        if "modifications" not in result.result:
            raise AssertionError("Expected 'modifications' in result.result")

    def test_xpr_plan_experiment_with_performance_history(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test XPR plan with existing performance history."""
        enhanced_engine.performance_history["optimize"] = [0.1, 0.15, 0.2, 0.25, 0.3]
        current_params = {"lr": 0.01}
        result = enhanced_engine.xpr_plan_experiment("optimize", current_params)

        if not result.success:
            raise AssertionError("Result should be successful")
        if result.metadata:
            if not result.metadata.get("adaptive"):
                raise AssertionError("Expected metadata.get('adaptive') to be True")

    def test_xpr_plan_experiment_no_improving_trend(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test XPR plan when trend is not improving."""
        enhanced_engine.performance_history["optimize"] = [0.5, 0.4, 0.3, 0.2]
        current_params = {"epochs": 100}
        result = enhanced_engine.xpr_plan_experiment("optimize", current_params)

        if not result.success:
            raise AssertionError("Result should be successful")
        if isinstance(result.result, dict):
            modified_epochs = result.result["modifications"]["epochs"]
            if not isinstance(modified_epochs, (int, float)):
                raise AssertionError("Expected modified_epochs to be int or float")

    def test_xpr_plan_experiment_exception_handling(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test XPR plan handles exceptions gracefully."""
        # Pass invalid data to trigger exception handling
        with patch.object(
            enhanced_engine,
            "analyze_performance_trend",
            side_effect=Exception("Test error"),
        ):
            result = enhanced_engine.xpr_plan_experiment("test", {})

            if not isinstance(result, XPRSkillResult):
                raise AssertionError("Result should be XPRSkillResult instance")
            if result.success:
                raise AssertionError("Result should be unsuccessful")
            if result.error is None or "not found" not in result.error:
                raise AssertionError(
                    f"Expected error with 'not found', got: {result.error}"
                )

    def test_extract_missing_module_success(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test extracting missing module from error."""
        error_msg = "No module named 'numpy'"
        result = enhanced_engine._extract_missing_module(error_msg)

        if result != "numpy":
            raise AssertionError(f"Expected result 'numpy', got: {result}")

    def test_extract_missing_module_no_match(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test extracting module when no match found."""
        error_msg = "Some other error message"
        result = enhanced_engine._extract_missing_module(error_msg)

        if result is not None:
            raise AssertionError("Expected result to be None")

    def test_xpr_job_debug_import_error(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test job debug with ImportError."""
        experiment_data = {
            "error": "ImportError: No module named 'missing_module'",
            "experiment": "test_exp",
            "file": "/path/to/file.py",
        }
        result = enhanced_engine.xpr_job_debug(experiment_data)

        if not result.success:
            raise AssertionError("Result should be successful")
        if isinstance(result.result, dict):
            if result.result["error_type"] != "import_error":
                raise AssertionError(
                    f"Expected error_type 'import_error', got: {result.result['error_type']}"
                )
            if "missing_module" not in result.result["dependencies"]:
                raise AssertionError("Expected 'missing_module' in dependencies")

    def test_xpr_job_debug_module_not_found(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test job debug with ModuleNotFoundError."""
        experiment_data = {
            "error": "ModuleNotFoundError: No module named 'test'",
            "experiment": "test",
            "file": "test.py",
        }
        result = enhanced_engine.xpr_job_debug(experiment_data)

        if not result.success:
            raise AssertionError("Result should be successful")
        if isinstance(result.result, dict):
            if result.result["error_type"] != "module_not_found":
                raise AssertionError(
                    f"Expected error_type 'module_not_found', got: {result.result['error_type']}"
                )

    def test_xpr_job_debug_permission_error(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test job debug with PermissionError."""
        experiment_data = {
            "error": "PermissionError: [Errno 13] Permission denied",
            "experiment": "test",
            "file": "/etc/passwd",
        }
        result = enhanced_engine.xpr_job_debug(experiment_data)

        if not result.success:
            raise AssertionError("Result should be successful")
        if isinstance(result.result, dict):
            if result.result["error_type"] != "permission_error":
                raise AssertionError(
                    f"Expected error_type 'permission_error', got: {result.result['error_type']}"
                )

    def test_xpr_job_debug_timeout(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test job debug with timeout error."""
        experiment_data = {
            "error": "Request timed out after 30 seconds",
            "experiment": "test",
        }
        result = enhanced_engine.xpr_job_debug(experiment_data)

        if not result.success:
            raise AssertionError("Result should be successful")
        if isinstance(result.result, dict):
            if result.result["error_type"] != "timeout_error":
                raise AssertionError(
                    f"Expected error_type 'timeout_error', got: {result.result['error_type']}"
                )

    def test_xpr_job_debug_generic_error(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test job debug with generic error."""
        experiment_data = {
            "error": "Something unexpected happened",
            "experiment": "test",
        }
        result = enhanced_engine.xpr_job_debug(experiment_data)

        if not result.success:
            raise AssertionError("Result should be successful")
        if isinstance(result.result, dict):
            if result.result["error_type"] != "generic_error":
                raise AssertionError(
                    f"Expected error_type 'generic_error', got: {result.result['error_type']}"
                )

    def test_xpr_job_debug_exception_handling(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test job debug exception handling."""

        # Trigger an exception during dictionary access
        class BadDict(dict):
            def get(self, *args):
                raise ValueError("Skill failed")

        result = enhanced_engine.xpr_job_debug(BadDict())

        if result.success:
            raise AssertionError("Result should be unsuccessful")
        if result.error is None:
            raise AssertionError("Expected error to not be None")

    def test_xpr_issue_fix_syntax_error(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test issue fix with SyntaxError."""
        experiment_data = {
            "error": "SyntaxError: invalid syntax at line 42",
            "experiment": "test",
        }
        result = enhanced_engine.xpr_issue_fix(experiment_data)

        if not result.success:
            raise AssertionError("Result should be successful")
        if isinstance(result.result, dict):
            if result.result["fix_type"] != "syntax_fix":
                raise AssertionError(
                    f"Expected fix_type 'syntax_fix', got: {result.result['fix_type']}"
                )
            if "42" not in result.result["generated_code"]:
                raise AssertionError("Expected '42' in generated_code")

    def test_xpr_issue_fix_file_not_found(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test issue fix with FileNotFoundError."""
        experiment_data = {
            "error": "FileNotFoundError: [Errno 2] No such file or directory: 'missing.txt'",
            "experiment": "test",
        }
        result = enhanced_engine.xpr_issue_fix(experiment_data)

        if not result.success:
            raise AssertionError("Result should be successful")
        if result.result["fix_type"] != "file_creation_fix":
            raise AssertionError(
                f"Expected fix_type 'file_creation_fix', got: {result.result['fix_type']}"
            )
        if "missing.txt" not in result.result["generated_code"]:
            raise AssertionError("Expected 'missing.txt' in generated_code")

    def test_xpr_issue_fix_import_error(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test issue fix with ImportError."""
        experiment_data = {
            "error": "ImportError: No module named 'requests'",
            "experiment": "test",
        }
        result = enhanced_engine.xpr_issue_fix(experiment_data)

        if not result.success:
            raise AssertionError("Result should be successful")
        if isinstance(result.result, dict):
            if result.result["fix_type"] != "import_fix":
                raise AssertionError(
                    f"Expected fix_type 'import_fix', got: {result.result['fix_type']}"
                )
            if "import requests" not in result.result["generated_code"]:
                raise AssertionError("Expected 'import requests' in generated_code")

    def test_xpr_issue_fix_exception_handling(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test issue fix exception handling."""

        # Trigger an exception during dictionary access
        class BadDict(dict):
            def get(self, *args):
                raise ValueError("Skill failed")

        result = enhanced_engine.xpr_issue_fix(BadDict())

        if result.success:
            raise AssertionError("Result should be unsuccessful")
        if result.error is None:
            raise AssertionError("Expected error to not be None")

    def test_xpr_issue_report_success(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test issue report generation."""
        experiment_data = {
            "error": "ImportError: No module named 'missing'",
            "experiment": "test_exp",
            "file": "/path/to/file.py",
            "metrics": {"accuracy": 0.5},
        }
        result = enhanced_engine.xpr_issue_report(experiment_data)

        if not result.success:
            raise AssertionError("Result should be successful")
        if isinstance(result.result, dict):
            assert "experiment_name" in result.result  # nosec: B101 - Test assertion
            assert "severity" in result.result  # nosec: B101 - Test assertion
            assert "recommendations" in result.result  # nosec: B101 - Test assertion

    def test_xpr_issue_report_exception_handling(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test issue report exception handling."""

        # Trigger an exception during dictionary access
        class BadDict(dict):
            def get(self, *args):
                raise ValueError("Skill failed")

        result = enhanced_engine.xpr_issue_report(BadDict())

        if result.success:
            raise AssertionError("Result should be unsuccessful")
        if result.error is None:
            raise AssertionError("Expected error to not be None")

    def test_assess_severity_critical(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test severity assessment for critical errors."""
        if enhanced_engine._assess_severity("Critical system failure") != "critical":
            raise AssertionError(
                "Expected severity 'critical' for Critical system failure"
            )
        if enhanced_engine._assess_severity("Fatal error occurred") != "critical":
            raise AssertionError(
                "Expected severity 'critical' for Fatal error occurred"
            )
        if enhanced_engine._assess_severity("System crash") != "critical":
            raise AssertionError("Expected severity 'critical' for System crash")

    def test_assess_severity_high(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test severity assessment for high severity."""
        if enhanced_engine._assess_severity("Error in processing") != "high":
            raise AssertionError("Expected severity 'high' for Error in processing")
        if enhanced_engine._assess_severity("Exception raised") != "high":
            raise AssertionError("Expected severity 'high' for Exception raised")
        if enhanced_engine._assess_severity("Task failed") != "high":
            raise AssertionError("Expected severity 'high' for Task failed")

    def test_assess_severity_medium(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test severity assessment for medium severity."""
        assert (  # nosec: B101 - Test assertion
            enhanced_engine._assess_severity("Warning: something happened") == "medium"
        )
        if enhanced_engine._assess_severity("Notice") != "medium":
            raise AssertionError("Expected severity 'medium' for Notice")

    def test_generate_recommendations_timeout(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test recommendation generation for timeout."""
        recs = enhanced_engine._generate_recommendations("Request timeout", {})

        if not any("timeout" in r.lower() for r in recs):
            raise AssertionError(
                "Expected at least one recommendation to contain 'timeout'"
            )
        if not any("duration" in r.lower() for r in recs):
            raise AssertionError(
                "Expected at least one recommendation to contain 'duration'"
            )

    def test_generate_recommendations_memory(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test recommendation generation for memory issues."""
        recs = enhanced_engine._generate_recommendations("Out of memory error", {})

        if not any("memory" in r.lower() for r in recs):
            raise AssertionError(
                "Expected at least one recommendation to contain 'memory'"
            )
        if not any("batch" in r.lower() for r in recs):
            raise AssertionError(
                "Expected at least one recommendation to contain 'batch'"
            )

    def test_generate_recommendations_permission(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test recommendation generation for permission issues."""
        recs = enhanced_engine._generate_recommendations("Permission denied", {})

        if not any("permission" in r.lower() for r in recs):
            raise AssertionError(
                "Expected at least one recommendation to contain 'permission'"
            )

    def test_generate_recovery_steps(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test recovery steps generation."""
        steps = enhanced_engine._generate_recovery_steps("any error")

        if len(steps) != 5:
            raise AssertionError(f"Expected 5 steps, got: {len(steps)}")
        if not all(isinstance(step, str) for step in steps):
            raise AssertionError("All steps should be strings")

    def test_analyze_root_cause_import(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test root cause analysis for ImportError."""
        cause = enhanced_engine._analyze_root_cause("ImportError: No module named 'x'")
        if "Missing or incompatible dependency" not in cause:
            raise AssertionError(
                f"Expected 'Missing or incompatible dependency' in cause, got: {cause}"
            )

    def test_analyze_root_cause_permission(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test root cause analysis for PermissionError."""
        cause = enhanced_engine._analyze_root_cause("PermissionError: denied")
        if "Insufficient file system permissions" not in cause:
            raise AssertionError(
                f"Expected 'Insufficient file system permissions' in cause, got: {cause}"
            )

    def test_analyze_root_cause_timeout(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test root cause analysis for timeout."""
        cause = enhanced_engine._analyze_root_cause("timeout occurred")
        if "Resource exhaustion" not in cause:
            raise AssertionError(
                f"Expected 'Resource exhaustion' in cause, got: {cause}"
            )

    def test_analyze_root_cause_generic(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test root cause analysis for generic errors."""
        cause = enhanced_engine._analyze_root_cause("Unknown error")
        if "Unknown error requiring investigation" not in cause:
            raise AssertionError(
                f"Expected 'Unknown error requiring investigation' in cause, got: {cause}"
            )

    def test_xpr_skill_chain_success(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test successful skill chaining."""
        # Register skills that work with dictionaries
        enhanced_engine.register_skill("skill1", lambda x: {"value": x["value"] + 1})
        enhanced_engine.register_skill("skill2", lambda x: {"value": x["value"] * 2})

        results = enhanced_engine.xpr_skill_chain({"value": 5}, ["skill1", "skill2"])

        if len(results) != 2:
            raise AssertionError(f"Expected 2 results, got {len(results)}")
        if not all(isinstance(r, XPRSkillResult) for r in results):
            raise AssertionError("All results should be XPRSkillResult instances")
        if not results[0].success:
            raise AssertionError("First result should be successful")
        if not results[1].success:
            raise AssertionError("Second result should be successful")

    def test_xpr_skill_chain_skill_not_found(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test skill chain with missing skill."""
        results = enhanced_engine.xpr_skill_chain({}, ["nonexistent_skill"])

        if len(results) != 1:
            raise AssertionError(f"Expected 1 result, got: {len(results)}")
        if results[0].success:
            raise AssertionError("First result should be unsuccessful")
        if results[0].error is None or "not found" not in results[0].error:
            raise AssertionError(
                f"Expected error with 'not found', got: {results[0].error}"
            )

    def test_xpr_skill_chain_exception_handling(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test skill chain exception handling."""

        def failing_skill(x):
            raise ValueError("Skill failed")

        enhanced_engine.register_skill("failing", failing_skill)
        results = enhanced_engine.xpr_skill_chain({}, ["failing"])

        if len(results) != 1:
            raise AssertionError(f"Expected 1 result, got: {len(results)}")
        if results[0].success:
            raise AssertionError("First result should be unsuccessful")

    def test_get_performance_summary_enhanced(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test enhanced performance summary."""
        # Add some execution reports
        for i in range(5):
            report = ExecutionReport(
                experiment_name=f"exp_{i}",
                success=i % 2 == 0,
                execution_time=float(i + 1),
                metadata={"adaptive": i % 3 == 0},
            )
            enhanced_engine.add_execution_report(report)

        summary = enhanced_engine.get_performance_summary()

        assert "avg_confidence" in summary  # nosec: B101 - Test assertion
        assert "skill_success_rate" in summary  # nosec: B101 - Test assertion
        assert "adaptive_optimizations" in summary  # nosec: B101 - Test assertion
        assert (
            summary["adaptive_optimizations"] == 2
        )  # i=0,3  # nosec: B101 - Test assertion


# =============================================================================
# REGISTER XPR SKILLS TESTS
# =============================================================================


class TestRegisterXPRSkills:
    """Test register_xpr_skills function."""

    def test_register_skills_basic(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test registering XPR skills on enhanced engine."""
        register_xpr_skills(enhanced_engine)
        # Should register plan_experiment
        assert (
            "plan_experiment" in enhanced_engine.skills
        )  # nosec: B101 - Test assertion
        # XPRAgentEngineEnhanced has xpr_* methods, so they should be registered
        assert "xpr_job_debug" in enhanced_engine.skills  # nosec: B101 - Test assertion
        assert "xpr_issue_fix" in enhanced_engine.skills  # nosec: B101 - Test assertion
        assert (
            "xpr_issue_report" in enhanced_engine.skills
        )  # nosec: B101 - Test assertion
        assert (
            "xpr_skill_chain" in enhanced_engine.skills
        )  # nosec: B101 - Test assertion
        assert (
            "xpr_plan_experiment" in enhanced_engine.skills
        )  # nosec: B101 - Test assertion

    def test_register_skills_on_base_engine(self, engine: XPRAgentEngine) -> None:
        """Test registering on base engine without xpr methods."""
        register_xpr_skills(engine)  # type: ignore[arg-type]

        # Should only register plan_experiment
        if "plan_experiment" not in engine.skills:
            raise AssertionError("Expected 'plan_experiment' in engine.skills")
        # These should not be registered on base engine
        assert "xpr_job_debug" not in engine.skills  # nosec: B101 - Test assertion
        assert "xpr_issue_fix" not in engine.skills  # nosec: B101 - Test assertion
        assert "xpr_issue_report" not in engine.skills  # nosec: B101 - Test assertion
        assert "xpr_skill_chain" not in engine.skills  # nosec: B101 - Test assertion
        assert (
            "xpr_plan_experiment" not in engine.skills
        )  # nosec: B101 - Test assertion

        # Should only register plan_experiment
        if "plan_experiment" not in engine.skills:
            raise AssertionError("Expected 'plan_experiment' in engine.skills")
        # These should not be registered on base engine
        assert "xpr_job_debug" not in engine.skills  # nosec: B101 - Test assertion


# =============================================================================
# XPR SKILL TYPE ENUM TESTS
# =============================================================================


class TestXPRSkillType:
    """Test XPRSkillType enum."""

    def test_skill_type_values(self) -> None:
        """Test all skill type enum values."""
        assert (
            XPRSkillType.PLAN_GENERATION.value == "plan_generation"
        )  # nosec: B101 - Test assertion
        assert (
            XPRSkillType.EXECUTION.value == "execution"
        )  # nosec: B101 - Test assertion
        assert XPRSkillType.ANALYSIS.value == "analysis"  # nosec: B101 - Test assertion
        assert (
            XPRSkillType.JOB_DEBUG.value == "job_debug"
        )  # nosec: B101 - Test assertion
        assert (
            XPRSkillType.ISSUE_FIX.value == "issue_fix"
        )  # nosec: B101 - Test assertion
        assert (
            XPRSkillType.ISSUE_REPORT.value == "issue_report"
        )  # nosec: B101 - Test assertion

    def test_xpr_skill_result_with_skill_type(self) -> None:
        """Test XPRSkillResult using enum."""
        result = XPRSkillResult(
            success=True,
            skill_type=XPRSkillType.PLAN_GENERATION.value,
            result={},
        )
        if result.skill_type != "plan_generation":
            raise AssertionError(
                f"Expected skill_type 'plan_generation', got: {result.skill_type}"
            )


# =============================================================================
# EDGE CASE AND ERROR HANDLING TESTS
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_execute_skill_with_none_args(self, engine: XPRAgentEngine) -> None:
        """Test executing skill with None arguments."""

        def skill_with_none(x=None):
            return x if x is not None else "default"

        engine.register_skill("none_test", skill_with_none)
        result = engine.execute_skill("none_test", None)

        if not result.success:
            raise AssertionError("Result should be successful")
        if result.result != "default":
            raise AssertionError(f"Expected result 'default', got: {result.result}")

    def test_execute_skill_returning_none(self, engine: XPRAgentEngine) -> None:
        """Test executing skill that returns None."""
        engine.register_skill("returns_none", lambda: None)
        result = engine.execute_skill("returns_none")

        if not result.success:
            raise AssertionError("Result should be successful")
        if result.result is not None:
            raise AssertionError(f"Expected result to be None, got: {result.result}")

    def test_performance_history_with_empty_list(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test performance trend with empty list."""
        enhanced_engine.performance_history["test"] = []
        result = enhanced_engine.analyze_performance_trend("test")

        if result["trend"] != 0.0:
            raise AssertionError(f"Expected trend 0.0, got: {result['trend']}")
        if result["volatility"] != 0.0:
            raise AssertionError(
                f"Expected volatility 0.0, got: {result['volatility']}"
            )

    def test_performance_history_with_single_value(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test performance trend with single value."""
        enhanced_engine.performance_history["test"] = [0.5]
        result = enhanced_engine.analyze_performance_trend("test")

        # With single value, volatility is 0 and trend depends on implementation
        # Implementation requires >= 2 values for non-zero trend calculation
        assert result["volatility"] == pytest.approx(
            0.0, abs=1e-3
        )  # nosec: B101 - Test assertion

    def test_skill_result_with_complex_metadata(self) -> None:
        """Test SkillResult with nested metadata."""
        complex_metadata = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "number": 42,
        }
        result = SkillResult(
            success=True,
            skill_type="test",
            metadata=complex_metadata,
        )
        if result.metadata and isinstance(result.metadata, dict):
            nested = result.metadata.get("nested")
            if isinstance(nested, dict):
                assert nested["key"] == "value"  # nosec: B101 - Test assertion

    def test_xpr_skill_result_post_init(self) -> None:
        """Test XPRSkillResult __post_init__ with None values."""
        result = XPRSkillResult(success=True, skill_type="test")

        assert result.metadata == {}  # nosec: B101 - Test assertion
        assert result.dependencies == []  # nosec: B101 - Test assertion
        assert result.recommendations == []  # nosec: B101 - Test assertion

    @pytest.mark.parametrize(
        "error_msg,expected_type",
        [
            ("ImportError: No module named 'x'", "import_error"),
            ("ModuleNotFoundError: No module named 'y'", "module_not_found"),
            ("PermissionError: denied", "permission_error"),
            ("timeout error occurred", "timeout_error"),
            ("Unknown error", "generic_error"),
        ],
    )
    def test_job_debug_error_pattern_matching(
        self,
        enhanced_engine: XPRAgentEngineEnhanced,
        error_msg: str,
        expected_type: str,
    ) -> None:
        """Test error pattern matching for various error types."""
        result = enhanced_engine.xpr_job_debug(
            {
                "error": error_msg,
                "experiment": "test",
            }
        )

        if not result.success:
            raise AssertionError("Result should be successful")
        assert (
            result.result["error_type"] == expected_type
        )  # nosec: B101 - Test assertion


# =============================================================================
# BOUNDARY VALUE TESTS
# =============================================================================


class TestBoundaryValues:
    """Test boundary values and extreme inputs."""

    def test_execute_skill_with_large_number(self, engine: XPRAgentEngine) -> None:
        """Test executing skill with very large number."""
        engine.register_skill("identity", lambda x: x)
        result = engine.execute_skill("identity", 1e308)

        if not result.success:
            raise AssertionError("Result should be successful")
        if result.result != 1e308:
            raise AssertionError(f"Expected result 1e308, got: {result.result}")

    def test_execute_skill_with_zero(self, engine: XPRAgentEngine) -> None:
        """Test executing skill with zero."""
        engine.register_skill("double", lambda x: x * 2)
        result = engine.execute_skill("double", 0)

        if not result.success:
            raise AssertionError("Result should be successful")
        if result.result != 0:
            raise AssertionError(f"Expected result 0, got: {result.result}")

    def test_execution_report_with_zero_time(self) -> None:
        """Test ExecutionReport with zero execution time."""
        report = ExecutionReport(
            experiment_name="test",
            success=True,
            execution_time=0.0,
        )
        if report.execution_time != 0.0:
            raise AssertionError(
                f"Expected execution_time 0.0, got: {report.execution_time}"
            )

    def test_performance_trend_with_negative_values(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test performance trend with negative values."""
        enhanced_engine.performance_history["test"] = [-0.5, -0.4, -0.3]
        result = enhanced_engine.analyze_performance_trend("test")

        if not pytest.approx(result["trend"], rel=1e-3) == -0.4:  # Average
            raise AssertionError(f"Expected trend approx -0.4, got: {result['trend']}")
        if result["volatility"] < 0:
            raise AssertionError(
                f"Expected volatility >= 0, got: {result['volatility']}"
            )

    def test_performance_trend_with_mixed_values(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test performance trend with mixed positive/negative values."""
        enhanced_engine.performance_history["test"] = [-0.5, 0.0, 0.5]
        result = enhanced_engine.analyze_performance_trend("test")

        if result["trend"] != 0.0:  # Average
            raise AssertionError(f"Expected trend 0.0, got: {result['trend']}")

    def test_xpr_skill_result_with_edge_confidence_values(self) -> None:
        """Test XPRSkillResult with edge confidence values."""
        result_min = XPRSkillResult(success=True, skill_type="test", confidence=0.0)
        result_max = XPRSkillResult(success=True, skill_type="test", confidence=1.0)

        if result_min.confidence != 0.0:
            raise AssertionError(
                f"Expected confidence 0.0, got: {result_min.confidence}"
            )
        if result_max.confidence != 1.0:
            raise AssertionError(
                f"Expected confidence 1.0, got: {result_max.confidence}"
            )


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestXPRIntegration:
    """Integration tests for XPR Agent Engine."""

    def test_full_workflow_simulation(self) -> None:
        """Test a full workflow simulation."""
        engine = XPRAgentEngineEnhanced()
        register_xpr_skills(engine)

        # Set up optimization strategy
        engine.set_optimization_strategy("training", {"lr": 0.01, "epochs": 100})

        # Simulate some performance history
        engine.performance_history["training"] = [0.1, 0.15, 0.2, 0.25]

        # Generate plan
        plan_result = engine.xpr_plan_experiment(
            "training", {"lr": 0.01, "epochs": 100}
        )
        assert plan_result.success is True  # nosec: B101 - Test assertion

        # Simulate execution and add report
        report = ExecutionReport(
            experiment_name="training_run_1",
            success=True,
            execution_time=10.5,
            metadata={"adaptive": True},
        )
        engine.add_execution_report(report)

        # Check performance summary
        summary = engine.get_performance_summary()
        if summary["total_experiments"] != 1:
            raise AssertionError(
                f"Expected total_experiments 1, got: {summary['total_experiments']}"
            )
        # avg_confidence should be 0 since we didn't set confidence on the report
        assert "avg_confidence" in summary  # nosec: B101 - Test assertion

    def test_error_recovery_workflow(
        self, enhanced_engine: XPRAgentEngineEnhanced
    ) -> None:
        """Test error detection and recovery workflow."""
        # Simulate an error
        error_data = {
            "error": "ImportError: No module named 'missing_dependency'",
            "experiment": "test_exp",
            "file": "/path/to/experiment.py",
        }

        # Debug the error
        debug_result = enhanced_engine.xpr_job_debug(error_data)
        assert debug_result.success is True  # nosec: B101 - Test assertion
        assert (
            debug_result.result["error_type"] == "import_error"
        )  # nosec: B101 - Test assertion

        # Generate fix
        fix_result = enhanced_engine.xpr_issue_fix(error_data)
        assert fix_result.success is True  # nosec: B101 - Test assertion

        # Generate report
        report_result = enhanced_engine.xpr_issue_report(error_data)
        assert report_result.success is True  # nosec: B101 - Test assertion
        assert (
            report_result.result["severity"] == "high"
        )  # nosec: B101 - Test assertion


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
