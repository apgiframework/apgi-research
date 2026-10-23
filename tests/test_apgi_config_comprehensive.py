"""
Comprehensive test suite for apgi_config.py

This test file covers all major components:
- ConfigManager singleton pattern
- Pydantic schema validation
- Environment variable loading
- File loading (JSON/YAML)
- Configuration migration
- Caching mechanisms
- Legacy compatibility
"""

import json
import os
import tempfile
from pathlib import Path
from typing import TypedDict
from unittest.mock import patch

import pytest

from utils.apgi_config import (
    APGIDynamicalParameters,
    APGIExperimentConfigSchema,
    APGIGlobalConfig,
    APGIMetricsConfigSchema,
    APGISecurityConfigSchema,
    ConfigManager,
    ExperimentConfig,
    SecurityConfig,
    compute_config_checksum,
    load_apgi_params,
    migrate_config,
    reset_config,
    validate_config_integrity,
    validate_startup_config,
)


class ExperimentConfigDict(TypedDict):
    version: str
    experiment_name: str
    tau_S: float
    beta: float
    theta_0: float
    alpha: float
    hierarchical_enabled: bool
    precision_gap_enabled: bool
    neuromodulator_tracking: bool


class SecurityConfigDict(TypedDict):
    audit_enabled: bool
    authz_enabled: bool
    subprocess_allowlist: list[str]
    require_secure_pickle: bool


class MetricsConfigDict(TypedDict):
    profiling_enabled: bool
    performance_budget_ms: float
    log_level: str
    json_logging: bool


class DynamicalConfigDict(TypedDict):
    tau_s: float
    tau_theta: float
    tau_m: float
    theta_0: float
    alpha: float
    beta_som: float
    beta_m: float
    gamma_m: float
    gamma_a: float
    lambda_s: float
    m_0: float
    theta_survival: float
    theta_neutral: float
    rho: float


class TestConfigManager:
    """Test ConfigManager class methods."""

    def test_singleton_pattern(self):
        """Test ConfigManager singleton pattern."""
        # Reset singleton
        reset_config()

        # Get two instances
        config1 = ConfigManager()
        config2 = ConfigManager()

        # Should be the same instance
        assert config1 is config2  # nosec: B101 - Test assertion
        assert config1._initialized  # nosec: B101 - Test assertion
        assert config2._initialized  # nosec: B101 - Test assertion

    def test_config_loading(self, tmp_path):
        """Test configuration loading from files."""
        test_config = {
            "experiment_name": "test_experiment",
            "tau_S": 0.4,
            "beta": 1.8,
            "theta_0": 0.6,
            "alpha": 6.0,
            "hierarchical_enabled": True,
            "precision_gap_enabled": True,
            "neuromodulator_tracking": True,
        }

        # Write test config file
        with open(tmp_path / "test_config.json", "w") as f:
            json.dump(test_config, f)

        # Mock the config file path
        with patch.object(ConfigManager, "_find_config_file") as mock_find:
            mock_find.return_value = tmp_path / "test_config.json"

            config = ConfigManager()
            loaded_config = {
                "experiment_name": config.get("experiment_name"),
                "tau_S": config.get("tau_S"),
                "beta": config.get("beta"),
                "theta_0": config.get("theta_0"),
                "alpha": config.get("alpha"),
                "hierarchical_enabled": config.get("hierarchical_enabled"),
                "precision_gap_enabled": config.get("precision_gap_enabled"),
                "neuromodulator_tracking": config.get("neuromodulator_tracking"),
            }

            # Verify all values were loaded correctly
            for key, expected_value in test_config.items():
                assert (
                    loaded_config[key] == expected_value
                ), f"Failed to load {key}"  # nosec: B101 - Test assertion

            # Verify source tracking
            assert config.get_source(
                "experiment_name"
            ) == str(  # nosec: B101 - Test assertion
                tmp_path / "test_config.json"
            )

    def test_environment_loading(self):
        """Test environment variable loading."""
        test_env_vars = {
            "APGI_EXPERIMENT_TEST_tau_S": "0.45",
            "APGI_EXPERIMENT_TEST_beta": "1.9",
            "APGI_EXPERIMENT_TEST_theta_0": "0.7",
            "APGI_EXPERIMENT_TEST_alpha": "7.0",
            "APGI_EXPERIMENT_TEST_hierarchical_enabled": "false",
            "APGI_EXPERIMENT_TEST_precision_gap_enabled": "false",
            "APGI_EXPERIMENT_TEST_neuromodulator_tracking": "false",
        }

        # Set environment variables
        for key, value in test_env_vars.items():
            os.environ[key] = value

        config = ConfigManager()

        # Verify environment variables take precedence
        for key, expected_value in test_env_vars.items():
            config_key = f"experiment_test_{key.lower()}"
            assert (  # nosec: B101 - Test assertion
                config.get(config_key) == expected_value
            ), f"Environment variable {key} not loaded correctly"

        # Verify source tracking for environment variables
        assert (
            config.get_source(f"experiment_test_{key.lower()}") == f"env:{key}"
        )  # nosec: B101 - Test assertion

    def test_yaml_support(self):
        """Test YAML configuration file support."""
        config = ConfigManager()

        # Test YAML file creation
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as tmp_yaml:
            tmp_yaml.write("""
version: 1.0.0
experiment:
  name: yaml_test
  parameters:
    tau_S: 0.35
""")
            tmp_yaml.flush()

            # Mock the file path
            with patch.object(ConfigManager, "_find_config_file") as mock_find:
                mock_find.return_value = Path(tmp_yaml.name)

                config = ConfigManager()
                loaded_version = config.get("version")
                loaded_experiment = config.get_experiment_config("yaml_test")

                assert loaded_version == "1.0.0"  # nosec: B101 - Test assertion
                assert (
                    loaded_experiment.experiment_name == "yaml_test"
                )  # nosec: B101 - Test assertion
                assert loaded_experiment.tau_S == 0.35  # nosec: B101 - Test assertion

    def test_caching_functionality(self):
        """Test configuration caching mechanisms."""
        config = ConfigManager()

        # Test experiment config caching
        with patch.object(ConfigManager, "get_cached_experiment_config") as mock_get:
            mock_get.return_value = APGIExperimentConfigSchema(
                experiment_name="cached_test",
                tau_S=0.25,
                beta=1.2,
                theta_0=0.4,
                alpha=4.0,
            )

            # First call should use cache
            result1 = config.get_experiment_config("cached_test")
            assert (
                result1.experiment_name == "cached_test"
            )  # nosec: B101 - Test assertion

            # Second call should use cache
            result2 = config.get_experiment_config("cached_test")
            assert (
                result2.experiment_name == "cached_test"
            )  # nosec: B101 - Test assertion

            # Verify cache was used
            mock_get.assert_called_once()

    def test_migration_functionality(self):
        """Test configuration migration functionality."""
        old_config = {
            "version": "0.9.0",
            "experiment_name": "old_experiment",
            "tau_S": 0.3,
        }

        new_config = migrate_config(old_config, "1.0.0")

        # Verify migration added new fields
        assert new_config["security_enabled"] is True  # nosec: B101 - Test assertion
        assert new_config["log_level"] == "INFO"  # nosec: B101 - Test assertion
        assert new_config["version"] == "1.0.0"  # nosec: B101 - Test assertion
        assert (
            new_config["experiment_name"] == "old_experiment"
        )  # nosec: B101 - Test assertion
        assert (
            new_config["tau_S"] == 0.3
        )  # Preserved old value  # nosec: B101 - Test assertion

    def test_legacy_compatibility(self):
        """Test legacy compatibility functions."""
        # Test legacy parameter loading
        legacy_params = load_apgi_params()
        expected_keys = ["tau_S", "beta", "theta_0", "alpha", "gamma_M", "lambda_S"]

        for key in expected_keys:
            assert (
                key in legacy_params
            ), f"Missing legacy key: {key}"  # nosec: B101 - Test assertion
            assert isinstance(
                legacy_params[key], (int, float)
            )  # nosec: B101 - Test assertion

    def test_error_handling(self):
        """Test error handling in configuration loading."""
        config = ConfigManager()

        # Test invalid JSON file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            tmp_file.write("invalid json content {")
            tmp_file.flush()

            with patch.object(ConfigManager, "_find_config_file") as mock_find:
                mock_find.return_value = Path(tmp_file.name)

                # Should not raise exception, should use defaults
                config = ConfigManager()
                assert config.get(
                    "tau_S", 0.35
                )  # Should use default  # nosec: B101 - Test assertion

    def test_config_validation(self):
        """Test configuration validation and integrity."""
        # Test checksum validation
        test_config = {"test_key": "test_value"}
        checksum = compute_config_checksum(test_config, "secret123")

        # Valid checksum
        assert validate_config_integrity(
            test_config, checksum, "secret123"
        )  # nosec: B101 - Test assertion

        # Invalid checksum
        assert not validate_config_integrity(
            test_config, checksum, "wrong_secret"
        )  # nosec: B101 - Test assertion

        # Test startup validation
        valid_config = APGIGlobalConfig(
            version="1.0.0",
            dynamical=APGIDynamicalParameters(),
            security=SecurityConfig(),
            experiment=ExperimentConfig(),
        )

        assert validate_startup_config(valid_config)  # nosec: B101 - Test assertion
        assert not validate_startup_config(
            {"invalid": "config"}
        )  # nosec: B101 - Test assertion


class TestPydanticSchemas:
    """Test Pydantic schema validation."""

    def test_experiment_config_validation(self):
        """Test APGIExperimentConfigSchema validation."""
        # Test valid configuration
        valid_config: ExperimentConfigDict = {
            "version": "1.0.0",
            "experiment_name": "test_experiment",
            "tau_S": 0.35,
            "beta": 1.5,
            "theta_0": 0.5,
            "alpha": 5.5,
            "hierarchical_enabled": True,
            "precision_gap_enabled": True,
            "neuromodulator_tracking": True,
        }

        config = APGIExperimentConfigSchema(
            version=valid_config["version"],
            experiment_name=valid_config["experiment_name"],
            tau_S=valid_config["tau_S"],
            beta=valid_config["beta"],
            theta_0=valid_config["theta_0"],
            alpha=valid_config["alpha"],
            hierarchical_enabled=valid_config["hierarchical_enabled"],
            precision_gap_enabled=valid_config["precision_gap_enabled"],
            neuromodulator_tracking=valid_config["neuromodulator_tracking"],
        )
        assert (
            config.experiment_name == "test_experiment"
        )  # nosec: B101 - Test assertion
        assert config.tau_S == 0.35  # nosec: B101 - Test assertion

        # Test invalid configuration
        with pytest.raises(ValueError):
            APGIExperimentConfigSchema(tau_S=1.0)  # Above max

        with pytest.raises(ValueError):
            APGIExperimentConfigSchema(tau_S=0.1)  # Below min

        with pytest.raises(ValueError):
            APGIExperimentConfigSchema(alpha=15.0)  # Above max

    def test_security_config_validation(self):
        """Test APGISecurityConfigSchema validation."""
        # Test valid configuration
        valid_config: SecurityConfigDict = {
            "audit_enabled": True,
            "authz_enabled": True,
            "subprocess_allowlist": ["git", "pytest", "python"],
            "require_secure_pickle": True,
        }

        config = APGISecurityConfigSchema(
            audit_enabled=valid_config["audit_enabled"],
            authz_enabled=valid_config["authz_enabled"],
            subprocess_allowlist=valid_config["subprocess_allowlist"],
            require_secure_pickle=valid_config["require_secure_pickle"],
        )
        assert config.audit_enabled is True  # nosec: B101 - Test assertion
        assert len(config.subprocess_allowlist) == 3  # nosec: B101 - Test assertion

        # Test invalid allowlist (not a list)
        with pytest.raises(ValueError):
            APGISecurityConfigSchema(subprocess_allowlist=["not_a_list"])

    def test_metrics_config_validation(self):
        """Test APGIMetricsConfigSchema validation."""
        # Test valid configuration
        valid_config: MetricsConfigDict = {
            "profiling_enabled": False,
            "performance_budget_ms": 600000.0,
            "log_level": "INFO",
            "json_logging": True,
        }

        config = APGIMetricsConfigSchema(
            profiling_enabled=valid_config["profiling_enabled"],
            performance_budget_ms=valid_config["performance_budget_ms"],
            log_level=valid_config["log_level"],
            json_logging=valid_config["json_logging"],
        )
        assert config.profiling_enabled is False  # nosec: B101 - Test assertion
        assert config.performance_budget_ms == 600000.0  # nosec: B101 - Test assertion

        # Test invalid log level
        with pytest.raises(ValueError):
            APGIMetricsConfigSchema(log_level="INVALID_LEVEL")

    def test_dynamical_parameters_validation(self):
        """Test APGIDynamicalParameters validation."""
        # Test valid configuration
        valid_config: DynamicalConfigDict = {
            "tau_s": 0.35,
            "tau_theta": 30.0,
            "tau_m": 1.5,
            "theta_0": 0.5,
            "alpha": 5.5,
            "beta_som": 1.5,
            "beta_m": 1.0,
            "gamma_m": -0.3,
            "gamma_a": 0.1,
            "lambda_s": 0.05,
            "m_0": 0.0,
            "theta_survival": 0.3,
            "theta_neutral": 0.7,
            "rho": 0.7,
        }

        config = APGIDynamicalParameters(**valid_config)
        assert (
            config.tau_s == 0.35
        )  # Note: field name is tau_s, not tau_S  # nosec: B101 - Test assertion
        # Pydantic validation happens automatically, no validate() method needed

        # Test consistency validation - should raise ValueError during creation
        with pytest.raises(
            ValueError, match="theta_survival must be less than theta_neutral"
        ):
            APGIDynamicalParameters(
                theta_survival=0.8,  # Should violate theta_survival < theta_neutral
                theta_neutral=0.7,
            )


class TestConfigurationUtilities:
    """Test configuration utility functions."""

    def test_config_checksum_computation(self):
        """Test configuration checksum computation."""
        config1 = {"key1": "value1", "key2": "value2"}
        config2 = {"key1": "value1", "key2": "value2"}

        checksum1 = compute_config_checksum(config1, "secret")
        checksum2 = compute_config_checksum(config2, "secret")

        assert (
            checksum1 == checksum2
        )  # Same config should have same checksum  # nosec: B101 - Test assertion
        assert len(checksum1) == 64  # SHA256 length  # nosec: B101 - Test assertion

    def test_config_sources_tracking(self):
        """Test configuration source tracking."""
        config = ConfigManager()

        # Load a config and check sources
        sources = config.get_all_sources()
        assert isinstance(sources, dict)  # nosec: B101 - Test assertion

        # Should have default source for unknown keys
        assert "unknown_key" in sources  # nosec: B101 - Test assertion
        assert sources["unknown_key"] == "default"  # nosec: B101 - Test assertion

    def test_config_reload(self):
        """Test configuration reload functionality."""
        config = ConfigManager()
        initial_sources = config.get_all_sources().copy()

        # Reload should clear cache and sources
        config.reload()

        # Sources should be reset to defaults
        new_sources = config.get_all_sources()
        assert (
            len(new_sources) < len(initial_sources) or len(new_sources) == 0
        )  # nosec: B101 - Test assertion


if __name__ == "__main__":
    pytest.main()
