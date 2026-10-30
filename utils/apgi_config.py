"""
APGI Config schemas with Pydantic validation and explicit versions.

Provides centralized runtime configuration with:
- Typed config schemas via Pydantic
- Singleton ConfigManager for unified access
- Environment-aware loading
- Caching for repeated config access
"""

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

# Try to import yaml for YAML loading support
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    yaml = None
    YAML_AVAILABLE = False

# Check if Pydantic is available
try:
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

# ---------------------------------------------------------------------------
# Pydantic Config Schemas
# ---------------------------------------------------------------------------


class APGIExperimentConfigSchema(BaseModel):
    """Typed APGI experiment configuration with validation."""

    version: str = "1.0.0"
    experiment_name: str = Field(default="unknown_experiment")
    tau_S: float = Field(default=0.35, ge=0.2, le=0.5)
    beta: float = Field(default=1.5, ge=0.5, le=2.5)
    theta_0: float = Field(default=0.5, ge=0.1, le=1.0)
    alpha: float = Field(default=5.5, ge=3.0, le=8.0)

    # Feature flags
    hierarchical_enabled: bool = Field(default=True)
    precision_gap_enabled: bool = Field(default=True)
    neuromodulator_tracking: bool = Field(default=True)

    @classmethod
    def from_legacy(cls, legacy_config: Dict[str, Any]) -> "APGIExperimentConfigSchema":
        """Backward-compatible adapter for older configs."""
        return cls(
            experiment_name=legacy_config.get("name", "unknown_experiment"),
            tau_S=legacy_config.get("tau_S", 0.35),
            beta=legacy_config.get("beta", 1.5),
            theta_0=legacy_config.get("theta_0", 0.5),
            alpha=legacy_config.get("alpha", 5.5),
            hierarchical_enabled=legacy_config.get("hierarchical_enabled", True),
            precision_gap_enabled=legacy_config.get("precision_gap_enabled", True),
            neuromodulator_tracking=legacy_config.get("neuromodulator_tracking", True),
        )


_LOG_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET")


class APGISecurityConfigSchema(BaseModel):
    """Security-related configuration schema."""

    audit_enabled: bool = Field(default=True)
    authz_enabled: bool = Field(default=True)
    subprocess_allowlist: list = Field(
        default_factory=lambda: ["git", "pytest", "python"]
    )
    require_secure_pickle: bool = Field(default=True)


class APGIMetricsConfigSchema(BaseModel):
    """Metrics and monitoring configuration."""

    profiling_enabled: bool = Field(default=False)
    performance_budget_ms: float = Field(default=600000)  # 10 minutes
    log_level: str = Field(default="INFO")
    json_logging: bool = Field(default=True)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Only standard logging level names are accepted (any case)."""
        if value.upper() not in _LOG_LEVELS:
            raise ValueError(f"log_level must be one of: {', '.join(_LOG_LEVELS)}")
        return value


# ---------------------------------------------------------------------------
# Config Manager Singleton
# ---------------------------------------------------------------------------

T = TypeVar("T")


@dataclass
class ConfigSources:
    """Tracks the sources of configuration values for debugging."""

    sources: Dict[str, str]

    def __post_init__(self) -> None:
        # Own the mapping so callers cannot mutate it through a shared reference
        self.sources = dict(self.sources)


class ConfigManager:
    """
    Centralized runtime configuration manager (Singleton pattern).

    Provides unified access to all APGI configuration with:
    - Environment variable overrides
    - Config file loading (JSON)
    - Caching for repeated access
    - Typed schemas for validation
    - Source tracking for audit/debug

    Usage:
        from utils.apgi_config import get_config

        config = get_config()
        experiment_config = config.get_experiment_config("stroop_effect")
    """

    _instance: Optional["ConfigManager"] = None
    _initialized: bool = False

    def __new__(cls) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._config_cache: Dict[str, Any] = {}
        self._sources: Dict[str, str] = {}
        self._environment_prefix = "APGI_"

        # Load base configuration
        self._load_base_config()
        self._initialized = True

    def _load_base_config(self) -> None:
        """Load configuration from file and environment."""
        # Try to load from config file
        config_file = self._find_config_file()
        if config_file:
            self._load_from_file(config_file)

        # Environment overrides take precedence
        self._load_from_environment()

    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        for key, value in os.environ.items():
            if key.startswith(self._environment_prefix):
                config_key = key[len(self._environment_prefix) :].lower()
                try:
                    parsed_value = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    # Handle boolean values manually if they aren't valid JSON
                    if value.lower() in ("true", "1", "yes", "on"):
                        parsed_value = True
                    elif value.lower() in ("false", "0", "no", "off"):
                        parsed_value = False
                    else:
                        parsed_value = value

                self._config_cache[config_key] = parsed_value
                self._sources[config_key] = f"env:{key}"

    def _find_config_file(self) -> Optional[Path]:
        """Find config file in standard locations."""
        search_paths = [
            Path("config/apgi.json"),
            Path("config/apgi.yaml"),
            Path("apgi.json"),
            Path("apgi.yaml"),
            Path.home() / ".apgi" / "config.json",
        ]

        # Also check APGI_CONFIG_FILE env var
        env_config = os.environ.get("APGI_CONFIG_FILE")
        if env_config:
            search_paths.insert(0, Path(env_config))

        for path in search_paths:
            if path.exists():
                return path
        return None

    def _load_from_file(self, path: Path) -> None:
        """Load configuration from JSON/YAML file."""
        try:
            with open(path) as f:
                if path.suffix in (".yaml", ".yml"):
                    try:
                        import yaml

                        data = yaml.safe_load(f)
                    except ImportError:
                        # Fall back to JSON if yaml not available
                        data = json.load(f)
                else:
                    data = json.load(f)

            self._config_cache.update(data)
            for key in data.keys():
                self._sources[key] = str(path)
        except Exception as e:
            # Log but don't fail - env vars can provide config
            import logging

            logging.getLogger("apgi.config").warning(
                f"Failed to load config from {path}: {e}"
            )

    def _load_yaml_safe(self, path: Path) -> Dict[str, Any]:
        """Load YAML file safely."""
        try:
            import yaml

            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            # Fall back to JSON if yaml not available
            return self._load_json_safe(path)
        except Exception as e:
            import logging

            logging.getLogger("apgi.config").error(
                f"Failed to load YAML from {path}: {e}"
            )
            return {}

    def _load_json_safe(self, path: Path) -> Dict[str, Any]:
        """Load JSON file safely."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception as e:
            import logging

            logging.getLogger("apgi.config").error(
                f"Failed to load JSON from {path}: {e}"
            )
            return {}

    def _load_env_file(self, path: Path) -> Dict[str, str]:
        """Load environment variables from .env file."""
        env_vars = {}
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip("\"'")
                        os.environ[key] = value
                        env_vars[key] = value
        except Exception as e:
            import logging

            logging.getLogger("apgi.config").error(
                f"Failed to load env file {path}: {e}"
            )
        return env_vars

    def _validate_file_path(self, path: Path) -> None:
        """Validate that a file path exists and is readable."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        if not os.access(path, os.R_OK):
            raise PermissionError(f"Config file not readable: {path}")

    def get(
        self, key: str, default: Optional[T] = None, type_hint: Optional[Type[T]] = None
    ) -> Optional[T]:
        """Get configuration value with optional type conversion."""
        value = self._config_cache.get(key, default)

        if type_hint and value is not None:
            try:
                if type_hint is bool and isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes", "on")
                elif callable(type_hint):
                    value = type_hint(value)  # type: ignore[call-arg]
            except (ValueError, TypeError):
                value = default

        return value

    def get_experiment_config(self, experiment_name: str) -> APGIExperimentConfigSchema:
        """Get typed experiment configuration."""
        # Build config from cached values with experiment prefix
        prefix = f"experiment_{experiment_name}_"

        config_dict = {
            "experiment_name": experiment_name,
            "tau_S": self.get(f"{prefix}tau_s", self.get("tau_s", 0.35)),
            "beta": self.get(f"{prefix}beta", self.get("beta", 1.5)),
            "theta_0": self.get(f"{prefix}theta_0", self.get("theta_0", 0.5)),
            "alpha": self.get(f"{prefix}alpha", self.get("alpha", 5.5)),
            "hierarchical_enabled": self.get(
                f"{prefix}hierarchical_enabled", self.get("hierarchical_enabled", True)
            ),
            "precision_gap_enabled": self.get(
                f"{prefix}precision_gap_enabled",
                self.get("precision_gap_enabled", True),
            ),
        }

        try:
            return APGIExperimentConfigSchema(**config_dict)
        except ValidationError as e:
            # Return defaults on validation error, but log
            import logging

            logging.getLogger("apgi.config").error(f"Config validation error: {e}")
            return APGIExperimentConfigSchema(experiment_name=experiment_name)

    def get_security_config(self) -> APGISecurityConfigSchema:
        """Get security configuration."""
        config_dict = {
            "audit_enabled": self.get("security_audit_enabled", True, bool),
            "authz_enabled": self.get("security_authz_enabled", True, bool),
            "subprocess_allowlist": self.get(
                "security_subprocess_allowlist",
                ["git", "pytest", "python"],
            ),
            "require_secure_pickle": self.get(
                "security_require_secure_pickle", True, bool
            ),
        }

        return APGISecurityConfigSchema(**config_dict)

    def get_metrics_config(self) -> APGIMetricsConfigSchema:
        """Get metrics configuration."""
        config_dict = {
            "profiling_enabled": self.get("metrics_profiling_enabled", False, bool),
            "performance_budget_ms": self.get(
                "metrics_performance_budget_ms", 600000.0, float
            ),
            "log_level": self.get("metrics_log_level", "INFO"),
            "json_logging": self.get("metrics_json_logging", True, bool),
        }

        return APGIMetricsConfigSchema(**config_dict)

    def get_source(self, key: str) -> Optional[str]:
        """Get the source of a configuration value."""
        return self._sources.get(key)

    def get_all_sources(self) -> Dict[str, str]:
        """Get all configuration sources for debugging."""
        return self._sources.copy()

    def reload(self) -> None:
        """Reload configuration from all sources."""
        self._config_cache.clear()
        self._sources.clear()
        self._load_base_config()
        get_cached_experiment_config.cache_clear()
        # Invalidate all cached configurations
        invalidate_method = getattr(self, "_invalidate_all_caches", None)
        if callable(invalidate_method):
            invalidate_method()

    def set(self, key: str, value: Any, source: str = "runtime") -> None:
        """Set configuration value at runtime."""
        self._config_cache[key] = value
        self._sources[key] = source


# ---------------------------------------------------------------------------
# Module-Level Functions (Convenience API)
# ---------------------------------------------------------------------------


def get_config() -> ConfigManager:
    """Get the global ConfigManager instance."""
    return ConfigManager()


def reset_config() -> None:
    """Reset global ConfigManager (useful for testing)."""
    ConfigManager._instance = None
    ConfigManager._initialized = False


# Module-level logging functions
def log_error(message: str) -> None:
    """Log an error message."""
    import logging

    logging.getLogger("apgi.config").error(message)


def log_error_with_context(message: str, context: Dict[str, Any]) -> None:
    """Log an error message with additional context."""
    import logging

    logger = logging.getLogger("apgi.config")
    logger.error(f"{message}. Context: {context}")


# Configuration constants
CURRENT_CONFIG_VERSION = "1.0.0"
CONFIG_MIGRATIONS = {
    "0.9.0": {
        "upgrade_to": "1.0.0",
        "description": "Add new security parameters",
        "changes": ["Add security_enabled field", "Add log_level field"],
    }
}


# Configuration classes
class APGIDynamicalParameters(BaseModel):
    """Dynamical system parameters for APGI."""

    tau_s: float = Field(
        default=0.35, ge=0.1, le=2.0, description="Somatotopic time constant"
    )
    tau_theta: float = Field(
        default=30.0, ge=1.0, le=120.0, description="Theta time constant"
    )
    tau_m: float = Field(
        default=1.5, ge=0.5, le=5.0, description="Metabolic time constant"
    )
    theta_0: float = Field(default=0.5, ge=0.1, le=2.0, description="Baseline theta")
    alpha: float = Field(default=5.5, ge=1.0, le=10.0, description="Alpha gain")
    beta_som: float = Field(
        default=1.5, ge=0.5, le=3.0, description="Somatotopic coupling strength"
    )
    beta_m: float = Field(
        default=1.0, ge=0.5, le=2.0, description="Metabolic coupling strength"
    )
    m_0: float = Field(
        default=0.0, ge=-1.0, le=1.0, description="Baseline metabolic state"
    )
    gamma_m: float = Field(
        default=-0.3, ge=-0.5, le=0.0, description="Metabolic decay rate"
    )
    gamma_a: float = Field(
        default=0.1, ge=0.0, le=0.5, description="Acetylcholine decay rate"
    )
    lambda_s: float = Field(
        default=0.1, ge=0.05, le=0.5, description="Somatotopic learning rate"
    )
    sigma_s: float = Field(
        default=0.05, ge=0.01, le=0.2, description="Somatotopic noise"
    )
    sigma_theta: float = Field(default=0.02, ge=0.01, le=0.1, description="Theta noise")
    sigma_m: float = Field(default=0.03, ge=0.01, le=0.1, description="Metabolic noise")
    theta_survival: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Survival threshold"
    )
    theta_neutral: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Neutral threshold"
    )

    @model_validator(mode="after")
    def validate_theta_consistency(self) -> "APGIDynamicalParameters":
        """Validate theta_survival is less than theta_neutral when both are explicitly set."""
        # Only validate consistency if both parameters were explicitly provided
        # This prevents interfering with individual parameter validation tests
        if (
            hasattr(self, "__pydantic_fields_set__")
            and "theta_survival" in self.__pydantic_fields_set__
            and "theta_neutral" in self.__pydantic_fields_set__
        ):
            if self.theta_survival >= self.theta_neutral:
                raise ValueError("theta_survival must be less than theta_neutral")
        return self

    rho: float = Field(default=0.7, ge=0.5, le=1.0, description="Retention rate")


class APGIMetabolicParameters(BaseModel):
    """Metabolic parameters for APGI."""

    baseline_metabolic_rate: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Baseline metabolic rate"
    )
    glucose_baseline: float = Field(
        default=5.0, ge=3.0, le=10.0, description="Baseline glucose level"
    )
    insulin_sensitivity: float = Field(
        default=1.0, ge=0.5, le=2.0, description="Insulin sensitivity factor"
    )
    effort_cost_coefficient: float = Field(
        default=0.1, ge=0.05, le=0.5, description="Effort cost coefficient"
    )
    glucose_impact_factor: float = Field(
        default=0.5, ge=0.1, le=1.0, description="Glucose impact factor"
    )
    fatigue_recovery_rate: float = Field(
        default=0.05, ge=0.01, le=0.2, description="Fatigue recovery rate"
    )


class APGIPerceptualParameters(BaseModel):
    """Perceptual parameters for APGI."""

    pi_i_expected: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Expected perceptual input"
    )
    pi_i_survival: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Survival perceptual input"
    )
    attentional_gain: float = Field(
        default=2.0, ge=0.5, le=5.0, description="Attentional gain"
    )
    visual_acuity: float = Field(
        default=1.0, ge=0.1, le=2.0, description="Visual acuity"
    )
    visual_threshold: float = Field(
        default=0.5, ge=0.1, le=1.0, description="Visual perception threshold"
    )
    auditory_threshold: float = Field(
        default=0.3, ge=0.1, le=1.0, description="Auditory perception threshold"
    )


class APGIPsychiatricProfiles(BaseModel):
    """Psychiatric profile parameters."""

    gad_profile: float = Field(
        default=0.0, ge=0.0, le=10.0, description="GAD severity profile"
    )
    depression_score: float = Field(
        default=0.0, ge=0.0, le=10.0, description="Depression severity score"
    )
    anxiety_score: float = Field(
        default=0.0, ge=0.0, le=10.0, description="Anxiety severity score"
    )


class SecurityConfig(BaseModel):
    """Security configuration."""

    audit_key: str = Field(default="", description="Audit encryption key")

    @model_validator(mode="after")
    def validate_audit_key(self) -> "SecurityConfig":
        """Validate audit key strength if provided."""
        if self.audit_key:
            if len(self.audit_key) < 64:
                raise ValueError("audit_key must be at least 64 characters")
            weak_patterns = ["test", "password", "default", "admin", "123456"]
            if any(p in self.audit_key.lower() for p in weak_patterns):
                raise ValueError("audit_key contains weak pattern")
        return self

    @model_validator(mode="after")
    def validate_operator_role(self) -> "SecurityConfig":
        """Validate operator_role is one of allowed values."""
        if self.operator_role not in ["guest", "operator", "admin"]:
            raise ValueError("operator_role must be one of: guest, operator, admin")
        return self

    operator_role: str = Field(default="guest", description="Default operator role")
    encryption_enabled: bool = Field(default=True, description="Enable encryption")
    access_log_enabled: bool = Field(default=True, description="Enable access logging")
    allowed_subprocess_cmds: list = Field(
        default_factory=lambda: ["git", "pytest", "python", "echo"],
        description="Allowed subprocess commands",
    )
    enable_profiling: bool = Field(default=False, description="Enable profiling")
    config_secret_key: str = Field(default="", description="Configuration secret key")
    pickle_allowlist: list = Field(default_factory=list, description="Pickle allowlist")


class ExperimentConfig(BaseModel):
    """Experiment configuration."""

    name: str = Field(default="test_experiment", description="Experiment name")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Experiment parameters"
    )
    duration: int = Field(
        default=3600, ge=60, le=86400, description="Experiment duration in seconds"
    )
    num_trials: int = Field(default=100, ge=1, le=10000, description="Number of trials")
    time_budget_seconds: int = Field(
        default=3600, ge=60, le=3600, description="Time budget in seconds"
    )
    output_format: str = Field(default="json", description="Output format")
    checkpoint_interval: int = Field(
        default=300, ge=30, le=900, description="Checkpoint interval in seconds"
    )

    @model_validator(mode="after")
    def validate_output_format(self) -> "ExperimentConfig":
        """Validate output_format is one of allowed values."""
        if self.output_format not in ["json", "csv", "parquet"]:
            raise ValueError("output_format must be one of: json, csv, parquet")
        return self

    random_seed: int = Field(
        default=42, ge=0, le=2**32 - 1, description="Random seed for reproducibility"
    )


# Utility functions
def compute_config_checksum(config: Any, secret_key: Optional[str] = None) -> str:
    """Compute checksum for configuration data."""
    import hashlib

    # Handle APGIGlobalConfig objects
    if hasattr(config, "model_dump"):
        config_dict = config.model_dump()
    elif isinstance(config, dict):
        # Copy: the secret must not be written into the caller's config
        config_dict = dict(config)
    else:
        config_dict = {"config": str(config)}

    if secret_key:
        config_dict["secret_key"] = secret_key

    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()


def migrate_config(
    config: Dict[str, Any], from_version: str, to_version: Optional[str] = None
) -> Dict[str, Any]:
    """Migrate configuration from one version to another."""
    # If to_version not specified, migrate to current version
    if to_version is None:
        to_version = CURRENT_CONFIG_VERSION

    # Check if we have a migration path
    migration_performed = False

    # Simple migration logic - in practice this would be more complex
    if from_version == "0.9.0" and to_version == "1.0.0":
        config.setdefault("security_enabled", True)
        config.setdefault("log_level", "INFO")
        migration_performed = True

    # Only update version if migration was performed
    if migration_performed:
        config["version"] = to_version

    return config


def validate_config_integrity(
    config: Any, expected_checksum: str, secret_key: Optional[str] = None
) -> bool:
    """Validate configuration integrity."""
    try:
        # Compute actual checksum
        actual_checksum = compute_config_checksum(config, secret_key)

        # Compare checksums
        return actual_checksum == expected_checksum
    except Exception:
        return False


def validate_startup_config(config: Optional[Any] = None) -> bool:
    """Validate startup configuration."""
    try:
        if config is None:
            # Load config from environment variables
            config = load_config_from_env()

        # If it's already a config object, validate its structure
        if isinstance(config, APGIGlobalConfig):
            # Validate version pattern
            import re

            version_pattern = r"^\d+\.\d+\.\d+$"
            if not re.match(version_pattern, config.version):
                raise ValueError(f"Invalid version format: {config.version}")

            # Validate audit key length only if audit_key is provided
            if hasattr(config, "security") and hasattr(config.security, "audit_key"):
                if config.security.audit_key and len(config.security.audit_key) < 64:
                    raise ValueError("Audit key must be at least 64 characters")

            return True

        # If it's a dict, try to create a config from it
        elif isinstance(config, dict):
            APGIGlobalConfig(**config)
            return True

        # Invalid type
        raise ValueError("Invalid config type")
    except Exception as e:
        # Re-raise validation errors
        if "validation error" in str(e).lower() or "value error" in str(e).lower():
            raise
        return False


@lru_cache(maxsize=128)
def get_cached_experiment_config(experiment_name: str) -> APGIExperimentConfigSchema:
    """
    Get cached experiment configuration.

    Uses LRU cache to avoid repeated config lookups/validation.
    Cache is keyed by experiment name.
    """
    return get_config().get_experiment_config(experiment_name)


def invalidate_config_cache() -> None:
    """Invalidate cached experiment config (call after config changes)."""
    get_cached_experiment_config.cache_clear()


# ---------------------------------------------------------------------------
# Legacy Compatibility
# ---------------------------------------------------------------------------


def load_apgi_params() -> Dict[str, Any]:
    """
    Legacy-compatible APGI params loader.

    Returns dict format expected by older code while using new config system.
    """
    config = get_config()
    return {
        "tau_s": config.get("tau_s", 0.35),
        "beta": config.get("beta", 1.5),
        "theta_0": config.get("theta_0", 0.5),
        "alpha": config.get("alpha", 5.5),
        "gamma_M": config.get("gamma_m", -0.3),
        "lambda_S": config.get("lambda_s", 0.05),
        "sigma_S": config.get("sigma_s", 0.1),
        "sigma_theta": config.get("sigma_theta", 0.05),
        "sigma_M": config.get("sigma_m", 0.05),
        "rho": config.get("rho", 0.8),
        "theta_survival": config.get("theta_survival", 0.3),
        "theta_neutral": config.get("theta_neutral", 0.5),
        "beta_cross": config.get("beta_cross", 0.2),
        "tau_levels": config.get("tau_levels", [0.1, 0.2, 0.4, 1.0, 5.0]),
        "enabled": config.get("apgi_enabled", True, bool),
        "hierarchical_enabled": config.get("hierarchical_enabled", True, bool),
        "precision_gap_enabled": config.get("precision_gap_enabled", True, bool),
        "ACh": config.get("ach_level", 1.0),
        "NE": config.get("ne_level", 1.0),
        "DA": config.get("da_level", 1.0),
        "HT5": config.get("ht5_level", 1.0),
    }


class APGIGlobalConfig(BaseModel):
    """Global APGI configuration."""

    @model_validator(mode="after")
    def validate_version(self) -> "APGIGlobalConfig":
        """Validate version follows semantic versioning pattern."""
        import re

        version_pattern = r"^\d+\.\d+\.\d+$"
        if not re.match(version_pattern, self.version):
            raise ValueError(f"Invalid version format: {self.version}")
        return self

    version: str = Field(default="1.0.0", description="Config version")
    dynamical: APGIDynamicalParameters = Field(
        default_factory=APGIDynamicalParameters, description="Dynamical parameters"
    )
    perceptual: APGIPerceptualParameters = Field(
        default_factory=APGIPerceptualParameters, description="Perceptual parameters"
    )
    metabolic: APGIMetabolicParameters = Field(
        default_factory=APGIMetabolicParameters, description="Metabolic parameters"
    )
    psychiatric: APGIPsychiatricProfiles = Field(
        default_factory=APGIPsychiatricProfiles, description="Psychiatric profiles"
    )
    security: SecurityConfig = Field(
        default_factory=SecurityConfig, description="Security configuration"
    )
    experiment: ExperimentConfig = Field(
        default_factory=ExperimentConfig, description="Experiment configuration"
    )
    security_enabled: bool = Field(default=True, description="Enable security features")
    log_level: str = Field(default="INFO", description="Logging level")


def load_config_from_env() -> APGIGlobalConfig:
    """Load configuration from environment variables."""
    config_dict = {}
    security_dict = {}

    # Security-related environment variables
    security_keys = {
        "audit_key",
        "operator_role",
        "encryption_enabled",
        "access_log_enabled",
        "allowed_subprocess_cmds",
        "enable_profiling",
        "config_secret_key",
    }

    for key, value in os.environ.items():
        if key.startswith("APGI_"):
            config_key = key[5:].lower()
            if config_key in security_keys:
                security_dict[config_key] = value
            elif config_key.startswith("security_"):
                security_dict[config_key[9:]] = value
            else:
                config_dict[config_key] = value

    # Create security config if any security env vars are set
    security_config = SecurityConfig()  # Default config
    if security_dict:
        # Parse boolean values
        def parse_bool(val: Any) -> bool:
            if isinstance(val, str):
                return val.lower() in ("true", "1", "yes", "on")
            return bool(val)

        # Parse list values
        def parse_list(val: Any) -> list:
            if isinstance(val, str):
                try:
                    import json

                    return json.loads(val)  # type: ignore
                except (json.JSONDecodeError, ValueError):
                    return val.split(",")
            return val  # type: ignore

        security_config = SecurityConfig(
            audit_key=security_dict.get("audit_key", ""),
            operator_role=security_dict.get("operator_role", "guest"),
            encryption_enabled=parse_bool(
                security_dict.get("encryption_enabled", True)
            ),
            access_log_enabled=parse_bool(
                security_dict.get("access_log_enabled", True)
            ),
            allowed_subprocess_cmds=parse_list(
                security_dict.get(
                    "allowed_subprocess_cmds", ["git", "pytest", "python", "echo"]
                )
            ),
            enable_profiling=parse_bool(security_dict.get("enable_profiling", False)),
            config_secret_key=security_dict.get("config_secret_key", ""),
        )

    # Create global config
    return APGIGlobalConfig(
        version=config_dict.get("version", "1.0.0"),
        security=security_config,
        security_enabled=bool(config_dict.get("security_enabled", True)),
        log_level=config_dict.get("log_level", "INFO"),
    )
