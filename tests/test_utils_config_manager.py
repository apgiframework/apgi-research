"""
Comprehensive tests for apgi_config.py functions.
This file aims to achieve 100% test coverage for all available functions.
"""

import json

# Import the actual functions that exist in apgi_config.py
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import mock_open, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.apgi_config import ConfigManager, ConfigSources, get_config, reset_config


class TestConfigSources:
    """Tests for ConfigSources dataclass."""

    def test_init_empty(self):
        """Test ConfigSources initialization with empty sources."""
        sources = ConfigSources(sources={})
        assert sources.sources == {}  # nosec: B101 - Test assertion

    def test_init_with_sources(self):
        """Test ConfigSources initialization with sources."""
        test_sources = {"key1": "file1", "key2": "env"}
        sources = ConfigSources(sources=test_sources)
        assert sources.sources == test_sources  # nosec: B101 - Test assertion

    def test_sources_copy(self):
        """Test that sources dict is independent."""
        sources1 = ConfigSources({"key": "value"})
        sources2 = ConfigSources(sources1.sources)
        sources2.sources["new_key"] = "new_value"
        assert "new_key" not in sources1.sources  # nosec: B101 - Test assertion


class TestConfigManagerUtilityFunctions:
    """Tests for standalone utility functions in ConfigManager."""

    def setup_method(self):
        """Reset config manager before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config manager after each test."""
        reset_config()

    def test_load_yaml_safe_valid(self):
        """Test _load_yaml_safe with valid YAML."""
        config = ConfigManager()

        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("test_key: test_value\nnumber_key: 42")
            temp_path = Path(f.name)

        try:
            result = config._load_yaml_safe(temp_path)
            assert result == {
                "test_key": "test_value",
                "number_key": 42,
            }  # nosec: B101 - Test assertion
        finally:
            temp_path.unlink(missing_ok=True)

    def test_load_yaml_safe_invalid(self):
        """Test _load_yaml_safe with invalid YAML."""
        config = ConfigManager()

        # Create temporary invalid YAML file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = Path(f.name)

        try:
            result = config._load_yaml_safe(temp_path)
            # "Safe" loaders report a failure as an empty config, never raise
            assert result == {}  # nosec: B101 - Test assertion
        finally:
            temp_path.unlink(missing_ok=True)

    def test_load_yaml_safe_no_yaml_module(self):
        """Test _load_yaml_safe when yaml module is not available."""
        config = ConfigManager()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("test_key: test_value")
            temp_path = Path(f.name)

        try:
            with patch.dict("sys.modules", {"yaml": None}):
                with patch(
                    "builtins.open", mock_open(read_data="test_key: test_value")
                ):
                    result = config._load_yaml_safe(temp_path)
                    # Should fall back to JSON parsing which fails for YAML content
                    assert result == {}  # nosec: B101 - Test assertion
        finally:
            temp_path.unlink(missing_ok=True)

    def test_load_json_safe_valid(self):
        """Test _load_json_safe with valid JSON."""
        config = ConfigManager()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json_data = {"test_key": "test_value", "number_key": 42}
            f.write(json.dumps(json_data))
            temp_path = Path(f.name)

        try:
            result = config._load_json_safe(temp_path)
            assert result == json_data  # nosec: B101 - Test assertion
        finally:
            temp_path.unlink(missing_ok=True)

    def test_load_json_safe_invalid(self):
        """Test _load_json_safe with invalid JSON."""
        config = ConfigManager()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"invalid": json content}')
            temp_path = Path(f.name)

        try:
            result = config._load_json_safe(temp_path)
            assert result == {}  # nosec: B101 - Test assertion
        finally:
            temp_path.unlink(missing_ok=True)

    def test_load_json_safe_file_not_found(self):
        """Test _load_json_safe when file doesn't exist."""
        config = ConfigManager()
        result = config._load_json_safe(Path("/nonexistent/file.json"))
        assert result == {}  # nosec: B101 - Test assertion

    def test_load_env_file_valid(self):
        """Test _load_env_file with valid file."""
        config = ConfigManager()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("TEST_KEY=test_value\nANOTHER_KEY=42")
            temp_path = Path(f.name)

        try:
            result = config._load_env_file(temp_path)
            assert result == {
                "TEST_KEY": "test_value",
                "ANOTHER_KEY": "42",
            }  # nosec: B101 - Test assertion
        finally:
            temp_path.unlink(missing_ok=True)

    def test_load_env_file_invalid(self):
        """Test _load_env_file with invalid format."""
        config = ConfigManager()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("INVALID_LINE_WITHOUT_EQUALS")
            temp_path = Path(f.name)

        try:
            result = config._load_env_file(temp_path)
            assert result == {}  # nosec: B101 - Test assertion
        finally:
            temp_path.unlink(missing_ok=True)

    def test_validate_file_path_exists(self):
        """Test _validate_file_path with existing file."""
        config = ConfigManager()

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Should not raise an exception
            config._validate_file_path(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def test_validate_file_path_not_exists(self):
        """Test _validate_file_path with non-existing file."""
        config = ConfigManager()
        with pytest.raises(FileNotFoundError):
            config._validate_file_path(Path("/nonexistent/file"))

    def test_validate_file_path_empty(self):
        """An empty path means the current directory, which is not a file."""
        config = ConfigManager()
        with pytest.raises(ValueError, match="not a file"):
            config._validate_file_path(Path(""))


class TestConfigManagerMissingCoverage:
    """Tests for missing coverage areas in ConfigManager."""

    def setup_method(self):
        """Reset config manager before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config manager after each test."""
        reset_config()

    def test_fallback_yaml_load_safe(self):
        """Test fallback behavior when YAML module is unavailable."""
        config = ConfigManager()

        # Create a temporary YAML-like file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("test_key: test_value\nnumber_key: 42")
            temp_path = Path(f.name)

        try:
            with patch.dict("sys.modules", {"yaml": None}):
                # Should fall back to JSON parsing or handle gracefully
                config._load_from_file(temp_path)
                # Should not crash and should have some fallback behavior
                assert len(config._config_cache) > 0  # nosec: B101 - Test assertion
        finally:
            temp_path.unlink(missing_ok=True)


class TestUtilityFunctions:
    """Tests for standalone utility functions."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_log_error_function(self):
        """Test standalone log_error function."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger_instance = mock_get_logger.return_value
            from utils.apgi_config import log_error

            log_error("test error")
            mock_logger_instance.error.assert_called_once_with("test error")

    def test_log_error_with_context_function(self):
        """Test standalone log_error_with_context function."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger_instance = mock_get_logger.return_value
            from utils.apgi_config import log_error_with_context

            log_error_with_context("Error message", {"key": "value"})
            mock_logger_instance.error.assert_called_once()


# Utility functions that should be available but aren't in apgi_config module
def create_config_profile(name: str, config: dict, directory: Path) -> bool:
    """Create a configuration profile."""
    try:
        profile_path = directory / f"{name}.json"
        with open(profile_path, "w") as f:
            json.dump({"name": name, "config": config}, f)
        return True
    except Exception:
        return False


def switch_config_profile(name: str, directory: Path) -> bool:
    """Switch to a configuration profile."""
    try:
        profile_path = directory / f"{name}.json"
        if profile_path.exists():
            with open(profile_path, "r") as f:
                profile_data = json.load(f)
                config = get_config()
                for key, value in profile_data.get("config", {}).items():
                    config.set(key, value, "profile")
            return True
        return False
    except Exception:
        return False


def rollback_config_profile() -> bool:
    """Rollback to previous configuration profile."""
    # This is a simplified implementation
    # In a real scenario, this would restore from backup
    return True


def list_config_profiles(directory: Path) -> list:
    """List configuration profiles in directory."""
    profiles = []
    try:
        for file_path in directory.glob("*.json"):
            with open(file_path, "r") as f:
                profile_data = json.load(f)
                profiles.append(profile_data)
        return profiles
    except Exception:
        return []


def compare_config_profiles(name1: str, name2: str, directory: Path) -> dict:
    """Compare two configuration profiles."""
    try:
        profile1_path = directory / f"{name1}.json"
        profile2_path = directory / f"{name2}.json"

        with open(profile1_path, "r") as f1:
            profile1_data = json.load(f1)

        with open(profile2_path, "r") as f2:
            profile2_data = json.load(f2)

        # Simple comparison logic
        differences = {}
        all_keys = set(profile1_data.get("config", {}).keys()) | set(
            profile2_data.get("config", {}).keys()
        )

        for key in all_keys:
            val1 = profile1_data.get("config", {}).get(key)
            val2 = profile2_data.get("config", {}).get(key)
            if val1 != val2:
                if key not in differences:
                    differences[key] = {"old": val1, "new": val2}
                else:
                    differences[key] = {"old": val1, "new": val2}

        return differences
    except Exception:
        return {}


def set_parameter(key: str, value: Any) -> None:
    """Set a configuration parameter."""
    config = get_config()
    config.set(key, value)


def get_batch_config(keys: list) -> dict:
    """Get multiple configuration parameters."""
    config = get_config()
    result: dict = {}
    for key in keys:
        result[key] = config.get(key)
    return result


def set_batch_parameter(params: dict) -> None:
    """Set multiple configuration parameters."""
    config = get_config()
    for key, value in params.items():
        config.set(key, value)


def get_max_workers() -> int:
    """Get maximum number of workers."""
    config = get_config()
    result = config.get("max_workers", default=4, type_hint=int)
    return result if result is not None else 4


def validate_config_file(file_path: Path) -> tuple[bool, list[str]]:
    """Validate a configuration file."""
    try:
        with open(file_path, "r") as f:
            config_data = json.load(f)
            # Basic validation - check if it's a valid JSON with required structure
            if isinstance(config_data, dict) and "tau_s" in config_data:
                return True, []
            else:
                return False, ["Missing required tau_s parameter"]
    except Exception as e:
        return False, [f"File validation error: {str(e)}"]
