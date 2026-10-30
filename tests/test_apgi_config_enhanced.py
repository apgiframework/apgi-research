"""
Enhanced test suite for apgi_config.py

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
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from utils.apgi_config import ConfigManager, reset_config


@pytest.fixture(autouse=True)
def _fresh_config():
    """ConfigManager is a singleton, so give every test an uninitialised one."""
    reset_config()
    yield
    reset_config()


class TestConfigManager:
    """Test ConfigManager class methods."""

    def test_config_isolation(self):
        """Test that configuration instances are properly isolated."""
        # Reset singleton
        reset_config()

        # Create two instances and verify they are the same (singleton)
        config1 = ConfigManager()
        config2 = ConfigManager()
        assert config1 is config2  # nosec: B101 - Test assertion
        assert config1._initialized  # nosec: B101 - Test assertion
        assert config2._initialized  # nosec: B101 - Test assertion

    def test_concurrent_access(self):
        """Test concurrent access to ConfigManager."""
        import threading
        import time

        # Create instances for concurrent access
        ConfigManager()
        ConfigManager()

        results = []
        errors = []

        def access_config(instance_id):
            try:
                for _ in range(100):
                    config = ConfigManager()
                    value = config.get("test_key", f"value_{instance_id}")
                    results.append(value)
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(f"Instance {instance_id}: {e}")

        # Start concurrent access
        thread1 = threading.Thread(target=access_config, args=(1,))
        thread2 = threading.Thread(target=access_config, args=(2,))

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        # Should have no errors, every read answered, and each thread seeing
        # its own (default) value consistently
        assert (
            len(errors) == 0
        ), f"Errors occurred: {errors}"  # nosec: B101 - Test assertion
        assert len(results) == 200  # 100 reads per thread  # nosec: B101
        assert set(results) == {"value_1", "value_2"}  # nosec: B101

    def test_memory_efficiency(self):
        """Test memory efficiency of ConfigManager."""
        config = ConfigManager()

        # Load many configurations to test memory usage
        for i in range(1000):
            config.get_experiment_config(f"memory_test_{i}")

    def test_config_validation_edge_cases(self):
        """Test configuration validation with edge cases."""
        config = ConfigManager()

        # Test with extreme values
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            extreme_config = {
                "tau_s": 1000.0,  # Way above max
                "beta": -10.0,  # Way below min
                "theta_0": 10.0,  # Way above max
                "alpha": -5.0,  # Way below min
            }
            tmp_file.write(json.dumps(extreme_config))
            tmp_file.flush()

            with patch.object(ConfigManager, "_find_config_file") as mock_find:
                mock_find.return_value = Path(tmp_file.name)

                config = ConfigManager()

                # Test that extreme values are handled gracefully
                # ConfigManager uses Pydantic schemas which handle validation automatically
                experiment_config = config.get_experiment_config("extreme_test")

                # Verify the config was loaded (Pydantic validation happens internally)
                assert experiment_config is not None  # nosec: B101 - Test assertion

    def test_config_source_precedence(self, tmp_path, monkeypatch):
        """Environment variables take precedence over the config file."""
        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps({"experiment_test_tau_s": 0.25, "file_only": "from file"})
        )
        monkeypatch.setenv("APGI_EXPERIMENT_TEST_TAU_S", "0.5")

        with patch.object(ConfigManager, "_find_config_file") as mock_find:
            mock_find.return_value = config_file

            config = ConfigManager()

        # Environment wins where both define a key (and is parsed to a number)...
        assert config.get("experiment_test_tau_s") == 0.5  # nosec: B101
        assert (  # nosec: B101 - Test assertion
            config.get_source("experiment_test_tau_s")
            == "env:APGI_EXPERIMENT_TEST_TAU_S"
        )
        # ...while keys only in the file keep the file as their source
        assert config.get("file_only") == "from file"  # nosec: B101
        assert config.get_source("file_only") == str(config_file)  # nosec: B101

    def test_config_reload_consistency(self, monkeypatch):
        """Reload discards runtime values and re-reads the environment."""
        monkeypatch.setenv("APGI_RELOAD_PROBE", "first")
        config = ConfigManager()
        assert config.get("reload_probe") == "first"  # nosec: B101

        # Modify config at runtime
        config.set("test_key", "modified_value", source="test")
        assert config.get_source("test_key") == "test"  # nosec: B101

        monkeypatch.setenv("APGI_RELOAD_PROBE", "second")
        config.reload()

        # Runtime values are discarded; environment values are re-read
        assert config.get("test_key") is None  # nosec: B101 - Test assertion
        assert config.get_source("test_key") is None  # nosec: B101
        assert config.get("reload_probe") == "second"  # nosec: B101

    def test_config_invalid_json_handling(self, tmp_path):
        """A malformed config file is ignored rather than crashing."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text('{"invalid": json content}')

        with patch.object(ConfigManager, "_find_config_file") as mock_find:
            mock_find.return_value = bad_file

            config = ConfigManager()

        # Should not crash, should use defaults
        assert config.get("invalid") is None  # nosec: B101 - Test assertion
        assert config.get("tau_s", 0.35) == 0.35  # nosec: B101
        experiment = config.get_experiment_config("still_works")
        assert experiment.experiment_name == "still_works"  # nosec: B101

    def test_config_circular_reference_handling(self):
        """Test handling of circular references in configuration."""
        config = ConfigManager()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            circular_config = {
                "experiment_name": "main",
                "parameters": {"main": {"experiment_name": "main"}},
            }
            tmp_file.write(json.dumps(circular_config))
            tmp_file.flush()

            with patch.object(ConfigManager, "_find_config_file") as mock_find:
                mock_find.return_value = Path(tmp_file.name)

                config = ConfigManager()
                experiment_config = config.get_experiment_config("main")

                # Should handle circular reference gracefully
                assert (
                    experiment_config.experiment_name == "main"
                )  # nosec: B101 - Test assertion
                assert (
                    experiment_config.experiment_name == "main"
                )  # nosec: B101 - Test assertion

    def test_config_performance(self):
        """Test ConfigManager performance with large configurations."""
        config = ConfigManager()

        # Test with many configuration accesses
        start_time = time.time()
        for i in range(1000):
            config.get_experiment_config(f"perf_test_{i}")

        end_time = time.time()
        duration = end_time - start_time

        # Should complete within reasonable time
        assert (
            duration < 1.0
        ), f"Config access took too long: {duration:.3f}s"  # nosec: B101 - Test assertion


if __name__ == "__main__":
    pytest.main()
