import io
import json as json_module
import warnings

import pytest

import utils.apgi_security as apgi_security_module
from utils.apgi_security import (
    DEFAULT_ALLOWED_SUBPROCESS_CMDS,
    PickleSecurityError,
    SecurePickleWrapper,
    SecureSubprocessError,
    SecureSubprocessWrapper,
    SubprocessSecurityPolicy,
    get_env_var,
    get_secure_pickle_wrapper,
    get_secure_subprocess_wrapper,
    get_sensitive_env_var,
    require_allowed_subprocess,
    secure_dump,
    secure_dumps,
    secure_load,
    secure_loads,
    secure_loads_json,
    secure_popen,
    secure_run,
    validate_config_checksum,
)


def test_secure_popen_blocks_disallowed_commands():
    """Test that secure_popen blocks commands not in allowlist."""
    with pytest.raises(SecureSubprocessError, match="not in the allowlist"):
        secure_popen(["rm", "-rf", "/"])

    with pytest.raises(SecureSubprocessError, match="not in the allowlist"):
        secure_popen("ls -la", shell=True)


def test_secure_loads_blocks_pickle_by_default():
    """Test that secure_loads blocks pickle by default."""
    payload = b"cos\nsystem\n(S'echo hacked'\ntR."
    with pytest.raises(PickleSecurityError, match="Pickle is disabled"):
        secure_loads(payload)


def test_config_signing_validation():
    config = {"alpha": 5.0, "beta": 1.0}
    secret = "test_key"

    import hashlib
    import hmac
    import json

    config_str = json.dumps(config, sort_keys=True)
    valid_hash = hmac.new(
        secret.encode("utf-8"), config_str.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    # Valid
    assert (
        validate_config_checksum(config, valid_hash, secret) is True
    )  # nosec: B101 - Test assertion

    # Bypass attempt (tampered value)
    tampered_config = {"alpha": 5.0, "beta": 10.0}
    assert (
        validate_config_checksum(tampered_config, valid_hash, secret) is False
    )  # nosec: B101 - Test assertion


# ---------------------------------------------------------------------------
# SubprocessSecurityPolicy
# ---------------------------------------------------------------------------


class TestSubprocessSecurityPolicy:
    def test_is_allowed_true_for_allowlisted_command(self):
        policy = SubprocessSecurityPolicy(allowed_commands=["git", "ls"])
        assert policy.is_allowed("git") is True

    def test_is_allowed_false_for_non_allowlisted_command(self):
        policy = SubprocessSecurityPolicy(allowed_commands=["git"])
        assert policy.is_allowed("rm") is False

    def test_is_allowed_true_when_allowlist_not_required(self):
        policy = SubprocessSecurityPolicy(
            allowed_commands=[], require_explicit_allowlist=False
        )
        assert policy.is_allowed("anything") is True


# ---------------------------------------------------------------------------
# SecureSubprocessWrapper
# ---------------------------------------------------------------------------


class TestSecureSubprocessWrapper:
    def test_default_policy_uses_env_var(self, monkeypatch):
        monkeypatch.setenv("APGI_ALLOWED_SUBPROCESS_CMDS", "foo, bar ,baz")
        policy = SecureSubprocessWrapper._default_policy()
        assert policy.allowed_commands == ["foo", "bar", "baz"]
        assert policy.require_explicit_allowlist is True

    def test_default_policy_uses_default_list_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("APGI_ALLOWED_SUBPROCESS_CMDS", raising=False)
        policy = SecureSubprocessWrapper._default_policy()
        assert policy.allowed_commands == DEFAULT_ALLOWED_SUBPROCESS_CMDS

    def test_extract_command_from_list_arg(self):
        wrapper = SecureSubprocessWrapper()
        cmd = wrapper._extract_command((["git", "status"],))
        assert cmd == "git"

    def test_extract_command_from_tuple_arg(self):
        wrapper = SecureSubprocessWrapper()
        cmd = wrapper._extract_command((("git", "status"),))
        assert cmd == "git"

    def test_extract_command_from_string_arg(self):
        wrapper = SecureSubprocessWrapper()
        cmd = wrapper._extract_command(("git status",))
        assert cmd == "git"

    def test_extract_command_empty_args_returns_empty_string(self):
        wrapper = SecureSubprocessWrapper()
        cmd = wrapper._extract_command(())
        assert cmd == ""

    def test_extract_command_empty_list_returns_empty_string(self):
        wrapper = SecureSubprocessWrapper()
        cmd = wrapper._extract_command(([],))
        assert cmd == ""

    def test_call_with_kwargs_args_disallowed(self):
        policy = SubprocessSecurityPolicy(allowed_commands=["git"])
        wrapper = SecureSubprocessWrapper(policy=policy)
        with pytest.raises(SecureSubprocessError, match="not in the allowlist"):
            wrapper(args=["rm", "-rf", "/"])

    def test_call_with_kwargs_args_empty_list(self):
        policy = SubprocessSecurityPolicy(allowed_commands=["git"])
        wrapper = SecureSubprocessWrapper(policy=policy)
        with pytest.raises(SecureSubprocessError):
            wrapper(args=[])

    def test_call_allows_whitelisted_command(self):
        policy = SubprocessSecurityPolicy(allowed_commands=["echo"])
        wrapper = SecureSubprocessWrapper(policy=policy)
        proc = wrapper(["echo", "hi"], stdout=-1)
        out, _ = proc.communicate()
        proc.wait()
        assert b"hi" in out

    def test_get_secure_subprocess_wrapper_returns_singleton(self, monkeypatch):
        monkeypatch.setattr(apgi_security_module, "_default_subprocess_wrapper", None)
        w1 = get_secure_subprocess_wrapper()
        w2 = get_secure_subprocess_wrapper()
        assert w1 is w2


# ---------------------------------------------------------------------------
# secure_run
# ---------------------------------------------------------------------------


class TestSecureRun:
    def test_secure_run_allows_whitelisted_list_command(self):
        result = secure_run(
            ["python3", "-c", "print('hello')"], capture_output=True
        )
        assert result.returncode == 0
        assert b"hello" in result.stdout

    def test_secure_run_allows_whitelisted_string_command(self):
        result = secure_run("python3 -c \"print('hello')\"", capture_output=True)
        assert result.returncode == 0

    def test_secure_run_blocks_disallowed_list_command(self):
        with pytest.raises(SecureSubprocessError, match="not in the allowlist"):
            secure_run(["rm", "-rf", "/"])

    def test_secure_run_blocks_disallowed_string_command(self):
        with pytest.raises(SecureSubprocessError, match="not in the allowlist"):
            secure_run("rm -rf /")

    def test_secure_run_blocks_empty_args(self):
        with pytest.raises(SecureSubprocessError):
            secure_run([])

    def test_secure_run_with_custom_policy(self, monkeypatch):
        monkeypatch.setattr(apgi_security_module, "_default_subprocess_wrapper", None)
        policy = SubprocessSecurityPolicy(allowed_commands=["echo"])
        result = secure_run(["echo", "custom"], policy=policy, capture_output=True)
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# SecurePickleWrapper
# ---------------------------------------------------------------------------


class TestSecurePickleWrapper:
    def test_loads_json_object(self):
        wrapper = SecurePickleWrapper()
        result = wrapper.loads(json_module.dumps({"a": 1}))
        assert result == {"a": 1}

    def test_loads_rejects_non_dict_json(self):
        wrapper = SecurePickleWrapper()
        with pytest.raises(ValueError, match="Expected JSON object"):
            wrapper.loads(json_module.dumps([1, 2, 3]))

    def test_loads_invalid_json_raises_value_error(self):
        wrapper = SecurePickleWrapper()
        with pytest.raises(ValueError, match="Invalid JSON data"):
            wrapper.loads("not json{{{")

    def test_loads_with_pickle_allowed_and_bytes_data(self):
        import pickle

        wrapper = SecurePickleWrapper(allow_pickle=True)
        data = pickle.dumps({"x": 1})
        result = wrapper.loads(data, use_pickle=True)
        assert result == {"x": 1}

    def test_loads_with_pickle_allowed_and_str_data(self):
        import pickle

        wrapper = SecurePickleWrapper(allow_pickle=True)
        # Protocol 0 pickle output for a simple dict is ASCII-safe, so it
        # round-trips through str -> utf-8 encode -> original bytes, exercising
        # the `isinstance(data, str)` branch in SecurePickleWrapper.loads.
        raw_bytes = pickle.dumps({"y": 2}, protocol=0)
        str_data = raw_bytes.decode("ascii")
        assert str_data.encode("utf-8") == raw_bytes
        result = wrapper.loads(str_data, use_pickle=True)
        assert result == {"y": 2}

    def test_loads_pickle_flag_but_wrapper_disallows(self):
        wrapper = SecurePickleWrapper(allow_pickle=False)
        with pytest.raises(ValueError):
            wrapper.loads(b"not valid json", use_pickle=True)

    def test_dumps_json_default(self):
        wrapper = SecurePickleWrapper()
        result = wrapper.dumps({"a": 1})
        assert isinstance(result, bytes)
        assert json_module.loads(result) == {"a": 1}

    def test_dumps_with_pickle_allowed(self):
        import pickle

        wrapper = SecurePickleWrapper(allow_pickle=True)
        result = wrapper.dumps({"a": 1}, use_pickle=True)
        assert pickle.loads(result) == {"a": 1}

    def test_dumps_non_serializable_uses_str_default(self):
        wrapper = SecurePickleWrapper()

        class Foo:
            def __str__(self):
                return "foo-repr"

        result = wrapper.dumps({"obj": Foo()})
        assert json_module.loads(result) == {"obj": "foo-repr"}

    def test_load_from_file_obj(self):
        wrapper = SecurePickleWrapper()
        file_obj = io.BytesIO(json_module.dumps({"k": "v"}).encode("utf-8"))
        result = wrapper.load(file_obj)
        assert result == {"k": "v"}

    def test_dump_to_file_obj(self):
        wrapper = SecurePickleWrapper()
        file_obj = io.BytesIO()
        wrapper.dump({"k": "v"}, file_obj)
        file_obj.seek(0)
        assert json_module.loads(file_obj.read()) == {"k": "v"}


class TestPickleWrapperFactoryFunctions:
    def test_get_secure_pickle_wrapper_default(self):
        wrapper = get_secure_pickle_wrapper()
        assert wrapper.allow_pickle is False

    def test_get_secure_pickle_wrapper_allow_pickle_singleton(self, monkeypatch):
        monkeypatch.setattr(apgi_security_module, "_unsafe_pickle_wrapper", None)
        w1 = get_secure_pickle_wrapper(allow_pickle=True)
        w2 = get_secure_pickle_wrapper(allow_pickle=True)
        assert w1 is w2
        assert w1.allow_pickle is True

    def test_secure_loads_json_helper(self):
        result = secure_loads_json(json_module.dumps({"a": "b"}))
        assert result == {"a": "b"}

    def test_secure_dumps_default_json(self):
        result = secure_dumps({"a": "b"})
        assert json_module.loads(result) == {"a": "b"}

    def test_secure_dumps_with_pickle(self):
        import pickle

        result = secure_dumps({"a": "b"}, use_pickle=True)
        assert pickle.loads(result) == {"a": "b"}

    def test_secure_load_helper(self):
        file_obj = io.BytesIO(json_module.dumps({"z": 9}).encode("utf-8"))
        result = secure_load(file_obj)
        assert result == {"z": 9}

    def test_secure_dump_helper(self):
        file_obj = io.BytesIO()
        secure_dump({"z": 9}, file_obj)
        file_obj.seek(0)
        assert json_module.loads(file_obj.read()) == {"z": 9}


# ---------------------------------------------------------------------------
# require_allowed_subprocess decorator
# ---------------------------------------------------------------------------


class TestRequireAllowedSubprocessDecorator:
    def test_decorator_executes_function_and_sets_policy(self):
        @require_allowed_subprocess(["git"])
        def my_func(x):
            return x * 2

        result = my_func(5)
        assert result == 10
        assert my_func._security_policy.allowed_commands == ["git"]

    def test_decorator_defaults_to_default_allowed_commands(self):
        @require_allowed_subprocess()
        def my_func():
            return "ok"

        result = my_func()
        assert result == "ok"
        assert my_func._security_policy.allowed_commands == DEFAULT_ALLOWED_SUBPROCESS_CMDS


# ---------------------------------------------------------------------------
# Legacy deprecated wrappers
# ---------------------------------------------------------------------------


class TestDeprecatedWrappers:
    def test_deprecated_secure_subprocess_popen_warns_and_delegates(self):
        with pytest.warns(DeprecationWarning):
            with pytest.raises(SecureSubprocessError):
                apgi_security_module._deprecated_secure_subprocess_popen(
                    ["rm", "-rf", "/"]
                )

    def test_deprecated_secure_loads_warns_and_delegates(self):
        # _deprecated_secure_loads calls secure_loads() with use_pickle=False,
        # so any non-JSON payload (including a pickle opcode stream) fails
        # JSON parsing and raises plain ValueError, not PickleSecurityError.
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ValueError):
                apgi_security_module._deprecated_secure_loads(
                    b"cos\nsystem\n(S'x'\ntR."
                )


# ---------------------------------------------------------------------------
# get_env_var / get_sensitive_env_var
# ---------------------------------------------------------------------------


class TestGetEnvVar:
    def test_returns_value_when_set(self, monkeypatch):
        monkeypatch.setenv("MY_TEST_VAR", "hello")
        assert get_env_var("MY_TEST_VAR") == "hello"

    def test_returns_default_when_unset_and_not_required(self, monkeypatch):
        monkeypatch.delenv("MY_UNSET_VAR", raising=False)
        assert get_env_var("MY_UNSET_VAR", default="fallback") == "fallback"

    def test_returns_none_when_unset_and_no_default(self, monkeypatch):
        monkeypatch.delenv("MY_UNSET_VAR", raising=False)
        assert get_env_var("MY_UNSET_VAR") is None

    def test_raises_when_required_and_unset(self, monkeypatch):
        monkeypatch.delenv("MY_UNSET_VAR", raising=False)
        with pytest.raises(ValueError, match="Required environment variable"):
            get_env_var("MY_UNSET_VAR", required=True)

    def test_raises_when_value_shorter_than_min_length(self, monkeypatch):
        monkeypatch.setenv("MY_SHORT_VAR", "abc")
        with pytest.raises(ValueError, match="must be at least"):
            get_env_var("MY_SHORT_VAR", min_length=10)

    def test_passes_when_value_meets_min_length(self, monkeypatch):
        monkeypatch.setenv("MY_LONG_VAR", "a" * 32)
        assert get_env_var("MY_LONG_VAR", min_length=10) == "a" * 32


class TestGetSensitiveEnvVar:
    def test_known_sensitive_key_no_warning(self, monkeypatch):
        monkeypatch.setenv("APGI_KMS_KEY", "k" * 40)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = get_sensitive_env_var("APGI_KMS_KEY")
        assert result == "k" * 40

    def test_unknown_key_warns(self, monkeypatch):
        monkeypatch.setenv("SOME_RANDOM_SECRET", "s" * 40)
        with pytest.warns(UserWarning, match="not in known sensitive keys"):
            result = get_sensitive_env_var("SOME_RANDOM_SECRET")
        assert result == "s" * 40

    def test_required_and_missing_raises(self, monkeypatch):
        monkeypatch.delenv("APGI_JWT_SECRET", raising=False)
        with pytest.raises(ValueError, match="Required environment variable"):
            get_sensitive_env_var("APGI_JWT_SECRET", required=True)

    def test_default_min_length_enforced(self, monkeypatch):
        monkeypatch.setenv("APGI_PSEUDONYM_SALT", "short")
        with pytest.raises(ValueError, match="must be at least"):
            get_sensitive_env_var("APGI_PSEUDONYM_SALT")


# ---------------------------------------------------------------------------
# validate_config_checksum edge cases
# ---------------------------------------------------------------------------


class TestValidateConfigChecksumEdgeCases:
    def test_uses_env_secret_when_not_provided(self, monkeypatch):
        import hashlib
        import hmac
        import json

        monkeypatch.setenv("APGI_CONFIG_SECRET_KEY", "env_secret")
        config = {"a": 1}
        config_str = json.dumps(config, sort_keys=True)
        expected = hmac.new(
            b"env_secret", config_str.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        assert validate_config_checksum(config, expected) is True

    def test_raises_when_no_secret_available(self, monkeypatch):
        monkeypatch.delenv("APGI_CONFIG_SECRET_KEY", raising=False)
        with pytest.raises(ValueError, match="APGI_CONFIG_SECRET_KEY"):
            validate_config_checksum({"a": 1}, "somehash")
