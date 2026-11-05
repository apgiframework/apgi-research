"""APGI-API-002 / APGI-IMP-002 / AUDIT-OPS-001: identity, signing key, tokens, lockdown."""

import pytest

from utils.apgi_authz import (AuthorizationContext, AuthorizationManager,
                              Permission, Role, set_authz_manager)


@pytest.fixture
def mgr(monkeypatch):
    monkeypatch.delenv("APGI_JWT_SECRET", raising=False)
    monkeypatch.delenv("APGI_ENV", raising=False)
    m = AuthorizationManager()
    set_authz_manager(m)
    return m


def test_production_requires_stable_signing_key(monkeypatch):
    monkeypatch.delenv("APGI_JWT_SECRET", raising=False)
    monkeypatch.setenv("APGI_ENV", "production")
    with pytest.raises(RuntimeError, match="APGI_JWT_SECRET"):
        AuthorizationManager()
    monkeypatch.setenv("APGI_JWT_SECRET", "x" * 40)
    assert AuthorizationManager().secret_key == "x" * 40


def test_token_identity_is_the_supplied_operator_not_the_environment(mgr, monkeypatch):
    monkeypatch.setenv("APGI_OPERATOR_ID", "attacker")
    op, _ = mgr.register_operator("alice", Role.OPERATOR)
    pair = mgr.create_token_pair(op.operator_id)
    assert pair.access_token.operator_id == op.operator_id
    assert pair.refresh_token.operator_id == op.operator_id
    jwt = mgr._create_jwt_token(pair.access_token)
    assert mgr.authenticate_with_jwt(jwt).operator_id == op.operator_id
    with pytest.raises(ValueError):
        mgr.create_token_pair("unknown")


def _cli():
    import apgi_cli

    return apgi_cli


def test_cli_env_role_does_not_grant_privilege(mgr, monkeypatch):
    for k in ("APGI_OPERATOR_TOKEN", "APGI_TRUSTED_LOCAL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("APGI_OPERATOR_ROLE", "admin")
    monkeypatch.setenv("APGI_OPERATOR_ID", "someone-else")
    op = _cli()._get_operator_identity()
    assert op.role == Role.GUEST


def test_cli_trusted_local_is_opt_in_and_never_in_production(mgr, monkeypatch):
    monkeypatch.delenv("APGI_OPERATOR_TOKEN", raising=False)
    monkeypatch.setenv("APGI_OPERATOR_ROLE", "operator")
    monkeypatch.setenv("APGI_TRUSTED_LOCAL", "1")
    assert _cli()._get_operator_identity().role == Role.OPERATOR
    monkeypatch.setenv("APGI_ENV", "production")
    assert _cli()._get_operator_identity().role == Role.GUEST


def test_cli_token_authenticates_or_fails(mgr, monkeypatch):
    op, secret = mgr.register_operator("bob", Role.ANALYST)
    monkeypatch.setenv("APGI_OPERATOR_TOKEN", secret)
    assert _cli()._get_operator_identity().operator_id == op.operator_id
    monkeypatch.setenv("APGI_OPERATOR_TOKEN", "forged")
    with pytest.raises(PermissionError):
        _cli()._get_operator_identity()


def test_emergency_lock_revokes_everything_and_blocks_authorization(mgr):
    op, secret = mgr.register_operator("carol", Role.OPERATOR)
    jwt = mgr._create_jwt_token(mgr.create_token_pair(op.operator_id).access_token)
    ctx = AuthorizationContext(op, "experiment", "e1", Permission.RUN_EXPERIMENT)
    assert mgr.authorize_action(ctx) is True
    assert mgr.emergency_lock("test") == 1
    assert mgr.authorize_action(ctx) is False
    assert mgr.authenticate_operator(secret) is None
    assert mgr.authenticate_with_jwt(jwt) is None
    assert mgr.emergency_unlock([op.operator_id]) == 1
    assert mgr.authorize_action(ctx) is True
