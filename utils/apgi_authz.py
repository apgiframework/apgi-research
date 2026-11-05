"""
Authorization and Identity Management for APGI System

Implements role-based access control (RBAC) and operator identity tracking.
Provides fine-grained permission checks for GUI and autonomous agent actions.
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from utils.apgi_logging import get_logger


class Role(Enum):
    """Operator roles with hierarchical permissions."""

    ADMIN = "admin"  # Full access
    OPERATOR = "operator"  # Run experiments, view results
    ANALYST = "analyst"  # View results only
    AUTONOMOUS_AGENT = "agent"  # Automated experiment execution
    GUEST = "guest"  # Read-only access


@dataclass
class AuthenticationToken:
    """JWT-like authentication token with expiration and refresh capabilities."""

    token_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operator_id: str = ""
    token_type: str = "access"  # access, refresh, session
    issued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=1)
    )
    scopes: List[str] = field(default_factory=list)
    is_revoked: bool = False

    def is_expired(self) -> bool:
        """Check if token is expired."""
        return datetime.now(timezone.utc) > self.expires_at

    def is_valid(self) -> bool:
        """Check if token is valid (not expired and not revoked)."""
        return not self.is_expired() and not self.is_revoked


@dataclass
class TokenPair:
    """Access and refresh token pair."""

    access_token: AuthenticationToken
    refresh_token: AuthenticationToken
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class Permission(Enum):
    """Fine-grained permissions."""

    # Experiment permissions
    RUN_EXPERIMENT = "run_experiment"
    MODIFY_EXPERIMENT = "modify_experiment"
    DELETE_EXPERIMENT = "delete_experiment"
    VIEW_EXPERIMENT = "view_experiment"

    # Configuration permissions
    MODIFY_CONFIG = "modify_config"
    VIEW_CONFIG = "view_config"

    # Data permissions
    EXPORT_DATA = "export_data"
    DELETE_DATA = "delete_data"
    VIEW_DATA = "view_data"

    # System permissions
    MANAGE_USERS = "manage_users"
    VIEW_AUDIT_LOG = "view_audit_log"
    MODIFY_SECURITY = "modify_security"

    # Autonomous agent permissions
    APPROVE_HYPOTHESIS = "approve_hypothesis"
    EXECUTE_PLAN = "execute_plan"


# Role-to-permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: set(Permission),  # All permissions
    Role.OPERATOR: {
        Permission.RUN_EXPERIMENT,
        Permission.MODIFY_EXPERIMENT,
        Permission.VIEW_EXPERIMENT,
        Permission.VIEW_CONFIG,
        Permission.EXPORT_DATA,
        Permission.VIEW_DATA,
        Permission.APPROVE_HYPOTHESIS,
        Permission.EXECUTE_PLAN,
    },
    Role.ANALYST: {
        Permission.VIEW_EXPERIMENT,
        Permission.VIEW_CONFIG,
        Permission.EXPORT_DATA,
        Permission.VIEW_DATA,
    },
    Role.AUTONOMOUS_AGENT: {
        Permission.RUN_EXPERIMENT,
        Permission.MODIFY_EXPERIMENT,
        Permission.VIEW_EXPERIMENT,
        Permission.VIEW_CONFIG,
        Permission.EXPORT_DATA,
        Permission.VIEW_DATA,
        Permission.EXECUTE_PLAN,
    },
    Role.GUEST: {
        Permission.VIEW_EXPERIMENT,
        Permission.VIEW_DATA,
    },
}


@dataclass
class OperatorIdentity:
    """Represents an operator's identity and credentials."""

    operator_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    username: str = ""
    role: Role = Role.GUEST
    token_hash: Optional[str] = None  # Securely store operator token hash
    email: Optional[str] = None
    department: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True

    def has_permission(self, permission: Permission) -> bool:
        """Check if operator has permission."""
        if not self.is_active:
            return False
        permissions = ROLE_PERMISSIONS.get(self.role, set())
        return permission in permissions

    def get_permissions(self) -> Set[Permission]:
        """Get all permissions for this operator."""
        if not self.is_active:
            return set()
        return ROLE_PERMISSIONS.get(self.role, set())


@dataclass
class AuthorizationContext:
    """Context for authorization decisions."""

    operator: OperatorIdentity
    resource_type: str  # experiment, config, data, etc.
    resource_id: str
    action: Permission
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class AuthorizationManager:
    """Manages authorization checks and decisions with enhanced token security."""

    def __init__(self, secret_key: Optional[str] = None) -> None:
        self.logger = get_logger("apgi.authz")
        self.operators: Dict[str, OperatorIdentity] = {}
        self.tokens: Dict[str, str] = {}  # legacy tokens -> operator_id
        self.auth_tokens: Dict[str, AuthenticationToken] = {}  # token_id -> token
        self.refresh_tokens: Dict[str, AuthenticationToken] = (
            {}
        )  # token_id -> refresh_token
        self.sessions: Dict[str, TokenPair] = {}  # session_id -> token_pair
        self.authorization_log: List[Dict] = []
        self.log_limit = 10000  # Cap the in-memory log
        # Load JWT secret from env var so tokens survive process restarts.
        # Fall back to a generated secret with a startup warning (suitable for
        # single-process dev use only).
        env_secret = os.environ.get("APGI_JWT_SECRET")
        if secret_key:
            self.secret_key = secret_key
        elif env_secret:
            self.secret_key = env_secret
        else:
            if os.environ.get("APGI_ENV", "").lower() in ("production", "prod"):
                # Fail closed: an ephemeral key in production silently breaks
                # token verification across restarts/workers (APGI-IMP-002).
                raise RuntimeError(
                    "APGI_JWT_SECRET must be set when APGI_ENV=production"
                )
            self.secret_key = secrets.token_urlsafe(32)
            self.logger.warning(
                "APGI_JWT_SECRET not set — using an ephemeral signing key. "
                "All tokens will be invalidated on process restart. "
                "Set APGI_JWT_SECRET to a stable secret for production use."
            )
        self.token_blacklist: Set[str] = set()  # revoked tokens
        self.locked: bool = False

    def _hash_token(self, token: str) -> str:
        """Hash a token for secure storage."""
        import hashlib

        return hashlib.sha256(token.encode()).hexdigest()

    def register_operator(
        self,
        username: str,
        role: Role,
        email: Optional[str] = None,
        department: Optional[str] = None,
    ) -> tuple[OperatorIdentity, str]:
        """Register a new operator and return identity and secret token."""
        token = str(uuid.uuid4())
        token_hash = self._hash_token(token)

        operator = OperatorIdentity(
            username=username,
            role=role,
            token_hash=token_hash,
            email=email,
            department=department,
        )
        self.operators[operator.operator_id] = operator
        self.tokens[token] = operator.operator_id

        self.logger.info(f"Registered operator {username} with role {role.value}")
        return operator, token

    def create_token_pair(
        self,
        operator_id: str,
        access_token_ttl: timedelta = timedelta(hours=1),
        refresh_token_ttl: timedelta = timedelta(days=30),
        scopes: Optional[List[str]] = None,
    ) -> TokenPair:
        """Create a new access and refresh token pair."""
        if operator_id not in self.operators:
            raise ValueError(f"Operator {operator_id} not found")

        scopes = scopes or ["read", "write"]

        # Create access token
        access_token = AuthenticationToken(
            operator_id=operator_id,
            token_type="access",  # nosec: B105 - token type, not password
            expires_at=datetime.now(timezone.utc) + access_token_ttl,
            scopes=scopes,
        )

        # Create refresh token
        refresh_token = AuthenticationToken(
            operator_id=operator_id,
            token_type="refresh",  # nosec: B105 - token type, not password
            expires_at=datetime.now(timezone.utc) + refresh_token_ttl,
            scopes=["refresh"],
        )

        # Create token pair
        token_pair = TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
        )

        # Store tokens
        self.auth_tokens[access_token.token_id] = access_token
        self.refresh_tokens[refresh_token.token_id] = refresh_token
        self.sessions[token_pair.session_id] = token_pair

        self.logger.info(f"Created token pair for operator {operator_id}")
        return token_pair

    def _create_jwt_token(self, token: AuthenticationToken) -> str:
        """Create a JWT-like token for the authentication token."""
        header = {"alg": "HS256", "typ": "JWT"}

        payload = {
            "token_id": token.token_id,
            "operator_id": token.operator_id,
            "token_type": token.token_type,
            "issued_at": token.issued_at.isoformat(),
            "expires_at": token.expires_at.isoformat(),
            "scopes": token.scopes,
        }

        # Encode header and payload
        header_b64 = (
            base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
        )
        payload_b64 = (
            base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        )

        # Create signature
        message = f"{header_b64}.{payload_b64}"
        signature = hmac.new(
            self.secret_key.encode(), message.encode(), hashlib.sha256
        ).digest()

        signature_b64 = base64.urlsafe_b64encode(signature).decode().rstrip("=")

        return f"{message}.{signature_b64}"

    def _parse_jwt_token(self, jwt_token: str) -> Optional[Dict[str, Any]]:
        """Parse and validate a JWT-like token."""
        try:
            parts = jwt_token.split(".")
            if len(parts) != 3:
                return None

            header_b64, payload_b64, signature_b64 = parts

            # Add padding back if needed
            header_b64 += "=" * (-len(header_b64) % 4)
            payload_b64 += "=" * (-len(payload_b64) % 4)
            signature_b64 += "=" * (-len(signature_b64) % 4)

            # Decode header and payload
            payload: Dict[str, Any] = json.loads(
                base64.urlsafe_b64decode(payload_b64).decode()
            )

            # Verify signature
            message = f"{parts[0]}.{parts[1]}"
            expected_signature = hmac.new(
                self.secret_key.encode(), message.encode(), hashlib.sha256
            ).digest()

            actual_signature = base64.urlsafe_b64decode(signature_b64)

            if not hmac.compare_digest(expected_signature, actual_signature):
                return None

            return payload

        except Exception as e:
            self.logger.error(f"Error parsing JWT token: {e}")
            return None

    def authenticate_with_jwt(self, jwt_token: str) -> Optional[OperatorIdentity]:
        """Authenticate an operator using a JWT-like token."""
        # Check if token is blacklisted
        if jwt_token in self.token_blacklist:
            self.logger.warning("Authentication failed: Token is blacklisted")
            return None

        # Parse and validate JWT token
        payload = self._parse_jwt_token(jwt_token)
        if not payload:
            self.logger.warning("Authentication failed: Invalid JWT token")
            return None

        token_id = payload.get("token_id")
        operator_id = payload.get("operator_id")
        token_type = payload.get("token_type")

        if (
            not token_id or not operator_id or token_type != "access"
        ):  # nosec: B105 - token type, not password
            self.logger.warning("Authentication failed: Invalid token payload")
            return None

        # Get stored token
        stored_token = self.auth_tokens.get(token_id)
        if not stored_token:
            self.logger.warning("Authentication failed: Token not found")
            return None

        # Check if token is valid
        if not stored_token.is_valid():
            self.logger.warning("Authentication failed: Token is expired or revoked")
            return None

        # Get operator
        operator = self.operators.get(operator_id)
        if not operator or not operator.is_active:
            self.logger.warning(
                f"Authentication failed: Operator {operator_id} is inactive or missing"
            )
            return None

        # Update last active
        operator.last_active = datetime.now(timezone.utc)
        return operator

    def refresh_access_token(self, refresh_jwt: str) -> Optional[str]:
        """Refresh an access token using a refresh token."""
        # Parse refresh token
        payload = self._parse_jwt_token(refresh_jwt)
        if not payload:
            self.logger.warning("Token refresh failed: Invalid refresh token")
            return None

        token_id = payload.get("token_id")
        operator_id = payload.get("operator_id")
        token_type = payload.get("token_type")

        if (
            not token_id or not operator_id or token_type != "refresh"
        ):  # nosec: B105 - token type, not password
            self.logger.warning("Token refresh failed: Invalid refresh token payload")
            return None

        # Get stored refresh token
        stored_token = self.refresh_tokens.get(token_id)
        if not stored_token or not stored_token.is_valid():
            self.logger.warning("Token refresh failed: Refresh token is invalid")
            return None

        # Get operator
        operator = self.operators.get(operator_id)
        if not operator or not operator.is_active:
            self.logger.warning("Token refresh failed: Operator is invalid")
            return None

        # Create new access token
        new_access_token = AuthenticationToken(
            operator_id=operator_id,
            token_type="access",  # nosec: B106 - token type, not password
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            scopes=stored_token.scopes,
        )

        # Store new access token
        self.auth_tokens[new_access_token.token_id] = new_access_token

        # Create JWT for new access token
        new_jwt = self._create_jwt_token(new_access_token)

        self.logger.info(f"Refreshed access token for operator {operator_id}")
        return new_jwt

    def revoke_token(self, jwt_token: str) -> bool:
        """Revoke a token by adding it to the blacklist."""
        payload = self._parse_jwt_token(jwt_token)
        if not payload:
            return False

        token_id = payload.get("token_id")
        token_type = payload.get("token_type")

        if token_id:
            # Mark token as revoked in storage
            if (
                token_type == "access" and token_id in self.auth_tokens
            ):  # nosec: B105 - token type, not password
                self.auth_tokens[token_id].is_revoked = True
            elif (
                token_type == "refresh" and token_id in self.refresh_tokens
            ):  # nosec: B105 - token type, not password
                self.refresh_tokens[token_id].is_revoked = True

        # Add to blacklist
        self.token_blacklist.add(jwt_token)

        self.logger.info(f"Revoked {token_type} token {token_id}")
        return True

    def authenticate_operator(self, token: str) -> Optional[OperatorIdentity]:
        """Authenticate an operator using their secret token."""
        operator_id = self.tokens.get(token)
        if not operator_id:
            self.logger.warning("Authentication failed: Invalid token")
            return None

        operator = self.operators.get(operator_id)
        if not operator or not operator.is_active:
            self.logger.warning(
                f"Authentication failed: Operator {operator_id} is inactive or missing"
            )
            return None

        # Verify token hash matches
        if operator.token_hash != self._hash_token(token):
            self.logger.error(
                f"Security breach: Token hash mismatch for operator {operator.username}"
            )
            return None

        operator.last_active = datetime.now(timezone.utc)
        return operator

    def emergency_lock(self, reason: str = "emergency lock") -> int:
        """Containment: deactivate every operator, revoke every issued token and
        refuse further authorization until ``emergency_unlock`` (AUDIT-OPS-001).

        Returns the number of operators deactivated.
        """
        self.locked = True
        count = 0
        for operator in self.operators.values():
            if operator.is_active:
                operator.is_active = False
                count += 1
        self.tokens.clear()
        self.sessions.clear()
        self.auth_tokens.clear()
        self.refresh_tokens.clear()
        self.logger.critical(f"EMERGENCY LOCK: {reason}; {count} operators deactivated")
        return count

    def emergency_unlock(self, operator_ids: Optional[List[str]] = None) -> int:
        """Lift the lock; operators must be reactivated explicitly by id."""
        self.locked = False
        count = 0
        for oid in operator_ids or []:
            op = self.operators.get(oid)
            if op and not op.is_active:
                op.is_active = True
                count += 1
        return count

    def get_operator(self, operator_id: str) -> Optional[OperatorIdentity]:
        """Get operator by ID. NOTE: This does NOT authenticate."""
        return self.operators.get(operator_id)

    def check_permission(
        self,
        operator: OperatorIdentity,
        permission: Permission,
    ) -> bool:
        """Check if operator has permission."""
        has_perm = operator.has_permission(permission)

        if not has_perm:
            self.logger.warning(
                f"Permission denied: {operator.username} lacks {permission.value}"
            )

        return has_perm

    def authorize_action(
        self,
        context: AuthorizationContext,
    ) -> bool:
        """Authorize an action with full context."""
        has_perm = (not self.locked) and context.operator.has_permission(
            context.action
        )

        # Log authorization decision
        log_entry = {
            "request_id": context.request_id,
            "timestamp": context.timestamp.isoformat(),
            "operator_id": context.operator.operator_id,
            "operator_name": context.operator.username,
            "resource_type": context.resource_type,
            "resource_id": context.resource_id,
            "action": context.action.value,
            "decision": "allowed" if has_perm else "denied",
        }
        self.authorization_log.append(log_entry)

        # Enforce log size limit (fixes BUG-L006)
        if len(self.authorization_log) > self.log_limit:
            self.authorization_log = self.authorization_log[-self.log_limit :]

        if not has_perm:
            self.logger.warning(
                f"Authorization denied: {context.operator.username} "
                f"cannot {context.action.value} on {context.resource_type}/{context.resource_id}"
            )
        else:
            self.logger.info(
                f"Authorization granted: {context.operator.username} "
                f"can {context.action.value} on {context.resource_type}/{context.resource_id}"
            )

        return has_perm

    def get_authorization_log(self, limit: int = 100) -> List[Dict]:
        """Get recent authorization log entries."""
        return self.authorization_log[-limit:]

    def deactivate_operator(self, operator_id: str) -> bool:
        """Deactivate an operator."""
        operator = self.operators.get(operator_id)
        if operator:
            operator.is_active = False
            self.logger.info(f"Deactivated operator {operator.username}")
            return True
        return False

    def update_operator_role(self, operator_id: str, new_role: Role) -> bool:
        """Update operator's role."""
        operator = self.operators.get(operator_id)
        if operator:
            old_role = operator.role
            operator.role = new_role
            self.logger.info(
                f"Updated operator {operator.username} role from {old_role.value} to {new_role.value}"
            )
            return True
        return False


# Global authorization manager instance
_authz_manager: Optional[AuthorizationManager] = None


def get_authz_manager() -> AuthorizationManager:
    """Get global authorization manager."""
    global _authz_manager
    if _authz_manager is None:
        _authz_manager = AuthorizationManager()
    return _authz_manager


def set_authz_manager(manager: AuthorizationManager) -> None:
    """Set global authorization manager (for testing)."""
    global _authz_manager
    _authz_manager = manager
