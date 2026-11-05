"""
Standardized CLI Entry Point Framework for APGI Experiments

Provides unified CLI handling, authorization enforcement, and logging setup
for all experiment entry points.

Usage:
    from apgi_cli import cli_entrypoint, require_auth

    @require_auth(Permission.RUN_EXPERIMENT)
    def run_my_experiment():
        ...

    if __name__ == "__main__":
        cli_entrypoint(run_my_experiment)
"""

import argparse
import logging
import os
import sys
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from apgi_audit import AuditEventType, get_audit_sink
from utils.apgi_authz import (
    AuthorizationContext,
    OperatorIdentity,
    Permission,
    Role,
    get_authz_manager,
)
from utils.apgi_logging import APGIContextLogger, get_logger


class CLIError(Exception):
    """Base exception for CLI-related errors."""

    pass


class AuthorizationError(CLIError):
    """Raised when authorization check fails."""

    pass


class ConfigurationError(CLIError):
    """Raised when configuration is invalid."""

    pass


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------


def create_standard_parser(description: str) -> argparse.ArgumentParser:
    """Create a standardized argument parser for experiment runners."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Standard arguments for all experiments
    parser.add_argument(
        "--trials",
        "-n",
        type=int,
        default=None,
        help="Number of trials to run",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path for results",
    )
    parser.add_argument(
        "--apgi-enabled",
        action="store_true",
        default=None,
        help="Enable APGI integration",
    )
    parser.add_argument(
        "--apgi-disabled",
        action="store_true",
        help="Disable APGI integration",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--operator",
        type=str,
        default=None,
        help="Operator ID for authorization",
    )
    parser.add_argument(
        "--role",
        type=str,
        default="operator",
        choices=[r.value for r in Role],
        help="Role for authorization",
    )

    return parser


# ---------------------------------------------------------------------------
# Authorization Enforcement
# ---------------------------------------------------------------------------


def require_auth(
    permission: Permission,
    resource_type: str = "experiment",
    resource_id: Optional[str] = None,
) -> Callable:
    """
    Decorator to enforce authorization at CLI boundaries.

    Usage:
        @require_auth(Permission.RUN_EXPERIMENT)
        def run_experiment():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get operator identity from environment or arguments
            operator = _get_operator_identity()

            # Determine resource ID
            res_id = resource_id or func.__name__

            # Create authorization context
            context = AuthorizationContext(
                operator=operator,
                resource_type=resource_type,
                resource_id=res_id,
                action=permission,
            )

            # Check authorization
            authz = get_authz_manager()
            if not authz.authorize_action(context):
                # Record denial in audit log
                audit = get_audit_sink()
                audit.record_event(
                    event_type=AuditEventType.AUTHORIZATION_DENIED,
                    operator_id=operator.operator_id,
                    operator_name=operator.username,
                    resource_type=resource_type,
                    resource_id=res_id,
                    action=f"{permission.value}_denied",
                    status="denied",
                    error_message=f"Operator lacks {permission.value} permission",
                )
                raise AuthorizationError(
                    f"Authorization denied: {operator.username} lacks {permission.value} "
                    f"permission for {resource_type}/{res_id}"
                )

            # Record successful authorization
            audit = get_audit_sink()
            audit.record_event(
                event_type=AuditEventType.AUTHORIZATION_GRANTED,
                operator_id=operator.operator_id,
                operator_name=operator.username,
                resource_type=resource_type,
                resource_id=res_id,
                action=f"{permission.value}_granted",
                status="success",
            )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def _get_operator_identity() -> OperatorIdentity:
    """Resolve the CLI caller's identity without trusting unverified env values.

    Precedence (APGI-API-002 / APGI-IMP-002):
      1. ``APGI_OPERATOR_TOKEN`` (JWT access token or registered secret token) is
         authenticated against the authorization manager; failure raises.
      2. Explicit local-trust mode (``APGI_TRUSTED_LOCAL=1`` and not production)
         may register the environment-declared name/role, as before.
      3. Otherwise the caller is a least-privilege GUEST. ``APGI_OPERATOR_ID`` and
         ``APGI_OPERATOR_ROLE`` alone never grant privileges.
    """
    authz = get_authz_manager()
    token = os.environ.get("APGI_OPERATOR_TOKEN")
    if token:
        operator = authz.authenticate_with_jwt(token) or authz.authenticate_operator(
            token
        )
        if operator is None:
            raise PermissionError("APGI_OPERATOR_TOKEN failed authentication")
        return operator

    username = os.environ.get("APGI_OPERATOR_NAME", "cli_user")
    trusted_local = (
        os.environ.get("APGI_TRUSTED_LOCAL") == "1"
        and os.environ.get("APGI_ENV", "").lower() not in ("production", "prod")
    )
    role = Role.GUEST
    if trusted_local:
        try:
            role = Role(os.environ.get("APGI_OPERATOR_ROLE", "operator"))
        except ValueError:
            role = Role.GUEST

    operator, _ = authz.register_operator(username=username, role=role)
    return operator


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------


def setup_cli_logging(verbose: bool = False) -> APGIContextLogger:
    """Set up structured logging for CLI execution."""
    logger = get_logger("apgi.cli")

    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    return APGIContextLogger(logger)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def cli_entrypoint(
    main_func: Callable,
    parser: Optional[argparse.ArgumentParser] = None,
    add_common_args: bool = True,
) -> None:
    """
    Standardized entry point wrapper for experiment scripts.

    Handles:
    - Argument parsing
    - Logging setup
    - Error handling
    - Exit codes

    Usage:
        def main(parsed_args):
            # Your experiment code
            return results

        if __name__ == "__main__":
            cli_entrypoint(main)
    """
    if parser is None and add_common_args:
        parser = create_standard_parser(main_func.__doc__ or "APGI Experiment")

    try:
        # Parse arguments if parser provided
        if parser is not None:
            args = parser.parse_args()
        else:
            args = None

        # Determine verbosity
        verbose = getattr(args, "verbose", False) if args else False

        # Setup logging
        logger = setup_cli_logging(verbose)
        logger.info(f"Starting {main_func.__name__}")

        # Run main function
        if args is not None:
            result = main_func(args)
        else:
            result = main_func()

        logger.info(f"Completed {main_func.__name__} successfully")

        # Output results if returned
        if result is not None:
            import json

            print(json.dumps(result, indent=2, default=str))

        sys.exit(0)

    except AuthorizationError as e:
        logger.error(f"Authorization failed: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(77)  # EX_CONFIG - permission issue

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(78)  # EX_CONFIG

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\nInterrupted", file=sys.stderr)
        sys.exit(130)  # Exit code for Ctrl+C

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# OpenAPI Spec Generation
# ---------------------------------------------------------------------------


def generate_openapi_spec(
    title: str,
    description: str,
    version: str = "1.0.0",
    base_path: str = "/api/v1",
    servers: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Generate OpenAPI 3.0 specification for APGI CLI endpoints.

    Args:
        title: API title
        description: API description
        version: API version
        base_path: Base path for API endpoints
        servers: List of server configurations

    Returns:
        OpenAPI specification dictionary
    """
    if servers is None:
        servers = [
            {"url": "http://localhost:8000", "description": "Development server"},
            {
                "url": "https://api.apgi-research.com",
                "description": "Production server",
            },
        ]

    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": title,
            "description": description,
            "version": version,
            "contact": {
                "name": "APGI Research Team",
                "email": "support@apgi-research.com",
            },
            "license": {"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
        },
        "servers": servers,
        "paths": {},
        "components": {
            "schemas": {
                "ExperimentRequest": {
                    "type": "object",
                    "properties": {
                        "trials": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10000,
                            "description": "Number of trials to run",
                        },
                        "output": {
                            "type": "string",
                            "format": "uri",
                            "description": "Output file path for results",
                        },
                        "apgi_enabled": {
                            "type": "boolean",
                            "description": "Enable APGI integration",
                        },
                        "operator": {
                            "type": "string",
                            "description": "Operator ID for authorization",
                        },
                        "role": {
                            "type": "string",
                            "enum": ["admin", "operator", "analyst", "agent", "guest"],
                            "description": "Role for authorization",
                        },
                        "verbose": {
                            "type": "boolean",
                            "description": "Enable verbose output",
                        },
                    },
                },
                "ExperimentResponse": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["success", "error", "running", "completed"],
                        },
                        "results": {
                            "type": "object",
                            "description": "Experiment results",
                        },
                        "error": {
                            "type": "string",
                            "description": "Error message if status is error",
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time",
                            "description": "Response timestamp",
                        },
                    },
                },
                "AuthorizationResponse": {
                    "type": "object",
                    "properties": {
                        "authorized": {"type": "boolean"},
                        "operator_id": {"type": "string"},
                        "permissions": {"type": "array", "items": {"type": "string"}},
                        "token": {
                            "type": "string",
                            "description": "Authentication token",
                        },
                    },
                },
                "ErrorResponse": {
                    "type": "object",
                    "properties": {
                        "error": {"type": "string"},
                        "message": {"type": "string"},
                        "code": {"type": "integer"},
                        "timestamp": {"type": "string", "format": "date-time"},
                    },
                },
            },
            "securitySchemes": {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT",
                }
            },
        },
        "security": [{"BearerAuth": []}],
    }

    # Add common paths
    spec["paths"] = {
        f"{base_path}/experiments": {
            "post": {
                "summary": "Run an experiment",
                "description": "Execute an APGI experiment with specified parameters",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ExperimentRequest"}
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Experiment started successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ExperimentResponse"
                                }
                            }
                        },
                    },
                    "401": {
                        "description": "Unauthorized",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        },
                    },
                    "500": {
                        "description": "Internal server error",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        },
                    },
                },
                "tags": ["experiments"],
            }
        },
        f"{base_path}/experiments/{{experiment_id}}": {
            "get": {
                "summary": "Get experiment status",
                "description": "Retrieve the current status and results of an experiment",
                "parameters": [
                    {
                        "name": "experiment_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Unique identifier for the experiment",
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Experiment status retrieved successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ExperimentResponse"
                                }
                            }
                        },
                    },
                    "404": {
                        "description": "Experiment not found",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        },
                    },
                },
                "tags": ["experiments"],
            },
            "delete": {
                "summary": "Cancel experiment",
                "description": "Cancel a running experiment",
                "parameters": [
                    {
                        "name": "experiment_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "Unique identifier for the experiment",
                    }
                ],
                "responses": {
                    "200": {"description": "Experiment cancelled successfully"},
                    "404": {
                        "description": "Experiment not found",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        },
                    },
                },
                "tags": ["experiments"],
            },
        },
        f"{base_path}/auth/authorize": {
            "post": {
                "summary": "Authorize operator",
                "description": "Authorize an operator for specific actions",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "operator_id": {"type": "string"},
                                    "permission": {
                                        "type": "string",
                                        "enum": [
                                            "run_experiment",
                                            "view_results",
                                            "manage_system",
                                        ],
                                    },
                                    "resource_type": {"type": "string"},
                                    "resource_id": {"type": "string"},
                                },
                                "required": ["operator_id", "permission"],
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Authorization successful",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/AuthorizationResponse"
                                }
                            }
                        },
                    },
                    "401": {
                        "description": "Authorization failed",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        },
                    },
                },
                "tags": ["authorization"],
            }
        },
        f"{base_path}/health": {
            "get": {
                "summary": "Health check",
                "description": "Check API health status",
                "responses": {
                    "200": {
                        "description": "API is healthy",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {
                                            "type": "string",
                                            "enum": ["healthy", "unhealthy"],
                                        },
                                        "timestamp": {
                                            "type": "string",
                                            "format": "date-time",
                                        },
                                        "version": {"type": "string"},
                                    },
                                }
                            }
                        },
                    }
                },
                "tags": ["system"],
            }
        },
    }

    return spec


def save_openapi_spec(
    spec: Dict[str, Any], output_path: str = "openapi.json", format: str = "json"
) -> None:
    """
    Save OpenAPI specification to file.

    Args:
        spec: OpenAPI specification dictionary
        output_path: Output file path
        format: Output format ('json' or 'yaml')
    """
    import json

    import yaml

    if format.lower() == "yaml":
        with open(output_path, "w") as f:
            yaml.dump(spec, f, default_flow_style=False, sort_keys=False)
    else:
        with open(output_path, "w") as f:
            json.dump(spec, f, indent=2, sort_keys=False)


def add_cli_command_to_spec(
    spec: Dict[str, Any],
    command_name: str,
    description: str,
    parameters: Optional[List[Dict[str, Any]]] = None,
    method: str = "POST",
) -> None:
    """
    Add a CLI command to the OpenAPI specification.

    Args:
        spec: OpenAPI specification to modify
        command_name: Name of the CLI command
        description: Description of the command
        parameters: List of command parameters
        method: HTTP method for the endpoint
    """
    if parameters is None:
        parameters = []

    path = f"/api/v1/cli/{command_name}"

    # Build request body schema from parameters
    properties = {}
    required = []

    for param in parameters:
        param_name = param["name"]
        properties[param_name] = {
            "type": param.get("type", "string"),
            "description": param.get("description", ""),
        }

        if param.get("required", False):
            required.append(param_name)

        # Add enum if specified
        if "choices" in param:
            properties[param_name]["enum"] = param["choices"]

    # Add the path to the spec
    spec["paths"][path] = {
        method.lower(): {
            "summary": command_name,
            "description": description,
            "tags": ["cli"],
            "requestBody": {
                "required": len(required) > 0,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        }
                    }
                },
            },
            "responses": {
                "200": {
                    "description": "Command executed successfully",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "status": {"type": "string"},
                                    "result": {"type": "object"},
                                    "message": {"type": "string"},
                                },
                            }
                        }
                    },
                },
                "400": {
                    "description": "Invalid request",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                        }
                    },
                },
                "401": {
                    "description": "Unauthorized",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                        }
                    },
                },
            },
        }
    }


# ---------------------------------------------------------------------------
# Legacy Compatibility
# ---------------------------------------------------------------------------


def standardized_main(
    run_func: Callable,
    experiment_name: str,
    args: Optional[List[str]] = None,
) -> int:
    """
    Legacy-compatible standardized main function.

    For backward compatibility with existing run_*.py files.
    """
    parser = create_standard_parser(f"Run {experiment_name} experiment")
    parsed_args = parser.parse_args(args)

    try:
        setup_cli_logging(parsed_args.verbose)
        result = run_func(parsed_args)
        if result is not None:
            import json

            print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


# Convenience exports
__all__ = [
    "cli_entrypoint",
    "require_auth",
    "create_standard_parser",
    "setup_cli_logging",
    "standardized_main",
    "CLIError",
    "AuthorizationError",
    "ConfigurationError",
    "generate_openapi_spec",
    "save_openapi_spec",
    "add_cli_command_to_spec",
]
