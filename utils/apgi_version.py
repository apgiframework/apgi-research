"""
Semantic versioning for APGI.
pyproject.toml is the single source of truth: a source checkout reads it
directly (an installed copy's metadata can be stale); otherwise the installed
package metadata is used.
"""

import re
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _read_pyproject_version() -> str | None:
    try:
        text = _PYPROJECT.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    return match.group(1) if match else None


def _read_installed_version() -> str:
    try:
        from importlib.metadata import version

        return version("autoresearch")
    except Exception:
        # Ultimate fallback if package is not installed
        return "1.1.0"


__version__ = _read_pyproject_version() or _read_installed_version()


def get_version() -> str:
    """Get the version from pyproject.toml (single source of truth)."""
    return __version__
