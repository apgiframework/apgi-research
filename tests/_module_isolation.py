"""Helpers that keep ``sys.modules`` mocks from leaking between test files.

Assigning mocks into ``sys.modules`` at import time of a test file applies to
the whole pytest session (collection happens before any test runs), so every
other file silently imports mocks instead of the real packages. These helpers
scope such mocks to a block, and restore only the entries they touched.
"""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Dict, Iterator, Mapping, Sequence

_MISSING = object()


@contextmanager
def mocked_modules(
    mocks: Mapping[str, Any] | None = None, fresh: Sequence[str] = ()
) -> Iterator[None]:
    """Temporarily install ``mocks`` in ``sys.modules`` and drop ``fresh`` names.

    On exit every touched entry is put back exactly as it was (or removed if it
    was absent). Entries that were not touched are left alone, so unrelated
    modules imported inside the block (numpy, torch, ...) are not reloaded.
    """
    mocks = dict(mocks or {})
    names = [*mocks, *fresh]
    saved: Dict[str, Any] = {name: sys.modules.get(name, _MISSING) for name in names}
    sys.modules.update(mocks)
    for name in fresh:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name, original in saved.items():
            if original is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def import_with_mocks(name: str, mocks: Mapping[str, Any]) -> ModuleType:
    """Import ``name`` afresh while ``mocks`` stand in for its dependencies.

    The returned module keeps references to the mocks, but ``sys.modules`` is
    restored afterwards so other files import the real module and packages.
    """
    with mocked_modules(mocks, fresh=[name]):
        return importlib.import_module(name)


@contextmanager
def installed_module(name: str, module: ModuleType) -> Iterator[None]:
    """Make ``name`` resolve to ``module`` (e.g. for ``patch("name.attr")``)."""
    with mocked_modules({name: module}):
        yield
