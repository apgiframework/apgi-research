"""Load the definitions in ``train.py`` without running the training script.

``train.py`` is a script: importing it builds the model, loads the tokenizer
from the user's cache and runs the whole training loop. Tests only need its
classes, functions and constants, so this loads everything *above* the
"Setup" section plus the helper functions defined further down, and skips the
statements that run training.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import ModuleType

from tests._module_isolation import installed_module

_TRAIN_PATH = Path(__file__).resolve().parent.parent / "train.py"
_SETUP_MARKER = "# Setup: tokenizer, model, optimizer, dataloader"


def load_train_definitions() -> ModuleType:
    """Return a ``train`` module holding its definitions but no side effects."""
    source = _TRAIN_PATH.read_text()
    setup_line = next(
        number
        for number, line in enumerate(source.splitlines(), start=1)
        if _SETUP_MARKER in line
    )

    keep = []
    for node in ast.parse(source).body:
        definition = isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.ClassDef,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        )
        # Constants and device detection live above the setup section; anything
        # after it (tokenizer, model, optimizer, training loop) is the run.
        setup_free = node.lineno < setup_line
        environ_write = isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Subscript) for target in node.targets
        )
        if (definition or setup_free) and not environ_write:
            keep.append(node)

    module = ModuleType("train")
    module.__file__ = str(_TRAIN_PATH)
    code = compile(ast.Module(body=keep, type_ignores=[]), str(_TRAIN_PATH), "exec")
    # @dataclass resolves the defining module through sys.modules, so the module
    # must be registered while its classes are created.
    with installed_module("train", module):
        exec(code, module.__dict__)  # nosec: B102 - trusted repository source
    return module
