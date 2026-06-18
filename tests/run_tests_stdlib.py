# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Stdlib-only test runner — fallback when pytest is not installed.

Discovers test_*.py files, finds top-level `test_*` functions, runs them, and
prints a pytest-like summary. Useful for offline/CI smoke checks without any
third-party dependency.

Usage:
    cd flyseek_extend
    python tests/run_tests_stdlib.py
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import traceback
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parent.parent


def _shim_pytest_if_missing() -> None:
    """Provide a minimal `pytest` namespace so test files can `import pytest`.

    Only `pytest.raises` is supported because that is all we use.
    """
    try:
        import pytest  # noqa: F401
        return
    except ImportError:
        pass

    import contextlib
    import re as _re
    import types

    class _RaisesCtx:
        def __init__(self, expected: type, match: str | None = None):
            self.expected = expected
            self.match = match

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                raise AssertionError(
                    f"DID NOT RAISE {self.expected.__name__}"
                )
            if not issubclass(exc_type, self.expected):
                return False  # let it propagate
            if self.match is not None and not _re.search(self.match, str(exc)):
                raise AssertionError(
                    f"raised {exc_type.__name__}({exc!r}) "
                    f"does not match {self.match!r}"
                )
            return True

    pytest_module = types.ModuleType("pytest")
    pytest_module.raises = lambda exc, match=None: _RaisesCtx(exc, match)
    pytest_module.fixture = lambda *a, **kw: (lambda f: f)
    pytest_module.mark = types.SimpleNamespace(
        parametrize=lambda *a, **kw: (lambda f: f)
    )
    sys.modules["pytest"] = pytest_module


def _load_test_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _collect_tests(module) -> list[tuple[str, Callable]]:
    return [
        (name, fn)
        for name, fn in inspect.getmembers(module, inspect.isfunction)
        if name.startswith("test_")
    ]


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    _shim_pytest_if_missing()

    test_files = sorted((REPO_ROOT / "tests").glob("test_*.py"))
    if not test_files:
        print("No test_*.py files found.")
        return 0

    passed = 0
    failed: list[tuple[str, str]] = []

    for test_file in test_files:
        rel = test_file.relative_to(REPO_ROOT)
        print(f"\n=== {rel} ===")
        try:
            module = _load_test_module(test_file)
        except Exception:
            failed.append((str(rel), traceback.format_exc()))
            print(f"  IMPORT FAILED: {rel}")
            continue

        for name, fn in _collect_tests(module):
            try:
                fn()
            except Exception:
                failed.append((f"{rel}::{name}", traceback.format_exc()))
                print(f"  FAIL  {name}")
            else:
                passed += 1
                print(f"  PASS  {name}")

    total = passed + len(failed)
    print("\n" + "=" * 60)
    if failed:
        print(f"FAILURES ({len(failed)}/{total}):")
        for name, tb in failed:
            print(f"\n--- {name} ---\n{tb}")
        print(f"\n{passed} passed, {len(failed)} failed.")
        return 1

    print(f"{passed} passed, 0 failed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
