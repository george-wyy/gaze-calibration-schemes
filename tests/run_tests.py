#!/usr/bin/env python3
"""Run the test suite without a test framework.

    python tests/run_tests.py

Equivalent to ``pytest tests/`` when pytest is available. This runner exists so that
verifying the re-implementations needs an interpreter and numpy, nothing else.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import traceback

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

MODULES = ["test_schemes"]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    passed = 0
    failed: list[str] = []
    for module_name in MODULES:
        module = load(module_name)
        for attr in sorted(dir(module)):
            if not attr.startswith("test_"):
                continue
            fn = getattr(module, attr)
            if not callable(fn):
                continue
            label = f"{module_name}.{attr}"
            try:
                fn()
            except Exception:  # noqa: BLE001 - the runner reports, it does not handle
                failed.append(label)
                print(f"FAIL  {label}")
                traceback.print_exc()
            else:
                passed += 1
                print(f"ok    {label}")
    total = passed + len(failed)
    print(f"\n{passed}/{total} passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
