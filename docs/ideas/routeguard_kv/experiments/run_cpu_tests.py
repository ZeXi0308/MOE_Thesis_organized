#!/usr/bin/env python3
"""Dependency-free runner for RouteGuard-KV CPU qualification tests."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import tempfile
import time


TEST_MODULES = (
    "test_kv_quant_cache",
    "test_route_lock",
    "test_r0a_artifacts",
    "test_r0a_analysis",
    "test_prepare_r0a_data",
    "test_run_gates",
    "test_olmoe_integration",
)


def main() -> None:
    started = time.time()
    passed = 0
    for module_name in TEST_MODULES:
        module = importlib.import_module(module_name)
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            parameters = inspect.signature(function).parameters
            if not parameters:
                function()
            elif list(parameters) == ["tmp_path"]:
                with tempfile.TemporaryDirectory(prefix="routeguard-kv-test-") as temporary:
                    function(Path(temporary))
            else:
                raise RuntimeError(f"unsupported test fixture for {module_name}.{name}: {list(parameters)}")
            passed += 1
            print(f"PASS {module_name}.{name}")
    print(f"CPU_TESTS_OK passed={passed} elapsed_s={time.time() - started:.3f}")


if __name__ == "__main__":
    main()
