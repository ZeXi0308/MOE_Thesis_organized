#!/usr/bin/env python3
"""Qualify already completed QMSum outputs after the launcher list-reader error.

Preserves original INCOMPLETE launcher receipts. Does not start inference or
modify any raw output; writes one separate qualification addendum.
"""
import json
from pathlib import Path

import C_QMSUM_BLOCK_V1 as setup


def main():
    block, resource, require = setup.block, setup.resource, setup.require
    cell = block.ROOT / "whole"
    original = resource.read(cell / "launcher-receipt.json")
    life = original["child_lifecycle"]
    require(original["status"] == "INCOMPLETE" and len(original["errors"]) == 1
            and original["errors"][0] == "ValueError: JSON object required: "
                + str(cell / "native/measured-outputs.json"), "unexpected original failure")
    require(life["exit_code"] == life["child_returncode"] == 0
            and life["child_reaped"] and not life["timed_out"]
            and life["launcher_error"] is None and life["interruption_signal"] is None
            and block.idle(original["gpu_after"])
            and original["stage_cleanup"]["status"] == "REMOVED",
            "native child or cleanup did not complete")
    native = cell / "native"
    old_read = resource.read

    def read(path):
        if Path(path) == native / "measured-outputs.json":
            value = json.loads(Path(path).read_text())
            require(isinstance(value, list), "measured output is not a list")
            return value
        return old_read(path)

    resource.read = read
    try:
        result = setup.qualify(native)
    finally:
        resource.read = old_read
    addendum = dict(status="QUALIFIED_COMPLETED_NATIVE_OUTPUTS", qualification=result,
        original_launcher_status="INCOMPLETE", gpu_rerun=False,
        original_launcher_receipt_sha256=resource.sha(cell / "launcher-receipt.json"),
        original_block_receipt_sha256=resource.sha(block.ROOT / "pair-receipt.json"),
        original_freeze_sha256=original["freeze_sha256"],
        postqual_source_sha256=resource.sha(Path(__file__)),
        explanation="Original native cell completed200 and exited0; launcher used dict-only resource.read on the output list. This separate addendum repeats the unchanged completion checks with JSON-list reading; all original receipts preserved.")
    target = block.BASE / "qmsum-whole-postqual-v1.json"
    resource.atomic_new(target, addendum)
    print(json.dumps(addendum))


if __name__ == "__main__":
    main()
