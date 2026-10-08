#!/usr/bin/env python3
"""Repair V1's class-dedent scheduler anchors; keep its policy/lifecycle."""
from __future__ import annotations

import argparse
from pathlib import Path
from types import MethodType

import C_QMSUM_BACKFILL_CELL_V1 as v1
from C_SPARE_CELL_HELPERS_V1 import dump, sha


def make_backfill_schedule(native_schedule, probe, restore):
    func, path, source = v1.pinned_scheduler_source(native_schedule)
    # inspect.getsource(method) is dedented four spaces by V1 before matching.
    failed_head = (
        "            if new_blocks is None:\n"
        "                # The request cannot be scheduled.\n\n"
        "                # NOTE: we need to untouch the request from the encode cache\n"
        "                # manager\n"
        "                if request.has_encoder_inputs:\n"
        "                    self.encoder_cache_manager.free(request)\n"
        "                break\n"
    )
    failed_backfill = failed_head.replace(
        "                break\n",
        "                if _c_qmsum_backfill_probe(self, request_queue, request):\n"
        "                    continue\n"
        "                break\n",
    )
    restore_at_end = (
        "        if step_skipped_waiting:\n"
        "            self.skipped_waiting.prepend_requests(step_skipped_waiting)\n"
    )
    restore_patched = restore_at_end + "        _c_qmsum_backfill_restore(self)\n"
    if source.count(failed_head) != 1 or source.count(restore_at_end) != 1:
        raise RuntimeError("pinned dedented scheduler waiting branch anchors differ")
    patched = source.replace(failed_head, failed_backfill).replace(
        restore_at_end, restore_patched)
    scope = dict(func.__globals__)
    scope.update(_c_qmsum_backfill_probe=probe, _c_qmsum_backfill_restore=restore)
    namespace = {}
    exec(compile(patched, str(path), "exec"), scope, namespace)
    return MethodType(namespace["schedule"], native_schedule.__self__)


def run(inputs, output, parent, model_dir, model_manifest):
    original = v1.make_backfill_schedule
    v1.make_backfill_schedule = make_backfill_schedule
    try:
        v1.run(inputs, output, parent, model_dir, model_manifest)
    finally:
        v1.make_backfill_schedule = original
        if output.is_dir():
            dump(output / "qmsum-backfill-v2-repair-source.json", dict(
                schema="c-qmsum-backfill-v2-repair-source-v1",
                v2_sha256=sha(Path(__file__)), v1_sha256=sha(Path(v1.__file__)),
                pinned_scheduler_sha256=v1.SCHEDULER_SHA,
                repair="V1 used undedented class indentation in its two exact source anchors; V2 matches the dedented method text only",
                scope="Source-anchor repair only; V1 policy, state hooks, inputs and lifecycle unchanged"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "output", "parent", "model-dir", "model-manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.inputs, args.output, args.parent, args.model_dir, args.model_manifest)
