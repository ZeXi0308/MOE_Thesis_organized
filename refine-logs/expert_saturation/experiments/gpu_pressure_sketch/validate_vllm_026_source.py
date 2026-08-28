"""Fail-closed source check for the vLLM 0.26 integration sketch."""

from __future__ import annotations

import argparse
from pathlib import Path


EXPECTED = {
    "vllm/model_executor/layers/fused_moe/router/base_router.py": (
        "self.capture_fn(topk_ids)",
        "topk_ids = self._apply_eplb_mapping(topk_ids)",
    ),
    "vllm/v1/worker/gpu_model_runner.py": (
        "self.req_indices.copy_to_gpu(total_num_scheduled_tokens)",
        "self.query_start_loc.copy_to_gpu()",
        "self._model_forward(",
        "event.synchronize()",
    ),
    "vllm/v1/engine/core.py": (
        "scheduler_output = self.scheduler.schedule(",
        "self.scheduler.update_from_output(",
    ),
    "vllm/model_executor/layers/fused_moe/routed_experts_capturer.py": (
        "self.device_buffer[:token_num_per_dp, layer_id, :]",
        "self.device_buffer.zero_()",
    ),
}


def validate(root: Path) -> list[str]:
    failures: list[str] = []
    for relative, needles in EXPECTED.items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing file: {relative}")
            continue
        text = path.read_text()
        for needle in needles:
            if needle not in text:
                failures.append(f"missing source anchor {needle!r} in {relative}")
    # Ordering matters: capture is of logical IDs before EPLB physical mapping.
    router = (root / "vllm/model_executor/layers/fused_moe/router/base_router.py").read_text()
    if router.index("self.capture_fn(topk_ids)") > router.index(
        "topk_ids = self._apply_eplb_mapping(topk_ids)"
    ):
        failures.append("capture hook moved after EPLB mapping")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vllm_root", type=Path)
    args = parser.parse_args()
    failures = validate(args.vllm_root)
    if failures:
        raise SystemExit("\n".join(failures))
    print("vLLM 0.26 source anchors: PASS")


if __name__ == "__main__":
    main()

