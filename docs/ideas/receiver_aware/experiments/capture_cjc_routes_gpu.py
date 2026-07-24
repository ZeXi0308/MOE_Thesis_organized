#!/usr/bin/env python3
"""Capture an identity-complete, route-real CJC contribution DAG on CUDA.

This producer records logical routing only.  It never fabricates expert-ready,
network-arrival or ACK timestamps; the reviewed replay derives those fields
from separately versioned LUT/config inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
SHARED = REPO_ROOT / "experiments" / "shared"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from capture_moe import patch_mixtral_moe  # noqa: E402


MODEL_SPECS = {
    "olmoe": {
        "model_id": "allenai/OLMoE-1B-7B-0924",
        "revision": "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
        "num_experts": 64,
        "top_k": 8,
    },
    "llm_jp": {
        "model_id": "llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M",
        "revision": "1d5983076dfc67aee4a77ec06a27027f5bab6055",
        "num_experts": 32,
        "top_k": 16,
    },
}


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=HERE.parent / "CJC_Phase2_冻结实验协议_2026-07-22.md")
    parser.add_argument("--mode", choices=("dev", "formal"), default="dev")
    parser.add_argument("--signoff", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--placement", choices=("contiguous", "round_robin"), default="contiguous")
    return parser.parse_args()


def _load_manifest(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise RuntimeError("data manifest schema_version must be 1")
    supplied = value.get("manifest_sha256")
    unhashed = dict(value)
    unhashed.pop("manifest_sha256", None)
    actual = sha256_bytes(canonical_json_bytes(unhashed))
    if supplied != actual:
        raise RuntimeError("data manifest self-hash mismatch")
    if value.get("sequence_tokens") != 128:
        raise RuntimeError("formal CJC capture requires exactly 128 tokens")
    requests = value.get("requests")
    if not isinstance(requests, list) or not requests:
        raise RuntimeError("data manifest has no requests")
    return value


def _source_hash() -> str:
    files = (Path(__file__), SHARED / "capture_moe.py")
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _require_signoff(
    path: Path | None,
    *,
    protocol_sha256: str,
    source_sha256: str,
    data_manifest_sha256: str,
) -> Mapping[str, Any]:
    if path is None or not path.is_file():
        raise RuntimeError("formal capture requires a Phase-4 SIGNED-OFF attestation")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "SIGNED-OFF",
        "protocol_sha256": protocol_sha256,
        "capture_source_sha256": source_sha256,
        "data_manifest_sha256": data_manifest_sha256,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise RuntimeError(f"formal signoff mismatch for {key}")
    return value


def _origin_lpt(requests: list[Mapping[str, Any]], ep_size: int) -> dict[str, int]:
    if ep_size < 1:
        raise ValueError("ep_size must be positive")
    loads = [0] * ep_size
    result: dict[str, int] = {}
    weighted = [(str(row["request_id"]), 128) for row in requests]
    for request_id, weight in sorted(weighted, key=lambda item: (-item[1], item[0])):
        rank = min(range(ep_size), key=lambda candidate: (loads[candidate], candidate))
        result[request_id] = rank
        loads[rank] += weight
    return result


def _expert_sender(expert_id: int, num_experts: int, ep_size: int, placement: str) -> int:
    if placement == "round_robin":
        return expert_id % ep_size
    return min(ep_size - 1, expert_id * ep_size // num_experts)


def _load_model_and_tokenizer(args: argparse.Namespace, spec: Mapping[str, Any]):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment capability
        raise RuntimeError("transformers is required for GPU route capture") from exc
    common = {
        "revision": spec["revision"],
        "cache_dir": str(args.cache_dir) if args.cache_dir else None,
        "local_files_only": not args.allow_download,
    }
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], **common)
    model = AutoModelForCausalLM.from_pretrained(
        spec["model_id"],
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        **common,
    )
    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; proxy fallback is forbidden")
    if args.output_dir.exists():
        raise RuntimeError("refusing to overwrite an existing output directory")
    manifest = _load_manifest(args.data_manifest)
    spec = MODEL_SPECS[args.model_key]
    expected_revision = f"{spec['model_id']}@{spec['revision']}"
    revisions = manifest.get("model_revisions")
    if not isinstance(revisions, dict) or revisions.get(args.model_key) != expected_revision:
        raise RuntimeError("data/model revision mismatch")
    protocol_sha = sha256_file(args.protocol)
    source_sha = _source_hash()
    manifest_sha = str(manifest["manifest_sha256"])
    signoff = None
    if args.mode == "formal":
        signoff = _require_signoff(
            args.signoff,
            protocol_sha256=protocol_sha,
            source_sha256=source_sha,
            data_manifest_sha256=manifest_sha,
        )

    requests = manifest["requests"]
    assert isinstance(requests, list)
    if len({str(row["request_id"]) for row in requests}) != len(requests):
        raise RuntimeError("duplicate request_id in data manifest")
    origin = _origin_lpt(requests, args.ep_size)
    expert_to_sender_by_model = {}
    for candidate in MODEL_SPECS.values():
        candidate_revision = f"{candidate['model_id']}@{candidate['revision']}"
        expert_to_sender_by_model[candidate_revision] = {
            str(expert_id): _expert_sender(
                expert_id,
                int(candidate["num_experts"]),
                args.ep_size,
                args.placement,
            )
            for expert_id in range(int(candidate["num_experts"]))
        }
    placement_payload = {
        "schema_version": 1,
        "ep_size": args.ep_size,
        "gpus_per_node": 4,
        "placement": args.placement,
        "expert_to_sender_by_model": expert_to_sender_by_model,
        "request_to_receiver": origin,
    }
    placement_sha = sha256_bytes(canonical_json_bytes(placement_payload))
    placement_payload["manifest_sha256"] = placement_sha

    model, tokenizer = _load_model_and_tokenizer(args, spec)
    recorder = patch_mixtral_moe(
        model,
        policy_name="full",
        record_routes=True,
        record_diagnostics=False,
    )
    expected_top_k = int(spec["top_k"])
    route_row_count = 0
    join_set_count = 0
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{args.output_dir.name}.partial-", dir=args.output_dir.parent
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        with (temporary / "route_trace.jsonl").open("x", encoding="utf-8") as trace:
            for request_index, request in enumerate(requests):
                request_id = str(request["request_id"])
                text = str(request["text"])
                text_sha = sha256_bytes(text.encode("utf-8"))
                if text_sha != request.get("text_sha256"):
                    raise RuntimeError(f"text hash mismatch for {request_id}")
                encoded = tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"]
                if tuple(input_ids.shape) != (1, 128):
                    raise RuntimeError(
                        f"request {request_id} is not exactly 128 valid tokens"
                    )
                before = len(recorder.route_batches)
                recorder.set_sample_id(request_index)
                with torch.inference_mode():
                    model(input_ids=input_ids.to("cuda:0"), use_cache=False)
                torch.cuda.synchronize()
                batches = recorder.route_batches[before:]
                if not batches:
                    raise RuntimeError("route-only capture emitted no route batches")
                forward_id = f"{request_id}:prefill:0"
                batch_id = "batch-0"
                seen_layers: set[int] = set()
                for route_event_index, batch in enumerate(batches):
                    experts = batch["selected_experts"]
                    weights = batch["routing_weights"]
                    if not isinstance(experts, torch.Tensor) or not isinstance(
                        weights, torch.Tensor
                    ):
                        raise RuntimeError("recorder emitted non-tensor routes")
                    if experts.shape != weights.shape or experts.shape != (
                        128,
                        expected_top_k,
                    ):
                        raise RuntimeError("captured route shape/top-k mismatch")
                    layer_id = int(batch["layer"])
                    if layer_id in seen_layers:
                        raise RuntimeError("duplicate layer in one forward route capture")
                    seen_layers.add(layer_id)
                    for token_position in range(128):
                        selected = [
                            int(value)
                            for value in experts[token_position].tolist()
                        ]
                        if len(set(selected)) != expected_top_k:
                            raise RuntimeError(
                                "duplicate expert in one token top-k closure"
                            )
                        token_id = f"{request_id}:tok:{token_position:03d}"
                        for topk_slot, expert_id in enumerate(selected):
                            if not 0 <= expert_id < int(spec["num_experts"]):
                                raise RuntimeError("expert id outside frozen model spec")
                            row = {
                                "schema_version": "cjc-route-v1",
                                "model_revision": expected_revision,
                                "data_manifest_sha256": manifest_sha,
                                "request_id": request_id,
                                "forward_id": forward_id,
                                "batch_id": batch_id,
                                "phase": "prefill",
                                "decode_step": 0,
                                "route_event_index": route_event_index,
                                "layer_id": layer_id,
                                "token_id": token_id,
                                "token_position": token_position,
                                "topk_slot": topk_slot,
                                "expert_id": expert_id,
                                "sender_rank": _expert_sender(
                                    expert_id,
                                    int(spec["num_experts"]),
                                    args.ep_size,
                                    args.placement,
                                ),
                                "receiver_rank": origin[request_id],
                                "valid": True,
                                "route_weight": float(
                                    weights[token_position, topk_slot].item()
                                ),
                                "route_source": "native_model_forward",
                                "placement_manifest_sha256": placement_sha,
                            }
                            trace.write(json.dumps(row, sort_keys=True) + "\n")
                            route_row_count += 1
                        join_set_count += 1
                # The recorder is only a per-forward transport here.  Keeping
                # every CPU route tensor would reintroduce an avoidable
                # whole-corpus memory term after the JSONL path became
                # streaming.
                recorder.route_batches.clear()
                recorder.routing_weight_batches.clear()

        (temporary / "placement.json").write_text(
            json.dumps(placement_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metadata = {
            "schema_version": 1,
            "status": "CAPTURE_ONLY" if args.mode == "formal" else "NOT_TESTED",
            "scientific_result": False,
            "evidence_boundary": "ROUTE_REAL / NO_READY_OR_NETWORK_TIMESTAMPS / NOT_RDMA",
            "mode": args.mode,
            "model_revision": expected_revision,
            "protocol_sha256": protocol_sha,
            "capture_source_sha256": source_sha,
            "data_manifest_sha256": manifest_sha,
            "placement_manifest_sha256": placement_sha,
            "route_rows": route_row_count,
            "join_sets": join_set_count,
            "top_k": expected_top_k,
            "gpu_name": torch.cuda.get_device_name(0),
            "signoff_sha256": sha256_file(args.signoff) if signoff is not None else None,
        }
        (temporary / "capture_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.rename(args.output_dir)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
