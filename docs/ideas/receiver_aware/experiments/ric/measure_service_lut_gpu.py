#!/usr/bin/env python3
"""Measure RIC-v1 service/control LUTs from calibration data on CUDA.

The LUT keeps every raw repeat and source tag.  Expert timings use each
selected expert's own natively routed hidden states.  H2D is explicitly
``H2D_NOT_RDMA``.  Host measurements exercise the one canonical RIC wire
path; this producer deliberately contains no second struct codec.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
EXPERIMENTS_ROOT = HERE.parent
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from capture_routes_gpu import (  # noqa: E402
    NATIVE_TOPK_SELECTION_RULE,
    RouteCaptureError,
    _gpu_compute_apps,
    _gpu_environment,
    _load_config,
    _load_data_manifest,
    _load_model_and_tokenizer,
    _normalizes_topk,
    _producer_source_sha256 as _capture_routes_source_sha256,
    _routes_from_logits,
    discover_moe_modules,
    model_load_reference,
    selected_layers,
)
from measure_capability_gpu import (  # noqa: E402
    _first_tensor,
    _reconstruct_real_contributions,
    _receiver_unpack,
    _sender_pack,
    _producer_source_sha256 as _capability_source_sha256,
    canonical_reduce,
)
from prepare_data import (  # noqa: E402
    DataPreparationError,
    _producer_source_sha256 as _prepare_data_source_sha256,
    add_self_hash,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from formal_provenance import (  # noqa: E402
    FormalProvenanceError,
    is_sha256,
    materialize_verified_signoff,
    verify_phase4_signoff,
)
from ric.schema import JoinIdentity  # noqa: E402
from ric.wire import (  # noqa: E402
    HEADER_BYTES,
    RECORD_BYTES,
    ContractCacheEntry,
    ContractMessage,
    ContractRecord,
    ContractTax,
    IdentityTable,
    SenderContractCache,
    apply_wire_contract,
    decode_contract,
    encode_contract,
    encoded_contract_bytes,
    join_identity_hash_parts,
)


REPO_ROOT = HERE.parents[4]
IDEA_ROOT = HERE.parents[1]
DEFAULT_CONFIG = IDEA_ROOT / "configs" / "ric_v1.json"
DEFAULT_PROTOCOL = IDEA_ROOT / "RIC_Phase2_冻结实验协议_2026-07-22.md"
FORMAL_ROWS = (1, 2, 4, 8, 16, 32, 64)
FORMAL_RECORD_COUNTS = tuple(range(1, 256))
FORMAL_SELECTED_EXPERTS = 4
FORMAL_WARMUPS = 10
FORMAL_TRIALS = 30


class ServiceLutError(RuntimeError):
    """A service-LUT measurement/provenance invariant failed."""


def _producer_source_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        HERE / "capture_routes_gpu.py",
        HERE / "measure_capability_gpu.py",
        HERE / "prepare_data.py",
        HERE / "formal_provenance.py",
        HERE / "capability_contract.py",
        HERE / "schema.py",
        HERE / "wire.py",
    ):
        digest.update(str(path.resolve().relative_to(REPO_ROOT.resolve())).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _require_formal_signoff(
    path: Path | None,
    *,
    protocol_sha256: str,
    config_sha256: str,
    source_sha256: str,
    data_manifest_sha256: str,
    data_producer_signoff_sha256: str,
    model_key: str,
    model_tree_manifest_sha256: str,
) -> Mapping[str, Any]:
    try:
        return verify_phase4_signoff(
            path,
            repo_root=REPO_ROOT,
            expected_fields={
                "stage": "measure_service_lut",
                "protocol_sha256": protocol_sha256,
                "config_sha256": config_sha256,
                "measure_service_lut_source_sha256": source_sha256,
                "measure_capability_source_sha256": _capability_source_sha256(),
                "capture_routes_source_sha256": _capture_routes_source_sha256(),
                "prepare_data_source_sha256": _prepare_data_source_sha256(),
                "data_manifest_sha256": data_manifest_sha256,
                "data_producer_signoff_sha256": data_producer_signoff_sha256,
                "model_key": model_key,
                "model_tree_manifest_sha256": model_tree_manifest_sha256,
            },
            required_source_paths=(
                Path(__file__),
                HERE / "capture_routes_gpu.py",
                HERE / "measure_capability_gpu.py",
                HERE / "prepare_data.py",
                HERE / "formal_provenance.py",
                HERE / "capability_contract.py",
                HERE / "schema.py",
                HERE / "wire.py",
            ),
        )
    except (FormalProvenanceError, DataPreparationError) as exc:
        raise ServiceLutError(str(exc)) from exc


def outcome_blind_experts(
    *,
    num_experts: int,
    count: int,
    selection_seed: int,
    model_revision: str,
    layer_id: int,
) -> list[int]:
    if count > num_experts:
        raise ServiceLutError("selected expert count exceeds model experts")
    return sorted(
        sorted(
            range(num_experts),
            key=lambda expert_id: sha256_bytes(
                f"{selection_seed}:{model_revision}:{layer_id}:{expert_id}".encode()
            ),
        )[:count]
    )


def _cuda_samples(operation: Callable[[], Any], *, warmups: int, trials: int) -> list[float]:
    import torch

    for _ in range(warmups):
        operation()
    torch.cuda.synchronize()
    result = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        result.append(float(start.elapsed_time(end) * 1000.0))
    return result


def _host_samples(operation: Callable[[], Any], *, warmups: int, trials: int) -> list[float]:
    sink = None
    for _ in range(warmups):
        sink = operation()
    result = []
    for _ in range(trials):
        start = time.perf_counter_ns()
        sink = operation()
        elapsed = time.perf_counter_ns() - start
        result.append(float(elapsed / 1000.0))
    if sink is None:
        raise ServiceLutError("host timing operation returned no sink")
    return result


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ServiceLutError("empty percentile")
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ServiceLutError("refusing to write empty LUT")
    fields = list(rows[0])
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if list(row) != fields:
                raise ServiceLutError("LUT row schema drift")
            writer.writerow(row)


def _summarize(raw_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = {}
    metadata: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    fields = (
        "component",
        "model_key",
        "layer_id",
        "expert_id",
        "rows",
        "record_count",
        "source",
        "payload_dtype",
        "payload_elements_per_row",
        "payload_element_size_bytes",
        "payload_bytes_per_contribution_row",
        "payload_layout_sha256",
        "contract_message_bytes",
        "configured_delay_us",
    )
    for row in raw_rows:
        key = tuple(row[field] for field in fields)
        grouped.setdefault(key, []).append(float(row["us"]))
        metadata[key] = row
    result = []
    for key in sorted(grouped, key=lambda item: tuple(str(value) for value in item)):
        values = grouped[key]
        result.append(
            {
                **{field: value for field, value in zip(fields, key)},
                "trial_count": len(values),
                "median_us": float(statistics.median(values)),
                "p95_us": _percentile(values, 0.95),
                "max_us": max(values),
            }
        )
    return result


def payload_descriptor(weighted_contributions: Any) -> dict[str, Any]:
    """Describe one contribution row from the real weighted output tensor."""

    shape = tuple(int(value) for value in weighted_contributions.shape)
    if len(shape) != 3 or shape[-1] <= 0:
        raise ServiceLutError("weighted contributions must be [token, topk_slot, elements]")
    element_size = int(weighted_contributions.element_size())
    if element_size <= 0:
        raise ServiceLutError("payload element size must be positive")
    descriptor = {
        "payload_dtype": str(weighted_contributions.dtype).removeprefix("torch."),
        "payload_elements_per_row": shape[-1],
        "payload_element_size_bytes": element_size,
        "payload_bytes_per_contribution_row": shape[-1] * element_size,
    }
    descriptor["payload_layout_sha256"] = sha256_bytes(canonical_json_bytes(descriptor))
    return descriptor


def analytic_network_transfer_us(message_bytes: int, link_gbps: float) -> float:
    """Serialize actual canonical message bytes on an analytic link."""

    if type(message_bytes) is not int or message_bytes <= 0:
        raise ServiceLutError("analytic transfer requires positive integer bytes")
    if not isinstance(link_gbps, (int, float)) or float(link_gbps) <= 0:
        raise ServiceLutError("analytic transfer requires positive link Gbps")
    return float(message_bytes * 8.0 / (float(link_gbps) * 1000.0))


def _contract_host_operations(config: Mapping[str, Any], record_count: int) -> Mapping[str, Callable[[], Any]]:
    """Build timed operations through the only canonical RIC wire API."""

    contract = config["contract"]
    if (
        int(contract["header_bytes"]) != HEADER_BYTES
        or int(contract["record_bytes"]) != RECORD_BYTES
    ):
        raise ServiceLutError("config byte accounting differs from canonical wire module")
    joins = tuple(
        JoinIdentity(
            request_id=f"lut-request-{index}",
            forward_id=f"lut-request-{index}:forward:0",
            batch_id="lut-batch-0",
            phase="prefill",
            decode_step=0,
            layer_id=1,
            token_id=f"lut-token-{index}",
            token_block_id=f"lut-block-{index}",
            receiver_rank=1,
            epoch=1,
        )
        for index in range(record_count)
    )
    identity_table = IdentityTable.from_joins(joins)
    frozen_hash_parts = tuple(join_identity_hash_parts(join) for join in joins)

    def state_build_contract_records() -> tuple[ContractRecord, ...]:
        return tuple(
            ContractRecord(
                join_key_hash64=hash64,
                layer_id=join.layer_id,
                missing_slot_mask=0x0001,
                identity_tag16=tag16,
                slack_bucket=3,
                flags=0,
            )
            for join, (hash64, tag16) in zip(joins, frozen_hash_parts)
        )
    records = state_build_contract_records()
    message = ContractMessage(
        sender_rank=0,
        receiver_rank=1,
        epoch=1,
        sequence=1,
        records=records,
    )
    encoded = encode_contract(message)
    entries = tuple(
        ContractCacheEntry(
            join_identity=join,
            receiver_rank=message.receiver_rank,
            epoch=message.epoch,
            sequence=message.sequence,
            missing_slot_mask=record.missing_slot_mask,
            slack_bucket=record.slack_bucket,
            flags=record.flags,
        )
        for join, record in zip(joins, records)
    )
    policy_cache = SenderContractCache(sender_rank=0)
    if policy_cache._sequence_fault(message) is not None:
        raise ServiceLutError("canonical policy-cache fixture has illegal epoch/sequence")
    policy_cache._commit(message, entries)
    policy_snapshot = policy_cache.snapshot()

    def hash_identity() -> tuple[tuple[int, int], ...]:
        return tuple(join_identity_hash_parts(join) for join in joins)

    def encode() -> bytes:
        return encode_contract(message)

    def decode() -> ContractMessage:
        return decode_contract(encoded)

    def collision_checked_identity_lookup() -> tuple[JoinIdentity, ...]:
        return tuple(
            identity_table.resolve(
                record,
                receiver_rank=message.receiver_rank,
                epoch=message.epoch,
            )
            for record in records
        )

    def apply() -> Any:
        # A fresh cache makes sequence=1 legal in every timed trial.  Reusing
        # the cache would silently time duplicate-sequence fallback instead.
        result = apply_wire_contract(
            encoded,
            cache=SenderContractCache(sender_rank=0),
            identity_table=identity_table,
            expected_sender_rank=0,
            tax=ContractTax(),
        )
        if not result.applied or result.fallback or len(result.entries) != record_count:
            raise ServiceLutError(f"canonical contract apply failed: {result.fault}")
        return result

    def epoch_sequence_apply() -> tuple[int | None, int | None]:
        cache = SenderContractCache(sender_rank=0)
        fault = cache._sequence_fault(message)
        if fault is not None:
            raise ServiceLutError(f"canonical epoch/sequence fixture failed: {fault}")
        cache._commit(message, entries)
        return (
            cache.current_epoch(message.receiver_rank),
            cache.last_sequence(message.receiver_rank, message.epoch),
        )

    def sender_policy_cache_lookup() -> tuple[tuple[int, int, bool], ...]:
        return tuple(
            (
                policy_snapshot[join].missing_slot_mask,
                policy_snapshot[join].slack_bucket,
                policy_snapshot[join].is_last_sibling,
            )
            for join in joins
        )

    def empty_harness() -> int:
        # ``_host_samples`` already wraps every operation in the same pair of
        # perf-counter calls.  Calling the clock again here would make the
        # harness strictly more expensive than a true empty operation and can
        # manufacture negative adjusted tax for small record counts.
        return 0

    return {
        "state_build_contract_record": state_build_contract_records,
        "host_hash_identity": hash_identity,
        "host_encode_contract": encode,
        "host_decode_contract": decode,
        "collision_checked_identity_lookup": collision_checked_identity_lookup,
        "host_apply_wire_contract": apply,
        "epoch_sequence_apply": epoch_sequence_apply,
        "sender_policy_cache_lookup": sender_policy_cache_lookup,
        "host_empty_harness": empty_harness,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=("olmoe", "llmjp"), required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("dev", "formal"), default="dev")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--signoff", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--warmups", type=int, default=FORMAL_WARMUPS)
    parser.add_argument("--trials", type=int, default=FORMAL_TRIALS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment capability
        raise ServiceLutError("CUDA PyTorch is required") from exc
    if not torch.cuda.is_available():
        raise ServiceLutError("CUDA is required for service LUT")
    if args.output_dir.exists():
        raise ServiceLutError("refusing to overwrite service-LUT output directory")
    try:
        compute_apps_before = _gpu_compute_apps()
    except RouteCaptureError as exc:
        raise ServiceLutError(str(exc)) from exc
    if args.mode == "formal" and (
        args.warmups != FORMAL_WARMUPS or args.trials != FORMAL_TRIALS
    ):
        raise ServiceLutError("formal LUT warmups/trials are source-frozen")
    config = _load_config(args.config)
    contract_cfg = config["contract"]
    if (
        int(contract_cfg.get("max_records_per_message", -1)) != 255
        or contract_cfg.get("record_count_tax_measurement_grid")
        != "all_integers_1_to_255"
        or contract_cfg.get("record_count_tax_lookup")
        != "exact_only_no_interpolation_or_extrapolation"
        or tuple(FORMAL_RECORD_COUNTS) != tuple(range(1, 256))
    ):
        raise ServiceLutError("contract tax grid differs from frozen exact 1..255 surface")
    protocol_sha = sha256_file(args.protocol)
    config_sha = sha256_file(args.config)
    manifest = _load_data_manifest(
        args.data_manifest,
        mode=args.mode,
        model_key=args.model_key,
        config=config,
        protocol_sha256=protocol_sha,
        config_sha256=config_sha,
    )
    if manifest.get("role") != "calibration":
        raise ServiceLutError("service LUT consumes calibration only")
    source_sha = _producer_source_sha256()
    manifest_sha = str(manifest["manifest_sha256"])
    data_producer_signoff_sha = manifest.get("signoff_sha256")
    spec = config["models"][args.model_key]
    if args.mode == "formal" and args.model_path is None:
        raise ServiceLutError("formal service LUT requires an explicit hashed --model-path")
    model_reference = model_load_reference(spec, args.model_path)
    model_tree_sha = model_reference[2].get("tree_manifest_sha256")
    signoff = None
    if args.mode == "formal":
        if not is_sha256(data_producer_signoff_sha):
            raise ServiceLutError("formal data producer signoff hash is missing")
        if not isinstance(model_tree_sha, str):
            raise ServiceLutError("formal local model tree has no manifest hash")
        signoff = _require_formal_signoff(
            args.signoff,
            protocol_sha256=protocol_sha,
            config_sha256=config_sha,
            source_sha256=source_sha,
            data_manifest_sha256=manifest_sha,
            data_producer_signoff_sha256=str(data_producer_signoff_sha),
            model_key=args.model_key,
            model_tree_manifest_sha256=model_tree_sha,
        )

    model_revision = f"{spec['repo_id']}@{spec['revision']}"
    model, tokenizer, transformers_version, model_source = _load_model_and_tokenizer(
        spec,
        cache_dir=args.cache_dir,
        allow_download=args.allow_download,
        model_path=args.model_path,
        model_reference=model_reference,
    )
    modules = discover_moe_modules(model)
    layer_ids = [row[0] for row in modules]
    frozen_layers = selected_layers(
        layer_ids,
        selection_seed=int(config["data"]["selection_seed"]),
        model_revision=model_revision,
        count=int(config["route_capture"]["selected_layer_count_per_model"]),
    )
    target_layer = frozen_layers[0]
    module_by_layer = {layer: module for layer, _name, module in modules}
    moe = module_by_layer[target_layer]
    selected_expert_ids = outcome_blind_experts(
        num_experts=int(spec["num_experts"]),
        count=FORMAL_SELECTED_EXPERTS,
        selection_seed=int(config["data"]["selection_seed"]),
        model_revision=model_revision,
        layer_id=target_layer,
    )
    pools: dict[int, list[Any]] = {expert_id: [] for expert_id in selected_expert_ids}
    pool_rows = {expert_id: 0 for expert_id in selected_expert_ids}
    route_specific_pool: dict[tuple[int, int], Any] = {}
    captured: dict[int, dict[str, Any]] = {}
    first_weighted = None

    def make_input_hook(layer_id: int):
        def input_hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            state = captured.setdefault(layer_id, {})
            if "hidden" in state:
                raise ServiceLutError(f"selected MoE layer {layer_id} called twice")
            state["hidden"] = _first_tensor(inputs).detach()

        return input_hook

    def make_gate_hook(layer_id: int):
        def gate_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            state = captured.setdefault(layer_id, {})
            if "router_logits" in state:
                raise ServiceLutError(f"selected gate layer {layer_id} called twice")
            state["router_logits"] = _first_tensor(output).detach()

        return gate_hook

    handles = []
    for layer_id in frozen_layers:
        selected_moe = module_by_layer[layer_id]
        handles.append(
            selected_moe.register_forward_pre_hook(make_input_hook(layer_id))
        )
        handles.append(
            selected_moe.gate.register_forward_hook(make_gate_hook(layer_id))
        )
    try:
        for request in manifest["requests"]:
            captured.clear()
            text = str(request["text"])
            if sha256_bytes(text.encode()) != request["text_sha256"]:
                raise ServiceLutError("calibration request text hash mismatch")
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            if tuple(encoded["input_ids"].shape) != (1, 128):
                raise ServiceLutError("LUT request is not exactly 128 tokens")
            with torch.inference_mode():
                model(
                    input_ids=encoded["input_ids"].to("cuda:0"),
                    use_cache=False,
                    output_router_logits=True,
                    return_dict=True,
                )
            torch.cuda.synchronize()
            if set(captured) != set(frozen_layers) or any(
                set(captured[layer]) != {"hidden", "router_logits"}
                for layer in frozen_layers
            ):
                raise ServiceLutError("selected-layer LUT capture is incomplete")
            for layer_id in frozen_layers:
                selected_moe = module_by_layer[layer_id]
                state = captured[layer_id]
                hidden = state["hidden"].reshape(-1, state["hidden"].shape[-1])
                experts, _weights = _routes_from_logits(
                    state["router_logits"],
                    top_k=int(spec["top_k"]),
                    normalize_topk=_normalizes_topk(selected_moe),
                    selection_rule=NATIVE_TOPK_SELECTION_RULE,
                    output_dtype=hidden.dtype,
                )
                if layer_id == target_layer and first_weighted is None:
                    first_weighted, _selected, _route_weights = (
                        _reconstruct_real_contributions(
                            moe=selected_moe,
                            hidden_states=state["hidden"],
                            router_logits=state["router_logits"],
                            top_k=int(spec["top_k"]),
                        )
                    )
                    first_weighted = first_weighted.detach()
                for expert_id in range(int(spec["num_experts"])):
                    token_idx = torch.where((experts == expert_id).any(dim=1))[0]
                    if token_idx.numel() == 0:
                        continue
                    key = (layer_id, expert_id)
                    if key not in route_specific_pool:
                        route_specific_pool[key] = hidden[token_idx[:1]].detach().cpu()
                    if (
                        layer_id == target_layer
                        and expert_id in pools
                        and pool_rows[expert_id] < max(FORMAL_ROWS)
                    ):
                        values = hidden[token_idx].detach().cpu()
                        pools[expert_id].append(values)
                        pool_rows[expert_id] += int(values.shape[0])
    finally:
        for handle in handles:
            handle.remove()
    if first_weighted is None:
        raise ServiceLutError("no real weighted contributions captured")
    if any(value < max(FORMAL_ROWS) for value in pool_rows.values()):
        raise ServiceLutError(f"insufficient own-routed expert rows: {pool_rows}")
    expected_route_specific_keys = {
        (layer_id, expert_id)
        for layer_id in frozen_layers
        for expert_id in range(int(spec["num_experts"]))
    }
    missing_route_specific = sorted(
        expected_route_specific_keys - set(route_specific_pool)
    )
    if missing_route_specific:
        raise ServiceLutError(
            "BLOCKED_ROUTE_SPECIFIC_SERVICE_COVERAGE: missing frozen calibration "
            f"(layer,expert) keys {missing_route_specific[:16]}"
        )
    payload_layout = payload_descriptor(first_weighted)

    raw_rows: list[dict[str, Any]] = []

    def append_samples(
        component: str,
        samples: Sequence[float],
        *,
        layer_id: int | None = None,
        expert_id: int = -1,
        rows: int = 0,
        record_count: int = 0,
        contract_message_bytes: int = 0,
        source: str,
    ) -> None:
        for trial, elapsed in enumerate(samples):
            raw_rows.append(
                {
                    "component": component,
                    "model_key": args.model_key,
                    "layer_id": target_layer if layer_id is None else int(layer_id),
                    "expert_id": expert_id,
                    "rows": rows,
                    "record_count": record_count,
                    "trial": trial,
                    "us": elapsed,
                    "source": source,
                    **payload_layout,
                    "contract_message_bytes": contract_message_bytes,
                    "configured_delay_us": 0.0,
                }
            )

    with torch.inference_mode():
        for layer_id in frozen_layers:
            selected_moe = module_by_layer[layer_id]
            for expert_id in range(int(spec["num_experts"])):
                activation = route_specific_pool[(layer_id, expert_id)].to(
                    device="cuda:0"
                )
                expert = selected_moe.experts[expert_id]
                append_samples(
                    "expert_execution_route_specific_row1",
                    _cuda_samples(
                        lambda expert=expert, activation=activation: expert(activation),
                        warmups=args.warmups,
                        trials=args.trials,
                    ),
                    layer_id=layer_id,
                    expert_id=expert_id,
                    rows=1,
                    source="measured_5090_cuda",
                )
        for expert_id in selected_expert_ids:
            own_pool = torch.cat(pools[expert_id], dim=0)
            expert = moe.experts[expert_id]
            for rows in FORMAL_ROWS:
                activation = own_pool[:rows].to(device="cuda:0")
                append_samples(
                    "expert_execution",
                    _cuda_samples(
                        lambda expert=expert, activation=activation: expert(activation),
                        warmups=args.warmups,
                        trials=args.trials,
                    ),
                    expert_id=expert_id,
                    rows=rows,
                    source="measured_5090_cuda",
                )
        reference_expert = moe.experts[selected_expert_ids[0]]
        reference_pool = torch.cat(pools[selected_expert_ids[0]], dim=0)
        for rows in FORMAL_ROWS:
            activation = reference_pool[:rows].to(device="cuda:0")
            output = _first_tensor(reference_expert(activation)).detach()
            packed_output = _sender_pack(output).detach()
            append_samples(
                "sender_pack",
                _cuda_samples(
                    lambda output=output: _sender_pack(output),
                    warmups=args.warmups,
                    trials=args.trials,
                ),
                rows=rows,
                source="measured_5090_cuda",
            )
            append_samples(
                "receiver_unpack",
                _cuda_samples(
                    lambda packed_output=packed_output: _receiver_unpack(packed_output),
                    warmups=args.warmups,
                    trials=args.trials,
                ),
                rows=rows,
                source="measured_5090_cuda",
            )
            host = output.cpu().pin_memory()
            staging = torch.empty_like(output)
            append_samples(
                "host_to_device_staging_not_rdma",
                _cuda_samples(
                    lambda host=host, staging=staging: staging.copy_(host, non_blocking=True),
                    warmups=args.warmups,
                    trials=args.trials,
                ),
                rows=rows,
                source="measured_5090_h2d_not_rdma",
            )
            siblings = first_weighted[:rows]
            append_samples(
                "canonical_reduction",
                _cuda_samples(
                    lambda siblings=siblings: canonical_reduce(siblings),
                    warmups=args.warmups,
                    trials=args.trials,
                ),
                rows=rows,
                source="measured_5090_cuda",
            )

    for record_count in FORMAL_RECORD_COUNTS:
        for component, operation in _contract_host_operations(config, record_count).items():
            append_samples(
                component,
                _host_samples(operation, warmups=args.warmups, trials=args.trials),
                record_count=record_count,
                contract_message_bytes=encoded_contract_bytes(record_count),
                source="measured_5090_host",
            )
    summary_rows = _summarize(raw_rows)
    primary_link_gbps = float(config["topology_proxy"]["primary_link_gbps"])
    delay_sensitivity_us = tuple(float(value) for value in config["contract"]["delay_sensitivity_us"])

    def append_derived_control_summary(
        component: str,
        *,
        record_count: int,
        source: str,
        elapsed_us: float,
        message_bytes: int,
        configured_delay_us: float,
    ) -> None:
        summary_rows.append(
            {
                "component": component,
                "model_key": args.model_key,
                "layer_id": target_layer,
                "expert_id": -1,
                "rows": 0,
                "record_count": record_count,
                "source": source,
                **payload_layout,
                "contract_message_bytes": message_bytes,
                "configured_delay_us": configured_delay_us,
                "trial_count": 1,
                "median_us": elapsed_us,
                "p95_us": elapsed_us,
                "max_us": elapsed_us,
            }
        )

    for record_count in FORMAL_RECORD_COUNTS:
        message_bytes = encoded_contract_bytes(record_count)
        append_derived_control_summary(
            "contract_transfer_analytic_primary_link",
            record_count=record_count,
            source="analytic_network",
            elapsed_us=analytic_network_transfer_us(message_bytes, primary_link_gbps),
            message_bytes=message_bytes,
            configured_delay_us=0.0,
        )
    for configured_delay_us in delay_sensitivity_us:
        append_derived_control_summary(
            "configured_contract_delay",
            record_count=0,
            source="synthetic_delay",
            elapsed_us=configured_delay_us,
            message_bytes=0,
            configured_delay_us=configured_delay_us,
        )

    # A conservative replay row uses the slowest selected expert median for
    # each row count; raw per-expert measurements remain available.
    for rows in FORMAL_ROWS:
        expert_summaries = [
            row
            for row in summary_rows
            if row["component"] == "expert_execution" and row["rows"] == rows
        ]
        conservative = max(expert_summaries, key=lambda row: float(row["median_us"]))
        summary_rows.append(
            {
                **conservative,
                "component": "expert_execution_conservative_max_selected_median",
                "expert_id": -1,
            }
        )

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{args.output_dir.name}.partial-", dir=args.output_dir.parent
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        raw_path = temporary / "service_lut_raw.csv"
        summary_path = temporary / "service_lut.csv"
        _write_csv(raw_path, raw_rows)
        _write_csv(summary_path, summary_rows)
        embedded_signoff_sha256 = (
            materialize_verified_signoff(args.signoff, temporary)
            if signoff is not None
            else None
        )
        metadata = add_self_hash(
            {
                "schema_version": "ric-service-lut-v1",
                "status": "LUT_ONLY" if args.mode == "formal" else "NOT_TESTED",
                "scientific_result": False,
                "evidence_boundary": (
                    "REAL_5090_EXPERT_PACK_HOST_CONTROL_H2D_NOT_RDMA / NO_NETWORK_P99"
                ),
                "mode": args.mode,
                "model_key": args.model_key,
                "model_revision": model_revision,
                "model_source": model_source,
                "model_tree_manifest_sha256": model_tree_sha,
                "transformers_version": transformers_version,
                "target_layer": target_layer,
                "route_specific_selected_layers": frozen_layers,
                "route_specific_expert_ids": list(range(int(spec["num_experts"]))),
                "route_specific_key_count": len(route_specific_pool),
                "route_specific_main_component": (
                    "expert_execution_route_specific_row1"
                ),
                "selected_experts": selected_expert_ids,
                "own_routed_rows": pool_rows,
                "batching_diagnostic_component": "expert_execution",
                "row_grid": list(FORMAL_ROWS),
                "record_count_grid": list(FORMAL_RECORD_COUNTS),
                "warmups": args.warmups,
                "trials": args.trials,
                **payload_layout,
                "contract_network_accounting": {
                    "source": "analytic_network",
                    "primary_link_gbps": primary_link_gbps,
                    "formula": "message_bytes * 8 / (link_gbps * 1000) us",
                    "message_bytes_by_record_count": {
                        str(count): encoded_contract_bytes(count)
                        for count in FORMAL_RECORD_COUNTS
                    },
                    "transfer_us_by_record_count": {
                        str(count): analytic_network_transfer_us(
                            encoded_contract_bytes(count), primary_link_gbps
                        )
                        for count in FORMAL_RECORD_COUNTS
                    },
                    "h2d_is_transfer_tax": False,
                },
                "configured_delay_accounting": {
                    "source": "synthetic_delay",
                    "values_us": list(delay_sensitivity_us),
                },
                "host_measurement_accounting": {
                    "raw_saved_before_harness_subtraction": True,
                    "harness_component": "host_empty_harness",
                    "negative_after_subtraction_clamped": False,
                    "additive_components": [
                        "state_build_contract_record",
                        "host_hash_identity",
                        "host_encode_contract",
                        "host_decode_contract",
                        "collision_checked_identity_lookup",
                        "epoch_sequence_apply",
                        "sender_policy_cache_lookup",
                    ],
                    "end_to_end_diagnostic_not_additive": "host_apply_wire_contract",
                },
                "data_path_measurement_accounting": {
                    "per_contribution_additive_components": [
                        "expert_execution_route_specific_row1",
                        "sender_pack",
                        "receiver_unpack",
                    ],
                    "per_join_once_only_component": "canonical_reduction",
                    "canonical_reduction_charged_once_per_join": True,
                },
                "protocol_sha256": protocol_sha,
                "config_sha256": config_sha,
                "data_manifest_sha256": manifest_sha,
                "data_producer_signoff_sha256": data_producer_signoff_sha,
                "measure_service_lut_source_sha256": source_sha,
                "service_lut_raw_sha256": sha256_file(raw_path),
                "service_lut_sha256": sha256_file(summary_path),
                "gpu_environment": _gpu_environment(
                    compute_apps_before=compute_apps_before
                ),
                "signoff_sha256": embedded_signoff_sha256,
            }
        )
        (temporary / "service_lut_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.rename(args.output_dir)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
