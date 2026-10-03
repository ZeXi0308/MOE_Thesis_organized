"""GPU capture, pure call planning, and numeric execution for SemanticFence.

The module keeps Torch and Transformers behind function boundaries.  In
particular, row identities and all four call planners can be imported and
tested on a host that has no Torch installation.  This file deliberately does
not provide a CLI, artifact lock, cuBLASLt tracing, or final verdict logic;
those belong to the parent runner.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


EXPERIMENT_DIR = Path(__file__).resolve().parent
CONTRACT_PATH = EXPERIMENT_DIR / "executor_contract.py"
CAPTURE_SCHEMA = "semanticfence-olmoe-capture-v1"
CALL_PLAN_SCHEMA = "semanticfence-call-plan-v1"
ARM_A = "A_isolated_m1"
ARM_B = "B_native_unrestricted"
ARM_C = "C_fixed_m64"
ARM_D = "D_semanticfence"
CALIBRATION_ARM = "calibration"
DEFAULT_M_VALUES = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_FIXED_M = 64
DEFAULT_WARMUPS = 3
DEFAULT_REPEATS = 10
FROZEN_ARM_ORDER = (ARM_A, ARM_B, ARM_C, ARM_D)


class GPUExecutionError(RuntimeError):
    """Capture, planning, or GPU execution is not interpretable."""


def _load_contract() -> Any:
    name = "semanticfence_executor_contract"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, CONTRACT_PATH)
    if spec is None or spec.loader is None:
        raise GPUExecutionError(f"cannot import {CONTRACT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CONTRACT = _load_contract()


def _torch() -> Any:
    """Import Torch only for tensor capture/materialization/execution."""

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - GPU host concern
        raise GPUExecutionError("Torch is required only for GPU execution") from exc
    return torch


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_text_sha256(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _tensor_storage_bytes(tensor: Any) -> bytes:
    """Return exact tensor storage bytes without converting BF16 numerically."""

    torch = _torch()
    value = tensor.detach().contiguous().view(torch.uint8).cpu()
    return value.numpy().tobytes()


def tensor_storage_sha256(tensor: Any) -> str:
    return hashlib.sha256(_tensor_storage_bytes(tensor)).hexdigest()


def bf16_storage_bytes(tensor: Any) -> bytes:
    torch = _torch()
    if tensor.dtype != torch.bfloat16:
        raise GPUExecutionError(f"expected BF16 tensor, got {tensor.dtype}")
    return _tensor_storage_bytes(tensor)


def strict_bf16_mismatch_count(left: bytes, right: bytes) -> int:
    """Count raw uint16 differences, including ``+0`` versus ``-0``."""

    if len(left) != len(right) or len(left) % 2:
        raise GPUExecutionError("raw BF16 references have incompatible sizes")
    return sum(
        left[index : index + 2] != right[index : index + 2]
        for index in range(0, len(left), 2)
    )


def positions_for_split(
    split: str, *, window_tokens: int = 16, evaluation_position: int = 15
) -> tuple[int, ...]:
    """Freeze calibration to all positions and evaluation to position 15."""

    if int(window_tokens) != 16 or int(evaluation_position) != 15:
        raise GPUExecutionError("SemanticFence capture positions are frozen at 16/15")
    if split == "calibration":
        return tuple(range(16))
    if split in {"evaluation", "semanticfence_eval_fresh"}:
        return (15,)
    raise GPUExecutionError(f"unsupported capture split: {split!r}")


@dataclass(frozen=True, slots=True)
class CapturedWindow:
    """One full 16-token OLMoE capture plus its native route ledger."""

    schema_version: str
    split: str
    document_sha256: str
    document_index: int
    offset: int
    window_token_ids: tuple[int, ...]
    selected_positions: tuple[int, ...]
    full_hidden_states: Any
    selected_experts: Any
    routing_weights: Any
    router_logits_sha256_by_layer: tuple[str, ...]

    @property
    def window_id(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": self.schema_version,
                "split": self.split,
                "document_sha256": self.document_sha256,
                "document_index": int(self.document_index),
                "offset": int(self.offset),
                "window_token_ids": list(self.window_token_ids),
            }
        )


@dataclass(frozen=True, slots=True)
class RowContext:
    """Non-tensor context needed to audit one routed expert row."""

    window_id: str
    absolute_token_position: int
    window_token_id: int
    routing_weight: float


@dataclass(frozen=True, slots=True)
class MaterializedRow:
    """A canonical RowRecord joined to its exact BF16 hidden row and context."""

    record: Any
    tensor: Any
    context: RowContext

    def __post_init__(self) -> None:
        if not isinstance(self.record, CONTRACT.RowRecord):
            raise GPUExecutionError("materialized row requires a RowRecord")

    @property
    def row_id(self) -> str:
        return self.record.row_id


def capture_olmoe_split(
    *,
    model: Any,
    tokenizer: Any,
    documents: Sequence[Mapping[str, Any]],
    split: str,
    token_offsets: Sequence[int] = (0, 256),
    window_tokens: int = 16,
    add_special_tokens: bool = False,
    evaluation_position: int = 15,
    device: str = "cuda",
) -> tuple[CapturedWindow, ...]:
    """Capture full OLMoE inputs and an exact native top-k route ledger.

    Every layer's native ``router_logits`` is compared bit-for-bit with a
    same-shape replay of that layer's gate over the captured MLP input.  The
    capture fails instead of materializing rows if even one layer disagrees.
    """

    torch = _torch()
    import torch.nn.functional as F

    positions = positions_for_split(
        split,
        window_tokens=int(window_tokens),
        evaluation_position=int(evaluation_position),
    )
    if tuple(int(value) for value in token_offsets) != (0, 256):
        raise GPUExecutionError("token offsets are frozen at (0, 256)")
    try:
        layers = tuple(model.model.layers)
        blocks = tuple(layer.mlp for layer in layers)
    except AttributeError as exc:
        raise GPUExecutionError("model is not a supported OLMoE causal LM") from exc
    layer_count = int(getattr(model.config, "num_hidden_layers", -1))
    hidden_size = int(getattr(model.config, "hidden_size", -1))
    expert_count = int(getattr(model.config, "num_experts", -1))
    top_k = int(getattr(model.config, "num_experts_per_tok", -1))
    if layer_count != len(blocks) or min(hidden_size, expert_count, top_k) <= 0:
        raise GPUExecutionError("OLMoE structure metadata is incomplete")

    captured_by_layer: dict[int, Any] = {}
    handles: list[Any] = []

    def make_hook(layer_index: int) -> Any:
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            if len(inputs) != 1:
                raise GPUExecutionError(
                    f"layer {layer_index} MLP hook expected one input"
                )
            states = inputs[0]
            expected = (1, int(window_tokens), hidden_size)
            if tuple(states.shape) != expected:
                raise GPUExecutionError(
                    f"layer {layer_index} hidden shape {tuple(states.shape)} != {expected}"
                )
            captured_by_layer[layer_index] = states.detach().clone()

        return hook

    for layer_index, block in enumerate(blocks):
        handles.append(block.register_forward_pre_hook(make_hook(layer_index)))

    results: list[CapturedWindow] = []
    try:
        ordered_documents = sorted(documents, key=lambda row: int(row["document_index"]))
        observed_indices = [int(row["document_index"]) for row in ordered_documents]
        if len(set(observed_indices)) != len(observed_indices):
            raise GPUExecutionError("capture documents contain duplicate indices")
        for document in ordered_documents:
            text = str(document["text"])
            text_digest = _canonical_text_sha256(text)
            if text_digest != document.get("text_sha256"):
                raise GPUExecutionError("capture document text hash mismatch")
            encoded = tokenizer(text, add_special_tokens=bool(add_special_tokens))
            token_ids = list(encoded["input_ids"])
            for raw_offset in token_offsets:
                offset = int(raw_offset)
                window = token_ids[offset : offset + int(window_tokens)]
                if len(window) != int(window_tokens):
                    raise GPUExecutionError(
                        f"document {document['document_index']} offset {offset} is short"
                    )
                captured_by_layer.clear()
                input_ids = torch.tensor(
                    [window], dtype=torch.long, device=torch.device(device)
                )
                with torch.inference_mode():
                    output = model(
                        input_ids=input_ids,
                        use_cache=False,
                        output_router_logits=True,
                        return_dict=True,
                    )
                if set(captured_by_layer) != set(range(layer_count)):
                    raise GPUExecutionError("full hidden-state capture is incomplete")
                router_logits = output.router_logits
                if router_logits is None or len(router_logits) != layer_count:
                    raise GPUExecutionError("native router-logit ledger is incomplete")

                hidden_gpu = tuple(captured_by_layer[index] for index in range(layer_count))
                selected_rows: list[Any] = []
                weight_rows: list[Any] = []
                router_hashes: list[str] = []
                for layer_index, native_value in enumerate(router_logits):
                    native = native_value.reshape(int(window_tokens), expert_count)
                    with torch.inference_mode():
                        replay = blocks[layer_index].gate(
                            hidden_gpu[layer_index].reshape(int(window_tokens), hidden_size)
                        )
                    if not torch.equal(native, replay):
                        raise GPUExecutionError(
                            f"layer {layer_index} native/replayed gate logits differ"
                        )
                    probabilities = F.softmax(native, dim=-1, dtype=torch.float)
                    weights, experts = torch.topk(
                        probabilities, k=top_k, dim=-1, sorted=True
                    )
                    if tuple(experts.shape) != (int(window_tokens), top_k):
                        raise GPUExecutionError("native top-k ledger shape mismatch")
                    selected_rows.append(experts.detach().cpu().clone())
                    weight_rows.append(weights.detach().cpu().clone())
                    router_hashes.append(tensor_storage_sha256(native))

                results.append(
                    CapturedWindow(
                        schema_version=CAPTURE_SCHEMA,
                        split="evaluation" if split == "semanticfence_eval_fresh" else split,
                        document_sha256=text_digest,
                        document_index=int(document["document_index"]),
                        offset=offset,
                        window_token_ids=tuple(int(value) for value in window),
                        selected_positions=positions,
                        full_hidden_states=torch.stack(
                            [value[0].detach().cpu() for value in hidden_gpu], dim=0
                        ).contiguous(),
                        selected_experts=torch.stack(selected_rows, dim=0).contiguous(),
                        routing_weights=torch.stack(weight_rows, dim=0).contiguous(),
                        router_logits_sha256_by_layer=tuple(router_hashes),
                    )
                )
    finally:
        for handle in handles:
            handle.remove()
    return tuple(results)


def materialize_routed_rows(
    captures: Iterable[CapturedWindow],
) -> tuple[MaterializedRow, ...]:
    """Create one RowRecord/tensor/context for every selected top-k pair."""

    torch = _torch()
    result: list[MaterializedRow] = []
    for capture in captures:
        if not isinstance(capture, CapturedWindow):
            raise GPUExecutionError("captures must contain CapturedWindow values")
        hidden = capture.full_hidden_states
        experts = capture.selected_experts
        weights = capture.routing_weights
        if hidden.ndim != 3 or experts.ndim != 3 or weights.shape != experts.shape:
            raise GPUExecutionError("captured hidden/route ledger shapes are invalid")
        layer_count, width, _hidden_size = map(int, hidden.shape)
        if int(experts.shape[0]) != layer_count or int(experts.shape[1]) != width:
            raise GPUExecutionError("capture layer/token dimensions disagree")
        if hidden.dtype != torch.bfloat16:
            raise GPUExecutionError("captured hidden states must be BF16")
        for token_position in capture.selected_positions:
            if token_position < 0 or token_position >= width:
                raise GPUExecutionError("selected token position is out of range")
            for layer in range(layer_count):
                hidden_row = hidden[layer, token_position].detach().contiguous().clone()
                hidden_sha256 = tensor_storage_sha256(hidden_row)
                expert_ids = [int(value) for value in experts[layer, token_position].tolist()]
                if len(set(expert_ids)) != len(expert_ids):
                    raise GPUExecutionError("native top-k ledger repeats an expert")
                for rank_index, expert_id in enumerate(expert_ids):
                    record = CONTRACT.RowRecord(
                        split=capture.split,
                        document_sha256=capture.document_sha256,
                        document_index=int(capture.document_index),
                        offset=int(capture.offset),
                        token_position=int(token_position),
                        layer=int(layer),
                        expert_id=int(expert_id),
                        route_rank=rank_index + 1,
                        hidden_sha256=hidden_sha256,
                    )
                    result.append(
                        MaterializedRow(
                            record=record,
                            tensor=hidden_row,
                            context=RowContext(
                                window_id=capture.window_id,
                                absolute_token_position=int(capture.offset)
                                + int(token_position),
                                window_token_id=int(
                                    capture.window_token_ids[token_position]
                                ),
                                routing_weight=float(
                                    weights[layer, token_position, rank_index].item()
                                ),
                            ),
                        )
                    )
    result.sort(key=lambda row: row.row_id)
    row_ids = [row.row_id for row in result]
    if len(set(row_ids)) != len(row_ids):
        raise GPUExecutionError("materialization produced duplicate row identities")
    return tuple(result)


def _records(rows: Iterable[Any]) -> tuple[Any, ...]:
    materialized = tuple(rows)
    result: list[Any] = []
    for value in materialized:
        row = value.record if isinstance(value, MaterializedRow) else value
        if not isinstance(row, CONTRACT.RowRecord):
            raise GPUExecutionError("planning rows must be RowRecord or MaterializedRow")
        result.append(row)
    row_ids = [row.row_id for row in result]
    if len(set(row_ids)) != len(row_ids):
        raise GPUExecutionError("planning input contains duplicate row identities")
    return tuple(result)


def _interleave_documents(rows: Sequence[Any]) -> tuple[Any, ...]:
    buckets: dict[str, deque[Any]] = defaultdict(deque)
    for row in sorted(rows, key=lambda value: value.row_id):
        buckets[row.document_sha256].append(row)
    ordered: list[Any] = []
    document_ids = sorted(buckets)
    while any(buckets[document_id] for document_id in document_ids):
        for document_id in document_ids:
            if buckets[document_id]:
                ordered.append(buckets[document_id].popleft())
    return tuple(ordered)


def build_calibration_packs(
    rows: Iterable[Any],
    *,
    m_values: Iterable[int] = DEFAULT_M_VALUES,
) -> tuple[Any, ...]:
    """Build deterministic, within-M disjoint packs for every layer/expert/M."""

    records = _records(rows)
    if any(row.split != "calibration" for row in records):
        raise GPUExecutionError("calibration packs may contain calibration rows only")
    normalized_ms = tuple(sorted({int(value) for value in m_values if int(value) > 1}))
    if not normalized_ms or any(value not in DEFAULT_M_VALUES for value in normalized_ms):
        raise GPUExecutionError("calibration M values must use the frozen M grid")
    grouped: dict[tuple[int, int], list[Any]] = defaultdict(list)
    for row in records:
        grouped[(row.layer, row.expert_id)].append(row)

    packs: list[Any] = []
    for (layer, expert_id), group in sorted(grouped.items()):
        interleaved = _interleave_documents(group)
        for m_value in normalized_ms:
            full_pack_count = len(interleaved) // m_value
            for pack_index in range(full_pack_count):
                start = pack_index * m_value
                pack_rows = tuple(interleaved[start : start + m_value])
                packs.append(
                    CONTRACT.Pack(
                        layer=layer, expert_id=expert_id, rows=pack_rows
                    )
                )
    packs.sort(key=lambda pack: (pack.layer, pack.expert_id, pack.m, pack.pack_id))
    return tuple(packs)


@dataclass(frozen=True, slots=True)
class PlannedCall:
    schema_version: str
    call_index: int
    arm: str
    layer: int
    expert_id: int
    rows: tuple[Any, ...]
    execution_m: int
    padding_rows: int
    expected_signatures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != CALL_PLAN_SCHEMA or int(self.call_index) < 0:
            raise GPUExecutionError("call plan schema/index is invalid")
        if not self.rows:
            raise GPUExecutionError("planned call must contain a real row")
        if int(self.padding_rows) < 0:
            raise GPUExecutionError("padding row count cannot be negative")
        if int(self.execution_m) != len(self.rows) + int(self.padding_rows):
            raise GPUExecutionError("execution M does not equal real plus padding rows")
        row_ids: list[str] = []
        for row in self.rows:
            if not isinstance(row, CONTRACT.RowRecord):
                raise GPUExecutionError("planned calls require RowRecord values")
            if row.layer != self.layer or row.expert_id != self.expert_id:
                raise GPUExecutionError("planned call crosses layer/expert boundary")
            row_ids.append(row.row_id)
        if len(set(row_ids)) != len(row_ids):
            raise GPUExecutionError("planned call duplicates a real row")

    @property
    def row_ids(self) -> tuple[str, ...]:
        return tuple(row.row_id for row in self.rows)

    @property
    def call_id(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": self.schema_version,
                "call_index": int(self.call_index),
                "arm": self.arm,
                "layer": int(self.layer),
                "expert_id": int(self.expert_id),
                "row_ids": list(self.row_ids),
                "execution_m": int(self.execution_m),
                "padding_rows": int(self.padding_rows),
                "expected_signatures": list(self.expected_signatures),
            }
        )


@dataclass(frozen=True, slots=True)
class ArmPlan:
    arm: str
    calls: tuple[PlannedCall, ...]

    @property
    def row_ids(self) -> tuple[str, ...]:
        return tuple(row_id for call in self.calls for row_id in call.row_ids)

    @property
    def padding_rows(self) -> int:
        return sum(call.padding_rows for call in self.calls)


def _group_records(records: Sequence[Any]) -> dict[tuple[int, int], tuple[Any, ...]]:
    grouped: dict[tuple[int, int], list[Any]] = defaultdict(list)
    for row in records:
        grouped[(row.layer, row.expert_id)].append(row)
    return {
        key: tuple(sorted(group, key=lambda row: row.row_id))
        for key, group in sorted(grouped.items())
    }


def _arm_plan(
    arm: str,
    specs: Iterable[tuple[int, int, tuple[Any, ...], int, int, tuple[str, ...]]],
    expected_rows: Sequence[Any],
) -> ArmPlan:
    calls = tuple(
        PlannedCall(
            schema_version=CALL_PLAN_SCHEMA,
            call_index=index,
            arm=arm,
            layer=layer,
            expert_id=expert_id,
            rows=rows,
            execution_m=execution_m,
            padding_rows=padding_rows,
            expected_signatures=signatures,
        )
        for index, (layer, expert_id, rows, execution_m, padding_rows, signatures) in enumerate(specs)
    )
    expected = {row.row_id for row in expected_rows}
    observed = [row_id for call in calls for row_id in call.row_ids]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise GPUExecutionError("arm plan does not cover every row exactly once")
    return ArmPlan(arm=arm, calls=calls)


def plan_arm_a(rows: Iterable[Any]) -> ArmPlan:
    records = _records(rows)
    # All four arms share the same (layer, expert, row-id) traversal.  Arm A
    # changes only the within-group pack size to M=1; otherwise its expert
    # weight/cache access order would be a latency confounder.
    specs = (
        (layer, expert_id, (row,), 1, 0, ())
        for (layer, expert_id), group in _group_records(records).items()
        for row in group
    )
    return _arm_plan(ARM_A, specs, records)


def plan_arm_b(rows: Iterable[Any], *, maximum_m: int = DEFAULT_FIXED_M) -> ArmPlan:
    records = _records(rows)
    grouped = _group_records(records)
    if any(len(group) > int(maximum_m) for group in grouped.values()):
        raise GPUExecutionError("natural group exceeds the frozen maximum M")
    specs = (
        (layer, expert_id, group, len(group), 0, ())
        for (layer, expert_id), group in grouped.items()
    )
    return _arm_plan(ARM_B, specs, records)


def plan_arm_c(rows: Iterable[Any], *, fixed_m: int = DEFAULT_FIXED_M) -> ArmPlan:
    records = _records(rows)
    if int(fixed_m) != DEFAULT_FIXED_M:
        raise GPUExecutionError("fixed control M is frozen at 64")
    grouped = _group_records(records)
    if any(len(group) > int(fixed_m) for group in grouped.values()):
        raise GPUExecutionError("real group is larger than fixed-M control")
    specs = (
        (
            layer,
            expert_id,
            group,
            int(fixed_m),
            int(fixed_m) - len(group),
            (),
        )
        for (layer, expert_id), group in grouped.items()
    )
    return _arm_plan(ARM_C, specs, records)


def plan_arm_d(
    rows: Iterable[Any],
    *,
    contract: Any,
    stack_digest: str,
) -> ArmPlan:
    """Use only authenticated allowed M values; unknowns split to M=1."""

    records = _records(rows)
    trusted = CONTRACT.validate_contract(contract)
    observed: dict[tuple[int, int, int], set[str]] = defaultdict(set)
    allowed: dict[tuple[int, int, int], set[str]] = defaultdict(set)
    if stack_digest == trusted.stack_digest:
        for entry in trusted.entries:
            key = (entry.layer, entry.expert_id, entry.m)
            observed[key].add(entry.signature)
            if entry.allowed:
                allowed[key].add(entry.signature)

    specs: list[tuple[int, int, tuple[Any, ...], int, int, tuple[str, ...]]] = []
    for (layer, expert_id), group in _group_records(records).items():
        # A deployable pre-call descriptor must predict one signature.  If
        # calibration produced multiple allowed signatures for the same
        # layer/expert/M, admission is ambiguous and therefore falls back.
        allowed_ms = {
            m_value
            for (entry_layer, entry_expert, m_value), signatures in allowed.items()
            if entry_layer == layer
            and entry_expert == expert_id
            and len(observed[(entry_layer, entry_expert, m_value)]) == 1
            and len(signatures) == 1
        }
        packs = CONTRACT.pack_distinct_rows(group, allowed_ms | {1})
        for pack in packs:
            signatures = tuple(sorted(allowed.get((layer, expert_id, pack.m), set())))
            if len(signatures) != 1:
                signatures = ()
            specs.append((layer, expert_id, pack.rows, pack.m, 0, signatures))
    return _arm_plan(ARM_D, specs, records)


def plan_four_arms(
    rows: Iterable[Any],
    *,
    contract: Any,
    stack_digest: str,
) -> Mapping[str, ArmPlan]:
    records = _records(rows)
    plans = {
        ARM_A: plan_arm_a(records),
        ARM_B: plan_arm_b(records),
        ARM_C: plan_arm_c(records),
        ARM_D: plan_arm_d(records, contract=contract, stack_digest=stack_digest),
    }
    expected = {row.row_id for row in records}
    expected_order = plans[ARM_A].row_ids
    for plan in plans.values():
        if set(plan.row_ids) != expected or len(plan.row_ids) != len(expected):
            raise GPUExecutionError(f"{plan.arm} row coverage is incomplete")
        if plan.row_ids != expected_order:
            raise GPUExecutionError(
                f"{plan.arm} changes the frozen logical row traversal order"
            )
    return plans


def frozen_arm_order(repeat_index: int) -> tuple[str, ...]:
    """Return the frozen Latin rotation used for one paired repeat."""

    if not isinstance(repeat_index, int) or repeat_index < 0:
        raise GPUExecutionError("repeat index must be a non-negative integer")
    offset = repeat_index % len(FROZEN_ARM_ORDER)
    return FROZEN_ARM_ORDER[offset:] + FROZEN_ARM_ORDER[:offset]


def calibration_call_plan(packs: Iterable[Any]) -> ArmPlan:
    trusted_packs = tuple(packs)
    records = tuple(row for pack in trusted_packs for row in pack.rows)
    specs = (
        (pack.layer, pack.expert_id, pack.rows, pack.m, 0, ())
        for pack in trusted_packs
    )
    # Calibration rows intentionally recur across different M treatments, so
    # this is not an exact-coverage arm and cannot use _arm_plan.
    calls = tuple(
        PlannedCall(
            schema_version=CALL_PLAN_SCHEMA,
            call_index=index,
            arm=CALIBRATION_ARM,
            layer=layer,
            expert_id=expert_id,
            rows=rows,
            execution_m=execution_m,
            padding_rows=padding_rows,
            expected_signatures=signatures,
        )
        for index, (layer, expert_id, rows, execution_m, padding_rows, signatures) in enumerate(specs)
    )
    del records
    return ArmPlan(arm=CALIBRATION_ARM, calls=calls)


@dataclass(frozen=True, slots=True)
class RowExecution:
    row_id: str
    reference_sha256: str
    repeat_sha256: tuple[str, ...]
    repeat_mismatch_counts: tuple[int, ...]
    bitwise_stable: bool
    all_exact_to_reference: bool


@dataclass(frozen=True, slots=True)
class ArmExecution:
    arm: str
    warmups: int
    repeats: int
    latency_ms: tuple[float, ...]
    rows: tuple[RowExecution, ...]
    representative_outputs: tuple[tuple[str, bytes], ...]
    representative_call_output_sha256: tuple[str, ...]
    call_count: int
    padding_rows: int

    def reference_output_map(self) -> dict[str, bytes]:
        return dict(self.representative_outputs)


@dataclass(frozen=True, slots=True)
class CalibrationPackExecution:
    pack: Any
    repeat_row_exact: tuple[tuple[bool, ...], ...]
    repeat_row_sha256: tuple[tuple[str, ...], ...]
    representative_full_output_sha256: str


@dataclass(frozen=True, slots=True)
class CalibrationExecution:
    reference: ArmExecution
    packs: tuple[CalibrationPackExecution, ...]


def _materialized_lookup(rows: Iterable[MaterializedRow]) -> dict[str, MaterializedRow]:
    lookup: dict[str, MaterializedRow] = {}
    for row in rows:
        if not isinstance(row, MaterializedRow):
            raise GPUExecutionError("GPU execution requires MaterializedRow values")
        if row.row_id in lookup:
            raise GPUExecutionError("GPU execution rows contain duplicate identities")
        lookup[row.row_id] = row
    return lookup


def _reference_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        if len(value) % 2:
            raise GPUExecutionError("BF16 reference byte length is odd")
        return value
    return bf16_storage_bytes(value)


def execute_arm(
    *,
    model: Any,
    plan: ArmPlan,
    rows: Iterable[MaterializedRow],
    reference_outputs: Mapping[str, Any] | None = None,
    warmups: int = DEFAULT_WARMUPS,
    repeats: int = DEFAULT_REPEATS,
    device: str = "cuda",
) -> ArmExecution:
    """Execute every planned call and compare every ``result[i]`` to M=1.

    Input packs and scatter indices are built before timing.  CUDA Events cover
    expert calls and canonical index-copy scatter.  For arm A, the first formal
    repeat becomes the returned canonical reference and later repeats establish
    its stability.  Arms B/C/D must receive A's reference map.
    """

    if int(warmups) != DEFAULT_WARMUPS or int(repeats) != DEFAULT_REPEATS:
        raise GPUExecutionError("GPU execution is frozen at 3 warmups / 10 repeats")
    if not isinstance(plan, ArmPlan) or not plan.calls:
        raise GPUExecutionError("execute_arm requires one non-empty ArmPlan")
    if plan.arm != ARM_A and reference_outputs is None:
        raise GPUExecutionError("non-reference arms require isolated M=1 outputs")

    torch = _torch()
    if not torch.cuda.is_available():
        raise GPUExecutionError("CUDA is unavailable")
    target_device = torch.device(device)
    lookup = _materialized_lookup(rows)
    expected_ids = tuple(sorted(plan.row_ids))
    if set(expected_ids) != set(lookup) or len(expected_ids) != len(lookup):
        raise GPUExecutionError("plan/materialized row coverage mismatch")
    canonical_index = {row_id: index for index, row_id in enumerate(expected_ids)}

    prepared: list[tuple[Any, Any, Any, int]] = []
    hidden_size: int | None = None
    for call in plan.calls:
        tensors: list[Any] = []
        for row_id in call.row_ids:
            materialized = lookup[row_id]
            value = materialized.tensor
            if value.ndim != 1 or value.dtype != torch.bfloat16:
                raise GPUExecutionError("hidden rows must be one-dimensional BF16")
            if tensor_storage_sha256(value) != materialized.record.hidden_sha256:
                raise GPUExecutionError("materialized hidden hash mismatch")
            current_size = int(value.numel())
            if hidden_size is None:
                hidden_size = current_size
            elif current_size != hidden_size:
                raise GPUExecutionError("hidden rows have inconsistent widths")
            tensors.append(value.to(device=target_device, dtype=torch.bfloat16))
        batch = torch.stack(tensors, dim=0)
        if call.padding_rows:
            padding = torch.zeros(
                (call.padding_rows, int(hidden_size)),
                device=target_device,
                dtype=torch.bfloat16,
            )
            batch = torch.cat((batch, padding), dim=0)
        if int(batch.shape[0]) != call.execution_m:
            raise GPUExecutionError("prebuilt execution batch has wrong M")
        scatter = torch.tensor(
            [canonical_index[row_id] for row_id in call.row_ids],
            device=target_device,
            dtype=torch.long,
        )
        try:
            expert = model.model.layers[call.layer].mlp.experts[call.expert_id]
        except (AttributeError, IndexError) as exc:
            raise GPUExecutionError("planned expert does not exist in model") from exc
        prepared.append((expert, batch, scatter, len(call.rows)))
    assert hidden_size is not None
    output_buffer = torch.empty(
        (len(expected_ids), hidden_size),
        device=target_device,
        dtype=torch.bfloat16,
    )

    def execute_once() -> Any:
        for expert, batch, scatter, real_count in prepared:
            result = expert(batch)
            expected_shape = (int(batch.shape[0]), hidden_size)
            if tuple(result.shape) != expected_shape or result.dtype != torch.bfloat16:
                raise GPUExecutionError("expert result shape/dtype mismatch")
            output_buffer.index_copy_(0, scatter, result[:real_count])
        return output_buffer

    with torch.inference_mode():
        for _ in range(DEFAULT_WARMUPS):
            execute_once()
        torch.cuda.synchronize(target_device)

    latency_ms: list[float] = []
    repeat_bytes: dict[str, list[bytes]] = {row_id: [] for row_id in expected_ids}
    with torch.inference_mode():
        for _repeat in range(DEFAULT_REPEATS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = execute_once()
            end.record()
            end.synchronize()
            latency_ms.append(float(start.elapsed_time(end)))
            result_cpu = result.detach().cpu().clone()
            if not bool(torch.isfinite(result_cpu).all().item()):
                raise GPUExecutionError("expert arm produced a non-finite output")
            for row_index, row_id in enumerate(expected_ids):
                repeat_bytes[row_id].append(bf16_storage_bytes(result_cpu[row_index]))

    supplied_references = (
        None
        if reference_outputs is None
        else {row_id: _reference_bytes(value) for row_id, value in reference_outputs.items()}
    )
    if supplied_references is not None and set(supplied_references) != set(expected_ids):
        raise GPUExecutionError("reference output coverage does not match the arm")

    row_results: list[RowExecution] = []
    representatives: list[tuple[str, bytes]] = []
    for row_id in expected_ids:
        values = repeat_bytes[row_id]
        reference = values[0] if supplied_references is None else supplied_references[row_id]
        hashes = tuple(hashlib.sha256(value).hexdigest() for value in values)
        mismatches = tuple(
            strict_bf16_mismatch_count(reference, value) for value in values
        )
        representatives.append((row_id, values[0]))
        row_results.append(
            RowExecution(
                row_id=row_id,
                reference_sha256=hashlib.sha256(reference).hexdigest(),
                repeat_sha256=hashes,
                repeat_mismatch_counts=mismatches,
                bitwise_stable=len(set(hashes)) == 1,
                all_exact_to_reference=all(value == 0 for value in mismatches),
            )
        )
    # One untimed representative pass binds every logical call to the later
    # trace worker without contaminating the CUDA latency samples.
    representative_call_hashes: list[str] = []
    with torch.inference_mode():
        for expert, batch, _scatter, _real_count in prepared:
            output = expert(batch).detach()
            representative_call_hashes.append(tensor_storage_sha256(output))
        torch.cuda.synchronize(target_device)
    return ArmExecution(
        arm=plan.arm,
        warmups=DEFAULT_WARMUPS,
        repeats=DEFAULT_REPEATS,
        latency_ms=tuple(latency_ms),
        rows=tuple(row_results),
        representative_outputs=tuple(representatives),
        representative_call_output_sha256=tuple(representative_call_hashes),
        call_count=len(plan.calls),
        padding_rows=plan.padding_rows,
    )


def execute_paired_arms(
    *,
    model: Any,
    plans: Mapping[str, ArmPlan],
    rows: Iterable[MaterializedRow],
    warmups: int = DEFAULT_WARMUPS,
    repeats: int = DEFAULT_REPEATS,
    device: str = "cuda",
    raw_output_dir: Path | None = None,
) -> Mapping[str, ArmExecution]:
    """Execute A/B/C/D as repeat-level, counterbalanced paired samples.

    Every arm is fully materialized before timing.  Within paired repeat ``i``
    the arm order is ``frozen_arm_order(i)``; latency vector element ``i`` is
    therefore the same pair for all arms.  The isolated A output from pair 0
    is the raw-BF16 reference for every arm and every repeat.
    """

    if int(warmups) != DEFAULT_WARMUPS or int(repeats) != DEFAULT_REPEATS:
        raise GPUExecutionError("paired execution is frozen at 3 warmups / 10 repeats")
    if set(plans) != set(FROZEN_ARM_ORDER):
        raise GPUExecutionError("paired execution requires exactly the four frozen arms")

    torch = _torch()
    if not torch.cuda.is_available():
        raise GPUExecutionError("CUDA is unavailable")
    target_device = torch.device(device)
    raw_dir = None if raw_output_dir is None else Path(raw_output_dir).resolve()
    if raw_dir is not None:
        if raw_dir.exists():
            raise GPUExecutionError("raw paired-output directory already exists")
        raw_dir.mkdir(parents=True)
    lookup = _materialized_lookup(rows)
    expected_ids = tuple(sorted(lookup))
    canonical_index = {row_id: index for index, row_id in enumerate(expected_ids)}

    prepared_by_arm: dict[str, tuple[tuple[Any, Any, Any, int], ...]] = {}
    output_by_arm: dict[str, Any] = {}
    hidden_size: int | None = None
    frozen_traversal: tuple[str, ...] | None = None
    for arm in FROZEN_ARM_ORDER:
        plan = plans[arm]
        if not isinstance(plan, ArmPlan) or plan.arm != arm or not plan.calls:
            raise GPUExecutionError(f"invalid paired plan for {arm}")
        if set(plan.row_ids) != set(expected_ids) or len(plan.row_ids) != len(expected_ids):
            raise GPUExecutionError(f"paired plan coverage mismatch for {arm}")
        if frozen_traversal is None:
            frozen_traversal = plan.row_ids
        elif plan.row_ids != frozen_traversal:
            raise GPUExecutionError(f"paired plan traversal mismatch for {arm}")

        prepared: list[tuple[Any, Any, Any, int]] = []
        for call in plan.calls:
            tensors: list[Any] = []
            for row_id in call.row_ids:
                materialized = lookup[row_id]
                value = materialized.tensor
                if value.ndim != 1 or value.dtype != torch.bfloat16:
                    raise GPUExecutionError("hidden rows must be one-dimensional BF16")
                if tensor_storage_sha256(value) != materialized.record.hidden_sha256:
                    raise GPUExecutionError("materialized hidden hash mismatch")
                width = int(value.numel())
                if hidden_size is None:
                    hidden_size = width
                elif width != hidden_size:
                    raise GPUExecutionError("hidden rows have inconsistent widths")
                tensors.append(value.to(device=target_device, dtype=torch.bfloat16))
            batch = torch.stack(tensors, dim=0)
            if call.padding_rows:
                batch = torch.cat(
                    (
                        batch,
                        torch.zeros(
                            (call.padding_rows, int(hidden_size)),
                            device=target_device,
                            dtype=torch.bfloat16,
                        ),
                    ),
                    dim=0,
                )
            if int(batch.shape[0]) != call.execution_m:
                raise GPUExecutionError("prebuilt paired batch has wrong M")
            scatter = torch.tensor(
                [canonical_index[row_id] for row_id in call.row_ids],
                device=target_device,
                dtype=torch.long,
            )
            try:
                expert = model.model.layers[call.layer].mlp.experts[call.expert_id]
            except (AttributeError, IndexError) as exc:
                raise GPUExecutionError("planned expert does not exist in model") from exc
            prepared.append((expert, batch, scatter, len(call.rows)))
        prepared_by_arm[arm] = tuple(prepared)
        output_by_arm[arm] = torch.empty(
            (len(expected_ids), int(hidden_size)),
            device=target_device,
            dtype=torch.bfloat16,
        )
    assert hidden_size is not None

    def execute_once(arm: str) -> Any:
        output = output_by_arm[arm]
        for expert, batch, scatter, real_count in prepared_by_arm[arm]:
            result = expert(batch)
            if tuple(result.shape) != (int(batch.shape[0]), hidden_size):
                raise GPUExecutionError("expert result shape mismatch")
            if result.dtype != torch.bfloat16:
                raise GPUExecutionError("expert result dtype mismatch")
            output.index_copy_(0, scatter, result[:real_count])
        return output

    with torch.inference_mode():
        for warmup_index in range(DEFAULT_WARMUPS):
            for arm in frozen_arm_order(warmup_index):
                execute_once(arm)
        torch.cuda.synchronize(target_device)

    latency_ms: dict[str, list[float]] = {arm: [] for arm in FROZEN_ARM_ORDER}
    repeat_bytes: dict[str, dict[str, list[bytes]]] = {
        arm: {row_id: [] for row_id in expected_ids} for arm in FROZEN_ARM_ORDER
    }
    with torch.inference_mode():
        for repeat_index in range(DEFAULT_REPEATS):
            for arm in frozen_arm_order(repeat_index):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                result = execute_once(arm)
                end.record()
                end.synchronize()
                latency_ms[arm].append(float(start.elapsed_time(end)))
                result_cpu = result.detach().cpu().clone()
                if not bool(torch.isfinite(result_cpu).all().item()):
                    raise GPUExecutionError("paired arm produced a non-finite output")
                full_raw = bf16_storage_bytes(result_cpu)
                expected_raw_size = len(expected_ids) * hidden_size * 2
                if len(full_raw) != expected_raw_size:
                    raise GPUExecutionError("paired raw BF16 byte size mismatch")
                if raw_dir is not None:
                    raw_path = raw_dir / f"pair_{repeat_index:02d}_{arm}.bf16"
                    with raw_path.open("xb") as handle:
                        handle.write(full_raw)
                        handle.flush()
                        os.fsync(handle.fileno())
                for row_index, row_id in enumerate(expected_ids):
                    start_byte = row_index * hidden_size * 2
                    repeat_bytes[arm][row_id].append(
                        full_raw[start_byte : start_byte + hidden_size * 2]
                    )

    references = {
        row_id: repeat_bytes[ARM_A][row_id][0] for row_id in expected_ids
    }
    executions: dict[str, ArmExecution] = {}
    with torch.inference_mode():
        for arm in FROZEN_ARM_ORDER:
            row_results: list[RowExecution] = []
            representatives: list[tuple[str, bytes]] = []
            for row_id in expected_ids:
                values = repeat_bytes[arm][row_id]
                reference = references[row_id]
                hashes = tuple(hashlib.sha256(value).hexdigest() for value in values)
                mismatches = tuple(
                    strict_bf16_mismatch_count(reference, value) for value in values
                )
                representatives.append((row_id, values[0]))
                row_results.append(
                    RowExecution(
                        row_id=row_id,
                        reference_sha256=hashlib.sha256(reference).hexdigest(),
                        repeat_sha256=hashes,
                        repeat_mismatch_counts=mismatches,
                        bitwise_stable=len(set(hashes)) == 1,
                        all_exact_to_reference=all(value == 0 for value in mismatches),
                    )
                )
            representative_call_hashes: list[str] = []
            for expert, batch, _scatter, _real_count in prepared_by_arm[arm]:
                representative_call_hashes.append(
                    tensor_storage_sha256(expert(batch).detach())
                )
            torch.cuda.synchronize(target_device)
            executions[arm] = ArmExecution(
                arm=arm,
                warmups=DEFAULT_WARMUPS,
                repeats=DEFAULT_REPEATS,
                latency_ms=tuple(latency_ms[arm]),
                rows=tuple(row_results),
                representative_outputs=tuple(representatives),
                representative_call_output_sha256=tuple(
                    representative_call_hashes
                ),
                call_count=len(plans[arm].calls),
                padding_rows=plans[arm].padding_rows,
            )
    return executions


def execute_calibration(
    *,
    model: Any,
    packs: Iterable[Any],
    rows: Iterable[MaterializedRow],
    repeats: int = DEFAULT_REPEATS,
    device: str = "cuda",
) -> CalibrationExecution:
    """Build all-row M=1 references, then measure every real calibration pack."""

    if int(repeats) != DEFAULT_REPEATS:
        raise GPUExecutionError("calibration repeats are frozen at 10")
    materialized = tuple(rows)
    if not materialized or any(row.record.split != "calibration" for row in materialized):
        raise GPUExecutionError("calibration execution requires calibration rows")
    trusted_packs = tuple(packs)
    if not trusted_packs:
        raise GPUExecutionError("calibration execution requires real M>1 packs")
    reference_plan = plan_arm_a(materialized)
    reference = execute_arm(
        model=model,
        plan=reference_plan,
        rows=materialized,
        reference_outputs=None,
        device=device,
    )
    reference_bytes = reference.reference_output_map()
    lookup = _materialized_lookup(materialized)
    torch = _torch()
    target_device = torch.device(device)
    executions: list[CalibrationPackExecution] = []
    for pack in trusted_packs:
        if not isinstance(pack, CONTRACT.Pack) or pack.m <= 1:
            raise GPUExecutionError("calibration pack is invalid")
        tensors = [
            lookup[row.row_id].tensor.to(device=target_device, dtype=torch.bfloat16)
            for row in pack.rows
        ]
        batch = torch.stack(tensors, dim=0)
        expert = model.model.layers[pack.layer].mlp.experts[pack.expert_id]
        repeat_exact: list[tuple[bool, ...]] = []
        repeat_hashes: list[tuple[str, ...]] = []
        representative_full_hash: str | None = None
        with torch.inference_mode():
            for repeat_index in range(DEFAULT_REPEATS):
                output = expert(batch).detach().cpu().clone()
                if output.dtype != torch.bfloat16 or not bool(torch.isfinite(output).all().item()):
                    raise GPUExecutionError("calibration expert output is invalid")
                if representative_full_hash is None:
                    representative_full_hash = tensor_storage_sha256(output)
                exact_flags: list[bool] = []
                hashes: list[str] = []
                for row_index, row in enumerate(pack.rows):
                    raw = bf16_storage_bytes(output[row_index])
                    hashes.append(hashlib.sha256(raw).hexdigest())
                    exact_flags.append(
                        strict_bf16_mismatch_count(reference_bytes[row.row_id], raw) == 0
                    )
                repeat_exact.append(tuple(exact_flags))
                repeat_hashes.append(tuple(hashes))
        assert representative_full_hash is not None
        executions.append(
            CalibrationPackExecution(
                pack=pack,
                repeat_row_exact=tuple(repeat_exact),
                repeat_row_sha256=tuple(repeat_hashes),
                representative_full_output_sha256=representative_full_hash,
            )
        )
    return CalibrationExecution(reference=reference, packs=tuple(executions))


__all__ = [
    "ARM_A",
    "ARM_B",
    "ARM_C",
    "ARM_D",
    "ArmExecution",
    "ArmPlan",
    "CapturedWindow",
    "CalibrationExecution",
    "CalibrationPackExecution",
    "GPUExecutionError",
    "MaterializedRow",
    "PlannedCall",
    "RowContext",
    "RowExecution",
    "bf16_storage_bytes",
    "build_calibration_packs",
    "calibration_call_plan",
    "capture_olmoe_split",
    "execute_arm",
    "execute_calibration",
    "execute_paired_arms",
    "frozen_arm_order",
    "materialize_routed_rows",
    "plan_arm_a",
    "plan_arm_b",
    "plan_arm_c",
    "plan_arm_d",
    "plan_four_arms",
    "positions_for_split",
    "strict_bf16_mismatch_count",
    "tensor_storage_sha256",
]
