from __future__ import annotations

"""Capture identity-complete native MoE routes or normalize an existing capture."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

try:
    from .core import Contribution, ProtocolError, sha256_file, write_json, write_routes
except ImportError:
    from core import Contribution, ProtocolError, sha256_file, write_json, write_routes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-csv", help="existing sample/layer/token/rank/expert capture")
    source.add_argument("--model", help="Hugging Face model id to execute")
    source.add_argument("--smoke", action="store_true", help="write deterministic non-evidence fixtures")
    parser.add_argument("--model-key")
    parser.add_argument("--model-revision")
    parser.add_argument("--dataset", default="wikitext103_docs")
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--phase", choices=("prefill", "decode"), default="decode")
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=16,
        help="maximum cached one-token steps when --phase decode",
    )
    parser.add_argument("--interarrival-us", type=float, default=5.0)
    parser.add_argument("--deadline-us", type=float, default=500.0)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="development only; formal capture requires CUDA")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


@dataclass(frozen=True)
class CachedDecodeStep:
    decode_step: int
    token_id: int
    absolute_position: int
    cache_length: int | None
    route_batches: tuple[Mapping[str, Any], ...]
    logits: Any | None = None


def _clear_recorder(recorder: object) -> None:
    getattr(recorder, "route_batches").clear()
    getattr(recorder, "routing_weight_batches").clear()


def _cache_sequence_length(cache: object) -> int | None:
    get_seq_length = getattr(cache, "get_seq_length", None)
    if callable(get_seq_length):
        return int(get_seq_length())
    try:
        first_key = cache[0][0]  # type: ignore[index]
        return int(first_key.shape[-2])
    except (IndexError, KeyError, TypeError, AttributeError):
        return None


def _snapshot_route_batches(recorder: object, *, expected_batch: int) -> tuple[Mapping[str, Any], ...]:
    batches: list[Mapping[str, Any]] = []
    layers: set[int] = set()
    for raw in getattr(recorder, "route_batches"):
        selected = raw["selected_experts"]
        weights = raw["routing_weights"]
        if selected.ndim != 2 or weights.shape != selected.shape:
            raise ProtocolError("router capture must be a matching [tokens, top_k] pair")
        if int(selected.shape[0]) != expected_batch:
            raise ProtocolError(
                "router capture row count does not match the expected input-token count "
                f"(observed={int(selected.shape[0])}, expected={expected_batch})"
            )
        layer = int(raw["layer"])
        if layer in layers:
            raise ProtocolError(f"router hook recorded layer {layer} more than once in one step")
        layers.add(layer)
        batches.append(
            {
                "sample_id": int(raw["sample_id"]),
                "layer": layer,
                "selected_experts": selected.clone(),
                "routing_weights": weights.clone(),
            }
        )
    if not batches:
        raise ProtocolError("cached decode produced no router records")
    return tuple(batches)


def run_cached_decode_steps(
    model: object,
    recorder: object,
    inputs: Mapping[str, Any],
    *,
    max_steps: int,
    eos_token_id: int | None,
    forced_decode_ids: Any | None = None,
    capture_logits: bool = False,
) -> list[CachedDecodeStep]:
    """Run a real prefill followed by cached, one-token decode steps.

    ``forced_decode_ids`` exists only for deterministic equivalence tests.  A
    normal capture greedily feeds the token predicted by the preceding step.
    The current producer intentionally supports batch size one: accepting a
    flattened multi-request recorder without an explicit row-to-request map
    would make identity attribution unsound.
    """

    try:
        import torch
    except ImportError as exc:
        raise ProtocolError("cached decode requires PyTorch") from exc
    if max_steps <= 0:
        raise ProtocolError("decode steps must be positive")
    input_ids = inputs.get("input_ids")
    if input_ids is None or input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise ProtocolError("identity-safe cached capture currently requires input_ids batch size one")
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    _clear_recorder(recorder)
    with torch.inference_mode():
        prefill = model(**dict(inputs), use_cache=True, return_dict=True)
    cache = getattr(prefill, "past_key_values", None)
    logits = getattr(prefill, "logits", None)
    if cache is None or logits is None:
        raise ProtocolError("prefill did not return logits and past_key_values")
    prompt_length = int(input_ids.shape[1])
    prefill_cache_length = _cache_sequence_length(cache)
    if prefill_cache_length is not None and prefill_cache_length != prompt_length:
        raise ProtocolError(
            f"prefill cache length {prefill_cache_length} != prompt length {prompt_length}"
        )
    _clear_recorder(recorder)  # prefill routes must never be labeled decode

    if forced_decode_ids is not None:
        if forced_decode_ids.ndim != 2 or int(forced_decode_ids.shape[0]) != 1:
            raise ProtocolError("forced decode ids must have shape [1, steps]")
        step_limit = min(max_steps, int(forced_decode_ids.shape[1]))
        next_token = forced_decode_ids[:, :1]
    else:
        step_limit = max_steps
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    steps: list[CachedDecodeStep] = []
    expected_signature: tuple[tuple[int, int], ...] | None = None
    for step in range(step_limit):
        if forced_decode_ids is not None:
            next_token = forced_decode_ids[:, step : step + 1]
        if eos_token_id is not None and bool(torch.all(next_token == eos_token_id).item()):
            break

        attention_mask = torch.cat(
            (attention_mask, torch.ones_like(attention_mask[:, :1])), dim=1
        )
        position_ids = attention_mask.long().cumsum(-1)[:, -1:] - 1
        with torch.inference_mode():
            output = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        cache = getattr(output, "past_key_values", None)
        output_logits = getattr(output, "logits", None)
        if cache is None or output_logits is None:
            raise ProtocolError(f"decode step {step} did not return logits and past_key_values")
        cache_length = _cache_sequence_length(cache)
        expected_length = prompt_length + step + 1
        if cache_length is not None and cache_length != expected_length:
            raise ProtocolError(
                f"decode cache length {cache_length} != expected {expected_length} at step {step}"
            )
        route_batches = _snapshot_route_batches(recorder, expected_batch=1)
        signature = tuple(
            sorted(
                (int(batch["layer"]), int(batch["selected_experts"].shape[1]))
                for batch in route_batches
            )
        )
        if expected_signature is None:
            expected_signature = signature
        elif signature != expected_signature:
            raise ProtocolError(
                f"router layer/top-k closure changed at decode step {step}: "
                f"expected={expected_signature}, observed={signature}"
            )
        steps.append(
            CachedDecodeStep(
                decode_step=step,
                token_id=int(next_token.item()),
                absolute_position=prompt_length + step,
                cache_length=cache_length,
                route_batches=route_batches,
                logits=(output_logits.detach().float().cpu() if capture_logits else None),
            )
        )
        _clear_recorder(recorder)
        if forced_decode_ids is None:
            next_token = torch.argmax(output_logits[:, -1, :], dim=-1, keepdim=True)
    return steps


def _request_fields(sample_id: int, offset: int, model: str, phase: str, gap: float, deadline: float) -> tuple[str, float, float]:
    local = sample_id - offset
    arrival = local * gap
    return f"{model}:{phase}:{sample_id:06d}", arrival, arrival + deadline


def _raw_int(row: Mapping[str, str], key: str, default: int) -> int:
    value = row.get(key)
    return default if value is None or value == "" else int(value)


def normalize_source(args: argparse.Namespace) -> list[Contribution]:
    model = args.model_key
    if not model:
        raise ProtocolError("--model-key is required with --source-csv")
    rows: list[Contribution] = []
    with Path(args.source_csv).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "layer", "token_position", "rank", "expert_id", "gate_weight"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ProtocolError(f"source capture missing columns {sorted(missing)}")
        for raw in reader:
            sample_id = int(raw["sample_id"])
            request_id, arrival, deadline = _request_fields(
                sample_id, args.offset, model, args.phase, args.interarrival_us, args.deadline_us
            )
            rows.append(
                Contribution(
                    model=model,
                    phase=args.phase,
                    request_id=request_id,
                    sample_id=sample_id,
                    arrival_us=arrival,
                    deadline_us=deadline,
                    layer=int(raw["layer"]),
                    token_position=int(raw["token_position"]),
                    rank=int(raw["rank"]),
                    expert_id=int(raw["expert_id"]),
                    gate_weight=float(raw["gate_weight"]),
                    src_replica=_raw_int(raw, "src_replica", 0),
                    input_event_id=str(
                        raw.get("input_event_id")
                        or f"{request_id}:{args.phase}:{int(raw['token_position']):06d}"
                    ),
                    token_id=_raw_int(raw, "token_id", int(raw["token_position"])),
                    decode_step=_raw_int(raw, "decode_step", -1),
                    layer_id=_raw_int(raw, "layer_id", int(raw["layer"])),
                    topk_slot=_raw_int(raw, "topk_slot", int(raw["rank"]) - 1),
                    source_rank=_raw_int(raw, "source_rank", _raw_int(raw, "src_replica", 0)),
                    target_replica=_raw_int(raw, "target_replica", -1),
                )
            )
    return rows


def smoke_rows(model: str, *, top_k: int, experts: int, samples: int, phase: str) -> list[Contribution]:
    rows: list[Contribution] = []
    for sample in range(samples):
        arrival = float(sample * 7)
        for layer in range(2):
            for token in range(4):
                chosen: list[int] = []
                candidate = (sample + token + layer) % experts
                for rank in range(1, top_k + 1):
                    while candidate in chosen:
                        candidate = (candidate + 1) % experts
                    chosen.append(candidate)
                    rows.append(
                        Contribution(
                            model=model,
                            phase=phase,
                            request_id=f"{model}:{phase}:{sample:06d}",
                            sample_id=sample,
                            arrival_us=arrival,
                            deadline_us=arrival + 180.0,
                            layer=layer,
                            token_position=token,
                            rank=rank,
                            expert_id=candidate,
                            gate_weight=1.0 / top_k,
                            src_replica=0,
                            input_event_id=f"{model}:{phase}:{sample:06d}:{token:06d}",
                            token_id=token,
                            decode_step=token if phase == "decode" else -1,
                            layer_id=layer,
                            topk_slot=rank - 1,
                            source_rank=0,
                            target_replica=-1,
                        )
                    )
                    candidate = (candidate + 3 + rank) % experts
    return rows


def _contributions_from_batches(
    *,
    batches: Sequence[Mapping[str, Any]],
    model_key: str,
    phase: str,
    request_id: str,
    sample_id: int,
    arrival_us: float,
    deadline_us: float,
    input_event_ids: Sequence[str],
    token_ids: Sequence[int],
    token_positions: Sequence[int],
    decode_steps: Sequence[int],
) -> list[Contribution]:
    count = len(input_event_ids)
    if not (len(token_ids) == len(token_positions) == len(decode_steps) == count):
        raise ProtocolError("route identity vectors have inconsistent lengths")
    rows: list[Contribution] = []
    for batch in batches:
        selected = batch["selected_experts"]
        weights = batch["routing_weights"]
        if int(selected.shape[0]) != count:
            raise ProtocolError("router rows do not match explicit input-event identities")
        layer = int(batch["layer"])
        for token_index in range(count):
            for slot in range(int(selected.shape[1])):
                rows.append(
                    Contribution(
                        model=model_key,
                        phase=phase,
                        request_id=request_id,
                        sample_id=sample_id,
                        arrival_us=arrival_us,
                        deadline_us=deadline_us,
                        layer=layer,
                        token_position=int(token_positions[token_index]),
                        rank=slot + 1,
                        expert_id=int(selected[token_index, slot].item()),
                        gate_weight=float(weights[token_index, slot].item()),
                        src_replica=0,
                        input_event_id=input_event_ids[token_index],
                        token_id=int(token_ids[token_index]),
                        decode_step=int(decode_steps[token_index]),
                        layer_id=layer,
                        topk_slot=slot,
                        source_rank=0,
                        target_replica=-1,
                    )
                )
    return rows


def capture_model(args: argparse.Namespace) -> list[Contribution]:
    try:
        import torch
    except ImportError as exc:
        raise ProtocolError("native capture requires PyTorch") from exc
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise ProtocolError("CUDA is required for native formal capture; --allow-cpu is development-only")

    shared = next(
        candidate / "experiments/shared"
        for candidate in Path(__file__).resolve().parents
        if (candidate / "experiments/shared").is_dir()
    )
    sys.path.insert(0, str(shared))
    from capture_moe import patch_mixtral_moe
    from modeling import load_model, load_tokenizer
    from prompts import get_prompts

    model_key = args.model_key or args.model
    texts = get_prompts(args.dataset, args.samples, offset=args.offset, split=args.split)
    tokenizer = load_tokenizer(args.model, local_files_only=args.offline, revision=args.model_revision)
    model, _ = load_model(
        args.model,
        dtype_name=args.dtype,
        local_files_only=args.offline,
        revision=args.model_revision,
    )
    recorder = patch_mixtral_moe(model, "full", num_receiver_groups=1, record_routes=True)
    recorder.update_contrib = lambda *a, **k: None
    recorder.update_receiver = lambda *a, **k: None
    recorder.update_error = lambda *a, **k: None
    recorder.update_pair_audit = lambda *a, **k: None

    output: list[Contribution] = []
    for local, text in enumerate(texts):
        sample_id = args.offset + local
        recorder.set_sample_id(sample_id)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.seq_len)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        request_id, arrival, deadline = _request_fields(
            sample_id, args.offset, model_key, args.phase, args.interarrival_us, args.deadline_us
        )
        if args.phase == "decode":
            steps = run_cached_decode_steps(
                model,
                recorder,
                inputs,
                max_steps=args.decode_steps,
                eos_token_id=tokenizer.eos_token_id,
            )
            if not steps:
                raise ProtocolError(
                    f"sample {sample_id} predicted EOS before any decode route was executed"
                )
            for step in steps:
                output.extend(
                    _contributions_from_batches(
                        batches=step.route_batches,
                        model_key=model_key,
                        phase="decode",
                        request_id=request_id,
                        sample_id=sample_id,
                        arrival_us=arrival,
                        deadline_us=deadline,
                        input_event_ids=(f"{request_id}:decode:{step.decode_step:06d}",),
                        token_ids=(step.token_id,),
                        token_positions=(step.absolute_position,),
                        decode_steps=(step.decode_step,),
                    )
                )
        else:
            _clear_recorder(recorder)
            with torch.inference_mode():
                model(**inputs, use_cache=True, return_dict=True)
            token_ids = [int(value) for value in inputs["input_ids"][0].tolist()]
            batches = _snapshot_route_batches(recorder, expected_batch=len(token_ids))
            output.extend(
                _contributions_from_batches(
                    batches=batches,
                    model_key=model_key,
                    phase="prefill",
                    request_id=request_id,
                    sample_id=sample_id,
                    arrival_us=arrival,
                    deadline_us=deadline,
                    input_event_ids=tuple(
                        f"{request_id}:prefill:{position:06d}"
                        for position in range(len(token_ids))
                    ),
                    token_ids=token_ids,
                    token_positions=tuple(range(len(token_ids))),
                    decode_steps=tuple(-1 for _ in token_ids),
                )
            )
            _clear_recorder(recorder)
        print(f"captured {local + 1}/{len(texts)}", flush=True)
    return output


def main() -> None:
    args = parse_args()
    if args.interarrival_us < 0 or args.deadline_us <= 0 or args.decode_steps <= 0:
        raise SystemExit("arrival/deadline parameters must be non-negative/positive")
    if args.smoke:
        model = args.model_key or "smoke-olmoe"
        top_k = 4 if "llm" not in model else 6
        rows = smoke_rows(model, top_k=top_k, experts=8, samples=min(args.samples, 8), phase=args.phase)
        evidence = "SMOKE_ONLY synthetic route identities; not model or GPU evidence"
        source_hash = None
    elif args.source_csv:
        rows = normalize_source(args)
        evidence = "normalized route capture; eligibility remains bounded by its upstream producer"
        source_hash = sha256_file(args.source_csv)
    else:
        rows = capture_model(args)
        evidence = (
            "cached one-token decode route capture; single-device development evidence only, "
            "not continuous batching, natural timing, EP, or an end-to-end denominator"
            if args.phase == "decode"
            else "native prefill route capture; single-device development evidence only"
        )
        source_hash = None
    write_routes(args.output, rows)
    write_json(
        Path(args.output).with_suffix(".meta.json"),
        {
            "schema": "bcrd-route-v2",
            "model": args.model_key or args.model or "smoke",
            "model_revision": args.model_revision,
            "rows": len(rows),
            "requests": len({row.request_id for row in rows}),
            "source_sha256": source_hash,
            "output_sha256": sha256_file(args.output),
            "smoke": bool(args.smoke),
            "formal_eligible": False,
            "scientific_result_eligible": False,
            "dataset": args.dataset,
            "split": args.split,
            "offset": args.offset,
            "samples_requested": args.samples,
            "seq_len": args.seq_len,
            "dtype": args.dtype,
            "phase": args.phase,
            "decode_steps_max": args.decode_steps,
            "arrival_provenance": "SYNTHETIC_CLI_PARAMETERS",
            "source_rank_provenance": "SINGLE_DEVICE_OBSERVED_RANK_0",
            "target_replica_provenance": "UNASSIGNED_SENTINEL_MINUS_ONE",
            "formal_blockers": [
                "no natural continuous-batching arrival or per-layer ready-time ledger",
                "instrumented model exactness is not yet qualified on both frozen formal models",
                "no dispatch, execution, combine, latency, or energy ledger",
                "source-csv eligibility remains bounded by its upstream producer",
            ],
            "evidence_boundary": evidence,
        },
    )
    print(f"wrote {len(rows)} contributions to {args.output}")


if __name__ == "__main__":
    main()
