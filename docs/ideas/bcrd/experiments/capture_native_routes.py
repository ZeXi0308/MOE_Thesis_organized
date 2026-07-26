from __future__ import annotations

"""Capture identity-complete native MoE routes or normalize an existing capture."""

import argparse
import csv
from pathlib import Path
import sys

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
    parser.add_argument("--interarrival-us", type=float, default=5.0)
    parser.add_argument("--deadline-us", type=float, default=500.0)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="development only; formal capture requires CUDA")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _request_fields(sample_id: int, offset: int, model: str, phase: str, gap: float, deadline: float) -> tuple[str, float, float]:
    local = sample_id - offset
    arrival = local * gap
    return f"{model}:{phase}:{sample_id:06d}", arrival, arrival + deadline


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
                    src_replica=int(raw.get("src_replica") or 0),
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
                            src_replica=sample % 2,
                        )
                    )
                    candidate = (candidate + 3 + rank) % experts
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
        with torch.no_grad():
            model(**inputs)
        request_id, arrival, deadline = _request_fields(
            sample_id, args.offset, model_key, args.phase, args.interarrival_us, args.deadline_us
        )
        for batch in recorder.route_batches:
            selected = batch["selected_experts"]
            weights = batch["routing_weights"]
            for token in range(selected.shape[0]):
                for rank_index in range(selected.shape[1]):
                    output.append(
                        Contribution(
                            model=model_key,
                            phase=args.phase,
                            request_id=request_id,
                            sample_id=sample_id,
                            arrival_us=arrival,
                            deadline_us=deadline,
                            layer=int(batch["layer"]),
                            token_position=token,
                            rank=rank_index + 1,
                            expert_id=int(selected[token, rank_index].item()),
                            gate_weight=float(weights[token, rank_index].item()),
                            src_replica=sample_id % 2,
                        )
                    )
        recorder.route_batches.clear()
        recorder.routing_weight_batches.clear()
        print(f"captured {local + 1}/{len(texts)}", flush=True)
    return output


def main() -> None:
    args = parse_args()
    if args.interarrival_us < 0 or args.deadline_us <= 0:
        raise SystemExit("arrival/deadline parameters must be non-negative/positive")
    if args.smoke:
        model = args.model_key or "smoke-olmoe"
        top_k = 4 if "llm" not in model else 6
        rows = smoke_rows(model, top_k=top_k, experts=8, samples=min(args.samples, 8), phase=args.phase)
        evidence = "SMOKE_ONLY synthetic route identities; not model or GPU evidence"
        source_hash = None
    elif args.source_csv:
        rows = normalize_source(args)
        evidence = "normalized native route capture; eligibility depends on upstream producer"
        source_hash = sha256_file(args.source_csv)
    else:
        rows = capture_model(args)
        evidence = "native model forward route capture; single-device routing evidence only"
        source_hash = None
    write_routes(args.output, rows)
    write_json(
        Path(args.output).with_suffix(".meta.json"),
        {
            "schema": "bcrd-route-v1",
            "model": args.model_key or args.model or "smoke",
            "model_revision": args.model_revision,
            "rows": len(rows),
            "requests": len({row.request_id for row in rows}),
            "source_sha256": source_hash,
            "output_sha256": sha256_file(args.output),
            "smoke": bool(args.smoke),
            "evidence_boundary": evidence,
        },
    )
    print(f"wrote {len(rows)} contributions to {args.output}")


if __name__ == "__main__":
    main()
