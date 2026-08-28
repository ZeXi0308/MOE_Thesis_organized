from __future__ import annotations

"""Freeze native route rows into bounded fixed-replica Oracle instances."""

import argparse
from pathlib import Path

try:
    from .core import ProtocolError, load_routes, read_json, sha256_file, stable_index, write_json, write_jsonl
except ImportError:
    from core import ProtocolError, load_routes, read_json, sha256_file, stable_index, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--gate1-summary", required=True)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--tokens-per-instance", type=int, default=2)
    parser.add_argument("--phase", choices=("prefill", "decode"))
    parser.add_argument("--calibration-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    gate1 = read_json(args.gate1_summary)
    if not isinstance(gate1, dict):
        raise ProtocolError("Gate 1 summary must be an object")
    accepted = {"PASS_GATE1"} | ({"SMOKE_ONLY"} if args.smoke else set())
    if gate1.get("status") not in accepted:
        raise ProtocolError(f"Gate 1 status {gate1.get('status')!r} does not authorize Gate 2")
    if args.replicas < 2 or args.tokens_per_instance <= 0:
        raise ValueError("replicas>=2 and tokens-per-instance>0 are required")
    if not 0 < args.calibration_fraction < 1:
        raise ValueError("calibration-fraction must lie in (0,1)")

    rows = load_routes(args.trace)
    by_layer = {}
    for row in rows:
        if args.phase and row.phase != args.phase:
            continue
        key = (row.model, row.phase, row.layer)
        token = (row.request_id, row.token_position)
        by_layer.setdefault(key, {}).setdefault(token, []).append(row)

    instances = []
    for (model, phase, layer), tokens in sorted(by_layer.items()):
        ordered = sorted(
            tokens.items(),
            # Interleave concurrent requests at the same token position rather
            # than exhausting one request before constructing the next window.
            key=lambda value: (value[0][1], min(item.arrival_us for item in value[1]), value[0][0]),
        )
        for start in range(0, len(ordered), args.tokens_per_instance):
            chosen = ordered[start : start + args.tokens_per_instance]
            if len(chosen) < args.tokens_per_instance:
                continue
            contributions = [item for _, token_rows in chosen for item in sorted(token_rows, key=lambda row: row.rank)]
            instance_id = f"{model}:{phase}:l{layer}:w{start // args.tokens_per_instance:06d}"
            request_ids = sorted({item.request_id for item in contributions})
            cluster_id = "|".join(request_ids)
            split = (
                "calibration"
                if stable_index(instance_id, 10_000, seed=args.seed) < int(args.calibration_fraction * 10_000)
                else "evaluation"
            )
            instances.append(
                {
                    "schema": "bcrd-instance-v1",
                    "instance_id": instance_id,
                    "model": model,
                    "phase": phase,
                    "layer": layer,
                    "replica_count": args.replicas,
                    "split": split,
                    "token_count": len(chosen),
                    "request_count": len({item.request_id for item in contributions}),
                    "request_ids": request_ids,
                    "cluster_id": cluster_id,
                    "contribution_count": len(contributions),
                    "contributions": [item.to_json() for item in contributions],
                }
            )
    if not instances:
        raise ProtocolError("no complete instances could be built")
    if not any(row["split"] == "calibration" for row in instances) or not any(
        row["split"] == "evaluation" for row in instances
    ):
        # Deterministic, non-random repair; this changes only the split label.
        instances[0]["split"] = "calibration"
        instances[-1]["split"] = "evaluation"

    write_jsonl(args.output, instances)
    meta = {
        "schema": "bcrd-instances-meta-v1",
        "smoke": bool(args.smoke),
        "instances": len(instances),
        "models": sorted({str(row["model"]) for row in instances}),
        "replicas": args.replicas,
        "tokens_per_instance": args.tokens_per_instance,
        "max_contributions": max(int(row["contribution_count"]) for row in instances),
        "trace_sha256": {str(path): sha256_file(path) for path in args.trace},
        "gate1_summary_sha256": sha256_file(args.gate1_summary),
        "output_sha256": sha256_file(args.output),
        "evidence_boundary": (
            "SMOKE_ONLY fixed-replica instances" if args.smoke else
            "frozen logical assignment instances; no physical EP execution"
        ),
    }
    write_json(Path(args.output).with_suffix(".meta.json"), meta)
    return meta


def main() -> None:
    args = parse_args()
    result = run(args)
    print(f"wrote {result['instances']} instances")


if __name__ == "__main__":
    main()
