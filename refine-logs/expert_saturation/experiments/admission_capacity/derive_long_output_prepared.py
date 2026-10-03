"""Derive a long-output prepared directory from a frozen short-output one.

The KV-pressure regime needs each request to occupy far more KV cache. The
cheapest way to get there without touching request identity is to keep the exact
same prompts, arrivals and cohort, and only extend the number of generated
tokens. This script does exactly that and proves it did nothing else:

  * `workload.json` is copied byte-for-byte, so `workload_sha256` is unchanged
    and every prompt token id, hash, document id and arrival time is identical
    to the sealed cohort.
  * `config.json` differs only in `output_tokens` plus explicit provenance
    fields recording the derivation. Every other key is asserted equal.

The point of the regime change is structural, and specific to MoE serving:
OLMoE-1B-7B pays HBM for 7B of expert weights while computing with ~1B, and this
model's KV cost is 128 KiB per token. Weight footprint therefore squeezes KV
capacity far harder than in a dense model of equal compute, which is what makes
KV a genuinely scarce resource here rather than an artificial constraint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

KV_BYTES_PER_TOKEN = 131072  # 16 layers * 16 kv heads * 128 head_dim * 2 (K,V) * 2 bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-prepared", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-tokens", required=True, type=int)
    parser.add_argument("--ttft-slo-s", type=float, default=None)
    parser.add_argument("--tpot-slo-s", type=float, default=None)
    args = parser.parse_args()
    if args.output_tokens < 2:
        parser.error("need at least two output tokens")
    if args.output_dir.exists():
        parser.error("output directory must be new")

    source_config = json.loads((args.source_prepared / "config.json").read_text())
    workload_bytes = (args.source_prepared / "workload.json").read_bytes()
    workload = json.loads(workload_bytes)
    digest = hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest()
    if digest != source_config["workload_sha256"]:
        raise ValueError("source prepared workload hash does not match its config")

    args.output_dir.mkdir(parents=True)
    # Byte-identical copy: identity is inherited, not regenerated.
    (args.output_dir / "workload.json").write_bytes(workload_bytes)
    copied_digest = hashlib.sha256(
        json.dumps(json.loads((args.output_dir / "workload.json").read_bytes()),
                   sort_keys=True).encode()).hexdigest()
    if copied_digest != source_config["workload_sha256"]:
        raise ValueError("copied workload hash changed")

    config = dict(source_config)
    previous_output = config["output_tokens"]
    config["output_tokens"] = args.output_tokens
    if args.ttft_slo_s is not None:
        config["ttft_slo_s"] = args.ttft_slo_s
    if args.tpot_slo_s is not None:
        config["tpot_slo_s"] = args.tpot_slo_s
    prompt_tokens = config["prompt_tokens"]
    requests = config["requests"]
    peak_tokens = requests * (prompt_tokens + args.output_tokens)
    config.update(
        derived_from=str(args.source_prepared),
        derived_rule="identical workload bytes; only output_tokens (and optional SLO) changed",
        derived_previous_output_tokens=previous_output,
        kv_bytes_per_token=KV_BYTES_PER_TOKEN,
        kv_peak_bytes_all_requests=peak_tokens * KV_BYTES_PER_TOKEN,
        kv_peak_tokens_all_requests=peak_tokens,
        final_context_tokens_per_request=prompt_tokens + args.output_tokens)
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    # Prove only the intended keys moved.
    changed = {k for k in set(config) | set(source_config)
               if config.get(k) != source_config.get(k)}
    allowed = {"output_tokens", "ttft_slo_s", "tpot_slo_s", "derived_from", "derived_rule",
               "derived_previous_output_tokens", "kv_bytes_per_token",
               "kv_peak_bytes_all_requests", "kv_peak_tokens_all_requests",
               "final_context_tokens_per_request"}
    unexpected = changed - allowed
    if unexpected:
        raise ValueError(f"unexpected config changes: {sorted(unexpected)}")

    for name in ("environment.json", "preflight.json"):
        source_file = args.source_prepared / name
        if source_file.exists():
            shutil.copy2(source_file, args.output_dir / f"inherited-{name}")

    summary = dict(
        source_prepared=str(args.source_prepared), output_dir=str(args.output_dir),
        workload_sha256=source_config["workload_sha256"],
        workload_bytes_identical=True,
        output_tokens=dict(before=previous_output, after=args.output_tokens),
        changed_config_keys=sorted(changed),
        requests=requests, prompt_tokens=prompt_tokens,
        final_context_tokens_per_request=prompt_tokens + args.output_tokens,
        kv_peak_gib_all_requests=peak_tokens * KV_BYTES_PER_TOKEN / 2 ** 30)
    (args.output_dir / "derivation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
