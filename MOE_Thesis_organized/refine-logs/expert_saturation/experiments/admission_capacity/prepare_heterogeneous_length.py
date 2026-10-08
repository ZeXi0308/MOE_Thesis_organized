#!/usr/bin/env python3
"""Freeze the homogeneous/heterogeneous output-length pair from sealed inputs.

Why this workload exists
------------------------
`20260913_completion_dispersion_r01` closed a two-term cost model

    d_wall = d_marginal_recompute + d_tail_waste

on four admission policies. Every one of those episodes ran 32 requests with an
identical 3072-in / 1024-out shape, so completions clustered and the drain was
short (completion span 2.1-2.5 s). The model's reach is unknown once completion
times are spread out. This script freezes the workload that tests exactly that.

Deficit alignment is the whole point
------------------------------------
Under `scheduler_reserve_full_isl`, admission reserves `ceil((P + L_i)/16)`
blocks per request, so the structural deficit is

    D = sum_i ceil((P + L_i)/16) - usable_blocks

In the original domain P + L is pinned at max_model_len (4096), which leaves no
room to raise the variance of L without lowering the deficit -- and the deficit
is what creates preemption in the first place. Shortening the prompt to 1024
buys that room. Both arms here are built to reserve the SAME total blocks, so
output-length dispersion is the only variable:

    homogeneous  42 x 2048            -> 42 x 192 = 8064 blocks
    heterogeneous 21 x 1024 + 21 x 3072 -> 21 x 128 + 21 x 256 = 8064 blocks

Both give deficit 8064 - 7671 = 393 blocks = 2.047 request footprints, against
2.035 in the original domain. Mean output length (2048) and total generated
tokens (86,016) are also equal. Only CV differs: 0 versus 0.500.

Prompt provenance and its limit
-------------------------------
No dataset arrow or `datasets` package is present on the run host, so prompts
are not re-tokenized. They are non-overlapping 1024-token slices of the already
sealed 3072-token sequences: slice 0 of all 32, then slice 1 of the first 10.
Every slice is verified to be a byte-exact sub-range of a sealed sequence whose
own sha256 still matches the sealed record, so prompt identity is inherited
rather than re-derived.

The cost is real and is recorded in the manifest: 10 of the 42 prompts come from
a document that also supplies another prompt, so these are NOT 42 independent
documents. That forbids any population or generalization claim. It does not
affect arm comparability, because both arms consume the identical 42 prompts in
the identical arrival order.
"""

import argparse, hashlib, json, math, statistics
from pathlib import Path

BLOCK = 16
USABLE_BLOCKS = 7671
PROMPT_TOKENS = 1024
N_REQUESTS = 42
HOMO_OUTPUT = 2048
HET_SHORT, HET_LONG = 1024, 3072
MAX_MODEL_LEN = 4096


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def token_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def reserved_blocks(prompt, out):
    """Blocks admission reserves for one request under reserve_full_isl."""
    return math.ceil(min(prompt + out, MAX_MODEL_LEN) / BLOCK)


def slice_plan(n_source):
    """(source index, slice index) for each of the N_REQUESTS prompts."""
    plan = [(i, 0) for i in range(n_source)]
    extra = N_REQUESTS - n_source
    if extra < 0:
        return plan[:N_REQUESTS]
    if extra > n_source:
        raise ValueError("not enough sealed sequences for a second slice pass")
    plan += [(i, 1) for i in range(extra)]
    return plan


def build(source_workload, source_config):
    seq = source_workload["actual_prompt_token_ids"]
    rows = source_workload["source_requests"]
    if len(seq) != len(rows):
        raise ValueError("sealed workload rows and sequences disagree")
    for row, ids in zip(rows, seq):
        if token_sha(ids) != row["prompt_token_ids_sha256"]:
            raise ValueError(f"sealed prompt {row['request_id']} fails its own sha256")

    plan = slice_plan(len(seq))
    prompts, records = [], []
    for order, (src, sl) in enumerate(plan):
        lo, hi = sl * PROMPT_TOKENS, (sl + 1) * PROMPT_TOKENS
        if hi > len(seq[src]):
            raise ValueError("slice exceeds the sealed sequence")
        ids = seq[src][lo:hi]
        if ids != seq[src][lo:hi] or len(ids) != PROMPT_TOKENS:
            raise ValueError("slice extraction is not byte exact")
        base = rows[src]
        rid = f"{base['request_id']}-s{sl}"
        prompts.append(ids)
        records.append(dict(
            request_id=rid,
            document_id=base["document_id"],          # shared across slices on purpose
            source_request_id=base["request_id"],
            slice_index=sl,
            slice_token_range=[lo, hi],
            arrival_order=order,
            article_title=base.get("article_title"),
            prompt_tokens=PROMPT_TOKENS,
            prompt_token_ids_sha256=token_sha(ids),
        ))

    if len({r["prompt_token_ids_sha256"] for r in records}) != N_REQUESTS:
        raise ValueError("duplicate prompt token sequences among the 42")

    # Alternate by arrival order so the two length classes interleave in the
    # queue rather than arriving as two blocks.
    het = [HET_SHORT if i % 2 == 0 else HET_LONG for i in range(N_REQUESTS)]
    homo = [HOMO_OUTPUT] * N_REQUESTS

    arms = {}
    for name, lengths in (("homogeneous", homo), ("heterogeneous", het)):
        blocks = sum(reserved_blocks(PROMPT_TOKENS, L) for L in lengths)
        arms[name] = dict(
            output_lengths=lengths,
            mean_output_tokens=statistics.mean(lengths),
            cv_output_tokens=statistics.pstdev(lengths) / statistics.mean(lengths),
            total_output_tokens=sum(lengths),
            reserved_blocks=blocks,
            deficit_blocks=blocks - USABLE_BLOCKS,
            deficit_request_footprints=(blocks - USABLE_BLOCKS) / (blocks / N_REQUESTS),
        )

    a, b = arms["homogeneous"], arms["heterogeneous"]
    if a["reserved_blocks"] != b["reserved_blocks"]:
        raise ValueError("arms do not reserve the same blocks; deficit is confounded")
    if a["total_output_tokens"] != b["total_output_tokens"]:
        raise ValueError("arms do not generate the same token count")
    if a["mean_output_tokens"] != b["mean_output_tokens"]:
        raise ValueError("arms do not share a mean output length")
    if b["cv_output_tokens"] <= a["cv_output_tokens"]:
        raise ValueError("heterogeneous arm is not more dispersed")
    if not 0 < a["deficit_blocks"]:
        raise ValueError("no structural deficit; preemption would not arise")

    arrival_gap = source_config["arrival_gap_s"]
    # `capture_episode` indexes this by regime name, so it must stay a mapping.
    # Only `steady` is used here; the sealed 32-request workload has the same shape.
    arrivals = {"steady": [i * arrival_gap for i in range(N_REQUESTS)]}
    workload = dict(
        schema="heterogeneous-length-pair-v1",
        source_requests=records,
        actual_prompt_token_ids=prompts,
        arrival_traces_s=arrivals,
        arrival_rule=f"fixed {arrival_gap}s gap by arrival_order, regime 'steady'",
        output_lengths_by_arm={k: v["output_lengths"] for k, v in arms.items()},
        document_identity_scope=(
            "prompts are non-overlapping slices of sealed sequences; "
            f"{N_REQUESTS - len(seq)} documents contribute two prompts each, so these "
            "are NOT independent documents and support no population claim"),
    )
    return workload, arms, records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sealed-prepared", type=Path, required=True,
                    help="sealed inputs_preparation/prepared/long directory")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    src_w = json.load(open(args.sealed_prepared / "workload.json"))
    src_c = json.load(open(args.sealed_prepared / "config.json"))
    if sha(src_w) != src_c["workload_sha256"]:
        raise ValueError("sealed source workload does not match its own sha256")

    workload, arms, records = build(src_w, src_c)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    config = dict(
        model=src_c["model"],
        requests=N_REQUESTS,
        prompt_tokens=PROMPT_TOKENS,
        output_tokens=None,                       # per-request; see output_lengths_by_arm
        output_length_mode="per_request",
        seed=src_c["seed"],
        ttft_slo_s=src_c["ttft_slo_s"],
        tpot_slo_s=src_c["tpot_slo_s"],
        arrival_gap_s=src_c["arrival_gap_s"],
        burst_size=src_c["burst_size"],
        source=dict(src_c["source"], derived_from="sealed 3072-token sequences, sliced"),
        arms=arms,
        fixed_kv_cache_memory_bytes=16089350144,
        usable_blocks=USABLE_BLOCKS,
        max_model_len=MAX_MODEL_LEN,
        engine_max_num_seqs=N_REQUESTS,
        input_preparation=(
            "prompts inherited from sealed sequences by byte-exact slicing; "
            "no re-tokenization, no dataset access; both arms consume identical prompts"),
        evidence_ceiling="NEW_RUN_DOMAIN_NOT_COMPARABLE_TO_N32_P3072_BY_ABSOLUTE_NUMBERS",
    )
    config["workload_sha256"] = sha(workload)

    json.dump(workload, open(out / "workload.json", "w"), indent=1)
    json.dump(config, open(out / "config.json", "w"), indent=1)

    print(f"wrote {out}/workload.json and config.json")
    print(f"  prompts: {N_REQUESTS} x {PROMPT_TOKENS} tokens, "
          f"{len({r['prompt_token_ids_sha256'] for r in records})} unique")
    print(f"  distinct source documents: {len({r['source_request_id'] for r in records})}")
    for name, a in arms.items():
        print(f"  {name:14s} mean={a['mean_output_tokens']:.0f} CV={a['cv_output_tokens']:.3f} "
              f"total_out={a['total_output_tokens']} reserved={a['reserved_blocks']} "
              f"deficit={a['deficit_blocks']:+d} blocks "
              f"({a['deficit_request_footprints']:.3f} footprints)")
    print(f"  workload_sha256 {config['workload_sha256']}")


if __name__ == "__main__":
    main()
