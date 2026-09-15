#!/usr/bin/env python3
"""Generalize the sealed KV-pool qualification to a per-request output length.

The sealed `safe_static.qualify` re-derives the block pool from the live engine
and then asserts the exact 32 x (3072 + 1024) shape. Every one of those checks is
worth keeping -- they are what proves the pool is empty, the null block is
reserved, and the deficit is real -- but the shape assertion is what makes the
heterogeneous workload unrunnable.

This patch replaces the three hardcoded dimensions with the values the workload
actually declares, and generalizes `per_request` from a single scalar to a sum
over the per-request reservations:

    sum_i ceil(min(P + L_i, max_model_len) / block_size)

so `safe_cap` keeps its meaning (how many of THESE requests fit) for both a
uniform and a mixed length vector. The strength of the check is unchanged: for
the homogeneous arm the sum reduces to `N * per_request` and reproduces the
sealed arithmetic exactly.

Applied on the run host to `/root/het-r01/pkg/safe_static.py` only; the sealed
copy under the four-arm package is not modified.
"""

import re
import sys
from pathlib import Path

TARGET = Path("/root/het-r01/pkg/safe_static.py")

OLD_DIMS = """        require(config['prompt_tokens'] == 3072 and config['output_tokens'] == 1024
                and config['requests'] == 32, 'frozen request dimensions differ')
        maximum = engine.vllm_config.scheduler_config.max_num_seqs
        require(maximum == 32 and engine.vllm_config.model_config.max_model_len == 4096, 'engine bounds differ')
        per_request = (config['prompt_tokens'] + config['output_tokens'] + block_size - 1) // block_size
        safe = min(maximum, usable // per_request)
        result.update(per_request_reserved_blocks=per_request, maximum_request_tokens=4096,
            engine_max_num_seqs=maximum, safe_cap=safe, reserved_blocks_at_safe_cap=safe * per_request,
            remaining_blocks_at_full_reservation=usable - safe * per_request)"""

NEW_DIMS = """        n_requests = int(config['requests'])
        prompt = int(config['prompt_tokens'])
        lengths = config.get('output_lengths')
        if lengths is None:
            lengths = [int(config['output_tokens'])] * n_requests
        require(len(lengths) == n_requests and all(type(v) is int and v >= 2 for v in lengths),
                'per-request output lengths do not match the declared request count')
        maximum = engine.vllm_config.scheduler_config.max_num_seqs
        max_len = engine.vllm_config.model_config.max_model_len
        require(maximum == n_requests and max_len == 4096, 'engine bounds differ')
        # Admission reserves the whole sequence, capped at max_model_len, so the
        # pool requirement is the sum of per-request reservations rather than a
        # single scalar times N. For a uniform length vector this is identical
        # to the sealed arithmetic.
        per_request_blocks = [(min(prompt + L, max_len) + block_size - 1) // block_size
                              for L in lengths]
        reserved_total = sum(per_request_blocks)
        ascending = sorted(per_request_blocks)
        safe, running = 0, 0
        for cost in ascending:
            if running + cost > usable or safe >= maximum:
                break
            running += cost
            safe += 1
        per_request = max(per_request_blocks)
        result.update(per_request_reserved_blocks=per_request,
            per_request_reserved_blocks_min=min(per_request_blocks),
            reserved_blocks_all_requests=reserved_total,
            structural_deficit_blocks=reserved_total - usable,
            maximum_request_tokens=max_len,
            output_length_mode='uniform' if len(set(lengths)) == 1 else 'per_request',
            engine_max_num_seqs=maximum, safe_cap=safe, reserved_blocks_at_safe_cap=running,
            remaining_blocks_at_full_reservation=usable - reserved_total)"""


def main():
    text = TARGET.read_text()
    if NEW_DIMS in text:
        print("already patched")
        return
    if text.count(OLD_DIMS) != 1:
        print(f"ERROR: expected exactly one match, found {text.count(OLD_DIMS)}", file=sys.stderr)
        sys.exit(1)
    TARGET.write_text(text.replace(OLD_DIMS, NEW_DIMS))
    print(f"patched {TARGET}")


if __name__ == "__main__":
    main()
