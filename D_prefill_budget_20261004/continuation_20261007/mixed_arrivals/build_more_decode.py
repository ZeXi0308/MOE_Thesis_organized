#!/usr/bin/env python3
"""CPU-only input preparation; prepared, not submitted for GPU execution.

Keep all 160 original records unchanged. Add 96 short requests using the same
16 GSM8K token templates six times each: these are explicit repetitions, not
new source examples or a natural-EOS/quality evaluation. Seed 2026100802 first
draws 96 independent uniform [0, 40) arrivals, then independently shuffles a
cyclic template-index list. No execution trace or performance result is read.
The endpoint KV figure is only a conservative token-position geometry bound,
with every request held at prompt+max-output length and no reclamation; it is
not observed peak residency, physical KV bytes, or a capacity measurement.
"""
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
SEED = 2026100802


def main():
    original = json.loads((ROOT / "workload.json").read_text())
    by_id = {r["request_id"]: r for r in original}
    background = [r for r in original if r["request_id"].startswith("high-background-")]
    long_requests = [r for r in original if r["request_id"].startswith("high-long-")]
    assert len(original) == len(by_id) == 160
    assert len(background) == 32 and len(long_requests) == 128
    templates = {}
    for r in background:
        assert r["source"] == "GSM8K" and r["max_tokens"] == 384
        key = r["source_index"]
        if key in templates:
            assert r["prompt_token_ids"] == templates[key]["prompt_token_ids"]
        else:
            templates[key] = r
    assert set(templates) == set(range(16))
    rng = random.Random(SEED)
    arrivals = [40 * rng.random() for _ in range(96)]
    template_indices = [i % 16 for i in range(96)]
    rng.shuffle(template_indices)
    extras = [dict(templates[k], request_id=f"high-background-extra-{i:03d}",
                   arrival_s=arrivals[i], max_tokens=384)
              for i, k in enumerate(template_indices)]
    rows = sorted(original + extras, key=lambda r: r["arrival_s"])
    preserved = all(r == by_id[r["request_id"]] for r in rows if r["request_id"] in by_id)
    assert preserved and len({r["request_id"] for r in rows}) == 256
    output = ROOT / "workload_more_decode.json"
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    prompt = sum(len(r["prompt_token_ids"]) for r in rows)
    generated = sum(r["max_tokens"] for r in rows)
    print(f"CPU-only prepared-not-submitted: {output}")
    print(f"seed={SEED}; requests=256; short=128; long=128; original160_fully_preserved={preserved}")
    print(f"total_prompt_tokens={prompt}; total_fixed_output_tokens={generated}; total={prompt + generated}")
    print(f"max_prompt_tokens={max(len(r['prompt_token_ids']) for r in rows)}; "
          f"max_prompt_plus_output_tokens={max(len(r['prompt_token_ids']) + r['max_tokens'] for r in rows)}")
    print(f"endpoint_KV_geometric_upper_bound_token_positions={prompt + generated} "
          "(all requests retained; no reclamation; not measured residency or physical bytes)")
    terminal_blocks = sorted(((len(r["prompt_token_ids"]) + r["max_tokens"] - 1 + 15) // 16
                              for r in rows), reverse=True)
    print(f"worst_192_terminal_16token_blocks={sum(terminal_blocks[:192])}; "
          f"normal_usable_blocks=36764; geometric_headroom={36764 - sum(terminal_blocks[:192])} "
          "(I+O-1 terminal geometry only; native admission/reservation still applies; not observed pressure)")


if __name__ == "__main__":
    main()
