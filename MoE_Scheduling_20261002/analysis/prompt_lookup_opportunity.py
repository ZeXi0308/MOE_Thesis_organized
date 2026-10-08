"""Fixed 2/5/4 n-gram opportunity on one observed token trajectory; CPU only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

SOURCE = "https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/vllm/v1/spec_decode/ngram_proposer.py"


def propose(prefix, max_model_len):
    k = min(4, max_model_len - len(prefix))
    if k <= 0:
        return []
    for n in range(min(5, len(prefix) - 1), 1, -1):
        suffix = prefix[-n:]
        for start in range(len(prefix) - n):  # Earlier matches may overlap the suffix.
            if prefix[start:start+n] == suffix:
                return prefix[start+n:start+n+k]
    return []


def summarize(rows):
    proposed = [r for r in rows if r[0]]
    return dict(prefixes=len(rows), prefixes_with_proposal=len(proposed),
        proposal_coverage=len(proposed)/len(rows) if rows else None,
        proposal_length_distribution=dict(sorted(Counter(r[0] for r in rows).items())),
        matching_prefix_distribution_all=dict(sorted(Counter(r[1] for r in rows).items())),
        matching_prefix_distribution_given_proposal=dict(sorted(Counter(r[1] for r in proposed).items())))


def analyze(root):
    read = lambda name: json.loads((root/name).read_text())
    raw, work, engine = read("raw.json"), read("workload.json"), read("engine_args.json")
    assert raw["status"] == "COMPLETE" and len(raw["requests"]) == 16
    prompts = {r["request_id"]: p for r, p in zip(work["source_requests"], work["actual_prompt_token_ids"])}
    all_rows, native_rows, results = [], [], []
    for request in raw["requests"]:
        prompt, output = prompts[request["request_id"]], request["output_token_ids"]
        assert request["status"] == "completed" and output and len(prompt) == request["prompt_tokens"]
        assert hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest() == request["prompt_token_ids_sha256"]
        rows = []
        for i in range(len(output)):
            candidate = propose(prompt + output[:i], engine["max_model_len"])
            matched = 0
            for a, b in zip(candidate, output[i:]):
                if a != b:
                    break
                matched += 1
            rows.append((len(candidate), matched))
        position, passes = 1, 0  # Native proposer starts after the first sampled output token.
        while position < len(output):
            position += min(1 + rows[position][1], len(output) - position)
            passes += 1
        all_rows.extend(rows)
        native_rows.extend(rows[1:])
        results.append(dict(request_id=request["request_id"], output_tokens=len(output),
            native_post_sample_prefixes=summarize(rows[1:]), baseline_decode_passes=len(output)-1,
            optimistic_decode_passes=passes, optimistic_passes_including_first_output=passes+1))
    baseline = sum(r["baseline_decode_passes"] for r in results)
    optimistic = sum(r["optimistic_decode_passes"] for r in results)
    return dict(status="OFFLINE_FIXED_TRAJECTORY_OPPORTUNITY_ONLY", parameters=dict(min_ngram=2, max_ngram=5, k=4),
        source_rule=SOURCE, rule="Longest suffix n-gram; earliest prior occurrence; overlap allowed; draft clipped by context and model limit.",
        input_path=str(root.resolve()), input_sha256={n: hashlib.sha256((root/n).read_bytes()).hexdigest()
            for n in ("raw.json", "workload.json", "engine_args.json")},
        all_generation_prefixes_including_prompt_only=summarize(all_rows),
        native_post_sample_prefixes=summarize(native_rows), requests=results,
        baseline_request_local_decode_passes=baseline, optimistic_request_local_decode_passes=optimistic,
        conditional_decode_pass_reduction=baseline-optimistic,
        conditional_decode_pass_reduction_fraction=(baseline-optimistic)/baseline,
        scope="Observed strings assumed unchanged. First output remains target-generated; subsequent advances are min(1+matched, remaining). Counts sum request-local decode passes, not batched engine calls; prefill chunks, verification width/cost, scheduling, KV and proposer overhead are excluded. Not GPU speedup, measured speculative acceptance, policy trajectory, or a novelty claim.")


if __name__ == "__main__":
    assert propose([1, 2, 9, 1, 2, 8, 1, 2], 100) == [9, 1, 2, 8]  # Earliest occurrence.
    assert propose([1, 2, 9, 2, 3, 1, 2, 3, 8, 1, 2, 3], 100) == [8, 1, 2, 3]  # Longest match wins.
    assert propose([7]*6, 100) == [7] and propose([1, 2, 3], 100) == []  # Overlap / no match.
    assert propose([7]*6, 6) == []
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parents[1]
    parser.add_argument("--input", type=Path, default=base/"results_kv_recovery_r02/02_swap16")
    parser.add_argument("--output", type=Path, default=base/"analysis/prompt_lookup_opportunity.json")
    args = parser.parse_args()
    result = analyze(args.input)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("requests", "input_sha256", "scope")}))
