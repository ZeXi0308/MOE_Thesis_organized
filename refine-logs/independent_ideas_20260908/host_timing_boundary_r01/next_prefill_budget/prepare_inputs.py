"""Derive paired natural-article token prefixes offline; refuse existing outputs."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def token_sha(ids):
    return sha(json.dumps(ids, separators=(",", ":")).encode())


def write_new(path, data):
    with path.open("x") as stream:
        stream.write(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=repo / "refine-logs/expert_saturation/outputs/admission_capacity/20260906_native_memory_pressure_r01/inputs_preparation/prepared")
    parser.add_argument("--output", type=Path, default=here / "prepared")
    args = parser.parse_args()
    sources = {f"{cohort}/{name}": args.source / cohort / name for cohort in ("short", "long") for name in ("config.json", "workload.json")}
    hashes = {name: sha(path.read_bytes()) for name, path in sources.items()}
    data = {name: json.loads(path.read_text()) for name, path in sources.items()}
    short, long = data["short/workload.json"], data["long/workload.json"]
    model = data["long/config.json"]["model"]
    assert model == data["short/config.json"]["model"]
    for i in range(16):
        a, b = short["source_requests"][i], long["source_requests"][i]
        s, t = short["actual_prompt_token_ids"][i], long["actual_prompt_token_ids"][i]
        assert all(a[k] == b[k] for k in ("request_id", "document_id", "document_sha256", "prompt", "dataset_row_index", "dataset_row_end_exclusive"))
        assert sha(a["prompt"].encode()) == a["document_sha256"]
        assert len(s) == 128 and len(t) == 3072 and s == t[:128]
        assert token_sha(s) == a["prompt_token_ids_sha256"] and token_sha(t) == b["prompt_token_ids_sha256"]
    assert len({r["document_sha256"] for r in long["source_requests"][:16]}) == 16
    args.output.mkdir(parents=True, exist_ok=False)
    for cohort in ("mixed", "all_short"):
        lengths = [2048 if cohort == "mixed" and i % 2 else 128 for i in range(16)]
        tokens = [long["actual_prompt_token_ids"][i][:n] for i, n in enumerate(lengths)]
        requests = []
        for i, ids in enumerate(tokens):
            original = long["source_requests"][i]
            request = {k: v for k, v in original.items() if k not in ("prompt", "prompt_sha256", "prompt_token_count", "prompt_token_ids_sha256")}
            request.update(source_index=i, prompt_token_count=len(ids), output_tokens=128, prompt_token_ids_sha256=token_sha(ids), source_short_prompt_token_ids_sha256=short["source_requests"][i]["prompt_token_ids_sha256"], source_long_prompt_token_ids_sha256=original["prompt_token_ids_sha256"])
            requests.append(request)
        workload = dict(schema="olmoe-admission-inputs-v1", source_requests=requests, actual_prompt_token_ids=tokens, arrival_traces_s={"steady": [i / 20 for i in range(16)]}, arrival_rule="steady: source index i arrives at i*0.05 seconds", document_identity_scope="same first 16 source articles in both cohorts; reused inputs, not a fresh holdout")
        config = dict(status="GPU_UNRUN", model=model, requests=16, prompt_tokens=128 if cohort == "all_short" else None, prompt_tokens_by_request=lengths, output_tokens=128, seed=20260905, arrival_gap_s=0.05, cohort=cohort, source=data["long/config.json"]["source"], source_root=str(args.source), sources_sha256=hashes, workload_sha256=sha(json.dumps(workload, sort_keys=True).encode()), token_input_rule="Actual token IDs are authoritative; truncate existing single-article prefixes only; no concatenation, replication, padding, or re-tokenization", token_hash_rule="SHA256 of compact JSON token ID array using separators=(',', ':')", cohort_rule="first 16 source articles; mixed: zero-based even index 128, odd index 2048; all_short: all 128", input_preparation="offline only; no GPU execution or performance measurement; runner declares SLO and engine settings")
        folder = args.output / cohort
        folder.mkdir()
        write_new(folder / "workload.json", workload)
        write_new(folder / "config.json", config)
    print(json.dumps(dict(status="INPUTS_PREPARED_GPU_UNRUN", output=str(args.output), cohorts=["mixed", "all_short"], requests_per_cohort=16)))
