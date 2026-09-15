#!/usr/bin/env python3
"""Offline fresh rows after the old manifest; refuse to overwrite prepared data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
sys.path.insert(0, str(ROOT / "docs/ideas/bcrd/experiments"))
import build_continuous_workloads as builder
import datasets
import transformers


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def dump(path, value):
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=HERE)
parser.add_argument("--dataset-arrow", type=Path, default=Path.home() / ".cache/huggingface/datasets/wikitext/wikitext-103-raw-v1/0.0.0" / builder.DATASET_REVISION / "wikitext-test.arrow")
args = parser.parse_args()
out = args.output_dir
assert not any((out / name).exists() for name in ("config.json", "workload.json", "provenance.json")), "prepared output already exists"
formal_path = ROOT / "docs/ideas/bcrd/experiments/configs/workloads/olmoe.formal.json"
formal = json.loads(formal_path.read_text())
prior = json.loads((HERE / "prior_identities.json").read_text())
old_docs, old_tokens = set(prior["prompt_sha256"]), set(prior["first128_token_ids_sha256"])
assert builder.sha256_file(args.dataset_arrow) == builder.DATASET_ARROW_SHA256 == formal["dataset"]["arrow_sha256"]
data = datasets.Dataset.from_file(str(args.dataset_arrow))
tokenizer, tokenizer_contract = builder.tokenizer_contract(formal["model"])
assert tokenizer_contract["files"] == formal["tokenizer"]["files"], "pinned tokenizer bytes changed"
nonempty = [i for i, row in enumerate(data) if row["text"].strip()]
assert nonempty[:128] == [r["dataset_row_index"] for r in formal["requests"]]
for row in formal["requests"]:
    text = data[row["dataset_row_index"]]["text"]
    assert text == row["prompt"] and hashlib.sha256(text.encode()).hexdigest() == row["prompt_sha256"]
    ids = tokenizer(text, add_special_tokens=True, truncation=True, max_length=128)["input_ids"]
    assert digest(ids) == row["prompt_token_ids_sha256"], "legacy prefix identity drift"
start = max(r["dataset_row_index"] for r in formal["requests"]) + 1
rows, prompts, seen_docs, seen_tokens, skipped = [], [], set(old_docs), set(old_tokens), []
for index in range(start, len(data)):
    text = data[index]["text"]
    sha = hashlib.sha256(text.encode()).hexdigest()
    ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"] if text.strip() else []
    prefix, token_sha = ids[:128], digest(ids[:128])
    reason = "short_or_empty" if len(ids) < 128 else "duplicate_prompt" if sha in seen_docs or token_sha in seen_tokens else None
    if reason:
        skipped.append(dict(dataset_row_index=index, reason=reason))
        continue
    rows.append(dict(request_id=f"fresh-native-row-{index:06d}", sample_id=len(rows),
        dataset_row_index=index, document_id=f"wikitext-test-row-{index:06d}-{sha[:16]}",
        document_sha256=sha, prompt=text, prompt_sha256=sha, prompt_token_count=128,
        untruncated_prompt_token_count=len(ids), prompt_token_ids_sha256=token_sha))
    prompts.append(prefix)
    seen_docs.add(sha)
    seen_tokens.add(token_sha)
    if len(rows) == 16:
        break
assert len(rows) == len(prompts) == len({r["request_id"] for r in rows}) == 16
base = ROOT / "refine-logs/expert_saturation/outputs/admission_capacity/20260905_pre_gpu_r01/prepared"
original = json.loads((base / "config.json").read_text())
arrivals = json.loads((base / "workload.json").read_text())["arrival_traces_s"]
assert arrivals == {"steady": [i * 0.1 for i in range(16)], "bursty": [(i // 4) / 3 * 15 * 0.1 for i in range(16)]}
workload = dict(schema="olmoe-admission-inputs-v1", source_requests=rows,
    actual_prompt_token_ids=prompts, arrival_traces_s=arrivals,
    arrival_rule="synthetic steady gap 0.1 s; groups of 4 at 0/0.5/1/1.5 s; inherited unchanged; native runner later applies scale 0.02",
    document_identity_scope="fresh WikiText rows and first128 token prefixes versus frozen prior admission prompts; not certified article-disjoint")
config = {key: original[key] for key in ("model", "requests", "prompt_tokens", "output_tokens", "seed", "ttft_slo_s", "tpot_slo_s")}
config.update(workload_sha256=hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest(),
    prepare_only=True, natural_stop=False, evidence_ceiling="CPU_INPUT_PREPARATION_GPU_UNRUN")
provenance = dict(dataset={**formal["dataset"], "selection_rule": "first16 rows after old manifest last row, with >=128 tokens and unseen full-text AND first128-token hashes"},
    arrow_path=str(args.dataset_arrow), observed_dataset_fingerprint=str(data._fingerprint),
    fingerprint_matches_legacy=str(data._fingerprint) == formal["dataset"]["fingerprint"],
    identity_check="frozen Arrow SHA and tokenizer files matched; old128 raw texts and token hashes reproduced; fingerprint difference retained, cause not established",
    old_manifest_last_row=start - 1, selected_dataset_row_indices=[r["dataset_row_index"] for r in rows], skipped_rows=skipped,
    prior_identities_sha256=builder.sha256_file(HERE / "prior_identities.json"), prior_unique_prompts=len(old_docs),
    tokenizer=tokenizer_contract, versions=dict(python=sys.version, datasets=datasets.__version__, transformers=transformers.__version__),
    arrivals=dict(kind="synthetic", arrival_gap_s=0.1, burst_size=4, inherited_from=str(base.relative_to(ROOT) / "workload.json"),
        native_arrival_scale_planned=0.02, burstgpt_source_used=False), gpu_status="UNRUN")
for row, ids in zip(rows, prompts):
    assert len(ids) == 128 and digest(ids) == row["prompt_token_ids_sha256"] and digest(ids) not in old_tokens
    assert row["prompt_sha256"] not in old_docs and all(type(t) is int and t >= 0 for t in ids)
out.mkdir(parents=True, exist_ok=True)
dump(out / "workload.json", workload)
dump(out / "config.json", config)
dump(out / "provenance.json", provenance)
with (out / "commands.txt").open("x") as handle:
    handle.write(shlex.join([sys.executable, *sys.argv]) + "\n")
print(json.dumps(dict(status="CPU_PREPARED_GPU_UNRUN", requests=16, selected_rows=provenance["selected_dataset_row_indices"],
    workload_sha256=config["workload_sha256"], old_prompt_overlap=0, legacy128_identity_checks="PASS")))
