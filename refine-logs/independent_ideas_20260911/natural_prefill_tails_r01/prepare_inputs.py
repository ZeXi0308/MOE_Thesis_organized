#!/usr/bin/env python3
"""Prepare 16 complete natural-length articles offline, without GPU execution."""
import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
TITLE = re.compile(r" = [^=\n]+ = \n")
EXPECTED = [(271, 790), (287, 1572), (406, 1846), (433, 1117),
            (464, 733), (557, 2602), (610, 1521), (748, 3011),
            (913, 2663), (944, 1739), (1038, 2546), (1380, 1012),
            (1582, 2736), (1633, 1379), (1648, 2404), (1690, 1432)]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-arrow", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "prepared/natural")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refuse existing prepared inputs: {args.output_dir}")
    from datasets import Dataset
    from transformers import AutoTokenizer
    from transformers.utils.hub import cached_file
    prior_dir = HERE.parent / "per_request_prefill_share_r01/prepared/mixed"
    base = json.loads((prior_dir / "config.json").read_text())
    prior = json.loads((prior_dir / "workload.json").read_text())
    assert sha(json.dumps(prior, sort_keys=True).encode()) == base["workload_sha256"]
    excluded = {r["document_id"] for r in prior["source_requests"]}
    assert len(excluded) == 16
    assert (base["requests"], base["seed"], base["arrival_gap_s"], base["output_tokens"]) == (16, 20260905, .05, 128)
    model, source = base["model"], base["source"]
    assert model["revision"] == model["tokenizer_revision"] == REVISION
    assert args.dataset_arrow.name == "wikitext-train-00000-of-00002.arrow"
    assert args.dataset_arrow.parent.name == DATASET_REVISION
    assert args.dataset_arrow.parents[2].name == "wikitext-103-raw-v1"
    assert file_sha(args.dataset_arrow) == source["arrow_sha256"]
    config_path = Path(cached_file(model["id"], "config.json", revision=REVISION, local_files_only=True))
    assert file_sha(config_path) == source["model_config_sha256"]
    assert json.loads(config_path.read_text())["max_position_embeddings"] == 4096
    for name, expected in source["tokenizer_files_sha256"].items():
        path = Path(cached_file(model["id"], name, revision=REVISION, local_files_only=True))
        assert file_sha(path) == expected
    tokenizer = AutoTokenizer.from_pretrained(model["id"], revision=REVISION, local_files_only=True)
    dataset = Dataset.from_file(str(args.dataset_arrow))
    records, prompts, start, pieces, previous = [], [], None, [], ""
    for index, row in enumerate(dataset):
        text = str(row["text"])
        is_title = (bool(TITLE.fullmatch(text)) and not previous.strip()
                    and index + 1 < len(dataset) and not str(dataset[index + 1]["text"]).strip())
        if is_title:
            if start is not None:
                rid = f"memory-train-article-{start:07d}"
                if rid not in excluded:
                    document = "".join(pieces)
                    ids = tokenizer(document, add_special_tokens=True, truncation=False, padding=False)["input_ids"]
                    if 1 <= len(ids) <= 3968:
                        document_sha = sha(document.encode())
                        records.append(dict(request_id=rid, document_id=rid, sample_id=start,
                            source_index=len(records), dataset_row_index=start, dataset_row_end_exclusive=index,
                            article_title=pieces[0].strip(), document_sha256=document_sha, prompt=document,
                            prompt_sha256=document_sha, prompt_token_count=len(ids), output_tokens=128,
                            prompt_token_ids_sha256=sha(json.dumps(ids, separators=(",", ":")).encode()),
                            original_document_token_count=len(ids)))
                        prompts.append(ids)
                        if len(records) == 16:
                            break
            start, pieces = index, []
        if start is not None:
            pieces.append(text)
        previous = text
    # Only the following valid title closes an article; never flush an open shard tail.
    assert [(r["dataset_row_index"], r["prompt_token_count"]) for r in records] == EXPECTED
    lengths = [len(ids) for ids in prompts]
    assert sum(lengths) == 29103 and max(lengths) + 128 <= 4096
    assert len({r["document_sha256"] for r in records}) == 16
    assert not excluded.intersection(r["document_id"] for r in records)
    identity = "16 distinct article IDs, excluding the previous campaign's 16 IDs; no broader holdout claim"
    provenance = {k: source[k] for k in ("dataset_id", "dataset_config", "split", "dataset_revision",
        "arrow_file", "arrow_sha256", "model_config_sha256", "model_max_position_embeddings", "tokenizer_files_sha256")}
    provenance.update(observed_dataset_fingerprint=str(dataset._fingerprint), excluded_document_ids=sorted(excluded),
        prior_inputs_sha256={str((prior_dir / name).relative_to(REPO)): file_sha(prior_dir / name)
                             for name in ("config.json", "workload.json")},
        selection_rule="first 16 complete articles in source row order, excluding prior IDs, with 1..3968 full tokens",
        reconstruction_rule="exact raw title row ' = title = \\n' with no equals inside title and empty neighboring rows; concatenate original rows unchanged until next valid title; exclude unclosed shard tail",
        identity_scope=identity)
    workload = dict(schema="olmoe-admission-inputs-v1", source_requests=records, actual_prompt_token_ids=prompts,
        arrival_traces_s={"steady": [i * .05 for i in range(16)]},
        arrival_rule="steady: source index i arrives at i*0.05 seconds", document_identity_scope=identity)
    config = dict(status="GPU_UNRUN", model=model, requests=16, prompt_tokens=None,
        prompt_tokens_by_request=lengths, output_tokens=128, seed=base["seed"], arrival_gap_s=.05,
        max_model_len=4096, cohort="natural", source=provenance,
        workload_sha256=sha(json.dumps(workload, sort_keys=True).encode()),
        token_input_rule="Full original-article token IDs; add_special_tokens=True, truncation=False, padding=False",
        token_hash_rule="SHA256 of compact JSON token ID array using separators=(',', ':')",
        input_preparation="offline CPU tokenizer only; source-only selection; no GPU or performance measurement")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, data in (("config.json", config), ("workload.json", workload)):
        with (args.output_dir / name).open("x") as stream:
            stream.write(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(dict(status="INPUTS_PREPARED_GPU_UNRUN", requests=16, total_prompt_tokens=sum(lengths),
                         prompt_tokens_by_request=lengths, output_dir=str(args.output_dir))))


if __name__ == "__main__":
    main()
