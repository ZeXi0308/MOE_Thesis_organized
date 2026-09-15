#!/usr/bin/env python3
"""Prepare eligible articles 33-64 offline; input pool only, no GPU protocol."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[7]
HERE = Path(__file__).resolve().parent
B = HERE.parents[1]
OLD = ROOT / "refine-logs/expert_saturation/outputs/admission_capacity/20260906_native_memory_pressure_r01/inputs_preparation"
spec = importlib.util.spec_from_file_location("original_preparation", OLD / "prepare_inputs.py")
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-arrow", required=True, type=Path)
    args = parser.parse_args()
    assert not any((HERE / n).exists() for n in ("workload.json", "manifest.json"))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    import datasets
    import transformers
    import tokenizers
    from datasets import Dataset
    from transformers import AutoTokenizer
    from transformers.utils.hub import cached_file

    prior = read(OLD / "prepared/short/workload.json")
    provenance = read(OLD / "prepared/inputs_report.json")["provenance"]
    model = read(OLD / "prepared/short/config.json")["model"]
    assert model["revision"] == model["tokenizer_revision"] == original.REVISION
    arrow = args.dataset_arrow.resolve()
    assert arrow.name == "wikitext-train-00000-of-00002.arrow"
    assert arrow.parent.name == original.DATASET_REVISION
    assert arrow.parents[2].name == "wikitext-103-raw-v1"
    arrow_hash = original.file_sha(arrow)
    assert arrow_hash == provenance["arrow_sha256"]
    tokenizer_manifest = read(ROOT / "docs/ideas/bcrd/experiments/configs/workloads/olmoe.formal.json")["tokenizer"]
    tokenizer_files = {}
    for name, expected in tokenizer_manifest["files"].items():
        path = Path(cached_file(model["id"], name, revision=original.REVISION, local_files_only=True))
        actual = original.file_sha(path)
        assert actual == expected == provenance["tokenizer_files_sha256"][name]
        tokenizer_files[name] = dict(path=str(path), sha256=actual)
    tokenizer = AutoTokenizer.from_pretrained(model["id"], revision=original.REVISION, local_files_only=True)
    assert tokenizer.__class__.__name__ == tokenizer_manifest["class"]
    dataset = Dataset.from_file(str(arrow))
    # Exact original top-level-title recognition and verbatim row concatenation.
    eligible, start, pieces = [], None, []
    for index, row in enumerate(dataset):
        text = str(row["text"])
        if re.fullmatch(r"= [^=].*? =", text.strip()):
            if start is not None:
                document = "".join(pieces)
                ids = tokenizer(document, add_special_tokens=True, truncation=False, padding=False)["input_ids"]
                if len(ids) >= 3072:
                    eligible.append(dict(start=start, stop=index, document=document, ids=ids,
                                         title=pieces[0].strip(), document_sha256=original.sha(document.encode())))
                    if len(eligible) == 64:
                        break
            start, pieces = index, []
        if start is not None:
            pieces.append(text)
    # Do not admit a shard's unterminated final article.
    assert len(eligible) == 64 and len(prior["source_requests"]) == 32
    for article, record, prefix in zip(eligible[:32], prior["source_requests"], prior["actual_prompt_token_ids"]):
        assert article["start"] == record["dataset_row_index"]
        assert article["stop"] == record["dataset_row_end_exclusive"]
        assert article["document"] == record["prompt"]
        assert article["document_sha256"] == record["document_sha256"]
        assert article["ids"][:128] == prefix
    records, prompts = [], []
    for ordinal, article in enumerate(eligible[32:], 33):
        rid = "memory-train-article-%07d" % article["start"]
        prefix = article["ids"][:128]
        records.append(dict(request_id=rid, document_id=rid, sample_id=article["start"],
            eligible_article_ordinal_1based=ordinal, dataset_row_index=article["start"],
            dataset_row_end_exclusive=article["stop"], article_title=article["title"],
            prompt=article["document"], document_sha256=article["document_sha256"],
            prompt_sha256=article["document_sha256"], original_document_token_count=len(article["ids"]),
            prompt_token_count=128, prompt_token_ids_sha256=original.token_sha(prefix)))
        prompts.append(prefix)
    new_ids = {r["document_id"] for r in records}
    new_hashes = {r["document_sha256"] for r in records}
    assert len(new_ids) == len(new_hashes) == len({original.token_sha(p) for p in prompts}) == 32
    assert not new_ids & {r["document_id"] for r in prior["source_requests"]}
    assert not new_hashes & {r["document_sha256"] for r in prior["source_requests"]}
    assert not {original.token_sha(p) for p in prompts} & {original.token_sha(p) for p in prior["actual_prompt_token_ids"]}
    references = [
        "shared_pool_qualification/prepared/workload.json",
        "shared_pool_qualification/revision02/prepared/workload.json",
        "shared_pool_lifecycle_qualification/prepared/workload.json",
        "shared_pool_lifecycle_qualification/revision02/prepared/workload.json",
        "shared_pool_performance/prepared/block0/workload.json",
        "shared_pool_performance/prepared/block1/workload.json",
        "full_stage_qualification/prepared/workload.json",
        "full_stage_performance/prepared/block0/workload.json",
        "full_stage_performance/prepared/block1/workload.json",
    ]
    prior_inputs = []
    for name in references:
        path = B / name
        used = read(path)["source_requests"]
        assert not new_ids & {r["document_id"] for r in used}
        assert not new_hashes & {r.get("document_sha256", r.get("prompt_sha256")) for r in used}
        prior_inputs.append(dict(path=name, sha256=original.file_sha(path), requests=len(used), overlap=0))
    scope = "Source-disjoint from original short32 and named X/F prepared inputs; same train shard and deterministic source-order selection, not independent GPU samples or a measured fresh holdout. Arrival schedule, output length, GPU matrix and performance verdict remain unspecified."
    workload = dict(schema="olmoe-admission-inputs-v1", source_requests=records,
                    actual_prompt_token_ids=prompts, input_pool_only=True, document_identity_scope=scope)
    workload_bytes = (json.dumps(workload, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    manifest = dict(status="INPUT_POOL_PREPARED_GPU_UNRUN", requests=32, prefix_tokens=128,
        selection_rule="First 64 complete articles eligible at >=3072 pinned-tokenizer tokens; reproduce original first32 exactly, then select eligible ordinal33..64 in source row order. No outcome/performance selection.",
        reconstruction_rule="Exact original re.fullmatch(r'= [^=].*? =', text.strip()); concatenate original row strings with no added separator, from one top-level title up to the next (exclusive). Exclude unterminated shard tail.",
        original_preparation=dict(path=str((OLD / 'prepare_inputs.py').relative_to(ROOT)), sha256=original.file_sha(OLD / 'prepare_inputs.py')),
        dataset=dict(id="wikitext", config="wikitext-103-raw-v1", split="train", revision=original.DATASET_REVISION,
                     arrow_path=str(arrow), arrow_sha256=arrow_hash, fingerprint=str(dataset._fingerprint)),
        tokenizer=dict(model_id=model["id"], revision=original.REVISION, tokenizer_class=tokenizer.__class__.__name__,
                       add_special_tokens=True, truncation=False, padding=False, files=tokenizer_files),
        preparation_environment=dict(python=sys.version, executable=sys.executable, datasets=datasets.__version__,
                                     transformers=transformers.__version__, tokenizers=tokenizers.__version__),
        selected_articles=[{k:r[k] for k in ("document_id", "eligible_article_ordinal_1based", "dataset_row_index", "dataset_row_end_exclusive", "article_title", "document_sha256", "original_document_token_count", "prompt_token_ids_sha256")} for r in records],
        original_short32=dict(path=str((OLD / 'prepared/short/workload.json').relative_to(ROOT)), sha256=original.file_sha(OLD / 'prepared/short/workload.json'), exact_reconstruction=True, document_id_overlap=0, document_hash_overlap=0, prefix_hash_overlap=0),
        mechanism_prepared_inputs=prior_inputs, identity_scope=scope,
        workload_sha256=original.sha(workload_bytes), script_sha256=original.file_sha(Path(__file__)),
        command=".venv/bin/python " + str(Path(__file__).relative_to(ROOT)) + " --dataset-arrow " + str(arrow))
    with (HERE / "workload.json").open("xb") as stream:
        stream.write(workload_bytes)
    with (HERE / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=manifest["status"], requests=32,
        first=records[0]["dataset_row_index"], last_end_exclusive=records[-1]["dataset_row_end_exclusive"],
        original_tokens_range=[min(r["original_document_token_count"] for r in records), max(r["original_document_token_count"] for r in records)],
        original_short32_overlap=0, mechanism_prepared_input_files=len(prior_inputs), workload_bytes=len(workload_bytes))))


if __name__ == "__main__":
    main()
