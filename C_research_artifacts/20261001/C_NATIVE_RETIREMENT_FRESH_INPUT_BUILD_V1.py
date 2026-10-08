#!/usr/bin/env python3
"""Build the preregistered fresh C policy-test input without GPU execution."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile


HERE = Path(__file__).resolve().parent
CORE_PATH = HERE / "20260930_c_development_burst_v2/prepare_development.py"
PLAN_PATH = HERE / "C_NATIVE_RETIREMENT_FRESH_INPUT_PLAN_20261001.md"
SCHEMA = "c-native-retirement-fresh-inputs-v1"
FIRST_RANK, LAST_RANK, COUNT = 832, 959, 128
CORE_SHA = "a1673e9a59f3640ca831b11084a906623034b97b8ed5912a4be6e6b79838e8bb"
PLAN_SHA = "984e507fa0cdc27a3da8ba68cddc8fcbe051b5f268fa9a510ac270e900074b1d"
OLD_CONFIG_SHA = "f6419ce604b9cb5aa6b047c8b8b4462e1ec58f2c08fb3eb1a13fe4d49109adde"
OLD_WORKLOAD_SHA = "89bc451017217cf9c4ca1d92805e00017624c81e5a88abc7d295357103cc94d3"
OLD_STATS_SHA = "d448b5c689166482eade715729b6fbec1a99468e67862c5f625851c6593e82ad"
PARQUET_SHA = "74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212"
ARRIVAL_SHA = "a6e1d41c88b40a336f287b77353332b5b051771de485626ac7c94ba1e1bf9fef"
MODEL_REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def compact_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def load_core():
    require(sha_file(CORE_PATH) == CORE_SHA and sha_file(PLAN_PATH) == PLAN_SHA,
            "frozen source helper or preregistered plan changed")
    spec = importlib.util.spec_from_file_location("c_fresh_pinned_v2", CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_old_inputs(folder: Path, tokenize):
    paths = {"config.json": OLD_CONFIG_SHA, "workload.json": OLD_WORKLOAD_SHA,
             "INPUT_STATS.json": OLD_STATS_SHA}
    require(all(sha_file(folder / name) == digest for name, digest in paths.items()),
            "old sustained input bytes changed")
    config = json.loads((folder / "config.json").read_text())
    workload = json.loads((folder / "workload.json").read_text())
    stats = json.loads((folder / "INPUT_STATS.json").read_text())
    rows, prompts = workload["source_requests"], workload["actual_prompt_token_ids"]
    require(config["requests"] == len(rows) == len(prompts) == COUNT
            and stats["selected_eligible_ranks"] == [704, 831]
            and config["model"]["revision"] == config["model"]["tokenizer_revision"] == MODEL_REVISION
            and config["source"]["active_source_sha256"] == PARQUET_SHA,
            "old cohort or source contract changed")
    for index, (row, ids) in enumerate(zip(rows, prompts)):
        require(row["source_index"] == index and row["eligible_source_rank"] == 704 + index
                and tokenize(row["prompt"]) == ids
                and compact_sha(ids) == row["prompt_token_ids_sha256"],
                f"old sustained tokenizer identity mismatch at rank {704 + index}")
    return config, workload, stats


def selected_articles(core, spec, source_root: Path, dataset_file: Path, tokenize):
    confirm, h_config, h_by_start, excluded, anchors = core.checked_core_sources(spec, source_root)
    h_builder = core.module_at("c_fresh_pinned_h_builder",
                               source_root / core.H_BASE_REL / "input_builder_base.py")
    dataset = core.ParquetTextRows(dataset_file)
    seen = set()
    articles = core.anchor_checked_articles(h_builder.closed_articles(dataset),
                                            tokenize, anchors, seen)
    selected, counts = confirm.select_articles(articles, tokenize, excluded,
                                               h_by_start, needed=LAST_RANK + 1)
    if len(seen) < len(anchors):
        for _ in articles:
            if len(seen) == len(anchors):
                break
    require(seen == set(anchors) and len(selected) == LAST_RANK + 1,
            "G/H anchors or 960 eligible articles were not reproduced")
    require(all(a[0] < a[1] <= b[0] for a, b in zip(selected, selected[1:])),
            "eligible article row intervals overlap or changed order")
    require(h_config["model"]["revision"] == MODEL_REVISION, "H model revision changed")
    return selected, counts, confirm, len(anchors)


def verify_old_slice(selected, old_workload, old_stats, confirm):
    previous_identity = confirm.logical_sha([[x[0], x[3], x[5]] for x in selected[:704]])
    require(previous_identity == old_stats["previous_eligible_identity_sha256"],
            "earlier eligible ranks 0..703 changed")
    for index, (row, ids) in enumerate(zip(old_workload["source_requests"],
                                           old_workload["actual_prompt_token_ids"])):
        start, end, prompt, document_sha, tokens, token_sha = selected[704 + index]
        require(row["eligible_source_rank"] == 704 + index
                and row["dataset_row_index"] == start
                and row["dataset_row_end_exclusive"] == end
                and row["prompt"] == prompt
                and row["document_sha256"] == document_sha
                and row["prompt_token_ids_sha256"] == token_sha
                and ids == tokens,
                f"old sustained article changed at rank {704 + index}")
    return confirm.logical_sha([[x[0], x[3], x[5]] for x in selected[:FIRST_RANK]])


def poisson_arrivals() -> list[float]:
    draw = random.Random(20261001)
    elapsed = 0.0
    arrivals = [0.0]
    for _ in range(COUNT - 1):
        elapsed += draw.expovariate(5.0)
        arrivals.append(round(elapsed, 6))
    require(len(arrivals) == COUNT and compact_sha(arrivals) == ARRIVAL_SHA
            and all(a <= b for a, b in zip(arrivals, arrivals[1:])),
            "single preregistered Poisson arrival draw changed")
    return arrivals


def build_inputs(selected, counts, confirm, anchor_count, old_config,
                 old_stats, previous_identity, spec, byte_checks):
    preceding, chosen = selected[:FIRST_RANK], selected[FIRST_RANK:LAST_RANK + 1]
    require(len(chosen) == COUNT, "fresh eligible rank slice incomplete")
    prior_ids = {f"memory-train-article-{x[0]:07d}" for x in preceding}
    prior_docs, prior_tokens = {x[3] for x in preceding}, {x[5] for x in preceding}
    require(len(prior_ids) == len(prior_docs) == len(prior_tokens) == FIRST_RANK,
            "previous eligible article identities are not unique")
    require(len({x[0] for x in chosen}) == len({x[3] for x in chosen})
            == len({x[5] for x in chosen}) == COUNT
            and not ({x[3] for x in chosen} & prior_docs)
            and not ({x[5] for x in chosen} & prior_tokens),
            "fresh articles duplicate an earlier eligible article")
    rows, prompts = [], []
    for index, (start, end, prompt, document_sha, ids, token_sha) in enumerate(chosen):
        request_id = f"memory-train-article-{start:07d}"
        require(request_id not in prior_ids and 256 <= len(ids) <= 3072
                and len(ids) + 1024 <= 4096,
                "fresh request identity or model token bound invalid")
        rows.append(dict(request_id=request_id, document_id=request_id,
            dataset_row_index=start, dataset_row_end_exclusive=end,
            article_title=prompt.splitlines()[0].strip(), source_index=index,
            eligible_source_rank=FIRST_RANK + index, prompt=prompt,
            document_sha256=document_sha, prompt_sha256=document_sha,
            prompt_token_count=len(ids), original_document_token_count=len(ids),
            prompt_token_ids_sha256=token_sha, max_output_tokens=1024))
        prompts.append(ids)
    arrivals = poisson_arrivals()
    workload = dict(schema=SCHEMA, source_requests=rows,
        actual_prompt_token_ids=prompts, arrival_traces_s={"poisson_v1": arrivals},
        arrival_rule="first 0; 127 cumulative Random(20261001).expovariate(5.0) gaps; round each cumulative timestamp to 6 decimals",
        output_contract=dict(max_output_tokens=1024, ignore_eos=False, min_tokens=0,
                             actual_output_lengths="UNKNOWN_UNTIL_EXECUTION"))
    lengths = [len(ids) for ids in prompts]
    old_source = old_config["source"]
    require(old_source["parquet_declaration"]["sha256"] == PARQUET_SHA
            and old_source["parquet_declaration"]["revision"] == spec["dataset"]["revision"],
            "prior Parquet declaration changed")
    source = dict(dataset_id=old_source["dataset_id"],
        dataset_config=old_source["dataset_config"], split=old_source["split"],
        dataset_revision=spec["dataset"]["revision"],
        tokenizer_files_sha256=spec["model"]["tokenizer_files_sha256"],
        selection="Next 128 eligible source-ordered complete articles, ranks 832..959, after prior224/H128 exclusions and earlier eligible ranks 0..831; no output, EOS, latency or policy-success selection",
        reconstruction_rule=old_source["reconstruction_rule"],
        active_source_kind="parquet", active_source_sha256=PARQUET_SHA,
        active_source_file="train-00000-of-00002.parquet",
        parquet_declaration=deepcopy(old_source["parquet_declaration"]),
        hf_commit_attested=False, arrow_equivalence_established=False,
        historic_receipt_lineage_complete=False,
        previous_sustained_input_sha256={"config.json": OLD_CONFIG_SHA,
                                         "workload.json": OLD_WORKLOAD_SHA,
                                         "INPUT_STATS.json": OLD_STATS_SHA},
        previous_sustained_all_128_row_text_token_ids_reproduced=True,
        previous_eligible_ranks=[0, FIRST_RANK - 1],
        previous_eligible_identity_sha256=previous_identity)
    config = dict(schema=SCHEMA, status="FRESH_INPUTS_GPU_UNRUN",
        purpose="Fresh C conditional-retirement policy test input, prepared before GPU comparison; not held-out confirmation",
        model=deepcopy(old_config["model"]), source=source, requests=COUNT,
        prompt_tokens=max(lengths), prompt_tokens_by_request=lengths,
        prompt_tokens_semantics=f"Observed full article lengths {min(lengths)}..{max(lengths)}; no padding/truncation; execution resource qualification uses the fixed 3072-token eligibility ceiling",
        max_output_tokens=1024, ignore_eos=False, min_tokens=0,
        output_tokens_semantics="Per-request cap, not known EOS or fixed output work",
        arrival_process="one preregistered Poisson draw at mean 5 requests/s",
        arrival_regime="poisson_v1", arrival_span_s=arrivals[-1],
        arrival_seed=20261001, arrival_rate_per_s=5.0,
        seed=20260905, workload_sha256=confirm.logical_sha(workload),
        requirements_sha256=byte_checks["requirements_sha256"],
        preregistered_plan_sha256=PLAN_SHA,
        input_preparation="Eligible ranks 832..959 after pinned prior224/H128 exclusion; all 192 G/H anchors, old 704..831 article/token IDs, and prior 0..703 identities reproduced",
        runtime_status="Input only; no GPU execution or runtime qualification",
        source_lineage_status="PARTIAL_HISTORIC_RECEIPTS_DECLARED",
        claim_ceiling="New relative to recorded C policy selection only; six historic receipt bytes unavailable and Parquet/old-Arrow whole-shard equivalence unestablished; no held-out or GPU confirmation")
    stats = dict(schema=SCHEMA, status="FRESH_INPUTS_GPU_UNRUN",
        selected_eligible_ranks=[FIRST_RANK, LAST_RANK],
        previous_eligible_ranks=[0, FIRST_RANK - 1],
        previous_eligible_identity_sha256=previous_identity,
        old_sustained_previous_704_identity_sha256=old_stats["previous_eligible_identity_sha256"],
        old_sustained_all_128_row_text_token_ids_checked=True,
        prompt_tokens_min=min(lengths), prompt_tokens_max=max(lengths),
        first_selected_row=chosen[0][0],
        last_selected_row_end_exclusive=chosen[-1][1],
        arrival_trace_compact_json_sha256=ARRIVAL_SHA,
        arrival_span_s=arrivals[-1], arrival_seed=20261001,
        arrival_rate_per_s=5.0, gh_anchor_count=anchor_count,
        examined_article_counts=counts, source_byte_checks=byte_checks,
        limitations=config["claim_ceiling"], generator_sha256=sha_file(Path(__file__)))
    return config, workload, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset-file", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--old-input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(args.output_dir.parent.is_dir()
            and not args.output_dir.exists() and not args.output_dir.is_symlink(),
            "output directory must be new under an existing directory")
    core = load_core()
    spec = core.requirements()
    require(args.dataset_file.name == "train-00000-of-00002.parquet"
            and sha_file(args.dataset_file) == PARQUET_SHA,
            "preregistered Parquet bytes changed")
    tokenizer_checks = {name: sha_file(args.tokenizer_dir / name)
                        for name in spec["model"]["tokenizer_files_sha256"]}
    require(tokenizer_checks == spec["model"]["tokenizer_files_sha256"],
            "pinned tokenizer/model config bytes changed")
    from tokenizers import Tokenizer, __version__ as tokenizers_version
    import pyarrow
    tokenizer = Tokenizer.from_file(str(args.tokenizer_dir / "tokenizer.json"))
    tokenize = lambda text: tokenizer.encode(text, add_special_tokens=True).ids
    old_config, old_workload, old_stats = load_old_inputs(args.old_input_dir, tokenize)
    selected, counts, confirm, anchor_count = selected_articles(
        core, spec, args.source_root, args.dataset_file, tokenize)
    previous_identity = verify_old_slice(selected, old_workload, old_stats, confirm)
    byte_checks = dict(status="LOCAL_BYTES_AND_ANCHORS_CHECKED",
        requirements_sha256=sha_file(core.REQUIREMENTS),
        source_core_files_sha256=spec["source_root_files_sha256"],
        parquet_sha256=PARQUET_SHA, tokenizer_files_sha256=tokenizer_checks,
        old_sustained_input_sha256={"config.json": OLD_CONFIG_SHA,
                                    "workload.json": OLD_WORKLOAD_SHA,
                                    "INPUT_STATS.json": OLD_STATS_SHA},
        source_helper_sha256=CORE_SHA, preregistered_plan_sha256=PLAN_SHA,
        tokenizers_version=tokenizers_version, pyarrow_version=pyarrow.__version__,
        python_version=sys.version.split()[0],
        scope="Local pinned bytes, 192 G/H anchors, old 128 exact token IDs; no HF commit attestation, complete historic receipts, or Arrow/Parquet whole-shard equality")
    config, workload, stats = build_inputs(selected, counts, confirm, anchor_count,
        old_config, old_stats, previous_identity, spec, byte_checks)
    stage = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.",
                                  dir=args.output_dir.parent))
    try:
        file_hashes = {}
        for name, value in (("config.json", config), ("workload.json", workload)):
            path = stage / name
            path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                       allow_nan=False) + "\n")
            file_hashes[name] = sha_file(path)
        stats["files_sha256"] = dict(file_hashes)
        (stage / "INPUT_STATS.json").write_text(json.dumps(
            stats, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(stage, args.output_dir)
    except BaseException:
        shutil.rmtree(stage)
        raise
    file_hashes["INPUT_STATS.json"] = sha_file(args.output_dir / "INPUT_STATS.json")
    print(json.dumps(dict(status=config["status"], output_dir=str(args.output_dir),
        source_ranks=[FIRST_RANK, LAST_RANK], requests=COUNT,
        prompt_tokens_min=stats["prompt_tokens_min"],
        prompt_tokens_max=stats["prompt_tokens_max"],
        arrival_span_s=stats["arrival_span_s"],
        arrival_sha256=ARRIVAL_SHA, files_sha256=file_hashes), indent=2))


if __name__ == "__main__":
    main()
