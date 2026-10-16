#!/usr/bin/env python3
"""Build six common 128-record workloads from the existing CPU scan candidates.

From the project root: python code/build_common_aux_workloads.py
No tokenizer/model loading, downloads, generation, truncation or overwrites.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("olmoe", "qwen3_awq", "granite3b")
CAPS = {"lcc": 3072, "multi_news": 3584}
COUNT, DEVELOPMENT, CONTEXT, OUTPUT, DRAFT = 128, 64, 4096, 128, 7
SELECTION = ("First 128 original-order source indices eligible under all three target tokenizers; "
             "LCC cap 3072, MultiNews cap 3584. Complete saved prompts are copied unchanged. "
             "No truncation, duplication, shuffle, reference-based or outcome-based selection.")
SCAN_OVERLAP = ("The source CPU scan ran while the static6 campaign was in progress. "
                "Its recorded single-CPU affinity and nice19 do not establish zero CPU interference; "
                "the original scan time window is preserved, without subtracting any inferred cost.")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def encoded(obj):
    return (json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def candidate_records(folder, entry, report, task, family):
    path = (folder / entry["path"]).resolve()
    require(path.is_relative_to(folder), "candidate path leaves the report directory")
    require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"],
            f"candidate hash/size mismatch: {path.name}")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    model = report["models"][family]
    require(data["source_sha256"] == report["source_sha256"], "candidate/source digest mismatch")
    require(data["model"] == data["tokenizer"] == model["target"], "candidate tokenizer identity mismatch")
    require(data["tokenizer_sha256"] == model["target_tokenizer_fingerprints"], "tokenizer fingerprints differ")
    require(data["chat_template"] == model["target_chat_template"], "chat template differs from scan")
    require(bool(data["chat_template"]) == (family != "granite3b"), "unexpected rendering mode")
    require(data["prompt_template"] == report["content_templates"][task], "content template changed")
    require(data["rendering"] == model["rendering"], "rendering semantics changed")
    require(data["prompt_range"] == [1, entry["prompt_limit"]] and entry["prompt_limit"] >= CAPS[task],
            "saved candidate limit cannot support the fixed selection cap")
    require(data["cpu_only"] is True and data["generation_performed"] is False, "unexpected candidate origin")
    for config in (model["target_config"], model["draft_config"]):
        require(config["max_position_embeddings"] >= CONTEXT, "target/draft cannot support 4096 context")
    require(len(data["records"]) == entry["records"], "candidate count differs from report")
    prefix, suffix = data["prompt_template"].split("{context}")
    records, source_ids, previous = {}, set(), -1
    vocab_limit = min(model["target_config"]["vocab_size"], model["draft_config"]["vocab_size"])
    for row in data["records"]:
        index, source_id, ids = row["source_index"], row["source_id"], row["prompt_token_ids"]
        require(type(index) is int and index > previous, "candidate indices are duplicated or out of original order")
        require(isinstance(source_id, str) and source_id not in source_ids, "duplicate/invalid source ID")
        require(row["task"] == task and row["source_member"] == data["source_member"], "record task/source mismatch")
        require(isinstance(ids, list) and 1 <= len(ids) <= entry["prompt_limit"] and
                all(type(i) is int and 0 <= i < vocab_limit for i in ids), "invalid token IDs/record length")
        if family == "olmoe":
            require(max(ids) < 50280, "OLMo target-only padding token appears in prompt content")
        prompt = row["prompt"]
        require(isinstance(prompt, str) and prompt.startswith(prefix) and prompt.endswith(suffix),
                "record does not preserve the complete content template")
        context = prompt[len(prefix):len(prompt)-len(suffix) if suffix else None]
        require(hashlib.sha256(context.encode()).hexdigest() == row["context_sha256"], "context digest mismatch")
        refs = row["references"]
        require(isinstance(refs, list) and refs and all(isinstance(x, str) for x in refs)
                and row["reference"] == refs[0], "reference schema differs from existing preparer")
        require(isinstance(row["rendered_sha256"], str) and len(row["rendered_sha256"]) == 64,
                "rendered prompt provenance missing")
        previous = index
        source_ids.add(source_id)
        records[index] = row
    return data, records, path


def build(folder, config_dir, manifest_path):
    folder, config_dir, manifest_path = (Path(p).resolve() for p in (folder, config_dir, manifest_path))
    for path in (folder, config_dir, manifest_path):
        require(path.is_relative_to(ROOT), "all builder inputs/outputs must remain in this project")
    outputs = {(task, family):config_dir/f"{task}_{family}_common128.json" for task in CAPS for family in FAMILIES}
    for path in [*outputs.values(), manifest_path]:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    report_path = folder / "feasibility.json"
    report, execution = read(report_path), read(folder / "cpu_execution.json")
    require(report["status"] == "complete_cpu_feasibility" and execution["returncode"] == 0, "scan incomplete")
    require(report["cpu_only"] is True and report["generation_performed"] is False, "unexpected feasibility origin")
    require(sha(folder/"scan.py") == report["scan_sha256"], "scan source digest mismatch")
    aux = read(ROOT/"configs/aux_workloads.json")
    require(sha(ROOT/"configs/aux_workloads.json") == report["config_sha256"], "source template config changed")
    require(aux["source"]["sha256"] == report["source_sha256"], "pinned archive identity changed")
    entries = {(x["task"],x["family"]):x for x in report["tokenized_candidates"]}
    require(len(entries) == len(report["tokenized_candidates"]) == 6 and set(entries) == set(outputs),
            "expected exactly the six target/task candidate files")
    loaded = {key:candidate_records(folder, entry, report, *key) for key,entry in entries.items()}
    populations, documents, artifacts = {}, {}, []
    builder_sha = sha(__file__)
    provenance = {"feasibility":relative(report_path),"feasibility_sha256":sha(report_path),
        "scan_source_sha256":report["scan_sha256"],"cpu_execution":execution,
        "scan_overlap_boundary":SCAN_OVERLAP,"builder":relative(__file__),"builder_sha256":builder_sha}
    for task, cap in CAPS.items():
        eligible = {family:{i for i,r in loaded[task,family][1].items() if len(r["prompt_token_ids"]) <= cap}
                    for family in FAMILIES}
        common = sorted(set.intersection(*(eligible[f] for f in FAMILIES)))
        require(len(common) >= COUNT, f"{task}: fewer than 128 common complete inputs")
        selected = common[:COUNT]
        expected = [r for r in report["cross_tokenizer_common_population"]
                    if r["task"] == task and r["prompt_limit"] == cap]
        require(len(expected) == 1 and expected[0]["eligible_all_three_tokenizers"] == len(common)
                and expected[0]["first_128_common_source_indices"] == selected, "common population differs from scan")
        for index in common:
            base = loaded[task,FAMILIES[0]][1][index]
            for family in FAMILIES[1:]:
                row = loaded[task,family][1][index]
                require(all(row[k] == base[k] for k in ("source_id","prompt","context_sha256","reference","references")),
                        "same-index target candidates disagree on source content/reference identity")
        common_ids = [loaded[task,FAMILIES[0]][1][i]["source_id"] for i in selected]
        split = {"development":{"start_offset":0,"end_offset_exclusive":DEVELOPMENT,
                                 "source_indices":selected[:DEVELOPMENT],"source_ids":common_ids[:DEVELOPMENT]},
                 "heldout":{"start_offset":DEVELOPMENT,"end_offset_exclusive":COUNT,
                              "source_indices":selected[DEVELOPMENT:],"source_ids":common_ids[DEVELOPMENT:]}}
        populations[task] = {"prompt_limit":cap,"common_eligible_count":len(common),"selected_count":COUNT,
                            "source_indices":selected,"source_ids":common_ids,"splits":split}
        for family in FAMILIES:
            candidate, rows, candidate_path = loaded[task,family]
            records = [rows[i] for i in selected]
            lens = [len(r["prompt_token_ids"]) for r in records]
            require(max(lens)+OUTPUT+DRAFT <= CONTEXT, "selected prompt/output/G7 exceeds 4096")
            model = report["models"][family]
            rendering = {"format":"raw continuation without chat tags" if family=="granite3b" else "target native chat template",
                "add_generation_prompt":family!="granite3b","encode_add_special_tokens":False,
                "truncate":False,"reference_in_prompt":False,
                "qwen_enable_thinking_override":None,
                "boundary":"Granite base raw continuation is not chat equivalence; Qwen's default thinking/template behavior is unchanged, with no override or generated-behavior validation."}
            data = {k:v for k,v in candidate.items() if k not in ("records","selection","prompt_range","rendering")}
            data.update(source_repo=aux["source"]["repo_id"], source_revision=aux["source"]["revision"],
                tokenizer_class=model["tokenizer_class"],transformers_version=report["transformers_version"],
                selection=SELECTION,prompt_range=[1,cap],rendering=rendering,scope=aux["scope"],
                reference_semantics=aux["tasks"][task]["reference_semantics"],config_sha256=report["config_sha256"],
                preparer_sha256=builder_sha,source_preparation={**provenance,"candidate":relative(candidate_path),
                    "candidate_sha256":entries[task,family]["sha256"],"candidate_records":len(rows)},
                family=family,common_population=populations[task],
                context_budget={"engine_max_model_len":CONTEXT,"output_tokens":OUTPUT,"proposal_budget":DRAFT,
                    "target_max_position_embeddings":model["target_config"]["max_position_embeddings"],
                    "draft_max_position_embeddings":model["draft_config"]["max_position_embeddings"],
                    "actual_max_prompt_tokens":max(lens),"minimum_remaining_tokens_after_output_and_G7":CONTEXT-max(lens)-OUTPUT-DRAFT,
                    "gpu_kv_capacity_and_throughput_tested":False},
                selection_report={"candidate_records":len(rows),"target_eligible_at_selected_cap":len(eligible[family]),
                    "common_eligible_count":len(common),"selected_count":COUNT}, records=records)
            path = outputs[task,family]
            payload = encoded(data)
            documents[path] = payload
            artifacts.append({"family":family,"task":task,"path":relative(path),"sha256":hashlib.sha256(payload).hexdigest(),
                "bytes":len(payload),"records":COUNT,"prompt_limit":cap,"min_prompt_tokens":min(lens),
                "max_prompt_tokens":max(lens),"mean_prompt_tokens":sum(lens)/COUNT,
                "minimum_remaining_tokens_after_output_and_G7":CONTEXT-max(lens)-OUTPUT-DRAFT,
                "source_indices":selected,"source_ids":common_ids,"candidate_sha256":entries[task,family]["sha256"]})
    manifest = {"status":"complete_common_auxiliary_population","created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection":SELECTION,"population":COUNT,"development_requests":DEVELOPMENT,"heldout_requests":COUNT-DEVELOPMENT,
        "source_sha256":report["source_sha256"],"source_revision":aux["source"]["revision"],
        "provenance":provenance,"tasks":populations,"artifacts":artifacts,"cpu_only":True,
        "tokenizer_rescan_performed":False,"generation_performed":False,
        "boundaries":[SCAN_OVERLAP,"Each target's two draft sources must reuse its same workload file and split.",
            "Equal source identity across targets preserves full task/context, not token counts or chat/base-model behavior equivalence.",
            "References are retained unchanged and checked for source identity; no reference text controls eligibility or selection.",
            "4096 prompt/output/G7 context legality does not establish GPU KV capacity, fixed actual concurrency, acceptance or performance."]}
    documents[manifest_path] = encoded(manifest)
    for path,payload in documents.items():
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feasibility",type=Path,default=ROOT/"runs/aux_workload_feasibility_r01")
    parser.add_argument("--config-dir",type=Path,default=ROOT/"configs")
    parser.add_argument("--manifest",type=Path,default=ROOT/"runs/auxiliary_population_preparation_r01/manifest.json")
    args=parser.parse_args()
    result=build(args.feasibility,args.config_dir,args.manifest)
    print(json.dumps({"status":result["status"],"manifest":str(args.manifest),
        "files":[{k:r[k] for k in ("path","records","min_prompt_tokens","max_prompt_tokens","minimum_remaining_tokens_after_output_and_G7")} for r in result["artifacts"]]}))


if __name__=="__main__":
    main()
