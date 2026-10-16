#!/usr/bin/env python3
"""One CPU-only full-prompt feasibility scan of the existing pinned archive."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
from pathlib import Path
import csv
import datetime
import gzip
import hashlib
import importlib.metadata
import json
import sys
import time
import zipfile

SCAN_STARTED_UNIX = time.time()
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
from prepare_aux_workloads import check_source, validate_row, tokenizer_fingerprints
from transformers import AutoTokenizer

def read(path):
    return json.loads(Path(path).read_text())

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_new(name, data):
    with (OUT / name).open("x") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")

config_path = ROOT / "configs/aux_workloads.json"
config = read(config_path)
source = ROOT.parent / config["source"]["shared_source"]
source_hash = check_source(source, config["source"])
thresholds = [2048, 3072, 3584, 3833, 3961]
plans = {"olmoe": "olmoe_dense_draft_reduced.json", "qwen3_awq": "qwen3_awq_eagle3_reduced.json",
         "granite3b": "granite3b_neural_draft_reduced.json"}
rows_by_task = {}
with zipfile.ZipFile(source) as archive:
    for task, spec in config["tasks"].items():
        rows = []
        with archive.open(spec["member"]) as stream:
            for index, line in enumerate(stream):
                if not line.strip():
                    continue
                row = json.loads(line)
                validate_row(row, task, index)
                rows.append((index, row))
        assert len(rows) == spec["expected_source_rows"]
        assert len({r["_id"] for _, r in rows}) == len(rows)
        rows_by_task[task] = rows

models, population, summaries, lengths, candidates = {}, [], [], {}, []
for family, plan_name in plans.items():
    plan = read(ROOT / "configs" / plan_name)
    target, draft = Path(plan["model"]), Path(plan["draft_model"])
    tc, dc = read(target / "config.json"), read(draft / "config.json")
    tok = AutoTokenizer.from_pretrained(target, local_files_only=True, trust_remote_code=False)
    ordinary_draft = plan["draft_source"] == "draft_model"
    dtok = AutoTokenizer.from_pretrained(draft, local_files_only=True, trust_remote_code=False) if ordinary_draft else None
    chat = bool(tok.chat_template)
    assert chat == (family != "granite3b"), "unexpected tokenizer rendering contract"
    context_limit = min(tc["max_position_embeddings"], dc["max_position_embeddings"])
    vocab = tok.get_vocab()
    dvocab = dtok.get_vocab() if dtok else None
    models[family] = {"target": str(target), "draft": str(draft), "draft_source": plan["draft_source"],
        "source_plan": str(ROOT / "configs" / plan_name), "source_plan_sha256": digest(ROOT / "configs" / plan_name),
        "target_config": {k:tc.get(k) for k in ("architectures", "max_position_embeddings", "rope_scaling", "rope_theta", "vocab_size", "bos_token_id", "eos_token_id", "pad_token_id")},
        "draft_config": {k:dc.get(k) for k in ("architectures", "max_position_embeddings", "rope_scaling", "rope_theta", "vocab_size", "draft_vocab_size", "bos_token_id", "eos_token_id", "pad_token_id")},
        "conservative_target_draft_context_limit": context_limit,
        "tokenizer_class": type(tok).__name__, "tokenizer_model_max_length": tok.model_max_length,
        "target_chat_template_present": chat, "target_chat_template": tok.chat_template,
        "draft_chat_template_present": bool(dtok.chat_template) if dtok else None,
        "rendering": "target chat template; add_generation_prompt=True; default template options" if chat else "same full content prompt as raw continuation; no chat delimiters",
        "target_tokenizer_fingerprints": tokenizer_fingerprints(target),
        "draft_tokenizer_fingerprints": tokenizer_fingerprints(draft),
        "target_vocab_len": len(vocab), "draft_vocab_len": len(dvocab) if dvocab else None,
        "target_vocab_entries_not_identically_mapped_in_draft": sum(dvocab.get(t) != i for t, i in vocab.items()) if dvocab else None,
        "independent_draft_reencode_mismatch_records": 0,
        "qwen_eagle_tokenizer_boundary": "EAGLE3 has no separate tokenizer; target input IDs use full input vocabulary, reduced draft output vocabulary is handled by native d2t/t2d mapping; this scan does not load or revalidate weights." if not ordinary_draft else None}
    for task, source_rows in rows_by_task.items():
        task_rows = []
        complete_records = []
        candidate_limit = min(3584, context_limit - 256 - 7)
        for index, row in source_rows:
            content = config["tasks"][task]["prompt_template"].format(context=row["context"])
            rendered = tok.apply_chat_template([{"role":"user", "content":content}], tokenize=False,
                add_generation_prompt=True) if chat else content
            ids = tok.encode(rendered, add_special_tokens=False, truncation=False)
            same_draft = (dtok.encode(rendered, add_special_tokens=False, truncation=False) == ids) if dtok else None
            if same_draft is False:
                models[family]["independent_draft_reencode_mismatch_records"] += 1
            bad = sum(i < 0 or i >= min(tc["vocab_size"], dc["vocab_size"]) for i in ids)
            p = {"family":family, "task":task, "source_index":index, "source_id":row["_id"],
                "prompt_tokens":len(ids), "content_sha256":hashlib.sha256(content.encode()).hexdigest(),
                "rendered_sha256":hashlib.sha256(rendered.encode()).hexdigest(),
                "max_token_id":max(ids), "out_of_target_or_draft_input_vocab_tokens":bad,
                "target_only_olmo_padding_tokens":sum(i >= 50280 for i in ids) if family == "olmoe" else None,
                "ordinary_draft_reencoding_equal":same_draft}
            population.append(p); task_rows.append(p)
            if 1 <= len(ids) <= candidate_limit:
                complete_records.append({"source_index":index,"source_id":row["_id"],
                    "task":task,"source_member":config["tasks"][task]["member"],
                    "prompt_token_ids":ids,"prompt":content,"rendered_sha256":p["rendered_sha256"],
                    "context_sha256":hashlib.sha256(row["context"].encode()).hexdigest(),
                    "reference":row["answers"][0],"references":row["answers"],
                    "language":row.get("language"),"source_length_statistic":row.get("length")})
        candidate_name = f"{task}_{family}_complete_candidates.json.gz"
        with gzip.open(OUT/candidate_name,"xt",encoding="utf-8",compresslevel=1) as stream:
            json.dump({"dataset":f"LongBench-derived/{task}","model":str(target),"tokenizer":str(target),
                "source":str(source),"source_sha256":source_hash,"source_member":config["tasks"][task]["member"],
                "prompt_range":[1,candidate_limit],"prompt_template":config["tasks"][task]["prompt_template"],
                "chat_template":tok.chat_template,"rendering":models[family]["rendering"],
                "tokenizer_sha256":models[family]["target_tokenizer_fingerprints"],
                "selection":"All complete eligible records in original order; no truncation or reference-based filtering. Final population selection has not been applied.",
                "cpu_only":True,"generation_performed":False,"records":complete_records},stream,ensure_ascii=False)
        candidates.append({"family":family,"task":task,"path":candidate_name,"prompt_limit":candidate_limit,
            "records":len(complete_records),"sha256":digest(OUT/candidate_name),"bytes":(OUT/candidate_name).stat().st_size})
        lengths[family, task] = {p["source_index"]:p["prompt_tokens"] for p in task_rows}
        lens = sorted(p["prompt_tokens"] for p in task_rows)
        for limit in thresholds:
            eligible = [p for p in task_rows if 1 <= p["prompt_tokens"] <= limit]
            selected = eligible[:128]
            summaries.append({"family":family, "task":task, "prompt_limit":limit,
                "source_rows":len(task_rows), "eligible_count":len(eligible), "enough_for_128":len(eligible)>=128,
                "source_min_tokens":min(lens), "source_max_tokens":max(lens),
                "minimum_prompt_limit_for_128":lens[127],
                "selected_128_max_tokens":max((p["prompt_tokens"] for p in selected), default=None) if len(selected)==128 else None,
                "selected_128_source_indices":[p["source_index"] for p in selected] if len(selected)==128 else [],
                "legal_for_128_output_plus_G7":limit+128+7<=context_limit,
                "legal_for_256_output_plus_G7":limit+256+7<=context_limit,
                "conservative_target_draft_context_limit":context_limit})
        print(json.dumps({"family":family,"task":task,"rows":len(lens),"min":min(lens),"max":max(lens),
                          "minimum_limit_for_128":lens[127],"eligible":{v:sum(n<=v for n in lens) for v in thresholds}}),flush=True)

joint = []
for task, source_rows in rows_by_task.items():
    joint_lens = {i:max(lengths[f, task][i] for f in plans) for i, _ in source_rows}
    for limit in thresholds:
        ids = [i for i, _ in source_rows if joint_lens[i] <= limit]
        selected = ids[:128]
        joint.append({"task":task,"prompt_limit":limit,"eligible_all_three_tokenizers":len(ids),
            "enough_for_128":len(ids)>=128,"minimum_common_limit_for_128":sorted(joint_lens.values())[127],
            "first_128_common_source_indices":selected if len(selected)==128 else [],
            "selected_common_max_tokens_by_family":{f:max(lengths[f, task][i] for i in selected) for f in plans} if len(selected)==128 else {}})

report = {"status":"complete_cpu_feasibility", "utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "started_unix_s":SCAN_STARTED_UNIX, "end_unix_s":time.time(),
    "elapsed_s":time.time()-SCAN_STARTED_UNIX,
    "cpu_only":True,"cuda_visible_devices":os.environ["CUDA_VISIBLE_DEVICES"],"generation_performed":False,
    "source":str(source),"source_sha256":source_hash,"config_sha256":digest(config_path),
    "scan_sha256":digest(__file__),"existing_preparer_sha256":digest(ROOT/"code/prepare_aux_workloads.py"),
    "transformers_version":importlib.metadata.version("transformers"), "models":models,
    "summaries":summaries,"cross_tokenizer_common_population":joint,"tokenized_candidates":candidates,
    "selection":"Full source scan; original JSONL order; complete context/content prompt; no truncation, duplication, shuffling, reference text inspection, reference-based filtering or generation.",
    "reference_handling":"Existing schema validation checks answers field presence/type only; no answer text is tokenized, scored or used for selection.",
    "content_templates":{t:s["prompt_template"] for t,s in config["tasks"].items()},
    "boundaries":["Native model config max_position_embeddings, not tokenizer sentinel model_max_length, sets this conservative context bound; no rope extension is assumed.",
        "Context legality reserves output length plus seven draft tokens; GPU KV feasibility and throughput are not evaluated.",
        "Granite base has no chat template. Raw continuation preserves the full content instruction/context but is not instruction-chat equivalent. Existing preparer requires explicit changes to support raw rendering and prompt limits above 2048; no existing source/config was changed.",
        "OLMoE and Qwen apply their own supplied chat templates; wrappers differ. Qwen default thinking/template behavior is preserved rather than silently disabled.",
        "Within each target family, both draft sources must reuse the same pretokenized population. Across families, equal original source IDs preserve task/context identity, not token lengths or model interface equivalence.",
        "LongBench-derived serving inputs are not an official LongBench evaluation; no proposal availability, acceptance or performance benefit is assumed."]}
write_new("feasibility.json",report)
for name, rows in [("population_lengths.csv",population),("counts.csv",summaries)]:
    with (OUT/name).open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)
print(json.dumps({"status":report["status"],"output":str(OUT),"rows":len(population),
    "started_unix_s":report["started_unix_s"],"end_unix_s":report["end_unix_s"],
    "common":[{k:r[k] for k in ("task","prompt_limit","eligible_all_three_tokenizers","minimum_common_limit_for_128")} for r in joint]}),flush=True)
