#!/usr/bin/env python3
"""CPU interface test: real installed lm-eval code, fake engine/output fixtures.

Run with CUDA_VISIBLE_DEVICES empty and the documented library path. This does
not construct a vLLM engine, load a model, fetch a dataset, or measure quality.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("CPU check requires CUDA_VISIBLE_DEVICES=''")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import eval_quality as q
import torch
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.models.vllm_causallms import VLLM

spec = q.generation_plan(ROOT / "configs/quality_generation_gsm8k_128.json")
assert len(spec["sample_ids"]) == 128
assert spec["sample_ids"] == sorted(set(spec["sample_ids"]))
spec = {**spec, "sample_ids": [2, 8, 13], "max_gen_toks": 4,
        "maximum_generated_tokens_per_arm": 12}
with tempfile.TemporaryDirectory(prefix="generation_quality_cpu_") as directory:
    ledger = q.GenerationLedger(directory, spec)
    requests, prompts = [], {}
    for index in spec["sample_ids"]:
        doc = {"question": f"CPU fixture {index}", "answer": "#### 7"}
        ledger.rows[index].update(doc=doc, doc_sha256=q.digest_json(doc), gold=doc["answer"])
        prompts[f"prompt-{index}"] = [index, 3, 4]
        requests.append(Instance("generate_until", doc,
            (f"prompt-{index}", {"until": ["Q:"], "do_sample": False, "max_gen_toks": 4}),
            0, metadata=(spec["task"], index, 1)))
    cls = q.recording_vllm(VLLM, ledger)
    model = object.__new__(cls)  # Deliberately never call the engine constructor.
    LM.__init__(model)
    model.tokenizer = SimpleNamespace(eos_token_id=0, eos_token="</s>", decode=lambda i: "</s>")
    model.tok_encode = lambda texts: [prompts[text] for text in texts]
    model._max_length, model._max_gen_toks = 32, 4
    model.custom_prefix_token_id = None
    model.batch_size, model.data_parallel_size = 2, 1
    # All constructor state read by the installed single-device generate_until
    # -> _model_generate path. LM.__init__ above supplies rank/cache_hook;
    # tokenizer/tok_encode and the fake engine are the deliberate test seams.
    # The real constructor normally sets this even when no truncation occurs.
    model.truncation_side = "left"
    model.think_end_token, model.lora_request = None, None
    calls = []

    def generate(inputs, sampling_params, **kwargs):
        calls.append(len(inputs))
        outputs = []
        for prompt, params in zip(inputs, sampling_params, strict=True):
            ids = prompt["prompt_token_ids"]
            index = ids[0]
            tokens, finish, stop = (([7, 0], "stop", None) if index == 2 else
                ([7], "stop", "Q:") if index == 8 else ([7] * 4, "length", None))
            outputs.append(SimpleNamespace(prompt_token_ids=ids, outputs=[SimpleNamespace(
                token_ids=tokens, text="The answer is 7.", finish_reason=finish,
                stop_reason=stop)], finished=True, request_id=str(index)))
        return outputs

    model.model = SimpleNamespace(generate=generate)
    for attribute in ("tokenizer", "tok_encode", "batch_size", "rank",
                      "cache_hook", "eot_token_id", "max_gen_toks", "max_length",
                      "truncation_side", "think_end_token", "data_parallel_size",
                      "model", "lora_request"):
        assert hasattr(model, attribute), f"fixture lacks generation state: {attribute}"
    assert model.generate_until(requests, disable_tqdm=True) == ["The answer is 7."] * 3
    assert calls == [2, 1]
    assert [ledger.rows[i]["status"] for i in spec["sample_ids"]] == ["eos", "stop_string", "cap"]
    ledger.rows[8]["status"], ledger.rows[13]["status"] = "in_flight", "not_started"
    ledger.fail(TimeoutError("fixture deadline"))
    summary = json.loads((Path(directory) / "generation_summary.json").read_text())
    assert summary["scoring_denominator"] == 3 and summary["metrics"] is None
    assert summary["termination_counts"] == {"eos": 1, "timeout": 1, "unfinished": 1}
    assert not torch.cuda.is_initialized()
    print(json.dumps({"actual_installed_generate_until_accounting": "passed",
        "engine_constructed": False, "cuda_initialized": False,
        "task_data_downloaded": False, "quality_measured": False}))
