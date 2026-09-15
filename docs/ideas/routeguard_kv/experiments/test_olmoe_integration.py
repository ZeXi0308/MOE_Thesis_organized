from __future__ import annotations

import copy

import torch
from transformers import OlmoeConfig, OlmoeForCausalLM

from kv_quant_cache import QuantizedDynamicCache
from r0a_artifacts import load_config
from run_r0a_5090 import _run_reference, _run_treatment


def test_actual_transformers_olmoe_decode_cache_and_route_lock_integration() -> None:
    torch.manual_seed(7)
    model_config = OlmoeConfig(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        norm_topk_prob=False,
    )
    model = OlmoeForCausalLM(model_config).eval()
    config = copy.deepcopy(
        load_config(__import__("pathlib").Path(__file__).parent / "configs/r0a_5090_v1.json")
    )
    config["model"]["expected"]["num_hidden_layers"] = 2
    config["model"]["expected"]["num_experts_per_tok"] = 2
    config["dataset"]["decode_steps"] = 3
    document = {
        "split": "unit",
        "document_index": 0,
        "text_sha256": "e" * 64,
    }
    token_ids = torch.randint(0, model_config.vocab_size, (1, 16))
    with torch.inference_mode():
        prefill = model(
            input_ids=token_ids[:, :8],
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        prompt_cache = prefill.past_key_values
        reference = _run_reference(
            model,
            prompt_cache,
            token_ids,
            document=document,
            prompt_length=8,
            config=config,
        )
        cache = QuantizedDynamicCache.from_prompt_cache(
            prompt_cache, target="k_only", config=model.config
        )
        row = _run_treatment(
            model,
            cache,
            token_ids,
            reference,
            document=document,
            prompt_length=8,
            target="k_only",
            arm="fully_locked",
            controller_mode="fully_locked",
            config=config,
            quantizer_ledger=cache.ledger,
        )
    assert reference.trajectory["reference_route_records"] == 6
    assert row["completed_steps"] == 3
    assert row["route_metrics"]["route_cell_count"] == 6
    assert row["route_metrics"]["executed_reference_set_mismatch_count"] == 0
    assert row["quantizer_ledger"] == {
        "snapshot_events": 2,
        "update_events": 6,
        "expected_snapshot_events": 2,
        "expected_update_events": 6,
        "pass": True,
    }

