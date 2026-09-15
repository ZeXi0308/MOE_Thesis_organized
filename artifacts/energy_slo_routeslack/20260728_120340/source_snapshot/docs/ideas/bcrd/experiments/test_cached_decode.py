from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
SHARED = REPO_ROOT / "experiments" / "shared"
for path in (HERE,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from capture_native_routes import (  # noqa: E402
    _contributions_from_batches,
    run_cached_decode_steps,
)
from core import validate_identity_conservation  # noqa: E402


class _Recorder:
    def __init__(self) -> None:
        self.sample_id = -1
        self.route_batches = []
        self.routing_weight_batches = []

    def set_sample_id(self, value: int) -> None:
        self.sample_id = int(value)


class _NativeRouterOutputModel:
    """Passively derive routes from Transformers' native router logits."""

    def __init__(self, model, recorder, torch_module) -> None:
        self.model = model
        self.recorder = recorder
        self.torch = torch_module

    def __call__(self, **kwargs):
        kwargs["output_router_logits"] = True
        kwargs.pop("return_dict", None)
        output = self.model.model(return_dict=True, **kwargs)
        for layer, router_logits in enumerate(output.router_logits):
            probabilities = self.torch.softmax(router_logits.float(), dim=-1)
            weights, experts = self.torch.topk(
                probabilities,
                k=int(self.model.config.num_experts_per_tok),
                dim=-1,
            )
            self.recorder.route_batches.append(
                {
                    "sample_id": self.recorder.sample_id,
                    "layer": layer,
                    "selected_experts": experts.detach().cpu(),
                    "routing_weights": weights.detach().cpu(),
                }
            )
            self.recorder.routing_weight_batches.append(weights.detach().cpu())
        return SimpleNamespace(
            logits=self.model.lm_head(output.last_hidden_state),
            past_key_values=output.past_key_values,
            router_logits=output.router_logits,
        )


class CachedDecodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
            from transformers import OlmoeConfig, OlmoeForCausalLM
        except ImportError as exc:  # pragma: no cover - explicit environment gate
            raise unittest.SkipTest(f"PyTorch/Transformers unavailable: {exc}")
        cls.torch = torch
        torch.manual_seed(7)
        config = OlmoeConfig(
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
        cls.native_model = OlmoeForCausalLM(config).eval()
        cls.recorder = _Recorder()
        cls.model = _NativeRouterOutputModel(cls.native_model, cls.recorder, torch)

    def test_cached_logits_match_full_prefix_and_routes_close_each_step(self) -> None:
        torch = self.torch
        prompt = torch.tensor([[4, 7, 9, 11, 13]], dtype=torch.long)
        forced = torch.tensor([[17, 19, 23]], dtype=torch.long)
        inputs = {"input_ids": prompt, "attention_mask": torch.ones_like(prompt)}
        self.recorder.set_sample_id(3)
        steps = run_cached_decode_steps(
            self.model,
            self.recorder,
            inputs,
            max_steps=3,
            eos_token_id=None,
            forced_decode_ids=forced,
            capture_logits=True,
        )
        self.assertEqual(len(steps), 3)

        all_rows = []
        request_id = "unit:decode:000003"
        for step_index, step in enumerate(steps):
            prefix = torch.cat((prompt, forced[:, : step_index + 1]), dim=1)
            with torch.inference_mode():
                recomputed = self.model(
                    input_ids=prefix,
                    attention_mask=torch.ones_like(prefix),
                    use_cache=False,
                    return_dict=True,
                ).logits[:, -1, :]
            torch.testing.assert_close(
                step.logits[:, -1, :],
                recomputed.detach().float().cpu(),
                rtol=1e-4,
                atol=1e-5,
            )
            self.assertEqual(step.cache_length, prompt.shape[1] + step_index + 1)
            self.assertEqual(len(step.route_batches), 2)
            self.assertTrue(
                all(tuple(batch["selected_experts"].shape) == (1, 2) for batch in step.route_batches)
            )
            all_rows.extend(
                _contributions_from_batches(
                    batches=step.route_batches,
                    model_key="unit-olmoe",
                    phase="decode",
                    request_id=request_id,
                    sample_id=3,
                    arrival_us=0.0,
                    deadline_us=1000.0,
                    input_event_ids=(f"{request_id}:decode:{step_index:06d}",),
                    token_ids=(step.token_id,),
                    token_positions=(step.absolute_position,),
                    decode_steps=(step_index,),
                )
            )

        summary = validate_identity_conservation(all_rows)
        self.assertEqual(summary["contributions"], 12)
        self.assertEqual(len({row.input_event_id for row in all_rows}), 3)
        self.assertEqual(len({row.contribution_id for row in all_rows}), 12)

    def test_eos_is_not_executed_and_max_step_is_respected(self) -> None:
        torch = self.torch
        prompt = torch.tensor([[2, 3, 5]], dtype=torch.long)
        forced = torch.tensor([[7, 8, 9]], dtype=torch.long)
        inputs = {"input_ids": prompt, "attention_mask": torch.ones_like(prompt)}
        self.recorder.set_sample_id(4)
        stopped = run_cached_decode_steps(
            self.model,
            self.recorder,
            inputs,
            max_steps=3,
            eos_token_id=7,
            forced_decode_ids=forced,
        )
        self.assertEqual(stopped, [])

        self.recorder.set_sample_id(4)
        limited = run_cached_decode_steps(
            self.model,
            self.recorder,
            inputs,
            max_steps=2,
            eos_token_id=None,
            forced_decode_ids=forced,
        )
        self.assertEqual(len(limited), 2)


if __name__ == "__main__":
    unittest.main()
