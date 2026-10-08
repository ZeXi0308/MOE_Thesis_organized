"""CPU structure tests; the optional torch test also runs on the deployment host.

These exercise request/row identity, prefix/bonus indexing and the original
native rejection arithmetic. They do not stand in for a GPU continuation test.
"""
import importlib.util
import itertools
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from suffix_runtime import (SuffixStepState, install_suffix_runtime,
                            packed_prefix_plan, rebuild_prefix_metadata)


class StateTests(unittest.TestCase):
    def make_state(self):
        # r0 is a prefill chunk; r1/r2 have two/one draft; r3 is AR.
        return SuffixStepState().begin_step(
            req_ids=["prefill", "two", "one", "ar"], num_scheduled_tokens=[4, 3, 2, 1],
            computed_starts=[0, 20, 30, 40], prompt_tokens=[8, 10, 10, 10],
            original_draft_counts=[0, 2, 1, 0],
            row_indices=[0, 0, 0, 0, 1, 1, 1, 2, 2, 3])

    def test_legal_rows_and_padding(self):
        state = self.make_state()
        self.assertEqual(state.eligible, [False, True, True, False])
        state.set_keep_drafts([0, 1, 0, 0], layer="layer0")
        self.assertEqual(state.alive_indices(12), [0, 1, 2, 3, 4, 5, 7, 9, 10, 11])
        state.set_keep_drafts([0, 0, 0, 0], layer="layer3")
        self.assertEqual(state.alive_indices(), [0, 1, 2, 3, 4, 7, 9])
        with self.assertRaises(ValueError):
            state.set_keep_drafts([0, 1, 0, 0])
        with self.assertRaises(ValueError):
            state.alive_indices(9)

    def test_unmapped_and_nondecode_rejected(self):
        with self.assertRaises(ValueError):
            SuffixStepState().begin_step(req_ids=["a", "b"], num_scheduled_tokens=[2, 1],
                computed_starts=[8, 8], prompt_tokens=[4, 4], original_draft_counts=[1, 0],
                row_indices=[0, 1, 0])
        state = SuffixStepState().begin_step(req_ids=["recovery"], num_scheduled_tokens=[4],
            computed_starts=[1], prompt_tokens=[8], original_draft_counts=[2])
        with self.assertRaises(ValueError):
            state.set_keep_drafts([1])

    def test_step_reset(self):
        state = self.make_state()
        state.set_keep_drafts([0, 0, 0, 0])
        state.sampled = True
        with self.assertRaises(ValueError):
            state.alive_indices()
        state.begin_step(req_ids=["one"], num_scheduled_tokens=[2], computed_starts=[31],
                         prompt_tokens=[10], original_draft_counts=[1], row_indices=[0, 0])
        self.assertEqual(state.keep_drafts, [1])
        self.assertEqual(state.step_id, 1)


class PrefixTests(unittest.TestCase):
    def test_original_example_keeps_prefix_not_tail(self):
        plan = packed_prefix_plan([3, 0, 2, 0, 1], [1, 0, 0, 0, 1])
        self.assertEqual(plan["logit_take"], [0, 1, 4, 5, 8, 9, 10])
        self.assertEqual(plan["draft_take"], [0, 5])
        self.assertEqual(plan["target_logits_indices"], [0, 5])
        self.assertEqual(plan["bonus_logits_indices"], [1, 2, 3, 4, 6])
        self.assertEqual(plan["cu_num_draft_tokens"], [1, 1, 1, 1, 2])

    def test_all_zero_selects_one_anchor_per_request(self):
        plan = packed_prefix_plan([1, 2, 0], [0, 0, 0])
        self.assertEqual(plan["logit_take"], [0, 2, 5])
        self.assertEqual(plan["draft_take"], [])
        self.assertEqual(plan["bonus_logits_indices"], [0, 1, 2])

    def test_exhaustive_prefix_and_native_rollback(self):
        for drafts in itertools.product(range(4), repeat=3):
            for kept in itertools.product(*(range(d + 1) for d in drafts)):
                plan = packed_prefix_plan(drafts, kept)
                labels = [(r, j) for r, d in enumerate(drafts) for j in range(d + 1)]
                selected = [labels[i] for i in plan["logit_take"]]
                self.assertEqual(selected, [(r, j) for r, k in enumerate(kept) for j in range(k + 1)])
                for r, k in enumerate(kept):
                    self.assertEqual(selected[plan["bonus_logits_indices"][r]], (r, k))
                    # Any natural mismatch before k or full prefix acceptance:
                    # generated = accepted drafts + one fallback/bonus token.
                    for accepted in range(k + 1):
                        original_computed = 100 + drafts[r] + 1
                        generated_count = accepted + 1
                        rejected = drafts[r] - max(generated_count - 1, 0)
                        self.assertEqual(original_computed - rejected, 100 + accepted + 1)

    def test_invalid_lengths(self):
        for a, b in (([1], [2]), ([1], []), ([], []), ([True], [0]), ([1], [-1])):
            with self.assertRaises(ValueError):
                packed_prefix_plan(a, b)


class HookTests(unittest.TestCase):
    def test_default_min_tokens_processor_is_allowed_but_custom_is_rejected(self):
        class MinTokens:
            pass

        class CustomProcessor:
            pass

        builtin = ModuleType("vllm.v1.sample.logits_processor.builtin")
        builtin.MinTokensLogitsProcessor = MinTokens
        for processor, allowed in ((MinTokens(), True), (CustomProcessor(), False)):
            sampling = SimpleNamespace(all_greedy=True, no_penalties=True,
                bad_words_token_ids={}, allowed_token_ids_mask=None,
                thinking_budget_state_holder=None,
                logitsprocs=SimpleNamespace(non_argmax_invariant=[processor]))
            sample = lambda logits, metadata: SimpleNamespace(
                sampled_token_ids=SimpleNamespace(shape=(1, 1)))
            runner = SimpleNamespace(speculative_config=SimpleNamespace(method="ngram"),
                use_async_scheduling=False, num_spec_tokens=2,
                parallel_config=SimpleNamespace(use_ubatching=False),
                model_config=SimpleNamespace(is_hybrid=False),
                input_batch=SimpleNamespace(req_ids=["r"], sampling_metadata=sampling),
                _prepare_inputs=lambda *args: None, _sample=sample)
            state = SuffixStepState().begin_step(req_ids=["r"], num_scheduled_tokens=[3],
                computed_starts=[10], prompt_tokens=[5], original_draft_counts=[2])
            state.set_keep_drafts([0])
            uninstall = install_suffix_runtime(runner, state)
            with patch.dict(sys.modules, {builtin.__name__: builtin}), patch(
                    "suffix_runtime.rebuild_prefix_metadata", return_value=("prefix_logits", None, {})):
                if allowed:
                    runner._sample("full_logits", SimpleNamespace(num_draft_tokens=[2]))
                    self.assertTrue(state.sampled)
                else:
                    with self.assertRaises(ValueError):
                        runner._sample("full_logits", SimpleNamespace(num_draft_tokens=[2]))
            uninstall()
            self.assertIs(runner._sample, sample)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch unavailable locally")
class TorchMetadataTests(unittest.TestCase):
    def test_native_metadata_fields_and_toy_greedy(self):
        import torch

        @dataclass
        class Metadata:
            draft_token_ids: object
            num_draft_tokens: list
            cu_num_draft_tokens: object
            cu_num_sampled_tokens: object
            target_logits_indices: object
            bonus_logits_indices: object
            logits_indices: object

            def __post_init__(self):
                self.max_spec_len = max(self.num_draft_tokens)

        if importlib.util.find_spec("vllm") is not None:
            from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
            Metadata = SpecDecodeMetadata

        def tensor(values):
            return torch.tensor(values, dtype=torch.int32)

        original = Metadata(tensor([1, 2, 4]), [2, 1, 0], tensor([2, 3, 3]),
                            tensor([3, 5, 6]), tensor([0, 1, 3]), tensor([2, 4, 5]),
                            tensor([10, 11, 12, 20, 21, 30]))
        logits = torch.full((6, 8), -10.)
        logits[torch.arange(6), torch.tensor([1, 2, 3, 4, 5, 6])] = 10.
        short, meta, _ = rebuild_prefix_metadata(logits, original, [1, 0, 0])
        self.assertEqual(short.argmax(-1).tolist(), [1, 2, 4, 6])
        self.assertEqual(meta.draft_token_ids.tolist(), [1])
        self.assertEqual(meta.logits_indices.tolist(), [10, 11, 20, 30])
        self.assertEqual(meta.bonus_logits_indices.tolist(), [1, 2, 3])
        self.assertEqual(meta.cu_num_sampled_tokens.tolist(), [2, 3, 4])
        self.assertEqual(meta.max_spec_len, 1)
        self.assertEqual(original.num_draft_tokens, [2, 1, 0])
        self.assertEqual(original.logits_indices.tolist(), [10, 11, 12, 20, 21, 30])
        # Fully cancelled first draft A can output A only once, never AA.
        anchors, meta, _ = rebuild_prefix_metadata(logits, original, [0, 0, 0])
        self.assertIsNone(meta)
        self.assertEqual(anchors.argmax(-1).tolist(), [1, 4, 6])


if __name__ == "__main__":
    unittest.main()
