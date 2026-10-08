"""CPU fake-tensor execution tests plus an optional native torch metadata test."""
import copy
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import importlib.util
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from suffix_compaction import SuffixCompaction, build_compact_context, compact_query_plan
from suffix_runtime import SuffixStepState, packed_prefix_plan


class Tensor:
    def __init__(self, data, dtype="int32", device="cpu"):
        self.data, self.dtype, self.device = copy.deepcopy(data), dtype, device
        self.shape = ((len(data), len(data[0])) if data and isinstance(data[0], list)
                      else (len(data),))
        self.ndim = len(self.shape)

    def __len__(self):
        return len(self.data)

    def index_select(self, dim, indices):
        assert dim == 0
        return Tensor([self.data[i] for i in indices.data], self.dtype, self.device)

    def index_copy_(self, dim, indices, source):
        assert dim == 0
        for i, row in zip(indices.data, source.data):
            self.data[i] = copy.deepcopy(row)
        return self

    def new_zeros(self, shape):
        return Tensor([[0] * shape[1] for _ in range(shape[0])], self.dtype, self.device)

    def tolist(self):
        return copy.deepcopy(self.data)


class Torch:
    long = "int64"
    tensor = staticmethod(lambda data, dtype="int32", device="cpu": Tensor(data, dtype, device))


class Mode(Enum):
    NONE = 0


@dataclass
class FlashAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: object
    max_seq_len: int
    seq_lens: object
    block_table: object
    slot_mapping: object
    use_cascade: bool = False
    common_prefix_len: int = 0
    dcp_context_kv_lens: object = None
    causal: bool = True
    sliding_window: object = (-1, -1)
    scheduler_metadata: object = "old_aot_schedule"
    prefix_scheduler_metadata: object = None


@dataclass
class Descriptor:
    num_tokens: int
    num_reqs: int
    uniform: bool = False


@dataclass
class Context:
    attn_metadata: dict
    slot_mapping: dict
    dp_metadata: object = None
    ubatch_slices: object = None
    is_padding: object = None
    cudagraph_runtime_mode: object = Mode.NONE
    batch_descriptor: object = None
    skip_compiled: bool = False
    moe_layer_index: int = 0


def state_fixture():
    # Three mandatory prefill rows, a two-draft decode, a one-draft decode, AR.
    return SuffixStepState().begin_step(req_ids=["p", "d2", "d1", "ar"],
        num_scheduled_tokens=[3, 3, 2, 1], computed_starts=[0, 9, 19, 29],
        prompt_tokens=[6, 5, 5, 5], original_draft_counts=[0, 2, 1, 0])


def context_fixture(tensor=Tensor):
    names = [f"model.layers.{i}.self_attn.attn" for i in range(3)]
    slots = tensor(list(range(100, 109)))
    metadata = FlashAttentionMetadata(9, 3, tensor([0, 3, 6, 8, 9]), 30,
                                     tensor([3, 12, 21, 30]),
                                     tensor([[1], [2], [3], [4]]), slots)
    context = Context(dict.fromkeys(names, metadata), dict.fromkeys(names, slots),
                      batch_descriptor=Descriptor(9, 4))
    return context, names


class FlashAttentionImpl:
    pass


class Layer:
    def __init__(self, index, get_context, controller, observed, fail=False):
        self.index, self.get_context, self.controller = index, get_context, controller
        self.observed, self.fail = observed, fail
        self.self_attn = SimpleNamespace(attn=SimpleNamespace(
            layer_name=f"model.layers.{index}.self_attn.attn",
            impl=FlashAttentionImpl(), kv_sharing_target_layer_name=None))

    def __call__(self, positions, hidden, residual):
        context = self.get_context()
        name = self.self_attn.attn.layer_name
        metadata, slots = context.attn_metadata[name], context.slot_mapping[name]
        qstarts, seqs = metadata.query_start_loc.data, metadata.seq_lens.data
        aligned = []
        for i, seq in enumerate(seqs):
            count = qstarts[i + 1] - qstarts[i]
            aligned.extend(range(seq - count, seq))
        assert aligned == positions.data, "causal query alignment shifted"
        assert len(slots) == len(hidden), "KV write mapping still has old rows"
        self.observed.append(dict(layer=self.index, rows=len(hidden), slots=slots.tolist(),
                                  positions=positions.tolist(), seq_lens=seqs[:]))
        context.moe_layer_index += 1
        if self.fail:
            raise RuntimeError("injected downstream failure")
        residual = hidden if residual is None else residual
        output = Tensor([[v + positions.data[r] + self.index + 1 for v in row]
                         for r, row in enumerate(hidden.data)], "float")
        if self.index == self.controller.decision_layer:
            self.controller.state.set_keep_drafts([0, 1, 0, 0])
            alive = set(self.controller.state.alive_indices())
            for i in range(len(output)):
                if i not in alive:
                    output.data[i] = [0] * len(output.data[i])
        return output, residual


class OlmoeModel:
    do_not_compile = True
    start_layer, end_layer = 0, 3

    def __init__(self, layers):
        self.layers = layers

    def embed_input_ids(self, inputs):
        return inputs

    def norm(self, hidden, residual=None):
        if residual is None:
            return hidden
        return Tensor([[a + b for a, b in zip(x, y)]
                       for x, y in zip(hidden.data, residual.data)], "float"), residual

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        hidden, residual = input_ids, None
        for layer in self.layers:
            hidden, residual = layer(positions, hidden, residual)
        return self.norm(hidden, residual)[0]


class OlmoeForCausalLM:
    def __init__(self, inner):
        self.model = inner
        self.logit_rows = []

    def compute_logits(self, hidden):
        self.logit_rows.append(len(hidden))
        return Tensor([[a, b, a + b] for a, b in hidden.data], "float")


def adapter_fixture(*, fail=False, unsupported=False):
    state = state_fixture()
    context, names = context_fixture()
    if unsupported:
        context.attn_metadata[names[1]].use_cascade = True
    current = {"context": context}
    module = ModuleType("vllm.forward_context")
    module.get_forward_context = lambda: current["context"]

    @contextmanager
    def override(value):
        old = current["context"]
        current["context"] = value
        try:
            yield
        finally:
            current["context"] = old

    module.override_forward_context = override
    controller = SimpleNamespace(state=state, num_layers=3, decision_layer=0,
        runtime=SimpleNamespace(torch=Torch, context={"phase":"measurement"}))
    observed = []
    layers = [Layer(i, module.get_forward_context, controller, observed, fail and i == 1)
              for i in range(3)]
    outer = OlmoeForCausalLM(OlmoeModel(layers))
    runner = SimpleNamespace(model=outer, use_async_scheduling=False,
        speculative_config=SimpleNamespace(method="ngram"),
        model_config=SimpleNamespace(enforce_eager=True, is_hybrid=False),
        parallel_config=SimpleNamespace(use_ubatching=False))
    with patch.dict(sys.modules, {module.__name__: module}):
        adapter = SuffixCompaction(runner, controller)
    return adapter, controller, outer, context, current, observed


class CompactionTests(unittest.TestCase):
    def test_mixed_plan_and_independent_kv_mapping(self):
        state = state_fixture()
        state.set_keep_drafts([0, 1, 0, 0])
        plan = compact_query_plan(state)
        self.assertEqual(plan["indices"], [0, 1, 2, 3, 4, 6, 8])
        self.assertEqual(plan["query_start_loc"], [0, 3, 5, 6, 7])
        self.assertEqual(plan["seq_lens"], [3, 11, 20, 30])
        self.assertEqual([s - q for s, q in zip(plan["seq_lens"], plan["query_lens"])],
                         state.computed_starts)
        context, names = context_fixture()
        new, reason = build_compact_context(context, names[1:], plan, total_rows=9, torch=Torch)
        self.assertIsNone(reason)
        self.assertIs(new.attn_metadata[names[0]], context.attn_metadata[names[0]])
        self.assertIs(new.attn_metadata[names[1]], new.attn_metadata[names[2]])
        self.assertEqual(new.slot_mapping[names[1]].tolist(), [100,101,102,103,104,106,108])
        self.assertIs(new.attn_metadata[names[1]].slot_mapping, new.slot_mapping[names[1]])
        self.assertIsNone(new.attn_metadata[names[1]].scheduler_metadata)
        self.assertIs(new.attn_metadata[names[1]].block_table, context.attn_metadata[names[1]].block_table)
        self.assertEqual(context.slot_mapping[names[1]].tolist(), list(range(100,109)))
        self.assertEqual(context.attn_metadata[names[1]].seq_lens.tolist(), [3,12,21,30])

    def test_forward_logits_prefix_bonus_and_context_restore(self):
        adapter, controller, outer, context, current, observed = adapter_fixture()
        values = [[i, i + 10] for i in range(9)]
        positions = [0,1,2,9,10,11,19,20,29]
        output = outer.model.forward(Tensor(values, "float"), Tensor(positions))
        self.assertEqual([r["rows"] for r in observed], [9,7,7])
        self.assertEqual(observed[1]["slots"], [100,101,102,103,104,106,108])
        self.assertIs(current["context"], context)
        self.assertFalse(controller.compact_active)
        self.assertEqual(context.moe_layer_index, 3)
        kept = [0,1,2,3,4,6,8]
        for i in range(9):
            expected = [2*v + 3*positions[i] + 6 for v in values[i]] if i in kept else [0,0]
            self.assertEqual(output.data[i], expected)
        packed = output.index_select(0, Tensor([2,3,4,5,6,7,8]))
        logits = outer.compute_logits(packed)
        self.assertEqual(outer.logit_rows, [5])
        self.assertEqual(logits.shape, (7,3))
        self.assertEqual(logits.data[3], [0,0,0])
        self.assertEqual(logits.data[5], [0,0,0])
        plan = packed_prefix_plan([0,2,1,0], [0,1,0,0])
        self.assertEqual(plan["logit_take"], [0,1,2,4,6])
        self.assertEqual(plan["bonus_logits_indices"], [0,2,3,4])
        report = adapter.report()
        self.assertEqual(report["compact_steps"], 1)
        self.assertEqual(report["compact_logits_steps"], 1)
        adapter.uninstall()
        self.assertNotIn("forward", vars(outer.model))
        self.assertNotIn("compute_logits", vars(outer))

    def test_failure_restores_context_and_execution_mapping(self):
        adapter, controller, outer, context, current, _ = adapter_fixture(fail=True)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            outer.model.forward(Tensor([[i,i] for i in range(9)]), Tensor([0,1,2,9,10,11,19,20,29]))
        self.assertIs(current["context"], context)
        self.assertFalse(controller.compact_active)
        self.assertIsNone(controller.compact_original_indices)
        self.assertEqual(adapter.records[0]["status"], "failed")

    def test_unsupported_metadata_falls_back_without_context_mutation(self):
        adapter, controller, outer, context, current, observed = adapter_fixture(unsupported=True)
        output = outer.model.forward(Tensor([[i,i] for i in range(9)]), Tensor([0,1,2,9,10,11,19,20,29]))
        self.assertEqual(len(output), 9)
        self.assertEqual([r["rows"] for r in observed], [9,9,9])
        self.assertIs(current["context"], context)
        self.assertFalse(adapter.records[0]["compact_applied"])
        self.assertIn("nonordinary", adapter.records[0]["fallback_reason"])


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch unavailable locally")
class NativeTensorTests(unittest.TestCase):
    def test_real_tensors_keep_physical_slots_and_causal_positions(self):
        import torch
        state = state_fixture()
        state.set_keep_drafts([0,1,0,0])
        context, names = context_fixture(lambda x: torch.tensor(x, dtype=torch.int32))
        if importlib.util.find_spec("vllm") is not None:
            from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata as NativeMetadata
            old = context.attn_metadata[names[0]]
            actual = NativeMetadata(num_actual_tokens=old.num_actual_tokens,
                max_query_len=old.max_query_len, query_start_loc=old.query_start_loc,
                max_seq_len=old.max_seq_len, seq_lens=old.seq_lens,
                block_table=old.block_table, slot_mapping=old.slot_mapping,
                use_cascade=False, common_prefix_len=0, cu_prefix_query_lens=None,
                prefix_kv_lens=None, suffix_kv_lens=None)
            context.attn_metadata = dict.fromkeys(names, actual)
        new, reason = build_compact_context(context, names[1:], compact_query_plan(state),
                                            total_rows=9, torch=torch)
        self.assertIsNone(reason)
        self.assertEqual(new.slot_mapping[names[1]].tolist(), [100,101,102,103,104,106,108])
        metadata = new.attn_metadata[names[1]]
        self.assertEqual((metadata.seq_lens - metadata.query_start_loc.diff()).tolist(), [0,9,19,29])
        self.assertEqual(context.attn_metadata[names[1]].seq_lens.tolist(), [3,12,21,30])


if __name__ == "__main__":
    unittest.main()
