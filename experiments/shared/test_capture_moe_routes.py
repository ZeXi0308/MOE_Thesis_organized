from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
from torch import nn


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from capture_moe import (  # noqa: E402
    MoeRecorder,
    _patched_mixtral_sparse_moe_forward,
    _patched_olmoe_sparse_moe_forward,
    _patched_qwen2_moe_sparse_moe_forward,
)
from policies import make_policy  # noqa: E402


class _DummyMoE(nn.Module):
    def __init__(self, kind: str, *, record_routes: bool) -> None:
        super().__init__()
        self.num_experts = 2
        self.top_k = 2
        self.gate = nn.Linear(4, self.num_experts, bias=False)
        self.experts = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.norm_topk_prob = True
        self.jitter_noise = 0.0
        self._idea_layer_id = 3
        self._idea_recorder = MoeRecorder(record_routes=record_routes)
        self._idea_recorder.set_sample_id(17)
        self._idea_record_diagnostics = False
        self._idea_cache_routing = False
        self._idea_lock_routing = False
        self._idea_grouped_owner_policy = None
        self._idea_creditreduce_endpoint = None
        self._idea_policy = make_policy("full")
        self._idea_num_receiver_groups = 1
        self._idea_receiver_mapping = "contiguous"
        if kind == "qwen":
            self.shared_expert = nn.Identity()
            self.shared_expert_gate = nn.Linear(4, 4, bias=False)


class RouteCaptureTest(unittest.TestCase):
    def test_route_only_capture_is_independent_of_diagnostics(self) -> None:
        cases = (
            ("mixtral", _patched_mixtral_sparse_moe_forward),
            ("olmoe", _patched_olmoe_sparse_moe_forward),
            ("qwen", _patched_qwen2_moe_sparse_moe_forward),
        )
        for kind, forward in cases:
            with self.subTest(kind=kind):
                moe = _DummyMoE(kind, record_routes=True)
                hidden = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)

                output, logits = forward(moe, hidden)

                self.assertEqual(output.shape, hidden.shape)
                self.assertEqual(logits.shape, (3, 2))
                self.assertEqual(len(moe._idea_recorder.route_batches), 1)
                route_batch = moe._idea_recorder.route_batches[0]
                self.assertEqual(route_batch["sample_id"], 17)
                self.assertEqual(route_batch["layer"], 3)
                self.assertEqual(tuple(route_batch["selected_experts"].shape), (3, 2))
                self.assertFalse(moe._idea_recorder.rank_stats)
                self.assertFalse(moe._idea_recorder.receiver_rank_stats)
                self.assertFalse(moe._idea_recorder.error_stats)

    def test_route_capture_respects_record_routes_false(self) -> None:
        moe = _DummyMoE("olmoe", record_routes=False)
        hidden = torch.ones((1, 2, 4), dtype=torch.float32)

        _patched_olmoe_sparse_moe_forward(moe, hidden)

        self.assertEqual(moe._idea_recorder.route_batches, [])


if __name__ == "__main__":
    unittest.main()
