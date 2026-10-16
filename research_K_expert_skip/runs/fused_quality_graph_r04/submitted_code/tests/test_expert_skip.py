import importlib.util
import math
import os
from pathlib import Path
import random
import sys
import unittest
from unittest import mock
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from expert_skip import (ExpertSkipPolicy, SkipConfig, reference_policy,
                         install_safe_alignment)

HAS_TORCH = importlib.util.find_spec("torch") is not None


class PolicySemantics(unittest.TestCase):
    def test_rank_gate_has_half_open_rank_and_gate_bins_without_singleton_guard(self):
        weights=[[.5,.25,.125,.125]]*2
        ids=[[0,1,2,3]]*2
        config=SkipConfig(mode='rank_gate',tail_rank=1,rank_end=3,
                          gate_min=.125,gate_threshold=.25,mass_budget=1.)
        _,result,stats=reference_policy(weights,ids,4,config)
        self.assertEqual(result,[[0,1,-1,3]]*2)
        self.assertEqual(stats['skipped_slots'],2)
        self.assertEqual(stats['distinct_before']-stats['distinct_after'],1)

    def test_unsorted_weights_rank_budget_and_scaled_renormalization(self):
        # Selected gate mass is 2, as with routed_scaling_factor=2.
        w, i, s = reference_policy([[.1, 1.6, .3]], [[5, 2, 7]], 8,
                                  SkipConfig(mode="marginal", tail_rank=1,
                                             gate_threshold=.2, mass_budget=.1))
        self.assertEqual(i, [[-1, 2, 7]])
        self.assertAlmostEqual(sum(w[0]), 2.0)
        self.assertAlmostEqual(s["removed_gate_mass_sum"], .05)
        self.assertEqual(s["distinct_before"]-s["distinct_after"], 1)

    def test_shared_tail_consumes_no_marginal_budget(self):
        weights = [[.8, .15, .05], [.8, .15, .05]]
        ids = [[0, 1, 2], [3, 4, 2]]
        common = dict(gate_threshold=.2, mass_budget=.21, tail_rank=1)
        _, im, sm = reference_policy(weights, ids, 5,
                                    SkipConfig(mode="marginal", **common),
                                    weight_bytes_per_expert=100)
        _, ig, sg = reference_policy(weights, ids, 5,
                                    SkipConfig(mode="gate", **common),
                                    weight_bytes_per_expert=100)
        self.assertEqual(im, [[0, -1, 2], [3, -1, 2]])
        self.assertEqual(ig, [[0, -1, -1], [3, -1, -1]])
        self.assertEqual(sm["skipped_slots"], 2)
        self.assertEqual(sm["logical_weight_bytes_before"]-sm["logical_weight_bytes_after"], 200)
        # A gate baseline can remove a shared expert if ALL of its slots drop.
        self.assertEqual(sg["distinct_before"]-sg["distinct_after"], 3)

    def test_global_exact_decrement_and_model_renorm_false(self):
        w, i, _ = reference_policy([[.08, .1, .01, .4]], [[0, 1, 2, 3]], 4,
                                  SkipConfig(mode="global", drop_j=2), False)
        self.assertEqual(i, [[-1, 1, -1, 3]])
        self.assertEqual(w, [[0, .1, 0, .4]])

    def test_ties_keep_first_slot_and_never_remove_last_expert(self):
        w, i, s = reference_policy([[0, 0, 0]], [[2, 0, 1]], 3,
                                  SkipConfig(mode="global", drop_j=100))
        self.assertEqual(i, [[2, -1, -1]])
        self.assertEqual(s["slots_after"], 1)
        self.assertTrue(all(math.isfinite(x) for x in w[0]))

    def test_dense_identity_and_reject_invalid_input(self):
        weights, ids = [[.2, .5, .3]], [[1, 0, 2]]
        w, i, _ = reference_policy(weights, ids, 3, SkipConfig())
        self.assertEqual((w, i), (weights, ids))
        with self.assertRaises(ValueError):
            reference_policy(weights, [[1, 1, 2]], 3, SkipConfig())
        with self.assertRaises(ValueError):
            reference_policy(weights, [[1, -1, 2]], 3, SkipConfig())

    def test_randomized_budget_singleton_and_scale_invariants(self):
        rng = random.Random(404)
        for batch in (1, 4, 17):
            for _ in range(30):
                ids = [rng.sample(range(23), 5) for _ in range(batch)]
                weights = [[rng.random() for _ in range(5)] for _ in range(batch)]
                cfg = SkipConfig(mode="marginal", tail_rank=2,
                                 gate_threshold=.3, mass_budget=.25)
                out_w, out_i, stats = reference_policy(weights, ids, 23, cfg)
                counts = [sum(e in row for row in ids) for e in range(23)]
                for wi, ii, wo, io in zip(weights, ids, out_w, out_i):
                    removed = [j for j in range(5) if io[j] == -1]
                    self.assertLess(len(removed), 5)
                    self.assertLessEqual(sum(wi[j] for j in removed)/sum(wi), .25+1e-12)
                    self.assertAlmostEqual(sum(wi), sum(wo))
                    self.assertTrue(all(counts[ii[j]] == 1 for j in removed))
                self.assertEqual(stats["skipped_slots"],
                                 stats["distinct_before"]-stats["distinct_after"])


@unittest.skipUnless(HAS_TORCH, "torch not installed on local CPU host")
class TorchParity(unittest.TestCase):
    def test_padding_preserved_but_excluded_from_actions_and_counts(self):
        import torch
        weights=torch.tensor([[.8,.1,.07,.03]]*4)
        ids=torch.tensor([[0,1,2,3],[4,5,6,7],[8,9,10,11],[0,1,2,3]])
        policy=ExpertSkipPolicy(SkipConfig(mode='marginal',tail_rank=2,
                              gate_threshold=.1,mass_budget=.11))
        w,i=policy('layer',weights,ids,16,False,100,active_tokens=torch.tensor(3))
        self.assertEqual(i.tolist(),[[0,1,-1,-1],[4,5,-1,-1],[8,9,-1,-1],[0,1,2,3]])
        torch.testing.assert_close(w[-1],weights[-1],rtol=0,atol=0)
        self.assertEqual(policy.snapshot()['totals']['tokens'],3)
        self.assertEqual(policy.snapshot()['totals']['skipped_slots'],6)
        uncounted=ExpertSkipPolicy(policy.config,collect_stats=False)
        uw,ui=uncounted('layer',weights,ids,16,False,100,active_tokens=torch.tensor(3))
        torch.testing.assert_close(uw,w,rtol=0,atol=0)
        torch.testing.assert_close(ui,i,rtol=0,atol=0)
        self.assertEqual(uncounted.snapshot()['layers'],{})

    def test_alignment_sentinel_preserves_zero_output_blocks(self):
        import torch
        base = "vllm.model_executor.layers.fused_moe"
        calls = []

        def native(ids, block, experts, expert_map, **kwargs):
            calls.append((ids, block, experts, expert_map, kwargs))
            return ids, expert_map, torch.tensor([0])

        modules = {}
        for suffix in (".moe_align_block_size", ".fused_moe", ".experts.triton_moe"):
            module = types.ModuleType(base+suffix)
            module.moe_align_block_size = native
            modules[base+suffix] = module
        with mock.patch.dict(sys.modules, modules):
            handle = install_safe_alignment()
            wrapper = modules[base+".fused_moe"].moe_align_block_size
            wrapper(torch.tensor([[1, -1], [2, 0]], dtype=torch.int32), 16, 3,
                    ignore_invalid_experts=True)
            ids, block, experts, mapping, kwargs = calls[-1]
            self.assertEqual(ids.tolist(), [[1, 3], [2, 0]])
            self.assertEqual(mapping.tolist(), [0, 1, 2, -1])
            self.assertEqual(experts, 4)
            self.assertFalse(kwargs["ignore_invalid_experts"])
            handle.restore()
            self.assertTrue(all(m.moe_align_block_size is native for m in modules.values()))

    def test_torch_randomized_reference_and_counters(self):
        import torch
        rng = random.Random(271)
        for mode in ("dense", "global", "gate", "marginal", "rank_gate"):
            cfg = SkipConfig(mode=mode, drop_j=2, tail_rank=2,
                             gate_threshold=.25, mass_budget=.3)
            policy = ExpertSkipPolicy(cfg)
            for b in (1, 4, 16):
                ids = [rng.sample(range(17), 4) for _ in range(b)]
                weights = [[rng.random() for _ in range(4)] for _ in range(b)]
                expected_w, expected_i, expected_s = reference_policy(weights, ids, 17,
                                                                     cfg, True, 123)
                out_w, out_i = policy("layer", torch.tensor(weights), torch.tensor(ids),
                                     17, True, 123)
                self.assertEqual(out_i.tolist(), expected_i)
                torch.testing.assert_close(out_w, torch.tensor(expected_w))
                stats = policy.snapshot(reset=True)["totals"]
                for key in expected_s:
                    self.assertAlmostEqual(stats[key], expected_s[key], places=5)
            self.assertEqual(policy.snapshot()["totals"]["events"], 0)


@unittest.skipUnless(HAS_TORCH and os.environ.get("EXPERT_SKIP_TEST_CUDA") == "1",
                     "set EXPERT_SKIP_TEST_CUDA=1 only while holding the GPU lock")
class CudaPolicyParity(unittest.TestCase):
    def test_runtime_active_tokens_and_padding_under_graph_replay(self):
        import torch
        weights=torch.tensor([[.8,.1,.07,.03]]*4,device='cuda')
        ids=torch.tensor([[0,1,2,3],[4,5,6,7],[8,9,10,11],[0,1,2,3]],device='cuda',dtype=torch.int32)
        active=torch.tensor(3,device='cuda',dtype=torch.int32)
        for backend in ('triton','triton_multi'):
            policy=ExpertSkipPolicy(SkipConfig(mode='marginal',tail_rank=2,
                gate_threshold=.1,mass_budget=.11,backend=backend))
            def call():
                return policy('layer',weights,ids,16,False,100,active_tokens=active)
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):call()
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                w,i=call()
            for real in (3,1,0):
                policy.reset()
                active.fill_(real)
                graph.replay()
                stats=policy.snapshot()['totals']
                self.assertEqual(stats['tokens'],real)
                self.assertEqual(stats['skipped_slots'],real*2)
                self.assertEqual(stats['events'],int(real>0))
                torch.testing.assert_close(w[real:],weights[real:],rtol=0,atol=0)
                torch.testing.assert_close(i[real:],ids[real:],rtol=0,atol=0)
            active.fill_(3)
    def test_triton_all_modes_unsorted_and_non_power_two_k(self):
        import torch
        torch.manual_seed(1701)
        for mode in ("dense", "global", "gate", "marginal", "rank_gate"):
            for batch, k, e in ((1, 6, 32), (8, 8, 64), (64, 8, 128)):
                config = dict(mode=mode, drop_j=2, tail_rank=2,
                              gate_threshold=.2, mass_budget=.25)
                weights = torch.rand(batch, k, device="cuda")
                ids = torch.stack([torch.randperm(e, device="cuda")[:k]
                                   for _ in range(batch)]).to(torch.int32)
                pt = ExpertSkipPolicy(SkipConfig(**config))
                wt, it = pt("layer", weights, ids, e, True, 100)
                st = pt.snapshot()["totals"]
                for backend in ("triton", "triton_multi"):
                    tr = ExpertSkipPolicy(SkipConfig(**config, backend=backend))
                    wg, ig = tr("layer", weights, ids, e, True, 100)
                    torch.testing.assert_close(ig, it, rtol=0, atol=0)
                    torch.testing.assert_close(wg, wt, rtol=2e-6, atol=1e-7)
                    sg = tr.snapshot()["totals"]
                    for key in st:
                        self.assertAlmostEqual(st[key], sg[key], delta=max(1e-5, abs(st[key])*1e-6))

    def test_triton_graph_replay_updates_device_counters(self):
        import torch
        cfg = SkipConfig(mode="marginal", tail_rank=2, gate_threshold=.2,
                         mass_budget=.2, backend="triton")
        policy = ExpertSkipPolicy(cfg)
        weights = torch.tensor([[.8, .1, .07, .03]], device="cuda").repeat(8, 1)
        ids = torch.arange(32, device="cuda", dtype=torch.int32).reshape(8, 4)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                policy("layer", weights, ids, 64, True, 10)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out_weights, out_ids = policy("layer", weights, ids, 64, True, 10)
        policy.reset()
        for _ in range(3):
            graph.replay()
        stats = policy.snapshot()["totals"]
        self.assertEqual(stats["events"], 3)
        self.assertEqual(stats["skipped_slots"], 48)
        self.assertEqual(stats["logical_weight_bytes_before"]-stats["logical_weight_bytes_after"], 480)
        self.assertTrue(torch.isfinite(out_weights).all().item())
        self.assertEqual((out_ids < 0).sum().item(), 16)


if __name__ == "__main__":
    unittest.main()
