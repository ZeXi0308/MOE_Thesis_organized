import importlib.util
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from expert_skip import ExpertSkipPolicy, SkipConfig, reference_policy

HAS_TORCH = importlib.util.find_spec("torch") is not None


class PolicySemantics(unittest.TestCase):
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
    def test_torch_randomized_reference_and_counters(self):
        import torch
        rng = random.Random(271)
        for mode in ("dense", "global", "gate", "marginal"):
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


if __name__ == "__main__":
    unittest.main()
