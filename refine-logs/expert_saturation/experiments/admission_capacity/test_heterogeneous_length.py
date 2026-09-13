#!/usr/bin/env python3
"""CPU checks for the per-request output-length patch, before any GPU run.

These exercise the patched `native_capture.capture_episode` against a fake
engine, because the failure this guards against is silent: if the per-request
length lookup were wrong, every episode would still complete and only the
generated token counts would be off, which is exactly the quantity the
experiment measures.

What is and is not covered
--------------------------
Covered: the length actually requested per request, the scalar fallback used by
warmup, rejection of a workload whose lengths do not cover the requests, and
rejection of a workload that sets both a scalar and a per-request mapping.

Not covered: the vLLM scheduler, KV allocation, the rotation action, or any
timing. This is a fixture, not a GPU result.
"""

import json, sys, types, unittest
from pathlib import Path

FROZEN = Path(__file__).resolve().parent.parent.parent / (
    "outputs/admission_capacity/20260913_heterogeneous_length_r01/execution/frozen")


def load_patched_capture():
    """Import the patched native_capture with a stub `vllm` package."""
    sys.path.insert(0, str(FROZEN))
    vllm = types.ModuleType("vllm")

    class SamplingParams:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    vllm.SamplingParams = SamplingParams
    sp = types.ModuleType("vllm.sampling_params")
    sp.RequestOutputKind = types.SimpleNamespace(CUMULATIVE="CUMULATIVE")
    sys.modules.setdefault("vllm", vllm)
    sys.modules.setdefault("vllm.sampling_params", sp)
    import native_capture
    return native_capture


class FakeScheduler:
    def __init__(self):
        self.requests, self.running, self.waiting = {}, [], []

    def get_request_counts(self):
        return len(self.running), len(self.waiting)

    def schedule(self):
        raise AssertionError("fake scheduler must not be driven")


class RecordingEngine:
    """Captures the SamplingParams handed to add_request, then stops."""

    def __init__(self):
        self.seen = {}
        self.engine_core = types.SimpleNamespace(
            engine_core=types.SimpleNamespace(scheduler=FakeScheduler()))
        self.vllm_config = types.SimpleNamespace(
            scheduler_config=types.SimpleNamespace(async_scheduling=False, stream_interval=1))

    def add_request(self, external_id, prompt, params, arrival_time=None):
        self.seen[external_id.split("/", 1)[1]] = (params.max_tokens, params.min_tokens,
                                                   params.ignore_eos)
        return f"internal-{len(self.seen)}"

    def has_unfinished_requests(self):
        return False

    def step(self):
        return []


def make_workload(ids, lengths=None):
    w = dict(
        source_requests=[dict(request_id=r, document_id=r) for r in ids],
        actual_prompt_token_ids=[[1, 2, 3] for _ in ids],
        arrival_traces_s={"steady": [0.0 for _ in ids]},
    )
    if lengths is not None:
        w["output_lengths_by_request"] = dict(zip(ids, lengths))
    return w


class PerRequestLengthTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nc = load_patched_capture()

    def run_episode(self, workload, config):
        engine = RecordingEngine()
        try:
            self.nc.capture_episode(engine, workload, config, regime="steady",
                                    arrival_scale=0.0, run_id="t", max_seconds=5)
        except Exception:
            pass                      # the fake engine finishes immediately
        return engine.seen

    def test_each_request_gets_its_own_length(self):
        ids = [f"r{i}" for i in range(4)]
        lengths = [1024, 3072, 1024, 3072]
        seen = self.run_episode(make_workload(ids, lengths), dict(cap=4))
        self.assertEqual(len(seen), 4, "every request must be admitted")
        for rid, want in zip(ids, lengths):
            self.assertEqual(seen[rid], (want, want, True),
                             f"{rid} must request exactly {want} tokens with ignore_eos")

    def test_scalar_fallback_still_works_for_warmup(self):
        ids = [f"w{i}" for i in range(3)]
        seen = self.run_episode(make_workload(ids), dict(cap=3, output_tokens=16))
        self.assertTrue(seen)
        for rid in ids:
            self.assertEqual(seen[rid], (16, 16, True))

    def test_incomplete_length_map_is_rejected(self):
        ids = ["a", "b", "c"]
        w = make_workload(ids, [8, 8, 8])
        del w["output_lengths_by_request"]["c"]
        with self.assertRaises(ValueError) as ctx:
            self.nc.capture_episode(RecordingEngine(), w, dict(cap=3),
                                    regime="steady", arrival_scale=0.0, run_id="t")
        self.assertIn("cover the source requests", str(ctx.exception))

    def test_scalar_and_map_together_is_rejected(self):
        ids = ["a", "b"]
        with self.assertRaises(ValueError) as ctx:
            self.nc.capture_episode(RecordingEngine(), make_workload(ids, [8, 8]),
                                    dict(cap=2, output_tokens=8),
                                    regime="steady", arrival_scale=0.0, run_id="t")
        self.assertIn("not both", str(ctx.exception))

    def test_degenerate_length_is_rejected(self):
        ids = ["a", "b"]
        with self.assertRaises(ValueError):
            self.nc.capture_episode(RecordingEngine(), make_workload(ids, [8, 1]),
                                    dict(cap=2), regime="steady", arrival_scale=0.0, run_id="t")


class WorkloadIdentityTest(unittest.TestCase):
    """The frozen pair workload must keep the deficit identical across arms."""

    @classmethod
    def setUpClass(cls):
        base = FROZEN / "inputs_preparation/prepared/pair"
        cls.workload = json.load(open(base / "workload.json"))
        cls.config = json.load(open(base / "config.json"))

    def test_arms_reserve_identical_blocks(self):
        a, b = self.config["arms"]["homogeneous"], self.config["arms"]["heterogeneous"]
        self.assertEqual(a["reserved_blocks"], b["reserved_blocks"])
        self.assertEqual(a["total_output_tokens"], b["total_output_tokens"])
        self.assertEqual(a["mean_output_tokens"], b["mean_output_tokens"])
        self.assertGreater(b["cv_output_tokens"], a["cv_output_tokens"])

    def test_deficit_is_positive_and_matches_the_original_domain(self):
        for arm in self.config["arms"].values():
            self.assertGreater(arm["deficit_blocks"], 0, "no deficit means no preemption")
            self.assertAlmostEqual(arm["deficit_request_footprints"], 2.035, delta=0.05,
                                   msg="deficit must stay near the original 2.035 footprints")

    def test_prompts_are_unique_and_correctly_sized(self):
        seqs = self.workload["actual_prompt_token_ids"]
        self.assertEqual(len(seqs), self.config["requests"])
        self.assertTrue(all(len(s) == self.config["prompt_tokens"] for s in seqs))
        self.assertEqual(len({tuple(s) for s in seqs}), len(seqs), "duplicate prompts")

    def test_document_reuse_is_declared(self):
        docs = {r["source_request_id"] for r in self.workload["source_requests"]}
        self.assertLess(len(docs), self.config["requests"],
                        "this workload does reuse documents; the claim must stay declared")
        self.assertIn("NOT independent documents",
                      self.workload["document_identity_scope"])

    def test_lengths_interleave_rather_than_block(self):
        het = self.config["arms"]["heterogeneous"]["output_lengths"]
        self.assertNotEqual(het[0], het[1], "length classes must alternate by arrival")
        self.assertEqual(len(set(het)), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
