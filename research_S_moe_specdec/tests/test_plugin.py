"""CPU contract checks; real vLLM/GPU smoke is a separate integration run."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import vllm_s_plugin as plugin


class Request:
    def __init__(self, rid, drafts, *, prefill=False):
        self.request_id = rid
        self.spec_token_ids = list(drafts)
        self.num_prompt_tokens = 4
        self.num_computed_tokens = 0 if prefill else 4
        self.is_prefill_chunk = prefill
        self.finished = False

    def is_finished(self):
        return self.finished


class Scheduler:
    def __init__(self, requests, native=False):
        self.running = requests
        self.requests = {r.request_id: r for r in requests}
        self.scheduler_config = NS(async_scheduling=False)
        self.num_spec_tokens = 7
        self.dynamic_sd_lookup = [0, 7] if native else None
        self.observed_lengths = []

    def schedule(self):
        self.observed_lengths = [len(r.spec_token_ids) for r in self.running]
        drafts = {r.request_id: r.spec_token_ids for r in self.running if r.spec_token_ids}
        counts = {r.request_id: len(r.spec_token_ids) + 1 for r in self.running}
        for req in self.running:
            req.spec_token_ids = []
        return NS(num_scheduled_tokens=counts, scheduled_spec_decode_tokens=drafts,
                  total_num_scheduled_tokens=sum(counts.values()),
                  num_spec_tokens_to_schedule=5)

    def make_spec_decoding_stats(self, *args):
        return None  # Equivalent to log_stats=False; still must observe samples.

    def update_from_output(self, output, finish=False):
        for rid, ids in output.scheduled_spec_decode_tokens.items():
            self.make_spec_decoding_stats(None, len(ids), min(1, len(ids)), None, rid)
            self.requests[rid].finished = finish
        return {"native_output": True}


class Acceptance:
    alpha = 0.7
    def __init__(self):
        self.calls = []

    def update(self, accepted, proposed):
        self.calls.append((accepted, proposed))


class PluginTests(unittest.TestCase):
    def setUp(self):
        self.original_active = plugin._ACTIVE
        self.original_methods = {name: getattr(Scheduler, name) for name in
                                 ("schedule", "update_from_output", "make_spec_decoding_stats")}
        Scheduler.schedule = plugin._schedule_wrapper(Scheduler.schedule)
        Scheduler.update_from_output = plugin._update_wrapper(Scheduler.update_from_output)
        Scheduler.make_spec_decoding_stats = plugin._stats_wrapper(Scheduler.make_spec_decoding_stats)

    def tearDown(self):
        plugin._ACTIVE = self.original_active
        for name, original in self.original_methods.items():
            setattr(Scheduler, name, original)

    def activate(self, **kwargs):
        runtime = plugin.Runtime(plugin.PluginConfig(**kwargs))
        plugin._ACTIVE = runtime
        runtime.start_episode("test")
        return runtime

    def test_truncation_precedes_native_allocation_and_preserves_full_proposal_cost(self):
        runtime = self.activate(static_gamma=2)
        scheduler = Scheduler([Request("a", range(7)), Request("b", range(3))])
        output = scheduler.schedule()
        self.assertEqual(scheduler.observed_lengths, [2, 2])
        self.assertEqual(output.num_spec_tokens_to_schedule, 7)
        self.assertEqual(runtime.steps[0]["truncated_by_controller"], 6)
        self.assertEqual(runtime.steps[0]["scheduled_drafts"], 4)
        self.assertEqual(output.num_scheduled_tokens, {"a": 3, "b": 3})

    def test_zero_gamma_removes_all_current_drafts(self):
        runtime = self.activate(mode="disabled")
        scheduler = Scheduler([Request("a", range(7))])
        output = scheduler.schedule()
        self.assertEqual(output.scheduled_spec_decode_tokens, {})
        self.assertEqual(runtime.steps[0]["scheduled_tokens"], 1)

    def test_moe_target_and_draft_same_layer_class_keep_separate_route_populations(self):
        # Granite target/drafter can share a RoutedExperts class and layer name.
        # Native DraftModelProposer calls are inside propose_draft_token_ids;
        # model_phase, not a name prefix, must distinguish the two populations.
        runtime = self.activate(static_gamma=4, expert_weight_bytes=16)
        runtime.trace = True
        scheduler = Scheduler([Request("a", range(4))])
        output = scheduler.schedule()

        class Layer:
            layer_name = "model.layers.0.mlp.experts"
            def __init__(self, experts):
                self.global_num_experts = experts

        class IDs:
            def __init__(self, rows):
                self.shape = (rows, 8)
            def __getitem__(self, key):
                return IDs(min(self.shape[0], key.stop))

        target, draft = Layer(40), Layer(32)
        hist_calls, forward_calls = [], []
        runtime._device_route_histogram = lambda ids, experts: hist_calls.append(
            (ids.shape[0], experts))
        forward = plugin._modular_wrapper(lambda layer, *args: forward_calls.append(layer))
        draft_forward = plugin._phase_wrapper("draft")(
            lambda runner: forward(draft, None, None, IDs(1)))
        def target_forward(runner, scheduler_output):
            forward(target, None, None, IDs(5))
            draft_forward(runner)
            self.assertEqual(runtime._phase, "target")
        plugin._phase_wrapper("target")(target_forward)(NS(), output)

        metadata = [meta for meta, _ in runtime._route_tensors]
        self.assertEqual(forward_calls, [target, draft])
        self.assertEqual(hist_calls, [(5, 40), (1, 32)])
        self.assertEqual([(r["model_phase"], r["tokens"], r["num_experts"])
                          for r in metadata], [("target", 5, 40), ("draft", 1, 32)])
        self.assertEqual(metadata[0]["verification_population"],
                         "all_scheduled_positions_including_rejected_drafts")
        self.assertEqual(metadata[1]["verification_population"], "executed_draft_positions")
        self.assertEqual(runtime._phase, "target")

    def test_dense_threshold_counts_decode_candidates(self):
        runtime = self.activate(mode="dense_heuristic", static_gamma=3, disable_above_batch=1)
        scheduler = Scheduler([Request("a", range(7)), Request("p", [], prefill=True)])
        scheduler.schedule()
        self.assertEqual(runtime.steps[0]["control_batch"], 1)
        self.assertEqual(runtime.steps[0]["gamma"], 3)
        self.assertEqual(runtime.steps[0]["phase"], "mixed")

    def test_native_lookup_and_current_drafts_are_unchanged(self):
        runtime = self.activate(mode="native")
        scheduler = Scheduler([Request("a", range(6))], native=True)
        output = scheduler.schedule()
        self.assertEqual(scheduler.observed_lengths, [6])
        self.assertEqual(output.num_spec_tokens_to_schedule, 5)
        self.assertEqual(runtime.steps[0]["truncated_by_controller"], 0)

    def test_native_stats_observed_even_without_native_log_stats(self):
        acceptance = Acceptance()
        runtime = self.activate(static_gamma=2, acceptance=acceptance)
        scheduler = Scheduler([Request("a", range(7))])
        output = scheduler.schedule()
        result = scheduler.update_from_output(output)
        self.assertEqual(result, {"native_output": True})
        self.assertEqual(acceptance.calls, [([1], [2])])
        self.assertEqual(runtime.steps[0]["accepted_drafts"], 1)

    def test_termination_censoring_preserves_raw_stats(self):
        acceptance = Acceptance()
        runtime = self.activate(static_gamma=2, acceptance=acceptance)
        scheduler = Scheduler([Request("a", range(7))])
        output = scheduler.schedule()
        scheduler.update_from_output(output, finish=True)
        self.assertEqual(acceptance.calls, [])
        self.assertEqual(runtime.steps[0]["accepted_drafts"], 1)
        self.assertEqual(runtime.steps[0]["ewma_censored_requests"], ["a"])

    def test_policy_receives_actual_paid_draft_budget(self):
        calls = []
        class Policy:
            def choose(self, batch, draft_gamma=None):
                calls.append((batch, draft_gamma))
                return {"gamma": 1}
        runtime = self.activate(mode="adaptive", policy=Policy())
        scheduler = Scheduler([Request("a", range(7))])
        scheduler.schedule()
        self.assertEqual(calls, [(1, 7)])
        self.assertEqual(runtime.steps[0]["scheduled_drafts"], 1)

    def test_uncovered_calibration_falls_back_but_bad_policy_still_raises(self):
        class Missing:
            def choose(self, batch, draft_gamma=None):
                raise ValueError(f"no calibrated candidate for batch={batch}: []")
        runtime = self.activate(mode="adaptive", policy=Missing())
        Scheduler([Request("a", range(7))]).schedule()
        self.assertEqual(runtime.steps[0]["gamma"], 0)
        self.assertEqual(runtime.steps[0]["decision"]["fallback"], "uncovered_calibration")
        class Broken:
            def choose(self, batch, draft_gamma=None):
                raise ValueError("malformed calibration payload")
        plugin._ACTIVE = plugin.Runtime(plugin.PluginConfig(mode="adaptive", policy=Broken()))
        with self.assertRaisesRegex(ValueError, "malformed"):
            Scheduler([Request("a", range(7))]).schedule()

    def test_zero_draft_terminal_request_is_censored(self):
        runtime = self.activate(static_gamma=0)
        req = Request("a", [])
        scheduler = Scheduler([req])
        output = scheduler.schedule()
        req.finished = True
        scheduler.update_from_output(output)
        self.assertEqual(runtime.steps[0]["ewma_censored_requests"], ["a"])

    def test_zero_proposal_availability_is_observed(self):
        from cost_model import AcceptanceEWMA
        acceptance = AcceptanceEWMA()
        runtime = self.activate(static_gamma=2, acceptance=acceptance)
        scheduler = Scheduler([Request("a", []), Request("b", [])])
        output = scheduler.schedule()
        scheduler.update_from_output(output)
        self.assertEqual(acceptance.expected_tokens(7), 1.0)
        self.assertTrue(runtime.steps[0]["availability_updated"])

    def test_episode_acceptance_snapshot_and_native_without_estimator(self):
        class SnapshotAcceptance(Acceptance):
            def __init__(self):
                super().__init__()
                self.snapshot_calls = 0

            def snapshot(self):
                self.snapshot_calls += 1
                return {"alpha_estimator": "exposure_ratio",
                        "alpha_prior": {"alpha": .7, "tested_count": 1., "accepted_count": .7},
                        "alpha_counts": {"observed_accepted": 1, "observed_tested": 2}}

        acceptance = SnapshotAcceptance()
        runtime = self.activate(mode="native", acceptance=acceptance)
        scheduler = Scheduler([Request("a", [1, 2])])
        output = scheduler.schedule()
        scheduler.update_from_output(output)
        self.assertEqual(acceptance.snapshot_calls, 0)
        result = runtime.end_episode()
        self.assertEqual(acceptance.snapshot_calls, 1)
        self.assertEqual(result["config"]["acceptance_class"], "SnapshotAcceptance")
        self.assertEqual(result["acceptance_snapshot"]["alpha_estimator"], "exposure_ratio")
        self.assertEqual(result["acceptance_snapshot"]["alpha_prior"]["tested_count"], 1.)
        self.assertEqual(result["acceptance_snapshot"]["alpha_counts"]["observed_tested"], 2)
        runtime = self.activate(mode="native")
        result = runtime.end_episode()
        self.assertIsNone(result["config"]["acceptance_class"])
        self.assertIsNone(result["acceptance_snapshot"])

    def test_selected_proposal_budget_changes_next_draft_request(self):
        self.activate(static_gamma=2, proposal_budget="selected")
        output = Scheduler([Request("a", range(7))]).schedule()
        self.assertEqual(output.num_spec_tokens_to_schedule, 2)

    def test_two_controllers_are_rejected(self):
        self.activate(static_gamma=2)
        with self.assertRaisesRegex(RuntimeError, "two simultaneous"):
            Scheduler([Request("a", range(7))], native=True).schedule()

    def test_graph_trace_requires_native_flag_but_graph_perf_can_run(self):
        runtime = self.activate(strict=False)
        runtime.end_episode()
        runtime._engine_eager = False
        runtime.start_episode("graph_perf", trace=False)
        runtime.end_episode()
        with self.assertRaisesRegex(RuntimeError, "bypasses Python"):
            runtime.start_episode("bad_graph_trace", trace=True)

    def test_native_mode_requires_engine_flag_before_capture(self):
        runtime = plugin.Runtime(plugin.PluginConfig(native_route_return=True))
        scheduler = Scheduler([Request("a", [])])
        scheduler.vllm_config = NS(model_config=NS(enforce_eager=False,
                                                  enable_return_routed_experts=False))
        with self.assertRaisesRegex(RuntimeError, "enable_return_routed_experts=True"):
            runtime._validate_scheduler(scheduler)
        scheduler.vllm_config.model_config.enable_return_routed_experts = True
        runtime._validate_scheduler(scheduler)
        self.assertTrue(runtime._native_route_enabled)

    def test_native_gpu_boundary_observes_every_verification_row_on_each_replay(self):
        runtime = self.activate(static_gamma=2, native_route_return=True,
                                expert_weight_bytes={"model.layers.0.mlp": 16,
                                                     "model.layers.2.mlp": 32})
        runtime.trace = True
        events, captured = [], []
        class Buffer:
            shape = (8, 3, 2)  # Dense layer 1 deliberately has no routed experts.
            version = 0
            def __getitem__(self, key):
                rows, layer_id, _ = key
                return (rows.stop, layer_id, self.version)
        buffer = Buffer()
        capturer = NS(get_device_buffer=lambda: buffer)
        layers = [NS(layer_name=f"model.layers.{index}.mlp", global_num_experts=4,
                     local_num_experts=4, quant_method=NS(is_monolithic=False)) for index in (0, 2)]
        modules = [NS(layer_id=index, routed_experts=layer,
                      router=NS(top_k=2, capture_fn=lambda ids: None))
                   for index, layer in zip((0, 2), layers)]
        runner = NS(model=NS(modules=lambda: modules), routed_experts_initialized=True,
                    routed_experts_capturer=capturer, input_batch=NS(req_ids=["a"]))
        runtime.register_native_layers(runner, capturer)
        self.assertEqual([row["native_layer_id"] for row in runtime.native_layer_mapping()], [0, 2])
        self.assertEqual([row["bytes_per_expert"] for row in runtime.native_layer_mapping()], [16, 32])
        def fake_route_capture(layer, ids, **kwargs):
            events.append("snapshot")
            captured.append((layer.layer_name, ids, kwargs["route_path"]))
        runtime.capture_routes = fake_route_capture
        def graph_replay(runner, output):
            # A replay never invokes the Python RoutedExperts.forward_modular hook.
            events.append("graph")
            buffer.version += 1
            return None
        execute = plugin._phase_wrapper("target")(graph_replay)
        request = Request("a", range(7))
        scheduler = Scheduler([request])
        scheduler.vllm_config = NS(model_config=NS(enforce_eager=False,
                                                  enable_return_routed_experts=True))
        for _ in range(2):
            request.spec_token_ids = list(range(7))
            output = scheduler.schedule()
            execute(runner, output)
        self.assertEqual(events, ["graph", "snapshot", "snapshot"] * 2)
        self.assertEqual([item[1] for item in captured], [(3, 0, 1), (3, 2, 1), (3, 0, 2), (3, 2, 2)])
        self.assertEqual([step["native_route_rows"] for step in runtime.steps], [3, 3])

    def test_unfilled_native_columns_are_detected_at_histogram_drain(self):
        runtime = self.activate()
        meta = {"tokens": 3, "top_k": 2, "bytes_per_expert": 16,
                "layer": "model.layers.0.mlp", "step": 0}
        runtime._finalize_route_histogram(meta, [6, 0, 0, 0, 0])
        self.assertTrue(any("unfilled native buffer" in error for error in runtime._route_errors))

    def test_actual_forward_graph_mode_recording_is_trace_only(self):
        runtime = self.activate()
        Scheduler([Request("a", [])]).schedule()
        calls = []
        def context():
            calls.append("context")
            return NS(cudagraph_runtime_mode="FULL", batch_descriptor=NS(num_tokens=8))
        wrapped = plugin._forward_mode_wrapper(lambda runner: "output", get_context=context)
        self.assertEqual(wrapped(None), "output")
        self.assertEqual(calls, [])
        runtime.trace = True
        wrapped(None)
        self.assertEqual(calls, ["context"])
        self.assertEqual(runtime.steps[0]["target_cudagraph_runtime_mode"], "FULL")
        self.assertEqual(runtime.steps[0]["target_cudagraph_padded_tokens"], 8)

    def test_graph_padding_requires_gpu_native_return_and_single_gpu_scope(self):
        with self.assertRaisesRegex(ValueError, "requires native_route_return"):
            plugin.PluginConfig(native_capture_padding=True)
        with self.assertRaisesRegex(ValueError, "native_route_backend='gpu'"):
            plugin.PluginConfig(native_capture_padding=True, native_route_return=True, native_route_backend="cpu")
        runtime = self.activate(native_capture_padding=True, native_route_return=True, strict=False)
        scheduler = Scheduler([Request("a", [])])
        scheduler.vllm_config = NS(model_config=NS(enforce_eager=False, enable_return_routed_experts=True))
        scheduler.parallel_config = NS(tensor_parallel_size=1, pipeline_parallel_size=1,
                                       data_parallel_size=2, enable_eplb=False)
        with self.assertRaisesRegex(RuntimeError, "single-GPU, single-DP"):
            runtime._validate_scheduler(scheduler)
        scheduler.parallel_config.data_parallel_size = 1
        scheduler.parallel_config.enable_dbo = True
        with self.assertRaisesRegex(RuntimeError, "DBO disabled"):
            runtime._validate_scheduler(scheduler)

    def test_graph_padding_drain_combines_populations_without_changing_valid_rows(self):
        runtime = self.activate()
        valid = {"tokens": 3, "top_k": 2, "bytes_per_expert": 16, "layer": "layer0", "step": 0}
        runtime._finalize_route_histogram(valid, [2, 2, 2, 0, 0])
        unchanged = dict(valid)
        physical = {"scheduled_tokens": 3, "padding_tokens": 2, "physical_tokens": 5,
                    "num_experts": 4, "top_k": 2, "bytes_per_expert": 16, "layer": "layer0", "step": 0}
        runtime._finalize_graph_padding(valid, physical, [1, 0, 1, 2, 0])
        self.assertEqual(valid, unchanged)
        self.assertEqual(physical["padding_expert_token_counts"], [1, 0, 1, 2])
        self.assertEqual(physical["physical_expert_token_counts"], [3, 2, 3, 2])
        self.assertEqual(physical["physical_distinct_experts"], 4)
        self.assertEqual(physical["padding_only_new_experts"], 1)
        self.assertEqual(physical["padding_only_new_expert_ids"], [3])
        self.assertEqual(physical["physical_selected_expert_storage_bytes"], 64)
        self.assertTrue(physical["complete"])
        zero = dict(physical, physical_tokens=3, padding_tokens=0)
        runtime._finalize_graph_padding(valid, zero, [0, 0, 0, 0, 0])
        self.assertEqual(zero["physical_expert_token_counts"], valid["expert_token_counts"])
        self.assertEqual(zero["padding_only_new_experts"], 0)
        self.assertTrue(zero["complete"])

    def test_graph_padding_detects_unfilled_and_invalid_histograms(self):
        runtime = self.activate()
        valid = {"expert_token_counts": [2, 2, 2, 0], "invalid_expert_assignments": 0}
        meta = {"scheduled_tokens": 3, "padding_tokens": 2, "physical_tokens": 5,
                "num_experts": 4, "top_k": 2, "bytes_per_expert": 16, "layer": "layer0", "step": 0}
        for histogram in ([4, 0, 0, 0, 0], [1, 0, 1, 1, 1], [0, 0, 0, 0, 0]):
            output = dict(meta)
            runtime._finalize_graph_padding(valid, output, histogram)
            self.assertFalse(output["complete"])
        self.assertTrue(any("Invalid/unfilled graph padding" in error for error in runtime._route_errors))

    def test_native_padding_missing_or_invalid_physical_count_is_incomplete(self):
        for padded in (None, 2, 9):
            with self.subTest(padded=padded):
                runtime = self.activate(native_route_return=True, native_capture_padding=True,
                                        static_gamma=2, expert_weight_bytes=16)
                runtime.trace = True
                scheduler = Scheduler([Request("a", range(7))])
                scheduler.vllm_config = NS(model_config=NS(enforce_eager=False, enable_return_routed_experts=True))
                output = scheduler.schedule()
                if padded is not None:
                    runtime._current["target_cudagraph_padded_tokens"] = padded
                class Buffer:
                    shape = (8, 1, 2)
                    def __getitem__(self, key):
                        return key
                runtime._native_layers = {0: NS(layer_name="layer0", global_num_experts=4)}
                runtime.capture_routes = lambda *args, **kwargs: None
                runner = NS(routed_experts_initialized=True, routed_experts_capturer=NS(get_device_buffer=lambda: Buffer()),
                            input_batch=NS(req_ids=["a"]))
                runtime.capture_native_gpu(runner, output)
                result = runtime.end_episode()
                self.assertFalse(result["summary"]["trace_complete"])
                self.assertEqual(result["graph_padding_routes"], [])
                self.assertTrue(any("cannot infer zero padding" in error for error in result["summary"]["route_errors"]))

    def test_native_padding_slices_physical_rows_and_has_no_trace_off_work(self):
        runtime = self.activate(native_route_return=True, native_capture_padding=True, static_gamma=2)
        scheduler = Scheduler([Request("a", range(7))])
        scheduler.vllm_config = NS(model_config=NS(enforce_eager=False, enable_return_routed_experts=True))
        output = scheduler.schedule()
        calls = []
        class Buffer:
            shape = (8, 1, 2)
            def __getitem__(self, key):
                rows, layer, _ = key
                return (rows.start or 0, rows.stop, layer)
        def get_buffer():
            calls.append("buffer")
            return Buffer()
        runtime._native_layers = {0: NS(layer_name="layer0", global_num_experts=4)}
        runtime.capture_routes = lambda layer, ids, **kwargs: (calls.append(("valid", ids)) or {"tokens": 3})
        runtime._capture_graph_padding = lambda layer, ids, meta, p: calls.append(("padding", ids, p))
        runner = NS(routed_experts_initialized=True, routed_experts_capturer=NS(get_device_buffer=get_buffer),
                    input_batch=NS(req_ids=["a"]))
        runtime.capture_native_gpu(runner, output)
        self.assertEqual(calls, [])
        runtime.trace = True
        runtime._current["target_cudagraph_padded_tokens"] = 5
        runtime.capture_native_gpu(runner, output)
        self.assertEqual(calls, ["buffer", ("valid", (0, 3, 0)), ("padding", (3, 5, 0), 5)])
        self.assertEqual(runtime.steps[0]["native_padding_route_rows"], 2)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed locally")
    def test_real_native_padding_deferred_drain_survives_buffer_reuse(self):
        import torch
        runtime = self.activate(native_route_return=True, native_capture_padding=True, static_gamma=2,
                                expert_weight_bytes=16)
        runtime.trace = True
        scheduler = Scheduler([Request("a", range(7))])
        scheduler.vllm_config = NS(model_config=NS(enforce_eager=False, enable_return_routed_experts=True))
        runtime._native_layers = {0: NS(layer_name="layer0", global_num_experts=4)}
        data = torch.tensor([[[0, 1]], [[1, 2]], [[0, 2]], [[2, 3]], [[3, 0]]])
        runner = NS(routed_experts_initialized=True, routed_experts_capturer=NS(get_device_buffer=lambda: data),
                    input_batch=NS(req_ids=["a"]))
        for padded in (5, 3):
            scheduler.running[0].spec_token_ids = list(range(7))
            output = scheduler.schedule()
            runtime._current.update(target_cudagraph_padded_tokens=padded, target_cudagraph_runtime_mode="FULL")
            runtime.capture_native_gpu(runner, output)
        self.assertEqual(runtime.routes, [])
        self.assertEqual(runtime.graph_padding_routes, [])
        data.zero_()  # Reused native buffer cannot change queued histograms.
        result = runtime.end_episode()
        self.assertTrue(result["summary"]["trace_complete"], result["summary"]["route_errors"])
        self.assertEqual([r["tokens"] for r in result["routes"]], [3, 3])
        physical, no_padding = result["graph_padding_routes"]
        self.assertEqual(physical["physical_expert_token_counts"], [3, 2, 3, 2])
        self.assertEqual(physical["padding_only_new_expert_ids"], [3])
        self.assertEqual(no_padding["physical_expert_token_counts"], [2, 2, 2, 0])
        self.assertEqual(no_padding["padding_tokens"], 0)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy not installed locally")
    def test_native_cpu_boundary_keeps_rejected_rows_before_scheduler_slicing(self):
        import numpy as np
        runtime = self.activate(static_gamma=2, native_route_return=True,
                                native_route_backend="cpu", keep_token_routes=True,
                                expert_weight_bytes=16)
        runtime.trace = True
        scheduler = Scheduler([Request("a", range(7))])
        scheduler.vllm_config = NS(model_config=NS(enforce_eager=False,
                                                  enable_return_routed_experts=True))
        output = scheduler.schedule()
        runtime._native_layers = {0: NS(layer_name="model.layers.0.mlp", global_num_experts=4)}
        # The last two verification rows contain experts absent from row 0.
        # A request-output/accepted-only reader would incorrectly measure D=2.
        data = np.array([[[0, 1]], [[1, 2]], [[2, 3]]], dtype=np.int32)
        native = NS(req_ids=["a"], routed_experts=NS(routing_data=data))
        runtime.capture_native_cpu(output, native)
        data[:] = 0  # Native pinned transit buffer may be reused next step.
        result = runtime.end_episode()
        self.assertEqual(result["routes"][0]["distinct_experts"], 4)
        self.assertEqual(result["routes"][0]["tokens"], 3)
        self.assertEqual(result["routes"][0]["expert_token_counts"], [1, 2, 2, 1])
        self.assertEqual(result["routes"][0]["topk_ids"], [[0, 1], [1, 2], [2, 3]])
        self.assertFalse(result["summary"]["uninstrumented_timing_eligible"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed locally")
    def test_real_tensor_histogram_and_deferred_drain(self):
        import torch
        runtime = self.activate(static_gamma=2, keep_token_routes=True)
        runtime.trace = True
        Scheduler([Request("a", range(7))]).schedule()
        layer = NS(layer_name="layers.0.mlp.experts", global_num_experts=4,
                   local_num_experts=4,
                   named_parameters=lambda recurse: [("w13_weight", torch.zeros(4, 2, 2))])
        ids = torch.tensor([[0, 1], [1, 2], [0, 2], [3, 3]])
        runtime.capture_routes(layer, ids)
        self.assertEqual(runtime.routes, [])  # No per-step host extraction.
        result = runtime.end_episode()
        route = result["routes"][0]
        self.assertEqual(route["expert_token_counts"], [2, 2, 2, 0])
        self.assertEqual(route["distinct_experts"], 3)
        self.assertEqual(route["selected_expert_storage_bytes"], 48)
        self.assertEqual(route["topk_ids"], [[0, 1], [1, 2], [0, 2]])


if __name__ == "__main__":
    unittest.main()
