from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import capture_routes_gpu as routes
import measure_capability_gpu as capability
import measure_service_lut_gpu as service_lut
import prepare_data as data


CONFIG_PATH = HERE.parents[1] / "configs" / "ric_v1.json"


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _explicit_model_spec(root: Path) -> dict[str, str]:
    files = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        files.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": data.sha256_file(path),
            }
        )
    return {
        "repo_id": "frozen/repo",
        "revision": "deadbeef",
        "expected_local_model_tree_manifest_sha256": data.sha256_bytes(
            data.canonical_json_bytes(files)
        ),
    }


def _manifest(config: dict[str, object], role: str) -> dict[str, object]:
    role_cfg = config["data"][role]
    formal_identity = config["data"]["formal_dataset_identity"]
    seed = int(config["data"]["selection_seed"])
    candidates = []
    for offset in range(int(role_cfg["document_count"])):
        text = f"request-{role}-{offset}"
        text_hash = data.sha256_bytes(text.encode())
        candidates.append(
            {
                "text": text,
                "text_sha256": text_hash,
                "source_row": int(role_cfg["candidate_row_start_inclusive"])
                + offset,
                "rank_sha256": data.frozen_concat_sha256(seed, text_hash),
                "token_lengths": {"olmoe": 129, "llmjp": 129},
            }
        )
    candidates.sort(key=lambda row: (row["rank_sha256"], row["source_row"]))
    requests = [
        {
            **row,
            "request_id": f"ric:{role}:{index:04d}:{row['text_sha256'][:12]}",
        }
        for index, row in enumerate(candidates)
    ]
    mode = "dev" if role == "calibration" else "formal"
    payload = {
        "schema_version": "ric-data-manifest-v1",
        "status": "NOT_TESTED" if mode == "dev" else "INPUT_ONLY",
        "scientific_result": False,
        "mode": mode,
        "role": role,
        "dataset_loader": "wikitext",
        "dataset_config": config["data"]["config"],
        "dataset_split": config["data"]["split"],
        "data_preparation_producer": formal_identity["producer"],
        "data_preparation_python_environment": formal_identity["python_environment"],
        "datasets_library_version": (
            formal_identity["datasets_library_version"]
            if mode == "formal"
            else "fixture"
        ),
        "dataset_repo_id": formal_identity["dataset_repo_id"],
        "dataset_revision": formal_identity["dataset_revision"],
        "dataset_source_urls_sha256": "9" * 64,
        "dataset_slice_fingerprint": (
            formal_identity[role]["dataset_slice_fingerprint"]
            if mode == "formal"
            else "fixture-fingerprint"
        ),
        "dataset_slice_row_count": int(
            role_cfg["candidate_row_end_exclusive"]
        )
        - int(role_cfg["candidate_row_start_inclusive"]),
        "dataset_slice_canonical_content_sha256": (
            formal_identity[role]["dataset_slice_canonical_content_sha256"]
            if mode == "formal"
            else "a" * 64
        ),
        "candidate_window": [
            role_cfg["candidate_row_start_inclusive"],
            role_cfg["candidate_row_end_exclusive"],
        ],
        "selection_seed": seed,
        "selection_method": config["data"]["selection_method"],
        "sequence_tokens": config["data"]["sequence_length"],
        "batch_size": config["data"]["batch_size"],
        "padding_allowed": config["data"]["padding_allowed"],
        "model_revisions": {
            key: f"{spec['repo_id']}@{spec['revision']}"
            for key, spec in config["models"].items()
        },
        "tokenizer_revisions": {
            key: f"{spec['repo_id']}@{spec['revision']}"
            for key, spec in config["models"].items()
        },
        "historical_exclusion_registry_sha256": "b" * 64,
        "protocol_sha256": data.sha256_file(data.DEFAULT_PROTOCOL),
        "config_sha256": data.sha256_file(CONFIG_PATH),
        "prepare_data_source_sha256": data._producer_source_sha256(),
        "selected_text_sha256": [row["text_sha256"] for row in requests],
        "requests": requests,
    }
    if role == "sealed":
        payload.update(
            {
                "sealed_reservation_sha256": "c" * 64,
                "sealed_nonce_sha256": "d" * 64,
                "calibration_manifest_sha256": "e" * 64,
                "calibration_manifest_file_sha256": "f" * 64,
                "calibration_selected_text_sha256": "1" * 64,
            }
        )
    return data.add_self_hash(payload)


class DataProducerTest(unittest.TestCase):
    def test_selection_hash_is_exact_frozen_concatenation_without_delimiter(self) -> None:
        selected = data.select_requests(
            ["text-0"],
            source_row_start=0,
            required_count=1,
            selection_seed=7,
            min_tokens=1,
            token_lengths=lambda _text: {"olmoe": 1, "llmjp": 1},
            historical_hashes=set(),
            role="calibration",
        )
        self.assertEqual(
            selected[0]["rank_sha256"],
            "5b991e2c17218193a4913fbfdb42adafb62f17ae540cc544fe0b6ff916bd050b",
        )
        self.assertNotEqual(
            selected[0]["rank_sha256"],
            "3bd54b915979e4e17a71ac0031ac15c974f0f64d25e7aae1b9f8b4419bd7ed22",
        )

    def test_self_hash_fails_closed(self) -> None:
        value = data.add_self_hash({"schema_version": "x", "answer": 1})
        self.assertEqual(data.validate_self_hash(value), value["manifest_sha256"])
        value["answer"] = 2
        with self.assertRaisesRegex(data.DataPreparationError, "mismatch"):
            data.validate_self_hash(value)

    def test_self_hash_is_stable_across_json_integer_key_roundtrip(self) -> None:
        value = data.add_self_hash(
            {"schema_version": "x", "nonlexicographic_integer_keys": {7: 1, 14: 2}}
        )
        self.assertEqual(
            value["nonlexicographic_integer_keys"], {"7": 1, "14": 2}
        )
        reloaded = json.loads(json.dumps(value))
        self.assertEqual(data.validate_self_hash(reloaded), value["manifest_sha256"])

    def test_self_hash_rejects_nonfinite_json(self) -> None:
        with self.assertRaisesRegex(data.DataPreparationError, "strict JSON"):
            data.add_self_hash({"schema_version": "x", "bad": float("nan")})

    def test_self_hash_rejects_keys_colliding_after_json_normalization(self) -> None:
        with self.assertRaisesRegex(data.DataPreparationError, "duplicate JSON object key"):
            data.add_self_hash(
                {"schema_version": "x", "bad": {1: "int", "1": "str"}}
            )

    def test_dev_cannot_request_sealed(self) -> None:
        with self.assertRaisesRegex(data.DataPreparationError, "forbidden"):
            data._validate_mode_role("dev", "sealed")

    def test_selection_does_not_skip_historical_collision(self) -> None:
        rows = [f"text-{index}" for index in range(8)]
        selected = data.select_requests(
            rows,
            source_row_start=100,
            required_count=3,
            selection_seed=7,
            min_tokens=2,
            token_lengths=lambda _text: {"olmoe": 3, "llmjp": 3},
            historical_hashes=set(),
            role="calibration",
        )
        collision = str(selected[0]["text_sha256"])
        with self.assertRaisesRegex(data.DataPreparationError, "Phase-2 amendment"):
            data.select_requests(
                rows,
                source_row_start=100,
                required_count=3,
                selection_seed=7,
                min_tokens=2,
                token_lengths=lambda _text: {"olmoe": 3, "llmjp": 3},
                historical_hashes={collision},
                role="calibration",
            )

    def test_historical_registry_uses_explicit_text_hash_fields_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text_hash = "a" * 64
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "text_sha256": text_hash,
                        "source_sha256": "b" * 64,
                    }
                ),
                encoding="utf-8",
            )
            registry = data.build_historical_registry(root)
            self.assertEqual(registry["text_hashes"], [text_hash])
            self.assertTrue(registry["complete"])
            data.validate_self_hash(registry, "registry_sha256")

class RouteProducerTest(unittest.TestCase):
    def test_route_rejects_local_tokenizer_length_drift(self) -> None:
        request = {"token_lengths": {"olmoe": 137, "llmjp": 141}}
        routes.validate_full_tokenizer_length(
            request,
            model_key="olmoe",
            observed_length=137,
            minimum_length=129,
        )
        with self.assertRaisesRegex(routes.RouteCaptureError, "differs from manifest"):
            routes.validate_full_tokenizer_length(
                request,
                model_key="olmoe",
                observed_length=136,
                minimum_length=129,
            )

    def test_manifest_rejects_dev_sealed(self) -> None:
        config = _config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sealed.json"
            path.write_text(json.dumps(_manifest(config, "sealed")), encoding="utf-8")
            with self.assertRaisesRegex(routes.RouteCaptureError, "forbidden"):
                routes._load_data_manifest(
                    path,
                    mode="dev",
                    model_key="olmoe",
                    config=config,
                    protocol_sha256=data.sha256_file(data.DEFAULT_PROTOCOL),
                    config_sha256=data.sha256_file(CONFIG_PATH),
                )

    def test_manifest_accepts_complete_calibration(self) -> None:
        config = _config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            payload = _manifest(config, "calibration")
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = routes._load_data_manifest(
                path,
                mode="dev",
                model_key="llmjp",
                config=config,
                protocol_sha256=data.sha256_file(data.DEFAULT_PROTOCOL),
                config_sha256=data.sha256_file(CONFIG_PATH),
            )
            self.assertEqual(loaded["manifest_sha256"], payload["manifest_sha256"])

    def test_contiguous_placement_and_route_blind_lpt(self) -> None:
        self.assertEqual(
            [routes.expert_sender(index, 8, 4) for index in range(8)],
            [0, 0, 1, 1, 2, 2, 3, 3],
        )
        requests = [{"request_id": value} for value in ("a", "b", "c", "d")]
        first = routes.origin_lpt(requests, 2)
        second = routes.origin_lpt(list(reversed(requests)), 2)
        self.assertEqual(first, second)
        # Independent golden: request SHA order is d,c,b,a.  Immediately
        # before b both loads are 128, and exact sha256(b"b1") is smaller
        # than sha256(b"b0"), so the request-specific tie must choose rank 1.
        self.assertLess(
            hashlib.sha256(b"b1").hexdigest(),
            hashlib.sha256(b"b0").hexdigest(),
        )
        self.assertEqual(first, {"d": 0, "c": 1, "b": 1, "a": 0})
        self.assertEqual(sorted(first.values()), [0, 0, 1, 1])

    def test_config_loader_rejects_nested_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text(
                '{"schema_version":"ric-config-v1",'
                '"status":"PHASE2_FROZEN_NO_SCIENTIFIC_RESULT",'
                '"nested":{"same":1,"same":2}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(routes.RouteCaptureError, "duplicate JSON key"):
                routes._load_config(path)

    def test_layer_selection_is_outcome_blind_and_stable(self) -> None:
        chosen = routes.selected_layers(
            [1, 3, 5, 7, 9, 11],
            selection_seed=2026072223,
            model_revision="model@revision",
            count=4,
        )
        self.assertEqual(
            chosen,
            routes.selected_layers(
                [11, 9, 7, 5, 3, 1],
                selection_seed=2026072223,
                model_revision="model@revision",
                count=4,
            ),
        )
        self.assertEqual(chosen, [1, 3, 5, 9])
        self.assertNotEqual(chosen, [1, 3, 7, 11])
        self.assertIn(routes.assigned_layer("request", chosen), chosen)

    def test_model_config_census_rejects_missing_moe_layer(self) -> None:
        model_config = SimpleNamespace(
            num_hidden_layers=2,
            num_experts=2,
            num_experts_per_tok=1,
            # Hugging Face generation top-k is not a MoE routing field.
            top_k=50,
        )
        module = lambda: SimpleNamespace(experts=[object(), object()])
        valid = [(0, "layers.0.moe", module()), (1, "layers.1.moe", module())]
        evidence = routes.validate_model_config_layer_census(
            model_config,
            valid,
            expected_num_experts=2,
            expected_top_k=1,
        )
        self.assertEqual(evidence["expected_layers"], [0, 1])
        with self.assertRaisesRegex(routes.RouteCaptureError, r"missing=\[1\]"):
            routes.validate_model_config_layer_census(
                model_config,
                valid[:1],
                expected_num_experts=2,
                expected_top_k=1,
            )

    def test_perturbed_routing_fails_independent_native_output_parity(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        class ScaleExpert(torch.nn.Module):
            def __init__(self, scale: float) -> None:
                super().__init__()
                self.scale = scale

            def forward(self, value):
                return value * self.scale

        moe = SimpleNamespace(experts=[ScaleExpert(1.0), ScaleExpert(3.0)])
        hidden = torch.ones((1, 2, 1), dtype=torch.float32)
        original_logits = torch.tensor([[[10.0, 0.0], [10.0, 0.0]]])
        perturbed_logits = torch.tensor([[[0.0, 10.0], [0.0, 10.0]]])
        original_experts, original_weights = routes._routes_from_logits(
            original_logits,
            top_k=1,
            normalize_topk=True,
            selection_rule=routes.NATIVE_TOPK_SELECTION_RULE,
            output_dtype=hidden.dtype,
        )
        native, original_experts, _ = routes.reconstruct_native_moe_output(
            moe=moe,
            hidden_states=hidden,
            selected_experts=original_experts,
            effective_weights=original_weights,
        )
        perturbed_experts, perturbed_weights = routes._routes_from_logits(
            perturbed_logits,
            top_k=1,
            normalize_topk=True,
            selection_rule=routes.NATIVE_TOPK_SELECTION_RULE,
            output_dtype=hidden.dtype,
        )
        perturbed, perturbed_experts, _ = routes.reconstruct_native_moe_output(
            moe=moe,
            hidden_states=hidden,
            selected_experts=perturbed_experts,
            effective_weights=perturbed_weights,
        )
        self.assertFalse(torch.equal(original_experts, perturbed_experts))
        with self.assertRaisesRegex(routes.RouteCaptureError, "output parity failed"):
            routes.validate_native_moe_output_parity(
                native,
                perturbed,
                tolerance_rule=_config()["route_capture"][
                    "native_moe_output_tolerance"
                ],
            )

    def test_route_epoch_is_one_based(self) -> None:
        self.assertEqual(routes.ROUTE_EPOCH, 1)
        source = (HERE / "capture_routes_gpu.py").read_text(encoding="utf-8")
        self.assertIn('"epoch": ROUTE_EPOCH', source)

    def test_explicit_model_path_avoids_repo_revision_cache_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            source, kwargs, manifest = routes.model_load_reference(
                _explicit_model_spec(root), root
            )
            self.assertEqual(source, str(root.resolve()))
            self.assertEqual(kwargs, {"local_files_only": True})
            self.assertNotIn("revision", kwargs)
            self.assertNotIn("cache_dir", kwargs)
            self.assertEqual(manifest["kind"], "explicit_local_directory")
            self.assertTrue(manifest["tree_manifest_sha256"])
            self.assertEqual(manifest["files"][0]["sha256"], data.sha256_file(root / "config.json"))

    def test_same_size_weight_bitflip_changes_local_tree_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "model-00001-of-00001.safetensors"
            shard.write_bytes(b"\x00" * 4096)
            first_spec = _explicit_model_spec(root)
            first = routes.model_load_reference(
                first_spec, root
            )[2]
            shard.write_bytes(b"\x00" * 4095 + b"\x01")
            with self.assertRaisesRegex(
                routes.RouteCaptureError, "tree hash differs from frozen config"
            ):
                routes.model_load_reference(first_spec, root)
            second = routes.model_load_reference(
                _explicit_model_spec(root), root
            )[2]
            self.assertEqual(first["files"][0]["size_bytes"], second["files"][0]["size_bytes"])
            self.assertNotEqual(first["files"][0]["sha256"], second["files"][0]["sha256"])
            self.assertNotEqual(first["tree_manifest_sha256"], second["tree_manifest_sha256"])

    def test_explicit_model_path_rejects_missing_expected_tree_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(
                routes.RouteCaptureError, "lacks a canonical expected tree hash"
            ):
                routes.model_load_reference(
                    {"repo_id": "frozen/repo", "revision": "deadbeef"}, root
                )

    def test_discovery_rejects_duplicate_moe_layer(self) -> None:
        class Module:
            gate = object()
            experts = [object(), object()]

        class Model:
            def named_modules(self):
                return [
                    ("model.layers.2.mlp", Module()),
                    ("model.layers.2.block_sparse_moe", Module()),
                ]

        with self.assertRaisesRegex(routes.RouteCaptureError, "multiple MoE"):
            routes.discover_moe_modules(Model())

    def test_formal_route_signoff_binds_manifest_and_model(self) -> None:
        with patch.object(
            routes, "verify_phase4_signoff", return_value={"status": "SIGNED-OFF"}
        ) as verify:
            result = routes._require_formal_signoff(
                None,
                protocol_sha256="p",
                config_sha256="c",
                source_sha256="s",
                data_manifest_sha256="d",
                data_producer_signoff_sha256="q",
                model_key="olmoe",
                model_tree_manifest_sha256="tree",
            )
        self.assertEqual(result["status"], "SIGNED-OFF")
        expected = verify.call_args.kwargs["expected_fields"]
        self.assertEqual(expected["stage"], "capture_routes")
        self.assertEqual(expected["model_key"], "olmoe")
        self.assertEqual(expected["data_manifest_sha256"], "d")
        self.assertEqual(expected["data_producer_signoff_sha256"], "q")
        self.assertEqual(expected["prepare_data_source_sha256"], data._producer_source_sha256())

    def test_route_source_uses_native_parity_not_patch(self) -> None:
        source = (HERE / "capture_routes_gpu.py").read_text(encoding="utf-8")
        self.assertIn("output_router_logits=True", source)
        self.assertIn("register_forward_hook", source)
        self.assertIn("validate_model_config_layer_census", source)
        self.assertIn("reconstruct_native_moe_output", source)
        self.assertIn("topk_expert_exact_native_capture", source)
        self.assertIn(
            "native_aten_topk_capture_plus_independent_moe_output_parity", source
        )
        self.assertNotIn(
            "native_output_router_logits_plus_gate_hook_parity", source
        )
        self.assertNotIn("patch_mixtral_moe", source)

    def test_tied_logits_replay_native_topk_slot_order(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        class IdentityExpert(torch.nn.Module):
            def forward(self, value):
                return value

        logits = torch.tensor([[[3.0, 2.0, 2.0, 2.0, 1.0]]])
        hidden = torch.ones((1, 1, 1), dtype=torch.float32)
        expected_weights, expected_experts = torch.topk(
            torch.softmax(logits.reshape(1, 5), dim=-1, dtype=torch.float32),
            3,
            dim=-1,
        )
        route_experts, route_weights = routes._routes_from_logits(
            logits,
            top_k=3,
            normalize_topk=False,
            selection_rule=routes.NATIVE_TOPK_SELECTION_RULE,
            output_dtype=hidden.dtype,
        )
        _output, replay_experts, replay_weights = routes.reconstruct_native_moe_output(
            moe=SimpleNamespace(experts=[IdentityExpert() for _ in range(5)]),
            hidden_states=hidden,
            selected_experts=route_experts,
            effective_weights=route_weights,
        )
        self.assertTrue(torch.equal(route_experts, expected_experts))
        self.assertTrue(torch.equal(replay_experts, expected_experts))
        self.assertTrue(torch.equal(route_weights, expected_weights))
        self.assertTrue(torch.equal(replay_weights, expected_weights))
        with self.assertRaisesRegex(routes.RouteCaptureError, "selection rule drift"):
            routes._routes_from_logits(
                logits,
                top_k=3,
                normalize_topk=False,
                selection_rule="stable-sort-substitution",
                output_dtype=hidden.dtype,
            )

    def test_raw_router_identity_rejects_shape_erased_by_flatten(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        flat = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
        differently_shaped = flat.reshape(1, 3, 4)
        with self.assertRaisesRegex(routes.RouteCaptureError, "raw router shape"):
            routes.validate_raw_router_tensor_identity(
                flat,
                differently_shaped,
                expected_shape=(3, 4),
            )

    def test_route_tuple_hash_binds_weight_dtype_and_finiteness(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        experts = torch.tensor([[1, 0]], dtype=torch.int64)
        bf16 = torch.tensor([[0.5, 0.25]], dtype=torch.bfloat16)
        fp32 = bf16.float()
        self.assertNotEqual(
            routes._route_tuple_sha256(experts, bf16),
            routes._route_tuple_sha256(experts, fp32),
        )
        with self.assertRaisesRegex(routes.RouteCaptureError, "non-finite"):
            routes._route_tuple_sha256(
                experts,
                torch.tensor([[float("nan"), 0.25]], dtype=torch.bfloat16),
            )

    def test_dispatch_observer_captures_actual_native_topk(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        active = {"value": 3}
        capture = routes.make_native_topk_capture_mode(
            active_layer=active,
            expected_num_experts=5,
            expected_top_k=3,
            expected_tokens=2,
        )
        probabilities = torch.tensor(
            [[0.4, 0.2, 0.2, 0.1, 0.1], [0.1, 0.1, 0.2, 0.2, 0.4]],
            dtype=torch.float32,
        )
        with capture:
            values, indices = torch.topk(probabilities, 3, dim=-1)
        captured_values, captured_indices = capture.calls[3]
        self.assertTrue(torch.equal(values, captured_values))
        self.assertTrue(torch.equal(indices, captured_indices))


class CapabilityProducerTest(unittest.TestCase):
    def test_type1_lcb_uses_frozen_500th_order_statistic(self) -> None:
        class BoundaryRandom:
            def __init__(self) -> None:
                self.calls = 0

            def randrange(self, _count: int) -> int:
                replicate = self.calls // 2
                self.calls += 1
                return 0 if replicate < 500 else 1

        with patch.object(
            capability.random, "Random", return_value=BoundaryRandom()
        ):
            lcb = capability.paired_effect_lcbs(
                {"boundary": [0.0, 1.0]},
                replicates=10000,
                order_statistic_one_based=500,
                seed=2026072226,
            )
        self.assertEqual(lcb["boundary"], 0.0)

    def test_profiler_stream_audit_rejects_default_stream_mix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "trace.json"
            trace = {
                "traceEvents": [
                    {"name": "work", "cat": "kernel", "args": {"stream": 13}},
                    {
                        "name": "copy",
                        "cat": "gpu_memcpy",
                        "args": {"stream": 13},
                    },
                ]
            }
            trace_path.write_text(json.dumps(trace), encoding="utf-8")
            self.assertEqual(capability.profiler_gpu_stream_ordinal(trace_path), 13)
            trace["traceEvents"].append(
                {
                    "name": "default",
                    "cat": "kernel",
                    "args": {"stream": 7},
                }
            )
            trace_path.write_text(json.dumps(trace), encoding="utf-8")
            with self.assertRaisesRegex(
                capability.CapabilityError, "mixed or missing GPU stream"
            ):
                capability.profiler_gpu_stream_ordinal(trace_path)

    def test_sender_local_selection_uses_one_sender_and_distinct_joins(self) -> None:
        routes_by_token = [
            [token % 8, 8 + token % 8]
            for token in range(96)
        ]
        plan = capability.select_sender_local_blocks(
            routes_by_token,
            request_id="request-0",
            layer_id=3,
            model_revision="model@revision",
            selection_seed=7,
            num_experts=64,
            ep_size=8,
            block_rows=4,
        )
        rows = [*plan["x_closing"], *plan["y_nonclosing"]]
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({row["token_index"] for row in rows}), 8)
        self.assertEqual({row["sender_rank"] for row in rows}, {plan["sender_rank"]})

    def test_sender_local_selection_blocks_insufficient_support(self) -> None:
        with self.assertRaisesRegex(capability.CapabilityError, "BLOCKED_G1"):
            capability.select_sender_local_blocks(
                [[0, 8], [1, 9]],
                request_id="request-0",
                layer_id=3,
                model_revision="model@revision",
                selection_seed=7,
                num_experts=64,
                ep_size=8,
                block_rows=2,
            )

    def test_tensor_hash_preserves_original_dtype(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        values = torch.tensor([1.0, 2.0])
        self.assertNotEqual(
            capability.tensor_sha256(values.float()),
            capability.tensor_sha256(values.half()),
        )
        self.assertNotEqual(
            routes._tensor_sha256(values.float()),
            routes._tensor_sha256(values.half()),
        )

    def test_canonical_reduce_is_slot_ordered_when_torch_available(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        siblings = torch.tensor([[[1.0], [2.0], [4.0]]])
        self.assertTrue(torch.equal(capability.canonical_reduce(siblings), torch.tensor([[7.0]])))

    def test_formal_capability_signoff_binds_model(self) -> None:
        with patch.object(
            capability,
            "verify_phase4_signoff",
            return_value={"status": "SIGNED-OFF"},
        ) as verify:
            capability._require_formal_signoff(
                None,
                protocol_sha256="p",
                config_sha256="c",
                source_sha256="s",
                data_manifest_sha256="d",
                data_producer_signoff_sha256="q",
                model_key="llmjp",
                model_tree_manifest_sha256="tree",
            )
        expected = verify.call_args.kwargs["expected_fields"]
        self.assertEqual(expected["stage"], "measure_capability")
        self.assertEqual(expected["model_key"], "llmjp")
        self.assertEqual(expected["data_producer_signoff_sha256"], "q")
        self.assertEqual(
            expected["capture_routes_source_sha256"],
            routes._producer_source_sha256(),
        )

    def test_capability_source_has_required_fixtures(self) -> None:
        source = (HERE / "measure_capability_gpu.py").read_text(encoding="utf-8")
        for required in (
            "native_unpatched_model_expert_execution",
            "baseline_nonclosing_first",
            "candidate_closing_first",
            "streaming",
            "full_layer_barrier",
            "canonical_reduce",
            "capability_raw.csv",
            "capability_action_trace.jsonl",
            "capability_cuda_trace_{profiler_release}.json",
            "torch.profiler",
            "queue_snapshot",
            "enqueue_ts_us",
            "service_end_ts_us",
        ):
            self.assertIn(required, source)
        self.assertIn("--model-path", source)
        self.assertIn("streaming_frontier_paired_lcb_us", source)
        self.assertIn("streaming_downstream_paired_lcb_us", source)
        self.assertIn("release_interaction_paired_lcb_us", source)
        self.assertIn("downstream_interaction_paired_lcb_us", source)
        self.assertIn("event_precedence_all_trials_pass", source)
        self.assertIn("lcb_order_statistic_one_based", source)
        self.assertIn("profiler_gpu_stream_ordinal", source)


class ServiceLutProducerTest(unittest.TestCase):
    def test_cli_parser_accepts_one_phase4_signoff_argument(self) -> None:
        argv = [
            "measure_service_lut_gpu.py",
            "--model-key",
            "olmoe",
            "--data-manifest",
            "data.json",
            "--output-dir",
            "out",
            "--signoff",
            "phase4.json",
        ]
        with patch.object(sys, "argv", argv):
            parsed = service_lut.parse_args()
        self.assertEqual(parsed.signoff, Path("phase4.json"))

    def test_outcome_blind_expert_selection_is_stable(self) -> None:
        first = service_lut.outcome_blind_experts(
            num_experts=32,
            count=4,
            selection_seed=2026072223,
            model_revision="model@revision",
            layer_id=3,
        )
        second = service_lut.outcome_blind_experts(
            num_experts=32,
            count=4,
            selection_seed=2026072223,
            model_revision="model@revision",
            layer_id=3,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(len(set(first)), 4)

    def test_contract_host_path_uses_canonical_wire_apply(self) -> None:
        operations = service_lut._contract_host_operations(_config(), 4)
        self.assertEqual(
            set(operations),
            {
                "state_build_contract_record",
                "host_hash_identity",
                "host_encode_contract",
                "host_decode_contract",
                "collision_checked_identity_lookup",
                "host_apply_wire_contract",
                "epoch_sequence_apply",
                "sender_policy_cache_lookup",
                "host_empty_harness",
            },
        )
        for name, operation in operations.items():
            if name != "host_empty_harness":
                self.assertIsNotNone(operation())
        applied = operations["host_apply_wire_contract"]()
        self.assertTrue(applied.applied)
        self.assertFalse(applied.fallback)
        self.assertEqual(operations["host_empty_harness"](), 0)
        self.assertEqual(len(applied.entries), 4)

    def test_payload_descriptor_comes_from_real_tensor_shape_and_dtype(self) -> None:
        class FakeTensor:
            shape = (128, 2, 2048)
            dtype = "torch.bfloat16"

            @staticmethod
            def element_size() -> int:
                return 2

        descriptor = service_lut.payload_descriptor(FakeTensor())
        self.assertEqual(descriptor["payload_dtype"], "bfloat16")
        self.assertEqual(descriptor["payload_elements_per_row"], 2048)
        self.assertEqual(descriptor["payload_element_size_bytes"], 2)
        self.assertEqual(descriptor["payload_bytes_per_contribution_row"], 4096)
        self.assertEqual(len(descriptor["payload_layout_sha256"]), 64)

    def test_analytic_network_uses_actual_bits_and_link_rate(self) -> None:
        self.assertAlmostEqual(
            service_lut.analytic_network_transfer_us(32, 200.0),
            0.00128,
        )
        with self.assertRaises(service_lut.ServiceLutError):
            service_lut.analytic_network_transfer_us(0, 200.0)

    def test_formal_lut_signoff_binds_model_and_manifest(self) -> None:
        with patch.object(
            service_lut,
            "verify_phase4_signoff",
            return_value={"status": "SIGNED-OFF"},
        ) as verify:
            service_lut._require_formal_signoff(
                None,
                protocol_sha256="p",
                config_sha256="c",
                source_sha256="s",
                data_manifest_sha256="d",
                data_producer_signoff_sha256="q",
                model_key="olmoe",
                model_tree_manifest_sha256="tree",
            )
        expected = verify.call_args.kwargs["expected_fields"]
        self.assertEqual(expected["stage"], "measure_service_lut")
        self.assertEqual(expected["data_manifest_sha256"], "d")
        self.assertEqual(expected["data_producer_signoff_sha256"], "q")
        self.assertEqual(
            expected["measure_capability_source_sha256"],
            capability._producer_source_sha256(),
        )

    def test_service_lut_source_has_all_required_components(self) -> None:
        source = (HERE / "measure_service_lut_gpu.py").read_text(encoding="utf-8")
        for required in (
            "expert_execution",
            "sender_pack",
            "receiver_unpack",
            "host_to_device_staging_not_rdma",
            "canonical_reduction",
            "host_hash_identity",
            "host_encode_contract",
            "host_decode_contract",
            "host_apply_wire_contract",
            "apply_wire_contract",
            "collision_checked_identity_lookup",
            "state_build_contract_record",
            "epoch_sequence_apply",
            "sender_policy_cache_lookup",
            "payload_bytes_per_contribution_row",
            "payload_layout_sha256",
            "contract_transfer_analytic_primary_link",
            'source="analytic_network"',
            'source="synthetic_delay"',
            "negative_after_subtraction_clamped",
            "service_lut_raw.csv",
            "service_lut.csv",
            "--model-path",
        ):
            self.assertIn(required, source)
        self.assertIn('"canonical_reduction_charged_once_per_join": True', source)
        self.assertIn("expert_execution_route_specific_row1", source)
        self.assertIn("BLOCKED_ROUTE_SPECIFIC_SERVICE_COVERAGE", source)


if __name__ == "__main__":
    unittest.main()
