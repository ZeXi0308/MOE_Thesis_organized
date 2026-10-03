from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

try:
    from .protocol import load_config
    from .raw_recompute import (
        RawBlock,
        RawRequest,
        _sha_config_paths,
        recompute_raw_gate,
        request_provenance_hashes,
        sha256_file,
        verify_bundle,
    )
    from .schema import ProtocolError
except ImportError:
    from protocol import load_config
    from raw_recompute import (
        RawBlock,
        RawRequest,
        _sha_config_paths,
        recompute_raw_gate,
        request_provenance_hashes,
        sha256_file,
        verify_bundle,
    )
    from schema import ProtocolError


CONFIG_PATH = Path(__file__).parent / "configs" / "gate0_v1.json"
RAW_SOURCE = Path(__file__).parent / "raw_recompute.py"
SIMPLE_POLICY = "per_tenant_request_concurrency_quota"


def digest(material: str) -> str:
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def fixture_config() -> dict[str, object]:
    config = deepcopy(load_config(CONFIG_PATH))
    selected = config["baseline_selection"]["frozen_strongest_simple_by_model"]
    selected["olmoe"] = SIMPLE_POLICY
    selected["llmjp"] = SIMPLE_POLICY
    return config


def raw_fixture() -> tuple[dict[str, object], list[RawRequest], list[RawBlock]]:
    config = fixture_config()
    requests: list[RawRequest] = []
    blocks: list[RawBlock] = []
    cells = (
        ("70pct", "ADV_TEXT"),
        ("30pct", "NAT_BENIGN"),
        ("70pct", "NAT_PATHOLOGICAL"),
    )
    for model in ("olmoe", "llmjp"):
        for load_cell, traffic_class in cells:
            scenarios = (
                (
                    "MATCHED_BENIGN",
                    "ATTACK_BASELINE",
                    "LEGAL_ORACLE",
                    "STRONGEST_SIMPLE",
                )
                if traffic_class == "ADV_TEXT"
                else ("CONTROL_DEFAULT", "CONTROL_ISOLATION")
            )
            for block_index in range(2):
                block_id = digest(f"{model}|{load_cell}|{traffic_class}|{block_index}")
                start = block_index * 10_000
                arrival_trace = digest(f"arrival|{model}|{load_cell}|{block_index}")
                victim_manifest = digest(f"victim|{model}|{load_cell}|{block_index}")
                budget_manifest = digest(f"budget|{model}|{load_cell}|{block_index}")
                attack_world = digest(f"attack-world|{model}|{load_cell}|{block_index}")
                control_world = digest(f"control-world|{model}|{load_cell}|{block_index}")
                for scenario in scenarios:
                    if scenario in {
                        "MATCHED_BENIGN",
                        "ATTACK_BASELINE",
                        "CONTROL_DEFAULT",
                    }:
                        policy = "production_default_fcfs_chunked_prefill"
                    elif scenario == "LEGAL_ORACLE":
                        policy = "future_known_exact_legal_oracle"
                    else:
                        policy = SIMPLE_POLICY
                    if traffic_class == "ADV_TEXT":
                        world = (
                            digest(f"benign-world|{model}|{load_cell}|{block_index}")
                            if scenario == "MATCHED_BENIGN"
                            else attack_world
                        )
                        duration = 2_000
                    else:
                        world = control_world
                        duration = 1_020 if scenario == "CONTROL_ISOLATION" else 1_000
                    blocks.append(
                        RawBlock(
                            model=model,
                            load_cell=load_cell,
                            traffic_class=traffic_class,
                            scenario=scenario,
                            block_id=block_id,
                            policy_id=policy,
                            request_world_sha256=world,
                            arrival_trace_sha256=arrival_trace,
                            victim_manifest_sha256=victim_manifest,
                            cotenant_budget_sha256=budget_manifest,
                            window_start_ns=start,
                            window_end_ns=start + duration,
                            queue_service_work_start=10.0,
                            queue_service_work_end=10.0,
                            queue_service_work_arrived=100.0,
                            oracle_status=(
                                "OPTIMAL" if scenario == "LEGAL_ORACLE" else "NOT_APPLICABLE"
                            ),
                            oracle_gap=0.0 if scenario == "LEGAL_ORACLE" else None,
                        )
                    )
                    latency = {
                        "MATCHED_BENIGN": 100,
                        "ATTACK_BASELINE": 140,
                        "LEGAL_ORACLE": 110,
                        "STRONGEST_SIMPLE": 115,
                        "CONTROL_DEFAULT": 100,
                        "CONTROL_ISOLATION": 100,
                    }[scenario]
                    for request_index in range(4):
                        victim = request_index < 2
                        pair_id = digest(
                            f"pair|{model}|{load_cell}|{traffic_class}|{block_index}|{request_index}"
                        )
                        role = "victim"
                        if not victim:
                            role = (
                                "cotenant"
                                if scenario in {
                                    "MATCHED_BENIGN",
                                    "CONTROL_DEFAULT",
                                    "CONTROL_ISOLATION",
                                }
                                else "attacker"
                            )
                        prompt_kind = "victim" if victim else (
                            "benign" if scenario == "MATCHED_BENIGN" else "cotenant"
                        )
                        if not victim and traffic_class == "ADV_TEXT" and scenario != "MATCHED_BENIGN":
                            prompt_kind = "attack"
                        prompt_hash = digest(
                            f"prompt|{model}|{load_cell}|{block_index}|{request_index}|{prompt_kind}"
                        )
                        output_hash = digest(
                            f"output|{model}|{load_cell}|{block_index}|{request_index}|"
                            + ("benign" if scenario == "MATCHED_BENIGN" and not victim else "fixed")
                        )
                        arrival = start + 10 + request_index
                        requests.append(
                            RawRequest(
                                model=model,
                                load_cell=load_cell,
                                traffic_class=traffic_class,
                                scenario=scenario,
                                block_id=block_id,
                                pair_id=pair_id,
                                tenant_id=(
                                    "tenant-victim" if victim else "tenant-cotenant"
                                ),
                                role=role,
                                request_id=f"request-{block_index}-{request_index}",
                                document_id=f"document-{block_index}-{request_index}",
                                document_cluster_id=(
                                    f"cluster-{model}-{load_cell}-{traffic_class}-{block_index}-{request_index}"
                                ),
                                prompt_hash=prompt_hash,
                                input_tokens=4,
                                max_new_tokens=1,
                                arrival_ns=arrival,
                                first_token_ns=arrival + latency,
                                completion_ns=arrival + latency + 1,
                                output_token_count=1,
                                output_hash=output_hash,
                                terminal_reason="COMPLETED",
                            )
                        )
    blocks = [
        replace(
            block,
            **request_provenance_hashes(
                [
                    row
                    for row in requests
                    if row.model == block.model
                    and row.load_cell == block.load_cell
                    and row.traffic_class == block.traffic_class
                    and row.block_id == block.block_id
                    and row.scenario == block.scenario
                ]
            ),
        )
        for block in blocks
    ]
    return config, requests, blocks


def write_jsonl(path: Path, rows: list[RawRequest] | list[RawBlock]) -> None:
    path.write_text(
        "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def artifact_entry(path: Path, *, schema: str, config_key: str) -> dict[str, object]:
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "row_count": len(path.read_text(encoding="utf-8").splitlines()),
        "format": "jsonl",
        "schema": schema,
        "config_key": config_key,
    }


class RawRecomputeTest(unittest.TestCase):
    def test_formal_hash_closure_includes_list_nested_tokenizers(self) -> None:
        paths = _sha_config_paths(load_config(CONFIG_PATH))
        self.assertIn("models.0.tokenizer_sha256", paths)
        self.assertIn("models.1.tokenizer_sha256", paths)

    def test_known_good_raw_fixture_recomputes_thresholds(self) -> None:
        config, requests, blocks = raw_fixture()
        result = recompute_raw_gate(
            config, requests, blocks, allow_small_fixture=True
        )
        self.assertEqual(result["status"], "RAW_RECOMPUTE_DIAGNOSTIC_ONLY")
        self.assertEqual(
            result["diagnostic_threshold_branch"], "ALL_THRESHOLDS_PASS"
        )
        self.assertFalse(result["formal_result"])
        self.assertNotIn("QUALIFIED_FOR_8XA100", str(result))

    def test_block_provenance_is_recomputed_from_request_rows(self) -> None:
        config, requests, blocks = raw_fixture()
        index = next(
            index
            for index, row in enumerate(requests)
            if row.scenario == "ATTACK_BASELINE" and row.role == "attacker"
        )
        requests[index] = replace(requests[index], prompt_hash="f" * 64)
        with self.assertRaisesRegex(ProtocolError, "recomputed request provenance"):
            recompute_raw_gate(config, requests, blocks, allow_small_fixture=True)

    def test_direct_dataclass_api_revalidates_timestamps(self) -> None:
        config, requests, blocks = raw_fixture()
        requests[0] = replace(
            requests[0], first_token_ns=requests[0].arrival_ns - 1
        )
        with self.assertRaisesRegex(ProtocolError, "timestamps are not monotonic"):
            recompute_raw_gate(config, requests, blocks, allow_small_fixture=True)

    def test_matched_benign_arm_cannot_carry_attacker_role(self) -> None:
        config, requests, blocks = raw_fixture()
        index = next(
            index
            for index, row in enumerate(requests)
            if row.scenario == "MATCHED_BENIGN" and row.role == "cotenant"
        )
        requests[index] = replace(requests[index], role="attacker")
        with self.assertRaisesRegex(ProtocolError, "invalid tenant role"):
            recompute_raw_gate(config, requests, blocks, allow_small_fixture=True)

    def test_goodput_window_is_recomputed_from_request_timestamps(self) -> None:
        config, requests, blocks = raw_fixture()
        result = recompute_raw_gate(
            config, requests, blocks, allow_small_fixture=True
        )
        controls = [
            row
            for row in result["metrics"]
            if row["traffic_class"] == "NAT_BENIGN"
        ]
        self.assertTrue(controls)
        self.assertTrue(
            all(row["benign_goodput_loss_point"] == 0.0 for row in controls)
        )

    def test_victim_output_mismatch_is_invalid(self) -> None:
        config, requests, blocks = raw_fixture()
        index = next(
            index
            for index, row in enumerate(requests)
            if row.scenario == "LEGAL_ORACLE" and row.role == "victim"
        )
        requests[index] = replace(requests[index], output_hash="f" * 64)
        with self.assertRaisesRegex(ProtocolError, "victim completion/output"):
            recompute_raw_gate(config, requests, blocks, allow_small_fixture=True)

    def test_missing_paired_arm_is_invalid(self) -> None:
        config, requests, blocks = raw_fixture()
        target = next(
            block
            for block in blocks
            if block.model == "olmoe"
            and block.traffic_class == "ADV_TEXT"
            and block.scenario == "LEGAL_ORACLE"
        )
        blocks = [block for block in blocks if block != target]
        requests = [
            row
            for row in requests
            if not (row.block_id == target.block_id and row.scenario == target.scenario)
        ]
        with self.assertRaisesRegex(ProtocolError, "scenarios"):
            recompute_raw_gate(config, requests, blocks, allow_small_fixture=True)

    def test_censored_request_cannot_leave_denominator(self) -> None:
        config, requests, blocks = raw_fixture()
        index = next(index for index, row in enumerate(requests) if row.role == "victim")
        requests[index] = replace(
            requests[index],
            first_token_ns=None,
            completion_ns=None,
            output_token_count=0,
            output_hash=None,
            terminal_reason="TIMED_OUT",
        )
        with self.assertRaisesRegex(ProtocolError, "CENSORED_REQUEST"):
            recompute_raw_gate(config, requests, blocks, allow_small_fixture=True)

    def test_oracle_timeout_is_not_a_heuristic_result(self) -> None:
        config, requests, blocks = raw_fixture()
        index = next(
            index for index, block in enumerate(blocks) if block.scenario == "LEGAL_ORACLE"
        )
        blocks[index] = replace(blocks[index], oracle_status="TIMEOUT", oracle_gap=None)
        with self.assertRaisesRegex(ProtocolError, "UNSOLVED_EXACT_STATE_LIMIT"):
            recompute_raw_gate(config, requests, blocks, allow_small_fixture=True)

    def test_queue_stability_is_derived_from_work_not_boolean(self) -> None:
        config, requests, blocks = raw_fixture()
        blocks = [
            replace(block, queue_service_work_end=100.0)
            if block.model == "olmoe"
            and block.traffic_class == "ADV_TEXT"
            and block.scenario == "ATTACK_BASELINE"
            else block
            for block in blocks
        ]
        result = recompute_raw_gate(
            config, requests, blocks, allow_small_fixture=True
        )
        self.assertEqual(result["status"], "INVALID_REQUEST_DAG")
        self.assertIn("QUEUE_UNSTABLE", str(result))

    def test_bundle_hash_and_completeness_verifier(self) -> None:
        _, requests, blocks = raw_fixture()
        config = load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request_path = root / "requests.jsonl"
            block_path = root / "blocks.jsonl"
            manifest_path = root / "manifest.json"
            write_jsonl(request_path, requests)
            write_jsonl(block_path, blocks)
            manifest = {
                "schema": "routeshield-raw-bundle-v1",
                "mode": "DEVELOPMENT",
                "config_sha256": sha256_file(CONFIG_PATH),
                "evaluator_source_sha256": sha256_file(RAW_SOURCE),
                "artifacts": {
                    "requests": artifact_entry(
                        request_path,
                        schema="routeshield-raw-request-v1",
                        config_key="required_evidence.raw_request_ledger_sha256",
                    ),
                    "blocks": artifact_entry(
                        block_path,
                        schema="routeshield-raw-block-v1",
                        config_key="required_evidence.raw_block_ledger_sha256",
                    ),
                },
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            verified = verify_bundle(
                manifest_path, config=config, config_path=CONFIG_PATH
            )
            self.assertEqual(verified.request_path, request_path.resolve())

            extra = root / "unlisted.json"
            extra.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, "unlisted file"):
                verify_bundle(manifest_path, config=config, config_path=CONFIG_PATH)

    def test_bundle_rejects_hash_mismatch_and_duplicate_json_key(self) -> None:
        _, requests, blocks = raw_fixture()
        config = load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request_path = root / "requests.jsonl"
            block_path = root / "blocks.jsonl"
            manifest_path = root / "manifest.json"
            write_jsonl(request_path, requests)
            write_jsonl(block_path, blocks)
            manifest = {
                "schema": "routeshield-raw-bundle-v1",
                "mode": "DEVELOPMENT",
                "config_sha256": sha256_file(CONFIG_PATH),
                "evaluator_source_sha256": sha256_file(RAW_SOURCE),
                "artifacts": {
                    "requests": artifact_entry(
                        request_path,
                        schema="routeshield-raw-request-v1",
                        config_key="required_evidence.raw_request_ledger_sha256",
                    ),
                    "blocks": artifact_entry(
                        block_path,
                        schema="routeshield-raw-block-v1",
                        config_key="required_evidence.raw_block_ledger_sha256",
                    ),
                },
            }
            manifest["artifacts"]["requests"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ProtocolError, "hash mismatch"):
                verify_bundle(manifest_path, config=config, config_path=CONFIG_PATH)

            manifest_path.write_text(
                '{"schema":"routeshield-raw-bundle-v1","schema":"duplicate"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ProtocolError, "duplicate JSON key"):
                verify_bundle(manifest_path, config=config, config_path=CONFIG_PATH)


if __name__ == "__main__":
    unittest.main()
