from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

try:
    from . import joinstream_gpu_backfeed as backfeed
except ImportError:
    import joinstream_gpu_backfeed as backfeed  # type: ignore


CPU_RESULT = Path(__file__).resolve().parents[4] / (
    "artifacts/joinstream_pilot/20260810_184136/joinstream_results.json"
)


class JoinStreamGPUBackfeedTest(unittest.TestCase):
    def test_csv_int_preserves_full_globaltimer_precision(self) -> None:
        start = 1_786_364_748_014_558_976
        end = start + 40_416
        self.assertEqual(backfeed._csv_int({"timestamp": str(start)}, "timestamp"), start)
        self.assertEqual(backfeed._csv_int({"timestamp": str(end)}, "timestamp") - start, 40_416)

    @staticmethod
    def _row(
        *,
        tail: float,
        residency: str,
        mode: str,
        repeat_kind: str,
        repeat_index: int,
        variant: str,
        slot: int,
    ) -> dict[str, object]:
        start = 1_000_000 + repeat_index * 100_000
        row_materialized = start + 10_000
        target = round(tail * 1000)
        whole_end = row_materialized + target
        if variant == "A_WholeBarrier":
            all_done = whole_end
            producer_end = whole_end
            publish = observe = entry = 0
            consumer_start = producer_end + 200
        elif variant == "B_AllDoneSham":
            all_done = whole_end
            publish = all_done + 50
            producer_end = publish + 50
            entry = start - 100
            observe = publish + 50
            consumer_start = observe + 50
        else:
            publish = row_materialized + 50
            entry = start - 100
            observe = publish + 50
            consumer_start = observe + 50
            all_done = max(whole_end, row_materialized + 300)
            producer_end = all_done + 50
        consumer_end = consumer_start + 1000
        total_end = max(producer_end, consumer_end)
        return {
            "schema_version": backfeed.RAW_SCHEMA,
            "mode": mode,
            "cell_id": f"tail={tail:g}__residency={residency}",
            "tail_gap_us": tail,
            "residency": residency,
            "repeat_kind": repeat_kind,
            "repeat_index": repeat_index,
            "permutation_slot": slot,
            "variant": variant,
            "producer_blocks": 2 if residency == "tail-friendly" else 16,
            "producer_block_size": 128,
            "consumer_grid_size": 1,
            "consumer_block_size": 128,
            "producer_launches": 1,
            "consumer_launches": 1,
            "input_hash": "input-ok",
            "work_contract_hash": "work-ok",
            "tail_fma_chunks_per_thread": round(tail * 10),
            "producer_start_ns": start,
            "join_close_ns": row_materialized - 10,
            "row_materialized_ns": row_materialized,
            "flag_publish_ns": publish,
            "consumer_entry_ns": entry,
            "consumer_observe_ns": observe,
            "consumer_start_ns": consumer_start,
            "consumer_end_ns": consumer_end,
            "producer_end_ns": producer_end,
            "all_blocks_done_ns": all_done,
            "total_end_ns": total_end,
            "producer_elapsed_ns": producer_end - start,
            "consumer_end_elapsed_ns": consumer_end - start,
            "total_elapsed_ns": total_end - start,
            "visibility_latency_ns": 0 if variant == "A_WholeBarrier" else observe - publish,
            "overlap_window_ns": producer_end - consumer_start,
            "tail_calibration_error_ns": (producer_end - row_materialized) - target,
            "row_hash": "row-ok",
            "consumer_hash": "consumer-ok",
            "reference_row_hash": "row-ok",
            "reference_consumer_hash": "consumer-ok",
            "correctness_pass": "true",
            "timestamp_contract_pass": "true",
            "cuda_error": "",
            "contributors_claimed": 4,
            "expected_contributors": 4,
            "join_counter_final": 4,
            "blocks_done_final": 2 if residency == "tail-friendly" else 16,
            "expected_blocks": 2 if residency == "tail-friendly" else 16,
        }

    @classmethod
    def _rows(cls) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for tail in backfeed.TAIL_GAPS_US:
            for residency in backfeed.RESIDENCIES:
                for mode in backfeed.MODES:
                    for repeat_kind, count in (
                        ("warmup", backfeed.EXPECTED_WARMUPS),
                        ("measured", backfeed.EXPECTED_REPEATS[mode]),
                    ):
                        for repeat in range(count):
                            order = backfeed.FROZEN_PERMUTATIONS[
                                repeat % len(backfeed.FROZEN_PERMUTATIONS)
                            ]
                            for slot, variant in enumerate(order):
                                rows.append(
                                    cls._row(
                                        tail=tail,
                                        residency=residency,
                                        mode=mode,
                                        repeat_kind=repeat_kind,
                                        repeat_index=repeat,
                                        variant=variant,
                                        slot=slot,
                                    )
                                )
        return rows

    @staticmethod
    def _write_fixture(directory: Path, rows: list[dict[str, object]]) -> tuple[Path, Path]:
        raw = directory / "joinstream_gpu_raw.csv"
        meta = directory / "joinstream_gpu_meta.json"
        with raw.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(backfeed.REQUIRED_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
        meta.write_text(
            json.dumps(
                {
                    "schema": "joinstream-gpu-meta-v1",
                    "status": "COMPLETED_RAW_NOT_ADJUDICATED",
                    "timer": {
                        "source": "ptx_%globaltimer",
                        "resolution_ns": 1,
                        "cross_variant_comparison": "per_trial_elapsed_from_producer_start",
                    },
                    "contract_failures": [],
                }
            ),
            encoding="utf-8",
        )
        return raw, meta

    def test_full_frozen_fixture_supports_and_retains_both_residencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw, meta = self._write_fixture(Path(tmp), self._rows())
            result = backfeed.analyze(raw, meta, CPU_RESULT)
        self.assertEqual(result["verdict"], "SUPPORT_GPU_ACTION_SPACE")
        self.assertEqual(result["novelty_positioning"], "SUPPORTS")
        self.assertEqual(len(result["gpu_cells"]), 8)
        self.assertEqual(len(result["cpu_backfeed_cells"]), 8)
        self.assertTrue(all(len(row["residencies"]) == 2 for row in result["cpu_backfeed_cells"]))
        self.assertGreaterEqual(result["gates"]["positive_cpu_cells"], 2)

    def test_missing_gpu_cell_fails_strict_eight_cell_contract(self) -> None:
        rows = [
            row
            for row in self._rows()
            if not (float(row["tail_gap_us"]) == 30.0 and row["residency"] == "near-saturating")
        ]
        with tempfile.TemporaryDirectory() as tmp:
            raw, meta = self._write_fixture(Path(tmp), rows)
            with self.assertRaisesRegex(backfeed.ContractError, "frozen 8 cells"):
                backfeed.analyze(raw, meta, CPU_RESULT)

    def test_missing_warmup_round_fails_before_aggregation(self) -> None:
        rows = self._rows()
        rows = [
            row
            for row in rows
            if not (
                row["mode"] == "utility"
                and row["repeat_kind"] == "warmup"
                and int(row["repeat_index"]) == 29
                and float(row["tail_gap_us"]) == 5.0
                and row["residency"] == "tail-friendly"
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            raw, meta = self._write_fixture(Path(tmp), rows)
            with self.assertRaisesRegex(backfeed.ContractError, "repeat indices incomplete"):
                backfeed.analyze(raw, meta, CPU_RESULT)

    def test_join_counter_failure_is_invalid_contract(self) -> None:
        rows = self._rows()
        rows[0]["join_counter_final"] = 3
        with tempfile.TemporaryDirectory() as tmp:
            raw, meta = self._write_fixture(Path(tmp), rows)
            with self.assertRaisesRegex(backfeed.ContractError, "K-way join"):
                backfeed.analyze(raw, meta, CPU_RESULT)

    def test_wrong_six_permutation_rotation_is_rejected(self) -> None:
        rows = self._rows()
        target = [
            row
            for row in rows
            if row["mode"] == "correctness"
            and row["repeat_kind"] == "warmup"
            and int(row["repeat_index"]) == 0
            and float(row["tail_gap_us"]) == 0.0
            and row["residency"] == "tail-friendly"
        ]
        by_variant = {row["variant"]: row for row in target}
        by_variant["A_WholeBarrier"]["permutation_slot"] = 1
        by_variant["B_AllDoneSham"]["permutation_slot"] = 0
        with tempfile.TemporaryDirectory() as tmp:
            raw, meta = self._write_fixture(Path(tmp), rows)
            with self.assertRaisesRegex(backfeed.ContractError, "six-permutation"):
                backfeed.analyze(raw, meta, CPU_RESULT)

    def test_paired_tail_work_mismatch_is_rejected(self) -> None:
        rows = self._rows()
        target = next(
            row
            for row in rows
            if row["mode"] == "utility"
            and row["repeat_kind"] == "measured"
            and int(row["repeat_index"]) == 0
            and float(row["tail_gap_us"]) == 15.0
            and row["residency"] == "tail-friendly"
            and row["variant"] == "C_JoinStream"
        )
        target["tail_fma_chunks_per_thread"] = int(
            target["tail_fma_chunks_per_thread"]
        ) + 1
        with tempfile.TemporaryDirectory() as tmp:
            raw, meta = self._write_fixture(Path(tmp), rows)
            with self.assertRaisesRegex(
                backfeed.ContractError, "tail_fma_chunks_per_thread"
            ):
                backfeed.analyze(raw, meta, CPU_RESULT)

    def test_late_consumer_entry_is_rejected(self) -> None:
        rows = self._rows()
        target = next(
            row
            for row in rows
            if row["mode"] == "utility"
            and row["repeat_kind"] == "measured"
            and int(row["repeat_index"]) == 0
            and float(row["tail_gap_us"]) == 5.0
            and row["residency"] == "near-saturating"
            and row["variant"] == "B_AllDoneSham"
        )
        target["consumer_entry_ns"] = int(target["producer_start_ns"]) + 1
        with tempfile.TemporaryDirectory() as tmp:
            raw, meta = self._write_fixture(Path(tmp), rows)
            with self.assertRaisesRegex(backfeed.ContractError, "pre-enter"):
                backfeed.analyze(raw, meta, CPU_RESULT)

    def test_mad_guard_is_applied_after_paired_values_exist(self) -> None:
        distribution = backfeed.summarize([1.0, 1.0, 1.0, 10.0], 0.001)
        self.assertEqual(distribution.median, 1.0)
        self.assertEqual(distribution.mad, 0.0)
        self.assertEqual(distribution.noise_guard, 0.001)

    def test_out_of_domain_residency_is_unavailable_not_global_invalid(self) -> None:
        cpu_cells, _ = backfeed.load_cpu_cells(CPU_RESULT)
        gpu_cells: list[dict[str, object]] = []
        for residency in backfeed.RESIDENCIES:
            scale = 1.0 if residency == "tail-friendly" else 1.0 / 3.0
            for tail in backfeed.TAIL_GAPS_US:
                x = tail * scale
                metric = lambda median, noise=0.01: {
                    "median": median,
                    "mad": 0.0,
                    "p10": median,
                    "p90": median,
                    "noise_guard": noise,
                    "count": 200,
                }
                gpu_cells.append(
                    {
                        "tail_gap_us": tail,
                        "residency": residency,
                        "metrics": {
                            "actual_tail_window_us": metric(x),
                            "critical_gain_vs_whole_us": metric(max(0.0, x - 0.1)),
                            "producer_tax_us": metric(0.0),
                        },
                    }
                )
        result = backfeed.backfeed_cpu_cells(cpu_cells, gpu_cells)
        target = next(row for row in result if row["query_headroom_us"] == 15.0)
        by_residency = {row["residency"]: row for row in target["residencies"]}
        self.assertEqual(by_residency["near-saturating"]["status"], "unavailable_no_extrapolation")
        self.assertTrue(by_residency["tail-friendly"]["positive"])
        self.assertTrue(target["positive"])

    def test_blocked_no_gpu_does_not_require_raw_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta = Path(tmp) / "meta.json"
            meta.write_text(json.dumps({"status": "BLOCKED_NO_GPU"}), encoding="utf-8")
            result = backfeed.analyze(Path(tmp) / "missing.csv", meta, CPU_RESULT)
        self.assertEqual(result["verdict"], "BLOCKED_NO_GPU")
        self.assertEqual(result["novelty_positioning"], "DOES_NOT_ADDRESS")


if __name__ == "__main__":
    unittest.main()
