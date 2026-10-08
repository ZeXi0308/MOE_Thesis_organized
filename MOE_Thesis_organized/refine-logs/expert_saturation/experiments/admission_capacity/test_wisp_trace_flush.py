import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import wisp_expert_groups
import wisp_v026_adapter as adapter


def runtime_at(path):
    path.mkdir()
    return adapter._Runtime(24, path, SimpleNamespace(cuda=SimpleNamespace(synchronize=Mock())),
                            SimpleNamespace(__file__=adapter.__file__), None)


def append_record(runtime, index):
    group = dict(miss=index + 1, evict=index, weight_copy_bytes=(index + 1) * 100,
                 load_cuda_span_ms=None)
    record = dict(call_id=runtime.next_call_id, measurement=index > 0,
        status="failed" if index == 3 else "complete", groups=[group], group_count=1,
        miss=index + 1, evict=index, weight_copy_bytes=(index + 1) * 100,
        context=dict(phase="measurement" if index else "warmup"), row_topk_experts=[[index]])
    runtime.records.append(record)
    runtime.events.append((group, (SimpleNamespace(elapsed_time=lambda end: index + 0.5), object())))


class TraceFlushTest(unittest.TestCase):
    def test_batches_equal_one_write_and_keep_global_call_ids_and_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            results = []
            for name, split in (("once", False), ("batches", True)):
                runtime = runtime_at(Path(directory) / name)
                for i in range(4):
                    self.assertEqual(runtime.next_call_id, i)
                    append_record(runtime, i)
                    if split and i in (0, 2):
                        report = runtime.flush_records(f"boundary-{i}")
                        self.assertEqual((len(runtime.records), len(runtime.events)), (0, 0))
                        self.assertEqual(report["flushed_calls_after"], i + 1)
                        self.assertLessEqual(report["sync_start_perf_ns"], report["sync_end_perf_ns"])
                summary = runtime.finalize()
                calls = (runtime.outdir / "calls.jsonl").read_bytes()
                self.assertEqual([r["call_id"] for r in map(json.loads, calls.splitlines())], list(range(4)))
                self.assertEqual(sum(r["bytes_written"] for r in runtime.flushes), len(calls))
                self.assertEqual((summary["all_calls"], summary["measurement_calls"], summary["failed_calls"]), (4, 3, 1))
                self.assertEqual(summary["all"], dict(miss=10, evict=6, weight_copy_bytes=1000, group_count=4, load_cuda_span_ms=8.0))
                self.assertEqual(summary["measurement"], dict(miss=9, evict=6, weight_copy_bytes=900, group_count=3, load_cuda_span_ms=7.5))
                self.assertEqual(summary["status"], "failed")
                self.assertEqual(runtime.next_call_id, 4)
                self.assertIs(runtime.finalize(), summary)
                results.append((calls, {k: v for k, v in summary.items() if k != "trace_flushes"}))
            self.assertEqual(*results)

    def test_finalize_after_last_episode_flush_keeps_existing_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = runtime_at(Path(directory) / "empty-finalize")
            runtime.flush_records("empty-initial-boundary")
            append_record(runtime, 0)
            runtime.flush_records("last-episode")
            before = (runtime.outdir / "calls.jsonl").read_bytes()
            summary = runtime.finalize()
            self.assertEqual((runtime.outdir / "calls.jsonl").read_bytes(), before)
            self.assertEqual(summary["all_calls"], 1)
            self.assertEqual(runtime.flushes[-1]["records_written"], 0)
            self.assertEqual(runtime.flushes[-1]["held_events_after"], 0)

    def test_partial_write_retains_references_and_blocks_retry_and_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = runtime_at(Path(directory) / "failed")
            for i in range(2):append_record(runtime, i)
            records, events = list(runtime.records), list(runtime.events)
            original_open, writes = Path.open, []
            class PartialWriter:
                def __enter__(self):
                    self.stream = original_open(runtime.outdir / "calls.jsonl", "xb")
                    return self
                def write(self, data):
                    writes.append(data)
                    return self.stream.write(data if len(writes) == 1 else data[:len(data) // 2])
                def __exit__(self, *args):self.stream.close()
            with patch.object(Path, "open", return_value=PartialWriter()):
                with self.assertRaises(OSError):runtime.flush_records("injected-write-failure")
            partial = (runtime.outdir / "calls.jsonl").read_bytes()
            self.assertEqual(partial, writes[0] + writes[1][:len(writes[1]) // 2])
            self.assertEqual(runtime.flushed_calls, 0)
            self.assertTrue(all(a is b for a, b in zip(runtime.records, records)))
            self.assertTrue(all(a is b for a, b in zip(runtime.events, events)))
            self.assertEqual((len(runtime.records), len(runtime.events)), (2, 2))
            for operation in (lambda: runtime.flush_records("retry"), lambda: runtime.finalize("failed"),
                              lambda: runtime.apply(None, None, None, None, None),
                              lambda: wisp_expert_groups._apply(runtime, None, None, None, None, None)):
                with self.assertRaises(RuntimeError):operation()
            self.assertEqual((runtime.outdir / "calls.jsonl").read_bytes(), partial)


if __name__ == "__main__":
    unittest.main()
