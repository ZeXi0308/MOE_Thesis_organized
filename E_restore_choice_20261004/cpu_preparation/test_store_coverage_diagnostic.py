"""Focused false-attribution checks for the CPU STORE diagnostic.

Run: python3 -B cpu_preparation/test_store_coverage_diagnostic.py
Uses small synthetic evidence only; no vLLM import, GPU or remote access.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnose_store_coverage import analyze, classify_gap, compare, completion_view, markdown, position_work


class StoreCoverageEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.followup = dict(hit_chunks=7, time=10.0, scheduler_step=10,
                             clock_domain="episode:test-a")
        self.job = dict(request_id="synthetic/request-alpha", job_id=41,
                        creation_record_index=3, mapping_complete=True,
                        logical_block_intervals=[[7, 9]],
                        content_version="prefix-v1", ack_step_end_s=5.0,
                        completion=dict(acknowledged_before_next_lookup=None,
                                        reports=[]))
        self.eviction = dict(kind="EVICT", request_id=self.job["request_id"],
                             job_id=41, creation_record_index=3,
                             storage_tier="Host", time=7.0,
                             clock_domain="episode:test-a", scheduler_step=7,
                             logical_block_interval=[7, 8],
                             content_version="prefix-v1")

    def category(self, jobs=None, builders_complete=True,
                 repeated_full_block=True, evictions=()):
        return classify_gap(7, self.followup, [self.job] if jobs is None else jobs,
                            builders_complete, repeated_full_block, evictions)[0]

    def acknowledge(self):
        self.job["completion"]["acknowledged_before_next_lookup"] = True

    def test_missing_ack_is_not_pending(self):
        self.assertEqual(self.category(), "INSUFFICIENT_EVIDENCE")

    def test_positive_pending_snapshot_after_lookup_is_distinct_from_no_ack(self):
        self.job["completion"]["reports"] = [dict(present_before=True,
            pending_count_before=1, entry_time_s=11.0, step=11)]
        self.assertEqual(self.category(), "NOT_YET_COMPLETED")
        # An older pending snapshot cannot rule out a terminal ACK before lookup.
        self.job["completion"]["reports"][0].update(entry_time_s=9.0, step=9)
        self.assertEqual(self.category(), "INSUFFICIENT_EVIDENCE")

    def test_ack_then_miss_without_eviction_is_inconclusive(self):
        self.acknowledge()
        self.assertEqual(self.category(), "INSUFFICIENT_EVIDENCE")

    def test_reused_job_id_cannot_join_another_creation(self):
        record = dict(operation="store_completion", record_index=5,
            phase="service", step_index=4, time_s=4.0, status="returned",
            before=[dict(job_id=41, creation_record_index=2,
                request_id=self.job["request_id"], reported_count=1,
                present_before=True, pending_count_before=1)],
            after=[dict(job_id=41, removed_by_native=True, pending_count_after=0)])
        view = completion_view([record], {"job_id": 41}, 3,
            self.job["request_id"], dict(step=10, decision_s=10.0))
        self.assertEqual(view["reports"], [])
        self.assertIsNone(view["acknowledged_before_next_lookup"])
        self.job["completion"] = view
        self.assertEqual(self.category(), "INSUFFICIENT_EVIDENCE")
        # The same identity requirement applies to supplied eviction records.
        self.acknowledge()
        self.eviction["creation_record_index"] = 2
        self.assertEqual(self.category(evictions=[self.eviction]),
                         "INSUFFICIENT_EVIDENCE")

    def test_cross_clock_or_changed_content_does_not_prove_eviction(self):
        self.acknowledge()
        for field, value in (("clock_domain", "episode:other-run"),
                             ("content_version", "prefix-v2")):
            with self.subTest(field=field):
                eviction = dict(self.eviction, **{field: value})
                self.assertEqual(self.category(evictions=[eviction]),
                                 "INSUFFICIENT_EVIDENCE")

    def test_explicit_ordered_eviction_of_same_lifecycle(self):
        self.acknowledge()
        self.assertEqual(self.category(evictions=[self.eviction]),
                         "COMPLETED_THEN_EVICTED")
        # A record before ACK or after lookup cannot explain this observed miss.
        for time in (4.0, 11.0):
            with self.subTest(time=time):
                self.assertEqual(self.category(evictions=[dict(self.eviction, time=time)]),
                                 "INSUFFICIENT_EVIDENCE")

    def test_not_created_requires_complete_window_and_full_block_recompute(self):
        self.assertEqual(self.category(jobs=[]), "NOT_CREATED")
        for complete, recomputed in ((False, True), (True, False), (False, False)):
            with self.subTest(complete=complete, recomputed=recomputed):
                self.assertEqual(self.category(jobs=[], builders_complete=complete,
                    repeated_full_block=recomputed), "INSUFFICIENT_EVIDENCE")
        # Unknown mapping invalidates completeness, even if the returned list
        # contains a job whose apparent interval does not cover the target.
        unknown = dict(self.job, mapping_complete=False)
        self.assertEqual(self.category(jobs=[unknown], builders_complete=False),
                         "INSUFFICIENT_EVIDENCE")

    def test_preceding_hole_and_later_hit_do_not_support_missing_block_claim(self):
        self.followup["hit_chunks"] = 6
        self.assertEqual(self.category(jobs=[]), "INSUFFICIENT_EVIDENCE")
        self.followup["hit_chunks"] = 8
        self.assertIsNone(self.category(jobs=[]))

    def test_request_identity_is_a_join_key_not_a_special_case(self):
        self.acknowledge()
        for request_id in ("unseen/albatross", "new-trace/request-9001"):
            with self.subTest(request_id=request_id):
                job = deepcopy(self.job)
                job["request_id"] = request_id
                eviction = dict(self.eviction, request_id=request_id)
                self.assertEqual(self.category(jobs=[job], evictions=[eviction]),
                                 "COMPLETED_THEN_EVICTED")
                # Retaining the old ID must break the evidence join.
                self.assertEqual(self.category(jobs=[job], evictions=[self.eviction]),
                                 "INSUFFICIENT_EVIDENCE")
                raw = {"scheduler_steps": [
                    {"time_s": 1.0, "scheduled": [{"request_id": request_id,
                        "start_computed": 0, "end_computed": 16}]},
                    {"time_s": 2.0, "scheduled": [{"request_id": request_id,
                        "start_computed": 8, "end_computed": 24}]}]}
                rows, totals = position_work(raw)
                self.assertEqual(set(rows), {request_id})
                self.assertEqual(totals[request_id], dict(scheduled_positions=32,
                    unique_positions=24, repeated_positions=8))


class WholeArrivalAccountingTests(unittest.TestCase):
    """A comparison must not turn an intersection into all-arrival accounting."""
    def setUp(self):
        self.raw = dict(requests=[dict(request_id='a', external_id='one', completed=False),
                                  dict(request_id='b', external_id='two', completed=False)],
            steps=[], scheduler_steps=[dict(time_s=1.0, scheduled=[dict(
                request_id='a', start_computed=0, end_computed=16)])])

    def read(self, raw):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'raw.json'
            path.write_text(json.dumps(raw))
            return analyze(path)

    def test_unscheduled_arrival_is_kept_but_missing_trace_is_unknown(self):
        result = self.read(self.raw)
        self.assertEqual(result['costs']['work_by_external_id']['two'],
                         dict(scheduled_positions=0, unique_positions=0, repeated_positions=0))
        del self.raw['scheduler_steps']
        self.assertIsNone(self.read(self.raw)['costs']['work_by_external_id'])

    def test_duplicate_identity_cannot_silently_overwrite_costs(self):
        for field in ('request_id', 'external_id'):
            with self.subTest(field=field):
                raw = deepcopy(self.raw)
                raw['requests'][1][field] = raw['requests'][0][field]
                with self.assertRaisesRegex(ValueError, 'Duplicate'):
                    self.read(raw)

    def test_partial_or_different_request_sets_have_no_whole_service_delta(self):
        first = self.read(self.raw)
        second = deepcopy(first)
        del second['costs']['work_by_external_id']['two']
        for count in (2, 1):
            with self.subTest(arrived=count):
                second['service']['arrived'] = count
                result = compare(first, second)
                self.assertEqual(result['status'], 'INCOMPARABLE')
                self.assertIsNone(result['all_request_position_delta'])
                rendered = markdown(dict(traces=[], comparison=result))
                self.assertIn('对照不可比较', rendered)
                self.assertNotIn('全部请求重复位置差', rendered)

    def test_same_all_arrivals_remain_comparable_without_invented_peer_group(self):
        first = self.read(self.raw)
        result = compare(first, deepcopy(first))
        self.assertEqual(result['status'], 'COMPARABLE_RECORDED_WORK')
        self.assertEqual(result['all_request_position_delta']['repeated_positions'], 0)
        self.assertIsNone(result['peer_repeated_position_delta'])  # no repair target
        self.assertIn('STORE 覆盖诊断', markdown(dict(traces=[], comparison=None)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
