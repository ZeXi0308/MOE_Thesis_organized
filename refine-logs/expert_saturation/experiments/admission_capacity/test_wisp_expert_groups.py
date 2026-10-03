"""CPU-only partition invariants; no kernel or performance validation."""
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import runpy
from types import SimpleNamespace as NS
import unittest

from native_pager_context import install_row_context
from wisp_expert_groups import matched_hash_protection, partition_experts, partition_protected_experts, partition_protected_experts_late, retention_plan


class ExpertGroupsTest(unittest.TestCase):
    def test_coverage_capacity_and_entry_misses_once(self):
        resident, active = set(range(24)), set(range(64))
        groups = partition_experts(active, resident, 24)
        self.assertEqual(Counter(e for g in groups for e in g), Counter(active))
        self.assertTrue(all(0 < len(g) <= 24 for g in groups))
        self.assertEqual(groups[0], list(range(24)))
        self.assertEqual([e for g in groups for e in g if e not in resident],
                         list(range(24, 64)))

    def test_resident_first_then_sorted_missing(self):
        self.assertEqual(partition_experts([1, 2, 4, 7, 8, 9], [1, 7, 20], 4),
                         [[1, 7, 2, 4], [8, 9]])

    def test_all_resident_needs_one_group(self):
        self.assertEqual(partition_experts([4, 1, 4], [1, 4, 7], 4), [[1, 4]])

    def test_empty_input(self):
        self.assertEqual(partition_experts([], [1, 4], 4), [])


class ProtectedGroupsTest(unittest.TestCase):
    def test_current_row_eligibility_frequency_and_padding_fallback(self):
        rows = [[0, 1], [2, 3], [2, 3], [2, 3], [4, 5], [6, 7]]
        metadata = [dict(computed_position=32, prompt_tokens=32)] + [
            dict(computed_position=i, prompt_tokens=128) for i in range(5)]
        context = dict(rows=metadata, row_request_order_verified=True, valid_row_start=0, valid_row_stop=6)
        plans = {m: retention_plan(rows, context, {0, 1, 4, 5}, 4, m, salt="seed0/layer0/step4")
                 for m in ("none", "frequency", "decode", "matched_hash")}
        self.assertEqual(plans["decode"][1]["chosen_protected_experts"], [0, 1])
        self.assertTrue(plans["decode"][1]["applied"])
        self.assertEqual(plans["frequency"][1]["chosen_protected_experts"], [2, 3])
        self.assertEqual(plans["frequency"][1]["protected_entry_resident"], [])
        self.assertEqual(len(plans["matched_hash"][1]["protected_entry_resident"]), 2)
        self.assertEqual(plans["none"][1]["fallback_reason"], "disabled")
        metadata[0]["computed_position"] = 31  # P-1 is still a prefill row.
        self.assertEqual(retention_plan(rows, context, {0, 1, 4, 5}, 4, "decode", salt="0")[1]["fallback_reason"], "not_mixed")
        context["valid_row_stop"] = 5
        plan, info = retention_plan(rows, context, {0, 1, 4, 5}, 4, "decode", salt="0")
        self.assertEqual(info["fallback_reason"], "unmapped_or_padded_rows")
        self.assertIsNone(info["real_decode_experts"])
        self.assertEqual([g["execute_experts"] for g in plan], partition_experts(range(8), {0, 1, 4, 5}, 4))

    def test_prompt_lengths_follow_actual_post_reorder_rows(self):
        batch = NS(req_ids=["b", "a"], num_reqs=2, num_computed_tokens_cpu=[31, 32], num_prompt_tokens=[32, 64])
        original = lambda *a: "prepared"
        runner = NS(use_async_scheduling=False, num_spec_tokens=0, parallel_config=NS(use_ubatching=False),
            input_batch=batch, req_indices=NS(np=[0, 0, 1]), _prepare_inputs=original)
        runtime = NS(context=dict(phase="measurement", step_id=4))
        pager = NS(_runtime=runtime, set_context=lambda **kw: setattr(runtime, "context", kw))
        uninstall = install_row_context(runner, pager)
        output = NS(total_num_scheduled_tokens=3, num_scheduled_tokens={"a": 1, "b": 2})
        self.assertEqual(runner._prepare_inputs(output, [2, 1]), "prepared")
        self.assertEqual([(r["internal_request_id"], r["computed_position"], r["prompt_tokens"])
                          for r in runtime.context["rows"]], [("b", 31, 32), ("b", 32, 32), ("a", 32, 64)])
        uninstall()
        self.assertIs(runner._prepare_inputs, original)

    def check_plan(self, active, resident, protected, cap):
        plan = partition_protected_experts(active, resident, protected, cap)
        current, seen, loaded = set(resident), set(), Counter()
        self.assertEqual(Counter(e for g in plan for e in g["execute_experts"]), Counter(active))
        if plan:
            self.assertTrue(active & resident <= set(plan[0]["execute_experts"]))
        for group in plan:
            execute, ensure = set(group["execute_experts"]), set(group["ensure_experts"])
            self.assertTrue(execute <= ensure and 0 < len(ensure) <= cap)
            self.assertEqual(ensure - execute, protected & seen)
            misses = ensure - current
            loaded.update(misses)
            # A legal adversarial eviction choice, independent of partition order.
            victims = sorted(current - ensure, reverse=True)[:max(0, len(current | ensure) - cap)]
            current.difference_update(victims)
            current.update(ensure)
            seen.update(execute)
            self.assertLessEqual(len(current), cap)
            self.assertTrue(protected & seen <= current)
            self.assertEqual(set(group["protected_after"]), protected & seen)
        self.assertEqual(loaded, Counter(active - resident))
        self.assertTrue(protected <= current)
        return plan

    def test_exhaustive_small_sets_and_matched_control(self):
        universe, cap = range(5), 3
        subsets = [set(s) for n in range(6) for s in combinations(universe, n)]
        for active in subsets:
            for resident in (s for s in subsets if len(s) <= cap):
                for protected in (s for s in subsets if s <= active and len(s) < cap):
                    plan = self.check_plan(active, resident, protected, cap)
                    control = set(matched_hash_protection(active, resident, protected, salt="seed0/layer0/step4"))
                    self.assertEqual((len(control), len(control & resident)), (len(protected), len(protected & resident)))
                    control_plan = self.check_plan(active, resident, control, cap)
                    self.assertEqual([(len(g["execute_experts"]), len(g["ensure_experts"])) for g in plan],
                                     [(len(g["execute_experts"]), len(g["ensure_experts"])) for g in control_plan])
                    if not protected:
                        self.assertEqual([g["execute_experts"] for g in plan], partition_experts(active, resident, cap))

    def test_cap24_protection_priority_and_deterministic_control(self):
        active, resident, protected = set(range(64)), set(range(24)), set(range(7)) | set(range(53, 64))
        plan = self.check_plan(active, resident, protected, 24)
        self.assertEqual(plan[0]["execute_experts"], list(range(24)))
        self.assertEqual(plan[1]["execute_experts"][:11], list(range(53, 64)))
        choose = lambda a, r, p: matched_hash_protection(a, r, p, salt="seed0/layer0/step4")
        self.assertEqual(choose(active, resident, protected),
                         choose(sorted(active, reverse=True), sorted(resident, reverse=True), sorted(protected, reverse=True)))
        self.assertEqual(partition_protected_experts([], resident, [], 24), [])
        with self.assertRaises(ValueError):
            partition_protected_experts(active, resident, set(range(24)), 24)
        with self.assertRaises(ValueError):
            partition_protected_experts(active, resident, {64}, 24)


class LateProtectedGroupsTest(unittest.TestCase):
    def test_small_sets_attain_bound_without_extra_loads(self):
        cap = 3
        subsets = [set(s) for n in range(5) for s in combinations(range(4), n)]
        for active in subsets:
            for entry in (s for s in subsets if len(s) <= cap):
                for protected in (s for s in subsets if s <= active and len(s) < cap):
                    plan = partition_protected_experts_late(active, entry, protected, cap)
                    a = len(protected & entry)
                    bound = max(bool(active), (len(active) - a + cap - a - 1) // (cap - a))
                    self.assertEqual(len(plan), bound)
                    self.assertEqual(Counter(e for g in plan for e in g["execute_experts"]), Counter(active))
                    for reverse in (False, True):
                        current, loaded, held = set(entry), Counter(), protected & entry
                        for g in plan:
                            execute, ensure = set(g["execute_experts"]), set(g["ensure_experts"])
                            held |= execute & protected
                            self.assertEqual(ensure, execute | held)
                            self.assertEqual(set(g["protected_after"]), held)
                            self.assertLessEqual(len(ensure), cap)
                            loaded.update(ensure - current)
                            victims = sorted(current - ensure, reverse=reverse)[:max(0, len(current | ensure) - cap)]
                            current = (current - set(victims)) | ensure
                            self.assertLessEqual(len(current), cap)
                            self.assertTrue(held <= current)
                        self.assertEqual(loaded, Counter(active - entry))
                        self.assertTrue(protected <= current)
                    self.assertTrue(all(not protected.intersection(g["execute_experts"]) for g in plan[:-1]))
                    if not protected:
                        self.assertEqual(plan, partition_protected_experts(active, entry, protected, cap))

    def test_order_preserves_selector_and_default_early(self):
        rows = [[0, 1], [2, 3], [2, 3], [4, 5], [6, 7]]
        context = dict(row_request_order_verified=True, valid_row_start=0, valid_row_stop=5,
                       rows=[dict(computed_position=32 if i == 0 else i, prompt_tokens=32) for i in range(5)])
        for mode in ("none", "frequency", "decode", "matched_hash"):
            args = (rows, context, {0, 1, 4, 5}, 4, mode)
            default = retention_plan(*args, salt="fixed")
            self.assertEqual(default, retention_plan(*args, salt="fixed", order="early"))
            late, info = retention_plan(*args, salt="fixed", order="late")
            self.assertEqual(info["chosen_protected_experts"], default[1]["chosen_protected_experts"])
            self.assertEqual(info["order"], "late")
            if mode == "none":self.assertEqual(late, default[0])
        with self.assertRaises(ValueError):retention_plan(*args, salt="fixed", order="unknown")


class GroupBudgetGuardTest(unittest.TestCase):
    def test_accept_reject_and_unchanged_default(self):
        rows = [[0, 1], [2, 3], [2, 3], [4, 5], [5, 6]]
        context = dict(row_request_order_verified=True, valid_row_start=0, valid_row_stop=len(rows),
            rows=[dict(computed_position=32 if i == 0 else i, prompt_tokens=32) for i in range(len(rows))])
        for resident, rejected in (({0, 1, 2, 4}, False), ({0, 1, 2, 3}, True)):
            args = (rows, context, resident, 4, "frequency")
            baseline = retention_plan(*args, salt="fixed", order="late")
            self.assertEqual(baseline, retention_plan(*args, salt="fixed", order="late", guard="none"))
            plan, info = retention_plan(*args, salt="fixed", order="late", guard="no_extra_groups")
            self.assertEqual(info["candidate_protected_experts"], [2, 3])
            self.assertTrue(info["candidate_eligible"])
            self.assertEqual((info["guard_rejected"], info["eligible"], info["applied"]), (rejected, not rejected, not rejected))
            self.assertEqual(info["candidate_group_lower_bound"], 3 if rejected else 2)
            self.assertEqual((info["ordinary_group_count"], info["executed_group_count"]), (2, 2))
            if rejected:
                self.assertEqual(info["chosen_protected_experts"], [])
                self.assertEqual(info["fallback_reason"], "extra_groups")
                self.assertEqual([g["execute_experts"] for g in plan], partition_experts(range(7), resident, 4))
            else:
                self.assertEqual(info["chosen_protected_experts"], [2, 3])
                self.assertEqual(plan, baseline[0])

    def test_invalid_guard_combinations(self):
        for mode, order, guard in (("none", "late", "no_extra_groups"), ("decode", "late", "no_extra_groups"),
                ("matched_hash", "late", "no_extra_groups"), ("frequency", "early", "no_extra_groups"),
                ("frequency", "late", "unknown")):
            with self.assertRaises(ValueError):
                retention_plan([], {}, set(), 24, mode, salt="fixed", order=order, guard=guard)

    def test_retained_6144_current_calls_guard_invariants(self):
        base = Path(__file__).resolve().parents[2] / "outputs/admission_capacity/20260912_wisp_olmoe_r01"
        campaign = base / "retention_lifecycle_performance"
        if not campaign.exists(): self.skipTest("retained local campaign unavailable")
        original = runpy.run_path(str(base / "retention_lifecycle_frozen/source/wisp_expert_groups.py"))["retention_plan"]
        counts = Counter()
        execution = json.loads((campaign / "results/execution.json").read_text())
        for cell in execution["cells"]:
            with (campaign / "results" / cell["label"] / "pager/calls.jsonl").open() as stream:
                for line in stream:
                    record = json.loads(line)
                    if not record["measurement"]: continue
                    active, entry = set(record["active_experts"]), set(record["entry_resident_experts"])
                    args = (record["row_topk_experts"], record["context"], entry, 24, "frequency")
                    prior, old = original(*args, salt="fixed", order="late")
                    default, info = retention_plan(*args, salt="fixed", order="late")
                    self.assertEqual(default, prior)
                    self.assertEqual({k:info[k] for k in old}, old)
                    plan, info = retention_plan(*args, salt="fixed", order="late", guard="no_extra_groups")
                    self.assertEqual(info["candidate_protected_experts"], old["chosen_protected_experts"])
                    self.assertEqual(info["candidate_group_lower_bound"], len(prior))
                    self.assertEqual(info["chosen_protected_experts"], [] if info["guard_rejected"] else old["chosen_protected_experts"])
                    self.assertEqual(len(plan), (len(active) + 23) // 24)
                    self.assertEqual(Counter(e for g in plan for e in g["execute_experts"]), Counter(active))
                    protected = set(info["chosen_protected_experts"])
                    for reverse in (False, True):
                        current, loaded, held = set(entry), Counter(), protected & entry
                        for group in plan:
                            execute, ensure = set(group["execute_experts"]), set(group["ensure_experts"])
                            held |= protected & execute
                            self.assertEqual((ensure, set(group["protected_after"])), (execute | held, held))
                            self.assertLessEqual(len(ensure), 24)
                            loaded.update(ensure-current)
                            victims = sorted(current-ensure, reverse=reverse)[:max(0, len(current | ensure)-24)]
                            current = (current-set(victims)) | ensure
                            self.assertTrue(held <= current and len(current) <= 24)
                        self.assertEqual(loaded, Counter(active-entry))
                        self.assertTrue(protected <= current)
                    counts.update(calls=1, rejected=int(info["guard_rejected"]), applied=int(info["applied"]))
        self.assertEqual(counts, Counter(calls=6144, rejected=1136, applied=1904))


if __name__ == "__main__":
    unittest.main()
