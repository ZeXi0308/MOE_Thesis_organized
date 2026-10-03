"""Synthetic DAG fixtures only; these tests are not EP or performance evidence."""
import copy
import unittest

from analyze_exposed_return import analyze, report


def node(name, start, end, resource, return_path=False, release=0):
    return dict(id=name, start_us=start, end_us=end, resources=[resource], request_ids=["r0"],
        phase="return_a2a" if return_path else "other", return_path=return_path,
        message_ids=["m0"] if return_path else [], release=dict(time_us=release,
        kind="external_event" if release else "request_arrival", request_id="r0",
        action_independent=True, evidence="synthetic fixture external timer" if release else "synthetic arrival"))


def fixture(nodes, dependencies, resources, completion):
    return dict(schema="ep-return-explicit-dag-v1", provenance="synthetic_fixture",
        coverage={k: True for k in ("causal_dependencies", "resource_serialization", "exogenous_releases",
                                   "request_completions", "all_moe_layers", "message_identity")},
        nodes=nodes, dependency_edges=[dict(source=a, target=b, evidence="synthetic causal edge") for a, b in dependencies],
        resource_sequences=[dict(resource_id=r, node_ids=ids, evidence="synthetic explicit serialized resource")
                            for r, ids in resources.items()], requests=[dict(request_id="r0", arrival_us=0,
        completion_node="done", completion_us=completion,
        tokens=[dict(token_index=0, completion_node="done", completion_us=completion)])])


class ExposedReturnTest(unittest.TestCase):
    def assert_bound(self, trace, saving):
        result = analyze(trace)
        self.assertEqual(result["status"], "SYNTHETIC_FIXTURE_ONLY", result["reasons"])
        self.assertFalse(result["scientific_go"])
        self.assertEqual(result["fixed_order_structural_conditional_bound"]["requests"][0]["completion_shift_us"], saving)
        self.assertIn("exporter", report(result))
        return result

    def test_overlapped_return_duration_is_not_request_saving(self):
        trace = fixture([node("expert", 0, 2, "compute0"), node("return", 2, 7, "network", True),
                         node("parallel", 0, 8, "compute1"), node("done", 8, 9, "sampling")],
                        [("expert", "return"), ("return", "done"), ("parallel", "done")],
                        {"compute0": ["expert"], "network": ["return"], "compute1": ["parallel"], "sampling": ["done"]}, 9)
        trace["negative_control_node_ids"] = ["expert"]
        result = self.assert_bound(trace, 0)
        self.assertTrue(result["negative_control"]["zero_completion_shift"])

    def test_resource_order_is_preserved_and_propagates_completion(self):
        trace = fixture([node("return", 0, 4, "serialized", True), node("work", 4, 7, "serialized"),
                         node("done", 7, 7, "marker")], [("work", "done")],
                        {"serialized": ["return", "work"], "marker": ["done"]}, 7)
        result = self.assert_bound(trace, 4)
        self.assertEqual(result["resource_serialization_edges"], 1)
        trace["resource_sequences"][0]["node_ids"] = ["work"]
        self.assertEqual(analyze(trace)["status"], "NOT_IDENTIFIED")

    def test_exogenous_release_stays_fixed_but_observed_start_is_rejected(self):
        trace = fixture([node("return", 0, 4, "network", True), node("work", 6, 8, "compute", release=6),
                         node("done", 8, 8, "marker")], [("return", "work"), ("work", "done")],
                        {"network": ["return"], "compute": ["work"], "marker": ["done"]}, 8)
        self.assert_bound(trace, 0)
        trace["nodes"][1]["release"]["kind"] = "observed_start"
        rejected = analyze(trace)
        self.assertEqual(rejected["status"], "NOT_IDENTIFIED")
        self.assertIsNone(rejected["fixed_order_structural_conditional_bound"])

    def test_missing_dependencies_wrong_identity_and_unexplained_gap_not_identified(self):
        original = fixture([node("return", 0, 4, "network", True), node("done", 4, 5, "sampling")],
                           [("return", "done")], {"network": ["return"], "sampling": ["done"]}, 5)
        for issue in ("chronology_only", "wrong_request_completion", "unexplained_gap"):
            with self.subTest(issue=issue):
                trace = copy.deepcopy(original)
                if issue == "chronology_only":
                    trace["dependency_edges"] = []
                elif issue == "wrong_request_completion":
                    trace["requests"][0]["completion_node"] = "missing"
                else:
                    trace["nodes"][1]["start_us"], trace["nodes"][1]["end_us"] = 5, 6
                result = analyze(trace)
                self.assertEqual(result["status"], "NOT_IDENTIFIED")
                self.assertIsNone(result["fixed_order_structural_conditional_bound"])


if __name__ == "__main__":
    unittest.main()
