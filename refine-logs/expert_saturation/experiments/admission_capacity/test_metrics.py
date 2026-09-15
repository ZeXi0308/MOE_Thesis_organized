import unittest

from metrics import summarize_requests, summarize_episode_requests


def request(request_id, arrival, tokens, status="completed", admission=None, completion=None):
    return {
        "request_id": request_id,
        "arrival_s": arrival,
        "admission_s": arrival if admission is None else admission,
        "token_times_s": tokens,
        "output_token_ids": list(range(len(tokens))),
        "status": status,
        "completion_s": (tokens[-1] if tokens else arrival) if status == "completed" and completion is None else completion,
    }


class RequestMetricsTest(unittest.TestCase):
    def summarize(self, rows, end=20, ttft=3, tpot=2):
        return summarize_requests(rows, observation_end_s=end, ttft_slo_s=ttft, tpot_slo_s=tpot)

    def test_queue_in_ttft_and_mean_tpot_not_completion_time(self):
        result = self.summarize([request("a", 10, [13, 14, 17], admission=12, completion=19)])
        row = result["per_request"][0]
        self.assertEqual((row["queue_s"], row["ttft_s"], row["tpot_s"]), (2, 3, 2))
        self.assertEqual(row["itl_s"], [1, 3])
        self.assertEqual(row["request_latency_s"], 9)
        self.assertTrue(row["slo_pass"])
        self.assertEqual(result["latency_s"]["itl"]["p50"], 2)
        self.assertTrue(result["latency_s"]["itl"]["small_sample"])

    def test_all_arrivals_denominator_retains_failed_and_unfinished(self):
        rows = [request("a", 10, [11, 13]), request("b", 12, [13], "failed"),
                request("c", 14, [], "unfinished"), request("d", 15, [16, 17], "unfinished")]
        result = self.summarize(rows)
        self.assertEqual([result[k] for k in ("n_arrived", "n_completed", "n_failed", "n_unfinished", "n_slo_pass")], [4, 1, 1, 2, 1])
        self.assertEqual(result["slo_attainment"], 0.25)
        self.assertEqual(result["goodput_rps"], 0.1)
        self.assertEqual(result["throughput_rps"], 0.1)
        self.assertEqual(result["latency_s"]["ttft"]["n"], 3)
        self.assertFalse(result["per_request"][1]["slo_pass"])
        self.assertFalse(result["per_request"][3]["slo_pass"])

    def test_single_token_tpot_is_null_and_only_completed_is_vacuous(self):
        result = self.summarize([request("a", 0, [1]), request("b", 0, [5]), request("c", 0, [1], "failed")])
        self.assertEqual(result["n_completed_tpot_vacuous"], 2)
        self.assertEqual(result["n_slo_pass"], 1)
        self.assertIsNone(result["per_request"][0]["tpot_s"])
        self.assertTrue(result["per_request"][0]["tpot_pass"])
        self.assertFalse(result["per_request"][2]["tpot_pass"])
        self.assertIsNone(result["latency_s"]["tpot"]["p99"])

    def test_observation_end_covers_arrival_token_and_completion(self):
        for row in [request("a", 21, [], "unfinished"), request("b", 0, [21], "unfinished"),
                    request("c", 0, [1], completion=21)]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.summarize([row])

    def test_reversed_times_and_misaligned_token_ids_are_invalid(self):
        bad = request("b", 0, [1, 2])
        bad["output_token_ids"] = [0]
        for row in [request("a", 0, [2, 1]), bad]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.summarize([row])

    def test_zero_duration_is_explicit(self):
        result = self.summarize([request("a", 20, [20])])
        self.assertIsNone(result["goodput_rps"])
        self.assertEqual(self.summarize([])["goodput_rps"], 0)

    def test_early_abort_keeps_planned_future_without_inventing_arrivals(self):
        failed = request("failed", 0, [1], "failed")
        waiting = request("waiting", 0, [], "unfinished")
        future = request("future", 10, [], "unfinished")
        waiting["admission_s"] = future["admission_s"] = None
        result = summarize_episode_requests([failed, waiting, future], observation_end_s=2,
                                            ttft_slo_s=3, tpot_slo_s=2)
        self.assertEqual((result["n_planned"], result["n_arrived"], result["n_not_yet_arrived"]), (3, 2, 1))
        self.assertEqual((result["n_failed"], result["n_unfinished"]), (1, 1))
        self.assertEqual(result["observation_duration_s"], 2)
        self.assertEqual(result["goodput_rps"], 0)
        self.assertEqual(len([failed, waiting, future]), 3)


if __name__ == "__main__":
    unittest.main()
