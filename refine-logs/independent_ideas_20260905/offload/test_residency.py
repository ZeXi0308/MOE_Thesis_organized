import unittest
from inspect_residency import capacity


class CapacityTest(unittest.TestCase):
    def setUp(self):
        self.config = dict(max_position_embeddings=8, num_attention_heads=2,
                           num_key_value_heads=1, num_hidden_layers=2, hidden_size=8)
        self.weights = dict(weight_bytes=128, nonexpert_bytes=32, expert_objects=4, bytes_per_expert=24)

    def test_kv_is_two_tensors_each_layer_and_each_live_request(self):
        result = capacity(self.config, self.weights, 3, 2, 1024)
        self.assertEqual(result["dense_bf16_kv_bytes"], 192)
        self.assertEqual(result["weights_plus_kv_lower_bound_bytes"], 320)
        self.assertEqual(result["optimistic_expert_slots"], 4)
        self.assertEqual(result["status"], "FIT_NOT_PROVEN")
        self.assertIsNone(result["fetch_stall_fraction"])

    def test_reserve_is_not_mislabeled_as_measured_necessary_pressure(self):
        result = capacity(self.config, self.weights, 1, 1, 256, 100)
        self.assertFalse(result["necessary_pressure"])
        self.assertEqual(result["optimistic_expert_slots"], 3)
        self.assertTrue(capacity(self.config, self.weights, 3, 2, 300)["necessary_pressure"])

    def test_out_of_context_or_invalid_memory_is_not_a_pressure_regime(self):
        for context, memory in ((9, 1024), (0, 1024), (2, float("nan"))):
            with self.assertRaises(ValueError):
                capacity(self.config, self.weights, 1, context, memory)


if __name__ == "__main__":
    unittest.main()
