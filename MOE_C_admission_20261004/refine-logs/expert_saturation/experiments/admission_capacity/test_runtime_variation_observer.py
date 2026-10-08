import unittest
from unittest.mock import patch

import runtime_variation_observer as module


class RuntimeVariationObserverTest(unittest.TestCase):
    def test_gc_pairing_cumulative_state_and_own_callback_cleanup(self):
        other = lambda phase, info: None
        with patch.object(module.gc, "callbacks", [other]):
            observer = module.RuntimeVariationObserver()
            with patch.object(module.time, "perf_counter_ns", side_effect=[100, 400, 500, 900]):
                observer._on_gc("start", dict(generation=1))
                self.assertEqual(len(observer._active), 1)
                observer._on_gc("stop", dict(generation=1, collected=3, uncollectable=1))
                observer._on_gc("start", dict(generation=1))
                observer._on_gc("stop", dict(generation=1, collected=2, uncollectable=0))
            snapshot = observer.snapshot()
            self.assertEqual(snapshot["gc"]["totals_by_generation"][1],
                dict(starts=2, completed=2, duration_ns=700, collected=5, uncollectable=1, unmatched_stops=0))
            self.assertEqual(snapshot["gc"]["active"], [])
            self.assertEqual([e["duration_ns"] for e in observer.events], [300, 400])
            self.assertTrue(all(e["thread_id"] == module.threading.get_ident() for e in observer.events))
            retained = observer.close()
            self.assertIs(retained, observer.events)
            observer.close()
            self.assertEqual(module.gc.callbacks, [other])
            self.assertEqual(len(retained), 2)

    def test_unavailable_fields_are_unknown_and_migration_remains_visible(self):
        with patch.object(module.gc, "callbacks", []):
            observer = module.RuntimeVariationObserver()
            observer._sched_getcpu = None
            with patch.object(module.Path, "read_text", side_effect=FileNotFoundError):
                snapshot = observer.snapshot()
            self.assertEqual(snapshot["cpu"]["status"], "unknown")
            self.assertIsNone(snapshot["cpu"]["scaling_cur_freq_khz"])
            self.assertIsNone(snapshot["cpu"]["core_before"])
            self.assertEqual(snapshot["rss"]["status"], "unknown")
            self.assertIsNone(snapshot["rss"]["bytes"])
            self.assertLessEqual(snapshot["start_perf_ns"], snapshot["end_perf_ns"])
            with patch.object(observer, "_core", side_effect=[4, 5]), \
                    patch.object(module.Path, "read_text", return_value="1900000\n"):
                cpu = observer._cpu_sample()
            self.assertEqual((cpu["core_before"], cpu["core_after"], cpu["migrated"]), (4, 5, True))
            self.assertEqual((cpu["status"], cpu["scaling_cur_freq_khz"]), ("migrated", 1900000))
            observer.close()


if __name__ == "__main__":
    unittest.main()
