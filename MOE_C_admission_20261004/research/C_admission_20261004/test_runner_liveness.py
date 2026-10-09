"""CPU regression for result lifetimes in the actual runner loops."""
import ast
from pathlib import Path
import statistics
import unittest
import weakref


class TrackedResult(dict):
    __slots__ = ("__weakref__",)


class RunnerLivenessTest(unittest.TestCase):
    def test_warmup_development_and_probe_results_are_released(self):
        path = Path(__file__).with_name("run.py")
        tree = ast.parse(path.read_text(), filename=str(path))
        main = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        cell = next(node for node in ast.walk(main)
                    if isinstance(node, ast.FunctionDef) and node.name == "cell")

        def actual_loop(scope, iterator):
            matches = [node for node in ast.walk(scope)
                       if isinstance(node, ast.For) and ast.unparse(node.iter) == iterator]
            self.assertEqual(len(matches), 1, iterator)
            # Execute production statements, including their cleanup placement.
            return compile(ast.Module(body=matches, type_ignores=[]), str(path), "exec")

        cases = (
            ("warmup", actual_loop(cell, "warm_specs")),
            ("development", actual_loop(main, "protocol['dev_caps']")),
            ("pressure_probe", actual_loop(main, "protocol['pressure_probes']")),
        )
        for kind, code in cases:
            with self.subTest(loop=kind):
                references = []
                calls = []

                def result(*args, **kwargs):
                    self.assertTrue(all(ref() is None for ref in references),
                        "Previous result is still alive at the next measurement/cell call")
                    value = TrackedResult(status="COMPLETE", actual_preemption_count=0,
                        requests=[dict(arrival_s=0., completion_s=1.)],
                        output_events=[dict(new_token_ids=[1])])
                    references.append(weakref.ref(value))
                    calls.append(1)
                    return value

                def read(filename):
                    if filename.name == "config.json":
                        return {}
                    return dict(source_requests=[dict(request_id="warm")],
                                actual_prompt_token_ids=[[1, 2]])

                namespace = dict(
                    ROOT=path.parent, path=path.parent, label="test", engine=object(),
                    warm_specs=[("short", 2, 2), ("short", 3, 3), ("long", 1, 1)],
                    protocol=dict(dev_caps=[128, 192, 256], pressure_probes=[
                        dict(label="low", count=64, gap=.5),
                        dict(label="near", count=192, gap=.1),
                        dict(label="high", count=192, gap=0.)]),
                    engine_max=256, async_enabled=False, dev={}, test={}, scores=[], probes=[],
                    statistics=statistics, cell=result, measure_episode=result,
                    read=read, subset=lambda workload, *args: workload,
                    set_empty_admission_cap=lambda *args: None,
                    dump=lambda *args: None)
                exec(code, namespace)
                self.assertEqual(len(calls), 3)
                self.assertTrue(all(ref() is None for ref in references),
                    "Last result is still alive after the loop")
                self.assertNotIn("wr" if kind == "warmup" else "raw", namespace)
                if kind == "development":
                    self.assertEqual(namespace["scores"], [
                        dict(cap=cap, mean_flow_s=1., preemptions=0)
                        for cap in (128, 192, 256)])
                elif kind == "pressure_probe":
                    self.assertEqual(namespace["probes"], [
                        dict(probe, actual_preemptions=0)
                        for probe in namespace["protocol"]["pressure_probes"]])


if __name__ == "__main__":
    unittest.main()
