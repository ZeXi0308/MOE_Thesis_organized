"""CPU check of the fair adapter seam in the r02 G64 diagnostic runner."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


HERE = Path(__file__).resolve().parent


class RunnerWiringTest(unittest.TestCase):
    def test_install_after_warmup_drain_and_before_measurement(self):
        spec = importlib.util.spec_from_file_location(
            'candidate_run_ltr_style', HERE / 'run_ltr_style.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)

        scheduler = SimpleNamespace(connector=SimpleNamespace(
            connector_scheduler=SimpleNamespace()))
        installed = []

        def fake_fair_install(actual_scheduler, **kwargs):
            self.assertIs(actual_scheduler, scheduler)
            self.assertEqual(kwargs, dict(vllm_config='pinned-config',
                block_size=16, threshold=30, quantum=10))
            installed.append('fair')
            scheduler.schedule = lambda: None
            scheduler._rotation_begin = lambda *args: None
            scheduler.connector.connector_scheduler._calc_num_offloadable_tokens = lambda *args: 0
            return dict(status='INSTALLED', threshold=30, quantum=10,
                        applied_rotations=0), lambda: installed.append('uninstall')

        data, undo = runner.install_fair_selected(scheduler,
            vllm_config='pinned-config', block_size=16, threshold=30,
            quantum=10, install=fake_fair_install)
        self.assertEqual(installed, ['fair'])
        self.assertEqual(data['ltr_config'], dict(threshold=30, quantum=10))
        self.assertEqual(data['store_scope'], 'selected')
        self.assertTrue(data['native_calc_overridden'])
        undo()
        self.assertEqual(installed, ['fair', 'uninstall'])

        tree = ast.parse((HERE / 'run_ltr_style.py').read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'main')
        calls = [(ast.unparse(n.func), n.lineno) for n in ast.walk(main)
                 if isinstance(n, ast.Call)]
        install_line = next(line for name, line in calls
                            if name == 'install_fair_selected')
        self.assertTrue(any(name == 'drain_offload' and line < install_line
                            for name, line in calls))
        self.assertTrue(any(name == 'engine.reset_prefix_cache' and line < install_line
                            for name, line in calls))
        self.assertTrue(any(name == 'set_empty_admission_cap' and line < install_line
                            for name, line in calls))
        self.assertTrue(any(name == 'capture_with_memory' and line > install_line
                            for name, line in calls))
        self.assertIn('from ltr_fair_native import install as install_ltr',
                      (HERE / 'run_ltr_style.py').read_text())


if __name__ == '__main__':
    unittest.main()
