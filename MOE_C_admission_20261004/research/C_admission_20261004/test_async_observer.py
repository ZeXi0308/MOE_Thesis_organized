"""Specific async observation risks: policy-neutral splice and progress labels."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from async_observer import patch_source, progress_class


class AsyncObservationTests(unittest.TestCase):
    def test_splice_adds_only_two_passive_calls_to_native_function(self):
        tree = ast.parse((Path(__file__).parent/'vendor/scheduler.py').read_text())
        scheduler = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
        function = next(n for n in scheduler.body if isinstance(n, ast.FunctionDef) and n.name == 'schedule')
        # Use exact whitespace/source as install does, rather than ast.unparse.
        import textwrap
        lines = (Path(__file__).parent/'vendor/scheduler.py').read_text().splitlines(True)
        source = textwrap.dedent(''.join(lines[function.lineno-1:function.end_lineno]))
        patched = patch_source(source)
        compile(patched, '<passive_async_test>', 'exec')

        class RemoveObserver(ast.NodeTransformer):
            removed = 0

            def visit_Assign(self, node):
                if any(isinstance(n, ast.Name) and n.id == '_c_async_row' for n in node.targets):
                    self.removed += 1
                    return None
                return self.generic_visit(node)

            def visit_Expr(self, node):
                if isinstance(node.value, ast.Call) and '_c_async_observer' in ast.unparse(node.value):
                    self.removed += 1
                    return None
                return self.generic_visit(node)

        strip = RemoveObserver()
        restored = strip.visit(ast.parse(patched))
        self.assertEqual(strip.removed, 2)
        self.assertEqual(ast.dump(restored), ast.dump(ast.parse(source)))

    def test_inflight_is_not_automatically_nonexecuting_recovery(self):
        req = NS(status=NS(name='RUNNING'), num_computed_tokens=120,
                 num_prompt_tokens=100, max_tokens=100, num_output_placeholders=1,
                 next_decode_eligible_step=3, num_tokens_with_spec=120)
        self.assertEqual(progress_class(req, 1, 3), 'scheduled_decode')
        req.num_computed_tokens = 199
        self.assertEqual(progress_class(req, 0, 3), 'terminal_inflight')
        req.num_computed_tokens = 120
        self.assertEqual(progress_class(req, 0, 2), 'decode_ineligible')
        req.status.name = 'WAITING_FOR_REMOTE_KVS'
        self.assertEqual(progress_class(req, 0, 3), 'WAITING_FOR_REMOTE_KVS')


if __name__ == '__main__':
    unittest.main()
