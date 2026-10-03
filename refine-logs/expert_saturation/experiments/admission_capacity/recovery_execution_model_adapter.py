"""Compose the execution obligation with the existing resource simulator.

No second resource/victim model is maintained. The supplied source is compiled
with a default-off resume-commit hook; its state transitions and selector remain
owned by the resource-model session. Native recompute scope only.
"""
import ast
import hashlib
from pathlib import Path


def load_simulator(path, *, funding_filter=False):
    source = Path(path).read_text()
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'simulate')
    fn.args.kwonlyargs += [ast.arg(arg='protect_native_recovery'), ast.arg(arg='natural_protection_events')]
    fn.args.kw_defaults += [ast.Constant(False), ast.Constant(None)]
    matches = [0, 0, 0]
    for node in ast.walk(fn):
        if isinstance(node, ast.While) and ast.unparse(node.test) == 'waiting and budget > 0':
            for i, item in enumerate(node.body):
                if ast.unparse(item) == 'waiting.pop(0)':
                    node.body[i:i] = ast.parse("""
if protect_native_recovery and protected is None and r['status'] == 'PREEMPTED' and r['output'] > 0 and total(r) - r['computed'] > 1:
    protected, output_at_start = rid, r['output']
    if natural_protection_events is not None:
        natural_protection_events.append(dict(step=step, target=rid, pending=total(r)-r['computed'], scheduled_tokens=tokens))
""").body
                    matches[0] += 1
                    break
        if isinstance(node, ast.If) and ast.unparse(node.test) == "protected is not None and states[protected]['output'] > output_at_start":
            node.test = ast.parse("protected is not None and (protected not in states or states[protected]['output'] > output_at_start)", mode='eval').body
            matches[1] += 1
        if isinstance(node, ast.Call) and ast.unparse(node.func) == 'tracker.decide':
            if funding_filter:
                if node.keywords:
                    raise ValueError('existing selector adapter must be reconciled')
                node.keywords.append(ast.keyword(arg='released_blocks', value=ast.parse(
                    "{r:states[r]['allocated'] for r in running}", mode='eval').body))
            matches[2] += 1
    if matches != [1, 1, 1]:
        raise ValueError(f'resource model integration points changed: {matches}')
    namespace = {}
    adapted = ast.fix_missing_locations(tree)
    exec(compile(adapted, str(path), 'exec'), namespace)
    original = namespace['simulate']

    def simulate(*args, **kwargs):
        action = kwargs.get('action', args[2] if len(args) > 2 else None)
        if kwargs.get('protect_native_recovery') and (action not in
                ('native', 'least', 'most', 'continuous_most') or any(kwargs.get(k)
                for k in ('saved_prefixes', 'observed_load_completions', 'staged_recovery_targets'))):
            raise ValueError('execution obligation currently qualified for native recompute only')
        return original(*args, **kwargs)
    return simulate, dict(source=str(path), source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                          adapted_ast_sha256=hashlib.sha256(ast.dump(adapted).encode()).hexdigest(),
                          funding_filter=funding_filter, default_protection=False)
