"""Same-current-input numerical qualification; logical output advances the model."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import runpy
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
WIDTHS = (1, 16, 32, 64, 128, 160)
EXPECTED = {'fused_moe': 'a8015d90908883d3dc459e7a508d2de56bfbdff678c5f36e23d0410ef5a04683',
            'moe_align_block_size': 'c3f7fc2087836f0160a32ab99ceb1d8f6e87c793679da34dd672e225cb7fa31b'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_record(record, kwargs):
    rows, mapping = record['row_topk_experts'], record['shared_plan']['expert_map_device']
    if len(rows) != kwargs['hidden_states'].shape[0] or not rows:
        raise ValueError('CPU route rows do not match actual input')
    if any(len(row) != 8 or len(set(row)) != 8 or any(type(e) is not int or not 0 <= e < 64 for e in row) for row in rows):
        raise ValueError('invalid logical top-k IDs')
    active = sorted({e for row in rows for e in row})
    if active != record['active_experts'] or len(mapping) != 384:
        raise ValueError('active union or plan map mismatch')
    slots = [mapping[e] for e in active]
    if len(set(slots)) != len(slots) or any(type(s) is not int or not 0 <= s < 384 for s in slots):
        raise ValueError('active physical map is not a valid injection')
    if kwargs['expert_map'].shape != (384,) or kwargs['w1'].shape[0] != 384 or kwargs['w2'].shape[0] != 384:
        raise ValueError('unexpected physical tensor dimensions')
    return active, mapping


def rotate_map(active, mapping):
    if len(active) < 2:
        raise ValueError('first-call negative has fewer than two active experts')
    result = list(mapping[:64])
    for expert, other in zip(active, active[1:] + active[:1]):
        result[expert] = mapping[other]
    return result


def metrics(torch, actual, reference):
    finite = bool(torch.isfinite(actual).all() & torch.isfinite(reference).all())
    delta = actual.float() - reference.float()
    den = float(torch.linalg.vector_norm(reference.float())) if finite else None
    norm = float(torch.linalg.vector_norm(delta)) if finite else None
    return dict(finite=finite, allclose=bool(torch.allclose(actual, reference, atol=.01, rtol=.01)),
                bit_equal=bool(torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))),
                maxabs=float(delta.abs().max()) if finite else None, reference_l2=den,
                relative_l2=norm / den if den else (0.0 if norm == 0 else None), atol=.01, rtol=.01)


def invoke(module, kernel, kwargs, logical, case, role):
    """Both roles use the real helper; only logical changes its invalid-ID option."""
    original = module._prepare_expert_assignment
    events = []
    case.setdefault('invocations', []).append(dict(role=role, helper_calls=events))
    def assignment(ids, config, num_tokens, top_k_num, global_num_experts, expert_map, **options):
        event = dict(global_num_experts=global_num_experts, map_shape=list(expert_map.shape),
                     rows=num_tokens, top_k=top_k_num, config=dict(config),
                     incoming_ignore_invalid_experts=options.get('ignore_invalid_experts'),
                     effective_ignore_invalid_experts=not logical, status='started')
        events.append(event)
        if (global_num_experts != (64 if logical else 384) or expert_map.shape != (global_num_experts,)
                or options.get('ignore_invalid_experts') is not True):
            raise RuntimeError('actual assignment interface differs from frozen contract')
        options['ignore_invalid_experts'] = not logical
        try:
            result = original(ids, config, num_tokens, top_k_num, global_num_experts, expert_map, **options)
            event['status'] = 'complete'
            return result
        except BaseException:
            event['status'] = 'failed'; raise
    module._prepare_expert_assignment = assignment
    try:
        result = kernel(**kwargs)
        if len(events) != 1:
            raise RuntimeError(f'expected one actual assignment helper call, observed {len(events)}')
        return result
    finally:
        module._prepare_expert_assignment = original
        case['assignment_restored'] = module._prepare_expert_assignment is original


class Qualification:
    def __init__(self, runtime, module, report):
        self.runtime, self.module, self.report = runtime, module, report
        self.original, self.covered = runtime.kernel, {}
        runtime.kernel = self.kernel

    def compare(self, kwargs, case):
        case['start_perf_ns'] = time.perf_counter_ns()
        try:
            reference = invoke(self.module, self.original, kwargs, False, case, 'physical_reference')
            logical = dict(kwargs, global_num_experts=64, expert_map=kwargs['expert_map'][:64])
            actual = invoke(self.module, self.original, logical, True, case, 'logical')
            case.update(metrics(self.runtime.torch, actual, reference))
            if not case['finite'] or not case['allclose']:
                raise RuntimeError('logical alignment numerical comparison failed')
            case['status'] = 'complete'
            return actual, reference
        except BaseException:
            case['status'] = 'failed'; raise
        finally:
            case['end_perf_ns'] = time.perf_counter_ns()

    def kernel(self, **kwargs):
        runtime, report = self.runtime, self.report
        if not runtime.measurement or runtime.validation_enabled or kwargs.get('global_num_experts') != 384:
            return self.original(**kwargs)
        record = runtime.records[-1]
        case = {k: record[k] for k in ('call_id', 'context', 'rows')}
        case.update(layer=record['layer_name'], status='started')
        report['actual_calls'].append(case)
        active, mapping = validate_record(record, kwargs)
        if runtime.shared_mode != 'oneshot' or record['context'].get('phase') != 'measurement':
            raise RuntimeError('qualification requires the declared ordinary measurement path')
        if any(kwargs[k] is not pool for k, pool in zip(('w1', 'w2'), runtime.shared_weights)):
            raise RuntimeError('kernel weights differ from the actual shared pool')
        case.update(cpu_map_valid=True, active_experts=active, active_physical_slots=[mapping[e] for e in active],
                    actual_shared_weight_identity=True, weight_shapes=[list(kwargs[k].shape) for k in ('w1', 'w2')])
        actual, reference = self.compare(kwargs, case)
        if len(report['actual_calls']) == 1:
            negative = dict(call_id=case['call_id'], layer=case['layer'], context=case['context'], rows=case['rows'],
                            active_experts=active, rotated_map=rotate_map(active, mapping), status='started')
            report['negative_control'] = negative
            bad = dict(kwargs, global_num_experts=64, expert_map=runtime.torch.tensor(
                negative['rotated_map'], dtype=runtime.torch.int32, device=kwargs['hidden_states'].device))
            output = invoke(self.module, self.original, bad, True, negative, 'negative_logical')
            negative.update(metrics(runtime.torch, output, reference))
            negative['status'] = 'complete' if negative['finite'] and not negative['allclose'] else 'failed'
            if negative['status'] != 'complete':
                raise RuntimeError('fixed first-call wrong-map negative did not fail allclose')
        if case['rows'] >= 160 and case['layer'] not in self.covered:
            self.covered[case['layer']] = []
            for width in WIDTHS:
                prefix = dict(call_id=case['call_id'], layer=case['layer'], context=case['context'], rows=width)
                report['prefix_cases'].append(prefix)
                if width == case['rows']:
                    prefix.update({k: case[k] for k in ('finite', 'allclose', 'bit_equal', 'maxabs', 'reference_l2', 'relative_l2', 'atol', 'rtol', 'status')})
                    prefix['reused_actual_comparison'] = True
                else:
                    inputs = dict(kwargs)
                    for key in ('hidden_states', 'topk_weights', 'topk_ids'):
                        inputs[key] = kwargs[key][:width].contiguous()
                    self.compare(inputs, prefix)
                self.covered[case['layer']].append(width)
        case['returned_path'] = 'logical64_false'
        return actual


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    out = args.output.resolve(); target = out / 'logical_alignment.json'
    if out.exists(): raise FileExistsError(out)
    report = dict(schema='logical_alignment_qualification_v1', status='NOT_QUALIFIED', actual_calls=[], prefix_cases=[],
                  negative_control=None, source_sha256={}, installed_sources={}, hooks_restored=False,
                  input_preservation='Frozen fused_experts_impl allocates separate output at line1704 and moe_sum writes it at1790; BF16 inputs/weights are read-only. No extra input clone. Logical output retained while auxiliary cases run.',
                  scope='Same-state numerical differential only. All reference, logical, prefix and negative costs retained; no task GT, quality or performance claim.')
    saved, hook, pager, original_install, failure = sys.argv, None, None, None, None
    saved_path = list(sys.path)
    report['start_perf_ns'] = time.perf_counter_ns()
    try:
        if '--pool-qualification' in saved or '--verify-kernel' in saved:
            raise ValueError('old bitwise qualification flags must be absent')
        for name, expected in json.loads((ROOT / 'sources.json').read_text()).items():
            actual = digest(ROOT / 'source' / name); report['source_sha256']['source/' + name] = actual
            if actual != expected: raise RuntimeError('frozen source hash mismatch: ' + name)
        report['source_sha256']['instrumentation/run_logical_alignment.py'] = digest(__file__)
        sys.path.insert(0, str(ROOT / 'source'))
        pager = importlib.import_module('wisp_v026_adapter'); original_install = pager.install
        def install(*a, **kw):
            nonlocal hook
            runtime = original_install(*a, **kw)
            if hook is None:
                for name, expected in EXPECTED.items():
                    module = importlib.import_module('vllm.model_executor.layers.fused_moe.' + name)
                    actual = digest(module.__file__)
                    report['installed_sources'][name] = dict(path=module.__file__, sha256=actual)
                    if actual != expected: raise RuntimeError('installed source hash mismatch: ' + name)
                hook = Qualification(runtime, importlib.import_module('vllm.model_executor.layers.fused_moe.fused_moe'), report)
            return runtime
        pager.install = install
        entry = str(ROOT / 'source/run_shared_pool_pager.py'); sys.argv = [entry, *saved[1:]]
        runpy.run_path(entry, run_name='__main__')
    except BaseException as exc:
        failure = exc; report['error'] = traceback.format_exc()
    finally:
        if hook is not None: hook.runtime.kernel = hook.original
        if pager is not None and original_install is not None: pager.install = original_install
        sys.argv = saved
        sys.path[:] = saved_path
        report['hooks_restored'] = (hook is None or hook.runtime.kernel is hook.original) and (pager is None or pager.install is original_install)
        report['coverage'] = hook.covered if hook else {}
        report['missing_coverage'] = {f'model.layers.{i}.mlp.experts': sorted(set(WIDTHS) - set(report['coverage'].get(f'model.layers.{i}.mlp.experts', []))) for i in range(16)}
        report['missing_coverage'] = {k: v for k, v in report['missing_coverage'].items() if v}
        if not failure and not report['missing_coverage'] and report['negative_control'] and report['hooks_restored']:
            report['status'] = 'QUALIFIED'
        report['end_perf_ns'] = time.perf_counter_ns()
        out.mkdir(parents=True, exist_ok=True)
        with target.open('x') as f: json.dump(report, f, indent=2, allow_nan=False); f.write('\n')
    if failure: raise failure
    if report['status'] != 'QUALIFIED': raise RuntimeError('logical alignment qualification has missing coverage')


if __name__ == '__main__':
    main()
