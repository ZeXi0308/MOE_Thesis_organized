#!/usr/bin/env python3
"""Compare repeated eager and graph outputs; record differences, do not waive them.

This isolates repeatability from graph capture at the actual large-batch shapes.
No tolerance is used to declare task quality or conversion correctness.
"""
import argparse
import itertools
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--tokens', default='2048,8192')
    p.add_argument('--repeats', type=int, default=4)
    p.add_argument('--seed', type=int, default=20261015)
    a = p.parse_args()
    import torch
    from format_backends import ExpertShape, build_backend
    from bench_expert_formats import make_routing
    from validate_nvfp4_layer import metrics
    meta = torch.load(a.weights, map_location='cpu', weights_only=True)['metadata']
    shape = ExpertShape(**meta['shape'])
    backend = build_backend('awq', shape, weights=a.weights, device='cuda:0')
    report = {'scope': 'same-input repeatability and capture diagnostic; not quality',
              'backend': backend.metadata, 'seed': a.seed, 'cases': [],
              'native_call_options': {'use_atomic_add': False, 'use_fp32_reduce': True}}
    def save():
        Path(a.out).write_text(json.dumps(report, indent=2) + '\n')
    with torch.inference_mode():
        for routing, tokens in itertools.product(('uniform', 'zipf'), map(int, a.tokens.split(','))):
            ids, scores, route = make_routing(tokens, shape, routing, a.seed, 1.2, None)
            ids, scores = ids.cuda(), scores.cuda()
            torch.manual_seed(a.seed + tokens)
            x = torch.randn((tokens, shape.d), dtype=torch.bfloat16, device='cuda:0')
            call = lambda: backend(x, scores, ids)
            for _ in range(5):
                call()
            torch.cuda.synchronize()
            eager = [call().clone() for _ in range(a.repeats)]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = call()
            captured = []
            for _ in range(a.repeats):
                graph.replay()
                captured.append(output.clone())
            torch.cuda.synchronize()
            def compare(left, right):
                value = metrics(left, right)
                value['different_elements'] = int((left != right).sum())
                value['elements'] = left.numel()
                value['both_finite'] = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
                return value
            case = {'tokens': tokens, 'routing': route, 'eager_repeat': [],
                    'graph_repeat': [], 'graph_vs_eager': []}
            for i in range(1, a.repeats):
                case['eager_repeat'].append(compare(eager[i], eager[0]))
                case['graph_repeat'].append(compare(captured[i], captured[0]))
            for i in range(a.repeats):
                case['graph_vs_eager'].append(compare(captured[i], eager[i]))
            report['cases'].append(case)
            save()
            print(json.dumps({'tokens': tokens, 'routing': routing,
                'max_relative_l2': {key: max(v['relative_l2'] for v in case[key])
                    for key in ('eager_repeat', 'graph_repeat', 'graph_vs_eager')}}), flush=True)
            del graph, output, eager, captured
    report['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
