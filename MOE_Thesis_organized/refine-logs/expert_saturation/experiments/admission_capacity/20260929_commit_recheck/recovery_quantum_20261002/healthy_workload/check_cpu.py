#!/usr/bin/env python3
"""Focused CPU checks: references, parser, cap/unfinished scoring, A schema.

All generated raw rows below are artificial fixtures, never model results.
No transformers, tokenizer, network or GPU is imported.
"""
import ast
from fractions import Fraction
import json
from pathlib import Path
import tempfile

from evaluate_quality import strict_answer, normalize_number, evaluate

ROOT = Path(__file__).resolve().parent


def arithmetic(node):
    if isinstance(node, ast.Expression):
        return arithmetic(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Fraction(str(node.value))
    if isinstance(node, ast.BinOp):
        a, b = arithmetic(node.left), arithmetic(node.right)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div):
            return a / b
        if isinstance(node.op, ast.FloorDiv):
            return Fraction(a // b)
    raise ValueError('Only literal arithmetic is accepted in reference checks')


def main():
    tasks = json.loads((ROOT/'tasks_text.json').read_text())['requests']
    refs = json.loads((ROOT/'reference_answers.json').read_text())['references']
    assert len(tasks) == len(refs) == 32
    assert [r['request_id'] for r in refs] == [r['request_id'] for r in tasks]
    for row in refs:
        assert arithmetic(ast.parse(row['independent_check_expression'], mode='eval')) == Fraction(row['gold'])
    parser_cases = {
        'So the answer is 35.': '35',
        'The answer is $1,430.00.': '1430',
        'Final answer: -5': '-5',
        '#### 1/2': '1/2',
        r'Thus \boxed{0.5}': '1/2',
        'We subtract 7 from 42 and get 35.': None,
        'The answer is 3. Final answer: 4.': '4',
        'The answer is 1/0': None,
        'Final answer: 12e3': None,
    }
    for text, expected in parser_cases.items():
        assert strict_answer(text) == expected, (text, strict_answer(text), expected)
    assert normalize_number('25,000') == '25000'
    config = dict(model={'id':'TEST_FIXTURE'}, workload_sha256='FIXTURE',
                  output_tokens=4, output_tokens_by_request={}, ignore_eos=False, min_tokens=0)
    rows = [dict(request_id=f'fixture{i}', document_id=f'fixture{i}',
                 prompt_token_ids_sha256='fixture') for i in range(4)]
    work = dict(source_requests=rows, arrival_traces_s={'steady':[0.0]*4})
    reference = {r['request_id']:{'gold':'35'} for r in rows}
    raw_rows, events = [], []
    for i, (count, reason, status) in enumerate([(1,'stop','completed'),(4,'length','completed'),(1,None,'unfinished')]):
        raw_rows.append(dict(request_id=f'fixture{i}', output_token_ids=[i+1]*count,
                             token_times_s=[.1*(j+1) for j in range(count)], arrival_s=0.0,
                             prompt_token_ids_sha256='fixture', max_output_tokens=4,
                             status=status, stop_reason=reason, completion_s=.5 if status=='completed' else None))
        for j in range(count):
            events.append(dict(request_id=f'fixture{i}', chunk_size=1, received_s=.1*(j+1),
                               finished=j==count-1 and status=='completed', finish_reason=reason))
    class FixtureTokenizer:
        def decode(self, ids, skip_special_tokens=True):
            return 'So the answer is 35.' if ids else ''
    with tempfile.TemporaryDirectory() as temp:
        d = Path(temp)
        (d/'config.json').write_text(json.dumps(config))
        (d/'raw.json').write_text(json.dumps(dict(status='INCOMPLETE',requests=raw_rows,
                                                  output_events=events,observation_end_s=1.0)))
        result = evaluate(d, work, config, reference, FixtureTokenizer())
    assert result['strict_accuracy'] == .5
    assert result['correct_and_natural_stop'] == 1
    assert result['cap_truncated_count'] == 1
    assert result['completed'] == 2 and result['requests'] == 4
    assert result['missing_requests'] == ['fixture3']
    assert result['actual_output_tokens_per_s'] == 6
    for name in ('prepare_inputs.py','evaluate_quality.py','check_cpu.py'):
        compile((ROOT/name).read_text(),name,'exec')
    print(json.dumps(dict(status='CPU_PASS',reference_arithmetic=32,parser_cases=len(parser_cases),
                          artificial_capture_checks=7,tokenization='NOT_RUN_BY_THIS_CHECK',native_runs=0)))


if __name__ == '__main__':
    main()
