#!/usr/bin/env python3
"""Prepare blind source-grounded review; reuse only byte-identical prior answers."""
import argparse
import hashlib
import json
from pathlib import Path
import random

BASE = Path(__file__).resolve().parent
def read(p): return json.loads(Path(p).read_text())
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def textsha(s): return hashlib.sha256(s.encode()).hexdigest()
def save(p, value):
    with p.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')


def prepare(root, output):
    rows = read(root / 'native/measured-outputs.json')
    source = read(BASE / 'docqa_inputs128_v1.json')['requests']
    answers = read(BASE / 'docqa_answers128_v1.json')['requests']
    prior = {x['request_id']: x for x in read(BASE / 'docqa16_quality_v1.json')['per_request']}
    old_bundle = {x['blind_id']: x for x in read(BASE / 'qa_review_batch_d_bundle_v1.json')}
    rows_by_id = {r['request_id']: r for r in rows}
    assert len(rows) == len(rows_by_id) == len(source) == len(answers) == 128
    assert set(rows_by_id) == {r['request_id'] for r in source}
    cases = []
    for s, key in zip(source, answers):
        rid = s['request_id']; r = rows_by_id[rid]
        assert key['request_id'] == rid and key['input_index'] == s['input_index']
        assert textsha(r['article_text']) == key['article_text_sha256'] == s['article_text_sha256']
        assert r['article_text'] == s['article_text'] and r['questions'] == s['questions']
        assert [{k: q[k] for k in ('question_id', 'question')} for q in key['questions']] == s['questions']
        cases.append((s, r, key))
    random.Random(202610021).shuffle(cases)
    mapping, fresh, reused = [], [], []
    for index, (s, r, key) in enumerate(cases):
        bid = f'E{index + 1:03d}'
        mapping.append(dict(blind_id=bid, request_id=s['request_id'], input_index=s['input_index']))
        item = dict(blind_id=bid, article_text=s['article_text'], questions=key['questions'], response=r['output_text'])
        old = prior.get(s['request_id'])
        same = False
        if old:
            b = old_bundle[old['blind_id']]
            same = all(item[k] == b[k] for k in ('article_text', 'questions', 'response'))
        if same:
            assert old['article_sha256'] == textsha(item['article_text'])
            assert old['response_sha256'] == textsha(item['response'])
            reused.append(dict(blind_id=bid, prior_blind_id=old['blind_id'], judgment=old,
                               reason='Exact article, all question/key contents, and response bytes unchanged.'))
        else:
            fresh.append(item)
    output.mkdir(exist_ok=False)
    save(output / 'identity-map.json', mapping)
    save(output / 'exact-reused-judgments.json', reused)
    groups = [fresh[i::3] for i in range(3)]
    for i, group in enumerate(groups):
        save(output / f'blind-source-batch-{i + 1}.json', group)
    receipt = dict(schema='c-docqa128-blind-preparation-v1', total=128,
        exact_reused=len(reused), fresh=len(fresh), fresh_batch_sizes=[len(x) for x in groups],
        raw_outputs_sha256=digest(root / 'native/measured-outputs.json'),
        answers_sha256=digest(BASE / 'docqa_answers128_v1.json'),
        prior_quality_sha256=digest(BASE / 'docqa16_quality_v1.json'),
        identity_map_sha256=digest(output / 'identity-map.json'))
    save(output / 'preparation-receipt.json', receipt)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(prepare(args.run_root, args.output)))
