#!/usr/bin/env python3
"""Read-only Q1 execution checks. Semantic judgments are kept separate."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from health_analyze_v2 import eos_ids, native_id_mapping, periodic, sha, token_ids


def analyze(run_root, source_file, expected_max_seqs=32):
    root, source_file = Path(run_root), Path(source_file)
    run = root / 'native'
    read = lambda name: json.loads((run / name).read_text())
    source = json.loads(source_file.read_text())
    planned = {r['request_id']: r for r in source['requests']}
    rows, rendered = read('measured-outputs.json'), read('rendered-inputs.json')
    rendered_by_id = {r['request_id']: r for r in rendered}
    issues = []
    def check(ok, reason):
        if not ok:
            issues.append(reason)
    check(len(planned) == len(rows) == len(rendered) == 16 and
          {r['request_id'] for r in rows} == set(planned) == set(rendered_by_id), 'request_inventory')
    parent = json.loads((root / 'launcher-receipt.json').read_text())
    model = json.loads((root / 'model-reverified.json').read_text())
    status, config, prov = read('status.json'), read('config.json'), read('provenance.json')
    check(parent['status'] == status['status'] == 'COMPLETE' and model['status'] == 'VERIFIED', 'lifecycle_or_model')
    check(read('source-input.json') == source and prov['input_sha256'] == sha(source_file), 'source_changed')
    check(prov['cell_sha256'] == sha(run / 'cell-source.py'), 'cell_source_changed')
    check(read('native-drain.json')['status'] == 'QUALIFIED', 'drain')
    reset = read('prefix-cache-reset.json')
    check(reset['reset_succeeded'] is True and reset['cached_hash_keys_after'] == 0 and
          reset['before']['status'] == reset['after']['status'] == 'QUALIFIED', 'cold_apc')
    expected = dict(temperature=0.0, max_tokens=512, min_tokens=0, ignore_eos=False, stop=[], seed=20260905)
    check(all(config['sampling'].get(k) == v for k, v in expected.items()), 'sampling')
    expected_args = dict(enable_prefix_caching=True, async_scheduling=False, max_num_seqs=expected_max_seqs,
        max_num_batched_tokens=1024, max_model_len=4096, kv_cache_memory_bytes=8589934592)
    check(all(config['engine_args'].get(k) == v for k, v in expected_args.items()), 'runtime_scope')
    check(read('tokenizer.json')['answer_cue'] is None and prov['reference_facts_loaded'] is False, 'template_or_reference_leak')
    eos = eos_ids(read('resolved-eos.json'))
    native = read('measured-native-sampling.json')
    mapping, id_errors = native_id_mapping(native, {'measured/' + rid for rid in planned})
    issues.extend(id_errors)
    for rid, params in native.items():
        stops = (params.get('stop_token_ids') or []) + (params.get('all_stop_token_ids') or [])
        check(params.get('stop') in (None, []) and params.get('ignore_eos') is False and
              params.get('min_tokens') == 0 and params.get('max_tokens') == 512 and
              params.get('_eos_token_id') in eos and set(stops).issubset(eos), rid + ':native_sampling')
    per_request = []
    for row in rows:
        rid = row['request_id']; render = rendered_by_id[rid]
        check(all(row[k] == render[k] for k in ('prompt', 'prompt_token_ids')), rid + ':rendered_identity')
        check(row['chat_user_content'] == planned[rid]['chat_user_content'], rid + ':source_identity')
        ids, times = row['output_token_ids'], row['token_times_s']
        check(token_ids(ids, allow_empty=True) and len(ids) <= 512 and len(times) == len(ids) and
              all(isinstance(t, (int, float)) and math.isfinite(t) and t >= 0 for t in times) and
              all(a <= b for a, b in zip(times, times[1:])), rid + ':tokens_or_timing')
        check(row['finished'] is True and row['finish_reason'] in ('stop', 'length'), rid + ':completion')
        check(row['finish_reason'] != 'length' or len(ids) == 512, rid + ':cap')
        natural = row['finish_reason'] == 'stop' and (row['stop_reason'] is None or row['stop_reason'] in eos)
        check(row['finish_reason'] != 'stop' or natural, rid + ':unknown_stop')
        per_request.append(dict(request_id=rid, natural_eos=natural, tokens=len(ids),
            empty=not row['output_text'].strip(), periodic_suffix=periodic(ids), finish_reason=row['finish_reason']))
    counts = dict(planned=16, completed=sum(r['finished'] is True for r in rows),
        natural_eos=sum(r['natural_eos'] for r in per_request), empty=sum(r['empty'] for r in per_request),
        periodic_suffix=sum(r['periodic_suffix'] is not None for r in per_request),
        length_capped=sum(r['finish_reason'] == 'length' for r in per_request),
        output_tokens=sum(r['tokens'] for r in per_request))
    structural_gate = counts['completed'] == 16 and counts['natural_eos'] >= 14 and counts['empty'] == 0 and counts['periodic_suffix'] <= 1
    return dict(validity='COMPLETE' if not issues else 'INVALID', issues=issues, counts=counts,
        structural_gate=structural_gate, semantic_quality='PENDING_SEPARATE_SOURCE_GROUNDED_JUDGMENT',
        per_request=per_request, native_to_external=mapping, expected_max_seqs=expected_max_seqs, analyzer_sha256=sha(__file__),
        source_sha256=sha(source_file), parent=parent,
        scope='Q1 execution integrity and fixed structural screen only; no performance comparison or semantic quality claim.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-max-seqs', type=int, choices=(1, 32), default=32)
    args = p.parse_args()
    result = analyze(args.run_root, args.source, args.expected_max_seqs)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: result[k] for k in ('validity', 'issues', 'counts', 'structural_gate')}))
