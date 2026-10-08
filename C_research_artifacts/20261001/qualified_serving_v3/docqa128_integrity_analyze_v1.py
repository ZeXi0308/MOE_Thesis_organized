#!/usr/bin/env python3
"""Read-only fixed QA128 execution integrity; no semantic or performance verdict."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from health_analyze_v2 import eos_ids, native_id_mapping, periodic, sha, token_ids

BASE = Path(__file__).resolve().parent
INPUTS = BASE / 'docqa_inputs128_v1.json'
RUN_NAME = 'qwen7b-native-docqa128-v1'
MODEL = 'Qwen/Qwen2.5-7B-Instruct'
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
Q0_SHA = '8afa5257e57e2a7e667e0209831626c65bd039c010b9e60a45933583ea6c54ab'
SAMPLING = dict(temperature=0.0, max_tokens=512, min_tokens=0,
                ignore_eos=False, stop=[], seed=20260905)
ENGINE = dict(dtype='bfloat16', seed=20260905, max_model_len=8192,
    max_num_seqs=128, max_num_batched_tokens=2048, kv_cache_memory_bytes=8589934592,
    gpu_memory_utilization=0.90, enable_chunked_prefill=True, enable_prefix_caching=True,
    scheduling_policy='fcfs', async_scheduling=False, scheduler_reserve_full_isl=True,
    long_prefill_token_threshold=0, stream_interval=1, enforce_eager=False,
    enable_return_routed_experts=False, trust_remote_code=False,
    tensor_parallel_size=1, pipeline_parallel_size=1)
THRESHOLDS = dict(planned=128, completed=128, empty_max=0,
                  natural_eos_min=112, periodic_suffix_max=8)


def digest_text(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def fixed_requests(source):
    """Validate the fixed source before consuming any run artifacts."""
    rows = source.get('requests')
    if not (source.get('schema') == 'c-docqa-inputs-v1' and
            source.get('task') == 'document_qa' and source.get('requests_planned') == 128 and
            source.get('max_model_len') == 8192 and isinstance(rows, list) and len(rows) == 128 and
            all(isinstance(r, dict) and isinstance(r.get('request_id'), str) and r['request_id'] for r in rows) and
            len({r['request_id'] for r in rows}) == 128 and
            [r.get('input_index') for r in rows] == list(range(128)) and
            all(type(r.get('input_index')) is int for r in rows) and
            source.get('arrival_traces_s') == [0.0] * 128 and
            isinstance(source.get('instruction'), str) and source['instruction'].strip() and
            source.get('sampling') == SAMPLING):
        raise ValueError('fixed QA128 source inventory/schema/configuration differs')
    for row in rows:
        questions = row.get('questions')
        if not (row.get('task') == 'document_qa' and row.get('max_output_tokens') == 512 and
                isinstance(questions, list) and len(questions) == 3 and
                all(isinstance(q, dict) and set(q) == {'question_id', 'question'} and
                    isinstance(q['question'], str) and q['question'].strip() for q in questions) and
                [q['question_id'] for q in questions] == ['Q1', 'Q2', 'Q3']):
            raise ValueError(row['request_id'] + ': source questions/task/cap differ')
        for name in ('article_text', 'chat_user_content'):
            if not (isinstance(row.get(name), str) and row[name].strip() and
                    digest_text(row[name]) == row.get(name + '_sha256')):
                raise ValueError(row['request_id'] + ': source content/hash differs')
        chat = source['instruction'] + '\n\nArticle:\n' + row['article_text'] + '\n\nQuestions:\n'
        chat += '\n'.join(f"{i + 1}. {q['question']}" for i, q in enumerate(questions))
        if row['chat_user_content'] != chat:
            raise ValueError(row['request_id'] + ': generation chat contains unexpected material')
    return rows


def analyze(run_root, source_file=INPUTS, expected_max_seqs=128):
    root, source_file = Path(run_root), Path(source_file)
    if source_file.resolve() != INPUTS.resolve() or expected_max_seqs != 128:
        raise ValueError('fixed QA128 source path or concurrency differs')
    source = json.loads(source_file.read_text())
    requests = fixed_requests(source)
    planned = {r['request_id']: r for r in requests}
    run, issues, hashes = root / 'native', [], {}

    def check(ok, reason):
        if not ok:
            issues.append(reason)

    def read(path, kind=dict):
        path = Path(path)
        try:
            value = json.loads(path.read_text())
            hashes[str(path)] = sha(path)
        except (OSError, ValueError, UnicodeError):
            issues.append('missing_or_invalid_json:' + str(path))
            return kind()
        if not isinstance(value, kind):
            issues.append('wrong_json_type:' + str(path))
            return kind()
        return value

    def file_sha(path):
        try:
            return sha(path)
        except OSError:
            issues.append('missing_file:' + str(path))
            return None

    def matching(actual, expected):
        return isinstance(actual, dict) and all(
            k in actual and actual[k] == v and type(actual[k]) is type(v)
            for k, v in expected.items())

    check(root.name == RUN_NAME, 'run_root_name')
    parent = read(root / 'launcher-receipt.json')
    model = read(root / 'model-reverified.json')
    manifest = read(BASE / 'qwen7b_model_manifest.json')
    freeze = read(BASE / 'native_docqa128_freeze_v1.json')
    protocol = read(BASE / 'native_docqa128_protocol_v1.json')
    source_sha = sha(source_file)
    expected_parent = dict(status='COMPLETE', action='native-docqa128', model_id=MODEL,
        revision=REVISION, inputs_sha256=source_sha, max_model_len=8192,
        generation_timeout_s=900, child_timeout_s=1200,
        answer_key_sha256=source.get('questions_answer_key_sha256'),
        manifest_sha256=file_sha(BASE / 'qwen7b_model_manifest.json'),
        freeze_sha256=file_sha(BASE / 'native_docqa128_freeze_v1.json'),
        protocol_sha256=file_sha(BASE / 'native_docqa128_protocol_v1.json'),
        qa16_quality_sha256=file_sha(BASE / 'docqa16_quality_v1.json'))
    check(matching(parent, expected_parent), 'parent_lifecycle_or_binding')
    check(matching(manifest, dict(model_id=MODEL, revision=REVISION)) and
          matching(model, dict(status='VERIFIED', model_id=MODEL, revision=REVISION,
              manifest_sha256=expected_parent['manifest_sha256'])) and
          isinstance(manifest.get('files'), list) and bool(manifest['files']) and
          model.get('files') == manifest['files'], 'verified_model_inventory')
    check(protocol.get('inputs_sha256') == source_sha and
          matching(protocol, dict(model_id=MODEL, revision=REVISION, **SAMPLING,
              max_model_len=8192, max_num_seqs=128, max_num_batched_tokens=2048,
              kv_cache_memory_bytes=8589934592, enable_prefix_caching=True,
              async_scheduling=False, scheduler_reserve_full_isl=True, scheduling_policy='fcfs')),
          'protocol_binding')
    frozen = freeze.get('files_sha256', {})
    for name in ('docqa_inputs128_v1.json', 'native_docqa128_cell_v1.py',
                 'docqa128_integrity_analyze_v1.py', 'native_docqa128_launch_v1.py',
                 'native_pressure_observer_v1.py', 'health_native.py', 'health_analyze_v2.py',
                 'qwen7b_model_manifest.json', 'native_docqa128_protocol_v1.json'):
        digest = file_sha(BASE / name)
        check(isinstance(frozen, dict) and digest is not None and frozen.get(name) == digest,
              'frozen_dependency:' + name)
    status, config = read(run / 'status.json'), read(run / 'config.json')
    prov, runtime = read(run / 'provenance.json'), read(run / 'runtime.json')
    tokenizer, resolved = read(run / 'tokenizer.json'), read(run / 'resolved-runtime.json')
    check(matching(status, dict(status='COMPLETE', expected_requests=128, request_count=128)), 'cell_lifecycle')
    check(read(run / 'source-input.json') == source and prov.get('input_sha256') == source_sha, 'source_changed')
    check(prov.get('cell_sha256') == file_sha(run / 'cell-source.py') ==
          file_sha(BASE / 'native_docqa128_cell_v1.py'), 'cell_source_changed')
    check(prov.get('q0_helper_sha256') == file_sha(BASE / 'health_native.py') == Q0_SHA and
          prov.get('observer_sha256') == file_sha(BASE / 'native_pressure_observer_v1.py'), 'helper_provenance')
    check(matching(prov, dict(model_id=MODEL, revision=REVISION, reference_facts_loaded=False,
          answer_keys_loaded=False, generation_chat_reconstructed=True,
          generation_input_fields=['instruction', 'article_text', 'questions.question'])), 'model_or_generation_provenance')
    engine = config.get('engine_args', {})
    check(matching(engine, ENGINE) and matching(config.get('sampling'), SAMPLING) and
          config.get('arrival') == 'all at zero' and 'answer_cue' in config and config['answer_cue'] is None,
          'runtime_or_sampling_scope')
    check(isinstance(prov.get('model_dir'), str) and bool(prov['model_dir']) and
          isinstance(engine, dict) and engine.get('model') == engine.get('tokenizer') ==
          parent.get('retained_stage') == prov.get('model_dir'), 'model_directory_binding')
    for label, receipt in (('parent', parent), ('runtime', runtime)):
        check(matching(receipt.get('runtime_environment'), {'VLLM_USE_FLASHINFER_SAMPLER': '0'}),
              label + ':sampler_environment')
    check(matching(resolved, dict(max_num_running_reqs=128, max_num_scheduled_tokens=2048,
          kv_cache_memory_bytes=8589934592, scheduler_reserve_full_isl=True)), 'resolved_runtime_scope')
    check(isinstance(tokenizer.get('chat_template'), str) and bool(tokenizer['chat_template']) and
          'answer_cue' in tokenizer and tokenizer['answer_cue'] is None, 'tokenizer_template_or_answer_cue')
    check(read(run / 'native-drain.json').get('status') == 'QUALIFIED', 'drain')
    reset = read(run / 'prefix-cache-reset.json')
    check(reset.get('reset_succeeded') is True and reset.get('cached_hash_keys_after') == 0 and
          matching(reset.get('before'), {'status': 'QUALIFIED'}) and
          matching(reset.get('after'), {'status': 'QUALIFIED'}), 'cold_apc')
    pressure = read(run / 'measured-pressure.json')
    calls, attempts = pressure.get('scheduler_calls'), pressure.get('allocation_attempts')
    check(pressure.get('schema') == 'c-native-pressure-observer-v1' and pressure.get('status') == 'COMPLETE' and
          matching(pressure.get('configuration'), dict(max_model_len=8192, scheduler_reserve_full_isl=True)) and
          isinstance(calls, list) and bool(calls) and isinstance(attempts, list) and bool(attempts) and
          pressure.get('schedule_calls') == len(calls) and pressure.get('allocation_calls') == len(attempts),
          'pressure_observation_incomplete')
    allowed_eos = eos_ids(read(run / 'resolved-eos.json'))
    check(bool(allowed_eos), 'model_eos_missing')
    native = read(run / 'measured-native-sampling.json')
    mapping, identity_issues = native_id_mapping(native, {'measured/' + rid for rid in planned})
    issues.extend(identity_issues)
    for rid, params in native.items():
        params = params if isinstance(params, dict) else {}
        extra, stops = params.get('stop_token_ids'), params.get('all_stop_token_ids')
        check(params.get('stop') in (None, []) and params.get('ignore_eos') is False and
              params.get('min_tokens') == 0 and params.get('max_tokens') == 512 and
              type(params.get('_eos_token_id')) is int and params['_eos_token_id'] in allowed_eos and
              token_ids(extra, allow_empty=True) and token_ids(stops, allow_empty=True) and
              set(extra + stops).issubset(allowed_eos), rid + ':native_sampling')

    def inventory(rows, label):
        index = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get('request_id'), str):
                issues.append(label + ':invalid_row')
                continue
            rid = row['request_id']
            check(rid not in index, label + ':duplicate:' + rid)
            index[rid] = row
        check(len(rows) == len(index) == 128 and set(index) == set(planned), label + ':request_inventory')
        return index

    rendered = inventory(read(run / 'rendered-inputs.json', list), 'rendered')
    outputs = inventory(read(run / 'measured-outputs.json', list), 'outputs')
    per_request = []
    for src in requests:
        rid = src['request_id']
        render, row = rendered.get(rid, {}), outputs.get(rid, {})
        for label, record in (('rendered', render), ('output', row)):
            check(all(record.get(k) == v for k, v in src.items()), rid + ':' + label + '_source_identity')
        prompt_ids = render.get('prompt_token_ids')
        valid_prompt = token_ids(prompt_ids)
        check(valid_prompt and len(prompt_ids) + 512 <= 8192 and
              render.get('prompt_tokens') == len(prompt_ids) and
              render.get('prompt_token_ids_sha256') == digest_text(json.dumps(prompt_ids, separators=(',', ':'))) and
              isinstance(render.get('prompt'), str) and bool(render['prompt']), rid + ':rendered_prompt')
        check(all(row.get(k) == render.get(k) for k in
                  ('prompt', 'prompt_token_ids', 'prompt_tokens', 'prompt_token_ids_sha256')),
              rid + ':rendered_identity')
        check(row.get('external_request_id') == 'measured/' + rid and row.get('arrival_s') == 0.0,
              rid + ':external_id_or_arrival')
        ids, times, text = row.get('output_token_ids'), row.get('token_times_s'), row.get('output_text')
        valid_ids = token_ids(ids, allow_empty=True)
        check(valid_ids and len(ids) <= 512, rid + ':output_tokens')
        ids = ids if valid_ids else []
        check(isinstance(times, list) and len(times) == len(ids) and
              all(type(t) in (int, float) and math.isfinite(t) and t >= 0 for t in times) and
              all(a <= b for a, b in zip(times, times[1:])), rid + ':token_timing')
        check(isinstance(text, str), rid + ':output_text')
        text = text if isinstance(text, str) else ''
        completed = row.get('finished') is True and row.get('finish_reason') in ('stop', 'length')
        check(completed, rid + ':completion')
        check(row.get('finish_reason') != 'length' or len(ids) == 512, rid + ':cap')
        stop = row.get('stop_reason')
        natural = completed and row.get('finish_reason') == 'stop' and bool(allowed_eos) and (
            stop is None or (type(stop) is int and stop in allowed_eos))
        check(row.get('finish_reason') != 'stop' or natural, rid + ':unknown_stop')
        per_request.append(dict(request_id=rid, input_index=src['input_index'], completed=completed,
            natural_eos=natural, tokens=len(ids), empty=not text.strip(), periodic_suffix=periodic(ids),
            finish_reason=row.get('finish_reason')))
    counts = dict(planned=128, completed=sum(r['completed'] for r in per_request),
        natural_eos=sum(r['natural_eos'] for r in per_request), empty=sum(r['empty'] for r in per_request),
        periodic_suffix=sum(r['periodic_suffix'] is not None for r in per_request),
        length_capped=sum(r['finish_reason'] == 'length' for r in per_request),
        output_tokens=sum(r['tokens'] for r in per_request))
    valid = not issues
    structural_gate = (valid and counts['completed'] == 128 and counts['natural_eos'] >= 112 and
                       counts['empty'] == 0 and counts['periodic_suffix'] <= 8)
    return dict(schema='c-docqa128-integrity-analysis-v1', validity='COMPLETE' if valid else 'INVALID',
        issues=issues, counts=counts, thresholds=THRESHOLDS, structural_gate=structural_gate,
        semantic_quality='PENDING_SEPARATE_SOURCE_GROUNDED_JUDGMENT',
        performance_qualification='NOT_EVALUATED', paper_go='NOT_ESTABLISHED',
        per_request=per_request, native_to_external=mapping, allowed_model_eos_ids=allowed_eos,
        expected_max_seqs=128, analyzer_sha256=sha(__file__), source_sha256=source_sha,
        parent=parent, original_file_sha256=hashes,
        scope='Fixed QA128 execution integrity and structural screen only. Pressure counts are not a GO; '
              'semantic quality, useful capacity, policy benefit and paper qualification remain separate.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=INPUTS)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-max-seqs', type=int, choices=(128,), default=128)
    args = parser.parse_args()
    result = analyze(args.run_root, args.source, args.expected_max_seqs)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('validity', 'issues', 'counts', 'structural_gate')}))
    raise SystemExit(0 if result['validity'] == 'COMPLETE' else 2)
