#!/usr/bin/env python3
"""Unmeasured arrival permutation: reverse within length classes, retain all tasks."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
SOURCE = BASE/'recovery_start_gate/simple_cap/inputs/cap256'


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build(destination):
    if destination.exists():
        raise FileExistsError(destination)
    spec = importlib.util.spec_from_file_location('frozen_input_loader', BASE/'pkg/run_recovery_cadence.py')
    loader = importlib.util.module_from_spec(spec); spec.loader.exec_module(loader)
    config, original = loader.load_inputs(SOURCE)
    rows, prompts = original['source_requests'], original['actual_prompt_token_ids']
    lengths = [len(tokens) for tokens in prompts]
    if (len(rows) != 256 or lengths != [3072, 512]*128
            or original['arrival_traces_s'] != {'steady': [i*.1 for i in range(256)]}
            or config['cap'] != 256 or config['engine_max_num_seqs'] != 256
            or config['output_tokens'] != 1024 or config['output_tokens_by_request']
            or config['ignore_eos'] is not True or config['min_tokens'] != 0
            or config['output_mode'] != 'fixed'):
        raise ValueError('Requires the original alternating256 / cap256 / fixed1024 source contract')
    # Pop from each original class's tail; arrival slots themselves never move.
    remaining = {size: [i for i, value in enumerate(lengths) if value == size] for size in (3072, 512)}
    order = [remaining[size].pop() for size in lengths]
    workload = dict(original, source_requests=[rows[i] for i in order],
                    actual_prompt_token_ids=[prompts[i] for i in order])
    result = dict(config, workload_sha256=canonical_sha(workload), arrival_permutation=dict(
        status='CPU_PREPARED_GPU_UNRUN', rule='Reverse source order within 3072-token and 512-token classes',
        scope='Untested arrival permutation of the same 256 requests; not new task data or independent task samples',
        source_inputs_from_B_root=str(SOURCE.relative_to(BASE)), source_workload_sha256=config['workload_sha256'],
        source_index_semantics='Each source_requests row is unchanged; source_index remains its original provenance, not the new arrival slot',
        invariant='Same request IDs, row contents, prompt tokens, per-request output budgets, 0.1s arrival slots and per-slot lengths',
        warmup_inputs_from_B_root='pkg/warmups',
        warmup_semantics='Keep the existing separate --warmup-inputs argument; original short first32 / long first2 and 16-token warmups; never derive from reordered measurement requests',
        inherited_output_metadata='workload.output_contract is preserved historical EOS metadata; execution remains governed by the unchanged fixed1024/ignore_eos=True config and fixed child'))
    # One identity/content/arrival check, using the exact production loader again
    # after the exclusive write. No GPU or tokenizer is imported.
    original_by_id = {row['request_id']: (row, tokens) for row, tokens in zip(rows, prompts)}
    reordered_by_id = {row['request_id']: (row, tokens) for row, tokens in zip(workload['source_requests'], workload['actual_prompt_token_ids'])}
    assert original_by_id == reordered_by_id and len(reordered_by_id) == 256
    assert [len(tokens) for tokens in workload['actual_prompt_token_ids']] == lengths
    assert workload['arrival_traces_s'] == original['arrival_traces_s']
    assert all(workload[key] == value for key, value in original.items()
               if key not in ('source_requests', 'actual_prompt_token_ids'))
    assert all(result[key] == value for key, value in config.items() if key != 'workload_sha256')
    for size in (3072, 512):
        assert [row['request_id'] for row, tokens in zip(workload['source_requests'], workload['actual_prompt_token_ids']) if len(tokens) == size] == [row['request_id'] for row, tokens in zip(rows, prompts) if len(tokens) == size][::-1]
    warmup_paths = [BASE/'pkg/warmups'/domain/name for domain in ('short', 'long') for name in ('config.json', 'workload.json')]
    warmup_before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in warmup_paths}
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in (('workload.json', workload), ('config.json', result)):
        with (destination/name).open('x') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n')
    loaded_config, loaded_workload = loader.load_inputs(destination)
    assert loaded_config == result and loaded_workload == workload
    assert warmup_before == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in warmup_paths}
    print(json.dumps(dict(status='PASS_CPU_INPUT_CHECK_GPU_UNRUN', output=str(destination), requests=256,
        changed_arrival_slots=sum(i != j for i, j in enumerate(order)), prompt_tokens=sum(lengths),
        output_token_budget=256*1024, workload_sha256=result['workload_sha256'],
        first_request=workload['source_requests'][0]['request_id'], last_request=workload['source_requests'][-1]['request_id'],
        warmup='Existing independent pkg/warmups files unchanged; no new warmup data')))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, default=ROOT/'inputs/cap256')
    build(parser.parse_args().output)


if __name__ == '__main__':
    main()
