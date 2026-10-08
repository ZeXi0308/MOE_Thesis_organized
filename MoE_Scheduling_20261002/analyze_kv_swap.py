"""Actual native CPU-KV A/B: natural output quality plus served/drained cost."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from analyze_natural import ROOT, analyze, comparison, load, ratio, sha
from kv_swap_src.kv_swap_metrics import scheduled_overlap

def connector_stages(directory, raw, cell):
    groups, copies = Counter(), Counter()
    for line in (directory / 'pager/calls.jsonl').open():
        call = json.loads(line)
        if call.get('measurement') and not call.get('validation_run'):
            key = call['context']['step_id']
            groups[key] += call['group_count']
            copies[key] += call['weight_copy_bytes']
    kinds, covered = defaultdict(Counter), Counter()
    steps, calls = raw['scheduler_steps'], raw['engine_calls']
    for call in calls:
        selected = steps[call['scheduler_step_start']:call['scheduler_step_stop']]
        keys = [s['step'] for s in selected]
        covered.update(keys)
        p = sum(r['prefill_tokens'] for s in selected for r in s['scheduled'])
        d = sum(r['decode_tokens'] for s in selected for r in s['scheduled'])
        kind = 'mixed' if p and d else 'pure_prefill' if p else 'pure_decode' if d else 'connector_only_or_idle'
        kinds[kind].update(engine_calls=1, scheduler_steps=len(selected), prefill_tokens=p, decode_tokens=d,
            total_engine_wall_s=call['return_s'] - call['start_s'], groups=sum(groups[k] for k in keys),
            weight_copy_bytes=sum(copies[k] for k in keys), calls_without_scheduler_step=int(not selected))
    return dict(by_kind={k: dict(v) for k, v in kinds.items()},
        total_engine_wall_s=sum(c['return_s'] - c['start_s'] for c in calls), engine_calls=len(calls),
        scheduler_steps=len(steps), scheduled_token_max=max((s['total_scheduled_tokens'] for s in steps), default=0),
        empty_scheduler_steps=sum(s['total_scheduled_tokens'] == 0 for s in steps),
        scheduler_steps_without_moe=sum(groups[s['step']] == 0 for s in steps),
        unassociated_scheduler_steps=[s['step'] for s in steps if covered[s['step']] != 1],
        unassociated_pager_step_ids=[k for k in groups if covered[k] != 1],
        measured_group_sum=sum(groups.values()), measured_copy_bytes_sum=sum(copies.values()),
        pager_totals_match=sum(groups.values()) == cell['group_count'] and sum(copies.values()) == cell['weight_copy_bytes'],
        scope='All captured engine calls retained, including zero-token/no-MoE calls. Post-capture drain is separately measured.')

def analyze_kv(directory, sources, arrivals, tokenizer, expected_budget=2048):
    cell = analyze(directory, sources, arrivals, tokenizer, expected_budget=expected_budget, stage_summarizer=connector_stages)
    raw = load(directory / 'raw.json')
    optional = lambda name: load(directory / name) if (directory / name).exists() else None
    cost = raw.get('kv_service_cost')
    if cost is None:
        cost = dict(served_s=max((r['completion_s'] for r in raw['requests'] if r['status'] == 'completed'), default=None),
            capture_end_s=raw['observation_end_s'], drained_s=None, scope='Legacy capture: served receipt observed; post-capture drain not instrumented.')
    transfers = optional('kv_offload_events.json')
    overlap = scheduled_overlap(raw)
    resources = optional('measurement_resources.json') or {}
    cell.update(kv_service_cost=cost,
        served_output_tokens_per_s=ratio(cell['output_tokens'], cost.get('served_s')) if cell['all_16_complete'] else None,
        drained_output_tokens_per_s=ratio(cell['output_tokens'], cost.get('drained_s')) if cell['all_16_complete'] else None,
        drained_minus_served_s=cost['drained_s'] - cost['served_s'] if cost.get('drained_s') is not None and cost.get('served_s') is not None else None,
        actual_resources=dict(after_init=optional('kv_resources_after_init.json'),
            before_measurement=resources.get('kv_offload'), after_measurement=optional('kv_resources_after_measurement.json'),
            cuda_memory=optional('cuda_memory.json')),
        completed_transfer_bytes=transfers.get('completed_transfer_bytes') if transfers is not None else None,
        transfer_observations=transfers,
        scheduled_position_overlap=raw.get('scheduled_position_overlap', overlap),
        overlap_source='captured' if 'scheduled_position_overlap' in raw else 'computed_from_actual_scheduled_intervals',
        captured_overlap_matches=all(raw['scheduled_position_overlap'].get(k) == v for k, v in overlap.items() if k != 'scope') if 'scheduled_position_overlap' in raw else None)
    return cell

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, default=ROOT / 'inputs/olmoe_gsm8k_natural16')
    parser.add_argument('--metadata', type=Path, default=ROOT.parent / 'C_research_artifacts/20261001/20261001_c_instruct_model_metadata_v1')
    parser.add_argument('--budget', type=int, default=2048)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for directory in (args.baseline, args.candidate):
        if not (directory / 'raw.json').is_file():
            parser.error(f'No metrics written: missing {directory}/raw.json')
    from tokenizers import Tokenizer
    workload = load(args.inputs / 'workload.json')
    tokenizer_path = args.metadata / 'tokenizer.json'
    expected_sha = next(r['sha256'] for r in load(args.metadata / 'metadata-receipt.json')['files'] if r['filename'] == 'tokenizer.json')
    if sha(tokenizer_path) != expected_sha:
        raise ValueError('local tokenizer differs from pinned receipt')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    cells = {name: analyze_kv(directory, workload['source_requests'], workload['arrival_traces_s']['steady'], tokenizer, args.budget)
             for name, directory in (('baseline', args.baseline), ('candidate', args.candidate))}
    pair = comparison('candidate', 'baseline', cells)
    pair['served_drained_ratios'] = {k: ratio(cells['candidate']['kv_service_cost'].get(k), cells['baseline']['kv_service_cost'].get(k))
        if all(c['all_16_complete'] for c in cells.values()) else None for k in ('served_s', 'capture_end_s', 'drained_s')}
    report = dict(evidence_type='ACTUAL_NATIVE_CPU_KV_AB', paths=dict(baseline=str(args.baseline), candidate=str(args.candidate)),
        workload_sha256=sha(args.inputs / 'workload.json'), tokenizer_sha256=expected_sha, cells=cells, comparison=pair,
        notes=['Natural scorer/latencies and per-request outputs are reused unchanged; variable output work prevents equal-work speedup claims.',
            'Served ends at final request output; drained includes cleanup/native push drain. Missing legacy drain/resources/transfer observations remain null.',
            'Full captured engine wall includes connector-only or idle calls, even with no scheduler step or MoE group; post-capture drain remains separately visible.',
            'Completed KV load/store bytes come from native transfer statistics and include observation through drain; lookup offers are not committed restore counts.',
            'Actual resource snapshots, scheduled-position overlap and original-prompt excess are distinct observations. Neither overlap nor prompt excess estimates elapsed-time savings.',
            'No offline speed forecast or future-policy replay is performed.'])
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(args.output)

if __name__ == '__main__':
    main()
