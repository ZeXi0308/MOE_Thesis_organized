"""Freeze one external-arrival contrast; no engine, GPU, or policy tuning.

Run from any directory: python3 -B cpu_preparation/prepare_two_burst_workload.py
The two equal cohorts individually fit the existing normal KV budget even at
their declared output caps. Actual overlap/preemption is an experimental result.
"""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'inputs_multinews320/pro6000_multinews_candidate_320.json'
OUTPUT = ROOT / 'inputs_multinews320/pro6000_multinews_two_burst160_20s.json'
SOURCE_SHA = '21c601c5aa8bda312ba3a4f10c88b8cc34bbecfbc2fb15f98daf2ac53f707995'


def main():
    source_bytes = SOURCE.read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == SOURCE_SHA
    original = json.loads(source_bytes)
    data = copy.deepcopy(original)
    rows = data['source_requests']
    assert len(rows) == 320
    cohorts = []
    for lo, hi, arrival in ((0, 160, 0.0), (160, 320, 20.0)):
        part = rows[lo:hi]
        prompt = sum(len(r['prompt_token_ids']) for r in part)
        output = sum(r['output_tokens'] for r in part)
        rounded = sum((len(r['prompt_token_ids']) + r['output_tokens'] + 15)//16*16
                      for r in part)
        assert rounded < 590576
        cohorts.append(dict(first_index=lo, end_index_exclusive=hi,
            arrival_s=arrival, prompt_tokens=prompt, output_budget=output,
            total_cap_positions=prompt+output, block_rounded_cap_positions=rounded))
        for row in part:
            row['arrival_s'] = arrival
    for before, after in zip(original['source_requests'], rows):
        assert {k:v for k,v in before.items() if k != 'arrival_s'} == {
            k:v for k,v in after.items() if k != 'arrival_s'}
    assert sum(r['output_tokens'] for r in rows) == 278528
    data.update(name='pro6000_multinews_two_burst160_20s',
        status='FROZEN_EXPLORATORY_WORKLOAD',
        arrival_trace='Two fixed external bursts: indices0..159 at0s,160..319 at20s. '
                      'No progress-dependent release or arrival-source pause.',
        arrival_derivation=dict(source_path=str(SOURCE.relative_to(ROOT)),
            source_sha256=SOURCE_SHA, generator='cpu_preparation/prepare_two_burst_workload.py',
            cohorts=cohorts, capacity_reference_tokens=590576,
            limitation='Cap arithmetic is not measured residence, a pressure guarantee, '
                       'or an oracle.20s is one prospective development setting, not a phase sweep.'),
        generation_semantics=dict(measured_natural_eos=False,
            required_run_group_flag='--fixed-output', temperature=0.0,
            min_tokens='per-request output_tokens', ignore_eos=True,
            output_tokens='fixed count; not an equal-content or quality claim'),
        repetition_disclosure=original['repetition_disclosure'].replace(
            'No prompt truncation, concatenation, padding, new download or forced output length.',
            'No prompt truncation, concatenation, padding or new download; this diagnostic fixes output counts.'),
        comparison_contract='Same arrivals/content/caps/resources/warmup within the H/R pair; '
            'only recovery mode differs. Compared with the old burst domain, only external arrival '
            'structure changes; the same new trace is also used for every warmup.',
        execution_boundary='Existing native full-policy H/R paths, default maintenance. '
            'No one-event target, STORE replay, new selector, victim, queue, or admission changes.',
        execution_condition='Single exploratory arrival-domain comparison; no parameter scan',
        queue_status='See current group status, not this immutable workload')
    payload = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
    if OUTPUT.exists():
        assert OUTPUT.read_text() == payload, 'Refuse to overwrite different frozen workload'
    else:
        OUTPUT.write_text(payload)
    print(json.dumps(dict(output=str(OUTPUT), sha256=hashlib.sha256(payload.encode()).hexdigest(),
                         cohorts=cohorts, requests=320, output_tokens=278528), indent=2))


if __name__ == '__main__':
    main()
