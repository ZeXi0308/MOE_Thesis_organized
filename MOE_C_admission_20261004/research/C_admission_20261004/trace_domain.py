"""Describe a preselected public trace prefix; never make online length predictions.

Azure GeneratedTokens is an observed result, not a client declaration. These
statistics cannot be used as an online policy input or a service simulation.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
from urllib.request import Request, urlopen


SOURCE = ('https://github.com/Azure/AzurePublicDataset/releases/download/'
          'dataset-llm-2024/AzureLLMInferenceTrace_code_1week.csv')
DESCRIPTION = 'https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md'


def distribution(values):
    values = sorted(values)
    def percentile(q):
        position = (len(values)-1)*q
        lower = int(position)
        return values[lower]+(values[min(lower+1, len(values)-1)]-values[lower])*(position-lower)
    return dict(count=len(values), min=values[0], mean=statistics.mean(values),
        p50=percentile(.5), p90=percentile(.9), p95=percentile(.95), p99=percentile(.99),
        max=values[-1], total=sum(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserved output must not exist.')
    if args.download:
        if args.prefix.exists():
            parser.error('Download destination already exists.')
        request = Request(SOURCE+'?download=1',
            headers={'User-Agent': 'C-admission-research', 'Range': 'bytes=0-524287'})
        with urlopen(request, timeout=45) as response:
            lines = [response.readline() for _ in range(4097)]
        if not all(lines) or not lines[-1].endswith(b'\n'):
            raise ValueError('Range did not contain 4096 complete data rows.')
        data = b''.join(lines)
        if data.splitlines()[0].decode('utf-8-sig').strip() != 'TIMESTAMP,ContextTokens,GeneratedTokens':
            raise ValueError('Unexpected public schema.')
        with args.prefix.open('xb') as handle:
            handle.write(data)
    with args.prefix.open(encoding='utf-8-sig', newline='') as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == 4096
    prompts = [int(r['ContextTokens']) for r in records]
    generated = [int(r['GeneratedTokens']) for r in records]
    times = [datetime.fromisoformat(r['TIMESTAMP']) for r in records]
    assert min(prompts) >= 0 and min(generated) >= 0
    span = (max(times)-min(times)).total_seconds()
    root = Path(__file__).resolve().parent
    cells = []
    for filename in sorted((root/'analysis').glob('westd-mc-budget-phase-r01.*.requests.json')):
        population = json.loads(filename.read_text())
        cells.append(dict(source=filename.name, requests=len(population),
            prompts=distribution([r['prompt_tokens'] for r in population]),
            generated=distribution([r['output_tokens'] for r in population]),
            hit_declared_limit=sum(r['length_stop'] for r in population),
            max_declarations=sorted({r['max_output_tokens'] for r in population})))
    report = dict(evidence='CPU_DESCRIPTOR_NOT_GPU_SERVICE_OR_PRODUCTION_REPLAY',
        new_gpu_runs=0, new_online_actions=0,
        public_source=dict(url=SOURCE, description=DESCRIPTION,
            attribution='Microsoft Azure Public Dataset; DynamoLLM, HPCA 2025, Stojkovic et al.',
            license='CC BY 4.0', license_url='https://creativecommons.org/licenses/by/4.0/',
            selection='First 4096 data rows in file order, fixed before reading values; no filtering.',
            modification='Original byte prefix only; the rest of the full-week file was not downloaded.',
            prefix_sha256=hashlib.sha256(args.prefix.read_bytes()).hexdigest()),
        public=dict(requests=len(records), prompts=distribution(prompts), generated=distribution(generated),
            first_timestamp=records[0]['TIMESTAMP'], last_timestamp=records[-1]['TIMESTAMP'],
            timestamp_span_s=span, request_count_per_span_s=len(records)/span,
            adjacent_time_inversions=sum(b<a for a,b in zip(times,times[1:])),
            zero_prompts=sum(p==0 for p in prompts), zero_generated=sum(g==0 for g in generated),
            generated_le128=sum(g<=128 for g in generated), generated_le256=sum(g<=256 for g in generated),
            generated_gt1024=sum(g>1024 for g in generated),
            prompt_plus_actual_gt4096=sum(p+g>4096 for p,g in zip(prompts,generated)),
            prompt_plus_declared1024_gt4096=sum(p+1024>4096 for p in prompts)),
        existing_v20=cells,
        limitations=['Public sample has no prompt text, client max_tokens, termination reason or application SLO.',
            'Public realized output lengths are unavailable to online admission; no policy receives them here.',
            'Token-count compatibility is descriptive; tokenizer/model equivalence is not established.',
            'One file prefix does not represent every production time window or the full dataset.',
            'Length/arrival differences alone imply neither memory contention nor a new-method benefit.'])
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps(report['public']))


if __name__ == '__main__':
    main()
