"""Freeze small PRO 6000 load set from complete, public natural prompts.

CPU only; no model load, network, CUDA import, context truncation or token padding.
Example:
  python prepare_pro6000_inputs.py  # requires the public tokenizers package
The local defaults read only raw dataset text and tokenizer metadata; no other
session's measured outcomes or policy decisions are used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


TOKENIZER_SHA256 = 'b1fb1517c84c6d516ff43adcebe7a1986ce7ed3e2533dc2c22b40337d2d24167'
KV_BYTES_PER_TOKEN = 131072  # 2 KV * 16 layers * 16 heads * 128 dims * BF16.


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


def read_selected(path):
    return {r['source_index']: r['source'] for r in json.loads(path.read_text())}


def write(path, obj):
    # Compact token arrays keep the frozen corpora small enough to transfer.
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(',', ':'),
                               allow_nan=False) + '\n')


def main():
    root = Path(__file__).resolve().parent
    sources = root / 'inputs_pro6000/sources'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gsm8k', type=Path,
                   default=sources/'gsm8k_test_selected.json')
    p.add_argument('--longqa', type=Path,
                   default=sources/'multifieldqa_en_selected.json')
    p.add_argument('--tokenizer', type=Path,
                   default=sources/'tokenizer.json')
    p.add_argument('--old-workload', type=Path, default=root/'workload.json')
    p.add_argument('--out', type=Path, default=root/'inputs_pro6000')
    a = p.parse_args()
    from tokenizers import Tokenizer
    assert sha256(a.tokenizer) == TOKENIZER_SHA256, 'Different tokenizer: do not reuse these bounds.'
    tokenizer = Tokenizer.from_file(str(a.tokenizer))
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    old = json.loads(a.old_workload.read_text())['source_requests']
    assert all(encode(row['prompt']) == row['prompt_token_ids'] for row in old)
    prefix = old[0]['prompt'].rsplit('Question:', 1)[0]
    assert prefix.count('Question:') == 8, 'Expected the existing E eight-shot prefix.'
    gsm = read_selected(a.gsm8k)
    longqa = read_selected(a.longqa)
    short_pool = []
    # Source indices fixed before observing the PRO 6000 performance runs.
    for index in range(128, 192):
        source = gsm[index]
        prompt = prefix + 'Question: ' + source['question'] + '\n<|assistant|>\nAnswer:'
        ids = encode(prompt)
        assert 600 <= len(ids) <= 1000 and len(ids) + 256 <= 4096
        short_pool.append(dict(task='gsm8k', example_index=index,
            document_id=f'gsm8k-test-{index:04d}', prompt=prompt,
            prompt_token_ids=ids, prompt_token_ids_sha256=digest(ids),
            gold=source['answer'].rsplit('####', 1)[-1].strip(),
            reference_answer=source['answer'], output_tokens=256,
            request_class='short', source_context_complete=True,
            synthetic_extension=False))
    long_pool = []
    observed_lengths = []
    for index, source in sorted(longqa.items()):
        # The entire original context and question are retained verbatim.
        prompt = ('<|endoftext|><|user|>\nRead the following text and answer the question.\n\n'
                  + source['context'] + '\n\nQuestion: ' + source['input']
                  + '\n<|assistant|>\nAnswer:')
        ids = encode(prompt)
        observed_lengths.append(dict(example_index=index, prompt_tokens=len(ids)))
        if 2500 <= len(ids) <= 3400:
            assert len(ids) + 512 <= 4096
            long_pool.append(dict(task='multifieldqa_en', example_index=index,
                document_id=source['_id'], prompt=prompt,
                prompt_token_ids=ids, prompt_token_ids_sha256=digest(ids),
                source_context_sha256=hashlib.sha256(source['context'].encode()).hexdigest(),
                gold=source['answers'], reference_answer=source['answers'],
                output_tokens=512, request_class='long',
                source_context_complete=True, synthetic_extension=False))
    assert len(long_pool) == 17, 'Source changed; inspect before replacing frozen workloads.'
    master = []
    short_index = long_index = 0
    counts = {}
    for index in range(256):
        if index % 4 == 0:
            source = short_pool[short_index]
            short_index += 1
        else:
            source = long_pool[long_index % len(long_pool)]
            long_index += 1
        sample_id = f"{source['task']}:{source['example_index']}"
        repetition = counts.get(sample_id, 0)
        counts[sample_id] = repetition + 1
        master.append(dict(source, request_id=f'E-pro6000-{index:04d}',
            arrival_s=0.0, source_sample_id=sample_id,
            repetition_index=repetition, independent_request_identity=True,
            prompt_tokens=len(source['prompt_token_ids'])))
    source_manifest = {
        'gsm8k': dict(path=str(a.gsm8k), sha256=sha256(a.gsm8k), rows=len(gsm),
            original_source_file_sha256='3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14',
            original_source_rows=1319,
            public_url='https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/test.jsonl',
            selected_test_indices=list(range(128, 192))),
        'longqa': dict(path=str(a.longqa), sha256=sha256(a.longqa), rows=len(longqa),
            original_source_file_sha256='0aac182fd317dcf6d74f8e1e0f3e61029407435346c2e0b3ff9fb45ae49c5c3f',
            original_source_rows=150,
            public_url='https://huggingface.co/datasets/zai-org/LongBench',
            revision='5e628be450b7e67fb7ae6e201bd6d8f7056f7672',
            member='data/multifieldqa_en.jsonl',
            selected_source_indices=[r['example_index'] for r in long_pool]),
        'tokenizer': dict(path=str(a.tokenizer), sha256=sha256(a.tokenizer),
            remote_model='/root/autodl-tmp/moe-research-20261002/model',
            remote_sha256_verified=True),
        'eight_shot_prefix': dict(path=str(a.old_workload), sha256=sha256(a.old_workload),
            reused_E_existing_prompts_retokenize_exact=True),
        'script_sha256': sha256(__file__),
    }
    a.out.mkdir(parents=True, exist_ok=True)
    receipt = dict(status='FROZEN_CPU_ONLY', source_manifest=source_manifest,
        unique_short_prompts=len(short_pool), unique_long_prompts=len(long_pool),
        long_prompt_lengths=[len(r['prompt_token_ids']) for r in long_pool], workloads=[])
    for n, pressure in [(64, 'low'), (192, 'near_capacity'), (256, 'high')]:
        rows = master[:n]
        prompt_tokens = sum(r['prompt_tokens'] for r in rows)
        output_tokens = sum(r['output_tokens'] for r in rows)
        full_tokens = sum(((r['prompt_tokens'] + r['output_tokens'] + 15)//16)*16 for r in rows)
        stats = dict(requests=n, short_requests=n//4, long_requests=n*3//4,
            unique_prompts=len({r['prompt_token_ids_sha256'] for r in rows}),
            prompt_tokens_total=prompt_tokens, output_token_caps_total=output_tokens,
            prompt_tokens_min=min(r['prompt_tokens'] for r in rows),
            prompt_tokens_max=max(r['prompt_tokens'] for r in rows),
            prompt_tokens_mean=statistics.mean(r['prompt_tokens'] for r in rows),
            nominal_complete_KV_gib=full_tokens*KV_BYTES_PER_TOKEN/1024**3,
            block_rounding_tokens=16,
            estimate_caveat='All requests jointly resident at output cap is a pressure estimate, '
                            'not observed GPU allocation or proof that preemption occurs.')
        obj = dict(schema='E.pro6000.workload.v1', name=f'pro6000_{pressure}_{n}',
            research_scope='Only legal recovery action Host restore versus native recompute.',
            source_manifest=source_manifest, source_requests=rows, statistics=stats,
            workload_kind='repeated_complete_natural_prompts_capacity_probe',
            repetition_disclosure='17 complete natural LongBench prompts are cycled with independent '
                'request identities. No context is truncated, repeated inside a prompt, or padded. '
                'This is a repeated-input capacity probe, not 192 distinct long documents.',
            generation_semantics='output_tokens are per-request caps; forcing exactly the cap with '
                'ignore_eos/min_tokens must be reported as fixed-output pressure probing. '
                'Natural EOS evaluation is a separate mode.',
            arrival_trace='All external arrivals are 0 seconds; rows retain master256 order.',
            suggested_fixed_resources=dict(dtype='bfloat16', max_model_len=4096,
                gpu_memory_utilization=0.9, kv_cache_memory_bytes=None,
                kv_offloading_size_gib=32, max_num_seqs=256,
                max_num_batched_tokens=2048, kv_bytes_per_token=KV_BYTES_PER_TOKEN),
            comparison_contract='Same file hash, arrival_s, output caps, warmup, resources, '
                'admission/victim/priority rules in every arm; normal-capacity no-recovery is valid.')
        filename = f'pro6000_{pressure}_{n}.json'
        write(a.out/filename, obj)
        receipt['workloads'].append(dict(file=filename, sha256=sha256(a.out/filename), **stats))
    write(a.out/'freeze_receipt.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
