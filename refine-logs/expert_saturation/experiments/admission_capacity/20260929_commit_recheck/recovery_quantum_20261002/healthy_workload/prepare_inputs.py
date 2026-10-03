#!/usr/bin/env python3
"""Prepare A-schema inputs using an already local, hash-pinned tokenizer.

No network, model load, or GPU. Requires only the tokenizers package and exact
local metadata; the output directory must be new.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=('instruct', 'base'), default='instruct')
    p.add_argument('--tokenizer-dir', type=Path, required=True)
    p.add_argument('--model-path', help='Optional already-local weights path for A EngineArgs.model')
    p.add_argument('--split', choices=('dev16', 'holdout16', 'all32'), default='dev16')
    p.add_argument('--arrival', choices=('burst', 'steady'), default='burst')
    p.add_argument('--arrival-gap', type=float, default=0.2)
    p.add_argument('--max-output-tokens', type=int, choices=(256, 512, 1024), default=512)
    p.add_argument('--usable-kv-blocks', type=int, default=4096)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.arrival_gap < 0 or args.usable_kv_blocks < 1:
        p.error('output must be new; arrival gap nonnegative; KV blocks positive')
    profile = read(ROOT / 'model_profiles.json')[args.model]
    hashes = {name: digest(args.tokenizer_dir / name) for name in profile['tokenizer_sha256']}
    if hashes != profile['tokenizer_sha256']:
        raise ValueError('Tokenizer files do not match the selected model revision')
    from tokenizer_backend import OfflineTokenizer
    tokenizer = OfflineTokenizer(args.tokenizer_dir)
    model = dict(profile['model'])
    if args.model_path:
        if not Path(args.model_path).is_dir():
            raise ValueError('model-path must be an existing local directory')
        model.update(hub_id=model['id'], id=str(Path(args.model_path).resolve()))

    def render(text):
        if args.model == 'instruct':
            return tokenizer.render_user(text)
        return text + '\nAnswer:'

    def encode_rows(tasks, check_source=False):
        rows, prompts = [], []
        for task in tasks:
            prompt = render(task['chat_user_content'])
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            if not ids or len(ids) + args.max_output_tokens > 4096:
                raise ValueError('Empty input or complete prompt plus output cap exceeds 4096')
            if check_source and args.model == 'instruct':
                if (prompt != task['original_instruct_rendered_prompt']
                        or token_hash(ids) != task['original_prompt_token_ids_sha256']):
                    raise ValueError('Retokenization differs from original pinned Instruct prompt')
            row = {k: task[k] for k in ('request_id', 'document_id', 'question')}
            row.update(prompt=prompt, prompt_token_ids_sha256=token_hash(ids))
            if 'example_index' in task:
                row['example_index'] = task['example_index']
            rows.append(row)
            prompts.append(ids)
        return rows, prompts

    def package(path, rows, prompts, arrivals, cap, *, warmup=False):
        work = dict(schema='a-healthy-gsm8k-tokenized-v1', task='warmup' if warmup else 'gsm8k',
                    source_requests=rows, actual_prompt_token_ids=prompts,
                    arrival_traces_s={'steady': arrivals},
                    sampling=dict(ignore_eos=warmup, min_tokens=cap if warmup else 0,
                                  max_tokens=cap, temperature=0.0, stop=[], stop_token_ids=[]))
        lengths = [len(ids) for ids in prompts]
        config = dict(status='CPU_TOKENIZED_GPU_UNRUN', model=model, requests=len(rows),
                      prompt_tokens=max(lengths), prompt_tokens_by_request=lengths,
                      output_tokens=cap, max_output_tokens=cap, output_tokens_by_request={},
                      output_mode='fixed' if warmup else 'eos', ignore_eos=warmup,
                      min_tokens=cap if warmup else 0, seed=20260905, cap=32,
                      engine_max_num_seqs=32, max_model_len=4096, max_num_batched_tokens=1024,
                      target_usable_kv_blocks=args.usable_kv_blocks, kv_block_bytes=2097152,
                      fixed_kv_cache_memory_bytes=(args.usable_kv_blocks + 1) * 2097152,
                      offload_gib=16, tokenizer_files_sha256=hashes,
                      tokenizer_backend=tokenizer.receipt,
                      reference_answers_sha256=digest(ROOT / 'reference_answers.json'),
                      tasks_text_sha256=digest(ROOT / 'tasks_text.json'),
                      sampling=work['sampling'], arrival_mode=args.arrival,
                      arrival_gap_s=args.arrival_gap if args.arrival == 'steady' else 0.0,
                      workload_sha256=hashlib.sha256(json.dumps(work, sort_keys=True).encode()).hexdigest())
        path.mkdir(parents=True, exist_ok=False)
        dump(path / 'workload.json', work)
        dump(path / 'config.json', config)
        return config

    tasks = read(ROOT / 'tasks_text.json')['requests']
    # Check both source partitions on every Instruct preparation, including the
    # independent partition that is not submitted in this particular episode.
    if args.model == 'instruct':
        encode_rows(tasks, check_source=True)
    selected = [t for t in tasks if args.split == 'all32' or t['split'] == args.split]
    rows, prompts = encode_rows(selected, check_source=True)
    arrivals = [i * args.arrival_gap if args.arrival == 'steady' else 0.0 for i in range(len(rows))]
    args.output.mkdir(parents=True, exist_ok=False)
    config = package(args.output / 'inputs', rows, prompts, arrivals, args.max_output_tokens)

    # Warmup questions are separate from scored inputs. No warmup outputs are scored.
    short = [dict(request_id=f'warmup-short-{i:02d}', document_id=f'warmup-short-{i:02d}',
                  question=f'What is {i + 3} plus 7?',
                  chat_user_content=f'Calculate {i + 3} plus 7. Explain briefly.') for i in range(32)]
    sr, sp = encode_rows(short)
    package(args.output / 'warmups' / 'short', sr, sp, [0.0] * 32, 16, warmup=True)
    long = []
    for i in range(2):
        sentence = 'A store records its inventory each morning and checks the total again each evening. '
        body = sentence * 100
        while len(tokenizer.encode(render(body), add_special_tokens=False)) < 2048:
            body += sentence
        long.append(dict(request_id=f'warmup-long-{i}', document_id=f'warmup-long-{i}',
                         question='Summarize the inventory process.',
                         chat_user_content=body + '\nSummarize the inventory process.'))
    lr, lp = encode_rows(long)
    package(args.output / 'warmups' / 'long', lr, lp, [0.0] * 2, 16, warmup=True)
    dump(args.output / 'preparation_receipt.json', dict(status='CPU_PREPARED_GPU_UNRUN',
         split=args.split, model_profile=args.model, tokenizer_files_sha256=hashes,
         tokenizer_backend=tokenizer.receipt,
         original_instruct_prompt_text_and_token_hashes_verified=32 if args.model == 'instruct' else 0,
         requests=len(rows), prompt_range=[min(map(len, prompts)), max(map(len, prompts))],
         initial_pages=sum((len(ids) + 15) // 16 for ids in prompts),
         all_requests_cap_upper_pages=sum((len(ids) + args.max_output_tokens + 15) // 16 for ids in prompts),
         usable_kv_blocks=args.usable_kv_blocks, config=config,
         note='Static page upper bound is capacity characterization, not measured natural output length.'))
    print(args.output)


if __name__ == '__main__':
    main()
