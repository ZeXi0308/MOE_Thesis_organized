#!/usr/bin/env python3
"""Save a fixed UltraChat training subset with the local model's chat template."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')


def main():
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    p = argparse.ArgumentParser()
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=512)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--shuffle-buffer', type=int, default=4096)
    p.add_argument('--revision')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    repo, split = 'HuggingFaceH4/ultrachat_200k', 'train_sft'
    revision = a.revision or HfApi().dataset_info(repo).sha
    state = {'state': 'RUNNING', 'dataset': repo, 'revision': revision,
             'split': split, 'samples_requested': a.samples, 'seed': a.seed,
             'selection': 'streaming bounded-buffer shuffle then take',
             'shuffle_buffer': a.shuffle_buffer, 'tokenizer': a.tokenizer,
             'start_utc_s': time.time(), 'scope': 'calibration training data; never quality evaluation data',
             'source': 'https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k'}
    def save():
        (a.output / 'manifest.json').write_text(json.dumps(state, indent=2) + '\n')
    save()
    try:
        tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
        if not tokenizer.chat_template:
            raise ValueError('Selected instruct tokenizer has no chat template')
        state['chat_template_sha256'] = hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest()
        dataset = load_dataset(repo, split=split, revision=revision, streaming=True)
        dataset = dataset.shuffle(seed=a.seed, buffer_size=a.shuffle_buffer).take(a.samples)
        n = 0
        with (a.output / 'calibration.jsonl').open('x') as out:
            for row in dataset:
                messages = row['messages']
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                if not text.strip():
                    raise ValueError('Empty calibration text')
                out.write(json.dumps({'prompt_id': row['prompt_id'], 'messages': messages, 'text': text}, ensure_ascii=False) + '\n')
                n += 1
        if n != a.samples:
            raise ValueError(f'Expected {a.samples} rows; received {n}')
        state.update(state='COMPLETE', samples_saved=n,
                     jsonl_sha256=hashlib.sha256((a.output / 'calibration.jsonl').read_bytes()).hexdigest())
    except BaseException as e:
        state.update(state='FAILED', error=f'{type(e).__name__}: {e}')
        raise
    finally:
        state['end_utc_s'] = time.time()
        save()
    print(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
