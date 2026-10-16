"""Prepare reproducible first-human ShareGPT prompts; no GPU initialization."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--count', type=int, default=160)
    p.add_argument('--min-prompt', type=int, default=32)
    p.add_argument('--max-prompt', type=int, default=256)
    args = p.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    source = Path(args.source)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    expected = '35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4'
    if digest != expected:
        raise ValueError('ShareGPT file differs from pinned revision 192ab2185289094fc556ec8ce5ce1e8e587154ca')
    data = json.loads(raw)
    records = []
    for idx, item in enumerate(data):
        turns = item.get('conversations', [])
        if len(turns) < 2 or turns[0].get('from') not in ('human', 'user'):
            continue
        text = turns[0].get('value', '')
        if not text:
            continue
        rendered = tokenizer.apply_chat_template([{'role':'user','content':text}], tokenize=False,
                                                  add_generation_prompt=True)
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        # Do not truncate a prompt or select by generated outcome.
        if not args.min_prompt <= len(ids) <= args.max_prompt:
            continue
        records.append({'source_index': idx, 'source_id': item.get('id'),
                        'prompt_token_ids': ids, 'prompt': text,
                        'rendered_sha256': hashlib.sha256(rendered.encode()).hexdigest()})
        if len(records) == args.count:
            break
    if len(records) < args.count:
        raise ValueError(f'only {len(records)} eligible prompts')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps({'dataset': 'ShareGPT_V3_unfiltered_cleaned_split',
        'source': str(source), 'source_sha256': digest,
        'source_revision': '192ab2185289094fc556ec8ce5ce1e8e587154ca',
        'model': args.model, 'selection': 'first eligible first-human turn, original order',
        'prompt_range': [args.min_prompt, args.max_prompt], 'records': records}, ensure_ascii=False) + '\n')
    print(json.dumps({'output': str(output), 'requests': len(records), 'source_sha256': hashlib.sha256(raw).hexdigest()}))


if __name__ == '__main__':
    main()
