#!/usr/bin/env python3
"""Build a distinct, unrun summarization domain from existing complete articles.

CPU tokenizer only; no model weights, CUDA, network, or output-dependent choices.
The resulting task is not a strategy comparison with the old continuation task.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

MODEL = 'allenai/OLMoE-1B-7B-0924-Instruct'
REVISION = '7f1c97f440f06ce36705e4f2b843edb5925f4498'
INSTRUCTION = ('Summarize the following article accurately and concisely. Cover its main '
               'points and conclusions. Use only information supported by the article. '
               'Return only the summary.')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode_json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def build(source, snapshot, output):
    if output.exists():
        raise ValueError('Output exists; preserve previous inputs')
    if snapshot.name != REVISION:
        raise ValueError('Expected the existing pinned Instruct snapshot')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', USE_TORCH='0',
                      USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True,
                                             trust_remote_code=False)
    model_config = json.loads((snapshot / 'config.json').read_text())
    generation = json.loads((snapshot / 'generation_config.json').read_text())
    if (model_config.get('max_position_embeddings') != 4096
            or model_config.get('eos_token_id') != 50279
            or generation.get('eos_token_id') != 50279 or tokenizer.eos_token_id != 50279):
        raise ValueError('Unexpected context or EOS identity')
    original_bytes = (source / 'workload.json').read_bytes()
    workload = json.loads(original_bytes)
    config = json.loads((source / 'config.json').read_text())
    articles = copy.deepcopy(workload['source_requests'])
    token_rows, lengths = [], []
    for row in articles:
        article = row['prompt']
        if digest(article.encode()) != row['prompt_sha256']:
            raise ValueError('Original article identity differs')
        messages = [dict(role='user', content=INSTRUCTION + '\n\nArticle:\n' + article)]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=False)
        if (not tokens or any(type(t) is not int for t in tokens)
                or tokens != tokenizer.encode(text, add_special_tokens=False)):
            raise ValueError('Chat template tokenization/roundtrip differs')
        row['article_text'] = article
        row['source_article_sha256'] = digest(article.encode())
        row['prompt'] = text
        row['prompt_sha256'] = digest(text.encode())
        row['prompt_token_count'] = len(tokens)
        row['prompt_token_ids_sha256'] = digest(json.dumps(tokens, separators=(',', ':')).encode())
        token_rows.append(tokens)
        lengths.append(len(tokens))
    # One common ceiling, derived solely from native geometry; never truncate an article.
    budget = min(1024, 4096 - max(lengths))
    if budget < 2 or len(articles) != 320:
        raise ValueError('Unexpected input count or insufficient native context')
    for row in articles:
        row['max_output_tokens'] = budget
    workload['source_requests'] = articles
    workload['actual_prompt_token_ids'] = token_rows
    workload['output_contract'] = dict(max_output_tokens=budget, ignore_eos=False, min_tokens=0,
        actual_output_lengths='UNKNOWN_UNTIL_EXECUTION',
        budget_basis='min(1024, 4096 - largest chat-template prompt length), common to all requests')
    order = sorted(range(len(articles)), key=lambda i: (lengths[i], articles[i]['request_id']))
    quality_ids = [articles[order[(2*k+1)*len(order)//16]]['request_id'] for k in range(8)]
    workload['task'] = dict(kind='ARTICLE_SUMMARIZATION_FEASIBILITY_UNRUN', instruction=INSTRUCTION,
        article_source_workload_sha256=digest(original_bytes), model=MODEL, revision=REVISION,
        quality_check_ids=quality_ids,
        quality_scope='Preselected length-stratified checks: supported facts, main-point coverage, '
                      'coherence, no continuation/repetition, and truncation. No gold summaries; '
                      'these checks cannot establish policy quality equivalence.',
        independence='Previously seen articles and arrival sequence; a new task domain, not holdout validation.')
    data = encode_json(workload)
    for key in ('output_tokens_by_request', 'budget_mixture'):
        config.pop(key, None)
    config.update(status='NATURAL_SUMMARY_CPU_PREPARED_GPU_UNRUN',
        model=dict(key='olmoe', id=MODEL, revision=REVISION, tokenizer_revision=REVISION,
                   dtype='bfloat16'), requests=320, prompt_tokens=max(lengths),
        prompt_tokens_by_request=lengths, output_tokens=budget, max_output_tokens=budget,
        workload_sha256=digest(json.dumps(workload, sort_keys=True).encode()),
        workload_file_sha256=digest(data), input_independence=workload['task']['independence'],
        prompt_tokens_semantics='Full original articles inside the official chat template; no truncation.',
        output_tokens_semantics='Common geometry-limited safety ceiling; natural termination unmeasured.',
        input_preparation='Original article order and external arrivals preserved; one fixed summary instruction.',
        fixed_kv_cache_memory_bytes=None, intended_usable_kv_bytes=None,
        resource_state='GPU UNRUN; Instruct normal capacity must be measured before service claims.')
    config['article_source'] = copy.deepcopy(config['source'])
    config['source']['tokenizer_files_sha256'] = {name: digest((snapshot / name).read_bytes())
        for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json')}
    config['source']['selection'] = 'All 320 previously selected complete articles; no output-dependent selection.'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'workload.json').write_bytes(data)
    (output / 'config.json').write_bytes(encode_json(config))
    print(json.dumps(dict(state=config['status'], requests=320, prompt_min=min(lengths),
        prompt_max=max(lengths), common_max_output=budget, eos_token_id=50279,
        quality_check_ids=quality_ids, workload_sha256=digest(data), output=str(output)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.snapshot, args.output)
