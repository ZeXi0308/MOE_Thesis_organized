#!/usr/bin/env python3
"""Source-only QA assembly and local-tokenizer preflight; never loads model weights."""
import argparse
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
CAP = 512
INSTRUCTION = ('Using only the article below, answer each of the three questions. '
    'Give a concise answer to each question, followed by a short quotation from '
    'the article that supports it. If the article does not support an answer, '
    'say so. Do not add outside facts.')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(name):
    return json.loads((BASE / name).read_text())


def write(name, data):
    with (BASE / name).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def assemble():
    original = read('summary128_inputs_v1.json')
    references = read('summary128_reference_v1.json')['references']
    names = ('docqa_questions_part_a_v1.json', 'docqa_questions_part_b_v1.json')
    annotations = [r for name in names for r in read(name)['requests']]
    annotations.sort(key=lambda r: r['input_index'])
    assert len(original['requests']) == len(references) == len(annotations) == 128
    assert [r['input_index'] for r in annotations] == list(range(128))
    assert len({r['request_id'] for r in annotations}) == 128
    rows = []
    for src, ref, ann in zip(original['requests'], references, annotations):
        assert src['request_id'] == ref['request_id'] == ann['request_id']
        assert src['input_index'] == ref['input_index'] == ann['input_index']
        article_sha = digest(src['article_text'].encode())
        assert article_sha == src['article_text_sha256'] == ref['article_text_sha256'] == ann['article_text_sha256']
        questions = ann['questions']
        assert len(questions) == 3 and [q['question_id'] for q in questions] == ['Q1', 'Q2', 'Q3']
        assert len({q['question'] for q in questions}) == 3
        for q, fact in zip(questions, ref['facts']):
            assert q['reference_fact_id'] == fact['fact_id']
            assert isinstance(q['question'], str) and q['question'].strip()
            assert q['answer_requirements'] and all(isinstance(s, str) and s.strip() for s in q['answer_requirements'])
            assert q['support_spans'] == fact['support_spans']
            for span in q['support_spans']:
                assert src['article_text'][span['start_char']:span['end_char_exclusive']] == span['text']
                assert digest(span['text'].encode()) == span['text_sha256']
        visible = [{k: q[k] for k in ('question_id', 'question')} for q in questions]
        content = INSTRUCTION + '\n\nArticle:\n' + src['article_text'] + '\n\nQuestions:\n'
        content += '\n'.join(f"{i + 1}. {q['question']}" for i, q in enumerate(visible))
        rows.append(dict(src, task='document_qa', questions=visible,
            chat_user_content=content, chat_user_content_sha256=digest(content.encode())))
    answers = dict(schema='c-docqa-answer-key-v1', requests=annotations,
        source_reference_sha256=digest((BASE / 'summary128_reference_v1.json').read_bytes()),
        question_parts_sha256={n: digest((BASE / n).read_bytes()) for n in names},
        scope='Source-only answer requirements; never passed to generation. All128 before any QA outputs.')
    write('docqa_answers128_v1.json', answers)
    candidate = dict(schema='c-docqa-candidate-inputs-v1', task='document_qa',
        source_workload_sha256=original['source_workload_sha256'],
        questions_answer_key_sha256=digest((BASE / 'docqa_answers128_v1.json').read_bytes()),
        requests_planned=128, source_selection='All existing ranks832..959, unchanged source order.',
        instruction=INSTRUCTION, sampling=original['sampling'], arrival_traces_s=[0.0] * 128,
        scope='Development document QA; no input/output filtering and no held-out claim.', requests=rows)
    assert candidate['sampling'] == dict(temperature=0.0, max_tokens=CAP, min_tokens=0,
                                        ignore_eos=False, stop=[], seed=20260905)
    write('docqa_inputs128_candidate_v1.json', candidate)
    print(json.dumps(dict(status='ASSEMBLED', requests=128, questions=384)))


def tokenize(model_dir):
    from transformers import AutoTokenizer
    candidate = read('docqa_inputs128_candidate_v1.json')
    assert candidate['schema'] == 'c-docqa-candidate-inputs-v1'
    manifest = read('qwen7b_model_manifest.json')
    marker = json.loads((model_dir / 'MODEL_IDENTITY.json').read_text())
    assert marker['model_id'] == manifest['model_id'] and marker['revision'] == manifest['revision']
    assert marker['manifest_sha256'] == digest((BASE / 'qwen7b_model_manifest.json').read_bytes())
    for item in manifest['files']:
        if not item['filename'].endswith('.safetensors'):
            assert digest((model_dir / item['filename']).read_bytes()) == item['sha256']
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    rows, prefix_keys = [], set()
    independent_prompt = independent_full = private_prompt = private_full = 0
    for row in candidate['requests']:
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': row['chat_user_content']}],
                                              tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        n, shareable = len(ids), (len(ids) - 1) // 16
        assert n > 0
        prompt_blocks, full_blocks = (n + 15) // 16, (n + CAP + 15) // 16
        independent_prompt += prompt_blocks; independent_full += full_blocks
        private_prompt += prompt_blocks - shareable; private_full += full_blocks - shareable
        parent = ''
        for index in range(shareable):
            parent = digest(json.dumps([parent, ids[index * 16:(index + 1) * 16]], separators=(',', ':')).encode())
            prefix_keys.add(parent)
        rows.append(dict(request_id=row['request_id'], input_index=row['input_index'], prompt_tokens=n,
                         prompt_token_ids_sha256=digest(json.dumps(ids, separators=(',', ':')).encode())))
    largest = max(r['prompt_tokens'] for r in rows) + CAP
    context = next((n for n in (4096, 8192) if largest <= n), None)
    config = json.loads((model_dir / 'config.json').read_text())
    assert context is None or context <= config['max_position_embeddings']
    preflight = dict(status='PASS_INPUT_GEOMETRY' if context else 'FAIL_CONTEXT',
        candidate_sha256=digest((BASE / 'docqa_inputs128_candidate_v1.json').read_bytes()),
        model_id=manifest['model_id'], revision=manifest['revision'], max_model_len=context,
        requests=rows, prompt_tokens_min=min(r['prompt_tokens'] for r in rows),
        prompt_tokens_max=max(r['prompt_tokens'] for r in rows), cap=CAP, assumed_block_size=16,
        independent_prompt_blocks=independent_prompt, independent_full_cap_blocks=independent_full,
        ideal_shared_prompt_blocks=len(prefix_keys) + private_prompt,
        ideal_shared_full_cap_blocks=len(prefix_keys) + private_full,
        scope='CPU tokenizer/ideal available-prefix geometry only; no native ownership or pressure result.')
    write('docqa_context_preflight_v1.json', preflight)
    if context is None:
        raise ValueError('No allowed context fits all128 complete inputs; no truncation/selection')
    final = dict(candidate, schema='c-docqa-inputs-v1', max_model_len=context)
    write('docqa_inputs128_v1.json', final)
    first = dict(final, requests_planned=16, requests=final['requests'][:16], arrival_traces_s=[0.0] * 16,
                 source_selection='Fixed first16 of all128; no output filtering.')
    write('docqa_inputs16_v1.json', first)
    print(json.dumps({k: v for k, v in preflight.items() if k != 'requests'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--assemble', action='store_true')
    action.add_argument('--tokenize', type=Path, metavar='MODEL_DIR')
    args = parser.parse_args()
    assemble() if args.assemble else tokenize(args.tokenize)
