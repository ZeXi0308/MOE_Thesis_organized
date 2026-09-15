"""Select source-ordered eligible articles 113..128; no GPU or outcome selection."""
import json
import os
from pathlib import Path
import re
import runpy
import sys

HERE = Path(__file__).resolve().parent
B = HERE.parent
C = B / 'fresh_cohort_r01'
P = B / 'logical_alignment_performance_r01'
prior = runpy.run_path(str(C / 'prepare_inputs.py'))
original, POOL, OLD, ROOT = (prior[k] for k in ('original', 'POOL', 'OLD', 'ROOT'))
read = lambda p: json.loads(p.read_text())


def main():
    destination = HERE / 'prepared/workload.json'
    assert not destination.exists() and not (HERE / 'input_provenance.json').exists()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      CUDA_VISIBLE_DEVICES='', TOKENIZERS_PARALLELISM='false')
    import datasets, transformers, tokenizers
    from datasets import Dataset
    from transformers import AutoTokenizer
    from transformers.utils.hub import cached_file
    previous = read(C / 'input_provenance.json')
    manifest = read(POOL / 'manifest.json'); model = manifest['tokenizer']
    arrow = Path(previous['dataset']['arrow_path'])
    assert original.file_sha(arrow) == manifest['dataset']['arrow_sha256']
    files = {}
    for name, expected in model['files'].items():
        path = Path(cached_file(model['model_id'], name, revision=original.REVISION, local_files_only=True))
        assert original.file_sha(path) == expected['sha256']
        files[name] = dict(path=str(path), sha256=expected['sha256'])
    tokenizer = AutoTokenizer.from_pretrained(model['model_id'], revision=original.REVISION, local_files_only=True)
    assert tokenizer.__class__.__name__ == model['tokenizer_class'] and model['revision'] == original.REVISION
    eligible, start, pieces = [], None, []
    for index, row in enumerate(Dataset.from_file(str(arrow))):
        text = str(row['text'])
        if re.fullmatch(r'= [^=].*? =', text.strip()):
            if start is not None:
                document = ''.join(pieces)
                ids = tokenizer(document, add_special_tokens=True, truncation=False, padding=False)['input_ids']
                if len(ids) >= 3072:
                    rid = 'memory-train-article-%07d' % start; prefix = ids[:128]; dh = original.sha(document.encode())
                    record = dict(request_id=rid, document_id=rid, sample_id=start, eligible_article_ordinal_1based=len(eligible)+1,
                        dataset_row_index=start, dataset_row_end_exclusive=index, article_title=pieces[0].strip(), prompt=document,
                        document_sha256=dh, prompt_sha256=dh, original_document_token_count=len(ids), prompt_token_count=128,
                        prompt_token_ids_sha256=original.token_sha(prefix))
                    eligible.append((record, prefix))
                    if len(eligible) == 128: break
            start, pieces = index, []
        if start is not None: pieces.append(text)
    assert len(eligible) == 128
    prior_paths = [OLD / 'prepared/short/workload.json', POOL / 'workload.json',
                   C / 'prepared/cohort0/workload.json', C / 'prepared/cohort1/workload.json', P / 'prepared/workload.json']
    earlier = [(r,p) for path in prior_paths for w in [read(path)] for r,p in zip(w['source_requests'],w['actual_prompt_token_ids'])]
    assert len(earlier) == 112
    for (r,p),(old,oldp) in zip(eligible[:112], earlier):
        assert p == oldp and all(r[k] == old[k] for k in ('document_id','prompt','document_sha256','dataset_row_index','dataset_row_end_exclusive'))
    chosen = eligible[112:]; records, prompts = zip(*chosen)
    ids = {r['document_id'] for r in records}; hashes = {r['document_sha256'] for r in records}
    assert len(ids) == len(hashes) == len({original.token_sha(p) for p in prompts}) == 16
    references = [ROOT / x['path'] for x in previous['prior_inputs']] + prior_paths[2:] + [B / 'logical_alignment_qualification_r01/prepared/workload.json']
    checks = []
    for path in dict.fromkeys(references):
        old = read(path); used, oldp = old['source_requests'], old['actual_prompt_token_ids']
        overlap = dict(document_id=len(ids & {r['document_id'] for r in used}),
            document_hash=len(hashes & {r.get('document_sha256',r.get('prompt_sha256')) for r in used}),
            shared_prefix=sum(p[:min(len(p),len(q))] == q[:min(len(p),len(q))] for p in prompts for q in oldp))
        assert oldp and all(oldp) and not any(overlap.values()), (str(path),overlap)
        checks.append(dict(path=str(path.relative_to(ROOT)),sha256=original.file_sha(path),requests=len(used),overlap=overlap))
    scope = 'Second independent B-line logical-alignment cohort, distinct from the named prior inputs only; not a claim of global non-use by other research sessions. Two engines per arm are not a statistical noise bound.'
    workload = dict(schema='olmoe-admission-inputs-v1', source_requests=list(records), actual_prompt_token_ids=list(prompts),
        arrival_traces_s={'steady':[i*.25 for i in range(16)]}, arrival_rule='request i arrives at i*0.25 seconds; all16 retained',document_identity_scope=scope)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('x') as f:json.dump(workload,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    provenance = dict(status='INPUTS_PREPARED_GPU_UNRUN',selection_rule='Reconstruct eligible1..112 exactly; use113..128 in source order, complete source articles>=3072 tokens, actual first128 only.',
        dataset=previous['dataset'],tokenizer=dict(model,files=files),original_first112_exact_reconstruction=True,prior_inputs=checks,
        workload_sha256=original.file_sha(destination),eligible_ordinals=[113,128],warmup_request_id=records[0]['request_id'],warmup_output_tokens=2,
        selected_articles=[{k:v for k,v in r.items() if k!='prompt'} for r in records],scope=scope,
        preparation_environment=dict(python=sys.version,datasets=datasets.__version__,transformers=transformers.__version__,tokenizers=tokenizers.__version__),
        source_sha256={str(p.relative_to(ROOT)):original.file_sha(p) for p in [Path(__file__),P/'prepare_inputs.py',C/'prepare_inputs.py',POOL/'prepare_pool.py',OLD/'prepare_inputs.py']})
    with (HERE/'input_provenance.json').open('x') as f:json.dump(provenance,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=provenance['status'],requests=16,eligible_ordinals=[113,128],prior_input_files=len(checks),all_overlaps=0,workload_sha256=provenance['workload_sha256'])))


if __name__ == '__main__': main()
