"""Offline eligible articles 65..96; two cohorts, no GPU or outcome selection."""
import argparse
import json
import os
from pathlib import Path
import re
import runpy
import sys

HERE = Path(__file__).resolve().parent
B = HERE.parent
POOL = B / 'new_host_20260913/finite_arrival_input_pool'
legacy = runpy.run_path(str(POOL / 'prepare_pool.py'))
original, ROOT, OLD = legacy['original'], legacy['ROOT'], legacy['OLD']
read = lambda p: json.loads(p.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-arrow', type=Path)
    args = parser.parse_args()
    destinations = [HERE / f'prepared/cohort{i}/workload.json' for i in range(2)]
    assert not any(p.exists() for p in destinations + [HERE / 'input_provenance.json'])
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', CUDA_VISIBLE_DEVICES='', TOKENIZERS_PARALLELISM='false')
    import datasets, transformers, tokenizers
    from datasets import Dataset
    from transformers import AutoTokenizer
    from transformers.utils.hub import cached_file
    manifest = read(POOL / 'manifest.json'); model = manifest['tokenizer']
    assert original.file_sha(POOL / 'prepare_pool.py') == manifest['script_sha256']
    assert original.file_sha(OLD / 'prepare_inputs.py') == manifest['original_preparation']['sha256']
    arrow = (args.dataset_arrow or Path(manifest['dataset']['arrow_path'])).resolve()
    assert arrow.is_file() and arrow.name == 'wikitext-train-00000-of-00002.arrow'
    assert arrow.parent.name == original.DATASET_REVISION and arrow.parents[2].name == 'wikitext-103-raw-v1'
    assert original.file_sha(arrow) == manifest['dataset']['arrow_sha256']
    files = {}
    for name, expected in model['files'].items():
        p = Path(cached_file(model['model_id'], name, revision=original.REVISION, local_files_only=True))
        assert original.file_sha(p) == expected['sha256']; files[name] = dict(path=str(p), sha256=expected['sha256'])
    assert model['revision'] == original.REVISION
    tokenizer = AutoTokenizer.from_pretrained(model['model_id'], revision=original.REVISION, local_files_only=True)
    assert tokenizer.__class__.__name__ == model['tokenizer_class']
    dataset = Dataset.from_file(str(arrow)); eligible, start, pieces = [], None, []
    for index, row in enumerate(dataset):
        text = str(row['text'])
        if re.fullmatch(r'= [^=].*? =', text.strip()):
            if start is not None:
                document = ''.join(pieces); ids = tokenizer(document, add_special_tokens=True, truncation=False, padding=False)['input_ids']
                if len(ids) >= 3072:
                    rid = 'memory-train-article-%07d' % start; prefix = ids[:128]; dh = original.sha(document.encode())
                    record = dict(request_id=rid, document_id=rid, sample_id=start, eligible_article_ordinal_1based=len(eligible)+1,
                        dataset_row_index=start, dataset_row_end_exclusive=index, article_title=pieces[0].strip(), prompt=document,
                        document_sha256=dh, prompt_sha256=dh, original_document_token_count=len(ids), prompt_token_count=128,
                        prompt_token_ids_sha256=original.token_sha(prefix))
                    eligible.append((record, prefix))
                    if len(eligible) == 96: break
            start, pieces = index, []
        if start is not None: pieces.append(text)
    assert len(eligible) == 96  # The unterminated shard tail is never admitted.
    priors = [OLD / 'prepared/short/workload.json', POOL / 'workload.json']
    previous = [(r,p) for path in priors for w in [read(path)] for r,p in zip(w['source_requests'],w['actual_prompt_token_ids'])]
    assert len(previous) == 64 and original.file_sha(POOL / 'workload.json') == manifest['workload_sha256']
    for (record,prefix),(old,old_prefix) in zip(eligible[:64],previous):
        assert prefix == old_prefix and all(record[k] == old[k] for k in ['document_id','dataset_row_index','dataset_row_end_exclusive','prompt','document_sha256'])
    chosen = eligible[64:]; records, prompts = zip(*chosen)
    identifiers = {r['document_id'] for r in records}; hashes = {r['document_sha256'] for r in records}
    prefix_hashes = {original.token_sha(p) for p in prompts}
    assert len(identifiers) == len(hashes) == len(prefix_hashes) == 32
    references = priors + [B / x['path'] for x in manifest['mechanism_prepared_inputs']]
    references += [B / x / 'prepared/workload.json' for x in ['finite_arrival_r01','jit_localization_r01','compile_domain_r01','compile_domain_r02','host_cost_r01']]
    checks = []
    for path in references:
        old = read(path); used, old_prompts = old['source_requests'], old['actual_prompt_token_ids']
        overlap = dict(document_id=len(identifiers & {r['document_id'] for r in used}),
            document_hash=len(hashes & {r.get('document_sha256',r.get('prompt_sha256')) for r in used}),
            prompt_hash=len(prefix_hashes & {original.token_sha(p) for p in old_prompts}),
            shared_prefix=sum(p[:min(len(p),len(q))] == q[:min(len(p),len(q))] for p in prompts for q in old_prompts))
        assert old_prompts and all(old_prompts) and not any(overlap.values()), (str(path),overlap)
        checks.append(dict(path=str(path.relative_to(ROOT)), sha256=original.file_sha(path), requests=len(used), overlap=overlap))
    scope = '32 distinct source-ordered documents form only two cohort experiment units; repeats/documents are not IID GPU samples. Not paper-level confirmation; no outcome selection.'
    outputs = []
    for i,path in enumerate(destinations):
        subset = chosen[i*16:(i+1)*16]
        workload = dict(schema='olmoe-admission-inputs-v1', source_requests=[r for r,p in subset], actual_prompt_token_ids=[p for r,p in subset],
            arrival_traces_s={'steady':[j*0.25 for j in range(16)]}, arrival_rule='request i arrives at i*0.25 seconds; all 16 preserved', document_identity_scope=scope)
        payload = (json.dumps(workload,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
        outputs.append((path,payload,dict(cohort=i,path=str(path.relative_to(HERE)),sha256=original.sha(payload),requests=16,
            eligible_ordinals=[65+i*16,80+i*16],warmup_request_id=subset[0][0]['request_id'],warmup_output_tokens=2)))
    provenance = dict(status='INPUTS_PREPARED_GPU_UNRUN', selection_rule='Reproduce original eligible1..64 exactly; choose65..96 in source order, complete articles>=3072 pinned-tokenizer tokens; retain first128 tokens.',
        reconstruction_rule=manifest['reconstruction_rule'],dataset=dict(manifest['dataset'],arrow_path=str(arrow),fingerprint=str(dataset._fingerprint)),tokenizer=dict(model,files=files),
        original_first64_exact_reconstruction=True,comparison_scope='Only named mechanism pools/qualification/prepared files; shared_prefix compares first min(new128, old prompt length) tokens.',
        prior_inputs=checks,cohorts=[o[2] for o in outputs],cohort_intersection=dict(document_id=0,document_hash=0,prompt_hash=0),
        selected_articles=[{k:v for k,v in r.items() if k!='prompt'} for r in records],identity_scope=scope,
        preparation_environment=dict(python=sys.version,executable=sys.executable,datasets=datasets.__version__,transformers=transformers.__version__,tokenizers=tokenizers.__version__),
        source_sha256={str(p.relative_to(ROOT)):original.file_sha(p) for p in [Path(__file__),POOL/'prepare_pool.py',POOL/'manifest.json',OLD/'prepare_inputs.py']},
        command='.venv/bin/python '+str(Path(__file__).relative_to(ROOT)))
    for path,payload,_ in outputs:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream: stream.write(payload)
    with (HERE/'input_provenance.json').open('x') as stream: json.dump(provenance,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=provenance['status'],cohorts=provenance['cohorts'],prior_input_files=len(checks),all_overlaps=0)))


if __name__ == '__main__': main()
