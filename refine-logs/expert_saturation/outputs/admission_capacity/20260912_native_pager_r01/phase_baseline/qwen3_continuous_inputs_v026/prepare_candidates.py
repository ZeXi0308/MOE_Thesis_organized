"""Offline article/prefix candidates only; no arrival or policy protocol."""
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path

os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
from datasets import Dataset
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'refine-logs/expert_saturation/experiments').is_dir())
O = ROOT / 'refine-logs/expert_saturation/outputs/admission_capacity'
E = ROOT / 'refine-logs/expert_saturation/experiments/admission_capacity'
spec = importlib.util.spec_from_file_location('prior_inputs', E / 'prepare_rotation_holdout.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
sha, file_sha, token_sha = prior.sha, prior.file_sha, prior.token_sha
base = O / '20260908_kv_budget_r01/inputs_preparation/prepared/short'
holdout = O / '20260913_rotation_holdout_r01/inputs'
report = json.loads((holdout / 'inputs_report.json').read_text())
arrow = Path(report['local_assets']['dataset_arrow'])
assert file_sha(arrow) == report['provenance']['arrow_sha256']
excluded_paths = [base / 'workload.json', *(holdout / f'cohort{i}/workload.json' for i in range(2))]
excluded = [r for p in excluded_paths for r in json.loads(p.read_text())['source_requests']]
assert len(excluded) == len({r['document_sha256'] for r in excluded}) == 96
assert all(sha(r['prompt'].encode()) == r['document_sha256'] for r in excluded)
cutoff = max(r['dataset_row_end_exclusive'] for r in excluded)
used = {r['document_sha256'] for r in excluded}
qwen = HERE.parent / 'qwen3_new_gpu_20260913'
metadata = json.loads((qwen / 'metadata_manifest.json').read_text())
for name, entry in metadata['files'].items():
    assert file_sha(qwen / 'model_metadata' / name) == entry['sha256'], name
tok = AutoTokenizer.from_pretrained(qwen / 'model_metadata', local_files_only=True, trust_remote_code=False)
for name in ['prepared/workload.json', 'qualification_prepared/workload.json']:
    old = json.loads((qwen / name).read_text())
    assert all(tok.encode(r['prompt'], add_special_tokens=False)[:len(ids)] == ids
               for r, ids in zip(old['source_requests'], old['actual_prompt_token_ids']))
rows, variants = [], {str(n): [] for n in [128, 256, 512]}
for start, stop, title, document in prior.articles(Dataset.from_file(str(arrow))):
    if start < cutoff or sha(document.encode()) in used:
        continue
    ids = tok.encode(document, add_special_tokens=False)
    if len(ids) < 512:
        continue
    digest = sha(document.encode()); used.add(digest)
    rows.append(dict(request_id=f'qwen-candidate-article-{start:07d}',
        document_id=f'memory-train-article-{start:07d}', dataset_row_index=start,
        dataset_row_end_exclusive=stop, article_title=title, prompt=document,
        document_sha256=digest, prompt_sha256=digest, full_document_token_count=len(ids),
        full_document_token_ids_sha256=token_sha(ids),
        prefix_token_ids_sha256={n: token_sha(ids[:int(n)]) for n in variants}))
    for n, values in variants.items():
        values.append(ids[:int(n)])
    if len(rows) == 16:
        break
assert len(rows) == 16 and len({r['document_sha256'] for r in rows}) == 16
assert all(a['dataset_row_end_exclusive'] <= b['dataset_row_index'] for a, b in zip(rows, rows[1:]))
model = json.loads((qwen / 'model_metadata/config.json').read_text())
block_tokens, maxseq, output_cap, token_budget = 16, 8, 64, 64
block_bytes = block_tokens * 2 * model['num_hidden_layers'] * model['num_key_value_heads'] * model['head_dim'] * 2
blocks = (512 * 2**20) // block_bytes
assert block_bytes == 1572864 and blocks == 341 and 512 + output_cap <= model['max_position_embeddings']
capacity = {n: dict(per_request_blocks=(int(n)+output_cap+15)//16,
    maxseq8_request_blocks=8*((int(n)+output_cap+15)//16),
    maxseq8_request_kv_bytes=8*((int(n)+output_cap+15)//16)*block_bytes,
    remaining_usable_blocks=blocks-1-8*((int(n)+output_cap+15)//16)) for n in variants}
assert all(v['remaining_usable_blocks'] >= 0 for v in capacity.values())
scope = ('PREPARED_UNRUN: 16 documents, each with three prefix candidates, not 48 independent samples. '
    'Excludes checked paging-chain original32 and rotation64 document intervals only; not globally untouched. '
    'No arrival, threshold, strategy, run order or measured output length frozen. Output cap64 is only a '
    'static capacity estimate; no dynamic feasibility, method validation, generalization or quality claim.')
identity = dict(repository=metadata['repository'], revision=metadata['revision'], tokenizer_class=type(tok).__name__,
    add_special_tokens=False, apply_chat_template=False, packages={n: importlib.metadata.version(n)
    for n in ['transformers', 'tokenizers', 'datasets']}, files_sha256={n: v['sha256'] for n,v in metadata['files'].items()})
workload = dict(status='PREPARED_UNRUN', scope=scope, tokenizer_identity=identity, source_requests=rows,
    actual_prompt_token_ids_by_length=variants, output_tokens_cap_for_capacity_estimate=64)
out = HERE / 'workload_candidates.json'
with out.open('x') as f:
    f.write(json.dumps(workload, indent=2, ensure_ascii=False) + '\n')
checks = dict(status='PASS_CPU_PREPARATION_ONLY', scope=scope, workload_sha256=file_sha(out),
    script_sha256=file_sha(Path(__file__)), reused_generator=str(E / 'prepare_rotation_holdout.py'),
    reused_generator_sha256=file_sha(E / 'prepare_rotation_holdout.py'), source_arrow=str(arrow),
    source_provenance={k: report['provenance'][k] for k in ['dataset_id','dataset_config','dataset_revision','split','arrow_sha256']},
    excluded_input_sha256={str(p): file_sha(p) for p in excluded_paths}, excluded_documents=96,
    excluded_row_prefix_end_exclusive=cutoff, selected_rows=[r['dataset_row_index'] for r in rows],
    prior_qwen_prepared_prefixes_reproduced=8, document_hash_and_interval_exclusion=True,
    kv_static_upper_bound=dict(block_tokens=16, block_bytes=block_bytes, requested_bytes=512*2**20,
        allocated_blocks=blocks, allocated_bytes=blocks*block_bytes, null_blocks=1, usable_blocks=blocks-1,
        assumed_maxseq=maxseq, token_budget=token_budget, output_cap=output_cap, by_prompt_length=capacity,
        assumptions='BF16 KV, no speculative/lookahead slots, no shared-prefix credit; fully materialized prompt+64 per running request. No GPU execution or allocator/workspace feasibility test.'))
with (HERE / 'input_checks.json').open('x') as f:
    f.write(json.dumps(checks, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({k: checks[k] for k in ['status','workload_sha256','selected_rows','kv_static_upper_bound']}))
