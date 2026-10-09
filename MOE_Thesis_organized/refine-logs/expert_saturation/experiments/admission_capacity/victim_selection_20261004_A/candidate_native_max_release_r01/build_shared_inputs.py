"""Fixed duplicated-article sharing diagnostic; input identities only, no outcomes."""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'candidate_native_mixed_budget_probe_r01/pkg/inputs/pro_high'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build(check=False):
    config = json.loads((SOURCE / 'config.json').read_text())
    workload = json.loads((SOURCE / 'workload.json').read_text())
    rows, prompts = workload['source_requests'], workload['actual_prompt_token_ids']
    assert len(rows) == 320 and len({r['request_id'] for r in rows}) == 320
    strata = defaultdict(list)
    for i, row in enumerate(rows):
        strata[row['prompt_length_stratum']].append(i)
    assert len(strata) == 64 and all(len(v) == 5 for v in strata.values())
    pairs = []
    for stratum, indices in sorted(strata.items()):
        first, second = indices[:2]
        original = rows[second]
        replacement = copy.deepcopy(rows[first])
        replacement.update(request_id=original['request_id'], source_index=second,
                           max_output_tokens=original['max_output_tokens'],
                           shared_prefix_source_request_id=rows[first]['request_id'],
                           replaced_source_document_id=original['document_id'])
        rows[first]['shared_prefix_source_request_id'] = rows[first]['request_id']
        rows[second] = replacement
        prompts[second] = list(prompts[first])
        pairs.append(dict(stratum=stratum, first_index=first, second_index=second,
                          first_request=rows[first]['request_id'], second_request=original['request_id'],
                          prompt_token_count=len(prompts[first])))
    metadata = dict(kind='SYNTHETIC_SHARED_ARTICLE_OPPORTUNITY_DIAGNOSTIC',
        rule='In each of the existing 64 prompt-length strata, copy the first request article to the second request; preserve all request IDs, arrival order/times, and assigned output budgets.',
        pairs=pairs, pair_count=64, paired_requests=128, unpaired_requests=192,
        unique_documents=256, no_outcome_information_used=True,
        source_workload_file_sha256=digest((SOURCE/'workload.json').read_bytes()),
        semantics='Repeated full natural article inputs, not 320 unique documents or independent tasks. Prefix reuse is native and not guaranteed before admission. This is a synthetic mechanism diagnostic, not production-quality evidence or a policy gain comparison with old inputs.')
    workload['shared_prefix_diagnostic'] = metadata
    lengths = list(map(len, prompts))
    config.update(enable_prefix_caching=True, shared_prefix_diagnostic=metadata,
        prompt_tokens=max(lengths), prompt_tokens_by_request=lengths,
        workload_sha256=digest(json.dumps(workload, sort_keys=True).encode()),
        status='SHARED_PREFIX_TAIL_DIAGNOSTIC_CPU_PREPARED_GPU_UNRUN',
        input_preparation=metadata['rule'], input_independence=metadata['semantics'],
        prompt_tokens_semantics='Untruncated natural articles; 64 input copies create 64 pairs and 192 singletons.')
    config['source']['shared_prefix_parent_selection'] = config['source']['selection']
    config['source']['selection'] = metadata['rule']
    stats = dict(requests=320, unique_request_ids=320, unique_documents=256,
        paired_requests=128, unpaired_requests=192, prompt_tokens_total=sum(lengths),
        prompt_tokens_min=min(lengths), prompt_tokens_max=max(lengths),
        prompt_only_logical_gib=sum((n+15)//16 for n in lengths)/512,
        ideal_shared_prompt_physical_gib=(sum((n+15)//16 for n in lengths)
            - sum(len(prompts[p['second_index']])//16 for p in pairs))/512,
        bound_semantics='Input page-count diagnostic; actual prefix hits, sharing, residency, output quantity and memory pressure are unknown until execution.',
        output_cap_counts=dict(Counter(config['output_tokens_by_request'].get(r['request_id'],1024) for r in rows)),
        workload_sha256=config['workload_sha256'], shared_prefix_diagnostic=metadata)
    for row, ids in zip(rows,prompts):
        assert row['prompt_token_count'] == len(ids)
        assert row['prompt_token_ids_sha256'] == digest(json.dumps(ids,separators=(',',':')).encode())
        assert row['prompt_sha256'] == digest(row['prompt'].encode())
        assert len(ids)+config['output_tokens_by_request'].get(row['request_id'],1024) <= 4096
    target=ROOT/'pkg/inputs/pro_high'
    for name,data in [('config.json',config),('workload.json',workload),('stats.json',stats)]:
        content=json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
        path=target/name
        if check:
            assert path.read_text()==content, f'Input drift: {path}'
        elif path.read_bytes() == (SOURCE/name).read_bytes() or path.read_text()==content:
            path.write_text(content)
        else:
            raise ValueError(f'Refusing to overwrite non-parent/non-derived input {path}')
    return {k:v for k,v in stats.items() if k!='shared_prefix_diagnostic'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    print(json.dumps(build(args.check),indent=2))
