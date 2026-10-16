"""Post-hoc description of fixed prompt-length classes; no causal matching."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from analyze import distribution


def main():
    session=ROOT/'runs/w2-48-ablations-r10'
    source=session/'metrics.json'
    rows=json.loads(source.read_text());assert len(rows)==8 and all(r['complete'] for r in rows)
    output=[];identity=None;hashes={}
    for row in rows:
        name=Path(row['cell']).name
        path=session/name/'raw.json';raw=json.loads(path.read_text())
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        by_id={r['request_id']:r for r in raw['requests']}
        current=sorted((r['request_id'],r['document_id'],r['prompt_tokens'],
                        r['prompt_token_ids_sha256'],r['arrival_s']) for r in raw['requests'])
        if identity is None:identity=current
        assert current==identity,'External request population changed'
        assert len(by_id)==len(row['requests'])==256
        assert {r['request_id'] for r in row['requests']}==set(by_id)
        stalled=emitted=0
        for length in (512,3072):
            population=[r for r in row['requests'] if by_id[r['request_id']]['prompt_tokens']==length]
            assert len(population)==128 and all(r['status']=='completed' for r in population)
            gaps=distribution([r['maxgap_s'] for r in population])
            ttft=distribution([r['ttft_s'] for r in population])
            flow=distribution([r['flow_s'] for r in population])
            count=sum(r['maxgap_s']>.2 for r in population)
            tokens=sum(r['output_tokens'] for r in population)
            stalled+=count;emitted+=tokens
            output.append(dict(cell=name,prompt_tokens=length,requests=len(population),
                stalled=count,stall_percent=100*count/len(population),
                gap_p99_s=gaps['p99'],ttft_p95_s=ttft['p95'],flow_mean_s=flow['mean'],
                output_tokens=tokens,eos=sum(r['finish_reason']=='stop' for r in population),
                length_limited=sum(r['finish_reason']=='length' for r in population)))
        assert stalled==row['stall_count'] and emitted==row['delivered_output_tokens']
    out=Path(__file__).resolve().parent
    (out/'request-classes.json').write_text(json.dumps(dict(rows=output,
        metrics_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),raw_sha256=hashes,
        semantics='Post-hoc descriptive split by pre-existing prompt length, 128 requests per class. '
        'Identical request/document/prompt/arrival identity across all eight arms. '
        'Generated outputs may differ; no quality or workload equivalence is inferred. '
        'Requests share one run per arm and are not independent system repetitions.'),indent=2)+'\n')
    with (out/'request-classes.tsv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0]),delimiter='\t')
        writer.writeheader();writer.writerows(output)
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
