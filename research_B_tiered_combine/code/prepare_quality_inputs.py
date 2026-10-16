"""Group WikiText raw lines into whole articles before choosing calibration/test."""
import argparse,hashlib,json,pathlib,re
import sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/".deps"))
import pyarrow.parquet as pq


def articles(lines):
    result=[];current=[];start=0;title=None
    for i,line in enumerate(lines):
        if re.match(r'^\s*= [^=].*[^=] =\s*$',line):
            if current and title is not None:
                result.append({'title':title,'first_row':start,'end_row':i,'text':''.join(current)})
            current=[line];start=i;title=line.strip()
        elif title is not None:current.append(line)
    if current and title is not None:
        result.append({'title':title,'first_row':start,'end_row':len(lines),'text':''.join(current)})
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--validation',required=True);p.add_argument('--test',required=True)
    p.add_argument('--output',required=True);p.add_argument('--calibration-docs',type=int,default=16);p.add_argument('--test-docs',type=int,default=32);a=p.parse_args()
    out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
    manifest={'source':'Salesforce/wikitext wikitext-2-raw-v1','unit':'article','selection':'first N articles with >=80 whitespace-separated words, before any candidate runs','splits':{}}
    for split,path,n in [('validation',a.validation,a.calibration_docs),('test',a.test,a.test_docs)]:
        data=pathlib.Path(path).read_bytes();lines=pq.read_table(path,columns=['text'])['text'].to_pylist()
        docs=[d for d in articles(lines) if len(d['text'].split())>=80]
        if len(docs)<n:raise ValueError(f'{split}: only {len(docs)} eligible articles for {n}')
        selected=[]
        for i,d in enumerate(docs[:n]):
            d.update({'id':f'{split}_{i:03d}','split':split,'text_sha256':hashlib.sha256(d['text'].encode()).hexdigest()});selected.append(d)
        (out/f'{split}.jsonl').write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in selected))
        manifest['splits'][split]={'parquet_sha256':hashlib.sha256(data).hexdigest(),'raw_rows':len(lines),'eligible_articles':len(docs),'selected':n}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))

if __name__=='__main__':main()
