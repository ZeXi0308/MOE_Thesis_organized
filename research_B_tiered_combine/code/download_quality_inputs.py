"""Download pinned public WikiText parquet inputs, preserving raw bytes locally."""
import argparse,hashlib,json,pathlib,time,urllib.request
REV='b08601e04326c79dfdd32d625aee71d232d685c3'
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=False)
manifest={'repo':'Salesforce/wikitext','revision':REV,'files':{},'errors':[]}
for split in ['validation','test']:
    rel=f'wikitext-2-raw-v1/{split}-00000-of-00001.parquet'
    for host in ['huggingface.co','hf-mirror.com']:
        url=f'https://{host}/datasets/Salesforce/wikitext/resolve/{REV}/{rel}'
        try:
            with urllib.request.urlopen(url,timeout=20) as r: data=r.read(10_000_000)
            if not data.startswith(b'PAR1') or not data.endswith(b'PAR1'):raise ValueError('not complete parquet')
            (out/f'{split}.parquet').write_bytes(data)
            manifest['files'][split]={'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
            print(split,len(data),flush=True);break
        except Exception as e:
            manifest['errors'].append({'url':url,'error':str(e)});print(host,split,type(e).__name__,str(e),flush=True)
    (out/'download.json').write_text(json.dumps(manifest,indent=2))
if len(manifest['files'])!=2:raise SystemExit('Download incomplete')
print('COMPLETE',flush=True)
