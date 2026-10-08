"""Extract own data-only research archive, then verify optional lossless logs."""
import argparse,hashlib,json,lzma,tarfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('destination',type=Path);a=p.parse_args()
a.destination.mkdir()
with tarfile.open(a.archive) as t:
 for m in t.getmembers():
  if Path(m.name).is_absolute() or '..' in Path(m.name).parts or not (m.isfile() or m.isdir()) or m.issym() or m.islnk():raise ValueError('unsafe archive member: '+m.name)
 t.extractall(a.destination)
verified=[]
for r in a.destination.rglob('compression-receipt.json'):
 receipt=json.loads(r.read_text());assert receipt['status']=='COMPLETE'
 for item in receipt['files']:
  n=item['source'];assert n in ('measured-pressure.json','measured-host-returns.jsonl','measured-steps.json')
  assert item['archive']==n+'.xz' and item['status']=='COMPRESSED_VERIFIED'
  packed=r.parent/'native'/item['archive'];raw=r.parent/'native'/n
  assert hashlib.sha256(packed.read_bytes()).hexdigest()==item['archive_sha256']
  h,size=hashlib.sha256(),0
  with lzma.open(packed,'rb') as src,raw.open('xb') as dst:
   while True:
    chunk=src.read(1024*1024)
    if not chunk:break
    size+=len(chunk);assert size<=item['source_bytes'];h.update(chunk);dst.write(chunk)
  assert size==item['source_bytes']==item['decoded_bytes'] and h.hexdigest()==item['source_sha256']==item['decoded_sha256']
  verified.append(str(raw))
print(json.dumps(dict(archive_bytes=a.archive.stat().st_size,archive_sha256=hashlib.sha256(a.archive.read_bytes()).hexdigest(),verified_restored_files=verified)))
