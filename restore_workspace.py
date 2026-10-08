#!/usr/bin/env python3
"""Restore and verify files represented by WORKSPACE_MANIFEST.json."""
import argparse,gzip,hashlib,json,os,pathlib,shutil,tempfile

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--verify-only',action='store_true',help='Verify without writing restored files')
 parser.add_argument('--prefix',default='',help='Only process matching original paths')
 args=parser.parse_args(); root=pathlib.Path(__file__).resolve().parent
 manifest=json.loads((root/'WORKSPACE_MANIFEST.json').read_text())
 count=0
 for item in manifest['files']:
  if not item['path'].startswith(args.prefix):continue
  relative=pathlib.PurePosixPath(item['path'])
  if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe manifest path')
  target=root/relative
  if 'symlink' in item:
   if not target.is_symlink() or os.readlink(target)!=item['symlink']:raise ValueError(f'Symlink mismatch: {relative}')
   continue
  parts=item.get('gzip_parts')
  if parts:
   with tempfile.TemporaryFile() as packed:
    for part in parts:
     with (root/part).open('rb') as fi:shutil.copyfileobj(fi,packed)
    packed.seek(0);h=hashlib.sha256();size=0
    temp=None
    if not args.verify_only:
     target.parent.mkdir(parents=True,exist_ok=True)
     if target.exists():
      with target.open('rb') as fi:existing=hashlib.file_digest(fi,'sha256').hexdigest()
      if existing!=item['sha256']:raise FileExistsError(f'Refusing to replace different file: {relative}')
     temp=tempfile.NamedTemporaryFile(dir=target.parent,delete=False)
    try:
     with gzip.GzipFile(fileobj=packed) as fi:
      while b:=fi.read(1024*1024):
       h.update(b);size+=len(b)
       if temp:temp.write(b)
     if h.hexdigest()!=item['sha256'] or size!=item['bytes']:raise ValueError(f'Checksum mismatch: {relative}')
     if temp:
      temp.close();os.chmod(temp.name,item['mode']);os.replace(temp.name,target)
    finally:
     if temp:
      temp.close()
      if os.path.exists(temp.name):os.unlink(temp.name)
  else:
   with target.open('rb') as fi:digest=hashlib.file_digest(fi,'sha256').hexdigest()
   if digest!=item['sha256']:raise ValueError(f'Checksum mismatch: {relative}')
  count+=1
  if count%1000==0:print(f'Verified {count} files',flush=True)
 print(f'OK: {count} files verified',flush=True)
if __name__=='__main__':main()
