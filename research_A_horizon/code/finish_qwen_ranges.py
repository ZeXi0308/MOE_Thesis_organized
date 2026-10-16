"""Finish verified mirror objects with four disjoint HTTP ranges at a time.

The prior downloader must be stopped and the asset continuation suspended.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time
import requests


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--asset-pid',type=int,required=True)
    args=p.parse_args();root=args.root
    receipt=root/'modelscope-download.json'
    status=json.loads(receipt.read_text())
    status.update(status='RUNNING',range_pid=os.getpid(),range_start_unix_s=time.time())
    receipt.write_text(json.dumps(status,indent=2)+'\n')
    blobs=root/'hf/models--Qwen--Qwen3-8B/blobs'
    try:
        for row in status['objects']:
            target=blobs/row['sha256'];partial=blobs/(row['sha256']+'.incomplete')
            if target.exists():
                assert target.stat().st_size==row['size'] and digest(target)==row['sha256']
                continue
            offset=partial.stat().st_size if partial.exists() else 0
            assert 0<=offset<=row['size']
            parts=blobs/(row['sha256']+'.range-parts');parts.mkdir(exist_ok=False)
            url='https://www.modelscope.cn/models/Qwen/Qwen3-8B/resolve/'+row['revision']+'/'+row['name']
            ranges=[(start,min(start+64*1024**2,row['size'])-1)
                    for start in range(offset,row['size'],64*1024**2)]
            def fetch(bounds):
                start,end=bounds;out=parts/str(start)
                for attempt in range(3):
                    try:
                        with requests.get(url,headers={'Range':f'bytes={start}-{end}'},
                                          stream=True,timeout=(10,60)) as response:
                            response.raise_for_status()
                            assert response.status_code==206
                            assert response.headers['Content-Range']==f'bytes {start}-{end}/{row["size"]}'
                            size=0
                            with out.open('wb') as f:
                                for chunk in response.iter_content(1024**2):
                                    size+=len(chunk)
                                    if size>end-start+1:raise ValueError('Oversized range')
                                    f.write(chunk)
                            if size!=end-start+1:raise IOError('Truncated range')
                        print(json.dumps(dict(file=row['name'],range_start=start,bytes=size)),flush=True)
                        return out
                    except (requests.RequestException,IOError):
                        if attempt==2:raise
            with ThreadPoolExecutor(max_workers=4) as pool:
                files=list(pool.map(fetch,ranges))
            assert (partial.stat().st_size if partial.exists() else 0)==offset
            with partial.open('ab') as dest:
                for file in files:
                    with file.open('rb') as source:shutil.copyfileobj(source,dest,8*1024**2)
            assert partial.stat().st_size==row['size'] and digest(partial)==row['sha256']
            partial.rename(target)
            shutil.rmtree(parts)
        status.update(status='COMPLETE',verified=[r['name'] for r in status['objects']])
    except BaseException as error:
        status.update(status='ERROR',error=type(error).__name__+': '+str(error).split('https://')[0])
        raise
    finally:
        status['end_unix_s']=time.time();receipt.write_text(json.dumps(status,indent=2)+'\n')
        command=Path('/proc',str(args.asset_pid),'cmdline')
        if command.exists() and b'finish_qwen_assets.py' in command.read_bytes():
            os.kill(args.asset_pid,signal.SIGCONT)


if __name__=='__main__':main()
