"""Resume byte-identical Qwen LFS objects from ModelScope into the private HF cache.

Run only after the previous downloader has exited. The pinned HF metadata is
the content authority; the alternate source must advertise matching hashes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import requests


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root
    hf=json.loads((root/'qwen3-hf-weights.json').read_text())
    ms={f['Name']:f for f in json.loads((root/'qwen3-modelscope-source.json').read_text())['Data']['Files']}
    objects=[]
    for row in hf:
        other=ms.get(row['name'])
        same=other and other.get('Sha256')==row['sha256'] and other.get('Size')==row['size']
        if row['name'].endswith('.safetensors'):assert same, row['name']
        if same:objects.append(dict(row,revision=other['Revision']))
    blobs=root/'hf/models--Qwen--Qwen3-8B/blobs';blobs.mkdir(parents=True,exist_ok=True)
    receipt=root/'modelscope-download.json'
    status=dict(status='RUNNING',start_unix_s=time.time(),objects=objects)
    receipt.write_text(json.dumps(status,indent=2)+'\n')

    def fetch(row):
        target=blobs/row['sha256'];partial=blobs/(row['sha256']+'.incomplete')
        if target.exists():
            assert target.stat().st_size==row['size'] and digest(target)==row['sha256']
            return row['name']
        url='https://www.modelscope.cn/models/Qwen/Qwen3-8B/resolve/'+row['revision']+'/'+row['name']
        offset=partial.stat().st_size if partial.exists() else 0
        assert offset<=row['size']
        failures=0
        while offset<row['size']:
            last=min(row['size']-1,offset+64*1024**2-1)
            try:
                with requests.get(url,headers={'Range':f'bytes={offset}-{last}'},
                                  stream=True,timeout=(10,60)) as response:
                    response.raise_for_status()
                    expected=f'bytes {offset}-{last}/{row["size"]}'
                    if response.status_code!=206 or response.headers.get('Content-Range')!=expected:
                        raise ValueError('Unexpected range response for '+row['name'])
                    with partial.open('ab') as out:
                        for chunk in response.iter_content(1024**2):
                            if offset+len(chunk)>last+1:raise ValueError('Oversized response')
                            out.write(chunk);offset+=len(chunk)
                if offset!=last+1:raise IOError('Truncated range')
                failures=0
            except (requests.RequestException,IOError):
                failures+=1
                if failures>=3:raise
                offset=partial.stat().st_size
            print(json.dumps(dict(file=row['name'],bytes=offset,total=row['size'])),flush=True)
        assert partial.stat().st_size==row['size'] and digest(partial)==row['sha256'], row['name']
        partial.rename(target)
        return row['name']

    try:
        with ThreadPoolExecutor(max_workers=3) as pool:status['verified']=list(pool.map(fetch,objects))
        status['status']='COMPLETE'
    except BaseException as error:
        status.update(status='ERROR',error=type(error).__name__+': '+str(error).split('https://')[0])
        raise
    finally:
        status['end_unix_s']=time.time();receipt.write_text(json.dumps(status,indent=2)+'\n')


if __name__=='__main__':main()
