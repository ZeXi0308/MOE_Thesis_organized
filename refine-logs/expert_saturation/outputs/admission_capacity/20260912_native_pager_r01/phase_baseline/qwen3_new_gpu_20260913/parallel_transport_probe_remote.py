import concurrent.futures,hashlib,json,time,urllib.request
url='https://modelscope.cn/api/v1/models/Qwen/Qwen3-30B-A3B/repo?Revision=8143878278a202cc0ff9ef1bbe00b2abb1adf86d&FilePath=model-00001-of-00016.safetensors'
size=16*1024**2
start=time.monotonic()
def get(i):
 lo=i*size;hi=lo+size-1;t=time.monotonic()
 with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':f'bytes={lo}-{hi}'}),timeout=20) as r:
  assert r.status==206 and r.headers.get('Content-Range')==f'bytes {lo}-{hi}/3999417504'
  data=r.read(size+1);assert len(data)==size
 return data,dict(part=i,bytes=len(data),wall_s=time.monotonic()-t)
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:parts=list(pool.map(get,range(4)))
 data=b''.join(x[0] for x in parts);elapsed=time.monotonic()-start
 sha=hashlib.sha256(data).hexdigest();assert sha=='a007259ef22ed51156307aab2e467108ab22b5991cee7e8191fdd2d213f17e1d'
 result=dict(status='PASS_BOUNDED_TRANSPORT',bytes=len(data),wall_s=elapsed,MBps=len(data)/elapsed/1e6,prefix_sha256=sha,parts=[x[1] for x in parts])
except Exception as e:result=dict(status='FAIL',error=type(e).__name__+': '+str(e),wall_s=time.monotonic()-start)
result['scope']='CPU-only4x16MiB fixed public source range sample, concurrent with retained HF download; no full-shard or GPU claim.'
print(json.dumps(result))
