import hashlib,json,time,urllib.request
url='https://modelscope.cn/api/v1/models/Qwen/Qwen3-30B-A3B/repo?Revision=8143878278a202cc0ff9ef1bbe00b2abb1adf86d&FilePath=model-00001-of-00016.safetensors'
limit=64*1024**2;count=0;digest=hashlib.sha256();start=time.monotonic();result=dict(public_url=url,scope='CPU-only64MiB bounded transport sample; not full shard integrity or GPU evidence. Concurrent with retained HFmirror download.')
try:
 with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':f'bytes=0-{limit-1}'}),timeout=20) as response:
  result.update(http_status=response.status,content_range=response.headers.get('Content-Range'),connect_s=time.monotonic()-start)
  while count<limit:
   data=response.read(min(1024**2,limit-count))
   if not data:break
   digest.update(data);count+=len(data)
except Exception as exc:result['error']=type(exc).__name__+': '+str(exc)
result.update(bytes=count,wall_s=time.monotonic()-start,prefix_sha256=digest.hexdigest());result['average_MBps']=count/result['wall_s']/1e6
print(json.dumps(result,indent=2))
