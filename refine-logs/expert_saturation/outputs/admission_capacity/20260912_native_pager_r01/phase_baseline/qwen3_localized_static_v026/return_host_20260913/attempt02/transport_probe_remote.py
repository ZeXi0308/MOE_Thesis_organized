"""One CPU-only 32MiB serial versus4x8MiB range probe during frozen loading."""
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,os,sys,time
from pathlib import Path
from urllib.request import Request,urlopen

stage=Path('/root/autodl-tmp/qwen3-localized-static-v026-launch-r02')
out=Path('/root/autodl-tmp/qwen3-localized-static-v026-r02')
MiB=2**20; segment=8*MiB; report=dict(status='FAILED',scope='Small-range transport diagnostic concurrent with existing serial loading; not4GB transfer rate, full-shard integrity, or inference performance.')

def loading():
    launch=json.loads((stage/'launch.json').read_text())
    assert launch['status']=='RUNNING' and launch['child_pid']==13275, 'unexpected live launch'
    for pid,name in ((13263,'run_remote.py'),(13275,'run_native_pager.py')):
        argv=Path('/proc',str(pid),'cmdline').read_bytes().split(b'\0')
        assert str(stage/name).encode() in argv, 'process does not belong to frozen stage'
    assert not (out/'comparison/numerical_qualification/attribution_gate.json').exists(), 'qualification gate already exists'
    events=(out/'loader_receipt.jsonl').read_text().splitlines()
    last=json.loads(events[-1]); assert last['event']=='DOWNLOAD', 'loader is not downloading'
    return dict(checked_unix_s=time.time(),parent_pid=13263,worker_pid=13275,last_loader_event=last)

def fetch(start,stop):
    began=time.perf_counter(); deadline=began+25; count=0
    digest=hashlib.sha256(); parts=[hashlib.sha256() for _ in range((stop-start+1)//segment)]
    request=Request(url,headers={'Range':f'bytes={start}-{stop}','Accept-Encoding':'identity'})
    with urlopen(request,timeout=25) as response:
        header_end=time.perf_counter()
        assert response.status==206, 'Range status must be206; body not read'
        assert response.headers.get('Content-Range')==f'bytes {start}-{stop}/{size}', 'wrong Content-Range; body not read'
        assert response.headers.get('Content-Encoding','identity')=='identity', 'encoded Range rejected'
        assert int(response.headers.get('Content-Length','-1'))==stop-start+1, 'wrong Content-Length; body not read'
        while count<stop-start+1:
            remaining=deadline-time.perf_counter(); assert remaining>0, 'request deadline'
            response.fp.raw._sock.settimeout(min(25,remaining))
            block=response.read(min(MiB,stop-start+1-count,segment-count%segment))
            assert block, 'short Range body'
            digest.update(block);parts[count//segment].update(block);count+=len(block)
        assert not response.read(1), 'excess Range body'
    end=time.perf_counter(); assert end<=deadline, 'request deadline'
    return dict(start=start,stop=stop,bytes=count,sha256=digest.hexdigest(),
        segment_sha256=[h.hexdigest() for h in parts],status=206,
        content_range=f'bytes {start}-{stop}/{size}',header_latency_s=header_end-began,
        body_wall_s=end-header_end,end_to_end_wall_s=end-began)

try:
    os.sched_setaffinity(0,{8}); assert os.sched_getaffinity(0)=={8}
    before=loading(); data=(stage/'qwen3.manifest.json').read_bytes()
    frozen=json.loads((stage/'protocol.json').read_text())
    assert hashlib.sha256(data).hexdigest()==frozen['package_files']['qwen3.manifest.json'], 'manifest SHA'
    manifest=json.loads(data)
    assert manifest['repository']=='Qwen/Qwen3-30B-A3B' and manifest['revision']=='ad44e777bcd18fa416d9da3bd8f70d33ebb85d39', 'frozen model revision'
    shard='model-00002-of-00016.safetensors';size=manifest['shards'][shard]['bytes']
    assert size>=32*MiB
    url=f"https://hf-mirror.com/{manifest['repository']}/resolve/{manifest['revision']}/{shard}"
    report.update(before=before,cpu_affinity=[8],manifest_sha256=hashlib.sha256(data).hexdigest(),
        shard=shard,origin='https://hf-mirror.com',revision=manifest['revision'],request_timeout_s=25)
    single=fetch(0,32*MiB-1);report['single']=single
    report['between']=loading(); began=time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(fetch,i*segment,(i+1)*segment-1) for i in range(4)]
        parallel=[future.result() for future in futures]
    parallel_wall=time.perf_counter()-began; report.update(parallel=parallel,parallel_arm_wall_s=parallel_wall)
    assert single['segment_sha256']==[r['sha256'] for r in parallel], 'ordered8MiB segment SHA mismatch'
    report.update(status='PASS',ordered_segment_sha_equal=True,total_payload_bytes=64*MiB,
        single_arm_wall_s=single['end_to_end_wall_s'],after=loading(),
        comparison='Serial32MiB fullSHA retained; four ordered segmentSHA match parallel ranges. Chunk hashes are not concatenated into a purported fullSHA.')
except BaseException as error:
    report.update(error_type=type(error).__name__,error=str(error) if isinstance(error,AssertionError) else 'Details omitted to avoid signed redirect URLs')
print(json.dumps(report),flush=True)
raise SystemExit(0 if report['status']=='PASS' else 1)
