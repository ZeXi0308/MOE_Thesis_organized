import concurrent.futures, hashlib, json, time, urllib.request

url = 'https://modelscope.cn/api/v1/models/Qwen/Qwen3-30B-A3B/repo?Revision=8143878278a202cc0ff9ef1bbe00b2abb1adf86d&FilePath=model-00001-of-00016.safetensors'
size = 16 * 1024**2

def get(i):
    lo = i * size
    hi = lo + size - 1
    row = dict(part=i, start=lo, stop=hi)
    start = time.monotonic()
    data = b''
    try:
        req = urllib.request.Request(url + '&range_start=' + str(lo), headers={'Range': f'bytes={lo}-{hi}', 'Cache-Control': 'no-cache'})
        with urllib.request.urlopen(req, timeout=20) as response:
            row.update(http_status=response.status, content_range=response.headers.get('Content-Range'), content_length=response.headers.get('Content-Length'))
            assert response.status == 206, row
            assert row['content_range'] == f'bytes {lo}-{hi}/3999417504', row
            data = response.read(size + 1)
            assert len(data) == size, len(data)
    except Exception as exc:
        row['error'] = type(exc).__name__ + ': ' + str(exc)
    row.update(bytes=len(data), wall_s=time.monotonic() - start, sha256=hashlib.sha256(data).hexdigest())
    return row, data

start = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    parts = list(pool.map(get, range(4)))
wall = time.monotonic() - start
data = b''.join(part[1] for part in parts)
sha = hashlib.sha256(data).hexdigest()
valid = all('error' not in part[0] for part in parts) and sha == 'a007259ef22ed51156307aab2e467108ab22b5991cee7e8191fdd2d213f17e1d'
print(json.dumps(dict(status='PASS_BOUNDED_TRANSPORT' if valid else 'FAIL', rows=[p[0] for p in parts], bytes=len(data), wall_s=wall, MBps=len(data)/wall/1e6, prefix_sha256=sha, scope='CPU-only immutable-source range diagnostic during model loading; no full-shard or GPU result. Cache-key query differs by range.')))
