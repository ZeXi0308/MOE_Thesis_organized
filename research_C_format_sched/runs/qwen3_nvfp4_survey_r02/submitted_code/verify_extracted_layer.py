#!/usr/bin/env python3
"""Verify a partial-download derivative against its now-complete source shard."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--subset', type=Path, required=True)
    p.add_argument('--download-manifest', type=Path, required=True)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    receipt = json.loads((a.subset / 'subset_source.json').read_text())
    downloaded = json.loads(a.download_manifest.read_text())
    if downloaded['state'] != 'COMPLETE' or downloaded['revision'] != receipt['revision']:
        raise ValueError('Source download is incomplete or has a different revision')
    index = json.loads((a.subset / 'model.safetensors.index.json').read_text())['weight_map']
    names = set(index.values())
    if len(names) != 1:
        raise ValueError('Expected a single-shard derivative')
    name = names.pop()
    source = a.source_dir / name
    expected = next(f for f in downloaded['files'] if f['name'] == name)
    digest = hashlib.sha256()
    with source.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if source.stat().st_size != expected['bytes'] or digest.hexdigest() != expected['sha256']:
        raise ValueError('Completed source shard differs from the download manifest')
    with source.open('rb') as f, (a.subset / name).open('rb') as g:
        source_header_n = struct.unpack('<Q', f.read(8))[0]
        source_header = json.loads(f.read(source_header_n))
        subset_header_n = struct.unpack('<Q', g.read(8))[0]
        subset_header = json.loads(g.read(subset_header_n))
        for key, expected_hash in receipt['tensor_sha256'].items():
            src, dst = source_header[key], subset_header[key]
            if (src['dtype'], src['shape']) != (dst['dtype'], dst['shape']):
                raise ValueError(f'Tensor shape/dtype mismatch: {key}')
            for stream, header_n, info in ((f, source_header_n, src), (g, subset_header_n, dst)):
                lo, hi = info['data_offsets']
                stream.seek(8 + header_n + lo)
                raw = stream.read(hi - lo)
                if len(raw) != hi - lo or hashlib.sha256(raw).hexdigest() != expected_hash:
                    raise ValueError(f'Tensor bytes mismatch: {key}')
    result = {'state': 'COMPLETE', 'verified_utc_s': time.time(),
              'revision': receipt['revision'], 'source_shard': name,
              'source_bytes': expected['bytes'], 'source_sha256': digest.hexdigest(),
              'verified_tensors': len(receipt['tensor_sha256']),
              'verified_tensor_bytes': receipt['selected_tensor_bytes'],
              'scope': 'All derivative tensor bytes match the completed fixed-revision shard',
              'original_receipt_preserved': str(a.subset / 'subset_source.json')}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
