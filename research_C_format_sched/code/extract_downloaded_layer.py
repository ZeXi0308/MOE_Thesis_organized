#!/usr/bin/env python3
"""Extract already-arrived tensor ranges while the original HF download continues.

This is a partial-checkpoint derivative, never a complete checkpoint. Retain its
source receipt and verify selected tensor bytes against the completed shard later.
"""
import hashlib
import json
from pathlib import Path
import shutil
import struct
import time

root = Path('/root/autodl-tmp/research_C_format_sched')
source_dir = root / 'checkpoints/qwen3_nvfp4_2538ded2'
source = next((source_dir / '.cache/huggingface/download').glob('*.incomplete'))
dest = root / 'checkpoints/qwen3_nvfp4_layer0_subset'
dest.mkdir(exist_ok=False)
with source.open('rb') as f:
    header_size = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(header_size))
    keys = sorted(k for k in header if k.startswith('model.layers.0.mlp.experts.'))
    if len(keys) != 1536:
        raise ValueError(f'Expected 1536 tensors; got {len(keys)}')
    required_end = 8 + header_size + max(header[k]['data_offsets'][1] for k in keys)
    if source.stat().st_size < required_end:
        raise RuntimeError('Not all selected tensor bytes have arrived yet')
    new_header = {'__metadata__': {'format': 'pt', 'scope': 'layer0 experts only'}}
    offset = 0
    for key in keys:
        info = header[key]
        size = info['data_offsets'][1] - info['data_offsets'][0]
        new_header[key] = {**info, 'data_offsets': [offset, offset + size]}
        offset += size
    encoded = json.dumps(new_header, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 8)
    output = dest / 'model-00001-of-00004.safetensors'
    hashes = {}
    with output.open('xb') as stream:
        stream.write(struct.pack('<Q', len(encoded)))
        stream.write(encoded)
        for key in keys:
            lo, hi = header[key]['data_offsets']
            f.seek(8 + header_size + lo)
            raw = f.read(hi - lo)
            if len(raw) != hi - lo:
                raise RuntimeError('Source range incomplete')
            hashes[key] = hashlib.sha256(raw).hexdigest()
            stream.write(raw)
shutil.copy2(source_dir / 'config.json', dest / 'config.json')
(dest / 'model.safetensors.index.json').write_text(json.dumps({'metadata': {'total_size': offset},
    'weight_map': {k: output.name for k in keys}}, indent=2) + '\n')
receipt = {'checkpoint': 'nvidia/Qwen3-30B-A3B-NVFP4',
    'revision': '2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3', 'created_utc_s': time.time(),
    'scope': 'layer0 experts only; not a loadable full model',
    'source_download_complete': False, 'source_file': str(source),
    'source_required_end': required_end, 'source_size_at_extraction': source.stat().st_size,
    'selected_tensor_bytes': offset, 'tensor_sha256': hashes,
    'completed_source_verification': 'pending'}
(dest / 'subset_source.json').write_text(json.dumps(receipt, indent=2) + '\n')
out = root / 'runs/checkpoint_probe'
out.mkdir(exist_ok=True)
shutil.copy2(dest / 'subset_source.json', out / 'subset_source.json')
print(json.dumps({'subset': str(dest), 'tensors': len(keys), 'tensor_bytes': offset}))
