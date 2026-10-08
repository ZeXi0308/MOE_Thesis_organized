"""Validate the exact measured prefix; never turn absent results into a pass."""
import argparse
import json
from pathlib import Path


def analyze(cell):
    status = json.loads((cell / 'status.json').read_text())
    if status['status'] != 'COMPLETE':
        raise ValueError('Incomplete GPU cell')
    data = json.loads((cell / 'kv-roundtrip.json').read_text())
    action = json.loads((cell / 'selective-store.json').read_text())
    transfers = json.loads((cell / 'offload-events.json').read_text())
    before, after = data['before'], data['after']
    if before is None or after is None:
        raise ValueError('Both actual snapshots are required')
    expected = {f'model.layers.{i}.self_attn.attn' for i in range(16)}
    for snap in (before, after):
        if snap['request'] != action['selected'] or snap['tokens'] != 3296:
            raise ValueError('Wrong request or logical prefix')
        if len(snap['blocks']) != 206 or len(set(snap['blocks'])) != 206:
            raise ValueError('Missing or aliased blocks')
        if set(snap['layers']) != expected:
            raise ValueError('Missing or extra layers')
        for layer in snap['layers'].values():
            d = layer['digest']
            if (d['bytes'], d['tokens'], d['dtype'], d['heads'], d['packed_head_size']) != (27000832, 3296, 'torch.bfloat16', 16, 256):
                raise ValueError('Unexpected KV layout or byte count')
        if snap['computed'] < snap['tokens']:
            raise ValueError('Prefix includes uncomputed tokens')
    if before['call'] != 328 or after['call'] <= before['call']:
        raise ValueError('Invalid snapshot ordering')
    loads = [j for e in transfers['completed_jobs'] for j in e['jobs']
             if j['request'] == action['selected'] and not j['is_store']]
    if not loads or loads[0]['job_id'] != data['load_job']:
        raise ValueError('First real load completion not matched')
    mismatch = [name for name in sorted(expected)
                if before['layers'][name]['digest'] != after['layers'][name]['digest']]
    identity_equal = before['token_prefix_sha256'] == after['token_prefix_sha256']
    result = dict(status='MATCH' if identity_equal and not mismatch else 'MISMATCH',
                  tokens=3296, layers=16, bytes_per_snapshot=432013312,
                  token_identity_equal=identity_equal, differing_layers=mismatch,
                  before_call=before['call'], after_call=after['call'],
                  before_computed=before['computed'], after_computed=after['computed'],
                  physical_blocks_equal=before['blocks'] == after['blocks'],
                  scope='Only the captured first restored full-chunk prefix. '
                        'No quality, other-request, later-load or performance certification. '
                        'After snapshot is at engine.step return; do not assume it precedes resumed compute.')
    if data['status'] != result['status']:
        raise ValueError('Recorder verdict differs from independently compared content')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cell', type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.cell), indent=2))
