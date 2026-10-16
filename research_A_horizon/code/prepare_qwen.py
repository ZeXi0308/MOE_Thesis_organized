"""Fetch a pinned official Qwen snapshot through a selectable HF endpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--endpoint', default='https://hf-mirror.com')
    args = p.parse_args()
    if len(args.revision) != 40 or any(c not in '0123456789abcdef' for c in args.revision):
        raise ValueError('Full commit SHA required')
    args.root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.root).free < 30 * 2**30:
        raise RuntimeError('Need 30 GiB before fetching model')
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(endpoint=args.endpoint)
    info = api.model_info('Qwen/Qwen3-8B', revision=args.revision, files_metadata=True)
    assert info.sha == args.revision
    record = dict(repo='Qwen/Qwen3-8B', revision=info.sha, endpoint=args.endpoint,
                  start_unix_s=time.time(), status='DOWNLOADING')
    receipt = args.root / 'qwen3-8b-download.json'
    receipt.write_text(json.dumps(record, indent=2))
    try:
        path = Path(snapshot_download(repo_id=record['repo'], revision=info.sha,
            endpoint=args.endpoint, cache_dir=str(args.root / 'hf'), max_workers=2,
            allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', 'README.md', 'LICENSE']))
        files = []
        for item in info.siblings:
            f = path / item.rfilename
            if not f.is_file():
                continue
            digest = hashlib.sha256()
            with f.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
                    digest.update(chunk)
            sha = digest.hexdigest()
            if item.lfs is not None:
                assert sha == item.lfs.sha256, item.rfilename
            files.append(dict(name=item.rfilename, bytes=f.stat().st_size, sha256=sha))
        index = json.loads((path / 'model.safetensors.index.json').read_text())
        assert set(index['weight_map'].values()) <= {row['name'] for row in files}
        cfg = json.loads((path / 'config.json').read_text())
        assert cfg['model_type'] == 'qwen3' and not cfg['use_sliding_window']
        kv_bytes = 2 * cfg['num_hidden_layers'] * cfg['num_key_value_heads'] * cfg['head_dim'] * 2
        record.update(status='COMPLETE', model_path=str(path), files=files,
                      bf16_kv_bytes_per_token=kv_bytes, end_unix_s=time.time())
    except BaseException as error:
        record.update(status='ERROR', error=repr(error), end_unix_s=time.time())
        raise
    finally:
        receipt.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
