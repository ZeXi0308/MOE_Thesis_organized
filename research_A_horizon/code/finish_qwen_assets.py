"""Finish snapshot and workload assets after the identified downloader exits.

CPU-only; no GPU experiments or automatic download retries are submitted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--downloader-pid', type=int, required=True)
    args = parser.parse_args()
    receipt = args.root/'models/qwen-assets.json'
    record = dict(status='WAITING_DOWNLOAD', pid=os.getpid(), downloader_pid=args.downloader_pid,
                  revision=REVISION, start_unix_s=time.time())
    def save():
        tmp = receipt.with_suffix('.tmp')
        tmp.write_text(json.dumps(record, indent=2)+'\n')
        tmp.replace(receipt)
    if receipt.exists():
        raise FileExistsError('Inspect existing asset preparation receipt before another submission')
    save()
    try:
        process = Path('/proc')/str(args.downloader_pid)
        while process.exists():
            try:
                command = (process/'cmdline').read_bytes()
            except FileNotFoundError:
                break
            if b'fetch_qwen_mirror.py' not in command:
                raise RuntimeError('Downloader PID identity changed or became zombie; inspect before proceeding')
            time.sleep(15)
        downloaded = json.loads((args.root/'models/modelscope-download.json').read_text())
        if downloaded['status'] != 'COMPLETE':
            raise RuntimeError('Mirror did not report verified completion')
        record['status'] = 'VERIFYING_SNAPSHOT'; save()
        env = dict(os.environ, HF_HUB_DISABLE_XET='1')
        subprocess.run([sys.executable, '-B', str(args.code/'prepare_qwen.py'),
            '--root', str(args.root/'models'), '--revision', REVISION], env=env, check=True)
        model = json.loads((args.root/'models/qwen3-8b-download.json').read_text())
        assert model['status'] == 'COMPLETE' and model['revision'] == REVISION
        old = json.loads((args.root/'configs/w2.json').read_text())['config']
        plan = json.loads((args.root/'configs/w2-24-calibration-r09.json').read_text())
        env.update(LD_LIBRARY_PATH=plan['library_path'], PYTHONPATH=plan['runtime_overlay'],
                   PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        record.update(status='BUILDING_INPUTS', model_path=model['model_path']); save()
        paths = []
        for family in ('W1', 'W2', 'W3'):
            output = args.root/'configs'/('qwen-'+family.lower()+'.json')
            subprocess.run([sys.executable, '-B', str(args.code/'build_workload.py'),
                '--source', old['source_path'], '--source-model', old['model_path'],
                '--model', model['model_path'], '--disable-thinking', '--family', family,
                '--output', str(output)], env=env, check=True)
            paths.append(output)
        def documents(path):
            data = json.loads(path.read_text())['workload']
            rows = data['source_requests']; tokens = data['actual_prompt_token_ids']
            assert len(rows) == len(tokens) == 256
            result = {row['document_id']: ids for row, ids in zip(rows, tokens)}
            assert len(result) == len(rows)
            return result
        assert documents(paths[1]) == documents(paths[2]), 'W3 must preserve W2 prompt multiset'
        record.update(status='COMPLETE', inputs=[dict(path=str(p),
            sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
            w2_w3_prompt_identity=True)
    except BaseException as error:
        record.update(status='ERROR', error=type(error).__name__+': '+str(error).split('https://')[0])
        raise
    finally:
        record['end_unix_s'] = time.time(); save()


if __name__ == '__main__':
    main()
