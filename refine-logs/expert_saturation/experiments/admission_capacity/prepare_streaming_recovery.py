"""Compose one open-population baseline diagnostic from the measured cohort3 stack."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


KV_BYTES = 8592031744  # 4096 usable blocks + the null block, 2 MiB per block.
PARENT_HOST_BYTES = 96636764160  # Existing shared container ceiling; not a per-tree cap.


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'expected one source anchor: {old[:90]!r}')
    return text.replace(old, new)


def patch_capture(text):
    text=replace(text,'    count = config["output_tokens"]\n',
        '    count = config["output_tokens"]\n'
        '    output_mode = config.get("output_mode", "fixed")\n'
        '    if output_mode not in ("fixed", "eos"):\n'
        '        raise ValueError("output_mode must be fixed or eos")\n')
    text=replace(text,'            prompt_tokens=len(ids), output_token_ids=[], token_times_s=[], status="unfinished")',
        '            prompt_tokens=len(ids), max_output_tokens=count, output_token_ids=[], token_times_s=[], status="unfinished")')
    text=replace(text,'max_tokens=count, min_tokens=count,\n                    ignore_eos=True,',
        'max_tokens=count, min_tokens=count if output_mode == "fixed" else 0,\n'
        '                    ignore_eos=output_mode == "fixed",')
    text=replace(text,'                    finish_reason=completion.finish_reason, prefix_valid=False)',
        '                    finish_reason=completion.finish_reason,\n'
        '                    native_stop_reason=getattr(completion, "stop_reason", None), prefix_valid=False)')
    text=replace(text,'                    if len(tokens) != count or completion.finish_reason != "length":',
        '                    if output_mode == "fixed" and (len(tokens) != count or completion.finish_reason != "length"):')
    text=replace(text,'                    row.update(status="completed", completion_s=received, stop_reason="length")',
        '                    if output_mode == "eos" and (completion.finish_reason not in ("stop", "length")\n'
        '                            or (completion.finish_reason == "length" and len(tokens) != count)):\n'
        '                        raise ValueError("native completion violated EOS/length-bound contract")\n'
        '                    row.update(status="completed", completion_s=received, stop_reason=completion.finish_reason,\n'
        '                        native_stop_reason=getattr(completion, "stop_reason", None))')
    text=replace(text,'        engine_steps=engine_steps,',
        '        engine_steps=engine_steps, output_mode=output_mode,')
    ast.parse(text)
    return text


def patch_pool_qualification(text):
    text=replace(text,"        require(config['prompt_tokens'] == 3072 and config['output_tokens'] == 1024\n"
        "                and config['requests'] == 32, 'frozen request dimensions differ')",
        "        require(1 <= config['prompt_tokens'] <= 3072 and config['output_tokens'] == 1024\n"
        "                and config['requests'] == 64, 'open episode upper bounds differ')")
    text=replace(text,'maximum_request_tokens=4096,',
        "maximum_request_tokens=config['prompt_tokens'] + config['output_tokens'],")
    text=replace(text,"        result.update(status='QUALIFIED', measurement_status='NOT_YET_RUN')",
        "        result.update(status='QUALIFIED', measurement_status='NOT_YET_RUN',\n"
        "            bound_semantics='Conservative P_max+output_limit bound, not actual EOS or physical future reservation')")
    ast.parse(text)
    return text


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',required=True,type=Path)
    p.add_argument('--inputs',required=True,type=Path)
    p.add_argument('--runner',required=True,type=Path)
    p.add_argument('--rotation',required=True,type=Path)
    p.add_argument('--host-observer',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    a=p.parse_args();assert not a.output_dir.exists(),'retain prior package'
    source=a.workspace/'refine-logs/expert_saturation/outputs/admission_capacity/20260914_recovery_holdout_comparison_r01/preparation/pkg'
    meta=json.loads((source.parent/'preparation.json').read_text())
    for name,h in meta['files_sha256'].items():assert sha(source/name)==h
    config=json.loads((a.inputs/'config.json').read_text());work=json.loads((a.inputs/'workload.json').read_text())
    assert config['requests']==64 and config['output_tokens']==1024
    assert config['min_tokens']==0 and config['ignore_eos'] is False
    assert work['arrival_traces_s']['steady']==[i*.5 for i in range(64)]
    assert len({r['document_id'] for r in work['source_requests']})==64
    assert all(256<=len(ids)<=3072 for ids in work['actual_prompt_token_ids'])
    assert config['prompt_tokens']==max(map(len,work['actual_prompt_token_ids']))
    assert hashlib.sha256(json.dumps(work,sort_keys=True).encode()).hexdigest()==config['workload_sha256']
    a.output_dir.mkdir(parents=True);pkg=a.output_dir/'pkg';pkg.mkdir()
    for name in ('native_capture.py','safe_static.py','memory_telemetry.py','metrics.py','absence_rotation.py'):
        text=(source/name).read_text()
        if name=='native_capture.py':text=patch_capture(text)
        if name=='safe_static.py':text=patch_pool_qualification(text)
        (pkg/name).write_text(text)
    for src,name in [(a.runner,'run_streaming_recovery.py'),(a.rotation,'rotation_native.py'),
                     (a.host_observer,'host_budget_observed.py')]:shutil.copy2(src,pkg/name)
    shutil.copy2(a.workspace/'refine-logs/expert_saturation/experiments/admission_capacity/host_budget_envelope.py',pkg/'host_budget_envelope.py')
    shutil.copytree(a.inputs,pkg/'inputs')
    shutil.copytree(source/'inputs_preparation/prepared',pkg/'warmups')
    labels=['stream-block0-native','stream-block0-most_output','stream-block1-most_output','stream-block1-native']
    variants=['native','most_output','most_output','native']
    cells=[dict(label=l,variant=v) for l,v in zip(labels,variants)]
    campaign=dict(status='PREPARED_GPU_UNRUN',cells=cells,kv_cache_bytes=KV_BYTES,usable_blocks=4096,
        parent_host_limit_bytes=PARENT_HOST_BYTES,host_scope='Existing shared container hard limit; process-tree RSS OBSERVED_ONLY',
        scope='Diagnostic native recovery existence in natural-length open population. No post-hoc pressure/arrival/EOS tuning; no-action/length-capped outcomes retained.',
        runtime_sources=meta['expected_runtime_sources'],source_package_sha256=meta['archive_sha256'])
    (pkg/'campaign.json').write_text(json.dumps(campaign,indent=2)+'\n')
    script='''#!/bin/bash
set -eu
PYTHON_BIN=${1:?Pass installed vLLM Python}
BASE=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$BASE/pkg"
export HF_HOME=/root/autodl-tmp/hf-cache HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
sha256sum -c SHA256SUMS
exec 9>/root/autodl-tmp/moe-research-gpu.lock
flock -n 9 || exit 90
test ! -e "$BASE/results"
mkdir "$BASE/results"
run() {
 label=$1; variant=$2
 pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) || exit 92
 test -z "$pids" || exit 93
 echo "START $(date -u +%FT%TZ) $label"
 "$PYTHON_BIN" -u host_budget_observed.py --parent-cgroup /sys/fs/cgroup --expected-parent-memory-max 96636764160 --declared-host-kv-offload-bytes 0 --output "$BASE/results/$label-host" --interval-s 1 -- timeout --kill-after=30 600 "$PYTHON_BIN" -u run_streaming_recovery.py --inputs inputs --warmup-inputs warmups --variant "$variant" --output-dir "$BASE/results/$label"
 echo "END $(date -u +%FT%TZ) $label"
}
'''
    script += ''.join(f"run {c['label']} {c['variant']}\n" for c in cells)
    script += 'echo "CAMPAIGN_FINISHED $(date -u +%FT%TZ)"\n'
    (pkg/'run.sh').write_text(script)
    files={str(x.relative_to(pkg)):sha(x) for x in sorted(pkg.rglob('*')) if x.is_file()}
    (pkg/'SHA256SUMS').write_text(''.join(f'{h}  {n}\n' for n,h in files.items()))
    archive=a.output_dir/'execution.tar.gz'
    with tarfile.open(archive,'w:gz') as t:t.add(pkg,arcname='pkg')
    receipt=dict(status='PREPARED_GPU_UNRUN',files_sha256=files,archive_sha256=sha(archive),cells=cells,
                 expected_runtime_sources=meta['expected_runtime_sources'],source_capture_sha256=sha(source/'native_capture.py'))
    (a.output_dir/'preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status=receipt['status'],archive_sha256=receipt['archive_sha256'])))


if __name__=='__main__':main()
