"""Prepare a native single-event pair; no network or GPU initialization."""
import hashlib
import json
from pathlib import Path
import shutil

base=Path(__file__).resolve().parents[2]
out=base/'outputs/admission_capacity/20260914_staged_store_probe_r01'
out.mkdir(exist_ok=False)
pkg=out/'pkg'
shutil.copytree(base/'outputs/admission_capacity/20260914_selective_store_once_r01/pkg',pkg)
exp=base/'experiments/admission_capacity'
for name in ['staged_store_once.py','staged_save_contract.py']:
    shutil.copy2(exp/name,pkg/name)
runner=pkg/'run_probe.py'
s=runner.read_text()
s=s.replace('from selective_store_once import install as install_selective_store','from staged_store_once import install as install_selective_store')
s=s.replace("install_selective_store(scheduler, save=args.selective_save=='on')", "install_selective_store(scheduler, vllm_config=engine.vllm_config, block_size=16, save=args.selective_save=='on')")
s=s.replace("'selective_store_once.py']", "'staged_store_once.py', 'staged_save_contract.py']")
s=s.replace("rotation_victim_order=args.victim_order,", "rotation_victim_order='most_output', action_scope='single_staged_event', preparation_step=329,")
s=s.replace("    root, out = Path(__file__).resolve().parent, args.output_dir", "    if args.offload_gib != 16:raise ValueError('Both arms require the same native 16GiB host cache')\n    root, out = Path(__file__).resolve().parent, args.output_dir")
runner.write_text(s)
# Exact copied installed sources, checked before invoking the GPU runner.
sources={}
for origin in ['20260914_kv_roundtrip_feasibility_r01/native_offload_source.json','20260914_load_ready_contract_r01/native_source.json']:
    for name,source in json.loads((base/'outputs/admission_capacity'/origin).read_text()).items():
        if name.endswith('.py'):sources[name]=hashlib.sha256(source.encode()).hexdigest()
(pkg/'runtime_source_hashes.json').write_text(json.dumps(sources,indent=2)+'\n')
(pkg/'preflight.py').write_text('''import hashlib, importlib.util, json
from pathlib import Path
root=Path(importlib.util.find_spec('vllm').origin).parent
expected=json.loads(Path(__file__).with_name('runtime_source_hashes.json').read_text())
for name,sha in expected.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha, name
print('Installed source hashes match; no GPU initialized')
''')
(pkg/'run.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/root/autodl-tmp/hf-cache HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
cd -- "$(dirname -- "$0")"
exec 9>/root/autodl-tmp/moe-research-gpu.lock
flock -n 9 || { echo ABORT_GPU_GROUP_LOCK_BUSY; exit 75; }
py=/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python
"$py" preflight.py
for arm in off on; do
  "$py" run_probe.py --domain long --reservation-policy full --completion-policy native --cap 32 --gpu-memory-utilization 0.90 --kv-cache-bytes 13960740864 --selective-save "$arm" --offload-gib 16 --output-dir "../results/save-$arm"
done
''')
(pkg/'run.sh').chmod(0o755)
for p in pkg.rglob('__pycache__'):shutil.rmtree(p)
manifest={str(p.relative_to(pkg)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(pkg.rglob('*')) if p.is_file()}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(out/'STATUS.md').write_text('''# PREPARED_LOCAL / GPU_UNRUN

One staged most-output event per arm, save-off then save-on; same 16GiB native host cache, d6 workload and warmups. This pair qualifies execution, not performance significance. No remote upload or driver. Respect Qwen and funding-filter queue, verify their real terminal state before launch. Group flock covers both arms and initialization; each original runner checks GPU processes and refuses existing result directories. First error stops; retain failures, no automatic retry.

Source hashes, CPU contract and actual native waiting-loop prefix checked. Full schedule, allocation, physical transfer, restoration correctness and request performance remain unverified. Actual save metadata must cover the planned block list and preemption must emit matching flush IDs. Next GPU evidence must establish original-target new output, eventual saved-victim load, complete requests and actual worker flush/transfer ordering; metadata alone is insufficient.
''')
print(out)
