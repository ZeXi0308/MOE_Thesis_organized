"""Prepare same-resource native offload on/off qualification; no GPU imports."""
from pathlib import Path
import shutil,hashlib,json
root=Path(__file__).resolve().parents[2];o=root/'outputs/admission_capacity'
p=o/'20260914_native_offload_baseline_r01';pkg=p/'pkg'
src=o/'20260914_d6_strong_baselines_r01/preparation/pkg'
pkg.mkdir(parents=True,exist_ok=False)
for name in ['run_probe.py','memory_telemetry.py','native_capture.py','metrics.py','safe_static.py','completion_headroom.py','absence_rotation.py','rotation_native.py']:
 shutil.copy2(src/name,pkg/name)
shutil.copytree(src/'inputs_preparation',pkg/'inputs_preparation')
shutil.copy2(Path(__file__).with_name('native_offload_observer.py'),pkg/'native_offload_observer.py')
s=(pkg/'run_probe.py').read_text()
s=s.replace('from metrics import summarize_episode_requests','from native_offload_observer import install as observe_offload, drain as drain_offload\nfrom metrics import summarize_episode_requests')
s=s.replace("    parser.add_argument('--output-dir'", "    parser.add_argument('--offload-gib', type=int, choices=[0,16], required=True)\n    parser.add_argument('--output-dir'")
s=s.replace("args.completion_policy not in ('native', 'headroom', 'rotate')", "args.completion_policy != 'native'")
s=s.replace("preemption_mode='native_recompute', rotation_config=vars(RotationConfig()))", "preemption_mode='native_recompute_or_external_kv', offload_gib=args.offload_gib, rotation_config=vars(RotationConfig()))")
s=s.replace("VLLM_BATCH_INVARIANT='0')", "VLLM_BATCH_INVARIANT='0', VLLM_USE_SIMPLE_KV_OFFLOAD='0')")
s=s.replace("'absence_rotation.py', 'rotation_native.py']}", "'absence_rotation.py', 'rotation_native.py', 'native_offload_observer.py']}")
s=s.replace("        dump(out/'engine_args.json', kwargs)","        kwargs.update(kv_offloading_size=args.offload_gib or None, kv_offloading_backend='native')\n        dump(out/'engine_args.json', kwargs)")
s=s.replace("        set_empty_admission_cap(engine, args.cap)","        dump(out/'warmup-offload-drain.json', drain_offload(engine))\n        if not engine.reset_prefix_cache(reset_connector=True):\n            raise RuntimeError('failed to clear warmup connector cache')\n        dump(out/'warmup-cache-reset.json', dict(success=True))\n        set_empty_admission_cap(engine, args.cap)")
a=s.index("        if args.completion_policy == 'rotate':\n            decisions, uninstall")
b=s.index('        try:\n            raw = capture_with_memory',a)
s=s[:a]+'''        expected = 'OffloadingConnector' if args.offload_gib else 'NoneType'
        if type(scheduler.connector).__name__ != expected:
            raise RuntimeError('effective connector mismatch')
        dump(out/'connector.json', dict(type=expected, config=str(engine.vllm_config.kv_transfer_config)))
        decisions = []
        offload_data, uninstall = observe_offload(scheduler.connector)
'''+s[b:]
s=s.replace("        finally:\n            uninstall()", "            dump(out/'post-request-drain.json', drain_offload(engine))\n        finally:\n            dump(out/'offload-events.json', offload_data)\n            uninstall()")
(pkg/'run_probe.py').write_text(s)
files={str(f.relative_to(pkg)):hashlib.sha256(f.read_bytes()).hexdigest() for f in pkg.rglob('*') if f.is_file()}
(p/'files.json').write_text(json.dumps(files,indent=2)+'\n')
(p/'STATUS.json').write_text(json.dumps(dict(status='CPU_PREPARED_UNRUN',gpu_runs=0,uploaded=False),indent=2)+'\n')
print(p)
