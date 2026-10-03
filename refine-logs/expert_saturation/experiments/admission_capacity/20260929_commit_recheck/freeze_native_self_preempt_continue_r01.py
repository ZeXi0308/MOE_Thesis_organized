#!/usr/bin/env python3
"""Freeze one seen-input action probe without altering completed confirmation."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

candidate = Path('candidate_native_self_preempt_continue_r01')
manifest = json.loads((candidate / 'manifest.json').read_text())
assert all(sha(candidate / name) == digest for name, digest in manifest.items())
msha = sha(candidate / 'manifest.json')
assert msha == '2770b1a25075fd153d18e89c25a4f166db9dbd993343b5ce652912f2a85a6376'
ctrl = 'run_native_self_preempt_continue_triplet_r01.py'
parent = json.loads(Path('NATIVE_RESIDENCY_FRESH_B1_PLAN_R01_20261002.json').read_text())
deps = parent['controller_dependencies'] + ['run_native_residency_fresh_b1_r01.py']
for name in [ctrl, *deps]:
    ast.parse(Path(name).read_text())
stage = Path('/private/tmp/moe-a-native-self-preempt-continue-stage-r01-20261002')
remote = '/root/' + stage.name
session = '/root/moe-a-native-self-preempt-continue-session-r01-20261002'
plan = {key: parent[key] for key in [
    'schema_version', 'authorized_gpu_uuid', 'approved_host_bytes', 'lock_path',
    'expected_lock_device_inode', 'python', 'hf_cache_dir', 'shared_model_source_cache',
    'model_revision', 'model_verifier_path', 'model_verifier_sha256',
    'cgroup_memory_max_file', 'approved_total_wall_seconds', 'warm_runtime_cache_source',
    'authorization_reference', 'input_workload_file_sha256', 'input_config_file_sha256',
    'input_stats_file_sha256']}
plan.update(
    session_dir=session, measurement_scope='SEEN_INPUT_SINGLE_ACTION_PROBE',
    hypothesis='After eligible current-request self-preemption, continuing the native running scan lets later running requests perform useful work and reduces worst request gap within fixed rate and flow budgets.',
    action_difference='Only density_continue bypasses the native outer break after qualified self-preemption. Same victim rule as density_break; same original successor, allocator, saving, ordinary backfill and input. Tail_break anchors the strongest control.',
    mechanism_criteria='Applied continuations > 0; all applied events actually visit the original successor; at least one suffix schedules positive tokens and subsequently produces a new output; preempted source schedules zero. No forced rotation, all128complete, nativefull and raw victim actions matched.',
    service_criteria='For continuation against BOTH controls: actual output rate >=97%, mean arrival-to-completion <=105%, strictly lower maximum request gap. Retain all20 goodput points and all request improvements/regressions. Any missing mechanism evidence or failed service criterion rejects this pilot hypothesis.',
    retention_rule='One frozen triplet tail_break,density_break,density_continue; all outputs retained. No threshold/input replacement or repeated trial to obtain a win. These128 inputs are already seen; any positive result requires new-input confirmation and close prior-art baseline.',
    claim_ceiling='Development probe of a new scheduling action. Not same-state causality, not proof that a skipped scan explains an entire long stall, and not a standalone novelty claim.',
    package_manifest_sha256=msha, controller_sha256=sha(ctrl),
    controller_dependencies=deps, controller_dependency_sha256={name:sha(name) for name in deps},
    analysis='analyze_native_self_preempt_continue_triplet_r01.py',
    frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    cells=[dict(arm=arm, package_dir=remote+'/'+label+'/'+candidate.name,
                output_dir='/root/moe-a-native-self-preempt-continue-output-'+label+'-r01-20261002',
                max_wall_seconds=900)
           for label, arm in zip(['first','second','third'], ['tail_break','density_break','density_continue'])])
pname = 'NATIVE_SELF_PREEMPT_CONTINUE_TRIPLET_PLAN_R01_20261002.json'
assert not Path(pname).exists() and not stage.exists()
write(pname, plan)
stage.mkdir()
for label in ['first','second','third']:
    shutil.copytree(candidate,stage/label/candidate.name)
for name in [ctrl,pname,*deps]:
    shutil.copyfile(name,stage/name)
archive = stage.with_suffix('.tar.gz')
assert not archive.exists()
with tarfile.open(archive,'w:gz') as target:
    target.add(stage,arcname=stage.name)
meta = dict(status='FROZEN_GPU_UNRUN',manifest=msha,controller=ctrl,controller_sha=sha(ctrl),
            plan=pname,plan_sha=sha(pname),stage=str(stage),remote_stage=remote,
            remote_session=session,archive=str(archive),archive_sha=sha(archive),
            archive_bytes=archive.stat().st_size)
write('NATIVE_SELF_PREEMPT_CONTINUE_FROZEN_R01_20261002.json',meta)
print(json.dumps(meta))
