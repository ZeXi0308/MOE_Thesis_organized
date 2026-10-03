#!/usr/bin/env python3
"""Freeze the closest public score adaptation with native and flow controls."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

candidate=Path('candidate_native_bidkv_score_r01')
manifest=json.loads((candidate/'manifest.json').read_text())
assert all(sha(candidate/name)==digest for name,digest in manifest.items())
msha=sha(candidate/'manifest.json')
assert msha=='24f1e31a0f116fe96b0e26696e59bcb6b43e85658e5d95a82d6835e86941166e'
ctrl='run_native_bidkv_score_triplet_r01.py'
plan=json.loads(Path('NATIVE_SELF_PREEMPT_CONTINUE_TRIPLET_PLAN_R01_20261002.json').read_text())
deps=plan['controller_dependencies']+['run_native_self_preempt_continue_triplet_r01.py']
for name in [ctrl,*deps]:ast.parse(Path(name).read_text())
stage=Path('/private/tmp/moe-a-native-bidkv-score-stage-r01-20261002')
remote='/root/'+stage.name
session='/root/moe-a-native-bidkv-score-session-r01-20261002'
plan.update(
    session_dir=session, measurement_scope='CLOSE_PUBLIC_SCORE_ADAPTATION_DEVELOPMENT_PROBE',
    hypothesis='Determine whether the official default BidKV score, in the same qualified native candidate set, improves over native tail within the fixed service budgets.',
    action_difference='Tail_break versus exact default BidKV score and tie order with the native running exit retained. Bidkv_continue additionally tests successor continuation under this different victim rule; it is a separate compatibility control, not a rescue of the failed density-continuation hypothesis.',
    mechanism_criteria='All128complete, nativefull, noforcedrotation, allvictimdecisions match actualrawpreemption. BidKV break must make at least one changed actual victim choice. Continue arm reports originalsuccessor visits, positive suffix tokens, subsequent outputs and source scheduled zero separately.',
    score_baseline_criteria='PRIMARY: bidkv_break versus tail_break has rate>=97%, meanflow<=105%, strictlylower maximum requestgap, plus validity and changedactualvictim. Report complete20pointgoodput and allrequests regardless of pass.',
    service_criteria='SECONDARY: bidkv_continue versus BOTH controls uses the same97% rate,105% meanflow and strictlylower maximumgap criteria, together with actual useful continuation evidence. Failure does not alter the primary score comparison.',
    retention_rule='One tail_break,bidkv_break,bidkv_continue triplet, allresultsretained. Default source coefficients0.5,0.3,epsilon1e-6, exactarrival/requestID ties. No tuning or seed/input replacement. Same now-seen128 requests; this is baseline qualification, not fresh confirmation.',
    claim_ceiling='Restricted score adaptation, not complete BidKV reproduction. Different runtime/backend/candidate restrictions are explicit. No new method claim or same-state causal effect.',
    source='bidkv_pinned_sources_20261002/SOURCE.json',
    source_commit='5ee80256d263d58b1e512d9d436d47e9bac564ba',
    package_manifest_sha256=msha,controller_sha256=sha(ctrl),
    controller_dependencies=deps,controller_dependency_sha256={name:sha(name) for name in deps},
    analysis='analyze_native_bidkv_score_triplet_r01.py',
    frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    cells=[dict(arm=arm,package_dir=remote+'/'+label+'/'+candidate.name,
                output_dir='/root/moe-a-native-bidkv-score-output-'+label+'-r01-20261002',max_wall_seconds=900)
           for label,arm in zip(['first','second','third'],['tail_break','bidkv_break','bidkv_continue'])])
pname='NATIVE_BIDKV_SCORE_TRIPLET_PLAN_R01_20261002.json'
assert not Path(pname).exists() and not stage.exists()
write(pname,plan);stage.mkdir()
for label in ['first','second','third']:shutil.copytree(candidate,stage/label/candidate.name)
for name in [ctrl,pname,*deps]:shutil.copyfile(name,stage/name)
archive=stage.with_suffix('.tar.gz');assert not archive.exists()
with tarfile.open(archive,'w:gz') as target:target.add(stage,arcname=stage.name)
meta=dict(status='FROZEN_GPU_UNRUN',manifest=msha,controller=ctrl,controller_sha=sha(ctrl),
          plan=pname,plan_sha=sha(pname),stage=str(stage),remote_stage=remote,remote_session=session,
          archive=str(archive),archive_sha=sha(archive),archive_bytes=archive.stat().st_size)
write('NATIVE_BIDKV_SCORE_FROZEN_R01_20261002.json',meta)
print(json.dumps(meta))
