#!/usr/bin/env python3
"""Freeze one current-retention probe against both measured BidKV controls."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

candidate=Path('candidate_native_current_guard_r01')
msha=sha(candidate/'manifest.json')
assert msha=='a9d0f1d18428149955bf6678a767a25e9dbfb9567ac774cb60812d0dea5ec5ef'
assert all(sha(candidate/name)==digest for name,digest in json.loads((candidate/'manifest.json').read_text()).items())
ctrl='run_native_current_guard_triplet_r01.py'
plan=json.loads(Path('NATIVE_BIDKV_SCORE_TRIPLET_PLAN_R01_20261002.json').read_text())
deps=plan['controller_dependencies']
for name in [ctrl,*deps]:ast.parse(Path(name).read_text())
stage=Path('/private/tmp/moe-a-native-current-guard-stage-r01-20261002')
remote='/root/'+stage.name;session='/root/moe-a-native-current-guard-session-r01-20261002'
plan.pop('score_baseline_criteria')
plan.update(
 session_dir=session,measurement_scope='SEEN_INPUT_CURRENT_VICTIM_EXCLUSION_ACTION_PROBE',
 hypothesis='Keeping the allocation-failed current request resident when a qualified alternative victim exists avoids creating an underfunded self-readmission, and reduces maximum gap within efficiency budgets versus both BidKV controls.',
 action_difference='Same official default BidKV score and candidate qualification. Guard excludes current only when unrestricted score selects current and another qualified unprocessed successor exists; re-ranks those alternatives. Unknown/singleton preserves native tail. Guard arm retains native break and allocation retry; no continuation, addedholds orforcedrotation.',
 mechanism_criteria='All128complete/nativefull/forced0/rawvictimdecisionsmatched. Atleastone guard-changed victim is actually preempted; current is scheduled for positive tokens and produces same-call output. Selected victim has zero same-call scheduled tokens; retain any current later-preempted failures and all displaced-victim outcomes.',
 service_criteria='Guard versus BOTH bidkv_break and bidkv_continue: actualoutputrate>=97%,meanflow<=105%,strictlylower maximumrequestgap. Full20goodput points and128request improvements/regressions retained. Guard action without servicecriteria is failure.',
 retention_rule='One triplet bidkv_break,bidkv_continue,bidkv_current_guard; allresultsretained, no tuning/input/seed replacement. Same already-seen128 inputs; positivepilot is not confirmation. Current-exclusion is a different action from the rejected densitycontinuation hypothesis.',
 claim_ceiling='New constrained victim action, not proof of independent novelty or a guaranteed stall bound. It moves interruptioncost to an alternative victim. No crossblock native comparison as an exact paired result.',
 motivation='A_NATIVE_CONTINUATION_MAX_GAP_CHAIN_R01_20261002.json',
 package_manifest_sha256=msha,controller_sha256=sha(ctrl),controller_dependency_sha256={name:sha(name) for name in deps},
 analysis='analyze_native_current_guard_triplet_r01.py',
 frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 cells=[dict(arm=arm,package_dir=remote+'/'+label+'/'+candidate.name,output_dir='/root/moe-a-native-current-guard-output-'+label+'-r01-20261002',max_wall_seconds=900)
        for label,arm in zip(['first','second','third'],['bidkv_break','bidkv_continue','bidkv_current_guard'])])
pname='NATIVE_CURRENT_GUARD_TRIPLET_PLAN_R01_20261002.json'
assert not Path(pname).exists() and not stage.exists()
write(pname,plan);stage.mkdir()
for label in ['first','second','third']:shutil.copytree(candidate,stage/label/candidate.name)
for name in [ctrl,pname,*deps]:shutil.copyfile(name,stage/name)
archive=stage.with_suffix('.tar.gz');assert not archive.exists()
with tarfile.open(archive,'w:gz') as target:target.add(stage,arcname=stage.name)
meta=dict(status='FROZEN_GPU_UNRUN',manifest=msha,controller=ctrl,controller_sha=sha(ctrl),plan=pname,plan_sha=sha(pname),stage=str(stage),remote_stage=remote,remote_session=session,archive=str(archive),archive_sha=sha(archive),archive_bytes=archive.stat().st_size)
write('NATIVE_CURRENT_GUARD_FROZEN_R01_20261002.json',meta)
print(json.dumps(meta))
