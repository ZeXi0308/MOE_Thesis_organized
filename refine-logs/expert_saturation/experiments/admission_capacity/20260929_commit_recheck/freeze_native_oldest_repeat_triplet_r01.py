#!/usr/bin/env python3
"""Freeze the single seen-input repeated-policy pilot before outcomes."""
import argparse, ast, datetime, hashlib, json, shutil, tarfile
from pathlib import Path

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--manifest-sha256', required=True)
parser.add_argument('--analysis-sha256', required=True)
args = parser.parse_args()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
package = root / 'candidate_native_oldest_repeat_r01'
assert sha(package / 'manifest.json') == args.manifest_sha256
for name, digest in json.loads((package / 'manifest.json').read_text()).items():
    assert sha(package / name) == digest, name
analysis = root / 'analyze_native_oldest_repeat_triplet_r01.py'
assert sha(analysis) == args.analysis_sha256
controller = root / 'run_native_oldest_repeat_triplet_r01.py'
s = controller.read_text()
assert s.count('PENDING_CANDIDATE_MANIFEST_DO_NOT_RUN') == 1
s = s.replace('PENDING_CANDIDATE_MANIFEST_DO_NOT_RUN', args.manifest_sha256)
ast.parse(s)
controller.write_text(s)
plan = json.loads((root / 'NATIVE_OLDEST_ADMISSION_TRIPLET_PLAN_R02_20261002.json').read_text())
plan.pop('revision_basis', None)
plan.update(
    session_dir='/root/moe-a-native-oldest-repeat-session-r01-20261002',
    measurement_scope='SEEN_INPUT_REPEATED_SIMPLE_POLICY_HEADROOM_PILOT',
    hypothesis='Repeated oldest paused admission with real capacity funding reduces full-cohort stalls versus native and queue-only without material output-rate or mean-flow cost.',
    action_difference='Same native-full/tail backend, fixed1s observed-output age, single qualified private running victim and Q1 hold. Repeated episodes: at most one active episode; retire on first new output or terminal, or cancellation; a cancelled same pause is not retried. New pauses may be selected. Native shadows, queue-only moves to head, fund additionally uses rechecked actual capacity transfer and native admission. All unrelated policies remain off.',
    mechanism_criteria='All128 complete, no failures or unfinished; native-full saving and identical requested settings. All event IDs and pause identities consistent; no cancelled same-pause retry or overlapping active episodes. Native/queue-only forced0 and no protection. Fund at least2 actual native forced preemptions across at least2 target sources; every actual commit joins native admission, first real output or terminal, legal Q1 lifecycle and displaced-victim completion. Every native victim choice joins raw preemption. Report no-action/cancellation as such.',
    service_criteria='Primary full-cohort target: fund per-request maximum-generation-gap p95 strictly below BOTH native and queue-only, actual output token rate at least97% and mean completed flow at most105% of BOTH. Report all128 per-request changes, maxgap median/p90/p95/global max, TTFT/completion, all20 fixed goodput points, all actual output totals/sequence/stop differences. No target-matched single-action criterion, no event-as-independent-replicate inference.',
    retention_rule='One serial block ordered queue_fund,queue_only,native, reversing prior pilot order. Same already-seen128 inputs; preserve every outcome. No threshold, seed or failed-cell rescue. Subsequent work requires a new stated hypothesis or fresh confirmation of frozen successful policy.',
    claim_ceiling='Exploratory native repeated simple-policy headroom and complete displaced-request costs. Not statistically stable benefit, identical-state counterfactual, hard stall bound, equal-work speedup, novelty or paper readiness. Closest known priority/funding/protection overlap remains.',
    source='A_NATIVE_OLDEST_ADMISSION_TRIPLET_RESULT_R03_20261002.json: actual single funding action and first output; repeated cost transfer remains untested.',
    package_manifest_sha256=args.manifest_sha256,
    controller_entry=controller.name, controller_sha256=sha(controller),
    analysis=analysis.name, analysis_sha256=args.analysis_sha256,
    analysis_status='FROZEN_BEFORE_ANY_REPEATED_GPU_OUTCOMES',
    frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
plan['controller_dependency_sha256'] = {n: sha(root/n) for n in plan['controller_dependencies']}
stage_name = 'moe-a-native-oldest-repeat-stage-r01-20261002'
plan['cells'] = [dict(arm=arm, package_dir='/root/'+stage_name+'/'+label+'/'+package.name,
    output_dir='/root/moe-a-native-oldest-repeat-output-'+label+'-r01-20261002', max_wall_seconds=900)
    for label,arm in zip(('first','second','third'),('queue_fund','queue_only','native'))]
plan_path = root / 'NATIVE_OLDEST_REPEAT_TRIPLET_PLAN_R01_20261002.json'
assert not plan_path.exists()
plan_path.write_text(json.dumps(plan,indent=2)+'\n')
stage = Path('/private/tmp') / stage_name
archive = Path(str(stage)+'.tar.gz')
assert not stage.exists() and not archive.exists()
stage.mkdir()
for name in (controller.name,plan_path.name,*plan['controller_dependencies']):
    shutil.copy2(root/name,stage/name)
for label in ('first','second','third'):
    shutil.copytree(package,stage/label/package.name)
for p in stage.rglob('*.py'): ast.parse(p.read_text())
with tarfile.open(archive,'w:gz') as tar: tar.add(stage,arcname=stage.name)
print(json.dumps(dict(plan=str(plan_path),plan_sha256=sha(plan_path),controller_sha256=sha(controller),
    analysis_sha256=args.analysis_sha256,manifest_sha256=args.manifest_sha256,
    archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size)))
