#!/usr/bin/env python3
"""Freeze the seen-input strong ordinary baseline challenge before outcomes."""
import argparse, ast, datetime, hashlib, json, shutil, tarfile
from pathlib import Path

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--manifest-sha256', required=True)
parser.add_argument('--analysis-sha256', required=True)
args = parser.parse_args()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
package = root / 'candidate_native_oldest_strong_r01'
assert sha(package / 'manifest.json') == args.manifest_sha256
for name, digest in json.loads((package / 'manifest.json').read_text()).items():
    assert sha(package / name) == digest, name
analysis = root / 'analyze_native_oldest_strong_triplet_r01.py'
assert sha(analysis) == args.analysis_sha256
controller = root / 'run_native_oldest_strong_triplet_r01.py'
s = controller.read_text()
assert s.count('PENDING_CANDIDATE_MANIFEST_DO_NOT_RUN') == 1
s = s.replace('PENDING_CANDIDATE_MANIFEST_DO_NOT_RUN', args.manifest_sha256)
ast.parse(s)
controller.write_text(s)
plan = json.loads((root / 'NATIVE_OLDEST_REPEAT_TRIPLET_PLAN_R02_20261002.json').read_text())
plan.pop('revision_basis', None)
plan.pop('resource_retry', None)
plan.update(
    session_dir='/root/moe-a-native-oldest-strong-session-r01-20261002',
    measurement_scope='SEEN_INPUT_STRONG_ORDINARY_BACKFILL_CHALLENGE',
    hypothesis='The repeated capacity-funding benefit persists beyond the established strong free-fit ordinary-backfill baseline.',
    action_difference='All native-full/tail. Native shadows repeated oldest episodes with no interventions. Ordinary uses the previously implemented free-fit waiting admission policy, no forced preemption, no oldest recovery/protection or legacy tracker. Fund preserves the exact repeated1s oldest/single-victim/Q1 algorithm with ordinary disabled. Only enable and expose the existing strong ordinary baseline; no candidate rule or threshold change.',
    mechanism_criteria='All128 complete, no failure/unfinished; native-full/tail/no unrelated selectors. Native/ordinary forced0 and no protection. Ordinary must have at least1 actual free-fit native admission and correct ordinary enabled identity. Fund repeated episode qualification unchanged: at least2 forced actions/2target sources, actual raw preemption/native admission/newoutput or terminal/Q1, allvictimcompletion, no samepause retry. Allnative decisions join raw.',
    service_criteria='Primary full-cohort target: fund per-request maximum-generation-gap p95 strictly below BOTH native and ordinary, actual output token rate at least97% and mean completed flow at most105% of BOTH. Report all128 per-request changes, maxgap median/p90/p95/global max, TTFT/completion, all20 fixed goodput points, all actual output totals/sequence/stop differences. No target-matched single-action criterion, no event-as-independent-replicate inference.',
    retention_rule='One serial block ordered ordinary,queue_fund,native. Same already-seen128 inputs; preserve every outcome. No threshold, seed or failed-cell rescue. Subsequent work requires a new stated hypothesis or fresh confirmation of frozen successful policy.',
    claim_ceiling='Exploratory native repeated simple-policy headroom and complete displaced-request costs. Not statistically stable benefit, identical-state counterfactual, hard stall bound, equal-work speedup, novelty or paper readiness. Closest known priority/funding/protection overlap remains.',
    source='A_NATIVE_OLDEST_REPEAT_TRIPLET_RESULT_R02_20261002.json: repeated mechanism and seen-input service positive; stronger ordinary baseline still untested.',
    package_manifest_sha256=args.manifest_sha256,
    controller_entry=controller.name, controller_sha256=sha(controller),
    analysis=analysis.name, analysis_sha256=args.analysis_sha256,
    analysis_status='FROZEN_BEFORE_ANY_STRONG_BASELINE_GPU_OUTCOMES',
    frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
plan['controller_dependencies'] = [n for n in plan['controller_dependencies'] if n!='run_native_oldest_repeat_triplet_r01.py']
plan['controller_dependency_sha256'] = {n: sha(root/n) for n in plan['controller_dependencies']}
stage_name = 'moe-a-native-oldest-strong-stage-r01-20261002'
plan['cells'] = [dict(arm=arm, package_dir='/root/'+stage_name+'/'+label+'/'+package.name,
    output_dir='/root/moe-a-native-oldest-strong-output-'+label+'-r01-20261002', max_wall_seconds=900)
    for label,arm in zip(('first','second','third'),('ordinary','queue_fund','native'))]
plan_path = root / 'NATIVE_OLDEST_STRONG_TRIPLET_PLAN_R01_20261002.json'
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
