#!/usr/bin/env python3
import argparse,ast,datetime,hashlib,json,shutil,tarfile
from pathlib import Path
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument("--manifest-sha256",required=True)
parser.add_argument("--analysis-sha256")
args=parser.parse_args()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
package=root/"candidate_native_victim_size_r01"
assert sha(package/"manifest.json")==args.manifest_sha256
for name,digest in json.loads((package/"manifest.json").read_text()).items():
 assert sha(package/name)==digest,name
analysis=root/"analyze_native_victim_size_triplet_r01.py"
if args.analysis_sha256:
 assert sha(analysis)==args.analysis_sha256
controller=root/"run_native_victim_size_triplet_r01.py"
s=controller.read_text();assert s.count("PENDING_CANDIDATE_MANIFEST_DO_NOT_RUN")==1
s=s.replace("PENDING_CANDIDATE_MANIFEST_DO_NOT_RUN",args.manifest_sha256)
ast.parse(s);controller.write_text(s)
plan=json.loads((root/"NATIVE_COMPLETION_DEFERRAL_TRIPLET_PLAN_R01_20261002.json").read_text())
plan.update(session_dir="/root/moe-a-native-victim-size-session-r01-20261002",
 measurement_scope="SEEN_INPUT_NATIVE_VICTIM_SIZE_PILOT",
 hypothesis="At single-decode page-allocation failure, evicting the smallest sufficient noncurrent suffix request may reduce its later physical readmission requirement and exposed recovery delay relative to native tail or maximum-sized victim. Reclaiming less may also cause more preemptions; count all costs.",
 action_difference="Native tail versus min_held_other versus max_held_other. Same native full saving, ordinary backfill, FCFS and Q1; full-running, guard, continuation and capacity-deferral off. Only choose among known unshared owned pure-decode noncurrent unprocessed suffix requests with held pages at least the computed allocation deficit. Min or max held size, ties later suffix index; ambiguous state or absent legal candidate falls back to native tail. Native preempt/retry remains unchanged.",
 mechanism_criteria="All three complete128/native-full/forced0. Both size rules change at least one native choice. Every changed choice has sufficient physical pages at decision, native preempt in matching raw call and positive failed-current output after that same-call retry; victim has no same-call output and later completion retained. Record all fallback/self-preempt events, later victim readmission/output delays and native preemption counts.",
 service_criteria="Minimum-sized victim versus BOTH native tail and maximum-sized victim: actual output rate >=97%, mean flow <=105%, strictly lower maximum request gap. Keep all128 per-request effects,20 fixed goodput points,natural sequence/stop/output-count changes. Descriptive seen pilot only.",
 retention_rule="One block ordered tail,min_held_other,max_held_other on the same already-seen128 inputs. No parameter/threshold/seed replacement or failed-cell retry. Failure stops unchanged-rule fresh expansion.",
 claim_ceiling="A direct test of a simple physical-size selector, not an established new algorithm or cost model. Host coverage, actual transfer completion and same-state counterfactual service are not inferred from page count.",
 opportunity="A_NATIVE_VICTIM_SIZE_OPPORTUNITY_R01_20261002.json",
 source="Pinned native scheduler and existing candidate-native path; size-only eviction comparison.",
 package_manifest_sha256=args.manifest_sha256,controller_sha256=sha(controller),
 controller_entry=controller.name,analysis=analysis.name,analysis_sha256=args.analysis_sha256,
 analysis_status=("SOURCE_FROZEN" if args.analysis_sha256 else
 "IMPLEMENTING_IN_PARALLEL_GOVERNED_BY_THIS_FROZEN_PLAN; source hash recorded separately before reading outcomes"),
 frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
plan["controller_dependency_sha256"]={name:sha(root/name) for name in plan["controller_dependencies"]}
stage_name="moe-a-native-victim-size-stage-r01-20261002"
plan["cells"]=[dict(arm=arm,package_dir="/root/"+stage_name+"/"+label+"/"+package.name,output_dir="/root/moe-a-native-victim-size-output-"+label+"-r01-20261002",max_wall_seconds=900)
 for label,arm in zip(("first","second","third"),("tail","min_held_other","max_held_other"))]
plan_path=root/"NATIVE_VICTIM_SIZE_TRIPLET_PLAN_R01_20261002.json"
assert not plan_path.exists()
plan_path.write_text(json.dumps(plan,indent=2)+"\n")
stage=Path("/private/tmp")/stage_name;archive=Path(str(stage)+".tar.gz")
assert not stage.exists() and not archive.exists();stage.mkdir()
for name in (controller.name,plan_path.name,*plan["controller_dependencies"]):shutil.copy2(root/name,stage/name)
for label in ("first","second","third"):shutil.copytree(package,stage/label/package.name)
for f in stage.rglob("*.py"):ast.parse(f.read_text())
with tarfile.open(archive,"w:gz") as tar:tar.add(stage,arcname=stage.name)
print(json.dumps(dict(plan=str(plan_path),plan_sha256=sha(plan_path),controller_sha256=sha(controller),
 analysis_sha256=args.analysis_sha256,manifest_sha256=args.manifest_sha256,archive=str(archive),
 archive_sha256=sha(archive),archive_bytes=archive.stat().st_size)))
