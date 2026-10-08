#!/usr/bin/env python3
"""Freeze the already selected three-arm deferral pilot once candidate is ready."""
import argparse, ast, datetime, hashlib, json, shutil, tarfile
from pathlib import Path

root=Path(__file__).resolve().parent
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
parser=argparse.ArgumentParser()
parser.add_argument("--manifest-sha256",required=True)
parser.add_argument("--analysis-sha256",required=True)
args=parser.parse_args()
package=root/"candidate_native_completion_deferral_r01"
assert sha(package/"manifest.json")==args.manifest_sha256
manifest=json.loads((package/"manifest.json").read_text())
for name,digest in manifest.items(): assert sha(package/name)==digest,name
analysis=root/"analyze_native_completion_deferral_triplet_r01.py"
assert sha(analysis)==args.analysis_sha256
controller=root/"run_native_completion_deferral_triplet_r01.py"
text=controller.read_text()
assert text.count("PENDING_CANDIDATE_MANIFEST_DO_NOT_RUN")==1
text=text.replace("PENDING_CANDIDATE_MANIFEST_DO_NOT_RUN",args.manifest_sha256)
ast.parse(text)
controller.write_text(text)
plan=json.loads((root/"NATIVE_BIDKV_FULL_RUNNING_TRIPLET_PLAN_R01_20261002.json").read_text())
plan.update(
 session_dir="/root/moe-a-native-completion-deferral-session-r01-20261002",
 measurement_scope="SEEN_INPUT_NATIVE_CAPACITY_DEFERRAL_MECHANISM_PILOT",
 hypothesis="At a native KV allocation failure, defer the failed current request without eviction when an already scheduled pure-decode prefix request can finish within its already owned KV capacity. Compare with unconditional scheduled-prefix-work deferral and native tail preemption.",
 action_difference="All arms use native tail, native full saving, ordinary backfill and Q1, with full-running BidKV, current guard and self-preempt continuation off. off retains native allocation/preemption. prefix_work skips the failed current request if any qualified already-scheduled prefix will do decode work. prefix_finish additionally requires positive remaining hard output cap to fit held_blocks*16-computed_tokens, choosing minimum remaining then arrival/request ID. Advance req_index, preserve current KV and scheduled prefix; no persistent quantum or hold.",
 mechanism_criteria="All three complete128/native-full/forced0. Tail has no deferrals; both work and finish have actual actions. Every deferral independently joins final native plan and raw events: failed current not preempted/no same-call output, positive scheduled prefix witness output/no same-call preemption, earlier index and safe pure-decode state. Finish witness remaining cap and physical capacity fit at decision. Retain later witness preemption/completion and every failed-current delay.",
 service_criteria="Separate from mechanism qualification: prefix_finish versus BOTH tail and prefix_work, actual output rate >=97%, mean flow <=105%, strictly lower maximum request gap. Report all128 requests, 20 fixed goodput points, per-request improvements/regressions, natural sequence/stop changes and actor costs.",
 retention_rule="One fixed block tail_break,prefix_work,prefix_finish in that order. Same already-seen128 inputs. Preserve all outcomes; no threshold/seed/input tuning. A useful pilot requires fresh confirmation and closest-method comparison before a paper performance claim.",
 claim_ceiling="Native action pilot on seen data, not novel-by-definition and not a hard completion or no-starvation guarantee. CacheOPT already uses predicted completion releases; owned-capacity hard-cap gating and allocation-failure deferral need separate contribution and performance evidence.",
 package_manifest_sha256=args.manifest_sha256,
 controller_sha256=sha(controller),
 controller_entry=controller.name,
 analysis=analysis.name,
 analysis_sha256=args.analysis_sha256,
 frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 authorization_reference="User authorizes both hourly GPU machines, installation/download and direct short experiments. This locked three-cell block is bounded 4800s total,900s/cell,90GiB; no resource purchases.",
 resource_retry="One new session, 1.25 GiB system-disk reserve after three private copies of the completed A runtime cache. No automatic retry after any cell starts.",
 source="A_NATIVE_COMPLETION_DEFERRAL_SOURCE_BOUNDARY_R01_20261002.json",
 opportunity="A_NATIVE_COMPLETION_FUNDED_DEFERRAL_OPPORTUNITY_R01_20261002.json",
)
plan.pop("source_commit",None)
plan["controller_dependencies"].append("run_native_bidkv_full_running_triplet_r01.py")
plan["controller_dependency_sha256"]={name:sha(root/name) for name in plan["controller_dependencies"]}
stage_name="moe-a-native-completion-deferral-stage-r01-20261002"
plan["cells"]=[dict(arm=arm,package_dir="/root/"+stage_name+"/"+label+"/"+package.name,output_dir="/root/moe-a-native-completion-deferral-output-"+label+"-r01-20261002",max_wall_seconds=900)
 for label,arm in zip(("first","second","third"),("tail_break","prefix_work","prefix_finish"))]
plan_path=root/"NATIVE_COMPLETION_DEFERRAL_TRIPLET_PLAN_R01_20261002.json"
assert not plan_path.exists()
plan_path.write_text(json.dumps(plan,indent=2)+"\n")
stage=Path("/private/tmp")/stage_name
archive=Path(str(stage)+".tar.gz")
assert not stage.exists() and not archive.exists()
stage.mkdir()
for name in (controller.name,plan_path.name,*plan["controller_dependencies"]):
 shutil.copy2(root/name,stage/name)
for label in ("first","second","third"):
 shutil.copytree(package,stage/label/package.name)
for f in stage.rglob("*.py"): ast.parse(f.read_text())
with tarfile.open(archive,"w:gz") as tar: tar.add(stage,arcname=stage.name)
print(json.dumps(dict(plan=str(plan_path),plan_sha256=sha(plan_path),controller_sha256=sha(controller),analysis_sha256=sha(analysis),package_manifest_sha256=args.manifest_sha256,stage=str(stage),archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size)))

