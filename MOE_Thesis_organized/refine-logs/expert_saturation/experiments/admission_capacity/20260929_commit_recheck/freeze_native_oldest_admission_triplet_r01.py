#!/usr/bin/env python3
"""Freeze one recovery-admission unit after candidate and analyzer exist."""
import argparse, ast, datetime, hashlib, json, shutil, tarfile
from pathlib import Path

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--manifest-sha256", required=True)
parser.add_argument("--analysis-sha256")
args = parser.parse_args()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
package = root / "candidate_native_oldest_admission_r01"
assert sha(package / "manifest.json") == args.manifest_sha256
for name, digest in json.loads((package / "manifest.json").read_text()).items():
    assert sha(package / name) == digest, name
analysis = root / "analyze_native_oldest_admission_triplet_r01.py"
if args.analysis_sha256:
    assert sha(analysis) == args.analysis_sha256
controller = root / "run_native_oldest_admission_triplet_r01.py"
s = controller.read_text()
assert s.count("PENDING_CANDIDATE_MANIFEST_DO_NOT_RUN") == 1
s = s.replace("PENDING_CANDIDATE_MANIFEST_DO_NOT_RUN", args.manifest_sha256)
ast.parse(s)
controller.write_text(s)
plan = json.loads((root / "NATIVE_COMPLETION_DEFERRAL_TRIPLET_PLAN_R01_20261002.json").read_text())
for obsolete in ("opportunity", "resource_retry"):
    plan.pop(obsolete, None)
plan.update(
    session_dir="/root/moe-a-native-oldest-admission-session-r01-20261002",
    measurement_scope="SEEN_INPUT_ONE_ACTION_RECOVERY_ADMISSION_MECHANISM_PILOT",
    hypothesis="For an already-output paused request, queue priority alone can leave its full native restoration unfunded; one bounded real capacity transfer may reduce time to its next output. All displaced-request and full-episode costs remain in the comparison.",
    action_difference="Native shadow anchor versus one queue-head movement versus same target selection plus prepare/commit recheck, at most one native victim preemption, native restoration and protection through first real new output. All arms disable legacy ordinary backfill and tracker rotations; run.sh ordinary-only variant is a compatibility entry and effective configuration records this. Only first eligible oldest paused target with observed-output age >=1s is selected. Full required capacity exceeds free pages but is covered by at most one qualified private running victim. One selection per arm, no replacement target after cancellation, no repeated policy.",
    trigger_basis="1s is a fixed research activation threshold, not a production SLO. Online age is measured from begin observing a request output-count increase, not from a client callback; this conservative observation time is logged. It separates a stalled stream from routine tens-of-milliseconds decode gaps observed in prior raw data. No threshold search follows this pilot.",
    mechanism_criteria="All128 requests complete per arm; same native-full save backend, effective ordinary_backfill false and legacy tracker off in all arms; no unrelated selector/deferral/continuation change. At most one selected target. Native and queue_only forced0 and no target protection; funded arm exactly one native actual preemption, full target native readmission, actual new output and eventual target/victim completion. Record cancellation or no action as such, without retrying selection.",
    service_criteria="Single-action headroom: only if all three online anchors select the same source request, compare its last-output-to-first-post-anchor-output gap and anchor-to-output delay. Funded must improve both delays versus native and queue_only, with actual output rate >=97% and mean flow <=105% of both. If anchors differ mark UNPAIRED, no same-target utility verdict. Report global maxgap, all128 per-request effects,20 fixed goodput points,natural sequence/stop/output-count differences regardless. This one-shot unit is not a repeated-policy or stable-performance confirmation.",
    retention_rule="One block ordered native,queue_only,queue_fund on same already-seen128 inputs. All outcomes retained; no target, threshold, seed, or failed-cell rescue. Cancel/no-action does not prove service failure; it identifies unavailable action. Positive descriptive result requires further original-scope evidence.",
    claim_ceiling="Native action and bounded one-action service headroom only. Target identity alone is not a common physical pre-state; independent runs remain policy-level observations. Queue order plus temporary protection is part of funded intervention cost. Not novelty, a hard stall bound, an Oracle bound, or a paper-ready method.",
    source="A_NATIVE_OLDEST_PAUSED_ADMISSION_NEXT_STEP_R01_20261002.json; pinned scheduler native restoration and existing prepare/commit implementation",
    package_manifest_sha256=args.manifest_sha256,
    controller_sha256=sha(controller), controller_entry=controller.name,
    analysis=analysis.name, analysis_sha256=args.analysis_sha256,
    analysis_status=("SOURCE_FROZEN_BEFORE_GPU_OUTCOMES" if args.analysis_sha256 else
                     "IMPLEMENTING_IN_PARALLEL_FIXED_CRITERIA; freeze source before reading performance or mechanism outcomes"),
    frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
plan["closest_method_boundary"] = {
    "CacheOPT_v2": {"source": "https://arxiv.org/html/2503.13773v2#S3.SS2", "section": "3.3.2 and 3.4", "overlap": "Prioritizes preempted requests by TBT slack and preempts running requests to fund critical requests. This pilot is not a reproduction and priority-plus-funding is not a new action."},
    "UniBoost": {"source": "https://arxiv.org/html/2606.18431#S3.SS3", "section": "3.2, 3.3 and Appendix Algorithm1", "overlap": "Unified running/waiting priorities with eviction; MemGuard protects dispatch until next attained-service threshold. Useful service after restoration already has a close method."},
    "scope": "A one-action mechanism/headroom test before any method claim. A positive result would require direct comparison with these closest policies and an independently supported difference."}
plan["controller_dependency_sha256"] = {n: sha(root/n) for n in plan["controller_dependencies"]}
stage_name = "moe-a-native-oldest-admission-stage-r01-20261002"
plan["cells"] = [dict(arm=arm,package_dir="/root/"+stage_name+"/"+label+"/"+package.name,
                     output_dir="/root/moe-a-native-oldest-admission-output-"+label+"-r01-20261002",max_wall_seconds=900)
                 for label, arm in zip(("first","second","third"),("native","queue_only","queue_fund"))]
plan_path = root / "NATIVE_OLDEST_ADMISSION_TRIPLET_PLAN_R01_20261002.json"
assert not plan_path.exists()
plan_path.write_text(json.dumps(plan, indent=2)+"\n")
stage = Path("/private/tmp") / stage_name
archive = Path(str(stage)+".tar.gz")
assert not stage.exists() and not archive.exists()
stage.mkdir()
for name in (controller.name,plan_path.name,*plan["controller_dependencies"]):
    shutil.copy2(root/name,stage/name)
for label in ("first","second","third"):
    shutil.copytree(package,stage/label/package.name)
for p in stage.rglob("*.py"):
    ast.parse(p.read_text())
with tarfile.open(archive,"w:gz") as tar:
    tar.add(stage,arcname=stage.name)
print(json.dumps(dict(plan=str(plan_path),plan_sha256=sha(plan_path),controller_sha256=sha(controller),
                     analysis_sha256=args.analysis_sha256,manifest_sha256=args.manifest_sha256,
                     archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size)))
