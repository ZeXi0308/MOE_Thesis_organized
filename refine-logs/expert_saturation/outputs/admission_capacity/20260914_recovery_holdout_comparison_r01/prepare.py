#!/usr/bin/env python3
"""Freeze the cohort3 native/most-output/fit/residual comparison; CPU only."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "AGENTS.md").is_file())
OUT_NAME = "preparation"
T = REPO / "refine-logs/expert_saturation/outputs/admission_capacity/20260914_restore_token_reservation_r01"
D = REPO / "refine-logs/expert_saturation/outputs/admission_capacity/20260914_d6_strong_baselines_r01"
H = REPO / "refine-logs/expert_saturation/outputs/admission_capacity/20260914_recovery_holdout_inputs_r01"
HOLDOUT_PREPARATION_SHA = "4775c84082d6afb6067b8060cda336e23fdb2a7d776fe70f72be4d09315c6c18"
COMMON = ("absence_rotation.py", "completion_headroom.py", "memory_telemetry.py",
          "metrics.py", "native_capture.py", "rotation_native.py", "safe_static.py")
ORDER = ("native", "most_output", "fit_scan", "guard_residual")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def change(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"patch anchor count={text.count(old)}: {old!r}")
    return text.replace(old, new)


def load(name, path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def actual_compatibility():
    """Compare retained successful d6/T runtime records before composing the wrapper."""
    d_result = D / "readback/results/block0-d6-most_output"
    d_engine, d_config, d_env, d_kv = [json.loads((d_result / name).read_text()) for name in
        ("engine_args.json", "config.json", "environment.json", "safe-cap-qualification.json")]
    t_result = T / "execution/readback/results/block0-fit_scan"
    t_engine, t_config, t_env, t_kv = [json.loads((t_result / name).read_text()) for name in
        ("engine_args.json", "config.json", "environment.json", "safe-cap-qualification.json")]
    engine_diff = {key: [d_engine.get(key), t_engine.get(key)] for key in set(d_engine) | set(t_engine)
                   if d_engine.get(key) != t_engine.get(key)}
    assert engine_diff == {}
    versions = ("python", "torch", "cuda", "vllm", "transformers")
    assert all(d_env[key] == t_env[key] for key in versions)
    assert d_env["vllm_source_sha256"] == t_env["vllm_source_sha256"]
    measurement = ("requests", "prompt_tokens", "output_tokens", "arrival_gap_s", "cap",
                   "domain", "max_seconds", "reservation_policy", "preemption_mode")
    assert all(d_config[key] == t_config[key] for key in measurement)
    assert (d_kv["usable_blocks"], t_kv["usable_blocks"]) == (6656, 6656)
    return dict(status="PASS", actual_records=dict(
            d6_strong_most_output_cell=str(d_result), token_reservation_fit_scan_cell=str(t_result)),
        engine_args=dict(common_fields=len(t_engine), differences=engine_diff, exactly_equal=True,
            target_source="shared 6656-block engine args"),
        runtime=dict(versions={key: t_env[key] for key in versions},
            vllm_source_sha256=t_env["vllm_source_sha256"], exact_runtime_sources_equal=True),
        measurement_fields={key: t_config[key] for key in measurement},
        resource_boundary=dict(strong_source_usable_blocks=6656, target_usable_blocks=6656,
            exact_match=True, implication="new cohort still requires new runtime measurements"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", default=OUT_NAME)
    args = parser.parse_args()
    if args.attempt != OUT_NAME and not args.attempt.startswith("preparation_attempt"):
        raise ValueError("attempt must be preparation or preparation_attempt*")
    out = HERE / args.attempt
    if out.exists():
        raise FileExistsError(f"refuse overwrite: {out}")

    t_meta = json.loads((T / "preparation/preparation.json").read_text())
    d_meta = json.loads((D / "preparation/status.json").read_text())
    d_final = json.loads((D / "STATUS.json").read_text())
    h_meta = json.loads((H / "PREPARATION.json").read_text())
    assert sha(H / "PREPARATION.json") == HOLDOUT_PREPARATION_SHA
    assert sha(T / "preparation/execution.tar.gz") == t_meta["archive_sha256"]
    assert sha(D / "preparation/package.tar.gz") == d_final["package_sha256"]
    t_pkg = T / "preparation/pkg"
    assert all(sha(t_pkg / name) == digest for name, digest in t_meta["files_sha256"].items())
    d_source = D / "preparation/pkg"
    assert all(sha(d_source / name) == sha(t_pkg / name) for name in COMMON)
    assert all(d_meta["files"][name] == sha(t_pkg / name) for name in COMMON)
    compatibility = actual_compatibility()

    out.mkdir()
    pkg = out / "pkg"
    pkg.mkdir()
    for name in t_meta["files_sha256"]:
        target = pkg / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(t_pkg / name, target)

    # Only the measured long cohort changes. Common short warmups stay byte-identical to T.
    holdout = H / "prepared"
    old_config = json.loads((pkg / "inputs_preparation/prepared/long/config.json").read_text())
    new_config = json.loads((holdout / "config.json").read_text())
    assert {key for key in set(old_config) | set(new_config)
            if old_config.get(key) != new_config.get(key)} == {"source", "workload_sha256", "input_preparation"}
    for name in ("config.json", "workload.json"):
        shutil.copy2(holdout / name, pkg / "inputs_preparation/prepared/long" / name)
    workload = json.loads((holdout / "workload.json").read_text())
    assert semantic(workload) == new_config["workload_sha256"]
    assert len(workload["source_requests"]) == len(workload["actual_prompt_token_ids"]) == 32
    assert workload["arrival_traces_s"]["steady"] == [round(i * .05, 10) for i in range(32)]

    receipts = pkg / "source_receipts"
    receipts.mkdir()
    receipt_sources = {
        "holdout_PREPARATION.json": H / "PREPARATION.json",
        "holdout_inputs_report.json": H / "prepared/inputs_report.json",
        "d6_strong_baseline_preparation.json": D / "preparation/status.json",
        "d6_strong_baseline_status.json": D / "STATUS.json",
        "token_reservation_preparation.json": T / "preparation/preparation.json",
        "token_reservation_fixture.json": T / "preparation/fixture-results.json",
    }
    for name, source in receipt_sources.items():
        shutil.copy2(source, receipts / name)
    (receipts / "compatibility.json").write_text(json.dumps(compatibility, indent=2) + "\n")

    probe = pkg / "run_probe.py"
    text = probe.read_text()
    text = change(text, '"""One fixed-pool four-arm request recovery episode."""',
                  '"""One matched cohort3 native/most-output/fit/residual episode."""')
    text = change(text, "from absence_rotation import RotationConfig\nfrom ltr_recompute_native import install as install_component",
                  "from absence_rotation import RotationConfig\nfrom rotation_native import install as install_rotation\nfrom ltr_recompute_native import install as install_component")
    start = text.index("def main():")
    end = text.index("    root, out = Path(__file__).resolve().parent, args.output_dir", start)
    replacement = '''def arm_spec(name):
    specs = {
        "native": dict(adapter="native", policy_family="native", completion_policy="native",
            victim_order="least_progress", boost=False, packing="native",
            complete_restores=False, reserve_ready_tokens=False),
        "most_output": dict(adapter="rotation", policy_family="strong_simple_rotation",
            completion_policy="rotate", victim_order="most_output", boost=False, packing="native",
            complete_restores=False, reserve_ready_tokens=False),
        "fit_scan": dict(adapter="component", policy_family="ltr_component",
            completion_policy="restore_token_reservation_fit_scan", victim_order="least_progress",
            boost=True, packing="fit_scan", complete_restores=False, reserve_ready_tokens=False),
        "guard_residual": dict(adapter="component", policy_family="ltr_component",
            completion_policy="restore_token_reservation_guard_residual", victim_order="least_progress",
            boost=True, packing="rank_prefix", complete_restores=True, reserve_ready_tokens=True),
    }
    return dict(specs[name])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['native','most_output','fit_scan','guard_residual'], required=True)
    parser.add_argument('--domain', choices=['long'], required=True)
    parser.add_argument('--reservation-policy', choices=['full'], required=True)
    parser.add_argument('--cap', type=int, choices=[32], required=True)
    parser.add_argument('--gpu-memory-utilization', type=float, choices=[0.90], required=True)
    parser.add_argument('--kv-cache-bytes', type=int, choices=[13960740864], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    spec = arm_spec(args.variant)
'''
    text = text[:start] + replacement + text[end:]
    old = '''    if args.cap != (29 if args.completion_policy == 'safe29' else 32):
        raise ValueError('cap/policy mismatch')
    config = dict(config, cap=args.cap, domain=args.domain, engine_max_num_seqs=32,
        metric_role='continuous request metrics primary; SLO 5s/.2s reference only',
        evidence_ceiling='NATIVE_INPROCESS_CAPACITY_QUALIFICATION', max_seconds=120)
    config.update(requested_arm='budget90', reservation_policy=args.reservation_policy, completion_policy=args.completion_policy, headroom_observer='fast',
                  fixed_kv_cache_memory_bytes=args.kv_cache_bytes,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  rotation_victim_order=args.victim_order,
                  preemption_mode='native_recompute', rotation_config=vars(RotationConfig()))
    config.update(completion_policy='ltr_component_' + args.fairness_boost,
        component=dict(boost=args.fairness_boost == 'on', threshold=200, quantum=10,
            base_order='FCFS', backend='priority packing / current-history reservation / native recompute'))
    config['completion_policy'] = 'ltr_packing_' + args.packing
    config['component']['packing'] = args.packing
    config['component']['complete_restores'] = args.complete_restores == 'on'
    config['component']['reserve_ready_tokens'] = args.reserve_ready_tokens == 'on'
    variant = 'fit_scan' if args.packing == 'fit_scan' else ('guard_residual' if args.reserve_ready_tokens == 'on' else 'guard_all')
    config['variant'] = variant
    config['completion_policy'] = 'restore_token_reservation_' + variant
'''
    new = '''    config = dict(config, cap=args.cap, domain=args.domain, engine_max_num_seqs=32,
        metric_role='continuous request metrics primary; SLO 5s/.2s reference only',
        evidence_ceiling='NATIVE_INPROCESS_CAPACITY_QUALIFICATION', max_seconds=120)
    config.update(requested_arm='budget90', reservation_policy=args.reservation_policy,
        completion_policy=spec['completion_policy'], policy_family=spec['policy_family'],
        variant=args.variant, headroom_observer='fast', fixed_kv_cache_memory_bytes=args.kv_cache_bytes,
        gpu_memory_utilization=args.gpu_memory_utilization, rotation_victim_order=spec['victim_order'],
        preemption_mode='native_recompute', rotation_config=vars(RotationConfig()))
    if spec['adapter'] == 'component':
        config['component'] = dict(boost=spec['boost'], threshold=200, quantum=10,
            base_order='FCFS', backend='priority packing / current-history reservation / native recompute',
            packing=spec['packing'], complete_restores=spec['complete_restores'],
            reserve_ready_tokens=spec['reserve_ready_tokens'])
'''
    text = change(text, old, new)
    old = '''        decisions, counters, obligations, uninstall = install_component(scheduler,
            vllm_config=engine.vllm_config, block_size=qualification['block_size'],
            boost=True, threshold=200, quantum=10, packing=args.packing,
            complete_restores=args.complete_restores == 'on',
            reserve_ready_tokens=args.reserve_ready_tokens == 'on')
        try:
'''
    new = '''        obligations = None
        if spec['adapter'] == 'component':
            decisions, counters, obligations, uninstall = install_component(scheduler,
                vllm_config=engine.vllm_config, block_size=qualification['block_size'],
                boost=True, threshold=200, quantum=10, packing=spec['packing'],
                complete_restores=spec['complete_restores'],
                reserve_ready_tokens=spec['reserve_ready_tokens'])
            decision_name = 'component-decisions.json'
        elif spec['adapter'] == 'rotation':
            decisions, uninstall = install_rotation(scheduler, vllm_config=engine.vllm_config,
                block_size=qualification['block_size'], expected_requests=32,
                rotation_config=RotationConfig(**config['rotation_config']), victim_order='most_output')
            decision_name = 'headroom-decisions.json'
        else:
            decisions, uninstall = install(scheduler, vllm_config=engine.vllm_config,
                block_size=qualification['block_size'], expected_requests=32,
                mode='native', observer='fast')
            decision_name = 'headroom-decisions.json'
        try:
'''
    text = change(text, old, new)
    text = change(text, "            if raw['status'] == 'COMPLETE':\n                obligations.finalize(len(decisions), {}, [r['internal_request_id'] for r in raw['requests'] if r['status']=='completed'])",
                  "            if raw['status'] == 'COMPLETE' and obligations is not None:\n                obligations.finalize(len(decisions), {}, [r['internal_request_id'] for r in raw['requests'] if r['status']=='completed'])")
    text = change(text, "            dump(out/'component-decisions.json', decisions)\n            dump(out/'restore-obligations.json', obligations.snapshot())",
                  "            dump(out/decision_name, decisions)\n            if obligations is not None:\n                dump(out/'restore-obligations.json', obligations.snapshot())")
    text = change(text, "        if args.completion_policy == 'headroom' and raw['status'] == 'COMPLETE':\n            if not any(d['held'] for d in decisions):\n                raw.update(status='INVALID_NO_ACTION', error='headroom never held a request')\n        if args.completion_policy == 'rotate' and raw['status'] == 'COMPLETE':",
                  "        if args.variant == 'most_output' and raw['status'] == 'COMPLETE':")
    probe.write_text(text)

    module = load("holdout_comparison_probe", probe)
    cells = []
    for block, variants in ((0, ORDER), (1, tuple(reversed(ORDER)))):
        for variant in variants:
            spec = module.arm_spec(variant)
            cells.append(dict(label=f"cohort3-block{block}-{variant}", cohort_id="cohort3",
                block=block, variant=variant, kv_cache_bytes=13960740864, usable_blocks=6656,
                cap=32, **spec))
    campaign = dict(schema_version=1, experiment_id="20260914_recovery_holdout_comparison_r01",
        status="CPU_PREPARED_GPU_UNRUN", cohort_id="cohort3", cells=cells,
        comparison="native_vs_most_output_vs_fit_scan_vs_guard_residual",
        question="On frozen disjoint cohort3, do native or strongest ordinary most_output cover the fit/residual pause-throughput tradeoff?",
        fixed_controls=dict(requests=32, prompt_tokens=3072, output_tokens=1024,
            arrival_gap_s=.05, cap=32, max_num_batched_tokens=1024, kv_cache_memory_bytes=13960740864,
            usable_blocks=6656, prefix_caching=False, ltr_threshold=200, ltr_quantum=10),
        claim_boundary="Strong-baseline qualification on one same-source document-disjoint cohort; no method winner or novelty claim before runtime.",
        input_preparation_sha256=HOLDOUT_PREPARATION_SHA)
    (pkg / "campaign.json").write_text(json.dumps(campaign, indent=2) + "\n")

    launcher = pkg / "run.sh"
    shell = launcher.read_text()
    start = shell.index("run() {")
    end = shell.index('echo "CAMPAIGN_FINISHED', start)
    runs = '''run() {
 label=$1; variant=$2
 pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) || exit 92
 if [ -n "$pids" ]; then echo "BLOCKED_BEFORE_GPU_INITIALIZATION $pids"; exit 93; fi
 echo "START $(date -u +%FT%TZ) $label"
 timeout 600 "$PYTHON_BIN" -u run_probe.py --variant "$variant" --domain long --reservation-policy full --cap 32 --gpu-memory-utilization 0.9 --kv-cache-bytes 13960740864 --output-dir "$BASE/results/$label" > "$BASE/results/$label.log" 2>&1
 echo "END $(date -u +%FT%TZ) $label"
}
''' + ''.join(f"run cohort3-block{block}-{variant} {variant}\n"
              for block, variants in ((0, ORDER), (1, tuple(reversed(ORDER)))) for variant in variants)
    launcher.write_text(shell[:start] + runs + shell[end:])

    for source in pkg.glob("*.py"):
        ast.parse(source.read_text(), filename=str(source))
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    subprocess.run([sys.executable, str(probe), "--help"], check=True, stdout=subprocess.DEVNULL)
    ltr = load("holdout_comparison_ltr", pkg / "ltr_recompute_native.py")
    rows = [ltr.Candidate("restore", -2, 0, True, 1, 1, 10),
            ltr.Candidate("ready", 0, 1, True, 1, 1, 1)]
    fit = ltr.plan(rows, 0, 5, 2, packing="fit_scan", reserve_ready_tokens=False)
    residual = ltr.plan(rows, 0, 5, 2, packing="rank_prefix", reserve_ready_tokens=True)
    assert (fit.tokens, residual.tokens) == ({"restore": 5}, {"restore": 4, "ready": 1})
    checks = dict(status="PASS", gpu_executions=0, uploads=0, compatibility=compatibility,
        input=dict(workload_sha256=new_config["workload_sha256"], requests=32,
            all_prompt_lengths_3072=all(len(ids) == 3072 for ids in workload["actual_prompt_token_ids"]),
            exact_steady_arrivals=True, preparation_sha256=HOLDOUT_PREPARATION_SHA),
        arm_specs={name: module.arm_spec(name) for name in ORDER},
        plan_fixture=dict(fit_scan=fit.tokens, guard_residual=residual.tokens),
        source=dict(common_a_t_modules_byte_identical=list(COMMON),
            ltr_sha256=sha(pkg / "ltr_recompute_native.py"), rotation_sha256=sha(pkg / "rotation_native.py")),
        run_order=[cell["label"] for cell in cells],
        boundary="CPU source/identity/plan checks only; no runtime result or winner.")
    (out / "preparation_checks.json").write_text(json.dumps(checks, indent=2) + "\n")

    inventory = {str(path.relative_to(pkg)): sha(path) for path in sorted(pkg.rglob("*"))
                 if path.is_file() and path.name != "SHA256SUMS"}
    (pkg / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in inventory.items()))
    archive = out / "execution.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name in [*inventory, "SHA256SUMS"]:
            bundle.add(pkg / name, arcname="pkg/" + name, recursive=False)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    metadata = dict(status="CPU_PREPARED_NATIVE_INTERFACE_UNRUN", gpu_executions=0, uploads=0,
        comparison=campaign["comparison"], repository_head=head,
        repository_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()),
        files_sha256=inventory, archive_sha256=sha(archive), cells=cells,
        holdout_preparation_sha256=HOLDOUT_PREPARATION_SHA,
        holdout_workload_sha256=new_config["workload_sha256"],
        source_receipts_sha256={name: sha(receipts / name) for name in sorted(p.name for p in receipts.iterdir())},
        source_archives=dict(token_reservation=t_meta["archive_sha256"],
            d6_strong_baseline=d_final["package_sha256"]),
        expected_runtime_sources=t_meta["expected_runtime_sources"], compatibility=compatibility,
        checks_sha256=sha(out / "preparation_checks.json"), command=shlex.join([sys.executable, *sys.argv]),
        source_script_sha256=sha(Path(__file__)), prepared_at=datetime.now(timezone.utc).isoformat(),
        scope="One document-disjoint native/strong-simple/component comparison, CPU prepared only; GPU UNRUN.")
    (out / "preparation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(dict(output=str(out), archive_sha256=metadata["archive_sha256"],
        metadata_sha256=sha(out / "preparation.json"), files=len(inventory), cells=len(cells))))


if __name__ == "__main__":
    main()
