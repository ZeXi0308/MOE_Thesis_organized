#!/usr/bin/env python3
"""One bounded full-cap/PF/PF/full-cap native AE-core ablation block."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shutil
import stat
import sys

import C_NATIVE_PAST_FUTURE_AE_BLOCK_V1 as prior
import C_NATIVE_PAST_FUTURE_AE_LAUNCHER_V1 as pilot

BASE, PARENT = pilot.BASE, pilot.PARENT
ROOT = BASE / "c-native-past-future-cap-ablation-dev-v1"
CELL_SOURCE = BASE / "C_NATIVE_PAST_FUTURE_CAP_ABLATION_CELL_V1.py"
ORDER = (("cap_1", "native_past_future_cap"),
         ("past_future_1", "native_past_future_ae"),
         ("past_future_2", "native_past_future_ae"),
         ("cap_2", "native_past_future_cap"))
INPUTS, INPUT_SHA = prior.INPUTS, prior.INPUT_SHA
CELL_SHA = "3479f00852a015f1786eedd164bd62ecd5281713d737a6a73d05794d85bdaee8"
CAP_ADAPTER_SHA = "aca4720a69cade192505c61e9d3da7844e89356cfd1a0d73f886c40bf1d1609f"
PF_ADAPTER_SHA, PF_PREDICTOR_SHA = prior.PF_ADAPTER_SHA, prior.PF_PREDICTOR_SHA
HELPER_SHA, PARENT_SHA = prior.HELPER_SHA, prior.PARENT_SHA
PRIOR_BLOCK_SHA = "240e22083a1b4480ac53273ded832cfb335273f910bbf76f07a706373e37ea8b"
MIN_START_FREE, MIN_CELL_FREE = 900 * 1024**2, 512 * 1024**2


def require(ok: bool, reason: str) -> None:
    prior.require(ok, reason)


def prior_gate() -> dict:
    pilot_provenance = prior.pilot_gate()
    root = BASE / "c-native-past-future-ae-matched-dev-v1"
    labels = [label for label, _ in prior.ORDER]
    block = pilot.read(root / "block-receipt.json")
    require(block.get("schema") == "c-native-past-future-ae-matched-block-v1"
            and block.get("status") == "COMPLETE" and block.get("error") is None
            and block.get("order") == block.get("completed") == labels
            and block.get("runner_source_sha256") == PRIOR_BLOCK_SHA,
            "previous PF matched block did not complete under frozen runner")
    cell_receipts = {}
    for index, (label, arm) in enumerate(prior.ORDER):
        path = root / label / "launcher-receipt.json"
        receipt = pilot.read(path)
        require(receipt.get("schema") == "c-native-past-future-ae-matched-cell-v1"
                and receipt.get("sequence_index") == index
                and receipt.get("label") == label and receipt.get("arm") == arm
                and receipt.get("status") == "ACCEPTED_FOR_MATCHED_BLOCK"
                and receipt.get("errors") == []
                and receipt.get("runner_source_sha256") == PRIOR_BLOCK_SHA
                and receipt.get("cell_source_sha256") == prior.CELL_SHA
                and receipt.get("input_file_sha256") == INPUT_SHA
                and receipt.get("batch_tokens") == 4096,
                "previous PF matched cell failed qualification")
        cell_receipts[label] = pilot.sha(path)
    return dict(pilot=pilot_provenance,
                matched_block_receipt_sha256=pilot.sha(root / "block-receipt.json"),
                matched_cell_receipt_sha256=cell_receipts)


def precheck() -> dict:
    require(pilot.sha(Path(pilot.__file__)) == HELPER_SHA
            and pilot.sha(Path(prior.__file__)) == PRIOR_BLOCK_SHA
            and pilot.sha(CELL_SOURCE) == CELL_SHA
            and pilot.sha(PARENT / "manifest.json") == PARENT_SHA
            and pilot.PYTHON.is_file() and pilot.HF_HOME.is_dir(),
            "frozen helper, previous block, new cell, parent or runtime changed")
    require(all(pilot.sha(INPUTS / name) == digest for name, digest in INPUT_SHA.items())
            and all(pilot.sha(BASE / name) == digest
                    for name, digest in pilot.EXTRA_SOURCES.items())
            and pilot.sha(BASE / "C_NATIVE_PAST_FUTURE_CAP_ABLATION_ADMISSION_V1.py")
                == CAP_ADAPTER_SHA
            and pilot.EXTRA_SOURCES.get("C_NATIVE_PAST_FUTURE_AE_ADMISSION_V1.py")
                == PF_ADAPTER_SHA
            and pilot.EXTRA_SOURCES.get("C_PAST_FUTURE_AE_PREDICTOR_V1.py")
                == PF_PREDICTOR_SHA,
            "viewed input or frozen AE-core source changed")
    require(not (ROOT / "block-start.json").exists()
            and not (ROOT / "block-start.json").is_symlink()
            and not any((ROOT / label).exists() or (ROOT / label).is_symlink()
                        for label, _ in ORDER)
            and not ROOT.is_symlink(), "ablation root already started; no retry")
    info = pilot.LOCK.lstat()
    require(stat.S_ISREG(info.st_mode)
            and f"{info.st_dev}:{info.st_ino}" == pilot.LOCK_INODE
            and pilot.GPU_UUID == "GPU-e4434c32-c4a4-2b81-55fa-271af38f3c36",
            "shared GPU UUID or lock inode changed")
    return prior_gate()


def qualify_cell(cell: Path, arm: str, code: int, timed_out: bool,
                 post: dict) -> dict:
    old_cell, old_sha = pilot.CELL, pilot.CELL_SOURCE_SHA
    try:
        pilot.CELL, pilot.CELL_SOURCE_SHA = cell, CELL_SHA
        result = pilot.qualify(code, timed_out, post)
    finally:
        pilot.CELL, pilot.CELL_SOURCE_SHA = old_cell, old_sha
    require(result["status"] == "ACCEPTED_PILOT_ONLY",
            f"{arm}: common cell qualification failed: {result['errors']}")
    config, args, env = (pilot.read(cell / name) for name in
                         ("config.json", "engine_args.json", "environment.json"))
    source = pilot.read(cell / "source-receipt.json")
    raw = pilot.read(cell / "raw.json")
    require(config.get("reservation_policy") == arm
            and config.get("max_num_batched_tokens") == 4096
            and config.get("policy_seed") == 20261001
            and config.get("seed") == args.get("seed") == 20260905
            and config.get("arrival_regime") == raw.get("regime") == "poisson_v1"
            and config.get("arrival_rate_per_s") == 5.0
            and config.get("arrival_span_s") == 22.682329
            and raw.get("arrival_scale") == 1.0
            and args.get("max_num_batched_tokens") == 4096
            and args.get("long_prefill_token_threshold") == 0
            and args.get("max_num_seqs") == 32
            and args.get("max_model_len") == 4096
            and args.get("async_scheduling") is False
            and args.get("enable_prefix_caching") is False
            and source.get("input_sha256") == INPUT_SHA
            and config.get("development_input_receipt", {}).get("input_sha256") == INPUT_SHA
            and env.get("fresh_input_sha256") == INPUT_SHA
            and env.get("pilot_source_sha256") == CELL_SHA
            and env.get("past_future_adapter_sha256") == PF_ADAPTER_SHA
            and env.get("past_future_predictor_sha256") == PF_PREDICTOR_SHA
            and env.get("cap_ablation_adapter_sha256") == CAP_ADAPTER_SHA,
            "same-input seed/batch/policy source changed")
    filename, policy = (("past-future-cap.json",
                         "author_AE_full_output_cap_ablation_native_batch4096")
                        if arm == "native_past_future_cap" else
                        ("past-future-ae.json",
                         "author_AE_statistical_peak_core_native_batch4096"))
    gate_path = cell / filename
    gate = pilot.read(gate_path)
    require(gate.get("status") == "DRAINED" and gate.get("drained") is True
            and gate.get("policy") == policy
            and gate.get("admitted") == gate.get("completed") == 128
            and gate.get("violations") == []
            and type(gate.get("physical_blocks_peak")) is int
            and 0 <= gate["physical_blocks_peak"] <= 4096
            and type(raw.get("actual_preemption_count")) is int
            and gate.get("preemptions") == raw["actual_preemption_count"],
            "AE arm failed capacity, lifecycle or native preemption accounting")
    require(not (cell / ("past-future-ae.json" if arm == "native_past_future_cap"
                          else "past-future-cap.json")).exists(),
            "both AE score rules appeared in one cell")
    summary = {key: gate[key] for key in
               ("policy", "admitted", "completed", "resumptions", "preemptions",
                "hold_calls", "decision_seconds", "physical_blocks_peak")}
    summary["gate_sha256"] = pilot.sha(gate_path)
    return dict(output_tokens=result["output_tokens"],
                raw_sha256=result["raw_sha256"],
                pressure_qualified=result["pressure_qualified"],
                policy_summary=summary)


def run_cell(index: int, label: str, arm: str, fd: int, env: dict,
             prior_provenance: dict) -> dict:
    free = shutil.disk_usage(BASE).free
    require(free >= MIN_CELL_FREE, f"before {label}: under 512 MiB free")
    before = pilot.probe()
    require(prior.idle(before), f"before {label}: GPU not idle")
    folder = ROOT / label
    folder.mkdir(exist_ok=False)
    cell = folder / arm
    command = [str(pilot.PYTHON), str(CELL_SOURCE),
               "--parent-package", str(PARENT), "--inputs-dir", str(INPUTS),
               "--arm", arm, "--output-dir", str(cell)]
    log_path = folder / "child.log"
    started = pilot.stamp()
    with log_path.open("x") as log:
        life = pilot.run_owned_child(command, cwd=BASE, env=env, log=log,
                                     lock_fd=fd, timeout_s=900)
        log.flush()
        os.fsync(log.fileno())
    try:
        post = pilot.probe()
    except Exception as exc:
        post = dict(probe_error=f"{type(exc).__name__}: {exc}")
    errors = []
    if not (life["exit_code"] == life["child_returncode"] == 0
            and life["child_reaped"] is True and life["timed_out"] is False
            and life["launcher_error"] is None and life["interruption_signal"] is None):
        errors.append("owned child exit/timeout/lifetime unqualified")
    if not prior.idle(post):
        errors.append("GPU did not drain after child exit")
    qualified = {}
    if not errors:
        try:
            qualified = qualify_cell(cell, arm, life["exit_code"], life["timed_out"], post)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    receipt = dict(schema="c-native-past-future-cap-ablation-cell-v1",
        sequence_index=index, label=label, arm=arm,
        status="ACCEPTED_FOR_ABLATION_BLOCK" if not errors else "INCOMPLETE",
        errors=errors, started_utc=started, ended_utc=pilot.stamp(),
        child_lifecycle=life, gpu_before=before, gpu_after=post,
        disk_free_before_bytes=free, log_sha256=pilot.sha(log_path),
        runner_source_sha256=pilot.sha(Path(__file__)),
        helper_source_sha256=HELPER_SHA, cell_source_sha256=CELL_SHA,
        cap_adapter_source_sha256=CAP_ADAPTER_SHA,
        pf_adapter_source_sha256=PF_ADAPTER_SHA,
        pf_predictor_source_sha256=PF_PREDICTOR_SHA,
        parent_manifest_sha256=PARENT_SHA, input_file_sha256=INPUT_SHA,
        lock_inode=pilot.LOCK_INODE, gpu_uuid=pilot.GPU_UUID,
        batch_tokens=4096, arrival_scale=1.0, prior_gate=prior_provenance,
        **qualified)
    pilot.atomic_new(folder / "launcher-receipt.json", receipt)
    print(json.dumps(dict(label=label, status=receipt["status"],
                          output_tokens=qualified.get("output_tokens"))), flush=True)
    return receipt


def main() -> int:
    provenance = precheck()
    env = prior.environment()
    fd = os.open(pilot.LOCK, os.O_RDWR | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        require(stat.S_ISREG(opened.st_mode)
                and f"{opened.st_dev}:{opened.st_ino}" == pilot.LOCK_INODE,
                "opened shared GPU lock inode changed")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps(dict(status="GPU_DEFERRED", reason="shared lock busy")), flush=True)
            return 75
        require(shutil.disk_usage(BASE).free >= MIN_START_FREE,
                "under 900 MiB free on data disk; no block started")
        ROOT.mkdir(parents=True, exist_ok=True)
        pilot.atomic_new(ROOT / "block-start.json", dict(
            schema="c-native-past-future-cap-ablation-block-v1",
            started_utc=pilot.stamp(), order=[label for label, _ in ORDER],
            runner_source_sha256=pilot.sha(Path(__file__)),
            helper_source_sha256=HELPER_SHA, cell_source_sha256=CELL_SHA,
            cap_adapter_source_sha256=CAP_ADAPTER_SHA,
            pf_adapter_source_sha256=PF_ADAPTER_SHA,
            pf_predictor_source_sha256=PF_PREDICTOR_SHA,
            parent_manifest_sha256=PARENT_SHA, input_file_sha256=INPUT_SHA,
            lock_inode=pilot.LOCK_INODE, gpu_uuid=pilot.GPU_UUID,
            batch_tokens=4096, arrival_scale=1.0, prior_gate=provenance))
        completed = []
        error = None
        try:
            for index, (label, arm) in enumerate(ORDER):
                receipt = run_cell(index, label, arm, fd, env, provenance)
                require(receipt["status"] == "ACCEPTED_FOR_ABLATION_BLOCK",
                        f"{label}: invalid cell; ablation block stopped")
                completed.append(label)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
        result = dict(schema="c-native-past-future-cap-ablation-block-v1",
            status="COMPLETE" if error is None else "INCOMPLETE",
            order=[label for label, _ in ORDER], completed=completed,
            error=error, ended_utc=pilot.stamp(),
            runner_source_sha256=pilot.sha(Path(__file__)))
        pilot.atomic_new(ROOT / "block-receipt.json", result)
        print(json.dumps(dict(status=result["status"], completed=completed,
                              error=error)), flush=True)
        return 0 if error is None else 70
    finally:
        # The helper reaps owned children; close fd only after all four/stop.
        os.close(fd)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"AE_CAP_ABLATION_PRECHECK_ERROR: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        raise SystemExit(70)
