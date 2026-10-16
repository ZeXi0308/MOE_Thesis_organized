#!/usr/bin/env python3
"""Read-only Qwen space plan, or explicitly execute K-private reflink/HTTP resume.

Never writes C, signals its downloader, migrates its files, or uses a GPU.
Reuses prepare_deepseek_mirror.download_shard for strict Range and final SHA256.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
GIB = 2**30
REPO = "Qwen/Qwen3-30B-A3B"
REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
MAPPING_SHA256 = "df2fdd2539cae01e24e4d52914d8d8c79e13bcdd9f9c5cb2c115ac18206f6edb"


def fingerprint(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular nonsymlink file: {path}")
    st = path.stat()
    return {"dev": st.st_dev, "ino": st.st_ino, "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def source_process(pid, source):
    base = Path("/proc") / str(pid)
    parts = (base / "stat").read_text().rsplit(")", 1)[1].split()
    if parts[0] != "T":
        raise RuntimeError("C downloader must remain stopped; no signal will be sent")
    argv = [os.fsdecode(a) for a in (base / "cmdline").read_bytes().split(b"\0") if a]
    if not any(Path(a).name == "prepare_quantization.py" for a in argv) or str(source) not in argv:
        raise RuntimeError("Source PID does not identify the expected C downloader")
    # Never record or print the rest of the process environment.
    flags = {os.fsdecode(k): os.fsdecode(v) for entry in (base / "environ").read_bytes().split(b"\0")
             if b"=" in entry for k, v in [entry.split(b"=", 1)]
             if k in (b"HF_HUB_DISABLE_XET", b"HF_HUB_ENABLE_HF_TRANSFER")}
    if flags.get("HF_HUB_DISABLE_XET") != "1" or flags.get("HF_HUB_ENABLE_HF_TRANSFER", "0") not in ("", "0"):
        raise RuntimeError("Only the verified ordinary HTTP writer is eligible")
    if (base / "exe").resolve() != Path(sys.executable).resolve():
        raise RuntimeError("Use the same Python environment as the source downloader")
    opened, positions = [], {}
    for fd in (base / "fd").iterdir():
        try:
            value = os.readlink(fd)
            if value.startswith(str(source) + "/") and value.endswith(".incomplete"):
                opened.append(value)
                info = (base / "fdinfo" / fd.name).read_text()
                positions[value] = next(int(line.split()[1]) for line in info.splitlines() if line.startswith("pos:"))
        except FileNotFoundError:
            pass
    return {"pid": pid, "state": parts[0], "start_ticks": int(parts[19]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "flags": flags, "open_partials": sorted(opened), "open_partial_positions": positions}


def http_contract(digest):
    import huggingface_hub
    from huggingface_hub import file_download
    writer = inspect.getsource(file_download._download_to_tmp_and_move)
    stream = inspect.getsource(file_download.http_get)
    if not all(s in writer for s in ('uuid.uuid4()', 'tmp_path.open("wb")', 'http_get(')) or not all(
            s in stream for s in ('temp_file.write(chunk)', 'new_resume_size += len(chunk)', 'headers["Range"]')):
        raise RuntimeError("Installed HF writer differs from the inspected sequential HTTP implementation")
    path = Path(file_download.__file__)
    return {"version": huggingface_hub.__version__, "file": str(path), "sha256": digest(path),
            "layout": "process-unique wb file; sequential HTTP chunks; Range resumes at written offset",
            "content_authentication": "partial layout/provenance only; full original SHA256 required before READY"}


def prefix_layout(path, row, weight_map):
    stamp = fingerprint(path)
    size = stamp["size"]
    if not 0 < size <= row["size"]:
        raise ValueError(f"Invalid partial length: {path}")
    with path.open("rb") as handle:
        header_size = int.from_bytes(handle.read(8), "little")
        if not 0 < header_size <= min(32 * 2**20, size - 8):
            raise ValueError(f"Partial does not contain a complete safetensors header: {path}")
        header = json.loads(handle.read(header_size))
        tensors = {key: val for key, val in header.items() if key != "__metadata__"}
        if not tensors or any(weight_map.get(key) != row["path"] for key in tensors):
            raise ValueError(f"Partial header disagrees with the pinned index: {path}")
        if max(value["data_offsets"][1] for value in tensors.values()) + 8 + header_size != row["size"]:
            raise ValueError(f"Partial header has the wrong full-file size: {path}")
        # Sparse/preallocated layout is not accepted as a continuous HTTP prefix.
        if os.lseek(handle.fileno(), 0, os.SEEK_DATA) != 0 or os.lseek(handle.fileno(), 0, os.SEEK_HOLE) < size:
            raise ValueError(f"Partial contains an unwritten hole: {path}")
    if fingerprint(path) != stamp:
        raise RuntimeError(f"Source changed during prefix inspection: {path}")
    return stamp


def assets(root, source, mapping, digest):
    directory = root / "ref_july/tokenizers/qwen3_bf16_ad44e777"
    if digest(directory / "source.json") != mapping["tokenizer_source"]["sha256"]:
        raise ValueError("K tokenizer source receipt changed")
    identity = json.loads((directory / "source.json").read_text())
    if identity["repo"] != REPO or identity["revision"] != REVISION:
        raise ValueError("Wrong K tokenizer identity")
    rows = [{"path": name, "size": value["size"], "sha256": value["sha256"], "source": str(directory / name)}
            for name, value in identity["files"].items()]
    for row in mapping["small_assets"]:
        path = source / row["path"]
        if not path.is_file():
            path = root.parent / "research_C_format_sched/runs/checkpoint_probe/qwen3_bf16" / row["path"]
        rows.append({**row, "source": str(path)})
    for row in rows:
        path = Path(row["source"])
        stamp = fingerprint(path)
        if stamp["size"] != row["size"] or digest(path) != row["sha256"] or fingerprint(path) != stamp:
            raise ValueError(f"Pinned small asset mismatch: {path}")
        row["source_stat"] = stamp
    return rows


def storage_plan(args, destination, helpers):
    stores = {"persistent": {"directory": str(destination), "reserve_bytes": args.reserve_gib * GIB,
                             "other_reserved_bytes": args.other_reserved_bytes}}
    placements = {}
    if args.fixed_placement:
        value = json.loads(args.fixed_placement.read_text())
        expected = {"rootfs": ["model-00009-of-00016.safetensors", "model-00016-of-00016.safetensors"],
                    "shm": ["model-00015-of-00016.safetensors"]}
        directories = {"rootfs": Path("/tmp/research_K_expert_skip/qwen3_bf16_ad44e777"),
                       "shm": Path("/dev/shm/research_K_expert_skip/qwen3_bf16_ad44e777")}
        for key, names in expected.items():
            item = value[key]
            path = Path(item["directory"])
            if path != directories[key] or path.resolve() != directories[key] or item["files"] != names:
                raise ValueError("Only the explicitly approved fixed three-shard placement is supported")
            minimum = 10 if key == "rootfs" else 20
            if item["reserve_gib"] < minimum:
                raise ValueError("Fixed filesystem reserve cannot be lowered")
            stores[key] = {"directory": str(path), "reserve_bytes": item["reserve_gib"] * GIB,
                           "other_reserved_bytes": 0}
            placements.update({name: key for name in names})
        if value["persistent_output_reservation_bytes"] < 2 * GIB:
            raise ValueError("Fixed placement retains >=2 GiB for persistent experiment output")
        stores["persistent"]["other_reserved_bytes"] += value["persistent_output_reservation_bytes"]
    for key, store in stores.items():
        store.update(helpers.filesystem_state(store["directory"]), clone_overhead_allowance_bytes=256 * 2**20)
        if key == "shm" and (store.get("filesystem_type") != "tmpfs" or store.get("mount_point") != "/dev/shm"):
            raise ValueError("Approved shm directory must resolve to /dev/shm tmpfs")
    if len({store["device_id"] for store in stores.values()}) != len(stores):
        raise ValueError("The three approved storage budgets must be on distinct filesystems")
    return stores, placements


def make_plan(args, helpers):
    root, source, destination = args.root.resolve(), args.source.resolve(), args.destination.resolve()
    if not destination.is_relative_to(root / "models") or destination == root / "models":
        raise ValueError("Destination must be a private directory inside K/models")
    mapping = json.loads((HERE / "pinned_files.json").read_text())
    if helpers.digest(HERE / "pinned_files.json") != MAPPING_SHA256:
        raise ValueError("Pinned HF/ModelScope mapping changed")
    if mapping["repo"] != REPO or mapping["hf_revision"] != REVISION or len(mapping["files"]) != 16:
        raise ValueError("Wrong fixed checkpoint mapping")
    process = source_process(args.source_pid, source)
    observed = json.loads((HERE / "source_inventory.json").read_text())
    if (str(source) != observed["source"] or process["pid"] != observed["source_pid"]
            or process["boot_id"] != observed["boot_id"] or process["start_ticks"] != int(observed["source_start_ticks"])):
        raise RuntimeError("C process identity differs from the supplied source inventory")
    contract = http_contract(helpers.digest)
    small = assets(root, source, mapping, helpers.digest)
    index = next(row for row in small if row["path"] == "model.safetensors.index.json")
    weight_map = json.loads(Path(index["source"]).read_text())["weight_map"]
    partials = list((source / ".cache/huggingface/download").glob("*.incomplete"))
    stores, placements = storage_plan(args, destination, helpers)
    if stores["persistent"]["device_id"] != source.stat().st_dev:
        raise ValueError("Reflink requires the same source/destination filesystem; no copy fallback")
    rows, monitored, c_append, c_restart = [], {}, 0, 0
    for original in mapping["files"]:
        row = {**original, "lfs_sha256": original["sha256"]}
        row["storage"] = placements.get(row["path"], "persistent")
        row["storage_directory"] = stores[row["storage"]]["directory"]
        complete = source / row["path"]
        choices = []
        for path in partials:
            if row["sha256"] in path.name.split("."):
                stamp = prefix_layout(path, row, weight_map)
                choices.append({"path": str(path), "stat": stamp})
                monitored[str(path)] = stamp
        row["partial_candidates"] = choices
        row["clone_source"] = max(choices, key=lambda value: value["stat"]["size"]) if choices else None
        if row["clone_source"] and (row["clone_source"]["path"] not in process["open_partials"]
                or process["open_partial_positions"].get(row["clone_source"]["path"]) != row["clone_source"]["stat"]["size"]):
            raise RuntimeError("Selected maximum prefix is not at the paused HTTP writer's current end offset")
        if complete.exists():
            stamp = fingerprint(complete)
            if (stamp["size"] != row["size"] or args.execute and helpers.digest(complete) != row["sha256"]
                    or fingerprint(complete) != stamp):
                raise ValueError(f"Complete source failed fixed SHA256: {complete}")
            monitored[str(complete)] = stamp
            row["clone_source"] = {"path": str(complete), "stat": stamp}
            row["clone_complete"] = True
            row["source_complete_sha_verified"] = args.execute
        else:
            row["clone_complete"] = False
            active_size = max([value["stat"]["size"] for value in choices
                               if value["path"] in process["open_partials"]] or [0])
            c_append += row["size"] - active_size
            c_restart += row["size"]
        directory = Path(row["storage_directory"])
        target = directory / row["path"]
        own_partial = directory / ".modelscope-partials" / (row["sha256"] + ".partial")
        link = destination / row["path"]
        if row["storage"] != "persistent" and (link.exists() or link.is_symlink()) and (
                not link.is_symlink() or link.resolve() != target):
            raise ValueError(f"Existing model entry conflicts with fixed placement: {link}")
        if target.is_symlink() or own_partial.is_symlink():
            raise ValueError("Storage payloads themselves must never be symlinks")
        present = target if target.exists() else own_partial if own_partial.exists() else None
        if present is not None:
            stamp = fingerprint(present)
            if present.stat().st_nlink != 1:
                raise ValueError(f"K payload must be isolated, not hardlinked: {present}")
            if present == target:
                if stamp["size"] != row["size"] or args.execute and helpers.digest(present) != row["sha256"]:
                    raise ValueError(f"Existing K shard failed full SHA256: {present}")
                row["existing_k_complete_sha_verified"] = args.execute
            elif stamp["size"]:
                prefix_layout(present, row, weight_map)
            effective = stamp["size"]
            row["existing_k_path"] = str(present)
        else:
            effective = row["clone_source"]["stat"]["size"] if row["clone_source"] else 0
        row["resume_bytes"] = effective
        row["new_tail_bytes"] = row["size"] - effective
        row["copy_prefix_bytes"] = (effective if row["storage"] != "persistent" and present is None else 0)
        rows.append(row)
    if source_process(args.source_pid, source) != process or any(fingerprint(Path(p)) != stamp for p, stamp in monitored.items()):
        raise RuntimeError("C PID or files changed during planning")
    charged_c = {"append": c_append, "restart": c_restart, "paused": 0}[args.c_reservation]
    new_bytes = sum(row["new_tail_bytes"] for row in rows)
    small_bytes = sum(row["size"] for row in small if not (destination / row["path"]).exists())
    for key, store in stores.items():
        store["new_tail_bytes"] = sum(row["new_tail_bytes"] for row in rows if row["storage"] == key)
        store["copy_prefix_bytes"] = sum(row["copy_prefix_bytes"] for row in rows if row["storage"] == key)
        store["new_small_asset_bytes"] = small_bytes if key == "persistent" else 0
        store["charged_c_remaining_bytes"] = charged_c if key == "persistent" else 0
        store["predicted_free_after_all_charged_bytes"] = (store["free_bytes"] - store["new_tail_bytes"]
            - store["copy_prefix_bytes"] - store["new_small_asset_bytes"] - store["charged_c_remaining_bytes"]
            - store["other_reserved_bytes"] - store["clone_overhead_allowance_bytes"])
        store["space_ok"] = store["predicted_free_after_all_charged_bytes"] >= store["reserve_bytes"]
    return {"status": "PLAN", "root": str(root), "source": str(source), "destination": str(destination),
            "weight_sha_verification_deferred": not args.execute,
            "repo": REPO, "revision": REVISION, "mapping_sha256": helpers.digest(HERE / "pinned_files.json"),
            "source_process": process, "http_contract": contract, "source_files": monitored,
            "files": rows, "small_assets": small, "filesystems": stores,
            "new_tail_bytes": new_bytes, "new_small_asset_bytes": small_bytes,
            "c_append_remaining_bytes": c_append, "c_restart_remaining_bytes": c_restart,
            "c_reservation": args.c_reservation, "charged_c_remaining_bytes": charged_c,
            "deferred_c_bytes": c_append if args.c_reservation == "paused" else 0,
            "other_reserved_bytes": args.other_reserved_bytes, "reserve_bytes": args.reserve_gib * GIB,
            "space_ok": all(store["space_ok"] for store in stores.values()),
            "partial_rule": "one largest verified-layout prefix per SHA; duplicates never summed or merged; full SHA gates READY",
            "reflink_rule": "Persistent files use FICLONE only; approved cross-filesystem prefix uses an isolated copy. C is never changed",
            "fixed_placement": str(args.fixed_placement) if args.fixed_placement else None,
            "fixed_placement_sha256": helpers.digest(args.fixed_placement) if args.fixed_placement else None}


def clone_source(source, destination, expected, copy_across_filesystems=False):
    if fingerprint(source) != expected:
        raise RuntimeError(f"Source changed before reflink: {source}")
    temporary = destination.with_name(destination.name + ".clone.tmp")
    with source.open("rb") as old, temporary.open("xb") as new:
        if copy_across_filesystems:
            shutil.copyfileobj(old, new, 8 * 2**20)
            new.flush()
        else:
            fcntl.ioctl(new.fileno(), 0x40049409, old.fileno())  # Linux FICLONE, no fallback
        os.fsync(new.fileno())
        old_stat, new_stat = os.fstat(old.fileno()), os.fstat(new.fileno())
        if (old_stat.st_dev, old_stat.st_ino) == (new_stat.st_dev, new_stat.st_ino):
            raise RuntimeError("Clone unexpectedly shares an inode")
    if fingerprint(source) != expected or temporary.stat().st_size != expected["size"]:
        raise RuntimeError("Source changed during clone; K temporary preserved for inspection")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    temporary.rename(destination)


def execute(args, plan, helpers):
    if not plan["space_ok"]:
        raise RuntimeError("Space plan fails one or more approved per-filesystem reserves")
    root, destination = Path(plan["root"]), Path(plan["destination"])
    run = root / "runs" / (time.strftime("qwen_partial_resume_%Y%m%d_%H%M%S_") + str(os.getpid()))
    run.mkdir(parents=True, exist_ok=False)
    receipt = {**plan, "status": "STARTING", "pid": os.getpid(), "started_unix": time.time(), "completed": []}
    def save():
        helpers.atomic_json(run / "source.json", receipt)
    save()
    stop = threading.Event()
    def interrupted(signum, frame):
        stop.set()
        raise InterruptedError(f"Download interrupted by signal {signum}; K partials preserved")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    def check_source():
        if source_process(args.source_pid, Path(plan["source"])) != plan["source_process"]:
            raise RuntimeError("Source PID/state changed; leaving C alone and stopping K")
        if any(fingerprint(Path(p)) != stamp for p, stamp in plan["source_files"].items()):
            raise RuntimeError("C source changed; K transfer stopped")
    try:
        destination.mkdir(parents=True, exist_ok=True)
        for store in plan["filesystems"].values():
            directory = Path(store["directory"])
            directory.mkdir(parents=True, exist_ok=True)
            partial_dir = directory / ".modelscope-partials"
            if partial_dir.is_symlink():
                raise ValueError("Private partial directory must not be a symlink")
            partial_dir.mkdir(exist_ok=True)
        check_source()
        for row in plan["files"]:
            if row.get("existing_k_path") or not row["clone_source"]:
                continue
            value = row["clone_source"]
            directory = Path(row["storage_directory"])
            target = directory / row["path"] if row["clone_complete"] else directory / ".modelscope-partials" / (row["sha256"] + ".partial")
            clone_source(Path(value["path"]), target, value["stat"], row["storage"] != "persistent")
            check_source()
        for row in plan["small_assets"]:
            target = destination / row["path"]
            if not target.exists():
                with target.open("xb") as handle:
                    handle.write(Path(row["source"]).read_bytes())
            if target.is_symlink() or helpers.digest(target) != row["sha256"]:
                raise ValueError(f"K small asset differs: {target}")
        # The helper checks is_set before each chunk/read. Cache these CPU-only
        # guards for two seconds rather than probing /proc on every MiB.
        mutex, last_check = threading.Lock(), [0.]
        class Guard:
            def is_set(self):
                with mutex:
                    if not stop.is_set() and time.monotonic() - last_check[0] >= 2:
                        check_source()
                        for key, store in plan["filesystems"].items():
                            directory = Path(store["directory"])
                            remaining = 0
                            for row in plan["files"]:
                                if row["storage"] != key:
                                    continue
                                target = directory / row["path"]
                                partial = directory / ".modelscope-partials" / (row["sha256"] + ".partial")
                                present = target if target.exists() else partial
                                remaining += row["size"] - (present.stat().st_size if present.exists() else 0)
                            actual = helpers.filesystem_state(directory)
                            if actual["device_id"] != store["device_id"] or (actual["free_bytes"] - remaining
                                    - store["charged_c_remaining_bytes"] - store["other_reserved_bytes"]
                                    - store["clone_overhead_allowance_bytes"] < store["reserve_bytes"]):
                                raise RuntimeError(f"Current {key} budget no longer preserves its reserve")
                        last_check[0] = time.monotonic()
                    return stop.is_set()
            def wait(self, seconds):
                return stop.wait(seconds)
        guard = Guard()
        guard.is_set()
        receipt["status"] = "DOWNLOADING"
        save()
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            pending = [pool.submit(helpers.download_shard, row, Path(row["storage_directory"]), args.chunk_mib * 2**20, args.retries, guard)
                       for row in plan["files"]]
            try:
                for future in as_completed(pending):
                    receipt["completed"].append(future.result())
                    save()
            except BaseException:
                stop.set()
                for future in pending:
                    future.cancel()
                raise
        check_source()
        for row in plan["files"]:
            if row["storage"] != "persistent":
                link, target = destination / row["path"], Path(row["storage_directory"]) / row["path"]
                if not link.exists() and not link.is_symlink():
                    link.symlink_to(target)
                if not link.is_symlink() or link.resolve() != target:
                    raise RuntimeError(f"Unexpected existing model entry: {link}")
        last_check[0] = 0.
        if guard.is_set():
            raise RuntimeError("Transfer cancelled before READY")
        receipt.update(status="READY", scope="All 16 full-shard SHA256 and required small assets verified; model load not tested")
    except BaseException as error:
        stop.set()
        receipt.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["ended_unix"] = time.time()
        save()
    print(json.dumps({"status": receipt["status"], "receipt": str(run / "source.json"), "destination": str(destination)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/research_K_expert_skip"))
    parser.add_argument("--source", type=Path, default=Path("/root/autodl-tmp/research_C_format_sched/checkpoints/qwen3_bf16_ad44e777"))
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--source-pid", type=int, default=21898)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--reserve-gib", type=float, default=20)
    parser.add_argument("--other-reserved-bytes", type=int, default=0)
    parser.add_argument("--fixed-placement", type=Path, help="explicit approved placement of shards9/16 on rootfs and15 on shm")
    parser.add_argument("--c-reservation", choices=("append", "restart", "paused"), default="append",
                        help="paused explicitly defers C's future allocation while its PID stays stopped; debt remains reported")
    args = parser.parse_args()
    if args.execute and args.c_reservation == "paused":
        parser.error("paused reservation is plan-only; execution must retain C future capacity")
    if not 1 <= args.max_workers <= 2 or not 1 <= args.chunk_mib <= 128 or not 0 <= args.retries <= 5 or args.reserve_gib < 20 or args.other_reserved_bytes < 0:
        parser.error("workers1..2, chunk1..128MiB, retries0..5, reserve>=20GiB and other reservations>=0 required")
    args.root = args.root.resolve()
    args.destination = args.destination or args.root / "models/qwen3_bf16_ad44e777"
    sys.path.insert(0, str(args.root / "code"))
    import prepare_deepseek_mirror as helpers
    if args.plan_only:
        print(json.dumps(make_plan(args, helpers), indent=2))
        return
    with (args.root / "qwen_partial_download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args, make_plan(args, helpers), helpers)


if __name__ == "__main__":
    main()
