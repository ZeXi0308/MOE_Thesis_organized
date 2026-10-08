#!/usr/bin/env python3
"""Freeze the structural-deficit sweep as an execution package.

Why sweep the KV pool
---------------------
Every result on the rotation mechanism so far sits at one operating point: a
structural deficit of 521 blocks, i.e. 2.035 request footprints. At that point
rotation cuts max ITL from 4.46 s to 1.01 s with no resolvable throughput cost.
One point cannot distinguish "this is a mechanism" from "this is a coincidence
of that point", and it says nothing about where the mechanism stops working.

The deficit is

    D = N * ceil((P + L) / block_size) - usable_blocks

With N, P and L pinned to the sealed workload, the only clean knob is the KV
pool, which the runner already threads through as `kv_cache_memory_bytes`. This
package turns the two hardcoded constants into one CLI argument and freezes a
sixteen-cell campaign over four deficits.

What is deliberately NOT varied
-------------------------------
N, prompt tokens, output tokens, arrival trace, cap, token budget, rotation
parameters and the engine flags all stay byte-identical to the sealed four-arm
package. If a deficit point cannot run without changing one of those, that point
is recorded as infeasible rather than rescued by moving a second variable.

Block accounting
----------------
vLLM reserves block 0 as the null block, so `usable = total - 1`. The sealed run
reported 7672 total for 16,089,350,144 bytes, giving 2,096,891.05 bytes/block.
Byte targets here are derived from that ratio and rounded down to a 2 MiB
boundary; the achieved block count is whatever the engine reports, and the
analyzer uses the reported value rather than the requested one.
"""

import argparse, hashlib, json, math, shutil, tarfile
from datetime import datetime, timezone
from pathlib import Path

BLOCK = 16
PROMPT, OUTPUT, N_REQUESTS = 3072, 1024, 32
PER_REQUEST_BLOCKS = math.ceil((PROMPT + OUTPUT) / BLOCK)      # 256
SEALED_BYTES, SEALED_TOTAL_BLOCKS = 16089350144, 7672
BYTES_PER_BLOCK = SEALED_BYTES / SEALED_TOTAL_BLOCKS

# deficit in request footprints -> label
DEFICITS = [(0.0, "d0"), (2.035, "d2"), (4.0, "d4"), (6.0, "d6")]
ARMS = [("native", "native"), ("rotate", "rotate")]


def file_sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def bytes_for_deficit(deficit_requests):
    """KV bytes whose usable block count yields the requested deficit."""
    usable = N_REQUESTS * PER_REQUEST_BLOCKS - round(deficit_requests * PER_REQUEST_BLOCKS)
    total = usable + 1                      # block 0 is the null block
    raw = total * BYTES_PER_BLOCK
    aligned = int(raw // (2 * 1024 * 1024)) * (2 * 1024 * 1024)
    return aligned, usable, total


def replace_once(text, old, new, what):
    if text.count(old) != 1:
        raise ValueError(f"expected one occurrence of {what}, found {text.count(old)}")
    return text.replace(old, new)


def patch_probe(text):
    """Turn the two hardcoded KV constants into a required CLI argument."""
    text = replace_once(
        text,
        "    parser.add_argument('--output-dir', type=Path, required=True)",
        "    parser.add_argument('--kv-cache-bytes', type=int, required=True,\n"
        "                        help='actual KV pool in bytes; sets the structural deficit')\n"
        "    parser.add_argument('--output-dir', type=Path, required=True)",
        "kv-cache-bytes argument")
    text = replace_once(
        text,
        "                  fixed_kv_cache_memory_bytes=16089350144,",
        "                  fixed_kv_cache_memory_bytes=args.kv_cache_bytes,",
        "config kv bytes")
    text = replace_once(
        text,
        "            kv_cache_memory_bytes=16089350144, scheduler_reserve_full_isl=args.reservation_policy == 'full',",
        "            kv_cache_memory_bytes=args.kv_cache_bytes, scheduler_reserve_full_isl=args.reservation_policy == 'full',",
        "engine kv bytes")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sealed-package", type=Path, required=True,
                    help="frozen/ directory of the sealed four-arm package")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    src, out = args.sealed_package, args.output_dir
    stage = out / "frozen"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    manifest = {}
    for path in sorted(src.iterdir()):
        if path.is_dir():
            continue
        manifest[path.name] = dict(source_sha256=file_sha(path))
        shutil.copy2(path, stage / path.name)
    shutil.copytree(src / "inputs_preparation", stage / "inputs_preparation")

    probe = stage / "run_probe.py"
    patched = patch_probe(probe.read_text())
    probe.write_text(patched)
    manifest["run_probe.py"]["patched_sha256"] = file_sha(probe)

    points = []
    for deficit, tag in DEFICITS:
        kv_bytes, usable, total = bytes_for_deficit(deficit)
        points.append(dict(tag=tag, deficit_requests=deficit, kv_cache_bytes=kv_bytes,
                           expected_usable_blocks=usable, expected_total_blocks=total,
                           required_blocks=N_REQUESTS * PER_REQUEST_BLOCKS,
                           expected_deficit_blocks=N_REQUESTS * PER_REQUEST_BLOCKS - usable))

    cells = []
    for block in (0, 1):
        order = ARMS if block == 0 else list(reversed(ARMS))
        for p in points:
            for arm, policy in order:
                cells.append(dict(label=f"block{block}-{p['tag']}-{arm}", block=block,
                                  deficit_tag=p["tag"], deficit_requests=p["deficit_requests"],
                                  kv_cache_bytes=p["kv_cache_bytes"],
                                  completion_policy=policy, cap=32, domain="long"))

    json.dump(dict(points=points, cells=cells,
                   per_request_blocks=PER_REQUEST_BLOCKS,
                   requests=N_REQUESTS, prompt_tokens=PROMPT, output_tokens=OUTPUT,
                   bytes_per_block=BYTES_PER_BLOCK),
              open(stage / "campaign.json", "w"), indent=1)

    archive = out / "execution.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                tar.add(p, arcname=str(p.relative_to(stage)))

    status = dict(status="PREPARED_UNRUN", gpu_executions=0,
                  frozen_utc=datetime.now(timezone.utc).isoformat(),
                  sealed_package=str(src), archive_sha256=file_sha(archive),
                  n_cells=len(cells), points=points, file_manifest=manifest,
                  note="only kv_cache_memory_bytes varies; N, inputs, arrival, cap, "
                       "rotation parameters and engine flags are unchanged")
    json.dump(status, open(out / "status.json", "w"), indent=1)

    print(f"archive sha256 {status['archive_sha256']}")
    print(f"{len(cells)} cells over {len(points)} deficit points")
    for p in points:
        print(f"  {p['tag']}: deficit {p['deficit_requests']:.3f} req "
              f"({p['expected_deficit_blocks']:+d} blocks), usable {p['expected_usable_blocks']}, "
              f"kv {p['kv_cache_bytes']:,} bytes")


if __name__ == "__main__":
    main()
