#!/usr/bin/env python3
"""Freeze the homogeneous/heterogeneous output-length campaign as an execution package.

What this changes in the sealed runner
--------------------------------------
The sealed `native_capture.py` fixes one output length for the whole episode:

    count = config["output_tokens"]
    ... SamplingParams(max_tokens=count, min_tokens=count, ignore_eos=True)

and validates completion against that single `count`. This package makes the
length per request, driven by a frozen list, and keeps every other code path
byte-identical. Three call sites move from the scalar to a lookup:

  * SamplingParams construction        -> length of the request being admitted
  * incremental output bound check     -> that request's length
  * completion check (len == count)    -> that request's length

`ignore_eos=True` and `min_tokens == max_tokens` are preserved, so each request
still generates a deterministic, exactly specified number of tokens. Nothing
about arrival, scheduling, KV accounting or the rotation action is touched.

`run_probe.py` gains `--arm {homogeneous,heterogeneous}` and drops the hard
assertion `requests == 32 and output_tokens == 1024`, replacing it with an
assertion that the arm's reserved-block total matches the frozen value, so a
workload edit that silently changes the structural deficit cannot run.

Everything else -- the four completion policies, the rotation parameters, the
fixed KV bytes, memory telemetry, metrics -- is copied unchanged from the sealed
four-arm package so results stay attributable to output-length dispersion.
"""

import argparse, hashlib, json, shutil, subprocess, sys, tarfile
from datetime import datetime, timezone
from pathlib import Path

# cohort/arm/policy in execution order; block 1 reverses block 0.
BLOCK0 = [("homogeneous", "native"), ("homogeneous", "rotate"),
          ("heterogeneous", "native"), ("heterogeneous", "rotate")]
BLOCK1 = list(reversed(BLOCK0))

EXPECTED_RESERVED_BLOCKS = 8064
EXPECTED_REQUESTS = 42


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(text, old, new, what):
    if text.count(old) != 1:
        raise ValueError(f"expected exactly one occurrence of {what}; found {text.count(old)}")
    return text.replace(old, new)


def patch_capture(text):
    """Make the fixed output length per-request."""
    text = replace_once(
        text,
        '    count = config["output_tokens"]\n'
        '    if type(count) is not int or count < 2 or type(config["cap"]) is not int or config["cap"] < 1:\n'
        '        raise ValueError("need positive cap and at least two fixed output tokens")',
        # Per-request lengths, indexed by the source request ID. The scalar form
        # stays legal so warmup episodes (which pass output_tokens=16) are
        # untouched; only a workload carrying explicit lengths switches modes.
        '    lengths_by_id = workload.get("output_lengths_by_request") or {}\n'
        '    count = config.get("output_tokens")\n'
        '    if lengths_by_id:\n'
        '        if count is not None:\n'
        '            raise ValueError("set either a scalar output_tokens or per-request lengths, not both")\n'
        '        if {s["request_id"] for s in sources} != set(lengths_by_id):\n'
        '            raise ValueError("per-request output lengths do not cover the source requests exactly")\n'
        '        if any(type(v) is not int or v < 2 for v in lengths_by_id.values()):\n'
        '            raise ValueError("each per-request output length must be an int >= 2")\n'
        '    elif type(count) is not int or count < 2:\n'
        '        raise ValueError("need at least two fixed output tokens")\n'
        '    if type(config["cap"]) is not int or config["cap"] < 1:\n'
        '        raise ValueError("need positive cap")\n'
        '    target = lambda rid: lengths_by_id.get(rid, count)',
        "output length setup")

    text = replace_once(
        text,
        "                params = SamplingParams(n=1, temperature=0.0, max_tokens=count, min_tokens=count,",
        "                want = target(source_id)\n"
        "                params = SamplingParams(n=1, temperature=0.0, max_tokens=want, min_tokens=want,",
        "SamplingParams construction")

    text = replace_once(
        text,
        "                if tokens[:len(previous)] != previous or len(tokens) < len(previous) or len(tokens) > count:",
        "                want = target(source_id)\n"
        "                if tokens[:len(previous)] != previous or len(tokens) < len(previous) or len(tokens) > want:",
        "incremental output bound")

    text = replace_once(
        text,
        '                    if len(tokens) != count or completion.finish_reason != "length":',
        '                    if len(tokens) != want or completion.finish_reason != "length":',
        "completion length check")
    return text


def patch_probe(text):
    """Accept --arm, load the paired workload, and re-check the deficit."""
    text = replace_once(
        text,
        "    if len(rows) != 32 or len(tokens) != 32 or config['output_tokens'] != 1024:\n"
        "        raise ValueError('frozen workload dimensions differ')\n"
        "    expected = 128 if domain == 'short' else 3072",
        "    if domain == 'pair':\n"
        f"        if len(rows) != {EXPECTED_REQUESTS} or len(tokens) != {EXPECTED_REQUESTS}:\n"
        "            raise ValueError('frozen pair workload dimensions differ')\n"
        "    elif len(rows) != 32 or len(tokens) != 32 or config['output_tokens'] != 1024:\n"
        "        raise ValueError('frozen workload dimensions differ')\n"
        "    expected = {'short': 128, 'long': 3072, 'pair': 1024}[domain]",
        "workload dimension check")

    text = replace_once(
        text,
        "    parser.add_argument('--domain', choices=['short', 'long'], required=True)",
        "    parser.add_argument('--domain', choices=['short', 'long', 'pair'], required=True)\n"
        "    parser.add_argument('--arm', choices=['homogeneous', 'heterogeneous'], default=None)",
        "domain argument")

    text = replace_once(
        text,
        "    inputs = {d: load_inputs(root, d) for d in ['short', 'long']}",
        "    inputs = {d: load_inputs(root, d) for d in ['short', 'long', 'pair']}\n"
        "    if (args.domain == 'pair') != (args.arm is not None):\n"
        "        raise ValueError('--arm is required for and only for the pair domain')",
        "input loading")

    # Attach the arm's per-request lengths and re-verify the structural deficit
    # before the engine is built, so a mismatched workload cannot consume GPU.
    text = replace_once(
        text,
        "    config = dict(config, cap=args.cap, domain=args.domain, engine_max_num_seqs=32,",
        "    if args.domain == 'pair':\n"
        "        arm = config['arms'][args.arm]\n"
        "        lengths = arm['output_lengths']\n"
        "        ids = [row['request_id'] for row in workload['source_requests']]\n"
        "        if len(lengths) != len(ids):\n"
        "            raise ValueError('arm lengths do not match the request count')\n"
        "        import math as _math\n"
        "        reserved = sum(_math.ceil(min(config['prompt_tokens'] + L, config['max_model_len']) / 16)\n"
        "                       for L in lengths)\n"
        f"        if reserved != {EXPECTED_RESERVED_BLOCKS}:\n"
        f"            raise ValueError(f'arm reserves {{reserved}} blocks, expected {EXPECTED_RESERVED_BLOCKS}; "
        "deficit would be confounded')\n"
        "        workload = dict(workload, output_lengths_by_request=dict(zip(ids, lengths)))\n"
        "        config = dict(config, arm=args.arm, output_tokens=None,\n"
        "                      reserved_blocks_checked=reserved)\n"
        "        engine_seqs = len(ids)\n"
        "    else:\n"
        "        engine_seqs = 32\n"
        "    config = dict(config, cap=args.cap, domain=args.domain, engine_max_num_seqs=engine_seqs,",
        "config assembly")

    # The pair domain admits up to 42 concurrent requests.
    text = text.replace("max_num_seqs=32", "max_num_seqs=engine_seqs")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sealed-package", type=Path, required=True,
                    help="frozen/ directory of the sealed four-arm execution package")
    ap.add_argument("--prepared", type=Path, required=True,
                    help="directory holding the paired workload.json and config.json")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    src, out = args.sealed_package, args.output_dir
    stage = out / "frozen"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    manifest = {}
    for path in sorted(src.iterdir()):
        if path.name == "inputs_preparation" or path.is_dir():
            continue
        manifest[path.name] = dict(source_sha256=file_sha(path))
        shutil.copy2(path, stage / path.name)

    for name, patcher in (("native_capture.py", patch_capture), ("run_probe.py", patch_probe)):
        text = (stage / name).read_text()
        patched = patcher(text)
        if patched == text:
            raise ValueError(f"{name} was not modified")
        (stage / name).write_text(patched)
        manifest[name]["patched_sha256"] = file_sha(stage / name)

    # Sealed 32-request inputs stay available so the runner's own identity
    # checks still exercise them; the new pair domain is added alongside.
    shutil.copytree(src / "inputs_preparation", stage / "inputs_preparation")
    pair = stage / "inputs_preparation/prepared/pair"
    pair.mkdir(parents=True)
    for f in ("workload.json", "config.json"):
        shutil.copy2(args.prepared / f, pair / f)
        manifest[f"prepared/pair/{f}"] = dict(source_sha256=file_sha(pair / f))

    cfg = json.load(open(pair / "config.json"))
    if cfg["requests"] != EXPECTED_REQUESTS:
        raise ValueError("prepared workload does not carry the expected request count")
    for name, arm in cfg["arms"].items():
        if arm["reserved_blocks"] != EXPECTED_RESERVED_BLOCKS:
            raise ValueError(f"arm {name} reserves {arm['reserved_blocks']} blocks")

    cells = []
    for block, order in ((0, BLOCK0), (1, BLOCK1)):
        for arm, policy in order:
            cells.append(dict(label=f"block{block}-{arm[:3]}-{policy}", block=block,
                              arm=arm, completion_policy=policy, cap=32, domain="pair"))
    json.dump(dict(cells=cells, prepared_workload_sha256=cfg["workload_sha256"],
                   reserved_blocks=EXPECTED_RESERVED_BLOCKS, requests=EXPECTED_REQUESTS),
              open(stage / "campaign.json", "w"), indent=1)

    archive = out / "execution.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(stage)))

    status = dict(
        status="PREPARED_UNRUN",
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        gpu_executions=0,
        sealed_package=str(src),
        archive_sha256=file_sha(archive),
        file_manifest=manifest,
        cells=[c["label"] for c in cells],
        note=("patched runner only; no GPU executed. The two arms reserve identical "
              "blocks so output-length dispersion is the only variable."),
    )
    json.dump(status, open(out / "status.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in status.items() if k != "file_manifest"}, indent=1))


if __name__ == "__main__":
    main()
