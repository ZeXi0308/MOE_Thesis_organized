#!/usr/bin/env python3
"""Run the pinned author's pure DFS reorder on actual full150 inputs; CPU only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

COMMIT = "3aca72b63c1dd6433193c0857b965acecdbfbb8a"
WORKLOAD_SHA = "3e34bc8a46abc1329744e22b6c97b18582a540f0abd98de55f012f560aac92fc"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_author_modules(source: Path):
    receipt = json.loads((source / "SOURCE_RECEIPT.json").read_text())
    if receipt["commit"] != COMMIT:
        raise ValueError("author revision differs")
    if any(name == "peek" or name.startswith("peek.") for name in sys.modules):
        raise RuntimeError("run in a fresh process; do not replace an imported PEEK package")
    # Empty namespaces satisfy absolute leaf imports. Author __init__.py is
    # deliberately never executed: it can patch installed inference engines.
    for name in ("peek", "peek.offline"):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    loaded = {}
    for leaf in ("prompt", "trie", "reorder"):
        relative = f"python/peek/offline/{leaf}.py"
        path = source / relative
        if sha(path) != receipt["files"][relative]["sha256"]:
            raise ValueError("author leaf changed: " + relative)
        name = "peek.offline." + leaf
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        loaded[leaf] = module
    return loaded, receipt


def analyze(input_dir: Path, source: Path) -> dict:
    workload_path = input_dir / "workload.json"
    if sha(workload_path) != WORKLOAD_SHA:
        raise ValueError("not the frozen full150 workload")
    rows = json.loads(workload_path.read_text())["requests"]
    if len(rows) != 150 or [r["source_index"] for r in rows] != list(range(150)):
        raise ValueError("source inventory differs")
    sequences = [r["prompt_token_ids"] for r in rows]
    modules, receipt = load_author_modules(source)
    start = time.perf_counter()
    order = modules["reorder"].reorder_for_prefix_sharing(sequences)
    elapsed = time.perf_counter() - start
    if sorted(order) != list(range(150)):
        raise ValueError("author result is not a complete permutation")
    trie = modules["trie"].PrefixTrie()
    for index, sequence in enumerate(sequences):
        trie.insert(sequence, index)
    coverage, group, depth = trie.sharing_score(min_depth=32)
    return dict(schema="c-longbench150-author-peek-dfs-order-v1",
        scope="CPU author Phase1 default reorder only; no GPU, cache simulation, policy outcome or full PEEK reproduction",
        source_commit=COMMIT, source_receipt_sha256=sha(source / "SOURCE_RECEIPT.json"),
        workload_sha256=WORKLOAD_SHA, requests=150,
        defaults=dict(trie_max_depth=trie.max_depth, min_sharing_depth=32,
            min_coverage=.1, min_avg_sharing_depth=64, max_duplication=.5),
        guard_statistics=dict(coverage=coverage, maximum_group_size=group,
            average_sharing_depth=depth,
            full_duplication_ratio=modules["reorder"]._full_duplication_ratio(sequences)),
        identity_order=order == list(range(150)),
        changed_positions=sum(i != value for i, value in enumerate(order)),
        source_indices_in_submission_order=order,
        request_ids_in_submission_order=[rows[i]["request_id"] for i in order],
        local_single_call_reorder_s=elapsed,
        timing_scope="Local CPU observation only, not remote online overhead or a benchmark",
        loaded_leaf_files={name: receipt["files"][name] for name in receipt["files"]
            if name.endswith(("/prompt.py", "/trie.py", "/reorder.py"))},
        omitted="Author package init, dispatch/wave cache refinement, eviction, online lanes and all engine patches")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.input_dir, args.source_dir)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({k: report[k] for k in (
        "identity_order", "changed_positions", "guard_statistics", "local_single_call_reorder_s")}))


if __name__ == "__main__":
    main()
