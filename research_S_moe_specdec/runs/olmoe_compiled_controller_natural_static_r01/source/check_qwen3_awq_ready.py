#!/usr/bin/env python3
"""Read-only Qwen3 AWQ/EAGLE asset gate. Never downloads or imports GPU code.

python code/check_qwen3_awq_ready.py --output runs/qwen3_awq_asset_ready/ready.json

Exit 0 means both ngram/EAGLE assets and plans are ready, 2 means pending,
1 means an invalid size/hash/config. Only the selected S/runs JSON is written.
When a required target file is missing, large-file hashing is deferred until
all required sizes are present. No prior observed_hash_verified flag is trusted.
Asset ready does not establish native loading, kernels, acceptance, or quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import download_draft as common

PROJECT = Path(__file__).resolve().parents[1]
OPTIONAL = {".gitattributes", "README.md", "recipe.yaml"}
SMALL_FILE_BYTES = 32 << 20


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def asset_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"Unsafe inventory name: {name}")
    return Path(root) / relative


def check_files(root, entries):
    """Reuse common.verify; avoid hashing large complete files on a missing set."""
    rows = []
    for entry in entries:
        path = asset_path(root, entry["name"])
        size = entry.get("bytes", entry.get("size"))
        if not isinstance(size, int) or size < 0 or not (entry.get("sha256") or entry.get("git_blob_sha1")):
            raise ValueError(f"Incomplete pinned inventory: {entry['name']}")
        actual = path.stat().st_size if path.is_file() else None
        rows.append({"name": entry["name"], "required": entry.get("required_for_native_load", True),
                     "expected_bytes": size, "observed_bytes": actual,
                     "expected_sha256": entry.get("sha256"), "expected_git_blob_sha1": entry.get("git_blob_sha1"),
                     "status": "missing" if actual is None else "size_mismatch" if actual != size else "size_verified_hash_pending"})
    complete_sizes = all(r["status"] == "size_verified_hash_pending" for r in rows if r["required"])
    for row, entry in zip(rows, entries):
        if row["status"] != "size_verified_hash_pending":
            continue
        if not complete_sizes and row["expected_bytes"] > SMALL_FILE_BYTES:
            row["status"] = "hash_deferred_until_required_sizes_complete"
            continue
        path = asset_path(root, row["name"])
        before = path.stat()
        try:
            actual_hash = common.verify(path, {**entry, "size": row["expected_bytes"]})
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                row["status"] = "changed_during_read"
            else:
                row.update(status="verified", verified_sha256=actual_hash)
        except (OSError, ValueError) as exc:
            row.update(status="hash_or_read_error", error=str(exc))
    return {"path": str(root), "ready": all(r["status"] == "verified" for r in rows if r["required"]),
            "hash_policy": "large files hashed only after all required sizes are complete; small files checked now",
            "files": rows}


def verified_json(group, name):
    row = next((r for r in group["files"] if r["name"] == name), {})
    return json.loads(asset_path(group["path"], name).read_text()) if row.get("status") == "verified" else None


def check(project):
    metadata_path = project / "configs/qwen3_awq_target.json"
    metadata = json.loads(metadata_path.read_text())
    entries = metadata["files"]
    if len(entries) != 17 or len({r["name"] for r in entries}) != 17:
        raise ValueError("Expected the fixed 17-file AWQ inventory")
    if {r["name"] for r in entries if not r["required_for_native_load"]} != OPTIONAL:
        raise ValueError("Only the three declared non-runtime files may be optional")
    target = check_files(Path(metadata["model_path"]), entries)
    config, index = verified_json(target, "config.json"), verified_json(target, "model.safetensors.index.json")
    errors, pending, checks = [], [], {}
    if config is None or index is None:
        pending.append("target config/index not yet hash verified")
    else:
        quant = config.get("quantization_config") or {}
        expected_shards = {r["name"] for r in entries if r["name"].endswith(".safetensors")}
        checks["indexed_shards_match_inventory"] = set(index.get("weight_map", {}).values()) == expected_shards
        checks["target_architecture_matches"] = config.get("architectures") == ["Qwen3MoeForCausalLM"]
        checks["target_compressed_tensors"] = quant.get("quant_method") == "compressed-tensors" and quant.get("format") == "pack-quantized"
        groups = list((quant.get("config_groups") or {}).values())
        checks["target_symmetric_w4_group128"] = bool(groups) and all(
            (g.get("weights") or {}).get("num_bits") == 4 and (g.get("weights") or {}).get("symmetric") is True
            and (g.get("weights") or {}).get("group_size") == 128 for g in groups)
    plans = {}
    for method in ("ngram", "eagle3"):
        for kind in ("smoke", "reduced"):
            path = project / f"configs/qwen3_awq_{method}_{kind}.json"
            plan = json.loads(path.read_text())
            name = f"{method}_{kind}"
            valid = plan["model"] == metadata["model_path"] and plan["model_revision"] == metadata["revision"]
            valid = valid and plan.get("quantization") == "compressed-tensors" and plan.get("draft_source") == method
            if method == "eagle3":
                valid = valid and plan.get("draft_revision") == metadata["eagle3"]["revision"]
                overrides = plan.get("speculative_overrides", {})
                valid = valid and "quantization" in overrides and overrides["quantization"] is None
                valid = valid and Path(plan["draft_model"]).resolve() != Path(plan["model"]).resolve()
            checks[f"plan_{name}"] = bool(valid)
            plans[name] = {"path": str(path.relative_to(project)), "sha256": digest(path), "plan": plan}
    draft_path = Path(plans["eagle3_smoke"]["plan"]["draft_model"])
    checks["eagle_plans_use_same_draft"] = str(draft_path) == plans["eagle3_reduced"]["plan"]["draft_model"]
    sources_path = project / "configs/draft_sources.json"
    source = json.loads(sources_path.read_text())["sources"]["qwen3_a3b_eagle3"]
    checks["eagle_pinned_source_matches_target_metadata"] = (
        source["revision"] == metadata["eagle3"]["revision"]
        and source["repo_id"] == metadata["eagle3"]["repo_id"])
    cache_path = draft_path.parent / "draft_metadata.json"
    draft = {"path": str(draft_path), "ready": False, "files": []}
    if not cache_path.is_file():
        pending.append("pinned local EAGLE file inventory unavailable; no network lookup attempted")
    else:
        cache = json.loads(cache_path.read_text())
        if cache.get("id") != source["repo_id"] or cache.get("sha") != source["revision"]:
            errors.append("local EAGLE inventory repository/revision mismatch")
        else:
            draft_entries = []
            for name in ("config.json", "model.safetensors"):
                item = next((r for r in cache["siblings"] if r["rfilename"] == name), None)
                if item is None:
                    errors.append(f"EAGLE inventory missing {name}")
                    continue
                lfs = item.get("lfs") or {}
                draft_entries.append({"name": name, "size": lfs.get("size", item.get("size")),
                    "sha256": lfs.get("sha256"), "git_blob_sha1": item.get("blobId") if not lfs else None})
            if len(draft_entries) == 2:
                known = source["known_files"]["model.safetensors"]
                if draft_entries[1]["size"] != known["size"] or draft_entries[1]["sha256"] != known["sha256"]:
                    errors.append("EAGLE weight inventory differs from pinned known weight")
                else:
                    draft = check_files(draft_path, draft_entries)
            draft_config = verified_json(draft, "config.json")
            if draft_config is not None:
                checks["draft_independent_dense_bf16"] = (
                    draft_config.get("architectures") == ["LlamaForCausalLMEagle3"]
                    and draft_config.get("model_type") == "llama" and draft_config.get("num_hidden_layers") == 1
                    and not draft_config.get("num_experts") and not draft_config.get("num_local_experts")
                    and draft_config.get("dtype", draft_config.get("torch_dtype")) == "bfloat16")
                checks["draft_has_no_quantization_config"] = not draft_config.get("quantization_config")
                if config is not None:
                    checks["draft_target_shared_config_keys"] = all(
                        draft_config.get(key) == config.get(key) for key in source["shared_config_keys"])
    errors.extend(name for name, passed in checks.items() if not passed)
    for group in (target, draft):
        for row in group["files"]:
            if not row["required"] or row["status"] == "verified":
                continue
            message = f"{group['path']}/{row['name']}: {row['status']}"
            (errors if row["status"] in ("size_mismatch", "hash_or_read_error") else pending).append(message)
    structural = all(checks.get(k) for k in (
        "indexed_shards_match_inventory", "target_architecture_matches", "target_compressed_tensors", "target_symmetric_w4_group128"))
    ngram_ready = bool(target["ready"] and structural and all(checks.get(f"plan_ngram_{x}") for x in ("smoke", "reduced")))
    eagle_ready = bool(ngram_ready and draft["ready"] and not errors and all(checks.get(k) for k in (
        "plan_eagle3_smoke", "plan_eagle3_reduced", "eagle_plans_use_same_draft", "draft_independent_dense_bf16",
        "draft_has_no_quantization_config", "draft_target_shared_config_keys")))
    ready = ngram_ready and eagle_ready
    return {"schema_version": 1, "checked_at_utc": common.timestamp(), "status": "ready" if ready else "invalid" if errors else "pending",
        "ready": ready, "ngram_ready": ngram_ready, "eagle3_ready": eagle_ready,
        "target_inventory": {"path": str(metadata_path.relative_to(project)), "sha256": digest(metadata_path),
            "repo_id": metadata["repo_id"], "revision": metadata["revision"], "file_count": 17},
        "target": target, "draft": draft, "configuration_checks": checks, "errors": errors, "pending": pending,
        "plans": {k:{x:y for x,y in value.items() if x != "plan"} for k,value in plans.items()},
        "scope": "asset/content and static plan readiness only; native load, SM120 kernels, quantized-target EAGLE acceptance and quality remain unverified",
        "read_only_assets": True, "downloads_attempted": False, "gpu_initialized": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--output", type=Path, default=Path("runs/qwen3_awq_asset_ready/ready.json"))
    args = parser.parse_args()
    project = args.project.resolve()
    output = (args.output if args.output.is_absolute() else project / args.output).resolve()
    if not output.is_relative_to((project / "runs").resolve()) or output.name != "ready.json":
        parser.error("Output must be an S/runs/.../ready.json file")
    report = check(project)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n")
    temp.replace(output)
    print(json.dumps({"output":str(output), "status":report["status"], "ready":report["ready"],
                      "errors":report["errors"], "pending":report["pending"]}))
    return 0 if report["ready"] else 1 if report["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
