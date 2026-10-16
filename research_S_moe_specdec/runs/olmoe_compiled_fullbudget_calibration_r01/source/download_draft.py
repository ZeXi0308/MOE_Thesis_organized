#!/usr/bin/env python3
"""Fetch the pinned EAGLE checkpoint and exact target tokenizer, using only stdlib.

All generated files, cached metadata, and resumable .part files stay in this
project's models/ directory. No target weights, GPU work, or global HF cache.

  python code/download_draft.py --resolve-only
  python code/download_draft.py --endpoint https://hf-mirror.com
  python code/download_draft.py --target-path /path/to/existing/Qwen3-30B-A3B

--resolve-only persists the pinned file inventories and reports exact bytes.
Full download prints draft_path and tokenizer_path for the runner. It verifies
content hashes and reports structural compatibility, not untested runtime or
undisclosed training-revision compatibility. Interrupted transfers are resumed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from http.client import IncompleteRead
import json
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


PROJECT = Path(__file__).resolve().parents[1]
MODELS = PROJECT / "models"
ENDPOINTS = ("https://huggingface.co", "https://hf-mirror.com")
TOKENIZER_NAMES = {
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "tokenizer.model", "vocab.json", "vocab.txt", "merges.txt", "added_tokens.json",
    "special_tokens_map.json", "chat_template.jinja", "LICENSE", "README.md",
}


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def confined(path):
    """Reject path traversal and pre-existing symlinks escaping project/models."""
    root = MODELS.absolute()
    if root.resolve() != root:
        raise ValueError("models/ must not be a symlink or have a symlinked ancestor")
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("all outputs must stay inside this project's models/")
    return resolved


def save_json(path, data):
    path = confined(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = confined(path.with_name(path.name + ".tmp"))
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def file_path(base, name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts:
        raise ValueError(f"unsafe repository path: {name!r}")
    return confined(base / name)


def request(url, offset=0):
    headers = {"User-Agent": "moe-specdec-pinned-download/1", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return urlopen(Request(url, headers=headers), timeout=60)


def inventory(repo, revision, endpoint, cache):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("a full immutable HF commit SHA is required")
    if cache.exists():
        data = json.loads(cache.read_text())
    else:
        url = f"{endpoint}/api/models/{repo}/revision/{revision}?blobs=true"
        with request(url) as response:
            data = json.load(response)
        if data.get("sha") != revision:
            raise ValueError("HF metadata did not resolve to the pinned commit")
        save_json(cache, data)
    if data.get("sha") != revision or data.get("id", repo) != repo:
        raise ValueError("cached metadata does not match pinned repository/revision")
    records = []
    for item in data["siblings"]:
        lfs = item.get("lfs") or {}
        size = lfs.get("size", item.get("size"))
        sha256 = lfs.get("sha256") or lfs.get("oid", "").removeprefix("sha256:")
        git_sha1 = item.get("blobId")
        if not isinstance(size, int) or size < 0:
            raise ValueError(f"missing file size: {item['rfilename']}")
        if not (re.fullmatch(r"[0-9a-f]{64}", sha256 or "") or
                (not lfs and re.fullmatch(r"[0-9a-f]{40}", git_sha1 or ""))):
            raise ValueError(f"missing content checksum: {item['rfilename']}")
        records.append({"name": item["rfilename"], "size": size,
                        "sha256": sha256 or None, "git_blob_sha1": git_sha1 if not lfs else None,
                        "repo_id": repo, "revision": revision,
                        "url": f"{endpoint}/{repo}/resolve/{revision}/{quote(item['rfilename'], safe='/')}"})
    return records


def verify(path, entry):
    if path.stat().st_size != entry["size"]:
        raise ValueError(f"size mismatch for {path.name}")
    sha = hashlib.sha256()
    git = hashlib.sha1(f"blob {entry['size']}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(chunk)
            git.update(chunk)
    if entry.get("sha256") and sha.hexdigest() != entry["sha256"]:
        raise ValueError(f"SHA256 mismatch for {path.name}; file retained for inspection")
    if entry.get("git_blob_sha1") and git.hexdigest() != entry["git_blob_sha1"]:
        raise ValueError(f"Git blob hash mismatch for {path.name}; file retained for inspection")
    return sha.hexdigest()


def download(entry, base, retries=4):
    dest = file_path(base, entry["name"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return verify(dest, entry)
    part = confined(dest.with_name(dest.name + ".part"))
    for attempt in range(retries):
        offset = part.stat().st_size if part.exists() else 0
        if offset > entry["size"]:
            raise ValueError(f"oversized partial file retained: {part}")
        if offset == entry["size"] and part.exists():
            digest = verify(part, entry)
            part.replace(dest)
            return digest
        try:
            with request(entry["url"], offset) as response:
                status = response.status
                if status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    expected = rf"bytes {offset}-\d+/{entry['size']}"
                    if not re.fullmatch(expected, content_range):
                        raise ValueError("unexpected partial-response range")
                    mode = "ab" if offset else "wb"
                elif status == 200:
                    mode = "wb"  # Endpoint ignored Range; restart safely.
                else:
                    raise ValueError(f"unexpected HTTP status {status}")
                with part.open(mode) as stream:
                    for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
                        stream.write(chunk)
                        if stream.tell() > entry["size"]:
                            raise ValueError("response exceeded expected file size")
            if part.stat().st_size != entry["size"]:
                raise OSError("incomplete transfer")
            digest = verify(part, entry)
            part.replace(dest)
            return digest
        except (HTTPError, URLError, OSError, IncompleteRead) as exc:
            if attempt + 1 == retries:
                # Do not persist signed redirected URLs or credentials in errors.
                raise RuntimeError(f"download failed for {entry['name']}: {type(exc).__name__}; rerun to resume") from None
            time.sleep(min(2 ** attempt, 8))
    raise AssertionError("unreachable")


def safetensors_header(path):
    with path.open("rb") as stream:
        size_bytes = stream.read(8)
        if len(size_bytes) != 8:
            raise ValueError("invalid safetensors header")
        size = struct.unpack("<Q", size_bytes)[0]
        if not 2 <= size <= min(path.stat().st_size - 8, 16 * 1024 * 1024):
            raise ValueError("invalid safetensors header length")
        header = json.loads(stream.read(size))
    return {k: {"dtype": v["dtype"], "shape": v["shape"]}
            for k, v in header.items() if k != "__metadata__"}


def compatibility(source, draft, tokenizer, target_path):
    dc = json.loads((draft / "config.json").read_text())
    tc = json.loads((tokenizer / "config.json").read_text())
    errors = []
    checks = {}
    for key, expected in source["expected_draft_config"].items():
        if dc.get(key) != expected:
            errors.append(f"draft config {key}: expected {expected!r}, got {dc.get(key)!r}")
    for role, cfg, expected in (("draft", dc, source["draft_architecture"]),
                                ("target", tc, source["target_architecture"])):
        if expected not in cfg.get("architectures", []):
            errors.append(f"unexpected {role} architecture")
    for key in source["shared_config_keys"]:
        checks[key] = {"draft": dc.get(key), "target": tc.get(key), "match": dc.get(key) == tc.get(key)}
        if not checks[key]["match"]:
            errors.append(f"draft/target {key} mismatch")
    # Target and draft have intentionally different layer counts and model_type.
    if tc.get("num_hidden_layers") != 48 or tc.get("num_experts_per_tok") != 8:
        errors.append("unexpected pinned target layer count or router top-k")
    if dc.get("dtype", dc.get("torch_dtype")) != "bfloat16" or tc.get("dtype", tc.get("torch_dtype")) != "bfloat16":
        errors.append("expected BF16 draft and target configurations")
    tensor_shapes = {}
    for path in draft.glob("*.safetensors"):
        tensor_shapes.update(safetensors_header(path))
    mappings = {k: v for k, v in tensor_shapes.items() if k.split(".")[-1] in ("d2t", "t2d")}
    if not mappings:
        errors.append("reduced draft vocabulary has no d2t/t2d mapping tensor; inspect checkpoint loader")
    pinned_snapshot_status = "structural_checks_passed" if not errors else "failed"
    local_target = {"status": "not supplied; exact local target not checked"}
    if target_path:
        local_config_path = target_path / "config.json"
        local_config_exists = local_config_path.is_file()
        local_config = json.loads(local_config_path.read_text()) if local_config_exists else None
        # _name_or_path and transformers metadata may change during export.
        structural = [*source["shared_config_keys"], "architectures", "model_type",
                      "num_hidden_layers", "num_experts", "num_experts_per_tok",
                      "moe_intermediate_size", "norm_topk_prob", "quantization_config"]
        mismatch = [key for key in structural if local_config.get(key) != tc.get(key)] if local_config_exists else []
        if local_config_exists and local_config.get("dtype", local_config.get("torch_dtype")) != "bfloat16":
            mismatch.append("dtype")
        token_files = [p for p in tokenizer.iterdir() if p.name in TOKENIZER_NAMES
                       and p.name not in ("config.json", "generation_config.json", "LICENSE", "README.md")]
        token_checks = {}
        for path in token_files:
            local_path = target_path / path.name
            exists = local_path.is_file()
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            actual = hashlib.sha256(local_path.read_bytes()).hexdigest() if exists else None
            token_checks[path.name] = {"exists": exists, "expected_sha256": expected,
                                       "actual_sha256": actual,
                                       "hash_matches": actual == expected if exists else None}
        token_matches = {name: check["hash_matches"] for name, check in token_checks.items()}
        missing_files = (["config.json"] if not local_config_exists else []) + [
            name for name, check in token_checks.items() if not check["exists"]]
        mismatched_files = [name for name, match in token_matches.items() if match is False]
        local_status = ("mismatch" if mismatch or mismatched_files else
                        "pending_local_target" if missing_files else "matched_config_and_tokenizer_only")
        local_target = {"path": str(target_path), "structural_mismatches": mismatch,
                        "config_exists": local_config_exists, "missing_files": missing_files,
                        "mismatched_tokenizer_files": mismatched_files, "tokenizer_files": token_checks,
                        "tokenizer_file_hash_matches": token_matches,
                        "status": local_status,
                        "target_weights": "not downloaded or rehashed; caller must retain the existing exact target snapshot provenance"}
        if mismatch or mismatched_files:
            errors.append("existing local target differs from the pinned config/tokenizer")
    status = ("failed" if errors else "pending_local_target" if local_target["status"] == "pending_local_target"
              else "structural_checks_passed")
    report = {"checked_at": timestamp(), "status": status, "pinned_snapshot_status": pinned_snapshot_status,
              "shared_config_checks": checks, "mapping_tensors": mappings,
              "tensor_shapes": tensor_shapes, "local_target": local_target,
              "limitations": source["compatibility_status"], "errors": errors}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/draft_sources.json")
    parser.add_argument("--source")
    parser.add_argument("--endpoint", choices=ENDPOINTS, default=ENDPOINTS[0])
    parser.add_argument("--resolve-only", action="store_true")
    parser.add_argument("--target-path", type=Path, help="read-only check against the existing target snapshot")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source_id = args.source or config["default_source"]
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", source_id):
        raise ValueError("invalid source id")
    source = config["sources"][source_id]
    base = confined(MODELS / source_id / source["revision"])
    base.mkdir(parents=True, exist_ok=True)
    # One writer per bundle; released on ordinary failures, no GPU lock needed.
    import fcntl
    with confined(base / ".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        draft = base / "draft"
        tokenizer = base / "target_tokenizer"
        draft_files = inventory(source["repo_id"], source["revision"], args.endpoint, base / "draft_metadata.json")
        target_files = inventory(source["target_repo_id"], source["target_revision"], args.endpoint, base / "target_metadata.json")
        token_files = [e for e in target_files if e["name"] in TOKENIZER_NAMES or e["name"].startswith("chat_templates/")]
        if not {"config.json", "tokenizer.json", "tokenizer_config.json"}.issubset({e["name"] for e in token_files}):
            raise ValueError("pinned target lacks required tokenizer/config files")
        for name, expected in source["known_files"].items():
            actual = next((e for e in draft_files if e["name"] == name), None)
            if actual is None or any(actual.get(k) != v for k, v in expected.items()):
                raise ValueError(f"metadata disagrees with independently recorded file hash/size: {name}")
        total = sum(e["size"] for e in draft_files + token_files)
        if total > source["max_download_bytes"]:
            raise ValueError("resolved bundle exceeds the configured byte limit")
        manifest = {"resolved_at": timestamp(), "source": source, "endpoint": args.endpoint,
                    "expected_total_bytes": total, "draft_path": str(draft), "tokenizer_path": str(tokenizer),
                    "files": [{**e, "role": role} for role, files in (("draft", draft_files), ("target_tokenizer", token_files)) for e in files],
                    "status": "resolved"}
        save_json(base / "download_manifest.json", manifest)
        print(json.dumps({k: manifest[k] for k in ("expected_total_bytes", "draft_path", "tokenizer_path", "status")}), flush=True)
        if args.resolve_only:
            return 0
        for entry in manifest["files"]:
            print(f"verifying/downloading {entry['role']}/{entry['name']} ({entry['size']} bytes)", flush=True)
            entry["verified_sha256"] = download(entry, base / entry["role"])
            entry["status"] = "verified"
            save_json(base / "download_manifest.json", manifest)
        report = compatibility(source, draft, tokenizer, args.target_path)
        save_json(base / "compatibility.json", report)
        manifest["status"] = "complete" if not report["errors"] else "downloaded_but_compatibility_failed"
        manifest["compatibility_status"] = report["status"]
        save_json(base / "download_manifest.json", manifest)
        print(json.dumps({"status": manifest["status"], "draft_path": str(draft),
                          "compatibility_status": report["status"],
                          "tokenizer_path": str(tokenizer), "compatibility_report": str(base / "compatibility.json")}), flush=True)
        return 1 if report["errors"] else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
