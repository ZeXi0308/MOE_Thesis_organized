"""No network: URL mapping and load_weights fetch wiring with transport mocks."""
import argparse
import ast
from functools import partial
import hashlib
import io
import json
from pathlib import Path
import re
from types import SimpleNamespace as NS


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    fix = Path(__file__).resolve().parent
    ready = fix.parent
    source = fix / "qwen_serial_loader.py"
    manifest = json.loads((ready / "qwen3.manifest.json").read_text())
    calls, events = [], []
    def require(ok, message):
        if not ok:
            raise ValueError(message)
    def urlopen(url, *, timeout):
        calls.append((url, timeout))
        return io.BytesIO(b"CPU transport mock")
    env = dict(require=require, re=re, urlopen=urlopen, partial=partial)
    tree = ast.parse(source.read_text())
    fetch_node = next(n for n in tree.body if getattr(n, "name", None) == "fetch_mirror")
    cls = next(n for n in tree.body if getattr(n, "name", None) == "QwenSerialLoader")
    load_node = next(n for n in cls.body if getattr(n, "name", None) == "load_weights")
    exec(compile(ast.Module(body=[fetch_node, load_node], type_ignores=[]), str(source), "exec"), env)
    prefix = "https://huggingface.co/Qwen/Qwen3-30B-A3B/resolve/" + manifest["revision"] + "/"
    for shard in manifest["shards"]:
        with env["fetch_mirror"](manifest, prefix + shard) as response:
            assert response.read() == b"CPU transport mock"
        assert calls[-1] == ("https://hf-mirror.com/" + (prefix + shard)[len("https://huggingface.co/"):], 120)
    first = prefix + next(iter(manifest["shards"]))
    bad_urls = [first.replace("https://", "http://", 1), first.replace("huggingface.co", "other.example", 1),
                first.replace("Qwen/Qwen3-30B-A3B", "other/model"),
                first.replace(manifest["revision"], "0" * 40), prefix + "unknown.safetensors", first + "?download=true"]
    for url in bad_urls:
        before = len(calls)
        try:
            env["fetch_mirror"](manifest, url)
        except ValueError as error:
            assert "frozen HF shard" in str(error)
        else:
            raise AssertionError("invalid input URL accepted")
        assert len(calls) == before
    paths = dict(manifest=ready / "qwen3.manifest.json", index=ready / "qwen3.index.json",
                 workspace=output, receipt=output / "mock_created_receipt.jsonl")
    def metadata():
        assert not paths["receipt"].exists()
        events.append("manifest_before_serial")
        return manifest
    instance = NS(paths=paths, _metadata=metadata, _validate_targets=lambda *args: None,
                  download_model=lambda config: events.append("metadata_validation"))
    def serial_spy(model, manifest_path, index_path, workspace, receipt, *, validate_loaded, fetch):
        events.append("serial")
        assert [manifest_path, index_path, workspace, receipt] == [paths[k] for k in ("manifest", "index", "workspace", "receipt")]
        assert validate_loaded is instance._validate_targets and fetch.func is env["fetch_mirror"]
        assert fetch.args == (manifest,)
        receipt.write_text("CPU serial-open mock\n")
        with fetch(first) as response:
            assert response.read() == b"CPU transport mock"
    env["load_serial"] = serial_spy
    env["load_weights"](instance, object(), object())
    assert events == ["metadata_validation", "manifest_before_serial", "serial"] and len(calls) == len(manifest["shards"]) + 1
    report = dict(status="PASS_CPU_TRANSPORT_MAPPING_ONLY", mapped_frozen_shards=len(manifest["shards"]),
                  rejected_urls=len(bad_urls), no_network_on_rejection=True, socket_timeout_s=120,
                  load_serial_calls=events.count("serial"), passed_fetch="partial(fetch_mirror, validated_manifest)",
                  fetch_after_receipt_creation=True, metadata_reentered_during_fetch=False,
                  loader_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  scope="actual fetch/load_weights AST; metadata/serial/HTTP transport mocks; no network or GPU")
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
