"""Materialize immutable per-model capture configs from the frozen P0-B protocol.

This is a mechanical bridge between the multi-cell machine protocol and the
single-model route capture CLI.  It does not inspect model routes or sealed
outputs.  Existing outputs are never overwritten.
"""

from __future__ import annotations

# --- shared-lib bootstrap (auto) ---
import sys
from pathlib import Path as _Path

def _ensure_shared_on_path() -> None:
    here = _Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        cand = p / "experiments" / "shared"
        if (cand / "capture_moe.py").exists():
            s = str(cand)
            if s not in sys.path:
                sys.path.insert(0, s)
            return
        if (p / "capture_moe.py").exists():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
            return

_ensure_shared_on_path()
del _ensure_shared_on_path, _Path
# --- end bootstrap ---

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCAL_REVISIONS = {
    "allenai/OLMoE-1B-7B-0924": "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
    "llm-jp/optimal-sparsity-math-d512-E32-k16-920M-A520M":
        "1d5983076dfc67aee4a77ec06a27027f5bab6055",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_lines(values: list[str]) -> str:
    return sha256_bytes(("\n".join(values) + "\n").encode("utf-8"))


def model_slug(name: str) -> str:
    if name.startswith("allenai/OLMoE"):
        return "olmoe"
    if name.startswith("llm-jp/"):
        return "llmjp"
    raise ValueError(f"unregistered primary model: {name}")


def verify_local_revision(model: str, revision: str) -> None:
    cache_name = "models--" + model.replace("/", "--")
    snapshot = Path.home() / ".cache" / "huggingface" / "hub" / cache_name / "snapshots" / revision
    if not snapshot.is_dir():
        raise FileNotFoundError(f"frozen local model snapshot is absent: {snapshot}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine-protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    protocol_path = Path(args.machine_protocol).expanduser().resolve()
    raw = protocol_path.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("schema") != "routefidelity_ep.p0b.machine_protocol.v1":
        raise ValueError("unexpected machine protocol schema")
    if protocol.get("status") != "FROZEN_BEFORE_FRESH_ROUTE_CAPTURE":
        raise ValueError("machine protocol is not capture-frozen")

    data = protocol["data"]
    history = protocol["historical_exclusion_registry"]
    historical_hashes = list(history["unique_document_hashes"])
    if len(historical_hashes) != int(history["unique_document_hash_count"]):
        raise RuntimeError("historical exclusion count drift")
    if len(set(historical_hashes)) != len(historical_hashes):
        raise RuntimeError("historical exclusion contains duplicates")
    if hash_lines(sorted(historical_hashes)) != history["unique_document_hashes_sha256"]:
        raise RuntimeError("historical exclusion digest drift")

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite capture-config directory: {output}")
    output.mkdir(parents=True)

    written: dict[str, str] = {}
    for cell in protocol["primary_cells"]:
        model = cell["model"]
        revision = LOCAL_REVISIONS[model]
        verify_local_revision(model, revision)
        config = {
            "schema_version": 1,
            "status": "FROZEN_BEFORE_MODEL_ROUTE_CAPTURE",
            "provenance": {
                "machine_protocol": str(protocol_path),
                "machine_protocol_file_sha256": sha256_bytes(raw),
                "machine_protocol_payload_sha256": protocol["machine_protocol_payload_sha256"],
                "primary_cell": cell["cell"],
            },
            "dataset": {
                "name": "wikitext2_docs",
                "split": data["split"],
                "seed": data["shuffle_seed"],
            },
            "model": {"name": model, "revision": revision},
            "architecture": {
                "num_experts": cell["experts"],
                "top_k": cell["top_k"],
            },
            "dtype": "bfloat16",
            "seq_len": data["tokenization"]["maximum_non_padding_tokens_per_request"],
            "ep_size": cell["ep_size"],
            "compute_seed": 0,
            "historical_exclusion": {
                "hashes": historical_hashes,
                "combined_hashes_sha256": history["unique_document_hashes_sha256"],
            },
            "phases": {
                phase: {
                    "offset": data["partitions"][phase]["offset"],
                    "n": data["partitions"][phase]["requests"],
                    "hash_of_hashes": data["partitions"][phase][
                        "expected_hash_of_document_hashes"
                    ],
                }
                for phase in ("calibration", "sealed")
            },
        }
        payload = (json.dumps(config, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        path = output / f"capture_{model_slug(model)}.json"
        path.write_bytes(payload)
        written[path.name] = sha256_bytes(payload)

    manifest = {
        "schema": "routefidelity_ep.p0b.capture_config_manifest.v1",
        "machine_protocol_file_sha256": sha256_bytes(raw),
        "configs": written,
    }
    manifest_payload = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    (output / "manifest.json").write_bytes(manifest_payload)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
