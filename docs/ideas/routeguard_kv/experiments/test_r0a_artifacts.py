from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from r0a_artifacts import (
    _cuda_driver_version,
    ArtifactError,
    assert_formal_approval,
    build_frozen_bindings,
    canonical_json_sha256,
    verify_frozen_bindings,
    write_json_no_overwrite,
)


def test_cuda_driver_version_is_recorded_from_nvidia_smi() -> None:
    class Result:
        returncode = 0
        stdout = "580.76.05\n"

    with patch("r0a_artifacts.subprocess.run", return_value=Result()) as run:
        assert _cuda_driver_version(True) == "580.76.05"
        run.assert_called_once_with(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    assert _cuda_driver_version(False) is None


def test_no_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    write_json_no_overwrite(path, {"b": 2, "a": 1})
    assert path.read_text(encoding="utf-8") == '{"a":1,"b":2}\n'
    try:
        write_json_no_overwrite(path, {"changed": True})
    except ArtifactError as exc:
        assert "overwrite" in str(exc)
    else:
        raise AssertionError("second write must fail closed")


def test_frozen_bindings_detect_source_change(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[4]
    config = repo / "docs/ideas/routeguard_kv/experiments/configs/r0a_5090_v1.json"
    manifest = tmp_path / "manifest.jsonl"
    source = tmp_path / "source.py"
    manifest.write_text('{"text_sha256":"' + "a" * 64 + '"}\n', encoding="utf-8")
    source.write_text("x = 1\n", encoding="utf-8")
    bindings = build_frozen_bindings(
        repo_root=repo,
        config_path=config,
        manifest_path=manifest,
        source_paths=[source],
    )
    verify_frozen_bindings(bindings, repo)
    source.write_text("x = 2\n", encoding="utf-8")
    try:
        verify_frozen_bindings(bindings, repo)
    except ArtifactError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("mutated frozen source must fail binding verification")


def test_approval_is_bound_to_exact_artifact_set(tmp_path: Path) -> None:
    config = {
        "approval": {"required_literal": "GPU Run Approved"},
    }
    bindings = {"schema_version": "routeguard-kv-frozen-bindings-v1", "files": {"x": "y"}}
    digest = canonical_json_sha256(bindings)
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps({"approval": "GPU Run Approved", "frozen_bindings_sha256": digest}),
        encoding="utf-8",
    )
    assert_formal_approval(approval, config, digest)
    try:
        assert_formal_approval(approval, config, "0" * 64)
    except ArtifactError as exc:
        assert "not bound" in str(exc)
    else:
        raise AssertionError("approval for a different binding must fail")
