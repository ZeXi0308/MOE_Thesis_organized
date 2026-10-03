from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_formal_refuses_before_output_without_approval(tmp_path: Path) -> None:
    experiment_dir = Path(__file__).resolve().parent
    config = experiment_dir / "configs/r0a_5090_v1.json"
    manifest = tmp_path / "sealed_manifest.jsonl"
    provenance = tmp_path / "provenance.json"
    registry = tmp_path / "historical_hash_registry.json"
    provenance.write_text("{}\n", encoding="utf-8")
    registry.write_text("{}\n", encoding="utf-8")
    with manifest.open("x", encoding="utf-8") as handle:
        for index in range(32):
            text = f"sealed placeholder {index}"
            import hashlib

            handle.write(
                json.dumps(
                    {
                        "schema_version": "routeguard-kv-document-v1",
                        "split": "sealed",
                        "document_index": index,
                        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "text": text,
                    }
                )
                + "\n"
            )
    output = tmp_path / "must_not_exist"
    result = subprocess.run(
        [
            sys.executable,
            str(experiment_dir / "run_r0a_5090.py"),
            "--config",
            str(config),
            "--manifest",
            str(manifest),
            "--data-provenance",
            str(provenance),
            "--historical-registry",
            str(registry),
            "--output-dir",
            str(output),
            "--phase",
            "formal",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "formal phase requires --approval" in result.stderr
    assert not output.exists()


def test_calibration_refuses_before_output_without_smoke_qualification(tmp_path: Path) -> None:
    experiment_dir = Path(__file__).resolve().parent
    output = tmp_path / "must_not_exist"
    result = subprocess.run(
        [
            sys.executable,
            str(experiment_dir / "run_r0a_5090.py"),
            "--config",
            str(experiment_dir / "configs/r0a_5090_v1.json"),
            "--manifest",
            str(tmp_path / "not_opened.jsonl"),
            "--data-provenance",
            str(tmp_path / "not_opened_provenance.json"),
            "--historical-registry",
            str(tmp_path / "not_opened_registry.json"),
            "--output-dir",
            str(output),
            "--phase",
            "calibration",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "calibration phase requires --qualification from smoke" in result.stderr
    assert not output.exists()
