from __future__ import annotations

import json
from pathlib import Path

from prepare_r0a_data import (
    hashes_from_artifact,
    parse_wikitext_articles,
    selection_sha256,
    text_sha256,
)
from r0a_artifacts import load_config, ordered_hash_of_hashes, sha256_file
from run_r0a_5090 import RunError, _validate_data_provenance


def test_article_parser_preserves_article_independence() -> None:
    rows = [
        "= First =",
        "a " * 300,
        "= = Subsection = =",
        "b " * 20,
        "= Second =",
        "short",
    ]
    articles = parse_wikitext_articles(rows, min_chars=500)
    assert len(articles) == 1
    assert articles[0].startswith("First")
    assert "Subsection" in articles[0]


def test_historical_scanner_only_accepts_recognized_hash_fields(tmp_path: Path) -> None:
    digest = "a" * 64
    ignored = "b" * 64
    path = tmp_path / "manifest.json"
    path.write_text(
        '{"nested":{"text_sha256":"' + digest + '"},"unrelated":"' + ignored + '"}',
        encoding="utf-8",
    )
    assert hashes_from_artifact(path) == {digest}


def test_selection_hash_is_salted_and_deterministic() -> None:
    digest = text_sha256("document")
    assert selection_sha256("salt", digest) == selection_sha256("salt", digest)
    assert selection_sha256("salt", digest) != selection_sha256("other", digest)


def test_data_provenance_binds_registry_and_manifest_order(tmp_path: Path) -> None:
    config_path = Path(__file__).parent / "configs/r0a_5090_v1.json"
    config = load_config(config_path)
    digest = "c" * 64
    registry_path = tmp_path / "historical_hash_registry.json"
    registry_path.write_text(
        json.dumps({"schema_version": "routeguard-kv-historical-hashes-v1", "hashes": []}) + "\n",
        encoding="utf-8",
    )
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "schema_version": "routeguard-kv-data-provenance-v1",
                "config_sha256": sha256_file(config_path),
                "dataset_repo_id": config["dataset"]["repo_id"],
                "dataset_config": config["dataset"]["config"],
                "dataset_revision": config["dataset"]["revision"],
                "dataset_split": config["dataset"]["split"],
                "required_tokens": config["dataset"]["required_tokens"],
                "historical_hash_registry_sha256": sha256_file(registry_path),
                "ordered_hash_of_hashes": {"smoke": ordered_hash_of_hashes([digest])},
                "selected_token_lengths": {digest: config["dataset"]["required_tokens"]},
                "dataset_fingerprint": "fixture-fingerprint",
                "historical_unique_count": 0,
                "eligible_count": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = _validate_data_provenance(
        provenance_path,
        registry_path,
        config_path=config_path,
        config=config,
        manifest=[{"text_sha256": digest}],
        phase="smoke",
    )
    assert result["dataset_fingerprint"] == "fixture-fingerprint"
    registry_path.write_text('{"schema_version":"wrong","hashes":[]}\n', encoding="utf-8")
    try:
        _validate_data_provenance(
            provenance_path,
            registry_path,
            config_path=config_path,
            config=config,
            manifest=[{"text_sha256": digest}],
            phase="smoke",
        )
    except RunError as exc:
        assert "data provenance validation failed" in str(exc)
    else:
        raise AssertionError("mutated historical registry must fail provenance validation")
