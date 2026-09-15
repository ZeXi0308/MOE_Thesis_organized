from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_eval_manifest import (  # noqa: E402
    EVAL_DOCUMENT_COUNT,
    FIXED_SELECTION_SALT,
    REQUIRED_TOKENS,
    PreparationError,
    canonical_json_bytes,
    canonical_text,
    load_exclusion_hashes,
    prepare_fresh_eval_manifest,
    selection_sha256,
    sha256_file,
    text_sha256,
)


def _long_document(name: str) -> str:
    return f"{name} " + ("token " * REQUIRED_TOKENS)


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value))


def _write_manifest(path: Path, texts: list[str]) -> None:
    with path.open("wb") as handle:
        for index, text in enumerate(texts):
            canonical = canonical_text(text)
            handle.write(
                canonical_json_bytes(
                    {
                        "document_index": index,
                        "split": path.stem,
                        "text": canonical,
                        "text_sha256": text_sha256(canonical),
                    }
                )
            )


def _fixture_exclusions(root: Path) -> tuple[dict[str, Path], list[str]]:
    excluded = [
        _long_document("historical"),
        _long_document("calibration"),
        _long_document("sealed"),
        _long_document("smoke"),
    ]
    historical = root / "historical.json"
    calibration = root / "calibration.jsonl"
    sealed = root / "sealed.jsonl"
    smoke = root / "smoke.jsonl"
    historical_digest = text_sha256(excluded[0])
    _write_json(
        historical,
        {
            "schema_version": "fixture",
            "hashes": [historical_digest],
            "prefix_hashes": [historical_digest],
        },
    )
    _write_manifest(calibration, [excluded[1]])
    _write_manifest(sealed, [excluded[2]])
    _write_manifest(smoke, [excluded[3]])
    return (
        {
            "historical_registry": historical,
            "calibration_manifest": calibration,
            "sealed_manifest": sealed,
            "smoke_manifest": smoke,
        },
        excluded,
    )


def _token_lengths(texts: list[str]) -> list[int]:
    return [len(text.split()) for text in texts]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class PrepareEvalManifestTest(unittest.TestCase):
    def test_canonical_text_and_fixed_selection_hash(self) -> None:
        self.assertEqual(canonical_text("a\r\nb\rc"), "a\nb\nc")
        digest = text_sha256("document")
        expected = hashlib.sha256(
            f"{FIXED_SELECTION_SALT}|{digest}".encode("utf-8")
        ).hexdigest()
        self.assertEqual(selection_sha256(digest), expected)

    def test_exclusions_merge_all_four_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources, excluded = _fixture_exclusions(root)
            hashes, report = load_exclusion_hashes(sources)
            self.assertEqual(hashes, {text_sha256(text) for text in excluded})
            self.assertEqual(set(report), set(sources))
            self.assertEqual(report["historical_registry"]["record_count"], 2)

    def test_preparation_is_exact_unique_disjoint_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources, excluded = _fixture_exclusions(root)
            candidates = excluded + [_long_document(f"candidate-{index}") for index in range(40)]
            candidates.append(candidates[-1])
            first = root / "first"
            second = root / "second"
            result = prepare_fresh_eval_manifest(
                candidate_documents=candidates,
                token_lengths=_token_lengths,
                exclusion_sources=sources,
                output_dir=first,
                source_provenance={"fixture": True},
            )
            prepare_fresh_eval_manifest(
                candidate_documents=reversed(candidates),
                token_lengths=_token_lengths,
                exclusion_sources=sources,
                output_dir=second,
                source_provenance={"fixture": True},
            )

            rows = _read_jsonl(first / "eval_manifest.jsonl")
            selected_hashes = [str(row["text_sha256"]) for row in rows]
            self.assertEqual(len(rows), EVAL_DOCUMENT_COUNT)
            self.assertEqual(len(set(selected_hashes)), EVAL_DOCUMENT_COUNT)
            self.assertTrue(all(int(row["token_length_at_least"]) >= REQUIRED_TOKENS for row in rows))
            self.assertFalse(set(selected_hashes) & {text_sha256(text) for text in excluded})
            self.assertEqual(
                first.joinpath("eval_manifest.jsonl").read_bytes(),
                second.joinpath("eval_manifest.jsonl").read_bytes(),
            )
            self.assertEqual(result["selected_document_count"], EVAL_DOCUMENT_COUNT)

            provenance = json.loads((first / "provenance.json").read_text(encoding="utf-8"))
            report = json.loads((first / "exclusion_report.json").read_text(encoding="utf-8"))
            hashes = json.loads((first / "artifact_hashes.json").read_text(encoding="utf-8"))
            self.assertEqual(provenance["eval_manifest_sha256"], sha256_file(first / "eval_manifest.jsonl"))
            self.assertEqual(report["selected_overlap_count"], 0)
            self.assertEqual(report["selected_document_count"], EVAL_DOCUMENT_COUNT)
            for name, digest in hashes["files"].items():
                self.assertEqual(digest, sha256_file(first / name))

    def test_short_documents_are_not_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources, _ = _fixture_exclusions(root)
            short = [f"short-{index}" for index in range(12)]
            long = [_long_document(f"long-{index}") for index in range(EVAL_DOCUMENT_COUNT)]
            output = root / "output"
            prepare_fresh_eval_manifest(
                candidate_documents=short + long,
                token_lengths=_token_lengths,
                exclusion_sources=sources,
                output_dir=output,
            )
            report = json.loads((output / "exclusion_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["short_candidate_count"], len(short))
            self.assertEqual(report["selected_document_count"], EVAL_DOCUMENT_COUNT)

    def test_insufficient_eligible_documents_fails_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources, _ = _fixture_exclusions(root)
            output = root / "output"
            with self.assertRaisesRegex(PreparationError, "exact-32"):
                prepare_fresh_eval_manifest(
                    candidate_documents=[
                        _long_document(f"candidate-{index}")
                        for index in range(EVAL_DOCUMENT_COUNT - 1)
                    ],
                    token_lengths=_token_lengths,
                    exclusion_sources=sources,
                    output_dir=output,
                )
            self.assertFalse(output.exists())

    def test_existing_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources, _ = _fixture_exclusions(root)
            output = root / "output"
            candidates = [
                _long_document(f"candidate-{index}") for index in range(EVAL_DOCUMENT_COUNT)
            ]
            prepare_fresh_eval_manifest(
                candidate_documents=candidates,
                token_lengths=_token_lengths,
                exclusion_sources=sources,
                output_dir=output,
            )
            before = (output / "eval_manifest.jsonl").read_bytes()
            with self.assertRaisesRegex(PreparationError, "refusing to overwrite"):
                prepare_fresh_eval_manifest(
                    candidate_documents=candidates,
                    token_lengths=_token_lengths,
                    exclusion_sources=sources,
                    output_dir=output,
                )
            self.assertEqual(before, (output / "eval_manifest.jsonl").read_bytes())

    def test_manifest_full_text_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources, _ = _fixture_exclusions(root)
            sources["sealed_manifest"].write_text(
                json.dumps({"text": "changed", "text_sha256": "0" * 64}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PreparationError, "full-text hash mismatch"):
                load_exclusion_hashes(sources)


if __name__ == "__main__":
    unittest.main()
