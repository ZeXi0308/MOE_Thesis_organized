from __future__ import annotations

import unittest

from triage_manifest import ManifestError, canonical_text, select_documents, text_sha256


class ManifestTests(unittest.TestCase):
    def test_canonicalization_only_changes_newlines(self) -> None:
        self.assertEqual(canonical_text(" a\r\nb\rc "), " a\nb\nc ")

    def test_selection_is_deterministic_disjoint_and_excludes(self) -> None:
        documents = [f"document {index}" for index in range(12)]
        excluded = {text_sha256(documents[0])}
        first = select_documents(
            documents, seed=4, calibration_count=3, sealed_count=5, excluded_hashes=excluded
        )
        second = select_documents(
            documents, seed=4, calibration_count=3, sealed_count=5, excluded_hashes=excluded
        )
        self.assertEqual(first, second)
        calibration, sealed = first
        calibration_hashes = {row["text_sha256"] for row in calibration}
        sealed_hashes = {row["text_sha256"] for row in sealed}
        self.assertFalse(calibration_hashes & sealed_hashes)
        self.assertNotIn(text_sha256(documents[0]), calibration_hashes | sealed_hashes)

    def test_insufficient_documents_fail(self) -> None:
        with self.assertRaises(ManifestError):
            select_documents(["one"], seed=1, calibration_count=1, sealed_count=1, excluded_hashes=[])


if __name__ == "__main__":
    unittest.main()
