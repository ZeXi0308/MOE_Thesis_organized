from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from triage_artifacts import ArtifactError, JsonlJournal, source_manifest, write_json_no_overwrite


class ArtifactTests(unittest.TestCase):
    def test_no_overwrite_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "value.json"
            write_json_no_overwrite(target, {"a": 1})
            with self.assertRaises(ArtifactError):
                write_json_no_overwrite(target, {"a": 2})
            journal = JsonlJournal(root / "raw.jsonl", resume=False)
            journal.append({"resume_key": "one", "value": 1})
            resumed = JsonlJournal(root / "raw.jsonl", resume=True)
            self.assertEqual(resumed.completed_keys, {"one"})
            with self.assertRaises(ArtifactError):
                resumed.append({"resume_key": "one", "value": 2})

    def test_source_manifest_changes_with_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.py"
            path.write_text("a")
            first = source_manifest([path])
            path.write_text("b")
            second = source_manifest([path])
            self.assertNotEqual(first["aggregate_sha256"], second["aggregate_sha256"])

    def test_source_manifest_relative_keys_are_relocatable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "nested" / "source.py"
            path.parent.mkdir()
            path.write_text("x", encoding="utf-8")
            manifest = source_manifest([path], root=root)
            self.assertEqual(list(manifest["files"]), ["nested/source.py"])


if __name__ == "__main__":
    unittest.main()
