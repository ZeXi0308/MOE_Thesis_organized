from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


HERE = Path(__file__).resolve().parent


class ReferenceImportOriginTest(unittest.TestCase):
    def test_capture_uses_shared_references_without_archive_path_injection(self) -> None:
        probe = textwrap.dedent(
            f"""
            from pathlib import Path
            import sys

            shared = Path({str(HERE)!r}).resolve()
            sys.path.insert(0, str(shared))
            before = tuple(sys.path)

            import capture_moe
            import creditreduce_reference
            import grouped_owner_combine

            assert Path(capture_moe.__file__).resolve().parent == shared
            assert Path(creditreduce_reference.__file__).resolve().parent == shared
            assert Path(grouped_owner_combine.__file__).resolve().parent == shared

            added = set(sys.path) - set(before)
            assert not any("docs/archive" in entry for entry in added)
            """
        )
        completed = subprocess.run(
            [sys.executable, "-B", "-c", probe],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
