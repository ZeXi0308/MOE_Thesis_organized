"""Regression for a replaced shared lock path during GPU session setup."""

import os
from pathlib import Path
import tempfile
import unittest

from serial_group import require_lock_identity


class LockIdentityTest(unittest.TestCase):
    def test_replaced_path_cannot_become_a_second_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu.lock"
            path.touch()
            fd = os.open(path, os.O_RDWR)
            try:
                state = os.fstat(fd)
                frozen = f"{state.st_dev}:{state.st_ino}"
                self.assertEqual(require_lock_identity(fd, path, frozen), frozen)
                path.unlink()
                path.touch()
                with self.assertRaisesRegex(ValueError, "no longer names the held inode"):
                    require_lock_identity(fd, path, frozen)
            finally:
                os.close(fd)


if __name__ == "__main__":
    unittest.main()
