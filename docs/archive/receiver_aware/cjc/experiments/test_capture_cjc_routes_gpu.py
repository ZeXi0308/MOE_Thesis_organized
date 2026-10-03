from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from capture_cjc_routes_gpu import (
    _expert_sender,
    _load_manifest,
    _origin_lpt,
    _require_signoff,
    canonical_json_bytes,
    sha256_bytes,
)


def _write_manifest(path: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "sequence_tokens": 128,
        "requests": [{"request_id": "r0", "text": "x", "text_sha256": "unused"}],
    }
    payload["manifest_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


class CjcCaptureTest(unittest.TestCase):
    def test_manifest_self_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            payload = _write_manifest(path)
            self.assertEqual(
                _load_manifest(path)["manifest_sha256"],
                payload["manifest_sha256"],
            )
            payload["sequence_tokens"] = 127
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "self-hash"):
                _load_manifest(path)

    def test_origin_lpt_is_route_blind_and_deterministic(self) -> None:
        requests = [{"request_id": key} for key in ("c", "a", "b", "d")]
        self.assertEqual(_origin_lpt(requests, 2), {"a": 0, "b": 1, "c": 0, "d": 1})

    def test_expert_placement_is_explicit(self) -> None:
        self.assertEqual(
            [_expert_sender(index, 8, 4, "contiguous") for index in range(8)],
            [0, 0, 1, 1, 2, 2, 3, 3],
        )
        self.assertEqual(
            [_expert_sender(index, 8, 4, "round_robin") for index in range(8)],
            [0, 1, 2, 3, 0, 1, 2, 3],
        )

    def test_formal_signoff_binds_all_capture_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signoff.json"
            payload = {
                "status": "SIGNED-OFF",
                "protocol_sha256": "p",
                "capture_source_sha256": "s",
                "data_manifest_sha256": "d",
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                _require_signoff(
                    path,
                    protocol_sha256="p",
                    source_sha256="s",
                    data_manifest_sha256="d",
                )["status"],
                "SIGNED-OFF",
            )
            with self.assertRaisesRegex(RuntimeError, "capture_source_sha256"):
                _require_signoff(
                    path,
                    protocol_sha256="p",
                    source_sha256="different",
                    data_manifest_sha256="d",
                )


if __name__ == "__main__":
    unittest.main()
