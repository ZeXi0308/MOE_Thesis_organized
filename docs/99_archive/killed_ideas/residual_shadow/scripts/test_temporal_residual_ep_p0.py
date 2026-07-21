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

import torch

from run_temporal_residual_ep_p0 import encode_sequence


def test_no_revisit_falls_back_to_fp8() -> None:
    raw = torch.tensor(
        [
            [[1.0, -2.0], [3.0, -4.0]],
            [[2.0, -1.0], [4.0, -3.0]],
        ],
        dtype=torch.bfloat16,
    )
    selected = torch.tensor([[0, 1], [2, 3]])
    temporal, stats = encode_sequence(raw, selected, "temporal_delta_mxfp4")
    fp8, _ = encode_sequence(raw, selected, "uniform_fp8")
    assert stats.revisits == 0
    assert torch.equal(temporal, fp8)


def test_revisit_tracks_identity_across_rank_change() -> None:
    raw = torch.tensor(
        [
            [[1.0, 2.0], [4.0, 8.0]],
            [[4.25, 8.25], [1.25, 2.25]],
        ],
        dtype=torch.bfloat16,
    )
    selected = torch.tensor([[3, 7], [7, 3]])
    _, stats = encode_sequence(raw, selected, "temporal_delta_mxfp4")
    assert stats.revisits == 2
    assert stats.same_rank_revisits == 0


def test_temporal_and_direct_controls_have_identical_payload() -> None:
    generator = torch.Generator().manual_seed(7)
    raw = torch.randn(4, 3, 32, generator=generator).to(torch.bfloat16)
    selected = torch.tensor([[0, 1, 2], [2, 0, 3], [2, 4, 0], [5, 2, 0]])
    _, temporal = encode_sequence(raw, selected, "temporal_delta_mxfp4")
    _, direct = encode_sequence(raw, selected, "revisit_abs_mxfp4")
    assert temporal.revisits == direct.revisits
    assert temporal.payload_bytes == direct.payload_bytes
    assert temporal.mode_bits == direct.mode_bits
