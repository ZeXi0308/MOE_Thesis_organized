#!/usr/bin/env python3
"""Plot complete per-request maximum-gap ECDFs from the frozen R02 triplet."""

import hashlib
import json
import math
import os
from pathlib import Path

PAPER = Path(__file__).resolve().parent
SOURCE = PAPER.parent / "A_NATIVE_OLDEST_REPEAT_TRIPLET_RESULT_R02_20261002.json"
SOURCE_SHA256 = "b1cc9b0487e3ed372f277606fd763c750d9c3aecc1948aa06dbd58da4482f5da"
FIGURES = PAPER / "figures"
MPL_CONFIG = PAPER / "tmp" / "mplconfig"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def gaps(arm: dict) -> list[float]:
    requests = arm["metrics"]["requests"]
    if len(requests) != 128 or not all(request["completed"] for request in requests):
        raise ValueError("Expected all 128 requests to complete in each arm")
    if len({request["request_id"] for request in requests}) != 128:
        raise ValueError("Expected 128 distinct requests in each arm")
    values = sorted(float(request["max_gap_s"]) for request in requests)
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("All maximum gaps must be finite and positive")
    return values


def main() -> None:
    source_bytes = SOURCE.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != SOURCE_SHA256:
        raise ValueError("Frozen source SHA-256 mismatch")
    data = json.loads(source_bytes)
    styles = (
        ("native", "Native", "#1f77b4", "-"),
        ("queue_only", "Queue only", "#ff7f0e", "--"),
        ("queue_fund", "Queue fund", "#2ca02c", "-"),
    )

    fig, ax = plt.subplots(figsize=(6.4, 3.5), constrained_layout=True)
    for key, label, color, linestyle in styles:
        values = gaps(data["arms"][key])
        x = [values[0] / 1.2, *values]
        y = [0.0, *(rank / len(values) for rank in range(1, len(values) + 1))]
        ax.step(x, y, where="post", label=label, color=color,
                linestyle=linestyle, linewidth=1.9)
    ax.set_xscale("log")
    ax.set_xlim(0.015, 16)
    ax.set_ylim(0, 1.01)
    ax.set_yticks((0, 0.25, 0.5, 0.75, 1.0))
    ax.set_xlabel("Per-request maximum output gap (s; log scale)")
    ax.set_ylabel("Fraction of requests")
    ax.grid(True, which="major", color="#d8d8d8", linewidth=0.6)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(FIGURES / f"oldest_repeat_r02_gap_ecdf.{extension}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
