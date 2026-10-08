#!/usr/bin/env python3
"""Plot separate request-weighted ECDFs for both capacity-victim pairs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
SOURCES = {
    1: HERE / "A_CAPACITY_VICTIM_PAIR_RESULT_R01_20261001.json",
    2: HERE / "A_CAPACITY_VICTIM_PAIR_B2_RESULT_R01_20261001.json",
}
B1_SHA256 = "f91eb78527a33bc3096804699ca84f20c9a030c999dd8f7b486325427c104439"
OUT = HERE / "A_CAPACITY_VICTIM_TWO_BLOCKS_FIGURE_R01_20261001"
METRICS = (
    ("max_gap_s", "Largest inter-output gap", "Seconds"),
    ("flow_s", "Arrival-to-completion flow", "Seconds"),
    ("ttft_s", "Time to first returned token", "Seconds"),
)
COLOR = {"off": "#2368a1", "on": "#d8731d"}
STYLE = {"off": "-", "on": "--"}


def read_block(block: int) -> tuple[dict, str]:
    source = SOURCES[block]
    raw = source.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if block == 1 and sha != B1_SHA256:
        raise ValueError("B1 audited result SHA-256 changed")
    result = json.loads(raw)
    status = result.get("status")
    if block == 1 and status != "CAPACITY_VICTIM_EXPLORATORY_PAIR_COMPLETE":
        raise ValueError("B1 result is incomplete")
    if block == 2 and not (isinstance(status, str) and status.endswith("PAIR_COMPLETE")):
        raise ValueError("B2 result is incomplete")
    return result, sha


def request_rows(results: dict[int, dict]) -> dict[tuple[int, str], list[dict]]:
    rows = {}
    reference = None
    for block in (1, 2):
        for arm in ("off", "on"):
            metrics = results[block]["metrics"][arm]
            requests = metrics["requests"]
            if (metrics["status"] != "COMPLETE" or metrics["completed"] != 128 or
                    metrics["failed"] != 0 or metrics["unfinished"] != 0 or
                    metrics["undefined_gap_requests"] != 0 or len(requests) != 128):
                raise ValueError(f"B{block}/{arm}: expected 128 complete metric rows")
            identity = sorted(
                (r["request_id"], r["document_id"], r["prompt_sha256"],
                 r["arrival_s"], r["max_output"]) for r in requests
            )
            if (len({r[0] for r in identity}) != 128 or
                    any(not r["completed"] for r in requests) or
                    (reference is not None and identity != reference)):
                raise ValueError(f"B{block}/{arm}: input cohort mismatch")
            reference = identity
            rows[block, arm] = requests
    return rows


def main() -> None:
    loaded = {block: read_block(block) for block in (1, 2)}
    results = {block: loaded[block][0] for block in (1, 2)}
    rows = request_rows(results)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.1), sharey=True)
    for row_index, block in enumerate((1, 2)):
        for col_index, (field, title, unit) in enumerate(METRICS):
            ax = axes[row_index, col_index]
            maxima = {}
            for arm in ("off", "on"):
                values = np.array(sorted(float(r[field]) for r in rows[block, arm]))
                if not np.all(np.isfinite(values)) or np.any(values < 0):
                    raise ValueError(f"B{block}/{arm}: invalid {field}")
                maxima[arm] = float(values[-1])
                fractions = np.arange(1, 129) / 128
                ax.step(
                    np.r_[0.0, values], np.r_[0.0, fractions], where="post",
                    color=COLOR[arm], linestyle=STYLE[arm], linewidth=2.0,
                    label=f"Capacity victim {arm}",
                )
            ax.set_xlim(0, max(maxima.values()) * 1.045)
            ax.set_ylim(0, 1.025)
            ax.grid(alpha=0.23, linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            if field == "max_gap_s":
                for arm in ("off", "on"):
                    summary = results[block]["metrics"][arm]["max_gap_request_max_s"]
                    if not math.isclose(maxima[arm], summary, rel_tol=1e-12):
                        raise ValueError(f"B{block}/{arm}: max-gap tail mismatch")
                    ax.plot(maxima[arm], 1.0, "o", color=COLOR[arm], markersize=4.5)
                ax.text(
                    0.98, 0.84,
                    f"off max {maxima['off']:.3f} s\non max {maxima['on']:.3f} s",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
                )
            if row_index == 0:
                ax.set_title(title, fontsize=11)
            if row_index == 1:
                ax.set_xlabel(unit)
            if col_index == 0:
                order = "off → on" if block == 1 else "on → off"
                ax.set_ylabel(f"B{block} ({order})\nFraction of requests ≤ x")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Capacity-qualified victim: separate request-weighted ECDFs", fontsize=14, y=0.985)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.948),
               ncol=2, frameon=False)
    fig.text(
        0.5, 0.035,
        "Each request weighs 1/128 within its own arm and block. Seen H128; natural outputs may differ. "
        "No pooling, CI, equal-work, blind, or per-action causal claim.",
        ha="center", fontsize=9, color="#444444",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.13,
                        hspace=0.36, wspace=0.18)
    OUT.mkdir(exist_ok=False)
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"capacity_victim_two_blocks_ecdf.{suffix}", dpi=200)
    plt.close(fig)
    print(f"B1 source SHA-256: {loaded[1][1]}")
    print(f"B2 source SHA-256: {loaded[2][1]}")
    print(f"wrote: {OUT}")


if __name__ == "__main__":
    main()
