#!/usr/bin/env python3
"""Plot request-weighted ECDFs for the fixed fit-first exploratory pair."""

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
SOURCE = HERE / "A_FIT_FIRST_PAIR_RESULT_R01_20261001.json"
SOURCE_SHA256 = "37613561779a464d44de5ccfb06c5415892101712923d8448f98796b4d410d87"
OUT = HERE / "A_FIT_FIRST_FIGURE_R01_20261001"


def main() -> None:
    raw = SOURCE.read_bytes()
    observed_sha = hashlib.sha256(raw).hexdigest()
    if observed_sha != SOURCE_SHA256:
        raise ValueError(f"source SHA-256 mismatch: {observed_sha}")
    result = json.loads(raw)
    if result.get("status") != "FIT_FIRST_EXPLORATORY_PAIR_COMPLETE":
        raise ValueError("pair is not complete")

    arms: dict[str, dict[str, dict]] = {}
    for arm in ("off", "on"):
        metrics = result["metrics"][arm]
        requests = metrics["requests"]
        if metrics["status"] != "COMPLETE" or metrics["completed"] != 128:
            raise ValueError(f"{arm}: expected 128 complete requests")
        if len(requests) != 128 or metrics["undefined_gap_requests"] != 0:
            raise ValueError(f"{arm}: incomplete request-level metric coverage")
        by_id = {request["request_id"]: request for request in requests}
        if len(by_id) != 128 or not all(r["completed"] for r in requests):
            raise ValueError(f"{arm}: duplicate or unfinished requests")
        arms[arm] = by_id

    if set(arms["off"]) != set(arms["on"]):
        raise ValueError("off/on request IDs differ")
    for request_id in arms["off"]:
        off, on = arms["off"][request_id], arms["on"][request_id]
        for key in ("prompt_sha256", "arrival_s", "max_output"):
            if off[key] != on[key]:
                raise ValueError(f"{request_id}: source cohort mismatch at {key}")

    specs = (
        ("max_gap_s", "Largest inter-output gap per request (s)"),
        ("flow_s", "Arrival-to-completion time per request (s)"),
        ("ttft_s", "Time to first returned token per request (s)"),
    )
    colors = {"off": "#2368a1", "on": "#d8731d"}
    styles = {"off": "-", "on": "--"}
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), sharey=True)
    for ax, (key, xlabel) in zip(axes, specs):
        maxima = []
        for arm in ("off", "on"):
            values = np.array(sorted(float(r[key]) for r in arms[arm].values()))
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(f"{arm}: invalid {key}")
            maxima.append(float(values[-1]))
            y = np.arange(1, len(values) + 1) / len(values)
            ax.step(
                np.r_[0.0, values], np.r_[0.0, y], where="post",
                color=colors[arm], linestyle=styles[arm], linewidth=2.0,
                label=f"fit-first {arm} (n=128)",
            )
        x_max = max(maxima) * 1.045
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.025)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        if key == "max_gap_s":
            if not math.isclose(maxima[1], result["metrics"]["on"]["max_gap_request_max_s"], rel_tol=1e-12):
                raise ValueError("on max-gap tail disagrees with audit summary")
            ax.plot(maxima[1], 1.0, "o", color=colors["on"], markersize=5)
            ax.annotate(
                f"on tail {maxima[1]:.3f} s",
                xy=(maxima[1], 1.0), xytext=(0.52, 0.82),
                textcoords="axes fraction", fontsize=9,
                arrowprops={"arrowstyle": "->", "color": colors["on"], "lw": 0.9},
            )

    axes[0].set_ylabel("Fraction of completed requests")
    axes[0].legend(frameon=False, loc="lower right", fontsize=9)
    fig.suptitle("Fit-first exploratory pair: request-weighted empirical CDFs", fontsize=14)
    fig.text(
        0.5, 0.015,
        "Same 128 seen H128 inputs; one ordered pair. Natural output sequences may differ. "
        "No pooling, confidence interval, equal-work, or per-action causal claim.",
        ha="center", va="bottom", fontsize=9, color="#444444",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    OUT.mkdir(exist_ok=False)
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"fit_first_request_ecdf.{suffix}", dpi=200)
    plt.close(fig)
    print(f"source SHA-256: {observed_sha}")
    print(
        "off/on max gap: "
        f"{max(float(r['max_gap_s']) for r in arms['off'].values()):.6f}/"
        f"{max(float(r['max_gap_s']) for r in arms['on'].values()):.6f} s"
    )
    print(f"wrote: {OUT}")


if __name__ == "__main__":
    main()
