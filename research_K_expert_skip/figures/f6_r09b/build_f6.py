"""Compact paper rendering of completed r09b controller timings; no GPU use.

The unchanged plot_results.py produced the accompanying CSV/source manifest.
This local rendering retains its figure stem and shows CPU/device panels;
eager wall measurements remain in microbench_run_points.csv and the paper text.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "runs/olmoe_fp8_matrix_r09b/microbench_policy.json"
STEM = "F6_E_64_k_8_NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition_60a476f2"
data = json.loads(SOURCE.read_text())
assert data["status"] == "PASS" and len(data["rows"]) == 12
assert all(r["experts"] == 64 and r["top_k"] == 8 for r in data["rows"])
assert all(r["eager_thread_cpu_clock"] == "time.thread_time_ns" for r in data["rows"])

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.30))
styles = [("torch", "Unfused PyTorch", "#2878b5", "o"),
          ("triton_multi", "Multi-kernel Triton", "#36954d", "s"),
          ("triton", "Fused Triton", "#ed7d22", "^")]
metrics = [("eager_thread_cpu_us_per_call", "Eager main-thread CPU"),
           ("captured_device_us_per_call", "Captured GPU elapsed time")]
for ax, (field, title) in zip(axes, metrics):
    for backend, label, color, marker in styles:
        rows = sorted((r for r in data["rows"] if r["backend"] == backend), key=lambda r: r["batch"])
        assert [r["batch"] for r in rows] == [1, 8, 32, 64]
        ax.plot([r["batch"] for r in rows], [r[field] for r in rows],
                label=label, color=color, marker=marker, markersize=4, linewidth=1.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 8, 32, 64])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_yscale("log")
    ax.set(xlabel="Routing rows, B", ylabel="Time per call (microseconds)", title=title)
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .91),
           ncol=3, fontsize=8, frameon=False)
fig.suptitle("F6: synthetic post-routing policy + counters (E=64, k=8)", y=.995, fontsize=11)
fig.subplots_adjust(left=.115, right=.99, top=.71, bottom=.26, wspace=.42)
fig.text(.5, .065, "One source execution; separate CPU/device clock boundaries, not a subtraction.\n"
         "CPU excludes final explicit CUDA synchronization, off-CPU waiting, and other threads.",
         ha="center", va="center", fontsize=7.5)
for extension in ("pdf", "png", "svg"):
    fig.savefig(HERE / f"{STEM}.{extension}", dpi=220)
plt.close(fig)

manifest_path = HERE / "results_manifest.json"
manifest = json.loads(manifest_path.read_text())
manifest["captions"][STEM] = (
    "Compact CPU/device panels from completed r09b synthetic policy + counters. "
    "Main-thread CPU uses time.thread_time_ns before final CUDA synchronization; "
    "captured-device time uses CUDA events. No wall-minus-device CPU estimate. "
    "Three implementations; one source execution, no independent-run intervals. "
    "All eager wall measurements remain in microbench_run_points.csv.")
manifest["local_rendering"] = {"script": "build_f6.py", "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "panels": [m[0] for m in metrics], "all_measured_fields_retained_in_csv": True}
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
