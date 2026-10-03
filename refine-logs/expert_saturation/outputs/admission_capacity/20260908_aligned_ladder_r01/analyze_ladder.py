#!/usr/bin/env python3
"""Retained 24-episode ladder analysis; no policy rerun or historical causal claim."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics as st
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
POLICY_PATH = BASE / "20260906_native_knee_r01/policy_probe/analyze_policy.py"
STEP_PATH = BASE / "20260908_step_cost_surface_r01/analyze_step_cost.py"
OLD = BASE / "20260906_native_knee_r01/policy_probe/gpu_results"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


policy = module("ladder_policy", POLICY_PATH)
step_cost = module("ladder_step_cost", STEP_PATH)
require = policy.require
INPUT_PATHS = set()


def read(path):
    INPUT_PATHS.add(str(Path(path).resolve()))
    return policy.read(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    return step_cost.quantiles(values) if values else {"n": 0, "p50": None}


def capture_evidence(root, group, maximum):
    path = root / f"{group}.console.log"
    INPUT_PATHS.add(str(path.resolve()))
    log = path.read_text()
    lists = [json.loads(x) for x in re.findall(r"cudagraph_capture_sizes['\"]?\s*:\s*(\[[\d, ]+\])", log)]
    counts = [int(x) for x in re.findall(r"Capturing CUDA graphs \(decode, FULL\):[^\r\n]*?\|\s*\d+/(\d+)", log)]
    require(bool(lists) and all(x == lists[0] for x in lists), "capture sizes absent or inconsistent")
    sizes = [x for x in lists[0] if x <= maximum]
    require(bool(counts) and set(counts) == {len(sizes)}, "decode capture count differs from sizes")
    return {"path": str(path), "logged_sizes": lists[0], "decode_sizes": sizes, "full_graph_counts": counts}


def alignment_summary(path, sizes):
    cell = step_cost.load_cell(path)
    result = {key: cell[key] for key in ("aligned", "n_steps", "n_receipts")}
    if not cell["aligned"]:
        return result
    pure = [r for r in cell["rows"] if r["prefill_tokens"] == 0 and r["decode_width"] > 0]
    widths = defaultdict(list)
    for row in pure:
        widths[row["decode_width"]].append(row["exec_ms"])
    buckets = [next(x for x in sizes if x >= r["decode_width"]) for r in pure]
    cost24 = [r["exec_ms"] for r in pure if 17 <= r["decode_width"] <= 24]
    result.update(
        n_pure_steps=len(pure), pure_step_ms=stats([r["exec_ms"] for r in pure]),
        pure_width_counts=dict(sorted(Counter(r["decode_width"] for r in pure).items())),
        pure_width_median=st.median([r["decode_width"] for r in pure]) if pure else None,
        pure_by_width_ms={w: stats(v) for w, v in sorted(widths.items())},
        bucket_step_counts=dict(sorted(Counter(buckets).items())),
        mean_padding_waste=st.mean(1 - r["decode_width"] / b for r, b in zip(pure, buckets)) if pure else None,
        capture_point_step_share=st.mean(r["decode_width"] in sizes for r in pure) if pure else None,
        actual_active_step_counts=dict(sorted(Counter(r["actual_active"] for r in cell["rows"]).items())),
        width17_24_pure_ms=stats(cost24),
        width17_24_historical_interval_diagnostic=(9.30 <= st.median(cost24) <= 9.51) if cost24 else None,
        interval_interpretation="Historical pooled-width interval only; neither a structural falsifier nor a new preregistered gate.")
    return result


def analyze_cell(directory, index, config, workload, sizes):
    plan = config["plans"][index]
    path = directory / f"cell-{index:03d}.json"
    out = dict(id=f"{directory.name}/cell-{index:03d}", path=str(path), group=directory.name,
               plan=plan, status="MISSING", qualified=False)
    if not path.exists():
        return out
    raw = read(path)
    out["raw_status"] = raw["status"]
    try:
        require(hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest() == config["workload_sha256"], "workload hash mismatch")
        saved = read(directory / f"metrics-{index:03d}.json")
        # summarize invokes the unchanged causal_check and preserves per-request metrics.
        out.update(policy.summarize(raw, config, workload, plan, saved), status=raw["status"])
        out["gpu_boundary_check"] = read(directory / f"checks-{index:03d}.json")
        out["alignment"] = alignment_summary(path, sizes)
        out["qualified"] = (raw["status"] == "COMPLETE" and out["metrics"]["n_completed"] == config["requests"]
                            and out["gpu_boundary_check"]["status"] == "PASS" and out["token_level_timing_resolved"]
                            and out["alignment"]["aligned"])
    except (KeyError, ValueError, TypeError, OSError, StopIteration) as exc:
        out.update(status="INVALID", error=str(exc), qualified=False)
    return out


def comparison(base, action):
    out = dict(base_id=base["id"], action_id=action["id"], status="MISSING_OR_UNQUALIFIED")
    if not (base["qualified"] and action["qualified"]):
        return out
    keys = ("goodput_rps", "throughput_rps", "attainment", "ttft_p50_s", "tpot_p50_s", "n_slo_pass")
    out.update(status="DESCRIPTIVE_SAME_ENGINE_RERUN", action_minus_base={k: action[k] - base[k] for k in keys},
               goodput_relative_change=action["goodput_rps"] / base["goodput_rps"] - 1 if base["goodput_rps"] else None)
    return out


def source_evidence(readback, environments):
    INPUT_PATHS.add(str((readback / "SOURCE_SHA256.txt").resolve()))
    manifest = {line.split()[1]: line.split()[0] for line in (readback / "SOURCE_SHA256.txt").read_text().splitlines() if line.strip()}
    files = []
    for name, expected in manifest.items():
        path = readback / name
        INPUT_PATHS.add(str(path.resolve()))
        actual = sha(path)
        observed = {g: e.get("source_sha256", {}).get(name) for g, e in environments.items()}
        files.append(dict(path=str(path), sha256=actual, manifest_sha256=expected, environment_sha256=observed,
                          matched=actual == expected and all(x == actual for x in observed.values())))
    fields = ("python", "torch", "cuda", "vllm", "transformers", "cpu_threads", "vllm_source_sha256", "execution_env")
    return dict(manifest_path=str(readback / "SOURCE_SHA256.txt"), files=files,
                all_execution_sources_match=all(x["matched"] for x in files),
                environment_fields_equal={k: environments["forward"].get(k) == environments["reverse"].get(k) for k in fields},
                environments=environments, limits="GPU state strings vary by sampling time; boundary checks do not certify continuous isolation.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=HERE / "readback/results")
    parser.add_argument("--output-dir", type=Path, default=HERE / "analysis")
    args = parser.parse_args()
    require(not args.output_dir.exists(), "output directory must be new; never overwrite analysis")
    configs, engines, envs, captures, cells = {}, {}, {}, {}, []
    for group in ("forward", "reverse"):
        directory = args.results_dir / group
        configs[group] = read(directory / "config.json")
        engines[group] = read(directory / "engine_args.json")
        envs[group] = read(directory / "environment.json")
        captures[group] = capture_evidence(args.results_dir, group, configs[group]["engine_max_num_seqs"])
        workload = read(directory / "workload.json")
        cells.extend(analyze_cell(directory, i, configs[group], workload, captures[group]["decode_sizes"])
                     for i in range(len(configs[group]["plans"])))
    paired, historical = [], []
    for group in ("forward", "reverse"):
        for regime in ("steady", "bursty"):
            rows = [c for c in cells if c["group"] == group and c["plan"]["regime"] == regime]
            feedback = next(c for c in rows if c["plan"]["policy"] == "feedback")
            statics = [c for c in rows if c["plan"]["policy"] == "static"]
            valid = [c for c in statics if c["qualified"]]
            best_value = max((c["goodput_rps"] for c in valid), default=None)
            best = [c for c in valid if c["goodput_rps"] == best_value]
            paired.append(dict(group=group, regime=regime, all_static_comparisons=[comparison(c, feedback) for c in statics],
                               best_static_ids=[c["id"] for c in best], best_static_goodput_rps=best_value,
                               feedback_goodput_rps=feedback.get("goodput_rps"),
                               feedback_beats_all_static=(feedback["goodput_rps"] > best_value)
                               if len(valid) == len(statics) and feedback["qualified"] else None))
            olddir = OLD / group
            oldconfig, oldworkload = read(olddir / "config.json"), read(olddir / "workload.json")
            index = next(i for i, p in enumerate(oldconfig["plans"]) if p["policy"] == "feedback" and p["regime"] == regime)
            old = analyze_cell(olddir, index, oldconfig, oldworkload, captures[group]["decode_sizes"])
            pair = comparison(old, feedback)
            pair.update(status="HISTORICAL_CROSS_CAMPAIGN_DIAGNOSTIC" if old["qualified"] and feedback["qualified"] else "UNQUALIFIED",
                        group=group, regime=regime, historical=old,
                        fair_config_equal={k: oldconfig.get(k) == configs[group].get(k) for k in policy.knee.FAIR_CONFIG},
                        engine_args_equal=read(olddir / "engine_args.json") == engines[group],
                        environment_path=str(olddir / "environment.json"),
                        old_environment=read(olddir / "environment.json"),
                        limitation="Different campaign times and engine processes; matching settings is not causal isolation of ladder.")
            if old["qualified"] and feedback["qualified"]:
                ow, nw = old["alignment"]["mean_padding_waste"], feedback["alignment"]["mean_padding_waste"]
                pair.update(old_padding_waste=ow, new_padding_waste=nw, waste_change=nw - ow,
                            F2_lower_padding_descriptive=nw < ow, F3_no_worse_tpot_descriptive=feedback["tpot_p50_s"] <= old["tpot_p50_s"])
            historical.append(pair)
    dependencies = [Path(__file__).resolve(), POLICY_PATH, Path(policy.knee.__file__),
                    Path(policy.knee.legacy.__file__), Path(sys.modules["metrics"].__file__), STEP_PATH]
    source = source_evidence(args.results_dir.parent, envs)
    complete = len(cells) == 24 and all(c["qualified"] for c in cells)
    result = dict(generated_utc=datetime.now(timezone.utc).isoformat(), status="MEASUREMENT_ONLY" if complete else "PARTIAL_OR_INVALID",
                  complete_expected_capture=complete, expected_episodes=24, qualified_episodes=sum(c["qualified"] for c in cells),
                  request_executions=sum(c.get("metrics", {}).get("n_completed", 0) for c in cells),
                  canonical="forward", repeat="reverse", cells=cells, same_engine_comparisons=paired,
                  historical_F2_F3_diagnostics=historical, capture_evidence=captures, source_evidence=source,
                  engine_args_equal=engines["forward"] == engines["reverse"], engine_args=engines, configs=configs,
                  input_paths=sorted(INPUT_PATHS),
                  analysis_dependencies=[dict(path=str(p), sha256=sha(p)) for p in dict.fromkeys(dependencies)],
                  executed_metrics_matches_analysis_metrics=sha(args.results_dir.parent / "metrics.py") == sha(Path(sys.modules["metrics"].__file__)),
                  limits=["Single model/GPU, 32 reused texts; 768 request executions are not 768 independent workloads.",
                          "F1 fixed historical milliseconds cannot falsify the staircase; actual width exposure is explicitly reported.",
                          "F2/F3 use historical campaign comparisons, not a contemporaneous old-ladder intervention arm.",
                          "No step-independent significance, causal Oracle, expert-signal increment, or production-capacity claim.",
                          "Capture padding is inferred from logged bucket sizes and actual pure-decode width, not measured HBM or executed CUDA kernels.",
                          "Best measured static is a descriptive hindsight reference, not an Oracle or held-out deployable selector.",
                          "Successful warmup has summaries but no retained full raw trajectories in inherited runner."])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "report.md").write_text(report(result))
    print(json.dumps({k: result[k] for k in ("status", "qualified_episodes", "request_executions", "same_engine_comparisons")}, indent=2))


def report(result):
    def fmt(x):
        return "—" if x is None else f"{x:.4f}"
    lines = ["# Retained capture-aligned ladder analysis", "", f"{result['status']}: {result['qualified_episodes']}/24 episodes; {result['request_executions']} request executions.", "", *[f"- {s}" for s in result["limits"]], "",
             "| Cell | Regime/arm | Status | Joint pass | Goodput | TTFT ms | TPOT ms | Active/decode/wait max | Pure width p50 | Padding waste | Width17–24 n / p50 ms |",
             "|---|---|---|---:|---:|---:|---:|---|---:|---:|---|"]
    for c in result["cells"]:
        if not c["qualified"]:
            lines.append(f"| {c['id']} | {c['plan']} | {c['status']} | — | — | — | — | — | — | — | — |")
            continue
        a, p = c["alignment"], c["plan"]
        lines.append(f"| {c['id']} | {p['regime']}/{p['policy']}{p['cap']} | {c['status']} | {c['n_slo_pass']} | {fmt(c['goodput_rps'])} | {fmt(c['ttft_p50_s']*1000)} | {fmt(c['tpot_p50_s']*1000)} | {c['max_active']}/{c['max_decode_requests']}/{c['max_native_waiting_after']} | {fmt(a['pure_width_median'])} | {fmt(a['mean_padding_waste'])} | {a['width17_24_pure_ms']['n']} / {fmt(a['width17_24_pure_ms']['p50'])} |")
    lines += ["", "## Within-engine feedback versus every measured static", ""]
    for p in result["same_engine_comparisons"]:
        lines.append(f"- {p['group']}/{p['regime']}: feedback {fmt(p['feedback_goodput_rps'])}, best static {p['best_static_ids']} = {fmt(p['best_static_goodput_rps'])}; feedback beats all static: {p['feedback_beats_all_static']}.")
        for pair in p["all_static_comparisons"]:
            change = pair.get("goodput_relative_change")
            lines.append(f"  - versus {pair['base_id']}: goodput change {fmt(change*100 if change is not None else None)}%.")
    lines += ["", "## Historical F2/F3 diagnostics only", "", "Comparisons below do not isolate ladder from campaign time or engine state.", ""]
    for p in result["historical_F2_F3_diagnostics"]:
        lines.append(f"- {p['group']}/{p['regime']}: waste {fmt(p.get('old_padding_waste'))} → {fmt(p.get('new_padding_waste'))}; F2 {p.get('F2_lower_padding_descriptive')}, F3 {p.get('F3_no_worse_tpot_descriptive')}; TPOT delta ms {fmt(p.get('action_minus_base',{}).get('tpot_p50_s',0)*1000)}.")
    lines += ["", "## Source and environment", "", f"All readback execution sources match manifest and both environments: {result['source_evidence']['all_execution_sources_match']}.",
              f"Engine arguments equal: {result['engine_args_equal']}; executed metrics equals analysis metrics: {result['executed_metrics_matches_analysis_metrics']}.",
              "Full per-request metrics, causal checks, width counts, source paths/hashes and historical environments are retained in analysis.json.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
