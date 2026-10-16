#!/usr/bin/env python3
"""F5: measured benefit versus batch across explicitly identified model/formats.

Example (from the project root; matplotlib required):
  python code/plot_cross_model.py \
    --bench olmoe=figures/results_r07/bench_points.csv \
    --output-dir figures/cross_model_r07

Repeat --bench MODEL_KEY=CSV for future BF16/FP8 tables from plot_results.py.
Model keys and E/k come from configs/models.json, never inferred from a curve.
Each run/repeat/condition has its own figure row: no pooling across runs,
formats, tuning, hardware, workloads or resource budgets. Native is matched
within that row and batch. --run-root ORIGINAL_RUN_ID=LOCAL_DIR repairs moved
CSV provenance paths; without accessible metadata, resource/tuning information
is explicitly unreported. --arms changes the displayed policies (default:
global_j4, gate_high, marginal_high); it does not select a best measured arm.

The range CSV lists observed batches above a descriptive threshold, without
interpolation, extrapolation or a significance claim. An absent model/format
has no curve. A single measured model is labelled ONE_MODEL_PARTIAL.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import textwrap


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def number(value):
    if value in (None, ""):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def mapping(value):
    key, separator, path = value.partition("=")
    if not separator or not key or not path:
        raise argparse.ArgumentTypeError("expected KEY=PATH")
    return key, Path(path)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Sources:
    def __init__(self):
        self.files = {}

    def add(self, path):
        path = Path(path).resolve()
        self.files[str(path)] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        return path

    def read(self, path):
        return json.loads(self.add(path).read_text())


def metadata(row, remaps, sources):
    root = remaps.get(row["run_id"], Path(row["run_id"]))
    config_path = root / "config.json"
    config = sources.read(config_path) if config_path.is_file() else {}
    args, kwargs = config.get("args", {}), config.get("model_kwargs", {})
    controller = {}
    for parent in (root.parent, root.parent.parent):
        path = parent / "status.json"
        if path.is_file():
            value = sources.read(path)
            if "command" in value:
                controller = value
                break
    tuning = [str(value).split("=", 1)[1] for value in controller.get("command", [])
              if str(value).startswith("VLLM_TUNED_CONFIG_FOLDER=")]
    return {"config_source": str(config_path) if config else "unreported",
            "gpu_memory_utilization": args.get("gpu_memory_utilization", "unreported"),
            "tensor_parallel_size": kwargs.get("tensor_parallel_size", "unreported"),
            "max_model_len": args.get("max_model_len", "unreported"),
            "vllm_version": config.get("vllm_version", "unreported"),
            "native_tuning_folder": tuning[0] if tuning else "unreported",
            "generation_semantics": config.get("generation_semantics", "unreported"),
            "source_code_hashes": config.get("source_sha256", {})}


def format_name(details):
    quant = str(details.get("quantization", "none")).lower()
    dtype = str(details.get("dtype", "unreported")).lower()
    if quant == "fp8":
        return "FP8 weights / " + dtype + " model dtype"
    if quant in ("none", "null", "") and dtype in ("bf16", "bfloat16"):
        return "BF16"
    return f"dtype={dtype}, quant={quant}"


def policy_label(arm, spec, top_k):
    if spec.get("mode") == "global_native":
        return f"{arm} (native k={top_k-int(spec['drop_j'])})"
    suffix = []
    for key, label in (("tail_rank", "r0"), ("gate_threshold", "tau"), ("mass_budget", "m")):
        if key in spec:
            suffix.append(f"{label}={spec[key]:g}")
    return arm + (" (" + ", ".join(suffix) + ")" if suffix else "")


def load(inputs, models, arms, remaps, sources):
    units = {}
    seen_inputs = set()
    for model_key, csv_path in inputs:
        if model_key not in models:
            raise ValueError(f"unknown model key {model_key}; use a key from --models-config")
        model = models[model_key]
        source = sources.add(csv_path)
        if source in seen_inputs:
            raise ValueError(f"duplicate input CSV: {source}")
        seen_inputs.add(source)
        with source.open(newline="", encoding="utf-8") as handle:
            for line, row in enumerate(csv.DictReader(handle), 2):
                if row["arm"] not in {*arms, "native"} or row.get("diagnostic", "").lower() == "true":
                    continue
                details = json.loads(row["condition_details"])
                known_paths = {key for key, value in models.items()
                               if details.get("model") in (value.get("path"), value.get("repo"))}
                if known_paths and model_key not in known_paths:
                    raise ValueError(f"CSV model path belongs to {sorted(known_paths)}, not declared {model_key}")
                condition = canonical(details)
                # Model identity is an explicit CLI declaration backed by the
                # registry. Preserve the recorded path separately; runtime
                # views and checkpoint relocation need not equal registry.path.
                identity = (model_key, row["run_id"], int(row["repeat"]), condition)
                if identity not in units:
                    meta = metadata(row, remaps, sources)
                    units[identity] = {"id": hashlib.sha256(canonical(identity).encode()).hexdigest()[:10],
                        "model_key": model_key, "model": model, "run_id": row["run_id"],
                        "repeat": int(row["repeat"]), "details": details, "metadata": meta, "rows": {}}
                unit = units[identity]
                point_key = (int(row["batch"]), row["arm"])
                if point_key in unit["rows"]:
                    raise ValueError(f"duplicate run/repeat/B/arm; not an independent run: {identity} {point_key}")
                row.update(csv_source=str(source), csv_record=line)
                unit["rows"][point_key] = row
    points = []
    for unit in units.values():
        model, details = unit["model"], unit["details"]
        specs = defaultdict(set)
        for (batch, arm), row in sorted(unit["rows"].items()):
            if arm == "native":
                continue
            native = unit["rows"].get((batch, "native"))
            if native is None:
                raise ValueError(f"missing same-run native baseline: {unit['run_id']} B={batch}")
            spec = json.loads(row["spec"] or "null") or {}
            specs[arm].add(canonical(spec))
            current_j, native_j = number(row["j_per_decode_token"]), number(native["j_per_decode_token"])
            duration, native_duration = number(row["energy_window_s"]), number(native["energy_window_s"])
            steady = all(value is not None and value >= 3 for value in (duration, native_duration))
            energy = 100 * (1 - current_j/native_j) if current_j is not None and native_j and native_j > 0 else None
            def pct(field):
                value = number(row[field])
                return None if value is None else value * 100
            points.append({"unit_id": unit["id"], "model_key": unit["model_key"],
                "model_repo": model["repo"], "model_revision": model.get("revision", "unreported"),
                "recorded_model_path": details.get("model"), "experts": model["experts"], "top_k": model["top_k"],
                "experts_per_selected_expert": model["experts"]/model["top_k"], "format": format_name(details),
                "run_id": unit["run_id"], "run_status": row["run_status"], "repeat": unit["repeat"],
                "batch": batch, "arm": arm, "policy_spec": canonical(spec),
                "tpot_reduction_pct": pct("tpot_saving_fraction"),
                "logical_footprint_reduction_pct": pct("logical_footprint_saving_fraction"),
                "j_per_token_reduction_pct": energy, "energy_steady_pair": steady,
                "energy_window_s": duration, "native_energy_window_s": native_duration,
                "tpot_ms": number(row["tpot_ms"]), "j_per_decode_token": current_j,
                "native_tpot_ms": number(native["tpot_ms"]), "native_j_per_decode_token": native_j,
                "condition_details": canonical(details), "run_metadata": canonical(unit["metadata"]),
                "csv_source": row["csv_source"], "csv_record": row["csv_record"],
                "native_csv_source": native["csv_source"], "native_csv_record": native["csv_record"],
                "logical_counter_source": row["logical_counter_source"], "energy_source": row["energy_source"]})
        for arm, values in specs.items():
            if len(values) != 1:
                raise ValueError(f"policy spec changes across B within one row: {unit['run_id']} {arm}")
    return units, points


METRICS = [("tpot_reduction_pct", "TPOT reduction vs native (%)"),
           ("logical_footprint_reduction_pct", "Logical-footprint reduction vs native (%)"),
           ("j_per_token_reduction_pct", "J / decode token reduction vs native (%)")]


def render(out, units, points, arms, status):
    if not points:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    shown = [unit for unit in units.values() if any(p["unit_id"] == unit["id"] for p in points)]
    shown.sort(key=lambda u: (u["model"]["experts"]/u["model"]["top_k"], u["model_key"], format_name(u["details"]), u["run_id"], u["repeat"]))
    fig, axes = plt.subplots(len(shown), 3, figsize=(15, 4.0*len(shown)+1.5), squeeze=False)
    for row_index, unit in enumerate(shown):
        model, details, meta = unit["model"], unit["details"], unit["metadata"]
        selected = [p for p in points if p["unit_id"] == unit["id"]]
        states = ','.join(sorted({p['run_status'] for p in selected}))
        titles = [f"{model['repo'].split('/')[-1]} | {format_name(details)}\nE/k={model['experts']}/{model['top_k']}={model['experts']/model['top_k']:g}; {Path(unit['run_id']).parent.name}, repeat {unit['repeat']} [{states}]",
            f"{details.get('execution', 'unreported')}; prefix {details.get('prompt_tokens')}, output {details.get('newtokens')}\nGPU fraction={meta['gpu_memory_utilization']}; TP={meta['tensor_parallel_size']}; maxlen={meta['max_model_len']}",
            f"Native tuning: {'/'.join(Path(meta['native_tuning_folder']).parts[-3:])}\nvLLM {meta['vllm_version']}; {details.get('hardware', 'unreported')}"]
        for column, (metric, ylabel) in enumerate(METRICS):
            ax = axes[row_index, column]
            usable_count = 0
            for index, arm in enumerate(arms):
                observed = sorted((p for p in selected if p['arm'] == arm), key=lambda p: p['batch'])
                if not observed:
                    continue
                usable = [p for p in observed if p[metric] is not None and
                          (metric != "j_per_token_reduction_pct" or p['energy_steady_pair'])]
                label = policy_label(arm, json.loads(observed[0]['policy_spec']), model['top_k'])
                color = f"C{index%10}"
                if usable:
                    # Preserve gaps: missing/short values are NaN, never zero.
                    bs = [p['batch'] for p in observed]
                    ys = [p[metric] if p in usable else float('nan') for p in observed]
                    ax.plot(bs, ys, 'o-', lw=1.6, ms=4, color=color, label=label)
                    usable_count += len(usable)
                short = [p for p in observed if metric == "j_per_token_reduction_pct" and p[metric] is not None and not p['energy_steady_pair']]
                if short:
                    ax.scatter([p['batch'] for p in short], [p[metric] for p in short], facecolors='none', edgecolors=color, label=label+' (short diagnostic)')
                if column == 0 and not usable and not short:
                    ax.plot([], [], color=color, label=label+' (unmeasured)')
            ax.axhline(0, color='.5', lw=.7)
            batches = sorted({p['batch'] for p in selected})
            ax.set_xscale('log', base=2)
            ax.set_xticks(batches, [str(b) for b in batches])
            ax.set(xlabel='Decode batch, B', ylabel=ylabel)
            ax.set_title('\n'.join(textwrap.fill(line, 68) for line in titles[column].splitlines()), fontsize=9)
            ax.grid(alpha=.2)
            if column == 0:
                ax.legend(frameon=False, fontsize=7, loc='best')
            if not usable_count:
                ax.text(.5, .5, 'No eligible measured points', transform=ax.transAxes, ha='center', color='.4')
    caption = ("Each row is one run/repeat/condition; formats, tuning, resource budgets and runs are not pooled. Points are measured batches, not independent token repetitions.\n"
        "E/k is descriptive: models also differ in architecture, routing, KV layout and workload behavior; this figure does not isolate E/k causally.\n"
        "Footprint compares realized policy trajectories with native, not within-route deletion or hardware DRAM traffic. Energy uses board-power integration; hollow points have a <3 s or unknown window in either paired arm.\n"
        "Zero is a descriptive baseline, not a significance or deployment threshold. Missing models/formats have no curve; no interpolation of a supported benefit region.")
    fig.suptitle('F5  Measured batch-dependent benefit | '+status, fontsize=13)
    fig.text(.025, .012, caption, va='bottom', fontsize=8, color='.35')
    fig.tight_layout(rect=(0, 1.0/(4.0*len(shown)+1.5), 1, .95))
    generated = []
    for extension in ('pdf', 'png', 'svg'):
        name = f'F5_cross_model.{extension}'
        fig.savefig(out/name, dpi=180)
        generated.append(name)
    plt.close(fig)
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bench', type=mapping, action='append', required=True, metavar='MODEL_KEY=CSV')
    parser.add_argument('--models-config', type=Path, default=Path(__file__).resolve().parents[1]/'configs/models.json')
    parser.add_argument('--arms', nargs='+', default=['global_j4', 'gate_high', 'marginal_high'])
    parser.add_argument('--run-root', type=mapping, action='append', default=[], metavar='ORIGINAL_RUN_ID=LOCAL_DIR')
    parser.add_argument('--threshold-pct', type=float, default=0, help='descriptive threshold for observed-batch lists; not a significance/SLO claim')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--replace-derived', action='store_true', help='replace only files listed in this tool\'s previous F5_manifest.json')
    args = parser.parse_args()
    if not math.isfinite(args.threshold_pct):
        parser.error('--threshold-pct must be finite')
    if len(set(args.arms)) != len(args.arms):
        parser.error('--arms contains duplicates')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        previous = args.output_dir/'F5_manifest.json'
        if not args.replace_derived or not previous.is_file():
            parser.error('choose an empty output directory or --replace-derived with an existing F5 manifest')
        for name in json.loads(previous.read_text()).get('generated_files', []):
            path = args.output_dir/name
            if path.parent.resolve() != args.output_dir.resolve() or not path.name.startswith('F5_'):
                raise ValueError('refusing to replace a file outside this tool\'s F5 outputs')
            if path.is_file():
                path.unlink()
    sources = Sources()
    models = sources.read(args.models_config)['models']
    sources.add(Path(__file__))
    units, points = load(args.bench, models, args.arms, dict(args.run_root), sources)
    measured_models = sorted({p['model_key'] for p in points})
    status = 'NO_PLOTTABLE_DATA' if not points else 'ONE_MODEL_PARTIAL' if len(measured_models) == 1 else 'MULTI_MODEL_MEASURED'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir/'F5_points.csv', points)
    ranges = []
    for unit in units.values():
        for arm in args.arms:
            for metric, _ in METRICS:
                selected = [p for p in points if p['unit_id'] == unit['id'] and p['arm'] == arm and p[metric] is not None
                            and (metric != 'j_per_token_reduction_pct' or p['energy_steady_pair'])]
                if selected:
                    ranges.append({'unit_id': unit['id'], 'model_key': unit['model_key'], 'format': format_name(unit['details']),
                        'run_id': unit['run_id'], 'repeat': unit['repeat'], 'arm': arm, 'metric': metric,
                        'threshold_pct': args.threshold_pct, 'observed_batches': canonical(sorted(p['batch'] for p in selected)),
                        'observed_batches_above_threshold': canonical(sorted(p['batch'] for p in selected if p[metric] > args.threshold_pct)),
                        'interpretation': 'descriptive observed points only; no significance, interpolation or deployment region'})
    write_csv(args.output_dir/'F5_observed_ranges.csv', ranges)
    figures = render(args.output_dir, units, points, args.arms, status)
    measured_formats = sorted({(p['model_key'], 'FP8' if json.loads(p['condition_details']).get('quantization') == 'fp8' else p['format']) for p in points})
    unmeasured_formats = sorted({(model, fmt) for model in models for fmt in ('BF16', 'FP8')}-set(measured_formats))
    info = {'schema_version': 1, 'status': status, 'measured_models': measured_models,
        'unmeasured_registry_models': sorted(set(models)-set(measured_models)), 'selected_arms': args.arms,
        'measured_model_formats': measured_formats, 'unmeasured_model_formats': unmeasured_formats,
        'points': len(points), 'threshold_pct': args.threshold_pct,
        'units': [{k:v for k,v in unit.items() if k != 'rows'} for unit in units.values()],
        'sources': list(sources.files.values()), 'generated_files': [*figures, *(['F5_points.csv'] if points else []), *(['F5_observed_ranges.csv'] if ranges else []), 'F5_manifest.json'],
        'identity_contract': 'Explicit CLI model key selects registry repo/revision/E/k; recorded checkpoint path is preserved separately. Registry identity is not a new checkpoint-hash verification.',
        'scope': 'Separate run/repeat/condition rows; no pooling or causal attribution to E/k; no confidence interval from tokens; descriptive benefit regions only.'}
    (args.output_dir/'F5_manifest.json').write_text(json.dumps(info, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:info[k] for k in ('status','measured_models','unmeasured_registry_models','points','generated_files')}, indent=2))


if __name__ == '__main__':
    main()
