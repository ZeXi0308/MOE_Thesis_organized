#!/usr/bin/env python3
"""T1 rank-by-gate whole-model quality table from actual quality_decode outputs.

Example:
  python code/plot_t1.py --quality runs/drop_rank_gate.json --output-dir figures/t1_r01

Each input summary is a run. Documents/tokens describe the tested population,
not independent system repetitions. Gate bins use selected-normalized fractions;
plot/table ranks are one-based. KL is full-vocabulary KL(native || policy).
Interventions repeat across all routed decode layers; cells are not additive
per-event/layer costs. Zero-event quality cells are NA, with raw metrics retained
in CSV. No valid measurements produces only a NO_MEASURED_DATA manifest.

Outputs: complete planned-bin/run CSV, paired-document CSV, measured group CSV,
run-mean summary CSV, LaTeX tables, PDF/PNG/SVG heatmaps, source SHA256 manifest.
Matplotlib is needed only when measured groups exist. No GPU dependencies.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import textwrap

from plot_results import Sources, artifact_path, canonical, counter_values, discover, read_json, slug


SCOPE = ("Whole-model consequence of repeatedly applying one rank/gate-bin policy across all routed decode layers; "
         "teacher-forced selected windows, full-vocabulary KL(native||policy); not additive event/layer costs or task accuracy.")


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"expected finite number, received {value!r}")
    return float(value)


def positive_int(value):
    value = finite(value)
    if value < 1 or value != int(value):
        raise ValueError("expected positive integer denominator")
    return int(value)


def exp(value):
    return math.exp(value) if value < 709 else None


def describe(values, prefix):
    return {prefix + suffix: fn(values) if values else None for suffix, fn in (
        ("_mean", statistics.mean), ("_median", statistics.median),
        ("_min", min), ("_max", max), ("_descriptive_sd", lambda v: statistics.stdev(v) if len(v) > 1 else None))}


def bin_fields(spec):
    rank0, rank1 = spec["tail_rank"], spec["rank_end"]
    low, high = finite(spec["gate_min"]), finite(spec["gate_threshold"])
    if not isinstance(rank0, int) or not isinstance(rank1, int) or not 0 <= rank0 < rank1 or not 0 <= low < high <= 1:
        raise ValueError("invalid explicit rank/gate interval")
    return {"rank_start_zero_based": rank0, "rank_end_exclusive": rank1,
            "rank_label_one_based": str(rank1) if rank1 == rank0 + 1 else f"{rank0+1}--{rank1}",
            "gate_min_inclusive": low, "gate_max_exclusive": high,
            "gate_label": f"[{low:g}, {high:g})"}


def read_tokens(record, manifest, sources):
    """Optional raw artifact; a referenced but absent file falls back to summaries."""
    if not record.get("token_file"):
        return None
    try:
        path = artifact_path(manifest, record["token_file"])
    except FileNotFoundError:
        sources.warn(f"Raw token artifact absent; using validated per-document summary: {manifest} {record['group']} {record['arm']}")
        return None
    sources.use(path)
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["prediction_phase"] != "decode":
            continue
        if row["group"] != record["group"] or row["arm"] != record["arm"] or row["output_position"] < 1:
            raise ValueError("token artifact group/arm/phase mismatch")
        key = (row["doc_key"], row["output_position"])
        if key in rows:
            raise ValueError("duplicate decode token metric")
        for field in ("nll", "baseline_nll", "kl_base_to_policy"):
            finite(row[field])
        rows[key] = row
    return rows


def paired_documents(native, candidate, manifest, sources):
    baseline, policy = native.get("per_document", {}), candidate.get("per_document", {})
    if not baseline or baseline.keys() != policy.keys():
        raise ValueError("native/candidate per-document populations differ or are missing")
    raw_native, raw_policy = read_tokens(native, manifest, sources), read_tokens(candidate, manifest, sources)
    for raw in (raw_native, raw_policy):
        if raw is not None and {doc for doc, _ in raw} != baseline.keys():
            raise ValueError("raw/summary document populations differ")
    if raw_native is not None and raw_policy is not None:
        if raw_native.keys() != raw_policy.keys():
            raise ValueError("native/candidate raw token populations differ")
        for key, row in raw_policy.items():
            other = raw_native[key]
            if row["target_token"] != other["target_token"] or not math.isclose(row["baseline_nll"], other["nll"], rel_tol=1e-7, abs_tol=1e-7):
                raise ValueError("raw token pairing or baseline NLL mismatch")
    output = []
    for doc in sorted(baseline):
        a, b = baseline[doc], policy[doc]
        tokens = positive_int(a["tokens"])
        if positive_int(b["tokens"]) != tokens:
            raise ValueError("document token denominator mismatch")
        kl, nll, base_nll = finite(b["mean_kl"]), finite(b["mean_nll"]), finite(a["mean_nll"])
        if raw_policy is not None:
            raw = [r for (d, _), r in raw_policy.items() if d == doc]
            if len(raw) != tokens:
                raise ValueError("raw/summary document token denominator mismatch")
            for actual, expected in ((statistics.mean(r["nll"] for r in raw), nll),
                                     (statistics.mean(r["baseline_nll"] for r in raw), base_nll),
                                     (statistics.mean(r["kl_base_to_policy"] for r in raw), kl)):
                if not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-7):
                    raise ValueError("raw/summary paired document metric mismatch")
        ppl, base_ppl = exp(nll), exp(base_nll)
        output.append({"doc_key": doc, "quality_tokens": tokens, "mean_kl_nats_per_token_raw": kl,
                       "policy_mean_nll": nll, "native_mean_nll": base_nll, "paired_mean_nll_delta": nll-base_nll,
                       "policy_ppl": ppl, "native_ppl": base_ppl,
                       "paired_document_ppl_delta_raw": ppl-base_ppl if ppl is not None and base_ppl is not None else None,
                       "pairing_validation": "raw_target_and_nll" if raw_native is not None and raw_policy is not None else "per_document_summary"})
    count = sum(r["quality_tokens"] for r in output)
    if count != positive_int(candidate["quality"]["tokens"]) or count != positive_int(native["quality"]["tokens"]):
        raise ValueError("group quality denominator differs from paired documents")
    return output


def load_counters(record, manifest, sources):
    inline = record.get("policy_counters")
    if record.get("policy_counter_file"):
        try:
            path = artifact_path(manifest, record["policy_counter_file"])
        except FileNotFoundError:
            path = None
        if path is not None:
            sources.use(path)
            counters = counter_values(read_json(path))
            if inline is not None:
                for field in ("tokens", "skipped_slots", "removed_gate_mass_sum", "slots_before"):
                    if not math.isclose(finite(counters[field]), finite(inline[field]), rel_tol=1e-8, abs_tol=1e-8):
                        raise ValueError("inline/artifact counter mismatch: " + field)
            return counters, str(path)
    if inline is not None:
        sources.warn(f"Worker artifact absent; using inline policy counters: {manifest} {record['group']} {record['arm']}")
        return inline, str(manifest) + f"#batches[{record['group']},{record['arm']}].policy_counters"
    raise ValueError("missing measured policy counters")


def load(paths, sources):
    groups, documents, bins = [], [], []
    for path in discover(paths, "quality", sources):
        sources.use(path)
        data = read_json(path)
        args = data.get("args", {})
        arms = {a["name"]: a["spec"] for a in data.get("arms", []) if (a.get("spec") or {}).get("mode") == "rank_gate"}
        for row in data.get("batches", []):
            if (row.get("spec") or {}).get("mode") == "rank_gate":
                if row["arm"] in arms and arms[row["arm"]] != row["spec"]:
                    raise ValueError("declared/executed arm spec differs: " + str(path))
                arms[row["arm"]] = row["spec"]
        if not arms:
            continue
        if data.get("status") != "PASS":
            sources.warn(f"Using only explicitly valid groups from non-PASS run ({data.get('status')}): {path}")
        condition = {k: args.get(k) for k in ("model", "dtype", "quantization", "batch", "prefix_tokens", "tokens", "articles")}
        condition.update(arms=arms, normalization=data.get("normalization"), splits=data.get("selected_splits"))
        key = canonical(condition)
        title = f"{Path(args.get('model','unspecified')).name} | {args.get('dtype','unspecified')} | quant={args.get('quantization') or 'none'} | B={args.get('batch','?')}"
        records = {}
        for row in data.get("batches", []):
            if row.get("valid") is True:
                pair = (row["group"], row["arm"])
                if pair in records:
                    raise ValueError("duplicate group/arm summary: " + str(path))
                records[pair] = row
        for arm, spec in arms.items():
            fields = {"condition": key, "condition_title": title, "run_id": str(path), "run_status": data.get("status"),
                      "arm": arm, "spec": canonical(spec), **bin_fields(spec)}
            measured, docrows = [], []
            for (group, name), record in sorted(records.items()):
                if name != arm:
                    continue
                try:
                    native = records[(group, "native")]
                    paired = paired_documents(native, record, path, sources)
                    counters, origin = load_counters(record, path, sources)
                    tokens = sum(r["quality_tokens"] for r in paired)
                    layer_tokens = positive_int(counters["tokens"])
                    skipped, before = finite(counters["skipped_slots"]), positive_int(counters["slots_before"])
                    removed = finite(counters["removed_gate_mass_sum"])
                    if not 0 <= skipped <= before or not 0 <= removed <= layer_tokens + 1e-6 or layer_tokens % tokens:
                        raise ValueError("invalid policy counter denominator or range")
                    event_state = "observed" if skipped else "no_selected_events"
                    nll = sum(r["policy_mean_nll"]*r["quality_tokens"] for r in paired)/tokens
                    base = sum(r["native_mean_nll"]*r["quality_tokens"] for r in paired)/tokens
                    kl = sum(r["mean_kl_nats_per_token_raw"]*r["quality_tokens"] for r in paired)/tokens
                    ppl, base_ppl = exp(nll), exp(base)
                    row = {**fields, "group": group, "event_state": event_state,
                           "quality_tokens": tokens, "documents": len(paired), "counter_token_layers": layer_tokens,
                           "routed_layer_count": layer_tokens//tokens, "slots_before": before, "skipped_slots": skipped,
                           "removed_gate_mass_sum": removed, "mean_removed_gate_mass_per_token_layer": removed/layer_tokens,
                           "removed_gate_mass_max": finite(counters["removed_gate_mass_max"]),
                           "tokens_with_skips_across_layers": finite(counters["tokens_with_skips"]),
                           "mean_kl_nats_per_token_raw": kl, "policy_mean_nll": nll, "native_mean_nll": base,
                           "ppl_delta_raw": ppl-base_ppl if ppl is not None and base_ppl is not None else None,
                           "counter_source": origin, "summary_source": str(path), "scope": SCOPE}
                    measured.append(row)
                    docrows.extend({**fields, "group": group, "group_event_state": event_state, **d,
                                    "event_scope": "skip counters are group/layer aggregates; document-specific skip occurrence is not recorded"} for d in paired)
                except (ValueError, KeyError, FileNotFoundError) as error:
                    sources.warn(f"Omitted T1 group {path} {group} {arm}: {error}")
            groups.extend(measured)
            documents.extend(docrows)
            if not measured:
                bins.append({**fields, "event_state": "unmeasured", "groups": 0, "scope": SCOPE})
                continue
            tokens = sum(r["quality_tokens"] for r in measured)
            layer_tokens = sum(r["counter_token_layers"] for r in measured)
            skipped = sum(r["skipped_slots"] for r in measured)
            removed = sum(r["removed_gate_mass_sum"] for r in measured)
            nll = sum(r["policy_mean_nll"]*r["quality_tokens"] for r in measured)/tokens
            base = sum(r["native_mean_nll"]*r["quality_tokens"] for r in measured)/tokens
            kl = sum(r["mean_kl_nats_per_token_raw"]*r["quality_tokens"] for r in measured)/tokens
            ppl, base_ppl = exp(nll), exp(base)
            delta = ppl-base_ppl if ppl is not None and base_ppl is not None else None
            bins.append({**fields, "event_state": "observed" if skipped else "no_selected_events",
                "groups": len(measured), "group_ids": canonical([r["group"] for r in measured]),
                "quality_tokens": tokens, "documents": len(docrows), "unique_documents": len({r["doc_key"] for r in docrows}),
                "document_population_sha256": hashlib.sha256(canonical(sorted((r["group"], r["doc_key"], r["quality_tokens"]) for r in docrows)).encode()).hexdigest(),
                "counter_token_layers": layer_tokens, "slots_before": sum(r["slots_before"] for r in measured),
                "skipped_slots": skipped, "removed_gate_mass_sum": removed,
                "mean_removed_gate_mass_per_token_layer": removed/layer_tokens,
                "removed_gate_mass_max": max(r["removed_gate_mass_max"] for r in measured),
                "tokens_with_skips_across_layers": sum(r["tokens_with_skips_across_layers"] for r in measured),
                "mean_kl_nats_per_token_raw": kl, "policy_mean_nll": nll, "native_mean_nll": base,
                "paired_mean_nll_delta": nll-base, "policy_ppl": ppl, "native_ppl": base_ppl, "ppl_delta_raw": delta,
                "mean_kl_nats_per_token": kl if skipped else None, "ppl_delta": delta if skipped else None,
                **describe([r["mean_kl_nats_per_token_raw"] for r in docrows], "document_mean_kl"),
                **describe([r["paired_mean_nll_delta"] for r in docrows], "document_paired_mean_nll_delta"),
                "scope": SCOPE})
    return groups, documents, bins


def summarize(bins):
    grouped = defaultdict(list)
    populations = defaultdict(set)
    for row in bins:
        grouped[(row["condition"], row["arm"])].append(row)
        if row["event_state"] != "unmeasured":
            populations[(row["condition"], row["run_id"])].add(row["document_population_sha256"])
    result = []
    for (_, _), rows in sorted(grouped.items()):
        good = [r for r in rows if r["event_state"] != "unmeasured"]
        first = rows[0]
        value = {k: first[k] for k in ("condition", "condition_title", "arm", "spec", *bin_fields(json.loads(first["spec"]))) }
        value.update(planned_runs=len(rows), measured_runs=len(good), runs_with_events=sum(r["event_state"] == "observed" for r in good),
                     run_ids=canonical([r["run_id"] for r in good]),
                     run_statuses=canonical(sorted({r.get("run_status") or "unknown" for r in good})),
                     coverage_varies_across_bins=any(len(populations[(r["condition"], r["run_id"])]) > 1 for r in good),
                     aggregation="arithmetic mean of run-level quality metrics over all measured runs; no token/document-based confidence intervals")
        if not good:
            value["event_state"] = "unmeasured"
        else:
            skipped = sum(r["skipped_slots"] for r in good)
            value.update(event_state="observed" if skipped else "no_selected_events",
                         quality_tokens=sum(r["quality_tokens"] for r in good), document_observations=sum(r["documents"] for r in good),
                         counter_token_layers=sum(r["counter_token_layers"] for r in good),
                         skipped_slots=skipped, removed_gate_mass_sum=sum(r["removed_gate_mass_sum"] for r in good),
                         mean_removed_gate_mass_per_token_layer=sum(r["removed_gate_mass_sum"] for r in good)/sum(r["counter_token_layers"] for r in good),
                         mean_kl_nats_per_token=statistics.mean(r["mean_kl_nats_per_token_raw"] for r in good) if skipped else None,
                         ppl_delta=statistics.mean(r["ppl_delta_raw"] for r in good) if skipped and all(r["ppl_delta_raw"] is not None for r in good) else None,
                         mean_kl_nats_per_token_run_values=canonical([r["mean_kl_nats_per_token_raw"] for r in good]),
                         ppl_delta_run_values=canonical([r["ppl_delta_raw"] for r in good]))
        result.append(value)
    return result


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, restval="NA")
        writer.writeheader()
        writer.writerows({k: "NA" if v is None else v for k, v in row.items()} for row in rows)


def latex_escape(value):
    return str(value).translate(str.maketrans({"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&",
                                              "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}"}))


def fmt(value):
    return "NA" if value is None else f"{value:.4g}"


def render(summary, output, generated, captions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    grouped = defaultdict(list)
    for row in summary:
        grouped[row["condition"]].append(row)
    for key, rows in grouped.items():
        if not any(r["measured_runs"] for r in rows):
            continue
        tag = slug(rows[0]["condition_title"]) + "_" + hashlib.sha256(key.encode()).hexdigest()[:8]
        stem = "T1_" + tag
        ordered = sorted(rows, key=lambda r: (r["rank_start_zero_based"], r["gate_min_inclusive"]))
        ranks = sorted({(r["rank_start_zero_based"], r["rank_end_exclusive"]) for r in rows})
        gates = sorted({(r["gate_min_inclusive"], r["gate_max_exclusive"]) for r in rows})
        cells = {}
        for row in rows:
            pair = ((row["rank_start_zero_based"], row["rank_end_exclusive"]), (row["gate_min_inclusive"], row["gate_max_exclusive"]))
            if pair in cells:
                raise ValueError("multiple arms occupy the same rank/gate cell; use separate configurations")
            cells[pair] = row
        fig, axes = plt.subplots(1, 2, figsize=(max(10, len(gates)*2.5), max(5.6, len(ranks)*.65+3.1)))
        for axis, field, title in zip(axes, ("mean_kl_nats_per_token", "ppl_delta"), ("Whole-model KL (nats / decode token)", "Pooled-window PPL difference")):
            matrix = np.full((len(ranks), len(gates)), np.nan)
            for i, rank in enumerate(ranks):
                for j, gate in enumerate(gates):
                    row = cells.get((rank, gate))
                    value = row.get(field) if row else None
                    if value is not None:
                        matrix[i, j] = value
            cmap = plt.get_cmap("viridis").copy()
            cmap.set_bad("#e7e7e7")
            valid = matrix[np.isfinite(matrix)]
            low, high = (float(valid.min()), float(valid.max())) if valid.size else (0, 1)
            if low == high:
                high = low + max(abs(low)*.1, 1e-6)
            image = axis.imshow(matrix, cmap=cmap, vmin=low, vmax=high, aspect="auto")
            for i, rank in enumerate(ranks):
                for j, gate in enumerate(gates):
                    row = cells.get((rank, gate))
                    if row is None or row["event_state"] == "unmeasured":
                        label = "NA\nunmeasured"
                    elif row["event_state"] == "no_selected_events":
                        label = "NA\n0 skips"
                    else:
                        label = fmt(row.get(field))
                    axis.text(j, i, label, ha="center", va="center", fontsize=9,
                              color="white" if math.isfinite(matrix[i,j]) and (matrix[i,j]-low)/(high-low)<.55 else "black")
            axis.set_xticks(range(len(gates)), [f"[{a:g}, {b:g})" for a,b in gates], rotation=25, ha="right")
            axis.set_yticks(range(len(ranks)), [str(b) if b==a+1 else f"{a+1}–{b}" for a,b in ranks])
            axis.set(xlabel="Selected-normalized gate fraction", ylabel="Selected-weight rank (one-based)", title=title)
            if valid.size:
                fig.colorbar(image, ax=axis, shrink=.8)
        caption = ("Each cell repeatedly drops one bin across all routed decode layers; not an additive event/layer cost. "
                   "Quality is averaged across measured runs; raw run and paired-document values are in CSV. "
                   "NA with 0 skips means no selected events, not evidence of safe skipping. Missing cells remain unmeasured. "
                   "No confidence intervals; tokens and documents are not system-run repetitions.")
        if any(r["coverage_varies_across_bins"] for r in rows):
            caption += " Measured document coverage differs across bins; inspect per-bin denominators before comparing."
        if any(any(s != "PASS" for s in json.loads(r["run_statuses"])) for r in rows):
            caption += " Includes valid groups from incomplete runs; see source manifest."
        fig.suptitle("T1 | " + rows[0]["condition_title"], fontsize=12)
        fig.text(.03, .02, "\n".join(textwrap.wrap(caption, 145)), fontsize=8, va="bottom")
        fig.tight_layout(rect=(0,.19,1,.94))
        for extension in ("pdf", "png", "svg"):
            name = stem + "." + extension
            fig.savefig(output/name, dpi=180, bbox_inches="tight")
            generated.append(name)
        plt.close(fig)
        captions[stem] = caption
        lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
                 r"\begin{tabular}{llrrrrrrrr}", r"\hline",
                 r"Rank & Gate bin & KL & $\Delta$PPL & Skipped slots & $\bar m$ & Docs & Tokens & Token-layers & Runs \\", r"\hline"]
        for row in ordered:
            values = [row["rank_label_one_based"], row["gate_label"], fmt(row.get("mean_kl_nats_per_token")),
                      fmt(row.get("ppl_delta")), fmt(row.get("skipped_slots")), fmt(row.get("mean_removed_gate_mass_per_token_layer")),
                      fmt(row.get("document_observations")), fmt(row.get("quality_tokens")),
                      fmt(row.get("counter_token_layers")), str(row["measured_runs"])]
            lines.append(" & ".join(latex_escape(v) for v in values) + r" \\")
        tex_caption = latex_escape(rows[0]["condition_title"] + ". " + caption +
                      " KL is full-vocabulary KL(native||policy). PPL is exponentiated pooled decode-window NLL. "
                      "Mean removed gate mass uses token-layers as denominator; Docs and Tokens count observations across runs.")
        tex_caption = tex_caption.replace("KL(native||policy)", r"$\mathrm{KL}(\mathrm{native}\Vert\mathrm{policy})$")
        lines.extend([r"\hline", r"\end{tabular}", r"\caption{" + tex_caption + "}",
                      r"\label{tab:" + stem + "}", r"\end{table*}"])
        name = stem + ".tex"
        (output/name).write_text("\n".join(lines)+"\n")
        generated.append(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("choose a new output directory; existing research results are not overwritten")
    sources = Sources()
    sources.use(Path(__file__))
    sources.use(Path(__file__).with_name("plot_results.py"))
    groups, documents, bins = load(args.quality, sources)
    summary = summarize(bins)
    args.output_dir.mkdir(parents=True)
    generated, captions = [], {}
    if groups:
        for name, rows in (("T1_groups.csv", groups), ("T1_documents.csv", documents), ("T1_bins_by_run.csv", bins), ("T1_summary.csv", summary)):
            write_csv(args.output_dir/name, rows)
            generated.append(name)
        render(summary, args.output_dir, generated, captions)
    manifest = {"status": "COMPLETE" if groups else "NO_MEASURED_DATA", "scope": SCOPE,
                "counts": {"measured_groups": len(groups), "paired_document_observations": len(documents),
                           "planned_run_bins": len(bins), "summary_bins": len(summary)},
                "sources": list(sources.paths.values()), "warnings": sources.warnings,
                "generated_files": generated, "captions": captions,
                "zero_event_semantics": "display KL/PPL are NA when skipped_slots=0; original quality values remain in raw CSV columns",
                "counter_denominators": "removed_gate_mass_sum/counter_token_layers; KL and NLL use scored model-output decode tokens",
                "statistics": "raw run metrics and run arithmetic means; paired document descriptors are not system repetitions; no CIs"}
    (args.output_dir/"T1_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps({"status": manifest["status"], "output_dir": str(args.output_dir), "counts": manifest["counts"]}))


if __name__ == "__main__":
    main()
