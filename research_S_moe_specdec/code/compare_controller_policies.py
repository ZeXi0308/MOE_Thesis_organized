#!/usr/bin/env python3
"""Compare executed, frozen controller policies on held-out request populations.

python code/compare_controller_policies.py --ar runs/EVAL_AR \
  --static runs/EVAL_STATIC --native-unrestricted runs/EVAL_TABLE \
  --native-threshold runs/EVAL_THRESHOLD --adaptive runs/EVAL_ADAPTIVE \
  --selection runs/SELECTION/regime_ID/selection.json \
  --calibration runs/FULLBUDGET/calibration/calibration.json \
  --output runs/controller_comparison --figures figures/controller_comparison

Missing directories/arms produce pending coverage, never synthetic results.
Each input must contain original perf JSON plus configuration/engine metadata.
Default workload slice is records[64:128], not source IDs 64..127. Fixed output
length is 128. The exact source IDs, workload hash, frozen tables, calibration
artifact and execution domain are checked before comparison. Selection scores
and post-hoc oracle curves are never treated as executed policy observations.

CPU-only, standard library; optional local PGFPlots rendering. --self-test
checks sample identity, executable policy identity and missing-counter handling.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile
import unittest

from plot_results import REGIME_FIELDS, _context, _finite, _requests, load_runs, write_csv


PROJECT = Path(__file__).resolve().parents[1]
ROLES = ("ar", "static", "native_unrestricted", "native_threshold", "adaptive")
LABELS = {"ar": "AR", "static": "Selected fixed gamma", "native_unrestricted": "Frozen native table",
          "native_threshold": "Frozen native threshold", "adaptive": "Adaptive"}
METRICS = ("effective_tpot_ms", "population_wall_s", "population_tokens_per_second",
           "wall_s", "tokens_per_second", "energy_j", "joules_per_token",
           "population_counter_energy_j", "population_counter_joules_per_token", "population_counter_elapsed_s")
CONTROL_METRICS = ("policy_choose_cpu_us_mean", "wrapper_cpu_us_mean", "acceptance_update_cpu_us_mean",
                   "schedule_instrumentation_cpu_us_mean", "wrapper_host_fraction", "schedule_host_fraction",
                   "discarded_available_token_fraction", "truncated_drafts_per_output_token")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path, default=None):
    return json.loads(Path(path).read_text()) if path and Path(path).is_file() else default


def output_csv(path, rows):
    write_csv(path, [{k:canonical(v) if isinstance(v, (list, dict)) else v for k,v in row.items()} for row in rows])


def locate_workload(plan, override=None):
    candidates = [Path(override)] if override else []
    name = plan.get("workload")
    if name:
        candidates.extend((Path(name), PROJECT / name, PROJECT / "configs" / Path(name).name))
    return next((p.resolve() for p in candidates if p.is_file()), None)


def sample_identity(record, config, workload_path, offset, count, output_length):
    """Check ordered workload records, preserving actual source IDs in evidence."""
    if workload_path is None:
        raise ValueError("workload unavailable: cannot verify the held-out request population")
    if sha(workload_path) != config.get("workload_sha256"):
        raise ValueError("workload SHA differs from runner configuration")
    records = load_json(workload_path)["records"][offset:offset+count]
    if len(records) != count:
        raise ValueError("expected workload slice is incomplete")
    expected = [r["source_index"] for r in records]
    requests = _requests(record)
    actual = [r.get("source_index") for r in requests]
    if len(requests) != count or len(set(actual)) != count or sorted(actual) != sorted(expected):
        raise ValueError("actual request IDs do not equal the prescribed held-out workload slice")
    if any(not r.get("finished") for r in requests):
        raise ValueError("request population includes unfinished requests")
    lengths = [len(r.get("output_token_ids", [])) for r in requests]
    if any(n != output_length for n in lengths):
        raise ValueError("fixed-output evaluation has a different output length or natural termination")
    if record.get("output_tokens") != sum(lengths):
        raise ValueError("reported output token total disagrees with complete request records")
    plan = config.get("plan", {})
    if int(plan.get("input_offset", 0)) != offset:
        raise ValueError("configured input_offset differs from evaluation protocol")
    shape_offset = int(plan.get("shape_warmup_offset", plan.get("input_offset", 0)))
    if (plan.get("warmup_each_batch") or plan.get("warmup_full_population")) and shape_offset != 0:
        raise ValueError("shape warmup is not restricted to development offset 0")
    warmup_offset = int(plan.get("warmup_offset", 80))
    if warmup_offset != 0:
        raise ValueError("initial warmup is not restricted to development offset 0")
    tpots = [_finite(r.get("effective_tpot_ms")) for r in requests]
    if any(t is None or t < 0 for t in tpots):
        raise ValueError("effective TPOT missing from one or more measured requests")
    return {"sample_source_indices": sorted(actual),
            "sample_indices_sha256": hashlib.sha256(canonical(sorted(actual)).encode()).hexdigest(),
            "request_count": count, "output_tokens_per_request": output_length,
            "effective_tpot_ms": statistics.mean(tpots), "input_offset": offset,
            "shape_warmup_offset": shape_offset, "warmup_offset": warmup_offset,
            "workload_file": str(workload_path), "all_finished": True,
            "finish_reason_counts":dict(Counter(str(r.get("finish_reason","not_recorded")) for r in requests)),
            "output_sequences_sha256":hashlib.sha256(canonical(sorted((r["source_index"],r["output_token_ids"]) for r in requests)).encode()).hexdigest()}


def validate_policy(role, row, engine, selection):
    mode = row.get("mode")
    gamma, budget = int(row["gamma"]), int(row["proposal_budget"])
    spec = engine.get("speculative_config") or {}
    table = spec.get("num_speculative_tokens_per_batch_size")
    native_budget = int(spec.get("num_speculative_tokens", 0))
    if native_budget != budget:
        raise ValueError("engine speculative maximum differs from recorded proposal budget")
    if role == "ar":
        if mode != "native" or gamma != 0 or budget != 0 or table is not None:
            raise ValueError("AR must execute native gamma=budget=0 without a table")
        return
    if role == "adaptive":
        if mode != "adaptive" or budget != 7 or table is not None:
            raise ValueError("adaptive must execute plugin adaptive with full proposal budget 7 and no native table")
        return
    if not selection or selection.get("status") != "selected_pending_evaluation":
        raise ValueError("frozen baseline selection is unavailable or incomplete")
    if mode != "native":
        raise ValueError("primary frozen baselines must use native execution")
    if role == "static":
        chosen = int(selection["static"]["selected_gamma"])
        if gamma != chosen or budget != chosen or table is not None:
            raise ValueError("executed fixed gamma does not match frozen direct-draft selection")
    else:
        expected = selection[role]
        if table != expected["table"] or gamma != int(expected["runner"]["gamma"]) or budget != int(expected["runner"]["proposal_budget"]):
            raise ValueError("executed native table/budget does not match its frozen selected role")


def selection_checks(selection, row, identity, source_hash):
    if not selection:
        return "selection_unavailable"
    for key in REGIME_FIELDS:
        if key in selection.get("regime", {}) and selection["regime"][key] != row[key]:
            raise ValueError(f"selection/evaluation regime mismatch: {key}")
    evaluation_ids = set(identity["sample_source_indices"])
    provenance = selection.get("source_provenance", [])
    if not provenance:
        raise ValueError("selection lacks sample/source provenance")
    for p in provenance:
        if p.get("sha256") == source_hash or str(Path(p["source"]).resolve()) == row["source"]:
            raise ValueError("selection source overlaps evaluation file")
        ids = p.get("sample_source_indices")
        if not ids:
            raise ValueError("selection sample identity is unavailable")
        if evaluation_ids & set(ids):
            raise ValueError("selection prompts overlap held-out evaluation")
    return "source_files_and_actual_prompt_ids_disjoint"


def population_counter_energy(record):
    """Span raw cumulative counters across waves, without integrating gaps.

    All original endpoint reads must be valid, ordered and on one NVML UUID.
    The resulting interval uses counter-read midpoints, not runner timestamps.
    """
    result={"population_counter_energy_j":None,"population_counter_joules_per_token":None,
            "population_counter_elapsed_s":None,"population_counter_status":"unavailable",
            "population_counter_boundary":"first-wave start to last-wave end cumulative NVML counter-read midpoints; includes inter-wave gaps and endpoint meter overhead; whole device, not exactly the runner population wall interval"}
    reads=[];uuids=[]
    for episode in record.get("episodes",[record]):
        energy=episode.get("energy") or {}
        uuid=(energy.get("device") or {}).get("uuid")
        if not uuid or energy.get("counter_valid") is not True:
            result["population_counter_status"]="missing_uuid_or_invalid_wave_counter";return result
        uuids.append(uuid)
        pair=[]
        for side in ("counter_start","counter_end"):
            read=energy.get(side) or {}
            required=("energy_mj","read_start_ns","monotonic_ns","read_end_ns")
            if "error" in read or any(type(read.get(k)) is not int or read[k]<0 for k in required):
                result["population_counter_status"]="missing_or_invalid_counter_read";return result
            if not read["read_start_ns"]<=read["monotonic_ns"]<=read["read_end_ns"]:
                result["population_counter_status"]="invalid_counter_read_bracket";return result
            pair.append(read)
        if pair[1]["energy_mj"]<=pair[0]["energy_mj"]:
            result["population_counter_status"]="nonadvancing_or_reset_wave_counter";return result
        reads.extend(pair)
    if not reads or len(set(uuids))!=1:
        result["population_counter_status"]="missing_reads_or_multiple_device_uuids";return result
    for left,right in zip(reads,reads[1:]):
        if right["read_start_ns"]<left["read_end_ns"] or right["monotonic_ns"]<=left["monotonic_ns"]:
            result["population_counter_status"]="counter_read_times_not_monotone";return result
        if right["energy_mj"]<left["energy_mj"]:
            result["population_counter_status"]="counter_decreased_or_reset_between_waves";return result
    first,last=reads[0],reads[-1]
    joules=(last["energy_mj"]-first["energy_mj"])/1000
    tokens=record.get("output_tokens")
    result.update(population_counter_energy_j=joules,
                  population_counter_joules_per_token=joules/tokens if tokens else None,
                  population_counter_elapsed_s=(last["monotonic_ns"]-first["monotonic_ns"])/1e9,
                  population_counter_status="validated_cumulative_counter_span",
                  population_counter_uuid=uuids[0],population_counter_wave_count=len(uuids),
                  population_counter_read_count=len(reads),population_counter_first_read=first,
                  population_counter_last_read=last,
                  population_counter_endpoint_bracket_width_s=((first["read_end_ns"]-first["read_start_ns"])+(last["read_end_ns"]-last["read_start_ns"]))/1e9)
    return result


def calibration_checks(path, row, record, config):
    if path is None or not Path(path).is_file():
        raise ValueError("adaptive measured full-budget calibration is unavailable")
    calibration = load_json(path)
    metadata = calibration.get("metadata", {})
    for key in REGIME_FIELDS:
        if key not in metadata or str(metadata[key]) != str(row[key]):
            raise ValueError(f"calibration/evaluation regime mismatch or missing field: {key}")
    records = calibration.get("records", [])
    cells = {(int(r["batch"]),int(r["gamma"]),r.get("proposal_budget")) for r in records}
    required = {(b,g,7) for b in (1,4,16,64) for g in range(8)}
    if not required <= cells:
        raise ValueError("adaptive calibration lacks a full B/gamma grid at proposal budget 7")
    for cell in records:
        if cell.get("evidence_kind") != "measured_complete_step_cost" or _finite(cell.get("total_step_ms")) is None:
            raise ValueError("calibration includes unmeasured/non-complete-step cost")
    source = Path(row["source"])
    actual = source.parent.parent / "calibration.json"
    if not actual.is_file():
        configured = config.get("args", {}).get("calibration")
        actual = Path(configured) if configured else actual
    if not actual.is_file() or sha(actual) != sha(path):
        raise ValueError("executed adaptive calibration artifact cannot be verified against the supplied table")
    evaluation_ids = {r["source_index"] for r in _requests(record)}
    verified_sources = []
    for name in sorted({r.get("source") for r in records if r.get("source")}):
        candidate = Path(name)
        # Source paths written remotely may be relocated with the whole project.
        if not candidate.is_file() and "/runs/" in name:
            candidate = PROJECT / "runs" / name.split("/runs/",1)[1]
        if not candidate.is_file():
            raise ValueError(f"calibration source unavailable for prompt-disjointness check: {name}")
        obj = load_json(candidate)
        ids = {r["source_index"] for r in _requests(obj)}
        if not ids or evaluation_ids & ids:
            raise ValueError("calibration source has missing or overlapping actual prompt IDs")
        verified_sources.append({"path":str(candidate.resolve()),"sha256":sha(candidate)})
    if not verified_sources:
        raise ValueError("calibration has no verifiable measured source files")
    return {"calibration_sha256":sha(path),"calibration_actual_artifact":str(actual.resolve()),
            "calibration_source_files":verified_sources}


def controls(record):
    grouped = defaultdict(list)
    snapshots = []
    for episode in record.get("episodes", [record]):
        hosts = episode.get("host_steps", [])
        plugin = episode.get("plugin", {})
        steps = plugin.get("steps", [])
        if len(hosts) != len(steps):
            raise ValueError("host/plugin step count mismatch")
        snapshots.append({"episode":episode.get("label"),
            "initial":episode.get("acceptance_snapshot_initial", episode.get("acceptance_snapshot_before", plugin.get("acceptance_snapshot_initial"))),
            "final":plugin.get("acceptance_snapshot")})
        for index, (host, step) in enumerate(zip(hosts, steps)):
            if host.get("index") != index or step.get("step") != index:
                raise ValueError("host/plugin step index mismatch")
            if step.get("episode") not in (None,episode.get("label")):
                raise ValueError("host/plugin episode mismatch")
            if int(step.get("scheduled_tokens",0)) > 0:
                grouped[step.get("phase","unknown")].append((host,step))
    output = []
    for phase,pairs in sorted(grouped.items()):
        row = {"phase":phase,"steps":len(pairs),"host_step_ms_total":sum(h["host_step_ms"] for h,_ in pairs)}
        fields = {"policy_choose_cpu_us":lambda s:_finite((s.get("decision") or {}).get("decision_cpu_us")),
                  "wrapper_cpu_us":lambda s:_finite(s.get("control_ns")),
                  "acceptance_update_cpu_us":lambda s:_finite(s.get("acceptance_update_ns")),
                  "schedule_instrumentation_cpu_us":lambda s:_finite(s.get("schedule_instrumentation_ns"))}
        for name,getter in fields.items():
            values = [v/(1 if name=="policy_choose_cpu_us" else 1000) for _,s in pairs if (v:=getter(s)) is not None]
            row[name+"_observed_steps"] = len(values)
            row[name+"_mean"] = statistics.mean(values) if values else None
            row[name+"_sum"] = sum(values) if len(values)==len(pairs) else None
            row[name+"_p95"] = sorted(values)[min(len(values)-1,math.ceil(.95*len(values))-1)] if values else None
        for key in ("drafts_available_before","truncated_by_controller","scheduled_drafts"):
            values=[v for _,s in pairs if (v:=_finite(s.get(key))) is not None]
            row[key+"_observed_steps"]=len(values)
            row[key]=sum(values) if len(values)==len(pairs) else None
        available,trimmed=row["drafts_available_before"],row["truncated_by_controller"]
        if available is not None and trimmed is not None and not 0<=trimmed<=available:
            raise ValueError("controller truncation count is outside available draft count")
        row["discarded_available_token_fraction"]=trimmed/available if available and trimmed is not None else None
        tokens=record.get("output_tokens")
        row["truncated_drafts_per_output_token"]=trimmed/tokens if trimmed is not None and tokens else None
        for name,prefix in (("wrapper_host_fraction","wrapper_cpu_us"),("schedule_host_fraction","schedule_instrumentation_cpu_us")):
            row[name]=row[prefix+"_sum"]/(row["host_step_ms_total"]*1000) if row[prefix+"_sum"] is not None and row["host_step_ms_total"] else None
        row["selected_gamma_histogram"]=dict(Counter(str(s.get("gamma")) for _,s in pairs))
        row["actual_decode_batch_histogram"]=dict(Counter(str(s.get("decode_batch")) for _,s in pairs))
        row["decision_source_histogram"]=dict(Counter((s.get("decision") or {}).get("source","not_reported") for _,s in pairs))
        row["fallback_histogram"]=dict(Counter((s.get("decision") or {}).get("fallback") for _,s in pairs if (s.get("decision") or {}).get("fallback")))
        row["terminal_steps"]=sum(bool(s.get("ewma_censored_requests")) for _,s in pairs)
        row["boundary"]="all scheduled steps in this phase, including terminal/drain; wrapper is nested in schedule instrumentation; trim counts are not wasted GPU time/energy"
        output.append(row)
    return output,snapshots


def summarize(rows, metrics, extra=()):
    grouped=defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in (*REGIME_FIELDS,"sample_indices_sha256","role","batch",*extra))].append(row)
    output=[]
    for key,values in sorted(grouped.items()):
        result=dict(zip((*REGIME_FIELDS,"sample_indices_sha256","role","batch",*extra),key))
        result.update(n_run_records=len(values),sources=[r["source"] for r in values],repeats=[r["repeat"] for r in values])
        for metric in metrics:
            raw=[r.get(metric) for r in values]
            present=[v for v in raw if _finite(v) is not None]
            result[metric+"_raw"]=raw
            result[metric+"_observed_runs"]=len(present)
            result[metric+"_median"]=statistics.median(present) if present else None
            result[metric+"_min"]=min(present) if present else None
            result[metric+"_max"]=max(present) if present else None
        output.append(result)
    return output


def render(rows, control_rows, directory):
    from plot_energy_progress import render_pgf,tex_escape
    groups=defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in (*REGIME_FIELDS,"sample_indices_sha256"))].append(row)
    files=[]
    colors={"ar":"black","static":"blue!75!black","native_unrestricted":"orange!85!black",
            "native_threshold":"green!55!black","adaptive":"purple"}
    def axes_for(values, metric, title, ylabel):
        if not any(r.get(metric+"_median") is not None for r in values): return None
        xlabel="Synchronous wave batch B" if all(r["admission_mode"]=="wave" for r in values) else "Admission cap B"
        lines=[r"\nextgroupplot[title={%s},ylabel={%s},xlabel={%s},xmode=log,log basis x=2,xtick={1,4,16,64},xticklabels={1,4,16,64},legend pos=north west]"%(tex_escape(title),tex_escape(ylabel),tex_escape(xlabel))]
        for role in ROLES:
            series=sorted((r for r in values if r["role"]==role and r.get(metric+"_median") is not None),key=lambda r:r["batch"])
            if not series: continue
            points=" ".join(f"({r['batch']},{r[metric+'_median']:.10g})" for r in series)
            raw=" ".join(f"({r['batch']},{v:.10g})" for r in series for v in r[metric+"_raw"] if v is not None)
            lines.extend([r"\addplot[color=%s,mark=*,mark size=2pt] coordinates {%s};\addlegendentry{%s}"%(colors[role],points,tex_escape(LABELS[role])),
                          r"\addplot[only marks,color=%s,mark=o,mark size=2.8pt,opacity=.5,forget plot] coordinates {%s};"%(colors[role],raw)])
        return "\n".join(lines)
    for key,values in sorted(groups.items()):
        matching=[r for r in control_rows if tuple(r[k] for k in (*REGIME_FIELDS,"sample_indices_sha256"))==key and r["phase"]=="decode"]
        specs=[("F3",values,[("effective_tpot_ms","(a) Complete-request effective TPOT","TPOT (ms)"),("population_wall_s","(b) Complete population window","Population wall time (s)")],
                r"Lines: medians of run records; open points: every raw run value. No confidence intervals.\\Frozen choices are executed policies; no evaluation-set oracle or step splicing."),
               ("F4",values,[("joules_per_token","(a) Sum of measurement windows","Energy (J / output token)"),("population_counter_joules_per_token","(b) Counter span including gaps","Energy (J / output token)")],
                r"Panel (a) sums wave windows. Panel (b) spans validated first-start to last-end raw cumulative NVML counters on one device.\\Counter-read midpoint boundaries include inter-wave gaps; they are not exactly runner wall-time boundaries. Missing counters remain absent."),
               ("F6",matching,[("policy_choose_cpu_us_mean","(a) Actual adaptive choose","CPU choose (microseconds)"),("wrapper_cpu_us_mean","(b) Controller wrapper","CPU wrapper (microseconds)"),("discarded_available_token_fraction","(c) Post-proposer truncation","Discarded / available draft tokens")],
                r"All scheduled decode steps, including terminal/drain; each point summarizes one measured run record.\\Wrapper includes choose/truncation and is nested in scheduler instrumentation; discarded tokens are not GPU time or energy.")]
        for prefix,data,metrics,note in specs:
            axes=[a for metric,title,ylabel in metrics if (a:=axes_for(data,metric,title,ylabel))]
            if not axes: continue
            tex="\n".join([r"\documentclass[border=5pt]{standalone}",r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{pgfplots}\usepgfplotslibrary{groupplots}",
                r"\pgfplotsset{compat=1.18}\begin{document}\sffamily\begin{tikzpicture}",
                r"\begin{groupplot}[group style={group size=%d by 1,horizontal sep=1.9cm},width=7.8cm,height=5.7cm,scale only axis,grid=major,grid style={gray!20},axis lines*=left,title style={font=\small},label style={font=\small},legend style={font=\scriptsize,draw=none}]"%len(axes),
                *axes,r"\end{groupplot}",r"\path (group c1r1.south west) -- node[below=1.3cm,align=center,font=\small] {%s} (group c%dr1.south east);"%(note,len(axes)),r"\end{tikzpicture}\end{document}"])
            name=prefix+"_executed_policies_"+hashlib.sha256(canonical(key).encode()).hexdigest()[:9]
            files+=render_pgf(tex,directory,name,["pdf","svg","png"])
    return files


def analyze(args):
    selection=load_json(args.selection)
    runs,control_rows,exclusions,provenance=[],[],[],[]
    seen={}
    coverage=[]
    for role in ROLES:
        paths=getattr(args,role)
        available=[str(Path(p)) for p in paths if Path(p).exists()]
        for p in paths:
            if not Path(p).exists(): exclusions.append({"role":role,"source":p,"reason":"input_not_available_yet"})
        loaded,rejected=load_runs(available)
        exclusions.extend({"role":role,**r} for r in rejected)
        accepted=[]
        for row in loaded:
            source=Path(row["source"])
            try:
                if source.suffix!=".json": raise ValueError("original run JSON required")
                if row["source"] in seen: raise ValueError(f"source already assigned to role {seen[row['source']]}")
                if row["batch"] not in args.batches: raise ValueError("batch outside declared evaluation protocol")
                record=load_json(source);config=load_json(source.parent/"configuration.json",{})
                engine=load_json(source.parent/"engine_args.json")
                if engine is None: raise ValueError("actual engine arguments unavailable")
                identity=sample_identity(record,config,locate_workload(config.get("plan",{}),args.workload),args.input_offset,args.population,args.output_tokens)
                validate_policy(role,row,engine,selection)
                source_hash=sha(source)
                independence=selection_checks(selection,row,identity,source_hash)
                if role not in ("ar",) and independence=="selection_unavailable":
                    raise ValueError("frozen selection provenance required for the strategy comparison")
                calibration=calibration_checks(args.calibration,row,record,config) if role=="adaptive" else {}
                control,snapshots=controls(record)
                initial_state=load_json(source.parent/"evaluation_initial_state.json")
                if initial_state:
                    if initial_state.get("warmup_offset")!=0 or initial_state.get("shape_warmup_offset")!=0 or initial_state.get("input_offset")!=args.input_offset:
                        raise ValueError("recorded evaluation initial-state offsets differ from the held-out protocol")
                energy_span=population_counter_energy(record)
                population_wall=_finite(record.get("population_wall_s"))
                if population_wall is not None and population_wall < row["wall_s"]-1e-8:
                    raise ValueError("population window is shorter than sum of measurement windows")
                complete={**row,**identity,"role":role,"source_sha256":source_hash,"population_wall_s":population_wall,
                    "population_tokens_per_second":row["output_tokens"]/population_wall if population_wall else None,
                    "population_inter_episode_gap_s":_finite(record.get("population_inter_episode_gap_s")),
                    "energy_j":_finite(record.get("energy_j")),"selection_independence":independence,
                    "native_table":(engine.get("speculative_config") or {}).get("num_speculative_tokens_per_batch_size"),
                    "energy_boundary":"sum of measured wave windows" if row["admission_mode"]=="wave" else "one measured complete population window",
                    "warm_state":"development shape warmup at offset 0; EWMA continues across B and repeat records within this resident engine",
                    "initial_acceptance_snapshot_status":"recorded" if initial_state and initial_state.get("acceptance_snapshot_before") is not None else "not_recorded",
                    "engine_evaluation_initial_state":initial_state,
                    "acceptance_snapshots":snapshots,**calibration,**energy_span}
                runs.append(complete);accepted.append(complete);seen[row["source"]]=role
                for c in control:
                    control_rows.append({**{k:complete[k] for k in (*REGIME_FIELDS,"sample_indices_sha256","role","batch","source","repeat")},**c})
                provenance.append({"role":role,"source":row["source"],"sha256":source_hash,
                    "configuration_sha256":sha(source.parent/"configuration.json"),"engine_args_sha256":sha(source.parent/"engine_args.json")})
            except (ValueError,KeyError,TypeError) as exc:
                exclusions.append({"role":role,"source":row["source"],"reason":str(exc)})
        for batch in args.batches:
            observed=[r for r in accepted if r["batch"]==batch]
            repeats=[r["repeat"] for r in observed]
            coverage.append({"role":role,"batch":batch,"expected_run_records":args.repeats,
                "observed_run_records":len(observed),"repeats":repeats,
                "status":"measured" if len(observed)==args.repeats and len(set(repeats))==len(repeats) else "pending_or_duplicate_or_extra_runs"})
    summary=summarize(runs,METRICS)
    control_summary=summarize(control_rows,CONTROL_METRICS,("phase",))
    weights=(selection or {}).get("normalized_batch_weights",{str(b):1/len(args.batches) for b in args.batches})
    weights={int(k):float(v) for k,v in weights.items()}
    objectives=[]
    objective_groups=defaultdict(list)
    for row in summary:
        objective_groups[tuple(row[k] for k in (*REGIME_FIELDS,"sample_indices_sha256","role"))].append(row)
    for key,values in objective_groups.items():
        cells={r["batch"]:r for r in values}
        full=set(cells)==set(weights)
        objectives.append({**dict(zip((*REGIME_FIELDS,"sample_indices_sha256","role"),key)),
            "weighted_run_median_effective_tpot_ms":sum(weights[b]*cells[b]["effective_tpot_ms_median"] for b in weights) if full else None,
            "batch_weights":weights,"status":"measured_all_batches" if full else "pending_missing_batches",
            "boundary":"executed evaluation policies; weighted medians of run records; no independent-engine confidence interval"})
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    for name,values in (("policy_runs.csv",runs),("policy_summary.csv",summary),("control_runs.csv",control_rows),
                        ("control_summary.csv",control_summary),("coverage.csv",coverage),("weighted_objective.csv",objectives)):
        output_csv(output/name,values)
    figures=[]
    if args.figures and summary:
        directory=Path(args.figures);directory.mkdir(parents=True,exist_ok=True)
        figures=render(summary,control_summary,directory)
    manifest={"schema_version":1,"status":"measured_comparison_complete" if all(r["status"]=="measured" for r in coverage) else "pending_or_partial",
        "protocol":{"input_offset":args.input_offset,"population":args.population,"output_tokens_per_request":args.output_tokens,"batches":args.batches,"repeats":args.repeats,"shape_warmup_offset":0},
        "selection":str(args.selection) if args.selection else None,"selection_sha256":sha(args.selection) if args.selection and Path(args.selection).is_file() else None,
        "calibration":str(args.calibration) if args.calibration else None,"coverage":coverage,"exclusions":exclusions,
        "source_provenance":provenance,"figure_artifacts":figures,"analysis_code_sha256":sha(__file__),
        "boundaries":["No offline oracle, selection score or spliced trajectory is a policy measurement.",
            "Policies use their actual engine configurations. In vLLM 0.26 V1, a native per-batch speculative table forces PIECEWISE capture, whereas fixed-budget compiled engines may use FULL_AND_PIECEWISE with a different capture grid. This is a whole-policy comparison, not an isolated decision-rule or matched-graph experiment; timed dispatch is not observed here.",
            "Raw repeat records retained; shared resident engines and continuous estimator state do not constitute independent engine repeats.",
            "TPOT uses all complete requests; complete-population wall includes inter-wave handling when recorded.",
            "Wave-window energy and validated whole-device cumulative-counter span energy are reported separately; counter span includes gaps but does not have exactly the runner wall-time endpoints. No sparse-power extrapolation across gaps.",
            "CPU metrics are nested scopes and must not be added or subtracted from wall time to estimate an uninstrumented counterfactual.",
            "Truncated draft counts are post-proposer work counts, not measured wasted GPU kernel time or energy.",
            "Only measured cells are plotted. Missing roles, fields, cost provenance or sample identities remain pending/excluded."]}
    (output/"comparison.json").write_text(json.dumps({**manifest,"runs":runs,"summary":summary,"control_runs":control_rows,"control_summary":control_summary,"weighted_objective":objectives},indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps({"status":manifest["status"],"accepted_run_records":len(runs),"excluded_inputs":len(exclusions),"figure_artifacts":len(figures),"output":str(output)}))
    return manifest


def self_test():
    class Checks(unittest.TestCase):
        def test_sample_ids_are_not_offsets(self):
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/"workload.json"
                path.write_text(json.dumps({"records":[{"source_index":100+i} for i in range(4)]}))
                config={"workload_sha256":sha(path),"plan":{"input_offset":2,"warmup_offset":0,"warmup_each_batch":True,"shape_warmup_offset":0}}
                record={"output_tokens":4,"requests":[{"source_index":i,"finished":True,"output_token_ids":[1,2],"effective_tpot_ms":3} for i in (102,103)]}
                self.assertEqual(sample_identity(record,config,path,2,2,2)["sample_source_indices"],[102,103])
                record["requests"][0]["source_index"]=2
                with self.assertRaises(ValueError): sample_identity(record,config,path,2,2,2)
        def test_native_roles_use_exact_frozen_table(self):
            selection={"status":"selected_pending_evaluation","native_threshold":{"table":[[1,4,3],[5,64,0]],"runner":{"gamma":7,"proposal_budget":7}}}
            row={"mode":"native","gamma":7,"proposal_budget":7}
            engine={"speculative_config":{"num_speculative_tokens":7,"num_speculative_tokens_per_batch_size":[[1,4,3],[5,64,0]]}}
            validate_policy("native_threshold",row,engine,selection)
            engine["speculative_config"]["num_speculative_tokens_per_batch_size"]=[[1,64,3]]
            with self.assertRaises(ValueError): validate_policy("native_threshold",row,engine,selection)
        def test_missing_timing_and_zero_truncation_are_distinct(self):
            record={"output_tokens":2,"label":"wave","host_steps":[{"index":0,"host_step_ms":10}],"plugin":{"steps":[{"step":0,"episode":"wave","phase":"decode","scheduled_tokens":2,"drafts_available_before":1,"truncated_by_controller":0,"scheduled_drafts":1,"control_ns":1000}]}}
            rows,_=controls(record)
            self.assertEqual(rows[0]["discarded_available_token_fraction"],0)
            self.assertIsNone(rows[0]["policy_choose_cpu_us_mean"])
            self.assertAlmostEqual(rows[0]["wrapper_host_fraction"],.0001)
            del record["plugin"]["steps"][0]["truncated_by_controller"]
            self.assertIsNone(controls(record)[0][0]["discarded_available_token_fraction"])
        def test_selection_rejects_same_prompts_in_different_files(self):
            selection={"source_provenance":[{"source":"other.json","sha256":"other","sample_source_indices":[123]}]}
            with self.assertRaises(ValueError): selection_checks(selection,{"source":"new.json"},{"sample_source_indices":[123]},"new")
        def test_counter_span_includes_gap_and_rejects_reset(self):
            def wave(start,end,a,b,uuid="gpu0"):
                read=lambda t,v:{"energy_mj":v,"read_start_ns":t-1,"monotonic_ns":t,"read_end_ns":t+1}
                return {"energy":{"device":{"uuid":uuid},"counter_valid":True,"counter_start":read(start,a),"counter_end":read(end,b)}}
            record={"output_tokens":10,"episodes":[wave(100,200,1000,1200),wave(300,400,1500,1800)]}
            result=population_counter_energy(record)
            self.assertEqual(result["population_counter_energy_j"],.8)
            self.assertEqual(result["population_counter_joules_per_token"],.08)
            record["episodes"][1]=wave(300,400,100,300)
            self.assertIsNone(population_counter_energy(record)["population_counter_energy_j"])
            record["episodes"][1]=wave(300,400,1500,1800,"gpu1")
            self.assertIsNone(population_counter_energy(record)["population_counter_energy_j"])
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    if not unittest.TextTestRunner().run(suite).wasSuccessful(): raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    for role in ROLES: parser.add_argument("--"+role.replace("_","-"),nargs="*",default=[])
    parser.add_argument("--selection",help="One complete frozen selection.json, not selection_manifest.json")
    parser.add_argument("--calibration",help="Measured full-budget table actually supplied to adaptive engine")
    parser.add_argument("--workload",help="Optional relocated workload; SHA must equal runner metadata")
    parser.add_argument("--input-offset",type=int,default=64)
    parser.add_argument("--population",type=int,default=64)
    parser.add_argument("--output-tokens",type=int,default=128)
    parser.add_argument("--batches",type=int,nargs="+",default=[1,4,16,64])
    parser.add_argument("--repeats",type=int,default=2)
    parser.add_argument("--output")
    parser.add_argument("--figures")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test: self_test();return
    if not args.output: parser.error("--output is required")
    analyze(args)


if __name__=="__main__": main()
