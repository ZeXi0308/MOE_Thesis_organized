#!/usr/bin/env python3
"""Count observable request-cap fields in pinned Exgentic span Parquet files.

Only IDs for deduplication and cap/length/finish/status metadata are read.
No prompt, response, tool-call content or per-session identifiers are emitted.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import pyarrow as pa
import pyarrow.parquet as pq

ATTRS = ("gen_ai.request.model", "gen_ai.request.max_tokens",
         "gen_ai.usage.output_tokens", "gen_ai.response.finish_reasons", "error.type")
def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def check_schema(schema: pa.Schema, path: Path) -> None:
    missing = {"session_id", "harness", "benchmark", "models", "spans"} - set(schema.names)
    if missing:
        raise ValueError(f"{path}: missing top-level fields {sorted(missing)}")
    typ = schema.field("spans").type
    if not (pa.types.is_list(typ) or pa.types.is_large_list(typ)):
        raise ValueError(f"{path}: spans must be a list of structs")
    typ = typ.value_type
    if not pa.types.is_struct(typ) or not {"span_id", "status", "attributes"} <= set(typ.names):
        raise ValueError(f"{path}: spans require span_id/status/attributes")
    status = typ.field("status").type
    if not pa.types.is_struct(status) or "code" not in status.names:
        raise ValueError(f"{path}: span status.code missing")
    attrs = typ.field("attributes").type
    if pa.types.is_struct(attrs):
        absent = set(ATTRS) - set(attrs.names)
        if absent:
            raise ValueError(f"{path}: span attributes missing {sorted(absent)}")
    elif not pa.types.is_map(attrs):
        raise ValueError(f"{path}: span attributes must be struct or map")


def attributes(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and all(isinstance(pair, (tuple, list)) and len(pair) == 2
                                       for pair in value):
        return dict(value)
    raise ValueError("span attributes value is not a map/struct")


def integer(value) -> int | None:
    if type(value) is int:
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value.strip())
    return None


def finish_set(value) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return set()
        if value.startswith("["):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return {"other_nonempty"}
        else:
            value = [value]
    if not isinstance(value, (list, tuple)):
        return {"other_nonempty"}
    reasons = set()
    for item in value:
        if item is None or not str(item).strip():
            continue
        name = str(item).strip().lower()
        reasons.add(name if re.fullmatch(r"[a-z0-9_.:-]{1,64}", name) else "other_nonempty")
    return reasons


def group() -> dict:
    return dict(spans=0, success=0, cap_positive=0, length_nonnegative=0,
                finish_nonempty=0, joint_cap_length_finish=0, success_joint=0,
                output_over_cap=0, output_below_cap=0, length_finish=0,
                length_finish_below_cap=0, success_joint_length_finish=0,
                success_joint_over_cap=0, cap_hist=Counter(), finish_hist=Counter(),
                success_joint_cap_hist=Counter(), success_joint_finish_hist=Counter())


def add(stats: dict, *, success: bool, cap: int | None, length: int | None,
        finishes: set[str]) -> None:
    stats["spans"] += 1
    stats["success"] += success
    stats["cap_positive"] += cap is not None
    stats["length_nonnegative"] += length is not None
    stats["finish_nonempty"] += bool(finishes)
    if cap is not None:
        stats["cap_hist"][str(cap)] += 1
    for reason in finishes:
        stats["finish_hist"][reason] += 1
    if cap is not None and length is not None:
        stats["output_over_cap"] += length > cap
        stats["output_below_cap"] += length < cap
        if finishes:
            stats["joint_cap_length_finish"] += 1
            stats["success_joint"] += success
            if success:
                stats["success_joint_cap_hist"][str(cap)] += 1
                for reason in finishes:
                    stats["success_joint_finish_hist"][reason] += 1
                stats["success_joint_length_finish"] += "length" in finishes
                stats["success_joint_over_cap"] += length > cap
    if "length" in finishes:
        stats["length_finish"] += 1
        stats["length_finish_below_cap"] += (cap is not None and length is not None
                                              and length < cap)


def freeze(stats: dict) -> dict:
    return {key: dict(sorted(value.items())) if isinstance(value, Counter) else value
            for key, value in stats.items()}


def analyze(input_dir: Path, dataset_id: str, revision: str) -> dict:
    files = sorted(input_dir.rglob("*.parquet"))
    if not input_dir.is_dir() or not files:
        raise FileNotFoundError(f"no Parquet files under {input_dir}")
    overall = group()
    by_model, by_harness = defaultdict(group), defaultdict(group)
    by_model_harness = defaultdict(group)
    top = Counter()
    seen_sessions, seen_spans, seen_global_spans = set(), set(), set()
    session_caps, session_model_caps = defaultdict(set), defaultdict(lambda: defaultdict(set))
    session_joint_caps = defaultdict(set)
    session_joint_model_caps = defaultdict(lambda: defaultdict(set))
    manifest, first_schema = [], None
    for path in files:
        reader = pq.ParquetFile(path)
        schema = reader.schema_arrow
        check_schema(schema, path)
        if first_schema is not None and not schema.equals(first_schema, check_metadata=False):
            raise ValueError(f"{path}: Parquet schema differs from first file")
        first_schema = schema
        manifest.append(dict(path=str(path.relative_to(input_dir)), bytes=path.stat().st_size,
                             sha256=digest(path), rows=reader.metadata.num_rows))
        span_fields = ("span_id", "status.code", *(f"attributes.{name}" for name in ATTRS))
        paths = [reader.schema.column(i).path for i in range(len(reader.schema))]
        selected = [path for path in paths if path.startswith("spans.") and any(
                    path.endswith("." + name) or path.endswith("." + name + ".list.element")
                    for name in span_fields)]
        if len(selected) != len(span_fields):
            raise ValueError(f"{path}: missing or duplicate required span leaf paths")
        columns = ["session_id", "harness", "benchmark", "models", *selected]
        for batch in reader.iter_batches(batch_size=64, columns=columns):
            for row in batch.to_pylist():
                top["rows"] += 1
                sid = row["session_id"]
                if sid is None or not str(sid).strip():
                    top["missing_session_id_rows"] += 1
                    sid = None
                else:
                    sid = str(sid)
                    top["duplicate_session_id_rows"] += sid in seen_sessions
                    seen_sessions.add(sid)
                harness = str(row["harness"]) if row["harness"] is not None else "(missing)"
                top["missing_harness_rows"] += harness == "(missing)"
                top["missing_benchmark_rows"] += row["benchmark"] is None
                top["models_present_rows"] += bool(row["models"])
                spans = row["spans"] or []
                top["rows_with_no_spans"] += not bool(spans)
                for span in spans:
                    top["spans"] += 1
                    span_id = span.get("span_id") if span else None
                    if span_id is None or not str(span_id).strip():
                        top["missing_span_id_spans"] += 1
                    else:
                        span_id = str(span_id)
                        if sid is not None:
                            key = (sid, span_id)
                            top["duplicate_session_span_ids"] += key in seen_spans
                            seen_spans.add(key)
                        top["duplicate_global_span_ids"] += span_id in seen_global_spans
                        seen_global_spans.add(span_id)
                    attrs = attributes(span.get("attributes") if span else None)
                    status = span.get("status") if span else None
                    code = status.get("code") if isinstance(status, dict) else None
                    if code is None:
                        top["missing_status_code_spans"] += 1
                    elif type(code) is not int:
                        top["noninteger_status_code_spans"] += 1
                    else:
                        top[f"status_code_{code}"] += 1
                    error = attrs.get("error.type")
                    error_empty = error is None or (isinstance(error, str) and not error.strip())
                    top["nonempty_error_type_spans"] += not error_empty
                    # Missing/noninteger status is never silently classified as success.
                    success = type(code) is int and code != 2 and error_empty
                    model = attrs.get("gen_ai.request.model")
                    model = str(model).strip() if model is not None else ""
                    if not model:
                        top["missing_request_model_spans"] += 1
                        model = "(missing)"
                    raw_cap = attrs.get("gen_ai.request.max_tokens")
                    raw_length = attrs.get("gen_ai.usage.output_tokens")
                    top["cap_present_spans"] += raw_cap is not None
                    top["length_present_spans"] += raw_length is not None
                    cap, length = integer(raw_cap), integer(raw_length)
                    cap = cap if cap is not None and cap > 0 else None
                    length = length if length is not None and length >= 0 else None
                    finishes = finish_set(attrs.get("gen_ai.response.finish_reasons"))
                    for stats in (overall, by_model[model], by_harness[harness],
                                  by_model_harness[(model, harness)]):
                        add(stats, success=success, cap=cap, length=length,
                            finishes=finishes)
                    if sid is not None and cap is not None:
                        session_caps[sid].add(cap)
                        if model != "(missing)":
                            session_model_caps[sid][model].add(cap)
                        if success and length is not None and finishes:
                            session_joint_caps[sid].add(cap)
                            if model != "(missing)":
                                session_joint_model_caps[sid][model].add(cap)
    distinct_caps = Counter(len(session_caps[sid]) for sid in seen_sessions)
    same_model_mixed = sum(any(len(caps) > 1 for caps in session_model_caps[sid].values())
                           for sid in seen_sessions)
    return dict(schema="c-exgentic-cap-fields-v1",
        source=dict(dataset_id=dataset_id, revision=revision,
                    files=manifest, file_count=len(manifest)),
        identity=dict(rows=top["rows"], unique_sessions=len(seen_sessions),
            duplicate_session_id_rows=top["duplicate_session_id_rows"],
            missing_session_id_rows=top["missing_session_id_rows"],
            spans=top["spans"], unique_global_span_ids=len(seen_global_spans),
            duplicate_global_span_ids=top["duplicate_global_span_ids"],
            duplicate_session_span_ids=top["duplicate_session_span_ids"],
            missing_span_id_spans=top["missing_span_id_spans"],
            rows_with_no_spans=top["rows_with_no_spans"]),
        coverage=dict(missing_status_code_spans=top["missing_status_code_spans"],
            noninteger_status_code_spans=top["noninteger_status_code_spans"],
            status_code_counts={key.removeprefix("status_code_"): value for key, value in top.items()
                                if key.startswith("status_code_")},
            nonempty_error_type_spans=top["nonempty_error_type_spans"],
            missing_request_model_spans=top["missing_request_model_spans"],
            cap_present_spans=top["cap_present_spans"],
            length_present_spans=top["length_present_spans"],
            missing_harness_rows=top["missing_harness_rows"],
            missing_benchmark_rows=top["missing_benchmark_rows"],
            models_present_rows=top["models_present_rows"],
            **freeze(overall)),
        session_caps=dict(distinct_positive_cap_count_hist=dict(sorted(distinct_caps.items())),
            sessions_with_multiple_positive_caps=sum(n > 1 for n in
                (len(session_caps[sid]) for sid in seen_sessions)),
            sessions_with_same_model_mixed_caps=same_model_mixed,
            success_joint_distinct_cap_count_hist=dict(sorted(Counter(
                len(session_joint_caps[sid]) for sid in seen_sessions).items())),
            success_joint_sessions_with_multiple_caps=sum(len(session_joint_caps[sid]) > 1
                                                          for sid in seen_sessions),
            success_joint_sessions_with_same_model_mixed_caps=sum(
                any(len(caps) > 1 for caps in session_joint_model_caps[sid].values())
                for sid in seen_sessions)),
        by_model={key: freeze(value) for key, value in sorted(by_model.items())},
        by_harness={key: freeze(value) for key, value in sorted(by_harness.items())},
        by_model_harness=[dict(model=model, harness=harness, **freeze(value))
                          for (model, harness), value in sorted(by_model_harness.items())])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.input_dir, args.dataset_id, args.revision)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
