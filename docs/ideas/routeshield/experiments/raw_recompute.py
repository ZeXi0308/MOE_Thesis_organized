from __future__ import annotations

"""Hash-bound raw paired-block recomputation for RouteShield Gate-0.

This module is deliberately development-only until the full request-DAG,
executed-dispatch, exactness-tensor, and Oracle-certificate validators close.
It recomputes request TTFT P99 and paired-block intervals; it cannot authorize
a formal scientific verdict by itself.
"""

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

try:
    from .schema import HEX64, ProtocolError
except ImportError:
    from schema import HEX64, ProtocolError


RAW_EVALUATOR_VERSION = "routeshield-raw-recompute-v1"
PRIMARY_SCENARIOS = (
    "MATCHED_BENIGN",
    "ATTACK_BASELINE",
    "LEGAL_ORACLE",
    "STRONGEST_SIMPLE",
)
CONTROL_SCENARIOS = ("CONTROL_DEFAULT", "CONTROL_ISOLATION")
ALL_SCENARIOS = frozenset((*PRIMARY_SCENARIOS, *CONTROL_SCENARIOS))
ROLES = frozenset({"victim", "attacker", "cotenant"})
TERMINALS = frozenset({"COMPLETED", "DROPPED", "CANCELLED", "TIMED_OUT"})
CELLS = (
    ("70pct", "ADV_TEXT"),
    ("30pct", "NAT_BENIGN"),
    ("70pct", "NAT_PATHOLOGICAL"),
)

REQUEST_FIELDS = frozenset(
    {
        "model",
        "load_cell",
        "traffic_class",
        "scenario",
        "block_id",
        "pair_id",
        "tenant_id",
        "role",
        "request_id",
        "document_id",
        "document_cluster_id",
        "prompt_hash",
        "input_tokens",
        "max_new_tokens",
        "arrival_ns",
        "first_token_ns",
        "completion_ns",
        "output_token_count",
        "output_hash",
        "terminal_reason",
    }
)
BLOCK_FIELDS = frozenset(
    {
        "model",
        "load_cell",
        "traffic_class",
        "scenario",
        "block_id",
        "policy_id",
        "request_world_sha256",
        "arrival_trace_sha256",
        "victim_manifest_sha256",
        "cotenant_budget_sha256",
        "window_start_ns",
        "window_end_ns",
        "queue_service_work_start",
        "queue_service_work_end",
        "queue_service_work_arrived",
        "oracle_status",
        "oracle_gap",
    }
)
ARTIFACT_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "size_bytes",
        "row_count",
        "format",
        "schema",
        "config_key",
    }
)
MANIFEST_FIELDS = frozenset(
    {"schema", "mode", "config_sha256", "evaluator_source_sha256", "artifacts"}
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ProtocolError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ProtocolError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def strict_json_loads(text: str, *, source: str) -> object:
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"{source}: invalid JSON") from exc


def strict_json_file(path: str | Path) -> object:
    source = Path(path)
    return strict_json_loads(source.read_text(encoding="utf-8"), source=str(source))


def strict_jsonl(path: str | Path) -> list[Mapping[str, object]]:
    source = Path(path)
    rows: list[Mapping[str, object]] = []
    with source.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                raise ProtocolError(f"{source}:{lineno}: blank JSONL rows are forbidden")
            parsed = strict_json_loads(line, source=f"{source}:{lineno}")
            if not isinstance(parsed, Mapping):
                raise ProtocolError(f"{source}:{lineno}: JSONL row must be an object")
            rows.append(parsed)
    if not rows:
        raise ProtocolError(f"{source}: JSONL file is empty")
    return rows


def _exact_fields(row: Mapping[str, object], expected: frozenset[str], kind: str) -> None:
    observed = set(row)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ProtocolError(f"{kind} fields mismatch; missing={missing}, extra={extra}")


def _text(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{key} must be a non-empty string")
    return value


def _sha(row: Mapping[str, object], key: str) -> str:
    value = _text(row, key)
    if not HEX64.fullmatch(value):
        raise ProtocolError(f"{key} must be lowercase SHA-256")
    return value


def _integer(row: Mapping[str, object], key: str, *, minimum: int = 0) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProtocolError(f"{key} must be an integer >= {minimum}")
    return value


def _optional_integer(row: Mapping[str, object], key: str) -> int | None:
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProtocolError(f"{key} must be null or a non-negative integer")
    return value


def _number(row: Mapping[str, object], key: str, *, minimum: float = 0.0) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{key} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted < minimum:
        raise ProtocolError(f"{key} must be finite and >= {minimum}")
    return converted


def _optional_number(row: Mapping[str, object], key: str) -> float | None:
    if row.get(key) is None:
        return None
    return _number(row, key)


@dataclass(frozen=True)
class RawRequest:
    model: str
    load_cell: str
    traffic_class: str
    scenario: str
    block_id: str
    pair_id: str
    tenant_id: str
    role: str
    request_id: str
    document_id: str
    document_cluster_id: str
    prompt_hash: str
    input_tokens: int
    max_new_tokens: int
    arrival_ns: int
    first_token_ns: int | None
    completion_ns: int | None
    output_token_count: int
    output_hash: str | None
    terminal_reason: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "RawRequest":
        _exact_fields(row, REQUEST_FIELDS, "raw request")
        scenario = _text(row, "scenario")
        role = _text(row, "role")
        terminal = _text(row, "terminal_reason")
        if scenario not in ALL_SCENARIOS:
            raise ProtocolError(f"unknown raw scenario: {scenario}")
        if role not in ROLES:
            raise ProtocolError(f"unknown raw role: {role}")
        if terminal not in TERMINALS:
            raise ProtocolError(f"unknown terminal_reason: {terminal}")
        first_token = _optional_integer(row, "first_token_ns")
        completion = _optional_integer(row, "completion_ns")
        output_count = _integer(row, "output_token_count")
        raw_output_hash = row.get("output_hash")
        output_hash = None
        if raw_output_hash is not None:
            if not isinstance(raw_output_hash, str) or not HEX64.fullmatch(raw_output_hash):
                raise ProtocolError("output_hash must be null or lowercase SHA-256")
            output_hash = raw_output_hash
        arrival = _integer(row, "arrival_ns")
        if terminal == "COMPLETED":
            if first_token is None or completion is None or output_hash is None:
                raise ProtocolError("completed request lacks first/completion/output evidence")
            if not arrival < first_token <= completion:
                raise ProtocolError("request timestamps are not monotonic")
            if output_count <= 0:
                raise ProtocolError("completed request must contain output tokens")
        elif (
            first_token is not None
            or completion is not None
            or output_count != 0
            or output_hash is not None
        ):
            raise ProtocolError("non-completed request cannot carry completion measurements")

        return cls(
            model=_text(row, "model"),
            load_cell=_text(row, "load_cell"),
            traffic_class=_text(row, "traffic_class"),
            scenario=scenario,
            block_id=_sha(row, "block_id"),
            pair_id=_sha(row, "pair_id"),
            tenant_id=_text(row, "tenant_id"),
            role=role,
            request_id=_text(row, "request_id"),
            document_id=_text(row, "document_id"),
            document_cluster_id=_text(row, "document_cluster_id"),
            prompt_hash=_sha(row, "prompt_hash"),
            input_tokens=_integer(row, "input_tokens", minimum=1),
            max_new_tokens=_integer(row, "max_new_tokens", minimum=1),
            arrival_ns=arrival,
            first_token_ns=first_token,
            completion_ns=completion,
            output_token_count=output_count,
            output_hash=output_hash,
            terminal_reason=terminal,
        )

    @property
    def ttft_ns(self) -> int:
        if self.first_token_ns is None:
            raise ProtocolError("TTFT requested for an incomplete request")
        return self.first_token_ns - self.arrival_ns


@dataclass(frozen=True)
class RawBlock:
    model: str
    load_cell: str
    traffic_class: str
    scenario: str
    block_id: str
    policy_id: str
    request_world_sha256: str
    arrival_trace_sha256: str
    victim_manifest_sha256: str
    cotenant_budget_sha256: str
    window_start_ns: int
    window_end_ns: int
    queue_service_work_start: float
    queue_service_work_end: float
    queue_service_work_arrived: float
    oracle_status: str
    oracle_gap: float | None

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "RawBlock":
        _exact_fields(row, BLOCK_FIELDS, "raw block")
        start = _integer(row, "window_start_ns")
        end = _integer(row, "window_end_ns", minimum=1)
        if end <= start:
            raise ProtocolError("block window_end_ns must exceed window_start_ns")
        scenario = _text(row, "scenario")
        if scenario not in ALL_SCENARIOS:
            raise ProtocolError(f"unknown block scenario: {scenario}")
        oracle_status = _text(row, "oracle_status")
        oracle_gap = _optional_number(row, "oracle_gap")
        if scenario == "LEGAL_ORACLE":
            if oracle_status not in {"OPTIMAL", "TIMEOUT", "STATE_LIMIT"}:
                raise ProtocolError("legal Oracle status is invalid")
            if oracle_status == "OPTIMAL" and oracle_gap != 0.0:
                raise ProtocolError("OPTIMAL Oracle requires exact gap=0")
        elif oracle_status != "NOT_APPLICABLE" or oracle_gap is not None:
            raise ProtocolError("non-Oracle blocks must use NOT_APPLICABLE/null")
        return cls(
            model=_text(row, "model"),
            load_cell=_text(row, "load_cell"),
            traffic_class=_text(row, "traffic_class"),
            scenario=scenario,
            block_id=_sha(row, "block_id"),
            policy_id=_text(row, "policy_id"),
            request_world_sha256=_sha(row, "request_world_sha256"),
            arrival_trace_sha256=_sha(row, "arrival_trace_sha256"),
            victim_manifest_sha256=_sha(row, "victim_manifest_sha256"),
            cotenant_budget_sha256=_sha(row, "cotenant_budget_sha256"),
            window_start_ns=start,
            window_end_ns=end,
            queue_service_work_start=_number(row, "queue_service_work_start"),
            queue_service_work_end=_number(row, "queue_service_work_end"),
            queue_service_work_arrived=_number(
                row, "queue_service_work_arrived", minimum=1e-12
            ),
            oracle_status=oracle_status,
            oracle_gap=oracle_gap,
        )

    @property
    def duration_ns(self) -> int:
        return self.window_end_ns - self.window_start_ns


def load_requests(path: str | Path) -> list[RawRequest]:
    return [RawRequest.from_mapping(row) for row in strict_jsonl(path)]


def load_blocks(path: str | Path) -> list[RawBlock]:
    return [RawBlock.from_mapping(row) for row in strict_jsonl(path)]


def _linear_quantile(values: Sequence[float], q: float) -> float:
    if not values or not 0.0 <= q <= 1.0:
        raise ProtocolError("quantile requires non-empty finite values and q in [0,1]")
    ordered = sorted(values)
    if any(not math.isfinite(value) for value in ordered):
        raise ProtocolError("quantile input contains a non-finite value")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(frozen=True)
class WeightedNearestRankP99:
    values: tuple[int, ...]
    cumulative_by_value: tuple[tuple[int, ...], ...]
    totals_by_block: tuple[int, ...]

    @classmethod
    def build(
        cls, values_by_block: Mapping[str, Sequence[int]], block_order: Sequence[str]
    ) -> "WeightedNearestRankP99":
        counts_by_value: dict[int, list[int]] = {}
        totals: list[int] = []
        for block_index, block_id in enumerate(block_order):
            values = list(values_by_block.get(block_id, ()))
            if not values:
                raise ProtocolError(f"block {block_id} has no victim TTFT values")
            totals.append(len(values))
            for value, count in Counter(values).items():
                counts_by_value.setdefault(value, [0] * len(block_order))[block_index] = count
        ordered_values = sorted(counts_by_value)
        cumulative = [0] * len(block_order)
        rows: list[tuple[int, ...]] = []
        for value in ordered_values:
            for index, count in enumerate(counts_by_value[value]):
                cumulative[index] += count
            rows.append(tuple(cumulative))
        return cls(tuple(ordered_values), tuple(rows), tuple(totals))

    def evaluate(self, multiplicities: Sequence[int]) -> float:
        if len(multiplicities) != len(self.totals_by_block):
            raise ProtocolError("bootstrap multiplicity vector has the wrong size")
        total = sum(
            count * multiplicity
            for count, multiplicity in zip(self.totals_by_block, multiplicities)
        )
        if total <= 0:
            raise ProtocolError("bootstrap sample contains no requests")
        target = math.ceil(0.99 * total)
        low, high = 0, len(self.values) - 1
        while low < high:
            middle = (low + high) // 2
            observed = sum(
                count * multiplicity
                for count, multiplicity in zip(
                    self.cumulative_by_value[middle], multiplicities
                )
            )
            if observed >= target:
                high = middle
            else:
                low = middle + 1
        return float(self.values[low])


def _stable_cell_seed(base_seed: int, model: str, cell: str) -> int:
    material = f"{RAW_EVALUATOR_VERSION}|{base_seed}|{model}|{cell}"
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _bootstrap_multiplicities(
    block_count: int, *, replicates: int, seed: int
) -> Iterable[list[int]]:
    rng = random.Random(seed)
    for _ in range(replicates):
        multiplicities = [0] * block_count
        for _ in range(block_count):
            multiplicities[rng.randrange(block_count)] += 1
        yield multiplicities


def _ratio_of_sums(
    numerators: Sequence[float], denominators: Sequence[float], multiplicities: Sequence[int]
) -> float:
    numerator = sum(value * count for value, count in zip(numerators, multiplicities))
    denominator = sum(value * count for value, count in zip(denominators, multiplicities))
    if numerator < 0 or denominator <= 0:
        raise ProtocolError("goodput ratio-of-sums has an invalid denominator")
    return numerator / denominator


def _canonical_records_sha256(records: Iterable[Mapping[str, object]]) -> str:
    encoded = json.dumps(
        sorted(records, key=lambda row: str(row["pair_id"])),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def request_provenance_hashes(
    requests: Sequence[RawRequest],
) -> dict[str, str]:
    """Recompute block provenance from request inputs, never producer summaries."""

    if not requests:
        raise ProtocolError("cannot hash an empty request world")
    arrival_records = [
        {"pair_id": row.pair_id, "arrival_ns": row.arrival_ns} for row in requests
    ]
    victim_records = [
        {
            "pair_id": row.pair_id,
            "tenant_id": row.tenant_id,
            "request_id": row.request_id,
            "document_id": row.document_id,
            "document_cluster_id": row.document_cluster_id,
            "prompt_hash": row.prompt_hash,
            "input_tokens": row.input_tokens,
            "max_new_tokens": row.max_new_tokens,
            "arrival_ns": row.arrival_ns,
        }
        for row in requests
        if row.role == "victim"
    ]
    cotenant_budget_records = [
        {
            "pair_id": row.pair_id,
            "tenant_id": row.tenant_id,
            "request_id": row.request_id,
            "input_tokens": row.input_tokens,
            "max_new_tokens": row.max_new_tokens,
            "arrival_ns": row.arrival_ns,
        }
        for row in requests
        if row.role != "victim"
    ]
    world_records = [
        {
            "pair_id": row.pair_id,
            "tenant_id": row.tenant_id,
            "role": row.role,
            "request_id": row.request_id,
            "document_id": row.document_id,
            "document_cluster_id": row.document_cluster_id,
            "prompt_hash": row.prompt_hash,
            "input_tokens": row.input_tokens,
            "max_new_tokens": row.max_new_tokens,
            "arrival_ns": row.arrival_ns,
        }
        for row in requests
    ]
    if not victim_records or not cotenant_budget_records:
        raise ProtocolError("each request world requires victim and cotenant work")
    return {
        "request_world_sha256": _canonical_records_sha256(world_records),
        "arrival_trace_sha256": _canonical_records_sha256(arrival_records),
        "victim_manifest_sha256": _canonical_records_sha256(victim_records),
        "cotenant_budget_sha256": _canonical_records_sha256(
            cotenant_budget_records
        ),
    }


def _observed_wall_clock_ns(requests: Sequence[RawRequest]) -> float:
    if not requests or any(row.completion_ns is None for row in requests):
        raise ProtocolError("wall-clock goodput requires completed raw requests")
    first_arrival = min(row.arrival_ns for row in requests)
    last_completion = max(
        row.completion_ns for row in requests if row.completion_ns is not None
    )
    if last_completion <= first_arrival:
        raise ProtocolError("recomputed wall-clock interval must be positive")
    return float(last_completion - first_arrival)


def _config_lookup(config: Mapping[str, Any], dotted: str) -> object:
    value: object = config
    for part in dotted.split("."):
        if isinstance(value, Mapping) and part in value:
            value = value[part]
            continue
        if isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
            continue
        else:
            raise ProtocolError(f"bundle binding references unknown config key: {dotted}")
    return value


def _sha_config_paths(value: object, *, path: str = "") -> set[str]:
    output: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if str(key).endswith("_sha256"):
                output.add(child)
            output.update(_sha_config_paths(item, path=child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{path}.{index}" if path else str(index)
            output.update(_sha_config_paths(item, path=child))
    return output


@dataclass(frozen=True)
class VerifiedBundle:
    mode: str
    manifest_sha256: str
    request_path: Path
    block_path: Path
    artifact_hashes: Mapping[str, str]


def _contained_file(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ProtocolError("artifact path must be a non-empty relative string")
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise ProtocolError(f"unsafe artifact path: {relative}")
    cursor = root
    for part in raw.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ProtocolError(f"symlink is forbidden in raw bundle: {relative}")
    resolved = cursor.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ProtocolError(f"artifact escapes bundle root or is not a file: {relative}")
    return resolved


def verify_bundle(
    manifest_path: str | Path,
    *,
    config: Mapping[str, Any],
    config_path: str | Path,
) -> VerifiedBundle:
    on_disk_config = strict_json_file(config_path)
    if on_disk_config != config:
        raise ProtocolError("in-memory config differs from the hash-bound config file")
    raw_manifest_file = Path(manifest_path)
    if raw_manifest_file.is_symlink():
        raise ProtocolError("raw bundle manifest cannot be a symlink")
    manifest_file = raw_manifest_file.resolve(strict=True)
    root = manifest_file.parent.resolve()
    parsed = strict_json_file(manifest_file)
    if not isinstance(parsed, Mapping):
        raise ProtocolError("raw bundle manifest must be an object")
    _exact_fields(parsed, MANIFEST_FIELDS, "raw bundle manifest")
    if parsed.get("schema") != "routeshield-raw-bundle-v1":
        raise ProtocolError("unsupported raw bundle schema")
    mode = _text(parsed, "mode")
    if mode not in {"DEVELOPMENT", "FORMAL"}:
        raise ProtocolError("raw bundle mode must be DEVELOPMENT or FORMAL")
    if _sha(parsed, "config_sha256") != sha256_file(config_path):
        raise ProtocolError("raw bundle config hash mismatch")
    if _sha(parsed, "evaluator_source_sha256") != sha256_file(__file__):
        raise ProtocolError("raw bundle evaluator source hash mismatch")
    artifacts = parsed.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) < {"requests", "blocks"}:
        raise ProtocolError("raw bundle must list requests and blocks artifacts")

    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    config_bindings: dict[str, str] = {}
    for name, raw_entry in artifacts.items():
        if not isinstance(name, str) or not isinstance(raw_entry, Mapping):
            raise ProtocolError("artifact entries must be named objects")
        _exact_fields(raw_entry, ARTIFACT_FIELDS, f"artifact {name}")
        path = _contained_file(root, raw_entry.get("path"))
        digest = _sha(raw_entry, "sha256")
        if sha256_file(path) != digest:
            raise ProtocolError(f"artifact hash mismatch: {name}")
        if path.stat().st_size != _integer(raw_entry, "size_bytes"):
            raise ProtocolError(f"artifact size mismatch: {name}")
        artifact_format = _text(raw_entry, "format")
        if artifact_format == "jsonl":
            row_count = len(strict_jsonl(path))
        elif artifact_format == "json":
            strict_json_file(path)
            row_count = 1
        elif artifact_format == "binary":
            row_count = 0
        else:
            raise ProtocolError(f"unsupported artifact format: {artifact_format}")
        if row_count != _integer(raw_entry, "row_count"):
            raise ProtocolError(f"artifact row count mismatch: {name}")
        config_key = _text(raw_entry, "config_key")
        if not config_key.endswith("_sha256"):
            raise ProtocolError("artifact config_key must name a frozen SHA-256 field")
        if config_key in config_bindings:
            raise ProtocolError(f"duplicate config hash binding: {config_key}")
        config_bindings[config_key] = digest
        expected = _config_lookup(config, config_key)
        if mode == "FORMAL" and expected != digest:
            raise ProtocolError(f"formal artifact differs from config binding: {config_key}")
        paths[name] = path
        hashes[name] = digest

    allowed_files = {manifest_file, *paths.values()}
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ProtocolError(f"raw bundle contains a symlink: {candidate}")
        if candidate.is_file() and candidate.resolve() not in allowed_files:
            raise ProtocolError(f"raw bundle contains an unlisted file: {candidate}")
    if len(set(paths.values())) != len(paths):
        raise ProtocolError("multiple artifact names bind the same file")

    if mode == "FORMAL":
        expected_bindings = _sha_config_paths(config)
        if set(config_bindings) != expected_bindings:
            missing = sorted(expected_bindings - set(config_bindings))
            extra = sorted(set(config_bindings) - expected_bindings)
            raise ProtocolError(
                f"formal bundle config bindings do not close; missing={missing}, extra={extra}"
            )
    expected_schemas = {
        "requests": "routeshield-raw-request-v1",
        "blocks": "routeshield-raw-block-v1",
    }
    for name, expected_schema in expected_schemas.items():
        if artifacts[name].get("schema") != expected_schema:
            raise ProtocolError(f"{name} artifact schema mismatch")
        if artifacts[name].get("format") != "jsonl":
            raise ProtocolError(f"{name} artifact must use JSONL format")
    return VerifiedBundle(
        mode=mode,
        manifest_sha256=sha256_file(manifest_file),
        request_path=paths["requests"],
        block_path=paths["blocks"],
        artifact_hashes=hashes,
    )


def _expected_scenarios(traffic_class: str) -> tuple[str, ...]:
    return PRIMARY_SCENARIOS if traffic_class == "ADV_TEXT" else CONTROL_SCENARIOS


def _policy_for_scenario(
    scenario: str, *, model: str, config: Mapping[str, Any]
) -> str:
    if scenario in {"MATCHED_BENIGN", "ATTACK_BASELINE", "CONTROL_DEFAULT"}:
        return "production_default_fcfs_chunked_prefill"
    if scenario == "LEGAL_ORACLE":
        return "future_known_exact_legal_oracle"
    selected = str(
        config["baseline_selection"]["frozen_strongest_simple_by_model"][model]
    )
    return selected


def _request_signature(row: RawRequest, *, include_prompt: bool) -> tuple[object, ...]:
    signature: tuple[object, ...] = (
        row.pair_id,
        row.tenant_id,
        row.request_id,
        row.input_tokens,
        row.max_new_tokens,
        row.arrival_ns,
    )
    if include_prompt:
        signature += (row.prompt_hash, row.document_id, row.document_cluster_id)
    return signature


def _validate_pairing(
    *,
    config: Mapping[str, Any],
    requests: Sequence[RawRequest],
    blocks: Sequence[RawBlock],
) -> dict[tuple[str, str, str], list[str]]:
    models = {str(row["key"]) for row in config["models"]}
    block_map: dict[tuple[str, str, str, str, str], RawBlock] = {}
    request_map: dict[tuple[str, str, str, str, str], list[RawRequest]] = {}
    for block in blocks:
        key = (
            block.model,
            block.load_cell,
            block.traffic_class,
            block.block_id,
            block.scenario,
        )
        if key in block_map:
            raise ProtocolError(f"duplicate block/scenario identity: {key}")
        block_map[key] = block
    seen_requests: set[tuple[str, str, str, str, str, str]] = set()
    for row in requests:
        identity = (
            row.model,
            row.load_cell,
            row.traffic_class,
            row.block_id,
            row.scenario,
            row.pair_id,
        )
        if identity in seen_requests:
            raise ProtocolError(f"duplicate paired request identity: {identity}")
        seen_requests.add(identity)
        key = identity[:5]
        if key not in block_map:
            raise ProtocolError(f"request row has no matching block row: {key}")
        block = block_map[key]
        if not block.window_start_ns <= row.arrival_ns < block.window_end_ns:
            raise ProtocolError("request arrival falls outside its block window")
        if row.completion_ns is not None and row.completion_ns > block.window_end_ns:
            raise ProtocolError("request completion falls outside its block window")
        request_map.setdefault(key, []).append(row)
    if set(request_map) != set(block_map):
        raise ProtocolError("a block/scenario contains no request rows")

    cell_blocks: dict[tuple[str, str, str], set[str]] = {}
    for model in models:
        for load_cell, traffic_class in CELLS:
            cell = (model, load_cell, traffic_class)
            block_ids = {
                key[3]
                for key in block_map
                if key[:3] == cell
            }
            if not block_ids:
                raise ProtocolError(f"missing frozen raw cell: {cell}")
            expected = set(_expected_scenarios(traffic_class))
            for block_id in block_ids:
                observed = {
                    key[4]
                    for key in block_map
                    if key[:4] == (*cell, block_id)
                }
                if observed != expected:
                    raise ProtocolError(
                        f"paired block {(*cell, block_id)} scenarios {observed} != {expected}"
                    )
            cell_blocks[cell] = block_ids
    unexpected_cells = {
        key[:3] for key in block_map if key[:3] not in set(cell_blocks)
    }
    if unexpected_cells:
        raise ProtocolError(f"raw bundle contains unexpected cells: {unexpected_cells}")

    for cell, block_ids in cell_blocks.items():
        model, _, traffic_class = cell
        expected_scenarios = _expected_scenarios(traffic_class)
        cluster_owner: dict[str, str] = {}
        for block_id in block_ids:
            block_rows = {
                scenario: block_map[(*cell, block_id, scenario)]
                for scenario in expected_scenarios
            }
            reference = block_rows[expected_scenarios[0]]
            for scenario, block in block_rows.items():
                if (
                    block.arrival_trace_sha256 != reference.arrival_trace_sha256
                    or block.victim_manifest_sha256 != reference.victim_manifest_sha256
                    or block.cotenant_budget_sha256 != reference.cotenant_budget_sha256
                    or block.window_start_ns != reference.window_start_ns
                ):
                    raise ProtocolError("paired block provenance/window mismatch")
                expected_policy = _policy_for_scenario(
                    scenario, model=model, config=config
                )
                if expected_policy.startswith("UNRESOLVED_"):
                    raise ProtocolError("strongest-simple baseline selection is unresolved")
                if scenario in {"STRONGEST_SIMPLE", "CONTROL_ISOLATION"} and (
                    expected_policy not in config["baselines"]
                ):
                    raise ProtocolError("selected strongest-simple policy is unregistered")
                if block.policy_id != expected_policy:
                    raise ProtocolError(
                        f"policy mismatch for {scenario}: {block.policy_id} != {expected_policy}"
                    )
            if traffic_class == "ADV_TEXT":
                worlds = {
                    block_rows[scenario].request_world_sha256
                    for scenario in ("ATTACK_BASELINE", "LEGAL_ORACLE", "STRONGEST_SIMPLE")
                }
                if len(worlds) != 1:
                    raise ProtocolError("A/O/S request worlds differ")
                statuses = {block_rows["LEGAL_ORACLE"].oracle_status}
                if statuses & {"TIMEOUT", "STATE_LIMIT"}:
                    raise ProtocolError("UNSOLVED_EXACT_STATE_LIMIT")
            else:
                if len({block.request_world_sha256 for block in block_rows.values()}) != 1:
                    raise ProtocolError("control default/isolation request worlds differ")

            scenario_rows = {
                scenario: request_map[(*cell, block_id, scenario)]
                for scenario in expected_scenarios
            }
            for scenario, rows in scenario_rows.items():
                expected_nonvictim_role = (
                    "cotenant"
                    if scenario
                    in {"MATCHED_BENIGN", "CONTROL_DEFAULT", "CONTROL_ISOLATION"}
                    else "attacker"
                )
                if any(
                    row.role not in {"victim", expected_nonvictim_role}
                    for row in rows
                ):
                    raise ProtocolError(
                        f"scenario {scenario} contains an invalid tenant role"
                    )
                victim_tenants = {
                    row.tenant_id for row in rows if row.role == "victim"
                }
                nonvictim_tenants = {
                    row.tenant_id for row in rows if row.role != "victim"
                }
                if (
                    len(victim_tenants) != 1
                    or len(nonvictim_tenants) != 1
                    or not victim_tenants.isdisjoint(nonvictim_tenants)
                ):
                    raise ProtocolError(
                        "paired blocks require one victim tenant and one cotenant/attacker tenant"
                    )
                block = block_rows[scenario]
                derived = request_provenance_hashes(rows)
                for key, observed_hash in derived.items():
                    if getattr(block, key) != observed_hash:
                        raise ProtocolError(
                            f"block {key} differs from recomputed request provenance"
                        )
            pair_sets = {scenario: {row.pair_id for row in rows} for scenario, rows in scenario_rows.items()}
            if len({frozenset(value) for value in pair_sets.values()}) != 1:
                raise ProtocolError("paired request sets differ across scenarios")
            for scenario, rows in scenario_rows.items():
                if any(row.terminal_reason != "COMPLETED" for row in rows):
                    raise ProtocolError("CENSORED_REQUEST")
                if any(row.max_new_tokens != 1 for row in rows):
                    raise ProtocolError("raw request violates frozen max_new_tokens=1")
                by_pair = {row.pair_id: row for row in rows}
                for pair_id in pair_sets[scenario]:
                    candidate = by_pair[pair_id]
                    reference_row = {
                        row.pair_id: row for row in scenario_rows[expected_scenarios[0]]
                    }[pair_id]
                    include_prompt = candidate.role == "victim" or traffic_class != "ADV_TEXT"
                    if _request_signature(candidate, include_prompt=include_prompt) != _request_signature(
                        reference_row, include_prompt=include_prompt
                    ):
                        raise ProtocolError("paired request budget/arrival/prompt mismatch")
                    if candidate.role == "victim":
                        if (
                            reference_row.role != "victim"
                            or candidate.output_hash != reference_row.output_hash
                            or candidate.output_token_count
                            != reference_row.output_token_count
                        ):
                            raise ProtocolError("victim completion/output set mismatch")
                    if traffic_class != "ADV_TEXT" and candidate.role != reference_row.role:
                        raise ProtocolError("control request roles differ")
                    if traffic_class == "ADV_TEXT" and scenario != "MATCHED_BENIGN":
                        attack_reference = {
                            row.pair_id: row
                            for row in scenario_rows["ATTACK_BASELINE"]
                        }[pair_id]
                        if candidate.role != attack_reference.role:
                            raise ProtocolError("A/O/S roles differ")
                        if candidate.role == "attacker" and (
                            candidate.prompt_hash != attack_reference.prompt_hash
                            or candidate.output_hash != attack_reference.output_hash
                            or candidate.output_token_count
                            != attack_reference.output_token_count
                        ):
                            raise ProtocolError("A/O/S attacker work or output differs")
                for row in rows:
                    if row.role == "victim":
                        owner = cluster_owner.setdefault(row.document_cluster_id, block_id)
                        if owner != block_id:
                            raise ProtocolError(
                                "victim document cluster crosses paired blocks"
                            )
    return {cell: sorted(block_ids) for cell, block_ids in cell_blocks.items()}


def _queue_stable(
    blocks: Sequence[RawBlock], *, tolerance: float
) -> bool:
    growth = sum(block.queue_service_work_end - block.queue_service_work_start for block in blocks)
    arrived = sum(block.queue_service_work_arrived for block in blocks)
    return growth <= tolerance * arrived


def recompute_raw_gate(
    config: Mapping[str, Any],
    requests: Iterable[RawRequest],
    blocks: Iterable[RawBlock],
    *,
    allow_small_fixture: bool = False,
) -> dict[str, object]:
    try:
        request_rows = [RawRequest.from_mapping(asdict(row)) for row in requests]
        block_rows = [RawBlock.from_mapping(asdict(row)) for row in blocks]
    except TypeError as exc:
        raise ProtocolError("raw recompute inputs must use the frozen dataclasses") from exc
    cell_blocks = _validate_pairing(
        config=config, requests=request_rows, blocks=block_rows
    )
    min_blocks = int(config["workloads"]["formal_p99_min_paired_blocks_per_cell"])
    min_requests = int(
        config["workloads"]["formal_p99_min_completed_victim_requests_per_cell"]
    )
    if allow_small_fixture:
        min_blocks = min(min_blocks, 2)
        min_requests = min(min_requests, 4)
    sample_shortfalls: list[str] = []
    for cell, block_ids in cell_blocks.items():
        if len(block_ids) < min_blocks:
            sample_shortfalls.append(
                f"{cell}: {len(block_ids)} blocks < required {min_blocks}"
            )
        for scenario in _expected_scenarios(cell[2]):
            count = sum(
                row.role == "victim"
                for row in request_rows
                if (row.model, row.load_cell, row.traffic_class) == cell
                and row.scenario == scenario
            )
            if count < min_requests:
                sample_shortfalls.append(
                    f"{(*cell, scenario)}: {count} victims < required {min_requests}"
                )
    if sample_shortfalls:
        return {
            "schema": "routeshield-raw-recompute-result-v1",
            "status": "BLOCKED_INSUFFICIENT_RAW_SAMPLE",
            "formal_result": False,
            "reason_codes": sample_shortfalls,
        }

    tolerance = float(config["statistics"]["queue_growth_max_fraction"])
    unstable = []
    for cell in cell_blocks:
        for scenario in _expected_scenarios(cell[2]):
            selected = [
                block
                for block in block_rows
                if (block.model, block.load_cell, block.traffic_class) == cell
                and block.scenario == scenario
            ]
            if not _queue_stable(selected, tolerance=tolerance):
                unstable.append((*cell, scenario))
    if unstable:
        return {
            "schema": "routeshield-raw-recompute-result-v1",
            "status": "INVALID_REQUEST_DAG",
            "formal_result": False,
            "reason_codes": [f"QUEUE_UNSTABLE:{cell}" for cell in unstable],
        }

    replicates = int(config["statistics"]["bootstrap_resamples"])
    if allow_small_fixture:
        replicates = min(replicates, 200)
    base_seed = int(config["statistics"]["bootstrap_seed"])
    raw_metrics: list[dict[str, object]] = []
    all_thresholds_pass = True
    threshold_failures: list[str] = []
    thresholds = config["statistics"]["thresholds"]

    for cell, block_ids in sorted(cell_blocks.items()):
        model, load_cell, traffic_class = cell
        scenario_names = _expected_scenarios(traffic_class)
        victim_values: dict[str, dict[str, list[int]]] = {
            scenario: {block_id: [] for block_id in block_ids}
            for scenario in scenario_names
        }
        goodput_numerators: dict[str, list[float]] = {scenario: [] for scenario in scenario_names}
        durations: dict[str, list[float]] = {scenario: [] for scenario in scenario_names}
        for scenario in scenario_names:
            for block_id in block_ids:
                selected_requests = [
                    row
                    for row in request_rows
                    if (row.model, row.load_cell, row.traffic_class, row.block_id, row.scenario)
                    == (*cell, block_id, scenario)
                ]
                victim_values[scenario][block_id] = [
                    row.ttft_ns for row in selected_requests if row.role == "victim"
                ]
                goodput_numerators[scenario].append(
                    float(sum(row.input_tokens for row in selected_requests))
                )
                block = next(
                    item
                    for item in block_rows
                    if (item.model, item.load_cell, item.traffic_class, item.block_id, item.scenario)
                    == (*cell, block_id, scenario)
                )
                durations[scenario].append(
                    _observed_wall_clock_ns(selected_requests)
                )

        indices = {
            scenario: WeightedNearestRankP99.build(victim_values[scenario], block_ids)
            for scenario in scenario_names
        }
        ones = [1] * len(block_ids)
        points = {scenario: index.evaluate(ones) for scenario, index in indices.items()}
        seed = _stable_cell_seed(base_seed, model, f"{load_cell}|{traffic_class}")

        if traffic_class == "ADV_TEXT":
            b, a, o, s = (points[scenario] for scenario in PRIMARY_SCENARIOS)
            if o > a or o > s:
                return {
                    "schema": "routeshield-raw-recompute-result-v1",
                    "status": "INVALID_REQUEST_DAG",
                    "formal_result": False,
                    "reason_codes": ["ORACLE_OBJECTIVE_OR_CERTIFICATE_MISMATCH"],
                }
            bootstrap_harm: list[float] = []
            bootstrap_gain: list[float] = []
            bootstrap_recovery: list[float] = []
            bootstrap_capture: list[float] = []
            unstable_harm = False
            unstable_oracle = False
            for multiplicities in _bootstrap_multiplicities(
                len(block_ids), replicates=replicates, seed=seed
            ):
                rb, ra, ro, rs = (
                    indices[scenario].evaluate(multiplicities)
                    for scenario in PRIMARY_SCENARIOS
                )
                bootstrap_harm.append(ra / rb - 1.0)
                bootstrap_gain.append((ra - ro) / ra)
                if ra <= rb:
                    unstable_harm = True
                    bootstrap_recovery.append(-1e12)
                else:
                    bootstrap_recovery.append((ra - ro) / (ra - rb))
                if ra <= ro:
                    unstable_oracle = True
                    bootstrap_capture.append(1e12)
                else:
                    bootstrap_capture.append((ra - rs) / (ra - ro))
            harm_point = a / b - 1.0
            gain_point = (a - o) / a
            recovery_point = (a - o) / (a - b) if a > b else -1e12
            capture_point = (a - s) / (a - o) if a > o else 1e12
            metric = {
                "model": model,
                "load_cell": load_cell,
                "traffic_class": traffic_class,
                "block_count": len(block_ids),
                "victim_requests_per_arm": {
                    scenario: sum(indices[scenario].totals_by_block)
                    for scenario in scenario_names
                },
                "p99_ttft_ns": points,
                "harm_point": harm_point,
                "harm_lcb": _linear_quantile(bootstrap_harm, 0.025),
                "oracle_gain_point": gain_point,
                "oracle_gain_lcb": _linear_quantile(bootstrap_gain, 0.025),
                "oracle_recovery_point": recovery_point,
                "oracle_recovery_lcb": _linear_quantile(bootstrap_recovery, 0.025),
                "simple_capture_point": capture_point,
                "simple_capture_ucb": _linear_quantile(bootstrap_capture, 0.975),
                "bootstrap_positive_harm_all": not unstable_harm,
                "bootstrap_positive_oracle_headroom_all": not unstable_oracle,
                "bootstrap_seed": seed,
                "bootstrap_resamples": replicates,
            }
            failures = []
            if (
                unstable_harm
                or harm_point < float(thresholds["harm_point_min"])
                or metric["harm_lcb"] <= float(thresholds["harm_lcb_strict_min"])
            ):
                failures.append("PHENOMENON_THRESHOLD_FAIL")
            if (
                unstable_oracle
                or gain_point < float(thresholds["oracle_gain_point_min"])
                or metric["oracle_gain_lcb"]
                <= float(thresholds["oracle_gain_lcb_strict_min"])
                or metric["oracle_recovery_lcb"]
                < float(thresholds["oracle_recovery_lcb_min"])
            ):
                failures.append("ORACLE_THRESHOLD_FAIL")
            if metric["simple_capture_ucb"] >= float(
                thresholds["simple_capture_ucb_strict_max"]
            ):
                failures.append("SIMPLE_CAPTURE_THRESHOLD_FAIL")
            if failures:
                all_thresholds_pass = False
                threshold_failures.extend(f"{model}:{failure}" for failure in failures)
            metric["threshold_failures"] = failures
            raw_metrics.append(metric)
        else:
            default, isolated = CONTROL_SCENARIOS
            point_default = _ratio_of_sums(
                goodput_numerators[default], durations[default], ones
            )
            point_isolated = _ratio_of_sums(
                goodput_numerators[isolated], durations[isolated], ones
            )
            point_loss = 1.0 - point_isolated / point_default
            bootstrap_losses = []
            for multiplicities in _bootstrap_multiplicities(
                len(block_ids), replicates=replicates, seed=seed
            ):
                default_goodput = _ratio_of_sums(
                    goodput_numerators[default], durations[default], multiplicities
                )
                isolated_goodput = _ratio_of_sums(
                    goodput_numerators[isolated], durations[isolated], multiplicities
                )
                bootstrap_losses.append(1.0 - isolated_goodput / default_goodput)
            loss_ucb = _linear_quantile(bootstrap_losses, 0.975)
            failures = []
            if loss_ucb >= float(thresholds["benign_goodput_loss_ucb_strict_max"]):
                failures.append("CONTROL_TAX_THRESHOLD_FAIL")
                all_thresholds_pass = False
                threshold_failures.append(f"{model}:CONTROL_TAX_THRESHOLD_FAIL")
            raw_metrics.append(
                {
                    "model": model,
                    "load_cell": load_cell,
                    "traffic_class": traffic_class,
                    "block_count": len(block_ids),
                    "goodput_default": point_default,
                    "goodput_isolation": point_isolated,
                    "benign_goodput_loss_point": point_loss,
                    "benign_goodput_loss_ucb": loss_ucb,
                    "bootstrap_seed": seed,
                    "bootstrap_resamples": replicates,
                    "threshold_failures": failures,
                }
            )

    return {
        "schema": "routeshield-raw-recompute-result-v1",
        "status": "RAW_RECOMPUTE_DIAGNOSTIC_ONLY",
        "formal_result": False,
        "diagnostic_threshold_branch": (
            "ALL_THRESHOLDS_PASS" if all_thresholds_pass else "THRESHOLD_FAILURES_PRESENT"
        ),
        "threshold_failures": sorted(set(threshold_failures)),
        "metrics": raw_metrics,
        "evaluator_version": RAW_EVALUATOR_VERSION,
        "evidence_boundary": (
            "Raw request/block recomputation only. Full-DAG, executed-dispatch, "
            "tensor exactness, and Oracle-certificate validation remain locked blockers."
        ),
    }


def recompute_bundle(
    manifest_path: str | Path,
    *,
    config: Mapping[str, Any],
    config_path: str | Path,
    allow_small_fixture: bool = False,
) -> dict[str, object]:
    bundle = verify_bundle(
        manifest_path, config=config, config_path=config_path
    )
    if bundle.mode == "FORMAL":
        return {
            "schema": "routeshield-raw-recompute-result-v1",
            "status": "BLOCKED_FORMAL_RAW_EVALUATOR_NOT_APPROVED",
            "formal_result": False,
            "manifest_sha256": bundle.manifest_sha256,
            "reason_codes": ["FORMAL_RAW_EVALUATOR_IMPLEMENTED=false"],
        }
    payload = recompute_raw_gate(
        config,
        load_requests(bundle.request_path),
        load_blocks(bundle.block_path),
        allow_small_fixture=allow_small_fixture,
    )
    return {
        **payload,
        "manifest_sha256": bundle.manifest_sha256,
        "artifact_hashes": dict(bundle.artifact_hashes),
    }


def rows_as_jsonl(rows: Iterable[RawRequest | RawBlock]) -> str:
    return "".join(
        json.dumps(asdict(row), sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
