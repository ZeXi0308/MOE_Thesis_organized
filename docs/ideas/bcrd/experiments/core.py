from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Iterable, Mapping, Sequence


class ProtocolError(RuntimeError):
    """An artifact violates the frozen experiment contract."""


ROUTE_COLUMNS = (
    "model",
    "phase",
    "request_id",
    "sample_id",
    "arrival_us",
    "deadline_us",
    "layer",
    "token_position",
    "rank",
    "expert_id",
    "gate_weight",
    "src_replica",
)


@dataclass(frozen=True)
class Contribution:
    model: str
    phase: str
    request_id: str
    sample_id: int
    arrival_us: float
    deadline_us: float
    layer: int
    token_position: int
    rank: int
    expert_id: int
    gate_weight: float
    src_replica: int = 0

    def __post_init__(self) -> None:
        if not self.model or not self.phase or not self.request_id:
            raise ValueError("model, phase and request_id must be non-empty")
        for name in ("sample_id", "layer", "token_position", "expert_id", "src_replica"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("rank must be a positive integer")
        if not math.isfinite(self.arrival_us) or self.arrival_us < 0:
            raise ValueError("arrival_us must be finite and non-negative")
        if not math.isfinite(self.deadline_us) or self.deadline_us < self.arrival_us:
            raise ValueError("deadline_us must be finite and >= arrival_us")
        if not math.isfinite(self.gate_weight) or self.gate_weight < 0:
            raise ValueError("gate_weight must be finite and non-negative")

    @property
    def contribution_id(self) -> str:
        return (
            f"{self.model}|{self.phase}|{self.request_id}|{self.layer}|"
            f"{self.token_position}|{self.rank}|{self.expert_id}"
        )

    def to_json(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "Contribution":
        try:
            return cls(
                model=str(row["model"]),
                phase=str(row["phase"]),
                request_id=str(row["request_id"]),
                sample_id=int(row["sample_id"]),
                arrival_us=float(row["arrival_us"]),
                deadline_us=float(row["deadline_us"]),
                layer=int(row["layer"]),
                token_position=int(row["token_position"]),
                rank=int(row["rank"]),
                expert_id=int(row["expert_id"]),
                gate_weight=float(row["gate_weight"]),
                src_replica=int(row.get("src_replica", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(f"invalid route row: {exc}") from exc


def load_routes(paths: Sequence[str | Path]) -> list[Contribution]:
    rows: list[Contribution] = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = set(ROUTE_COLUMNS) - set(reader.fieldnames or ())
            if missing:
                raise ProtocolError(f"{path}: missing route columns {sorted(missing)}")
            rows.extend(Contribution.from_mapping(row) for row in reader)
    validate_identity_conservation(rows)
    return rows


def write_routes(path: str | Path, rows: Sequence[Contribution]) -> None:
    validate_identity_conservation(rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROUTE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(row.to_json() for row in rows)


def validate_identity_conservation(rows: Sequence[Contribution]) -> dict[str, int]:
    if not rows:
        raise ProtocolError("route trace is empty")
    identities = [row.contribution_id for row in rows]
    if len(identities) != len(set(identities)):
        raise ProtocolError("duplicate routed contribution identity")
    by_token: dict[tuple[str, str, str, int, int], list[Contribution]] = {}
    for row in rows:
        key = (row.model, row.phase, row.request_id, row.layer, row.token_position)
        by_token.setdefault(key, []).append(row)
    for key, token_rows in by_token.items():
        ranks = sorted(row.rank for row in token_rows)
        if ranks != list(range(1, len(ranks) + 1)):
            raise ProtocolError(f"non-contiguous top-k ranks for token {key}: {ranks}")
        experts = [row.expert_id for row in token_rows]
        if len(experts) != len(set(experts)):
            raise ProtocolError(f"duplicate expert for token {key}")
    return {
        "contributions": len(rows),
        "tokens": len(by_token),
        "requests": len({(row.model, row.phase, row.request_id) for row in rows}),
    }


@dataclass(frozen=True)
class CurvePoint:
    rows: int
    median_us: float
    p95_us: float

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.median_us <= 0 or self.p95_us < self.median_us:
            raise ValueError("invalid service-curve point")


class ServiceCatalog:
    """Fail-closed interpolation of measured expert service curves."""

    def __init__(self, points: Mapping[tuple[str, int], Sequence[CurvePoint]]) -> None:
        self._points: dict[tuple[str, int], tuple[CurvePoint, ...]] = {}
        for key, values in points.items():
            ordered = tuple(sorted(values, key=lambda point: point.rows))
            if not ordered or len({point.rows for point in ordered}) != len(ordered):
                raise ProtocolError(f"invalid or duplicate curve rows for {key}")
            if any(b.median_us < a.median_us for a, b in zip(ordered, ordered[1:])):
                raise ProtocolError(f"non-monotone median curve for {key}")
            self._points[key] = ordered
        if not self._points:
            raise ProtocolError("service catalog is empty")

    @classmethod
    def from_csv(cls, path: str | Path) -> "ServiceCatalog":
        grouped: dict[tuple[str, int], list[CurvePoint]] = {}
        with Path(path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"model", "layer", "rows", "median_us", "p95_us"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ProtocolError(f"service curve missing columns {sorted(missing)}")
            for row in reader:
                key = (str(row["model"]), int(row["layer"]))
                grouped.setdefault(key, []).append(
                    CurvePoint(int(row["rows"]), float(row["median_us"]), float(row["p95_us"]))
                )
        return cls(grouped)

    def keys(self) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(self._points))

    def estimate_us(self, model: str, layer: int, rows: int, *, conservative: bool = False) -> float:
        if rows <= 0:
            raise ValueError("rows must be positive")
        points = self._points.get((model, layer)) or self._points.get((model, -1))
        if points is None:
            raise ProtocolError(f"no service curve for model={model!r}, layer={layer}")
        if rows < points[0].rows or rows > points[-1].rows:
            raise ProtocolError(
                f"rows={rows} outside measured curve [{points[0].rows},{points[-1].rows}] "
                f"for model={model!r}, layer={layer}"
            )
        field = "p95_us" if conservative else "median_us"
        for point in points:
            if rows == point.rows:
                return float(getattr(point, field))
        for left, right in zip(points, points[1:]):
            if left.rows < rows < right.rows:
                ratio = (rows - left.rows) / (right.rows - left.rows)
                return float(getattr(left, field)) + ratio * (
                    float(getattr(right, field)) - float(getattr(left, field))
                )
        raise AssertionError("in-range interpolation failed")


def stable_index(material: str, modulo: int, *, seed: int = 0) -> int:
    if modulo <= 0:
        raise ValueError("modulo must be positive")
    digest = hashlib.sha256(f"{seed}|{material}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def percentile(values: Sequence[float], q: float) -> float:
    if not values or not 0 <= q <= 1:
        raise ValueError("percentile needs non-empty values and q in [0,1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])


def bootstrap_mean_ci(
    values: Sequence[float], *, replicates: int = 2000, seed: int = 20260725
) -> tuple[float, float, float]:
    if not values:
        raise ValueError("bootstrap values cannot be empty")
    point = sum(values) / len(values)
    if len(values) == 1 or replicates <= 0:
        return point, point, point
    rng = random.Random(seed)
    means = [
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(replicates)
    ]
    return point, percentile(means, 0.025), percentile(means, 0.975)


def clustered_bootstrap_mean_ci(
    values: Sequence[float],
    cluster_ids: Sequence[str],
    *,
    replicates: int = 2000,
    seed: int = 20260725,
) -> tuple[float, float, float]:
    if len(values) != len(cluster_ids) or not values:
        raise ValueError("clustered bootstrap needs equal non-empty values and cluster ids")
    grouped: dict[str, list[float]] = {}
    for value, cluster_id in zip(values, cluster_ids):
        grouped.setdefault(cluster_id, []).append(float(value))
    cluster_means = [sum(items) / len(items) for _, items in sorted(grouped.items())]
    return bootstrap_mean_ci(cluster_means, replicates=replicates, seed=seed)


@dataclass(frozen=True)
class ReplayConfig:
    replica_count: int
    hold_us: float = 0.0
    remote_latency_us: float = 0.0
    remote_bytes_per_row: int = 0
    conservative_curve: bool = False

    def __post_init__(self) -> None:
        if self.replica_count < 2:
            raise ValueError("BCRD requires at least two replicas")
        if self.hold_us < 0 or self.remote_latency_us < 0 or self.remote_bytes_per_row < 0:
            raise ValueError("replay costs must be non-negative")


def _make_batches(
    contributions: Sequence[Contribution], assignments: Sequence[int], config: ReplayConfig
) -> dict[int, list[dict[str, object]]]:
    grouped: dict[tuple[int, int], list[tuple[int, Contribution, float]]] = {}
    for index, (item, replica) in enumerate(zip(contributions, assignments)):
        if replica < 0 or replica >= config.replica_count:
            raise ProtocolError(f"illegal replica {replica} for contribution {index}")
        ready = item.arrival_us + (config.remote_latency_us if item.src_replica != replica else 0.0)
        grouped.setdefault((replica, item.expert_id), []).append((index, item, ready))

    by_replica: dict[int, list[dict[str, object]]] = {r: [] for r in range(config.replica_count)}
    for (replica, expert), items in grouped.items():
        items.sort(key=lambda value: (value[2], value[1].deadline_us, value[0]))
        cursor = 0
        while cursor < len(items):
            first_ready = items[cursor][2]
            seal = first_ready + config.hold_us
            end = cursor + 1
            while end < len(items) and items[end][2] <= seal + 1e-12:
                end += 1
            chunk = items[cursor:end]
            by_replica[replica].append(
                {
                    "expert_id": expert,
                    "indices": tuple(value[0] for value in chunk),
                    "ready_us": max(value[2] for value in chunk),
                    "deadline_us": min(value[1].deadline_us for value in chunk),
                    "rows": len(chunk),
                }
            )
            cursor = end
    return by_replica


def simulate_assignment(
    contributions: Sequence[Contribution],
    assignments: Sequence[int],
    catalog: ServiceCatalog,
    config: ReplayConfig,
) -> dict[str, object]:
    """Replay one layer with per-replica EDF batches and request fork-join."""
    if not contributions or len(contributions) != len(assignments):
        raise ProtocolError("assignment must cover every contribution exactly once")
    model = contributions[0].model
    layer = contributions[0].layer
    if any(item.model != model or item.layer != layer for item in contributions):
        raise ProtocolError("one replay instance must contain one model and one layer")
    batches = _make_batches(contributions, assignments, config)
    completion: dict[int, float] = {}
    launches = 0
    total_service = 0.0
    for replica, pending in batches.items():
        now = min((float(batch["ready_us"]) for batch in pending), default=0.0)
        queue = list(pending)
        while queue:
            ready = [batch for batch in queue if float(batch["ready_us"]) <= now + 1e-12]
            if not ready:
                now = min(float(batch["ready_us"]) for batch in queue)
                continue
            batch = min(
                ready,
                key=lambda value: (
                    float(value["deadline_us"]),
                    float(value["ready_us"]),
                    int(value["expert_id"]),
                ),
            )
            duration = catalog.estimate_us(
                model, layer, int(batch["rows"]), conservative=config.conservative_curve
            )
            now += duration
            total_service += duration
            launches += 1
            for index in batch["indices"]:  # type: ignore[union-attr]
                completion[int(index)] = now
            queue.remove(batch)
    if len(completion) != len(contributions):
        raise AssertionError("replay lost contributions")

    request_completion: dict[str, float] = {}
    request_arrival: dict[str, float] = {}
    request_deadline: dict[str, float] = {}
    for index, item in enumerate(contributions):
        request_completion[item.request_id] = max(
            request_completion.get(item.request_id, -math.inf), completion[index]
        )
        request_arrival[item.request_id] = min(
            request_arrival.get(item.request_id, math.inf), item.arrival_us
        )
        request_deadline[item.request_id] = min(
            request_deadline.get(item.request_id, math.inf), item.deadline_us
        )
    latencies = [request_completion[key] - request_arrival[key] for key in request_completion]
    on_time = sum(request_completion[key] <= request_deadline[key] for key in request_completion)
    remote = sum(item.src_replica != replica for item, replica in zip(contributions, assignments))
    return {
        "requests": len(request_completion),
        "contributions": len(contributions),
        "on_time": on_time,
        "slo_attainment": on_time / len(request_completion),
        "mean_completion_us": sum(latencies) / len(latencies),
        "p50_completion_us": percentile(latencies, 0.50),
        "p95_completion_us": percentile(latencies, 0.95),
        "p99_completion_us": percentile(latencies, 0.99),
        "makespan_us": max(request_completion.values()) - min(request_arrival.values()),
        "total_service_us": total_service,
        "launches": launches,
        "remote_assignments": remote,
        "remote_bytes": remote * config.remote_bytes_per_row,
        "request_completion_us": request_completion,
    }


def objective_key(metrics: Mapping[str, object]) -> tuple[float, ...]:
    return (
        float(metrics["on_time"]),
        -float(metrics["p99_completion_us"]),
        -float(metrics["mean_completion_us"]),
        -float(metrics["total_service_us"]),
        -float(metrics["remote_assignments"]),
    )


def relative_latency_gain(baseline: float, candidate: float) -> float:
    if baseline <= 0 or candidate < 0 or not math.isfinite(baseline + candidate):
        raise ValueError("latency gain needs finite non-negative values and positive baseline")
    return (baseline - candidate) / baseline


def read_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_instances(path: str | Path) -> list[dict[str, object]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ProtocolError(f"{path}:{lineno}: invalid JSON") from exc
    if not rows:
        raise ProtocolError("instance file is empty")
    return rows


def write_jsonl(path: str | Path, rows: Iterable[object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
