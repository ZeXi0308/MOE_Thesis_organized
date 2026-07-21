"""Frozen P0-B primitives for RouteFidelity-EP.

This module deliberately contains no capture path and no command-line entry
point.  It provides small, deterministic reference implementations used to
validate route representations before any backend or latency claim is made.

Evidence boundary:

* C0/C1 values are logical record counts, not wire bytes or operator latency.
* S1-R preserves expert degree independently inside every request-layer group.
* S2/S3 helpers measure a documented canonical representation, not a backend
  serialization format.
* Cluster bootstrap resamples request/article clusters; it never treats tokens
  as independent observations.
"""


# --- shared-lib bootstrap (auto) ---
import sys
from pathlib import Path as _Path

def _ensure_shared_on_path() -> None:
    here = _Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        cand = p / "experiments" / "shared"
        if (cand / "capture_moe.py").exists():
            s = str(cand)
            if s not in sys.path:
                sys.path.insert(0, s)
            return
        if (p / "capture_moe.py").exists():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
            return

_ensure_shared_on_path()
del _ensure_shared_on_path, _Path
# --- end bootstrap ---

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
from hashlib import sha256
import math
from pathlib import Path
import struct
from typing import Callable, Hashable, Mapping, Sequence

import numpy as np


C0_EXPANDED = "C0_EXPANDED"
C1_UNIQUE_OWNER = "C1_UNIQUE_OWNER"


def _readonly_int_array(
    values: object, *, dtype: np.dtype, ndim: int, name: str
) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {array.shape}")
    array = np.array(array, dtype=dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class RouteIR:
    """Token-level route IR with explicit architecture and home-rank metadata."""

    experts: np.ndarray
    request_id: np.ndarray
    layer_id: np.ndarray
    token_position: np.ndarray
    home_rank: np.ndarray
    num_experts: int
    top_k: int
    home_rank_source: str = "observed"

    def __post_init__(self) -> None:
        if int(self.num_experts) <= 0:
            raise ValueError("num_experts must be positive and explicit")
        if int(self.top_k) <= 0:
            raise ValueError("top_k must be positive and explicit")
        if int(self.top_k) > int(self.num_experts):
            raise ValueError("top_k cannot exceed num_experts")

        experts = _readonly_int_array(
            self.experts, dtype=np.int32, ndim=2, name="experts"
        )
        request_id = _readonly_int_array(
            self.request_id, dtype=np.int64, ndim=1, name="request_id"
        )
        layer_id = _readonly_int_array(
            self.layer_id, dtype=np.int32, ndim=1, name="layer_id"
        )
        token_position = _readonly_int_array(
            self.token_position,
            dtype=np.int64,
            ndim=1,
            name="token_position",
        )
        home_rank = _readonly_int_array(
            self.home_rank, dtype=np.int32, ndim=1, name="home_rank"
        )

        rows = experts.shape[0]
        if experts.shape[1] != int(self.top_k):
            raise ValueError(
                f"experts has width {experts.shape[1]}, expected top_k={self.top_k}"
            )
        for name, array in (
            ("request_id", request_id),
            ("layer_id", layer_id),
            ("token_position", token_position),
            ("home_rank", home_rank),
        ):
            if len(array) != rows:
                raise ValueError(f"{name} has {len(array)} rows, expected {rows}")
        if rows == 0:
            raise ValueError("RouteIR cannot be empty")
        if np.any(experts < 0) or np.any(experts >= int(self.num_experts)):
            raise ValueError("expert id is outside explicit [0, num_experts) range")
        if np.any(layer_id < 0) or np.any(token_position < 0):
            raise ValueError("layer_id and token_position must be non-negative")
        if np.any(home_rank < 0):
            raise ValueError("home_rank must be non-negative")
        ordered = np.sort(experts, axis=1)
        if ordered.shape[1] > 1 and np.any(ordered[:, 1:] == ordered[:, :-1]):
            raise ValueError("each token route must contain top_k unique experts")
        if not str(self.home_rank_source).strip():
            raise ValueError("home_rank_source must describe observed or synthetic origin")

        object.__setattr__(self, "experts", experts)
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "layer_id", layer_id)
        object.__setattr__(self, "token_position", token_position)
        object.__setattr__(self, "home_rank", home_rank)
        object.__setattr__(self, "num_experts", int(self.num_experts))
        object.__setattr__(self, "top_k", int(self.top_k))

    @property
    def token_count(self) -> int:
        return int(self.experts.shape[0])

    def with_experts(self, experts: np.ndarray) -> "RouteIR":
        """Return an IR with changed routes and unchanged exogenous metadata."""

        return RouteIR(
            experts=experts,
            request_id=self.request_id,
            layer_id=self.layer_id,
            token_position=self.token_position,
            home_rank=self.home_rank,
            num_experts=self.num_experts,
            top_k=self.top_k,
            home_rank_source=self.home_rank_source,
        )


def deterministic_home_ranks(
    request_id: np.ndarray,
    layer_id: np.ndarray,
    token_position: np.ndarray,
    ep_size: int,
) -> np.ndarray:
    """Frozen synthetic home-rank control: (request + layer + position) mod EP."""

    if int(ep_size) <= 0:
        raise ValueError("ep_size must be positive")
    request = np.asarray(request_id, dtype=np.int64)
    layer = np.asarray(layer_id, dtype=np.int64)
    position = np.asarray(token_position, dtype=np.int64)
    if request.ndim != 1 or layer.shape != request.shape or position.shape != request.shape:
        raise ValueError("request_id, layer_id, and token_position must be aligned 1-D arrays")
    return np.mod(request + layer + position, int(ep_size)).astype(np.int32)


def load_route_csv(
    path: str | Path,
    *,
    num_experts: int,
    top_k: int,
    request_id_column: str = "sample_id",
    home_rank_column: str = "home_rank",
    synthetic_home_ep_size: int | None = None,
) -> RouteIR:
    """Load a ranked route CSV without inferring architecture constants.

    Required columns are ``request_id_column``, ``layer``, ``token_position``,
    ``rank``, and ``expert_id``.  ``rank`` must be exactly ``1..top_k`` for
    each token.  Home rank must either be present in ``home_rank_column`` or be
    explicitly synthesized by passing ``synthetic_home_ep_size``.  The two
    sources are mutually exclusive to keep the evidence boundary auditable.
    """

    if int(num_experts) <= 0 or int(top_k) <= 0:
        raise ValueError("num_experts and top_k must be positive and explicit")
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        required = {
            request_id_column,
            "layer",
            "token_position",
            "rank",
            "expert_id",
        }
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"route CSV missing required columns: {missing}")
        has_home = home_rank_column in fieldnames
        if has_home and synthetic_home_ep_size is not None:
            raise ValueError("observed and synthetic home-rank sources are mutually exclusive")
        if not has_home and synthetic_home_ep_size is None:
            raise ValueError(
                f"route CSV lacks {home_rank_column!r}; explicitly pass synthetic_home_ep_size"
            )

        parsed: list[tuple[int, int, int, int, int, int | None]] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                home_value = (
                    int(row[home_rank_column])
                    if has_home and row[home_rank_column] not in (None, "")
                    else None
                )
                parsed.append(
                    (
                        int(row[request_id_column]),
                        int(row["layer"]),
                        int(row["token_position"]),
                        int(row["rank"]),
                        int(row["expert_id"]),
                        home_value,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid integer field at CSV line {line_number}") from exc

    if not parsed:
        raise ValueError("route CSV is empty")
    if has_home and any(row[5] is None for row in parsed):
        raise ValueError(f"{home_rank_column!r} contains empty values")
    parsed.sort(key=lambda row: (row[0], row[1], row[2], row[3]))

    groups: dict[tuple[int, int, int], list[tuple[int, int, int | None]]] = defaultdict(list)
    for request, layer, position, rank, expert, home in parsed:
        groups[(request, layer, position)].append((rank, expert, home))

    experts: list[list[int]] = []
    request_ids: list[int] = []
    layers: list[int] = []
    positions: list[int] = []
    observed_homes: list[int] = []
    expected_ranks = list(range(1, int(top_k) + 1))
    for (request, layer, position), rows in sorted(groups.items()):
        ranks = [row[0] for row in rows]
        if ranks != expected_ranks:
            raise ValueError(
                f"token {(request, layer, position)} ranks are {ranks}, "
                f"expected {expected_ranks}"
            )
        token_experts = [row[1] for row in rows]
        if any(expert < 0 or expert >= int(num_experts) for expert in token_experts):
            raise ValueError(
                f"token {(request, layer, position)} has expert outside explicit E={num_experts}"
            )
        if len(set(token_experts)) != int(top_k):
            raise ValueError(f"token {(request, layer, position)} has duplicate experts")
        if has_home:
            homes = {int(row[2]) for row in rows if row[2] is not None}
            if len(homes) != 1:
                raise ValueError(
                    f"token {(request, layer, position)} has inconsistent home_rank values"
                )
            observed_homes.append(next(iter(homes)))
        experts.append(token_experts)
        request_ids.append(request)
        layers.append(layer)
        positions.append(position)

    request_array = np.asarray(request_ids, dtype=np.int64)
    layer_array = np.asarray(layers, dtype=np.int32)
    position_array = np.asarray(positions, dtype=np.int64)
    if has_home:
        home_array = np.asarray(observed_homes, dtype=np.int32)
        source = f"csv:{home_rank_column}"
    else:
        assert synthetic_home_ep_size is not None
        home_array = deterministic_home_ranks(
            request_array, layer_array, position_array, synthetic_home_ep_size
        )
        source = f"synthetic_mod:ep_size={int(synthetic_home_ep_size)}"

    return RouteIR(
        experts=np.asarray(experts, dtype=np.int32),
        request_id=request_array,
        layer_id=layer_array,
        token_position=position_array,
        home_rank=home_array,
        num_experts=int(num_experts),
        top_k=int(top_k),
        home_rank_source=source,
    )


@dataclass(frozen=True)
class PerRequestCrossDomainCost:
    """Logical C0/C1 record counts, aggregated independently per request."""

    contract: str
    request_id: np.ndarray
    logical_records: np.ndarray
    local_records: np.ndarray
    cross_domain_records: np.ndarray

    def __post_init__(self) -> None:
        request_id = _readonly_int_array(
            self.request_id, dtype=np.int64, ndim=1, name="request_id"
        )
        logical = _readonly_int_array(
            self.logical_records, dtype=np.int64, ndim=1, name="logical_records"
        )
        local = _readonly_int_array(
            self.local_records, dtype=np.int64, ndim=1, name="local_records"
        )
        cross = _readonly_int_array(
            self.cross_domain_records,
            dtype=np.int64,
            ndim=1,
            name="cross_domain_records",
        )
        if not (len(request_id) == len(logical) == len(local) == len(cross)):
            raise ValueError("per-request cost arrays must have equal length")
        if np.any(logical != local + cross):
            raise ValueError("logical_records must equal local_records + cross_domain_records")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "logical_records", logical)
        object.__setattr__(self, "local_records", local)
        object.__setattr__(self, "cross_domain_records", cross)

    @property
    def total_logical_records(self) -> int:
        return int(self.logical_records.sum())

    @property
    def total_cross_domain_records(self) -> int:
        return int(self.cross_domain_records.sum())

    def as_rows(self) -> list[dict[str, int | str]]:
        return [
            {
                "contract": self.contract,
                "request_id": int(request),
                "logical_records": int(logical),
                "local_records": int(local),
                "cross_domain_records": int(cross),
            }
            for request, logical, local, cross in zip(
                self.request_id,
                self.logical_records,
                self.local_records,
                self.cross_domain_records,
            )
        ]


def lower_cross_domain_cost(
    route: RouteIR,
    expert_to_rank: np.ndarray,
    rank_to_domain: np.ndarray,
    *,
    contract: str,
) -> PerRequestCrossDomainCost:
    """Lower a route into C0/C1 dispatch records and count cross-domain records.

    A record is cross-domain when the expert owner and token home rank belong
    to different topology domains.  C0 emits one record per routed expert;
    C1 emits one record per unique owner rank reached by that token.
    """

    mapping = np.asarray(expert_to_rank, dtype=np.int64)
    domains = np.asarray(rank_to_domain, dtype=np.int64)
    if mapping.ndim != 1 or len(mapping) != route.num_experts:
        raise ValueError("expert_to_rank must have exactly num_experts entries")
    if domains.ndim != 1 or len(domains) == 0:
        raise ValueError("rank_to_domain must be a non-empty 1-D array")
    if np.any(mapping < 0) or np.any(mapping >= len(domains)):
        raise ValueError("placement references a rank missing from rank_to_domain")
    if np.any(route.home_rank >= len(domains)):
        raise ValueError("RouteIR home_rank is missing from rank_to_domain")
    if contract not in (C0_EXPANDED, C1_UNIQUE_OWNER):
        raise ValueError(f"unknown reference contract: {contract}")

    request_values, request_slot = np.unique(route.request_id, return_inverse=True)
    owners = mapping[route.experts]
    home_domains = domains[route.home_rank][:, None]
    if contract == C0_EXPANDED:
        row_logical = np.full(route.token_count, route.top_k, dtype=np.int64)
        row_cross = np.count_nonzero(domains[owners] != home_domains, axis=1)
    else:
        sorted_owners = np.sort(owners, axis=1)
        unique_owner = np.ones(sorted_owners.shape, dtype=bool)
        if route.top_k > 1:
            unique_owner[:, 1:] = sorted_owners[:, 1:] != sorted_owners[:, :-1]
        row_logical = np.count_nonzero(unique_owner, axis=1)
        row_cross = np.count_nonzero(
            unique_owner & (domains[sorted_owners] != home_domains), axis=1
        )
    request_count = len(request_values)
    logical = np.bincount(
        request_slot, weights=row_logical, minlength=request_count
    ).astype(np.int64)
    cross = np.bincount(
        request_slot, weights=row_cross, minlength=request_count
    ).astype(np.int64)
    local = logical - cross

    return PerRequestCrossDomainCost(
        contract=contract,
        request_id=request_values,
        logical_records=logical,
        local_records=local,
        cross_domain_records=cross,
    )


def _pair_counter(routes: np.ndarray) -> Counter[tuple[int, int]]:
    counter: Counter[tuple[int, int]] = Counter()
    for row in routes:
        ordered = sorted(int(value) for value in row)
        for left in range(len(ordered)):
            for right in range(left + 1, len(ordered)):
                counter[(ordered[left], ordered[right])] += 1
    return counter


def _counter_tv(
    before: Mapping[Hashable, int], after: Mapping[Hashable, int]
) -> float:
    before_total = float(sum(before.values()))
    after_total = float(sum(after.values()))
    if before_total == 0.0 and after_total == 0.0:
        return 0.0
    if before_total == 0.0 or after_total == 0.0:
        return 1.0
    keys = set(before) | set(after)
    return 0.5 * sum(
        abs(before.get(key, 0) / before_total - after.get(key, 0) / after_total)
        for key in keys
    )


@dataclass(frozen=True)
class S1RCertificate:
    group_count: int
    attempted_swaps: int
    accepted_swaps: int
    duplicate_rows: int
    degree_tv_mean: float
    degree_tv_max: float
    token_jaccard_mean: float
    pair_distance_mean: float
    pair_distance_max: float

    @property
    def acceptance_rate(self) -> float:
        if self.attempted_swaps == 0:
            return 0.0
        return self.accepted_swaps / self.attempted_swaps


@dataclass(frozen=True)
class S1RResult:
    route: RouteIR
    certificate: S1RCertificate


def s1r_exact_degree(
    route: RouteIR,
    *,
    seed: int,
    swap_multiplier: int = 8,
) -> S1RResult:
    """Degree-preserving double-edge rewiring within each request-layer.

    Token degree (top-k), request-layer expert degrees, route simplicity, and
    all exogenous metadata are invariant.  Co-activation and rank ordering may
    change.  Low accepted-swap count or Jaccard near one is a diagnostic, not a
    hidden retry or a fabricated success.
    """

    if int(swap_multiplier) < 0:
        raise ValueError("swap_multiplier must be non-negative")
    rng = np.random.default_rng(int(seed))
    out = np.array(route.experts, copy=True)
    group_indices: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (request, layer) in enumerate(zip(route.request_id, route.layer_id)):
        group_indices[(int(request), int(layer))].append(index)

    attempted_total = 0
    accepted_total = 0
    degree_tvs: list[float] = []
    pair_distances: list[float] = []

    for key in sorted(group_indices):
        indices = np.asarray(group_indices[key], dtype=np.int64)
        before = np.array(route.experts[indices], copy=True)
        row_sets = [set(int(value) for value in row) for row in out[indices]]
        attempts = int(swap_multiplier) * int(before.size)
        attempted_total += attempts
        accepted = 0
        if len(indices) >= 2:
            for _ in range(attempts):
                local_left, local_right = rng.integers(0, len(indices), size=2)
                if local_left == local_right:
                    continue
                left_rank = int(rng.integers(0, route.top_k))
                right_rank = int(rng.integers(0, route.top_k))
                left_index = int(indices[local_left])
                right_index = int(indices[local_right])
                left_expert = int(out[left_index, left_rank])
                right_expert = int(out[right_index, right_rank])
                if (
                    left_expert == right_expert
                    or right_expert in row_sets[local_left]
                    or left_expert in row_sets[local_right]
                ):
                    continue
                out[left_index, left_rank] = right_expert
                out[right_index, right_rank] = left_expert
                row_sets[local_left].remove(left_expert)
                row_sets[local_left].add(right_expert)
                row_sets[local_right].remove(right_expert)
                row_sets[local_right].add(left_expert)
                accepted += 1
        accepted_total += accepted

        after = out[indices]
        before_degree = np.bincount(before.reshape(-1), minlength=route.num_experts)
        after_degree = np.bincount(after.reshape(-1), minlength=route.num_experts)
        degree_tvs.append(
            0.5
            * float(np.abs(before_degree - after_degree).sum())
            / float(before.size)
        )
        pair_distances.append(_counter_tv(_pair_counter(before), _pair_counter(after)))

    duplicate_rows = int(
        sum(len(set(int(value) for value in row)) != route.top_k for row in out)
    )
    jaccards = []
    for before_row, after_row in zip(route.experts, out):
        before_set = set(int(value) for value in before_row)
        after_set = set(int(value) for value in after_row)
        jaccards.append(len(before_set & after_set) / len(before_set | after_set))

    certificate = S1RCertificate(
        group_count=len(group_indices),
        attempted_swaps=attempted_total,
        accepted_swaps=accepted_total,
        duplicate_rows=duplicate_rows,
        degree_tv_mean=float(np.mean(degree_tvs)),
        degree_tv_max=float(np.max(degree_tvs)),
        token_jaccard_mean=float(np.mean(jaccards)),
        pair_distance_mean=float(np.mean(pair_distances)),
        pair_distance_max=float(np.max(pair_distances)),
    )
    return S1RResult(route=route.with_experts(out), certificate=certificate)


def _hyperedge_histograms(
    route: RouteIR, group_keys: Sequence[tuple[int, ...]]
) -> dict[tuple[int, ...], Counter[tuple[int, ...]]]:
    if len(group_keys) != route.token_count:
        raise ValueError("group_keys must align with RouteIR tokens")
    histograms: dict[tuple[int, ...], Counter[tuple[int, ...]]] = defaultdict(Counter)
    for key, row in zip(group_keys, route.experts):
        histograms[tuple(int(value) for value in key)][
            tuple(sorted(int(value) for value in row))
        ] += 1
    return histograms


def _canonical_histogram_bytes(
    route: RouteIR,
    *,
    magic: bytes,
    histograms: Mapping[tuple[int, ...], Counter[tuple[int, ...]]],
) -> bytes:
    """Serialize a route histogram using the frozen little-endian toy format."""

    payload = bytearray()
    payload.extend(magic)
    payload.extend(struct.pack("<III", route.num_experts, route.top_k, len(histograms)))
    for key in sorted(histograms):
        payload.extend(struct.pack("<I", len(key)))
        for value in key:
            payload.extend(struct.pack("<q", int(value)))
        entries = histograms[key]
        payload.extend(struct.pack("<I", len(entries)))
        for hyperedge, count in sorted(entries.items()):
            if len(hyperedge) != route.top_k:
                raise AssertionError("canonical hyperedge width changed")
            payload.extend(struct.pack(f"<{route.top_k}I", *hyperedge))
            payload.extend(struct.pack("<Q", int(count)))
    return bytes(payload)


def canonical_s2_size_bytes(route: RouteIR) -> int:
    """Toy S2 payload size: per-request-layer unordered hyperedge histograms.

    This retains request identity while discarding temporal order.  The helper
    is a deterministic toy serialization only; the frozen P0-B 70% gate uses
    the runner's explicit bit-packed accounting instead.
    """

    keys = [
        (int(request), int(layer))
        for request, layer in zip(route.request_id, route.layer_id)
    ]
    histograms = _hyperedge_histograms(route, keys)
    return len(_canonical_histogram_bytes(route, magic=b"RFS2\x01", histograms=histograms))


def canonical_s3_size_bytes(route: RouteIR, window_id: np.ndarray) -> int:
    """Canonical S3 payload size: request-layer-window hyperedge histograms."""

    windows = np.asarray(window_id, dtype=np.int64)
    if windows.ndim != 1 or len(windows) != route.token_count:
        raise ValueError("window_id must be a 1-D array aligned with RouteIR tokens")
    if np.any(windows < 0):
        raise ValueError("window_id must be non-negative")
    keys = [
        (int(request), int(layer), int(window))
        for request, layer, window in zip(route.request_id, route.layer_id, windows)
    ]
    histograms = _hyperedge_histograms(route, keys)
    return len(_canonical_histogram_bytes(route, magic=b"RFS3\x01", histograms=histograms))


def canonical_s4_size_bytes(route: RouteIR) -> int:
    """Canonical S4 size for the full ordered route oracle.

    The frozen toy format stores explicit architecture constants followed by
    one record per token: request, layer, token position, home rank, and the
    unordered top-k expert set.  Its byte size is independent of expert order.
    It is a representation-size reference, not
    a claim about backend frames or wire bytes.
    """

    header_bytes = len(b"RFS4\x01") + struct.calcsize("<IIQ")
    per_token_bytes = struct.calcsize("<qqqI") + struct.calcsize(
        f"<{route.top_k}I"
    )
    return header_bytes + route.token_count * per_token_bytes


def placement_hash(expert_to_rank: np.ndarray) -> str:
    """Stable SHA-256 over a normalized expert-to-rank mapping."""

    mapping = np.asarray(expert_to_rank, dtype=np.int64)
    if mapping.ndim != 1 or len(mapping) == 0:
        raise ValueError("placement must be a non-empty 1-D array")
    if np.any(mapping < 0):
        raise ValueError("placement ranks must be non-negative")
    payload = b"RouteFidelityPlacementV1\x00"
    payload += struct.pack("<Q", len(mapping))
    payload += np.asarray(mapping, dtype="<i8").tobytes(order="C")
    return sha256(payload).hexdigest()


def balanced_placements(
    num_experts: int,
    ep_size: int,
    *,
    random_count: int = 128,
    seed: int = 20260718,
) -> dict[str, np.ndarray]:
    """Create deterministic contiguous, round-robin, and random placements."""

    num_experts = int(num_experts)
    ep_size = int(ep_size)
    random_count = int(random_count)
    if num_experts <= 0 or ep_size <= 0:
        raise ValueError("num_experts and ep_size must be positive")
    if num_experts % ep_size != 0:
        raise ValueError("balanced placement requires num_experts divisible by ep_size")
    if random_count < 0:
        raise ValueError("random_count must be non-negative")
    per_rank = num_experts // ep_size
    result: dict[str, np.ndarray] = {
        "contiguous": np.arange(num_experts, dtype=np.int32) // per_rank,
        "round_robin": np.arange(num_experts, dtype=np.int32) % ep_size,
    }
    rng = np.random.default_rng(int(seed))
    rank_slots = np.repeat(np.arange(ep_size, dtype=np.int32), per_rank)
    for index in range(random_count):
        permutation = rng.permutation(num_experts)
        mapping = np.empty(num_experts, dtype=np.int32)
        mapping[permutation] = rank_slots
        result[f"random_{index:03d}"] = mapping
    for name, mapping in list(result.items()):
        frozen = np.array(mapping, dtype=np.int32, copy=True)
        frozen.setflags(write=False)
        result[name] = frozen
    return result


def placement_manifest_hash(placements: Mapping[str, np.ndarray]) -> str:
    """Freeze a named placement pool, including names and individual hashes."""

    digest = sha256(b"RouteFidelityPlacementManifestV1\x00")
    for name in sorted(placements):
        encoded = str(name).encode("utf-8")
        digest.update(struct.pack("<I", len(encoded)))
        digest.update(encoded)
        digest.update(bytes.fromhex(placement_hash(placements[name])))
    return digest.hexdigest()


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    """Dependency-free Kendall tau-b with ties handled in both rankings."""

    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.ndim != 1 or right.shape != left.shape:
        raise ValueError("x and y must be aligned 1-D arrays")
    if len(left) < 2:
        raise ValueError("Kendall tau-b requires at least two observations")
    if np.any(~np.isfinite(left)) or np.any(~np.isfinite(right)):
        raise ValueError("Kendall tau-b inputs must be finite")

    concordant = 0
    discordant = 0
    tie_left = 0
    tie_right = 0
    for first in range(len(left) - 1):
        for second in range(first + 1, len(left)):
            delta_left = left[first] - left[second]
            delta_right = right[first] - right[second]
            if delta_left == 0.0 and delta_right == 0.0:
                continue
            if delta_left == 0.0:
                tie_left += 1
            elif delta_right == 0.0:
                tie_right += 1
            elif delta_left * delta_right > 0.0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + tie_left)
        * (concordant + discordant + tie_right)
    )
    if denominator == 0.0:
        return float("nan")
    return (concordant - discordant) / denominator


@dataclass(frozen=True)
class SelectionRegret:
    selected_config: str
    exact_best_config: str
    exact_best_configs: tuple[str, ...]
    selected_exact_score: float
    exact_best_score: float
    absolute_regret: float
    relative_regret: float


def selection_regret(
    exact_scores: Sequence[float],
    surrogate_scores: Sequence[float],
    *,
    config_ids: Sequence[str] | None = None,
    minimize: bool = True,
) -> SelectionRegret:
    """Evaluate the true regret of the config selected by a surrogate."""

    exact = np.asarray(exact_scores, dtype=np.float64)
    surrogate = np.asarray(surrogate_scores, dtype=np.float64)
    if exact.ndim != 1 or surrogate.shape != exact.shape or len(exact) == 0:
        raise ValueError("exact_scores and surrogate_scores must be aligned non-empty arrays")
    if np.any(~np.isfinite(exact)) or np.any(~np.isfinite(surrogate)):
        raise ValueError("regret inputs must be finite")
    ids = tuple(str(index) for index in range(len(exact))) if config_ids is None else tuple(str(value) for value in config_ids)
    if len(ids) != len(exact) or len(set(ids)) != len(ids):
        raise ValueError("config_ids must be unique and aligned with scores")

    direction = 1.0 if minimize else -1.0
    selected_index = min(
        range(len(ids)), key=lambda index: (direction * surrogate[index], ids[index])
    )
    best_value = float(np.min(exact) if minimize else np.max(exact))
    best_indices = [index for index, value in enumerate(exact) if value == best_value]
    best_index = min(best_indices, key=lambda index: ids[index])
    if minimize:
        absolute = float(exact[selected_index] - best_value)
    else:
        absolute = float(best_value - exact[selected_index])
    absolute = max(0.0, absolute)
    if best_value == 0.0:
        relative = 0.0 if absolute == 0.0 else float("inf")
    else:
        relative = absolute / abs(best_value)
    return SelectionRegret(
        selected_config=ids[selected_index],
        exact_best_config=ids[best_index],
        exact_best_configs=tuple(sorted(ids[index] for index in best_indices)),
        selected_exact_score=float(exact[selected_index]),
        exact_best_score=best_value,
        absolute_regret=absolute,
        relative_regret=float(relative),
    )


@dataclass(frozen=True)
class ClusterBootstrapSummary:
    tau_b_point: float
    tau_b_ci_low: float
    tau_b_ci_high: float
    relative_regret_point: float
    relative_regret_ci_low: float
    relative_regret_ci_high: float
    cluster_count: int
    bootstrap_count: int
    tau_b_replicates: np.ndarray
    relative_regret_replicates: np.ndarray

    def __post_init__(self) -> None:
        tau = np.asarray(self.tau_b_replicates, dtype=np.float64).copy()
        regret = np.asarray(self.relative_regret_replicates, dtype=np.float64).copy()
        tau.setflags(write=False)
        regret.setflags(write=False)
        object.__setattr__(self, "tau_b_replicates", tau)
        object.__setattr__(self, "relative_regret_replicates", regret)


def _default_score_reduce(rows: np.ndarray) -> np.ndarray:
    return np.mean(rows, axis=0)


def _finite_quantile(values: np.ndarray, probability: float) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    return float(np.quantile(finite, probability))


def cluster_bootstrap_ranking(
    exact_row_scores: np.ndarray,
    surrogate_row_scores: np.ndarray,
    cluster_ids: Sequence[Hashable],
    *,
    config_ids: Sequence[str] | None = None,
    bootstrap_count: int = 10_000,
    seed: int = 20260718,
    confidence: float = 0.95,
    minimize: bool = True,
    score_reduce: Callable[[np.ndarray], np.ndarray] | None = None,
    seed_reduce: str = "mean",
) -> ClusterBootstrapSummary:
    """Cluster-bootstrap tau-b and selected-config regret.

    ``exact_row_scores`` is ``[rows, configs]``.  Surrogate scores may be the
    same shape or ``[synthesis_seeds, rows, configs]``.  Every resample draws
    request/article cluster IDs with replacement and includes all rows from a
    selected cluster.  ``score_reduce`` maps selected rows to one score per
    config; the default is a row mean.  A custom reducer can implement a P99,
    but callers remain responsible for whether that statistic is meaningful.
    """

    exact = np.asarray(exact_row_scores, dtype=np.float64)
    surrogate = np.asarray(surrogate_row_scores, dtype=np.float64)
    clusters = np.asarray(cluster_ids, dtype=object)
    if exact.ndim != 2 or exact.shape[1] < 2:
        raise ValueError("exact_row_scores must be [rows, at least two configs]")
    if surrogate.ndim == 2:
        if surrogate.shape != exact.shape:
            raise ValueError("2-D surrogate scores must match exact scores")
        surrogate = surrogate[None, :, :]
    elif surrogate.ndim == 3:
        if surrogate.shape[1:] != exact.shape:
            raise ValueError("3-D surrogate scores must be [seeds, rows, configs]")
    else:
        raise ValueError("surrogate_row_scores must be 2-D or 3-D")
    if clusters.ndim != 1 or len(clusters) != exact.shape[0]:
        raise ValueError("cluster_ids must align with score rows")
    if np.any(~np.isfinite(exact)) or np.any(~np.isfinite(surrogate)):
        raise ValueError("bootstrap scores must be finite")
    if int(bootstrap_count) <= 0:
        raise ValueError("bootstrap_count must be positive")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if seed_reduce not in ("mean", "median"):
        raise ValueError("seed_reduce must be 'mean' or 'median'")
    ids = tuple(str(index) for index in range(exact.shape[1])) if config_ids is None else tuple(str(value) for value in config_ids)
    if len(ids) != exact.shape[1] or len(set(ids)) != len(ids):
        raise ValueError("config_ids must be unique and aligned with config columns")
    reducer = score_reduce or _default_score_reduce

    cluster_rows: dict[Hashable, list[int]] = {}
    for row_index, cluster in enumerate(clusters.tolist()):
        try:
            cluster_rows.setdefault(cluster, []).append(row_index)
        except TypeError as exc:
            raise ValueError("cluster IDs must be hashable") from exc
    unique_clusters = list(cluster_rows)
    if len(unique_clusters) < 2:
        raise ValueError("cluster bootstrap requires at least two clusters")

    def aggregate(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        exact_scores = np.asarray(reducer(exact[indices]), dtype=np.float64)
        per_seed = np.stack(
            [np.asarray(reducer(seed_scores[indices]), dtype=np.float64) for seed_scores in surrogate]
        )
        if exact_scores.shape != (exact.shape[1],) or per_seed.shape[1:] != (exact.shape[1],):
            raise ValueError("score_reduce must return one score per config")
        surrogate_scores = (
            np.mean(per_seed, axis=0)
            if seed_reduce == "mean"
            else np.median(per_seed, axis=0)
        )
        return exact_scores, surrogate_scores

    all_indices = np.arange(exact.shape[0], dtype=np.int64)
    exact_point, surrogate_point = aggregate(all_indices)
    tau_point = kendall_tau_b(exact_point, surrogate_point)
    regret_point = selection_regret(
        exact_point, surrogate_point, config_ids=ids, minimize=minimize
    ).relative_regret

    rng = np.random.default_rng(int(seed))
    tau_replicates = np.empty(int(bootstrap_count), dtype=np.float64)
    regret_replicates = np.empty(int(bootstrap_count), dtype=np.float64)
    for replicate in range(int(bootstrap_count)):
        sampled = rng.integers(0, len(unique_clusters), size=len(unique_clusters))
        indices = np.fromiter(
            (
                row
                for cluster_index in sampled
                for row in cluster_rows[unique_clusters[int(cluster_index)]]
            ),
            dtype=np.int64,
        )
        exact_scores, surrogate_scores = aggregate(indices)
        tau_replicates[replicate] = kendall_tau_b(exact_scores, surrogate_scores)
        regret_replicates[replicate] = selection_regret(
            exact_scores, surrogate_scores, config_ids=ids, minimize=minimize
        ).relative_regret

    alpha = (1.0 - float(confidence)) / 2.0
    return ClusterBootstrapSummary(
        tau_b_point=float(tau_point),
        tau_b_ci_low=_finite_quantile(tau_replicates, alpha),
        tau_b_ci_high=_finite_quantile(tau_replicates, 1.0 - alpha),
        relative_regret_point=float(regret_point),
        relative_regret_ci_low=_finite_quantile(regret_replicates, alpha),
        relative_regret_ci_high=_finite_quantile(regret_replicates, 1.0 - alpha),
        cluster_count=len(unique_clusters),
        bootstrap_count=int(bootstrap_count),
        tau_b_replicates=tau_replicates,
        relative_regret_replicates=regret_replicates,
    )
