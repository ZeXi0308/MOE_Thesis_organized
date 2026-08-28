from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence


@dataclass(frozen=True)
class Request:
    request_id: int
    experts: frozenset[int]


def overlap(a: Request, b: Request) -> int:
    return len(a.experts & b.experts)


def pair_random(requests: Sequence[Request], seed: int) -> list[tuple[Request, Request]]:
    if len(requests) % 2:
        raise ValueError("request count must be even")
    items = list(requests)
    random.Random(seed).shuffle(items)
    return list(zip(items[::2], items[1::2]))


def pair_greedy(requests: Sequence[Request], *, maximize: bool) -> list[tuple[Request, Request]]:
    if len(requests) % 2:
        raise ValueError("request count must be even")
    remaining = sorted(requests, key=lambda item: item.request_id)
    pairs = []
    while remaining:
        first = remaining.pop(0)
        ranked = sorted(
            remaining,
            key=lambda item: (
                -overlap(first, item) if maximize else overlap(first, item),
                item.request_id,
            ),
        )
        second = ranked[0]
        remaining.remove(second)
        pairs.append((first, second))
    return pairs


def union_invocations(pairs: Sequence[tuple[Request, Request]]) -> int:
    return sum(len(a.experts | b.experts) for a, b in pairs)
