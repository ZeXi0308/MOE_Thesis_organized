from __future__ import annotations

import random
import os
from pathlib import Path

from datasets import Dataset, load_dataset

BUILTIN_PROMPTS = [
    "Mixture of Experts models route each token to a small subset of feed-forward experts.",
    "The combine phase aggregates expert outputs with router gate weights.",
    "Communication-efficient inference is important for large language model serving.",
    "A receiver port can become the bottleneck when many expert outputs arrive together.",
    "Rank-aware approximate combine uses the router ranking as an importance proxy.",
    "The lowest ranked expert output may contribute less to the final hidden state.",
    "Layer sensitivity profiling helps decide where approximation is acceptable.",
    "A static lookup table is easier to deploy than online per-token optimization.",
    "Uniform quantization is a simple baseline but ignores routing importance.",
    "Dropping an expert output can save bytes but may introduce numerical bias.",
    "Perplexity and logit KL can measure the quality impact of approximation.",
    "The first experiment should validate the contribution distribution before optimization.",
]


def _slice_with_offset(values: list[str], num_samples: int, offset: int) -> list[str]:
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")
    if offset + num_samples > len(values):
        raise RuntimeError(
            f"requested [{offset}:{offset + num_samples}] from only {len(values)} available prompts"
        )
    return values[offset : offset + num_samples]


def get_builtin_prompts(num_samples: int, offset: int = 0) -> list[str]:
    required = offset + num_samples
    if required <= len(BUILTIN_PROMPTS):
        return BUILTIN_PROMPTS[offset:required]
    out: list[str] = []
    while len(out) < required:
        out.extend(BUILTIN_PROMPTS)
    return out[offset:required]


def get_wikitext2_prompts(
    num_samples: int,
    min_chars: int = 80,
    offset: int = 0,
    split: str = "validation",
    seed: int | None = None,
    config: str = "wikitext-2-raw-v1",
) -> list[str]:
    try:
        dataset = load_dataset("wikitext", config, split=split)
    except (ConnectionError, OSError):
        dataset = _load_cached_wikitext_arrow(split, config=config)
    prompts: list[str] = []
    for row in dataset:
        text = " ".join(str(row["text"]).split())
        if len(text) < min_chars:
            continue
        prompts.append(text)
    if seed is not None:
        random.Random(seed).shuffle(prompts)
    return _slice_with_offset(prompts, num_samples, offset)


def get_wikitext2_documents(
    num_samples: int,
    min_chars: int = 500,
    offset: int = 0,
    split: str = "validation",
    seed: int | None = None,
    config: str = "wikitext-2-raw-v1",
) -> list[str]:
    """Return one independent prompt per WikiText article.

    WikiText stores an article as a top-level ``= title =`` row followed by
    paragraph and subsection rows.  The older line-level sampler could return
    many adjacent rows from the same article and then incorrectly treat them
    as independent bootstrap samples.  This parser keeps the article as the
    sampling/bootstrapping unit.

    ``config`` selects the underlying HF dataset config.  ``wikitext-2-raw-v1``
    is the original ~121-document pool (validation=60, test=61 after the
    ``min_chars`` filter) that has been reused across nearly every experiment
    in this project; ``wikitext-103-raw-v1`` is the same corpus family but
    with roughly 30x as many articles, and no prior experiment in this
    project has ever touched it, so it is the simplest way to get genuinely
    fresh sealed documents without inventing a new exclusion-tracking scheme.
    """
    try:
        dataset = load_dataset("wikitext", config, split=split)
    except (ConnectionError, OSError):
        dataset = _load_cached_wikitext_arrow(split, config=config)

    documents: list[str] = []
    title: str | None = None
    body: list[str] = []

    def flush() -> None:
        nonlocal title, body
        if title is None:
            return
        text = " ".join([title, *body]).strip()
        if len(text) >= min_chars:
            documents.append(text)
        title = None
        body = []

    for row in dataset:
        text = " ".join(str(row["text"]).split())
        is_top_level_title = (
            text.startswith("= ")
            and text.endswith(" =")
            and not text.startswith("= =")
        )
        if is_top_level_title:
            flush()
            title = text.strip("= ")
        elif text and title is not None:
            body.append(text)
    flush()

    if seed is not None:
        random.Random(seed).shuffle(documents)
    return _slice_with_offset(documents, num_samples, offset)


def _load_cached_wikitext_arrow(split: str, config: str = "wikitext-2-raw-v1") -> Dataset:
    """Load a materialized WikiText split when Hub metadata is unavailable."""
    cache_roots: list[Path] = []
    if datasets_cache := os.environ.get("HF_DATASETS_CACHE"):
        cache_roots.append(Path(datasets_cache).expanduser())
    if hf_home := os.environ.get("HF_HOME"):
        cache_roots.append(Path(hf_home).expanduser() / "datasets")
    if xdg_cache := os.environ.get("XDG_CACHE_HOME"):
        cache_roots.append(Path(xdg_cache).expanduser() / "huggingface" / "datasets")
    cache_roots.append(Path.home() / ".cache" / "huggingface" / "datasets")

    filename = f"wikitext-{split}.arrow"
    for cache_root in dict.fromkeys(cache_roots):
        candidates = sorted(
            cache_root.glob(f"wikitext/{config}/**/{filename}"),
            reverse=True,
        )
        if candidates:
            return Dataset.from_file(str(candidates[0]))
    searched = ", ".join(str(path) for path in dict.fromkeys(cache_roots))
    raise RuntimeError(
        f"WikiText split {split!r} ({config}) is unavailable offline; searched: {searched}"
    )


def get_prompts(
    dataset: str,
    num_samples: int,
    offset: int = 0,
    split: str = "validation",
    seed: int | None = None,
) -> list[str]:
    if dataset == "builtin":
        return get_builtin_prompts(num_samples, offset=offset)
    if dataset == "wikitext2":
        return get_wikitext2_prompts(num_samples, offset=offset, split=split, seed=seed)
    if dataset == "wikitext2_docs":
        return get_wikitext2_documents(
            num_samples, offset=offset, split=split, seed=seed
        )
    if dataset == "wikitext103_docs":
        # Same corpus family, ~30x larger pool. No prior experiment in this
        # project has ever used wikitext-103-raw-v1, so any offset here is
        # genuinely fresh without needing a historical-exclusion registry.
        return get_wikitext2_documents(
            num_samples,
            offset=offset,
            split=split,
            seed=seed,
            config="wikitext-103-raw-v1",
        )
    raise ValueError(f"unknown dataset: {dataset}")
