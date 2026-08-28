from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache

from kv_quant_cache import (
    QuantizedDynamicCache,
    asymmetric_int4_qdq,
    assert_no_storage_aliases,
    clone_dynamic_cache,
)


def _prompt_cache() -> DynamicCache:
    cache = DynamicCache()
    keys = torch.linspace(-1.0, 1.0, 2 * 3 * 4 * 8, dtype=torch.float32).reshape(2, 3, 4, 8)
    values = keys + 4.0
    cache.update(keys, values, 0)
    return cache


def test_asymmetric_int4_formula_and_constant_vectors() -> None:
    x = torch.tensor([[[[-1.0, -0.2, 0.3, 1.0], [3.25, 3.25, 3.25, 3.25]]]])
    actual = asymmetric_int4_qdq(x)
    # FP32 evaluation gives z=7 for this vector, then q=[0,5,9,14].
    expected_first = torch.tensor([-14.0 / 15.0, -4.0 / 15.0, 4.0 / 15.0, 14.0 / 15.0])
    torch.testing.assert_close(actual[0, 0, 0], expected_first)
    assert torch.equal(actual[0, 0, 1], x[0, 0, 1])


def test_k_only_quantizes_snapshot_and_new_writes_without_touching_v() -> None:
    prompt = _prompt_cache()
    cache = QuantizedDynamicCache.from_prompt_cache(prompt, target="k_only")
    assert not torch.equal(cache.layers[0].keys, prompt.layers[0].keys)
    assert torch.equal(cache.layers[0].values, prompt.layers[0].values)
    new_k = torch.tensor([[[[-1.0, -0.7, -0.2, 0.0, 0.1, 0.2, 0.8, 1.0]]]]).expand(2, 3, 1, 8)
    new_v = torch.randn(2, 3, 1, 8)
    cache.update(new_k, new_v, 0)
    assert torch.equal(cache.layers[0].values[..., -1:, :], new_v)
    assert torch.equal(cache.layers[0].keys[..., -1:, :], asymmetric_int4_qdq(new_k))
    assert [(event.phase, event.layer_idx) for event in cache.ledger.events] == [
        ("snapshot", 0),
        ("update", 0),
    ]


def test_v_only_and_kv_target_isolation() -> None:
    prompt = _prompt_cache()
    v_only = QuantizedDynamicCache.from_prompt_cache(prompt, target="v_only")
    both = QuantizedDynamicCache.from_prompt_cache(prompt, target="kv")
    assert torch.equal(v_only.layers[0].keys, prompt.layers[0].keys)
    assert not torch.equal(v_only.layers[0].values, prompt.layers[0].values)
    assert not torch.equal(both.layers[0].keys, prompt.layers[0].keys)
    assert not torch.equal(both.layers[0].values, prompt.layers[0].values)


def test_cache_clones_have_no_storage_aliases() -> None:
    prompt = _prompt_cache()
    left = clone_dynamic_cache(prompt)
    right = QuantizedDynamicCache.from_prompt_cache(prompt, target="identity")
    assert_no_storage_aliases([prompt, left, right])
    left.layers[0].keys.add_(10.0)
    assert not torch.equal(left.layers[0].keys, prompt.layers[0].keys)
    assert torch.equal(right.layers[0].keys, prompt.layers[0].keys)
