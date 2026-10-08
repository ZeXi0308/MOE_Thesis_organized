#!/usr/bin/env python3
"""Small, standalone vLLM 0.26 FA2 correctness check; no model/engine/TP group.

Run after other GPU work has stopped:
  python test_compaction_attention_gpu.py --output compaction_attention_gpu.json
  # If the script and runtime sources were uploaded separately:
  python test_compaction_attention_gpu.py --src-dir /path/to/suffix_compact_src

Tests BF16 Q/K/V and cache against a CPU FP32 dense causal reference computed
from those same quantized inputs. FA2 does not accept FP32 Q/K/V. Tolerances are
absolute + relative, per element: FA2/reference 0.01 + 0.01*abs(reference),
full/compact 0.004 + 0.004*abs(full), FP32 reference/reference 1e-6 + 1e-6*abs(ref).
KV writes and untouched cache contents must be bit-exact. These are attention
interface tests, not model-output/greedy-token bit-equivalence tests.

The standalone harness explicitly sets the native HND KV layout used by the
serving runs; it does not ask a KV connector or construct an engine config.

Native interfaces checked against the v0.26.0 source:
  vllm/v1/attention/backends/flash_attn.py
  vllm/model_executor/layers/attention/attention.py (unified_kv_cache_update)
  vllm/forward_context.py
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


LAYER = "model.layers.1.self_attn.attn"
BLOCK_SIZE, NUM_BLOCKS, HEADS, KV_HEADS, HEAD_SIZE = 16, 24, 4, 2, 64
DENSE_TOL = dict(atol=0.01, rtol=0.01)
PAIR_TOL = dict(atol=0.004, rtol=0.004)
FP32_TOL = dict(atol=1e-6, rtol=1e-6)


def run(args):
    import torch
    import vllm
    from vllm.forward_context import (
        BatchDescriptor, ForwardContext, get_forward_context,
        is_forward_context_available, override_forward_context,
    )
    from vllm.model_executor.layers.attention.attention import unified_kv_cache_update
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionBackend, FlashAttentionImpl, FlashAttentionMetadata,
    )
    from vllm.v1.attention.backends.fa_utils import is_fa_version_supported
    from vllm.v1.attention.backends.utils import get_kv_cache_layout, set_kv_cache_layout

    if vllm.__version__.split("+")[0] != "0.26.0":
        raise RuntimeError(f"This bounded test requires vLLM 0.26.0, got {vllm.__version__}")
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("This test requires one available NVIDIA CUDA GPU")
    if not torch.cuda.is_bf16_supported() or not is_fa_version_supported(2):
        raise RuntimeError("This device/build does not support BF16 FlashAttention 2")
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise RuntimeError("This test only supports a single process / TP1")
    if is_forward_context_available():
        raise RuntimeError("Run this script independently, outside an engine forward")
    # Match the real serving layout through vLLM's native override. Without this,
    # get_kv_cache_stride_order() asks the KV connector for its default, which
    # requires a current VllmConfig even though this standalone test has no engine.
    # set_kv_cache_layout also clears the native getter's functools cache.
    set_kv_cache_layout("HND")
    assert get_kv_cache_layout() == "HND"

    src = Path(args.src_dir).resolve()
    sys.path.insert(0, str(src))
    from suffix_compaction import build_compact_context, compact_query_plan
    from suffix_runtime import SuffixStepState

    device, dtype = torch.device("cuda:0"), torch.bfloat16
    scale = HEAD_SIZE ** -0.5
    impl = FlashAttentionImpl(num_heads=HEADS, head_size=HEAD_SIZE, scale=scale,
        num_kv_heads=KV_HEADS, alibi_slopes=None, sliding_window=None,
        kv_cache_dtype="auto")
    # Select the real FA2 branch directly; no mocked kernels/config/engine.
    # The native support check above makes an unavailable FA2 an explicit error.
    impl.vllm_flash_attn_version = 2
    assert impl.dcp_world_size == impl.pcp_world_size == 1
    assert not FlashAttentionBackend.forward_includes_kv_cache_update
    shape = FlashAttentionBackend.get_kv_cache_shape(
        NUM_BLOCKS, BLOCK_SIZE, KV_HEADS, HEAD_SIZE)
    order = FlashAttentionBackend.get_kv_cache_stride_order()
    assert order == (0, 1, 2, 3), f"Unexpected HND physical stride order: {order}"
    inverse = tuple(order.index(i) for i in range(len(order)))

    def clone_cache(cpu_values):
        # Match the backend's real logical shape AND preferred physical strides.
        storage = torch.empty(tuple(shape[i] for i in order), dtype=dtype, device=device)
        cache = storage.permute(inverse)
        cache.copy_(cpu_values)
        return cache

    def make_context(meta, slots, cache):
        one = torch.ones((), dtype=torch.float32, device=device)
        layer = SimpleNamespace(impl=impl, kv_cache=cache,
                                _q_scale=one, _k_scale=one, _v_scale=one)
        return ForwardContext(no_compile_layers={LAYER: layer},
            attn_metadata={LAYER: meta}, slot_mapping={LAYER: slots},
            batch_descriptor=BatchDescriptor(meta.num_actual_tokens, 5),
            skip_compiled=True)

    def execute(context, q, k, v):
        output = torch.empty_like(q)
        with override_forward_context(context):
            assert get_forward_context() is context
            # Real vLLM context lookup -> real reshape_and_cache_flash kernel.
            unified_kv_cache_update(k, v, LAYER)
            layer = context.no_compile_layers[LAYER]
            returned = impl.forward(layer, q, k, v, layer.kv_cache,
                                    context.attn_metadata[LAYER], output)
            assert returned.data_ptr() == output.data_ptr()
        assert not is_forward_context_available(), "ForwardContext leaked"
        torch.cuda.synchronize()
        return output.float().cpu()

    def compare(actual, expected, tolerance, label):
        diff = (actual.float() - expected.float()).abs()
        torch.testing.assert_close(actual, expected, **tolerance, msg=label)
        return dict(max_abs=float(diff.max()), mean_abs=float(diff.mean()), **tolerance)

    def expected_update(base, slots_cpu, keys_cpu, values_cpu):
        expected = base.clone()
        for row, slot in enumerate(slots_cpu):
            block, offset = divmod(slot, BLOCK_SIZE)
            expected[block, :, offset, :HEAD_SIZE] = keys_cpu[row]
            expected[block, :, offset, HEAD_SIZE:] = values_cpu[row]
        return expected

    reports = []
    for seed in args.seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)

        def random_bf16(*dims):
            return torch.randn(dims, generator=generator, dtype=torch.float32).to(dtype)

        # pure prefill; context-bearing prefill crossing a page; N2 decode
        # crossing a page; N4 decode crossing a page; ordinary AR decode.
        counts, computed = [5, 7, 3, 5, 1], [0, 13, 15, 31, 17]
        drafts, prompts = [0, 0, 2, 4, 0], [9, 40, 8, 8, 8]
        state = SuffixStepState().begin_step(
            req_ids=["prefill", "extend", "n2", "n4", "ar"],
            num_scheduled_tokens=counts, computed_starts=computed,
            prompt_tokens=prompts, original_draft_counts=drafts)
        rows, old_starts = state.total_rows, [0]
        for count in counts:
            old_starts.append(old_starts[-1] + count)
        blocks_cpu = [[11, 4, 17], [2, 19, 6], [14, 1, 21],
                      [8, 23, 3], [16, 10, 5]]
        # Unique, non-contiguous physical blocks, not row-index-as-slot mapping.
        assert len(set(sum(blocks_cpu, []))) == 15
        slots_cpu = [blocks_cpu[r][pos // BLOCK_SIZE] * BLOCK_SIZE + pos % BLOCK_SIZE
                     for r, (c, q) in enumerate(zip(computed, counts))
                     for pos in range(c, c + q)]
        assert len(set(slots_cpu)) == rows
        q_cpu = random_bf16(rows, HEADS, HEAD_SIZE)
        logical_k = [random_bf16(c + q, KV_HEADS, HEAD_SIZE)
                     for c, q in zip(computed, counts)]
        logical_v = [random_bf16(c + q, KV_HEADS, HEAD_SIZE)
                     for c, q in zip(computed, counts)]
        k_cpu = torch.cat([k[c:] for k, c in zip(logical_k, computed)])
        v_cpu = torch.cat([v[c:] for v, c in zip(logical_v, computed)])
        base = random_bf16(*shape)  # Includes deliberately stale unwritten tails.
        for r, c in enumerate(computed):
            for pos in range(c):
                block, offset = blocks_cpu[r][pos // BLOCK_SIZE], pos % BLOCK_SIZE
                base[block, :, offset, :HEAD_SIZE] = logical_k[r][pos]
                base[block, :, offset, HEAD_SIZE:] = logical_v[r][pos]
        q, k, v = [t.to(device) for t in (q_cpu, k_cpu, v_cpu)]
        slots = torch.tensor(slots_cpu, dtype=torch.int64, device=device)
        meta = FlashAttentionMetadata(num_actual_tokens=rows,
            max_query_len=max(counts),
            query_start_loc=torch.tensor(old_starts, dtype=torch.int32, device=device),
            max_seq_len=max(c + n for c, n in zip(computed, counts)),
            seq_lens=torch.tensor([c + n for c, n in zip(computed, counts)],
                                 dtype=torch.int32, device=device),
            block_table=torch.tensor(blocks_cpu, dtype=torch.int32, device=device),
            # Intentionally wrong metadata slots: the writer MUST use context slots.
            slot_mapping=torch.full_like(slots, -1), use_cascade=False,
            common_prefix_len=0, cu_prefix_query_lens=None,
            prefix_kv_lens=None, suffix_kv_lens=None, causal=True)

        def dense_reference(query_counts):
            result = []
            for req, nquery in enumerate(query_counts):
                for j in range(nquery):
                    length = computed[req] + j + 1
                    query = q_cpu[old_starts[req] + j].float()
                    keys = logical_k[req][:length].float().repeat_interleave(
                        HEADS // KV_HEADS, dim=1)
                    values = logical_v[req][:length].float().repeat_interleave(
                        HEADS // KV_HEADS, dim=1)
                    weights = torch.einsum("hd,thd->ht", query, keys) * scale
                    result.append(torch.einsum("ht,thd->hd", weights.softmax(-1), values))
            return torch.stack(result)

        full_context = make_context(meta, slots, clone_cache(base))
        full = execute(full_context, q, k, v)
        full_expected_cache = expected_update(base, slots_cpu, k_cpu, v_cpu)
        torch.testing.assert_close(full_context.no_compile_layers[LAYER].kv_cache.cpu(),
                                   full_expected_cache, atol=0, rtol=0)
        reference_full = dense_reference(counts)
        full_error = compare(full, reference_full, DENSE_TOL, "full vs FP32 dense")

        for label, keep in [("no_cut", drafts), ("partial", [0, 0, 1, 2, 0]),
                            ("fixed0", [0, 0, 0, 0, 0])]:
            # Monotone legal reductions: partial then fixed0 share one step state.
            state.set_keep_drafts(keep)
            plan = compact_query_plan(state)
            original = make_context(meta, slots, clone_cache(base))
            compact, reason = build_compact_context(original, [LAYER], plan,
                                                    total_rows=rows, torch=torch)
            assert reason is None and compact is not None, reason
            cm = compact.attn_metadata[LAYER]
            torch.testing.assert_close(cm.seq_lens - cm.query_start_loc.diff(),
                torch.tensor(computed, device=device, dtype=torch.int32), atol=0, rtol=0)
            assert cm.block_table is meta.block_table
            assert original.attn_metadata[LAYER] is meta
            assert original.slot_mapping[LAYER] is slots
            torch.testing.assert_close(meta.query_start_loc.cpu(),
                torch.tensor(old_starts, dtype=torch.int32), atol=0, rtol=0)
            indices = torch.tensor(plan["indices"], device=device, dtype=torch.int64)
            selected_slots = [slots_cpu[i] for i in plan["indices"]]
            assert cm.slot_mapping.cpu().tolist() == selected_slots
            assert compact.slot_mapping[LAYER].cpu().tolist() == selected_slots
            # Keep read/write sources deliberately distinct through the REAL writer.
            compact = replace(compact, attn_metadata={LAYER: replace(cm,
                slot_mapping=torch.full_like(cm.slot_mapping, -1))})
            cq, ck, cv = [t.index_select(0, indices) for t in (q, k, v)]
            actual = execute(compact, cq, ck, cv)
            expected = expected_update(base, selected_slots,
                k_cpu[plan["indices"]], v_cpu[plan["indices"]])
            torch.testing.assert_close(compact.no_compile_layers[LAYER].kv_cache.cpu(),
                                       expected, atol=0, rtol=0)
            reference = dense_reference(plan["query_lens"])
            full_kept = full[plan["indices"]]
            record = dict(seed=seed, case=label, original_rows=rows,
                compact_rows=len(plan["indices"]), plan=plan,
                physical_slots=selected_slots, full_dense_error=full_error,
                compact_dense_error=compare(actual, reference, DENSE_TOL, label+" dense"),
                full_compact_error=compare(actual, full_kept, PAIR_TOL, label+" prefix"),
                reference_prefix_error=compare(reference,
                    reference_full[plan["indices"]], FP32_TOL, label+" reference"),
                kv_exact=True, independent_slot_mapping=True)
            if label == "fixed0":
                # Negative control: shrinking Q but retaining old seq lengths shifts
                # FA2's right-aligned causal boundary. It MUST produce a detectable error.
                wrong_meta = replace(cm, seq_lens=meta.seq_lens, max_seq_len=meta.max_seq_len)
                wrong = make_context(wrong_meta, compact.slot_mapping[LAYER],
                                     clone_cache(full_expected_cache))
                wrong_output = execute(wrong, cq, ck, cv)
                error = (wrong_output - reference).abs()
                violations = error > (DENSE_TOL["atol"] + DENSE_TOL["rtol"]*reference.abs())
                assert bool(violations.any()), "Fixture failed to detect stale seq_lens"
                record["stale_seq_negative_control"] = dict(
                    max_abs=float(error.max()), violating_elements=int(violations.sum()))
                # Opposite slot control: valid metadata + all--1 context slots causes
                # zero writes, proving that the independent mapping governs the kernel.
                skipped = make_context(cm, torch.full_like(cm.slot_mapping, -1), clone_cache(base))
                with override_forward_context(skipped):
                    unified_kv_cache_update(ck, cv, LAYER)
                torch.cuda.synchronize()
                torch.testing.assert_close(skipped.no_compile_layers[LAYER].kv_cache.cpu(),
                                           base, atol=0, rtol=0)
                assert not is_forward_context_available()
                record["independent_slots_negative_control"] = "all -1 context slots: zero writes"
            reports.append(record)

    return dict(status="PASS", scope="vLLM 0.26 FA2 TP1 paged causal attention only",
        torch_version=torch.__version__, vllm_version=vllm.__version__,
        device=torch.cuda.get_device_name(0), fa_version=impl.vllm_flash_attn_version,
        dtype=str(dtype), reference_dtype="CPU FP32 on BF16-quantized inputs",
        cache_shape=list(shape), cache_layout=get_kv_cache_layout(),
        cache_stride_order=list(order),
        source_sha256={name: hashlib.sha256((src/name).read_bytes()).hexdigest()
                       for name in ("suffix_compaction.py", "suffix_runtime.py")},
        cases=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src-dir", default=str(Path(__file__).parent/"suffix_compact_src"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 17])
    parser.add_argument("--output", type=Path, help="Optional full JSON result path")
    args = parser.parse_args()
    started = time.perf_counter()
    report = run(args)  # Failures intentionally raise; never report a skipped GPU as PASS.
    report["wall_s_including_imports"] = time.perf_counter() - started
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
