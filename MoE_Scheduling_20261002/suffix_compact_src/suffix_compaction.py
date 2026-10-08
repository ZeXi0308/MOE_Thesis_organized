"""One-time suffix compaction for eager, single-GPU OLMoE / FlashAttention.

Only execution tensors and a temporary attention context shrink. Scheduler
counts, request order, block tables, physical KV slots and native rollback stay
unchanged. The original sampler receives the original packed logits shape.
"""
from contextlib import ExitStack
from dataclasses import replace
import time

from suffix_runtime import packed_prefix_plan


def compact_query_plan(state):
    """CPU plan preserving request order and every mandatory query row."""
    counts = [k + 1 if eligible else q for q, k, eligible in zip(
        state.num_scheduled_tokens, state.keep_drafts, state.eligible)]
    if len(counts) != len(state.req_ids) or any(q < 1 for q in counts):
        raise ValueError("invalid compact query counts")
    indices, cumulative = [], [0]
    for start, old, new in zip(state.row_starts, state.num_scheduled_tokens, counts):
        if new > old:
            raise ValueError("compaction cannot grow a request")
        indices.extend(range(start, start + new))
        cumulative.append(cumulative[-1] + new)
    if indices != state.alive_indices():
        raise ValueError("compact rows differ from the qualified surviving prefix")
    seq_lens = [c + q for c, q in zip(state.computed_starts, counts)]
    return dict(indices=indices, query_lens=counts, query_start_loc=cumulative,
                seq_lens=seq_lens, context_lens=list(state.computed_starts),
                num_actual_tokens=len(indices), max_query_len=max(counts),
                max_seq_len=max(seq_lens))


def metadata_fallback_reason(metadata, slot_mapping, *, total_rows, num_reqs):
    if type(metadata).__name__ != "FlashAttentionMetadata":
        return "not_flash_attention"
    if (metadata.use_cascade or metadata.common_prefix_len != 0
            or metadata.causal is not True
            or metadata.dcp_context_kv_lens is not None
            or getattr(metadata, "mm_prefix_range_tensor", None) is not None
            or getattr(metadata, "rswa_prefix_lens", None) is not None
            or getattr(metadata, "rswa_window", None) is not None
            or metadata.sliding_window not in (None, (-1, -1))):
        return "nonordinary_attention_metadata"
    if (metadata.num_actual_tokens != total_rows
            or len(metadata.query_start_loc) != num_reqs + 1
            or len(metadata.seq_lens) != num_reqs
            or metadata.block_table.shape[0] != num_reqs
            or slot_mapping is None or len(slot_mapping) != total_rows
            or len(metadata.slot_mapping) != total_rows):
        return "attention_padding_or_row_mismatch"
    return None


def build_compact_context(context, layer_names, plan, *, total_rows, torch):
    """Copy context/metadata; return (new_context, None) or (None, reason).

    ForwardContext.slot_mapping is the actual cache-write input. Replacing only
    FlashAttentionMetadata.slot_mapping would leave KV writes on the old rows.
    No tensor owned by the runner is modified in-place.
    """
    if (not isinstance(context.attn_metadata, dict)
            or not isinstance(context.slot_mapping, dict)):
        return None, "microbatch_or_non_dict_context"
    if (context.dp_metadata is not None or context.ubatch_slices is not None
            or context.is_padding is not None
            or getattr(context.cudagraph_runtime_mode, "name", None) != "NONE"):
        return None, "parallel_padded_or_cudagraph_context"
    nreq = len(plan["query_lens"])
    for name in layer_names:
        metadata = context.attn_metadata.get(name)
        reason = metadata_fallback_reason(metadata, context.slot_mapping.get(name),
                                          total_rows=total_rows, num_reqs=nreq)
        if reason:
            return None, reason + ":" + name
    metadata_map, slot_map = dict(context.attn_metadata), dict(context.slot_mapping)
    metadata_cache, slot_cache = {}, {}
    for name in layer_names:
        old = context.attn_metadata[name]
        old_slots = context.slot_mapping[name]
        if id(old_slots) not in slot_cache:
            indices = torch.tensor(plan["indices"], dtype=torch.long, device=old_slots.device)
            slot_cache[id(old_slots)] = old_slots.index_select(0, indices)
        new_slots = slot_cache[id(old_slots)]
        key = (id(old), id(old_slots))
        if key not in metadata_cache:
            metadata_cache[key] = replace(old,
                num_actual_tokens=plan["num_actual_tokens"],
                query_start_loc=torch.tensor(plan["query_start_loc"],
                                             dtype=old.query_start_loc.dtype,
                                             device=old.query_start_loc.device),
                seq_lens=torch.tensor(plan["seq_lens"], dtype=old.seq_lens.dtype,
                                      device=old.seq_lens.device),
                max_query_len=plan["max_query_len"], max_seq_len=plan["max_seq_len"],
                slot_mapping=new_slots, scheduler_metadata=None,
                prefix_scheduler_metadata=None)
        metadata_map[name], slot_map[name] = metadata_cache[key], new_slots
    descriptor = context.batch_descriptor
    if descriptor is not None:
        descriptor = replace(descriptor, num_tokens=plan["num_actual_tokens"],
                             num_reqs=nreq, uniform=False)
    return replace(context, attn_metadata=metadata_map, slot_mapping=slot_map,
                   batch_descriptor=descriptor, is_padding=None, skip_compiled=True), None


class SuffixCompaction:
    """Install after SuffixController; expose report() and uninstall()."""

    def __init__(self, runner, controller):
        from vllm.forward_context import get_forward_context, override_forward_context

        self.runner, self.controller = runner, controller
        self.torch = controller.runtime.torch
        self.get_context, self.override_context = get_forward_context, override_forward_context
        self.outer, self.inner = runner.model, runner.model.model
        parallel = runner.parallel_config
        if (type(self.outer).__name__ != "OlmoeForCausalLM"
                or type(self.inner).__name__ != "OlmoeModel"
                or not getattr(self.inner, "do_not_compile", False)
                or not getattr(runner.model_config, "enforce_eager", False)
                or getattr(runner.model_config, "is_hybrid", False)
                or runner.use_async_scheduling
                or runner.speculative_config.method != "ngram"
                or any(getattr(parallel, name, 1) != 1 for name in (
                    "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"))
                or getattr(runner, "dcp_world_size", 1) != 1
                or parallel.use_ubatching
                or getattr(parallel, "use_sequence_parallel_moe", False)
                or self.inner.start_layer != 0
                or self.inner.end_layer != controller.num_layers):
            raise ValueError("compaction requires eager single-GPU CPU-ngram OLMoE")
        self.layers = list(self.inner.layers)
        self.layer_names = [layer.self_attn.attn.layer_name for layer in self.layers]
        for layer in self.layers:
            attention = layer.self_attn.attn
            if (type(attention.impl).__name__ != "FlashAttentionImpl"
                    or attention.kv_sharing_target_layer_name is not None):
                raise ValueError("compaction requires ordinary FlashAttention without KV sharing")
        self.records = []
        self.compacted_step_id = None
        self.last_record = None
        self.controller.compact_active = False
        self.controller.compact_original_indices = None
        self.original_forward, self.original_logits = self.inner.forward, self.outer.compute_logits
        self.had_forward = "forward" in vars(self.inner)
        self.had_logits = "compute_logits" in vars(self.outer)
        self.forward_hook, self.logits_hook = self._forward, self._compute_logits
        self.inner.forward, self.outer.compute_logits = self.forward_hook, self.logits_hook

    def _forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        state, torch = self.controller.state, self.torch
        self.compacted_step_id, self.last_record = None, None
        self.controller.compact_active = False
        self.controller.compact_original_indices = None
        record = dict(step_id=state.step_id, phase=self.controller.runtime.context.get("phase"),
                      status="started", compact_applied=False, fallback_reason=None,
                      layer_rows=[], logits_compacted=False)
        self.records.append(record)
        self.last_record = record
        if not state.active or state.sampled or intermediate_tensors is not None:
            record.update(status="fallback", fallback_reason="no_qualified_synchronous_step")
            return self.original_forward(input_ids, positions, intermediate_tensors, inputs_embeds)
        base_context, compact_context = self.get_context(), None
        compact_indices, original_rows = None, None
        started = time.perf_counter()
        try:
            with ExitStack() as scope:
                hidden = inputs_embeds if inputs_embeds is not None else self.inner.embed_input_ids(input_ids)
                residual = None
                original_rows = len(hidden)
                for index, layer in enumerate(self.layers):
                    record["layer_rows"].append(dict(layer=index, input_rows=len(hidden),
                        original_rows=original_rows, compact=self.controller.compact_active))
                    hidden, residual = layer(positions, hidden, residual)
                    if (index == self.controller.decision_layer
                            and state.keep_drafts != state.original_draft_counts):
                        compact_started = time.perf_counter()
                        if (original_rows != state.total_rows or positions.ndim != 1
                                or len(positions) != original_rows
                                or residual is None or len(residual) != original_rows):
                            record["fallback_reason"] = "padded_or_unfamiliar_hidden_shape"
                            continue
                        plan = compact_query_plan(state)
                        compact_context, reason = build_compact_context(
                            base_context, self.layer_names[index + 1:], plan,
                            total_rows=original_rows, torch=torch)
                        if reason:
                            record["fallback_reason"] = reason
                            continue
                        compact_indices = torch.tensor(plan["indices"], dtype=torch.long,
                                                       device=hidden.device)
                        hidden = hidden.index_select(0, compact_indices)
                        residual = residual.index_select(0, compact_indices)
                        positions = positions.index_select(0, compact_indices)
                        scope.enter_context(self.override_context(compact_context))
                        self.controller.compact_active = True
                        self.controller.compact_original_indices = list(plan["indices"])
                        self.compacted_step_id = state.step_id
                        record.update(compact_applied=True, after_layer=index, plan=plan,
                                      compact_setup_host_s=time.perf_counter() - compact_started)
                if residual is not None:
                    hidden, _ = self.inner.norm(hidden, residual)
                else:
                    hidden = self.inner.norm(hidden)
                if compact_indices is not None:
                    scatter_started = time.perf_counter()
                    full_hidden = hidden.new_zeros((original_rows, hidden.shape[-1]))
                    full_hidden.index_copy_(0, compact_indices, hidden)
                    hidden = full_hidden
                    record["hidden_scatter_host_s"] = time.perf_counter() - scatter_started
                record.update(status="complete", returned_rows=len(hidden))
                if not record["compact_applied"] and record["fallback_reason"] is None:
                    record["fallback_reason"] = "no_cut"
                return hidden
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            # Keep the cold-start MoE layer-string cursor consistent with the
            # original context even though a nested copy was used for the tail.
            if compact_context is not None:
                base_context.moe_layer_index = compact_context.moe_layer_index
            self.controller.compact_active = False
            self.controller.compact_original_indices = None
            record["forward_host_s"] = time.perf_counter() - started

    def _compute_logits(self, hidden_states):
        state = self.controller.state
        if self.compacted_step_id != state.step_id or not state.active or state.sampled:
            return self.original_logits(hidden_states)
        started = time.perf_counter()
        expected = sum(state.original_draft_counts) + len(state.req_ids)
        if len(hidden_states) != expected:
            raise ValueError("LM-head input lost original packed request segments")
        plan = packed_prefix_plan(state.original_draft_counts, state.keep_drafts)
        take = self.torch.tensor(plan["logit_take"], dtype=self.torch.long,
                                 device=hidden_states.device)
        kept_logits = self.original_logits(hidden_states.index_select(0, take))
        if kept_logits is None:
            raise ValueError("single-rank OLMoE logits unexpectedly absent")
        logits = kept_logits.new_zeros((expected, kept_logits.shape[-1]))
        logits.index_copy_(0, take, kept_logits)
        self.last_record.update(logits_compacted=True, logits_original_rows=expected,
                                logits_executed_rows=len(take), logits_kept_indices=plan["logit_take"],
                                logits_adapter_host_s=time.perf_counter() - started)
        return logits

    def report(self):
        return dict(enabled=True, scope="eager_single_gpu_olmoe_cpu_ngram_flash_attention",
                    scheduler_counts_modified=False, kv_blocks_reallocated=False,
                    source_timing="all compaction and logits work is inside engine.step",
                    steps=self.records,
                    compact_steps=sum(r["compact_applied"] for r in self.records),
                    compact_logits_steps=sum(r["logits_compacted"] for r in self.records))

    def uninstall(self):
        if self.inner.forward is self.forward_hook:
            if self.had_forward:
                self.inner.forward = self.original_forward
            else:
                del self.inner.forward
        if self.outer.compute_logits is self.logits_hook:
            if self.had_logits:
                self.outer.compute_logits = self.original_logits
            else:
                del self.outer.compute_logits
        self.controller.compact_active = False
        self.controller.compact_original_indices = None
