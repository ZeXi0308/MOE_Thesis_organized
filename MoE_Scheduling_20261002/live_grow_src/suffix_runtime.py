"""Legal speculative suffix state and sampling hooks for synchronous OLMoE.

The MoE wrapper owns gather/execute/scatter.  This module establishes actual
post-reorder input row identities and samples only surviving prefix logits.
It deliberately leaves SchedulerOutput and its original draft counts intact:
native scheduler rejection accounting rolls back every uncommitted position.

Prototype scope: vLLM 0.26 CPU ngram, greedy, ordinary causal attention, TP/PP1,
eager, no hybrid state, no penalties/grammar/custom logits processors.  KV
physical slots are not freed or cleared here; native valid-length accounting
must exclude rejected suffixes and overwrite them before subsequent use.
"""
from __future__ import annotations

from numbers import Integral
import time


def _ints(values, name, minimum=0):
    values = list(values)
    if any(not isinstance(v, Integral) or isinstance(v, bool) or v < minimum for v in values):
        raise ValueError(f"invalid {name}")
    return [int(v) for v in values]


def packed_prefix_plan(original_counts, keep_counts):
    """CPU-only index plan; logits have one original (draft_count+1) segment/request.

    In particular, keep=0 selects the anchor's logits, which produce one target
    token.  It never selects the old bonus at the end of the original segment.
    """
    original = _ints(original_counts, "original draft counts")
    keep = _ints(keep_counts, "kept draft counts")
    if len(original) != len(keep) or not original or any(k > d for d, k in zip(original, keep)):
        raise ValueError("kept draft counts must fit original request segments")
    logit_take, draft_take, target, bonus, cu_draft, cu_sampled = [], [], [], [], [], []
    old_logit = old_draft = new_logit = new_draft = 0
    for d, k in zip(original, keep):
        logit_take.extend(range(old_logit, old_logit + k + 1))
        draft_take.extend(range(old_draft, old_draft + k))
        target.extend(range(new_logit, new_logit + k))
        bonus.append(new_logit + k)
        old_logit += d + 1
        old_draft += d
        new_logit += k + 1
        new_draft += k
        cu_sampled.append(new_logit)
        cu_draft.append(new_draft)
    return dict(logit_take=logit_take, draft_take=draft_take,
                target_logits_indices=target, bonus_logits_indices=bonus,
                cu_num_draft_tokens=cu_draft, cu_num_sampled_tokens=cu_sampled)


class SuffixStepState:
    """Shared by runner hooks and an external MoE policy; no torch dependency."""

    def __init__(self):
        self.step_id = -1
        self.active = False
        self.steps = []
        self.events = []
        self.req_ids = []
        self.num_scheduled_tokens = []
        self.row_starts = []
        self.computed_starts = []
        self.prompt_tokens = []
        self.original_draft_counts = []
        self.keep_drafts = []
        self.eligible = []
        self.total_rows = 0
        self.sampled = False

    def begin_step(self, *, req_ids, num_scheduled_tokens, computed_starts,
                   prompt_tokens, original_draft_counts, row_indices=None):
        ids = list(req_ids)
        counts = _ints(num_scheduled_tokens, "scheduled token counts", 1)
        starts = _ints(computed_starts, "computed positions")
        prompts = _ints(prompt_tokens, "prompt lengths", 1)
        drafts = _ints(original_draft_counts, "original draft counts")
        n = len(ids)
        if (not n or any(len(v) != n for v in (counts, starts, prompts, drafts))
                or any(not isinstance(rid, str) or not rid for rid in ids)
                or len(set(ids)) != n or any(d + 1 > c for d, c in zip(drafts, counts))):
            raise ValueError("request identities/counts do not align")
        expected = [i for i, count in enumerate(counts) for _ in range(count)]
        if row_indices is not None and _ints(row_indices, "runner row indices") != expected:
            raise ValueError("runner rows do not follow post-reorder request segments")
        row_starts, cursor = [], 0
        for count in counts:
            row_starts.append(cursor)
            cursor += count
        self.step_id += 1
        self.active = True
        self.sampled = False
        self.req_ids, self.num_scheduled_tokens = ids, counts
        self.row_starts, self.computed_starts = row_starts, starts
        self.prompt_tokens, self.original_draft_counts = prompts, drafts
        self.keep_drafts = list(drafts)
        # Exclude prefill/recovery chunks and any unfamiliar verification shape.
        self.eligible = [s >= p and c == d + 1 and d > 0
                         for s, p, c, d in zip(starts, prompts, counts, drafts)]
        self.total_rows = cursor
        self.steps.append(self.snapshot())
        return self

    def snapshot(self):
        return dict(step_id=self.step_id, req_ids=list(self.req_ids),
                    num_scheduled_tokens=list(self.num_scheduled_tokens),
                    row_starts=list(self.row_starts), computed_starts=list(self.computed_starts),
                    prompt_tokens=list(self.prompt_tokens),
                    original_draft_counts=list(self.original_draft_counts),
                    keep_drafts=list(self.keep_drafts), eligible=list(self.eligible),
                    total_rows=self.total_rows)

    def set_keep_drafts(self, keep_drafts, *, layer=None):
        keep = _ints(keep_drafts, "kept draft counts")
        if not self.active or self.sampled or len(keep) != len(self.keep_drafts):
            raise ValueError("suffix mutation requires an active unsampled step")
        if any(k > old for k, old in zip(keep, self.keep_drafts)):
            raise ValueError("a discarded suffix cannot become alive again")
        if any(k != old and not eligible
               for k, old, eligible in zip(keep, self.keep_drafts, self.eligible)):
            raise ValueError("prefill/recovery/non-speculative rows must be retained")
        if keep != self.keep_drafts:
            self.events.append(dict(step_id=self.step_id, layer=layer,
                                    before=list(self.keep_drafts), after=list(keep)))
            self.keep_drafts = keep
            self.steps[-1]["keep_drafts"] = list(keep)

    def alive_indices(self, total_rows=None):
        if not self.active or self.sampled:
            raise ValueError("no active unsampled row mapping")
        total = self.total_rows if total_rows is None else _ints([total_rows], "tensor row count")[0]
        if total < self.total_rows:
            raise ValueError("MoE tensor is shorter than the scheduled input")
        result = []
        for start, count, keep, eligible in zip(
                self.row_starts, self.num_scheduled_tokens, self.keep_drafts, self.eligible):
            result.extend(range(start, start + (keep + 1 if eligible else count)))
        # Padding remains intact; it has no qualified request identity.
        result.extend(range(self.total_rows, total))
        return result


def rebuild_prefix_metadata(logits, metadata, keep_drafts):
    """Return packed prefix logits and a fresh metadata object (no scheduler edits)."""
    import torch

    original = _ints(metadata.num_draft_tokens, "metadata draft counts")
    plan = packed_prefix_plan(original, keep_drafts)
    if logits is None or logits.ndim != 2 or len(logits) != sum(original) + len(original):
        raise ValueError("logits do not match original packed verification segments")
    if len(metadata.draft_token_ids) != sum(original):
        raise ValueError("draft tensor does not match original segment lengths")
    take_logits = torch.tensor(plan["logit_take"], dtype=torch.long, device=logits.device)
    short_logits = logits.index_select(0, take_logits).contiguous()
    if not any(keep_drafts):
        # Use the ordinary sampler for a fully anchor-only batch.  All future
        # bookkeeping still receives the original scheduler_output.
        return short_logits, None, plan
    take_drafts = torch.tensor(plan["draft_take"], dtype=torch.long,
                               device=metadata.draft_token_ids.device)
    fields = {}
    for name in ("target_logits_indices", "bonus_logits_indices",
                 "cu_num_draft_tokens", "cu_num_sampled_tokens"):
        template = getattr(metadata, name)
        fields[name] = torch.tensor(plan[name], dtype=template.dtype, device=template.device)
    fields.update(draft_token_ids=metadata.draft_token_ids.index_select(0, take_drafts).contiguous(),
                  num_draft_tokens=list(keep_drafts),
                  logits_indices=metadata.logits_indices.index_select(
                      0, take_logits.to(metadata.logits_indices.device)).contiguous())
    # Construction recalculates max_spec_len in SpecDecodeMetadata.__post_init__.
    return short_logits, type(metadata)(**fields), plan


def install_suffix_runtime(runner, state):
    """Install two instance hooks; return uninstall(). Install after engine loading."""
    if not isinstance(state, SuffixStepState):
        raise TypeError("state must be a SuffixStepState")
    spec = runner.speculative_config
    if (runner.use_async_scheduling or not runner.num_spec_tokens or spec is None
            or spec.method != "ngram" or runner.parallel_config.use_ubatching
            or getattr(runner.parallel_config, "tensor_parallel_size", 1) != 1
            or getattr(runner.parallel_config, "pipeline_parallel_size", 1) != 1
            or getattr(runner.model_config, "is_hybrid", False)):
        raise ValueError("suffix hooks require synchronous TP/PP1 non-hybrid CPU ngram")
    originals = {name: getattr(runner, name) for name in ("_prepare_inputs", "_sample")}
    had_instance = {name: name in vars(runner) for name in originals}

    def prepare(scheduler_output, num_scheduled_tokens):
        result = originals["_prepare_inputs"](scheduler_output, num_scheduled_tokens)
        batch = runner.input_batch
        n = batch.num_reqs
        ids = list(batch.req_ids)
        counts = _ints(num_scheduled_tokens, "runner scheduled counts", 1)
        if (len(ids) != n or set(ids) != set(scheduler_output.num_scheduled_tokens)
                or counts != [scheduler_output.num_scheduled_tokens[rid] for rid in ids]
                or sum(counts) != scheduler_output.total_num_scheduled_tokens):
            raise ValueError("scheduler and post-reorder input batch differ")
        drafts = scheduler_output.scheduled_spec_decode_tokens
        original_counts = [len(drafts.get(rid, ())) for rid in ids]
        metadata = result[1]
        if metadata is not None and list(metadata.num_draft_tokens) != original_counts:
            raise ValueError("runner metadata and scheduled draft identities differ")
        if metadata is None and any(original_counts):
            raise ValueError("scheduled drafts lack speculative metadata")
        state.begin_step(req_ids=ids, num_scheduled_tokens=counts,
                         computed_starts=batch.num_computed_tokens_cpu[:n],
                         prompt_tokens=batch.num_prompt_tokens[:n],
                         original_draft_counts=original_counts,
                         row_indices=runner.req_indices.np[:sum(counts)])
        return result

    def sample(logits, spec_decode_metadata):
        if not state.active or state.sampled:
            raise ValueError("sampling without exactly one prepared input batch")
        started = time.perf_counter()
        sampling = runner.input_batch.sampling_metadata
        if not sampling.all_greedy:
            raise ValueError("content-dependent suffix cancellation is restricted to greedy")
        if list(runner.input_batch.req_ids) != state.req_ids:
            raise ValueError("request order changed after row mapping")
        changed = state.keep_drafts != state.original_draft_counts
        if changed:
            from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor

            if spec_decode_metadata is None or list(spec_decode_metadata.num_draft_tokens) != state.original_draft_counts:
                raise ValueError("sampling metadata lost the original draft accounting")
            if (not sampling.no_penalties or sampling.bad_words_token_ids
                    or sampling.allowed_token_ids_mask is not None
                    or getattr(sampling, "thinking_budget_state_holder", None) is not None
                    or any(type(processor) is not MinTokensLogitsProcessor
                           for processor in sampling.logitsprocs.non_argmax_invariant)):
                raise ValueError("prototype suffix sampling excludes penalties and custom constraints")
            # v0.26 always installs its built-in MinTokens processor for spec
            # decode, even at min_tokens=0. Its target path consumes the new
            # draft counts; its bonus path uses committed output lengths only.
            logits, spec_decode_metadata, plan = rebuild_prefix_metadata(
                logits, spec_decode_metadata, state.keep_drafts)
            state.steps[-1]["sample_prefix_plan"] = plan
        state.steps[-1]["sample_repack_host_ms"] = (time.perf_counter() - started) * 1000
        output = originals["_sample"](logits, spec_decode_metadata)
        state.sampled = True
        state.steps[-1]["sampled"] = True
        state.steps[-1]["sampler_output_width"] = int(output.sampled_token_ids.shape[-1])
        return output

    runner._prepare_inputs, runner._sample = prepare, sample

    def uninstall():
        for name, hook in (("_prepare_inputs", prepare), ("_sample", sample)):
            if getattr(runner, name, None) is hook:
                if had_instance[name]:
                    setattr(runner, name, originals[name])
                else:
                    delattr(runner, name)
        state.active = False

    return uninstall
