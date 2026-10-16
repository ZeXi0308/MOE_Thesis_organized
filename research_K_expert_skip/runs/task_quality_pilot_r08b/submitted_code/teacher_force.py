"""Request-keyed teacher forcing and full-distribution quality for vLLM 0.26.

Use logits_processors=["teacher_force:TeacherForceLogitsProcessor"]. Metrics
are collected before this processor masks logits. Sampling parameters must
have neutral penalties, no logit bias, no min-p and no allowed-token filter.
One CPU float32 reference distribution is retained per request/position for
the current comparison group; the last candidate releases it. This is quality
instrumentation and must never be timed as inference performance.

Interface verified against the v0.26.0 tag:
https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/sample/logits_processor/interface.py
https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/sample/logits_processor/__init__.py
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path

try:
    from vllm.v1.sample.logits_processor.interface import LogitsProcessor
except ModuleNotFoundError as exc:
    if exc.name != "vllm":
        raise
    # Permit dependency-free state-machine checks on the CPU authoring host.
    # Real inference imports vLLM first and therefore uses its actual ABC.
    LogitsProcessor = object

EXTRA_KEY = "expert_skip_teacher_force"


def update_request_states(states, update, make_state):
    """Apply the documented removed -> added -> moved persistent-batch order."""
    if update is None:
        return
    for index in update.removed:
        states.pop(index, None)
    for index, params, prompt_ids, output_ids in update.added:
        state = make_state(params, prompt_ids, output_ids)
        if state is None:
            states.pop(index, None)
        else:
            states[index] = state
    for first, second, direction in update.moved:
        kind = getattr(direction, "name", str(direction))
        if kind == "SWAP":
            first_state, second_state = states.pop(first, None), states.pop(second, None)
            if first_state is not None:
                states[second] = first_state
            if second_state is not None:
                states[first] = second_state
        elif kind == "UNIDIRECTIONAL":
            state = states.pop(first, None)
            states.pop(second, None)
            if state is not None:
                states[second] = state
        else:
            raise ValueError(f"Unrecognized vLLM move: {direction}")


def quality_summary(rows):
    """Token-weighted NLL/PPL, never a mean of individual-sequence PPLs."""
    if not rows:
        return {"tokens": 0, "mean_nll": None, "ppl": None, "mean_kl": None}
    mean_nll = math.fsum(r["nll"] for r in rows) / len(rows)
    return {
        "tokens": len(rows),
        "documents": len({(r["group"], r["doc_key"]) for r in rows}),
        "mean_nll": mean_nll,
        "ppl": math.exp(mean_nll) if mean_nll < 709 else None,
        "mean_kl": math.fsum(r["kl_base_to_policy"] for r in rows) / len(rows),
        "max_kl": max(r["kl_base_to_policy"] for r in rows),
        "min_kl": min(r["kl_base_to_policy"] for r in rows),
        "mean_nll_delta": math.fsum(r["nll"] - r["baseline_nll"] for r in rows) / len(rows),
    }


class TeacherForceLogitsProcessor(LogitsProcessor):
    @classmethod
    def validate_params(cls, sampling_params):
        config = (sampling_params.extra_args or {}).get(EXTRA_KEY)
        if config is None:
            return
        required = {"group", "doc_key", "arm", "role", "forced_tokens",
                    "metrics_path", "expected_batch", "prefix_tokens"}
        missing = required - config.keys()
        if missing:
            raise ValueError(f"Teacher-forcing configuration missing {sorted(missing)}")
        tokens = config["forced_tokens"]
        if not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("forced_tokens must be nonnegative integer token IDs")
        if config["role"] not in {"baseline", "candidate"}:
            raise ValueError("role must be baseline or candidate")
        if sampling_params.max_tokens != len(tokens) or not sampling_params.ignore_eos:
            raise ValueError("max_tokens must match forced_tokens and ignore_eos must be true")
        for name, neutral in [("presence_penalty", 0), ("frequency_penalty", 0),
                              ("repetition_penalty", 1), ("min_p", 0), ("min_tokens", 0)]:
            if getattr(sampling_params, name, neutral) != neutral:
                raise ValueError(f"Quality capture requires neutral {name}")
        for name in ("logit_bias", "allowed_token_ids", "bad_words", "stop", "stop_token_ids"):
            if getattr(sampling_params, name, None):
                raise ValueError(f"Quality capture forbids {name}")
        if not Path(config["metrics_path"]).is_absolute():
            raise ValueError("metrics_path must be absolute")

    def __init__(self, vllm_config, device, is_pin_memory):
        self._requests = {}
        self._reference = {}
        self._reference_bytes = 0
        self._group = None
        self._seen = set()
        self._call = 0

    def is_argmax_invariant(self):
        return False

    def _new_state(self, params, prompt_ids, output_ids):
        config = (params.extra_args or {}).get(EXTRA_KEY)
        if config is None:
            return None
        self.validate_params(params)
        if prompt_ids is None or len(prompt_ids) != config["prefix_tokens"]:
            raise ValueError("Prompt length differs from controlled prefix length")
        if config["role"] == "baseline" and config["group"] != self._group:
            if self._reference:
                raise RuntimeError("Previous group's references were not released by its last arm")
            self._group = config["group"]
            self._seen.clear()
        if config["group"] != self._group:
            raise RuntimeError("Candidate has no matching baseline group")
        # Keep the actual running list reference, never a copy.
        return {"config": dict(config), "output_ids": output_ids}

    def update_state(self, batch_update):
        update_request_states(self._requests, batch_update, self._new_state)

    def apply(self, logits):
        if not self._requests:
            return logits  # Engine initialization/warm-up without task requests.
        import torch

        if logits.ndim != 2 or set(self._requests) != set(range(logits.shape[0])):
            raise RuntimeError("Logits rows do not match controlled request batch")
        signatures = set()
        for state in self._requests.values():
            cfg, output = state["config"], state["output_ids"]
            signatures.add((cfg["group"], cfg["arm"], len(output)))
            if logits.shape[0] != cfg["expected_batch"]:
                raise RuntimeError(f"Requested decode B={cfg['expected_batch']}, observed {logits.shape[0]}")
        if len(signatures) != 1:
            raise RuntimeError("Controlled equal-length requests are not at the same output position")
        self._call += 1
        by_path = defaultdict(list)
        for index, state in sorted(self._requests.items()):
            cfg, output = state["config"], state["output_ids"]
            position = len(output)
            forced = cfg["forced_tokens"]
            if position >= len(forced) or list(output) != forced[:position]:
                raise RuntimeError("Teacher-forced transcript diverged or generated too many tokens")
            target = forced[position]
            if target >= logits.shape[-1]:
                raise ValueError("Forced token outside vocabulary")
            key = (cfg["group"], cfg["doc_key"], position)
            seen_key = (cfg["arm"], *key)
            if seen_key in self._seen:
                raise RuntimeError(f"Duplicate prediction for {seen_key}")
            self._seen.add(seen_key)
            q = torch.log_softmax(logits[index].float(), dim=-1)
            nll = float(-q[target].item())
            if cfg["role"] == "baseline":
                if key in self._reference:
                    raise RuntimeError("Duplicate baseline prediction")
                p_cpu = q.detach().cpu().clone()
                needed = p_cpu.numel() * p_cpu.element_size()
                budget = int(cfg.get("reference_max_mib", 2048)) * 1024 * 1024
                if self._reference_bytes + needed > budget:
                    raise MemoryError("Reference logit cache exceeds reference_max_mib; reduce tokens per batch")
                if not cfg.get("release_reference", False):
                    self._reference[key] = (p_cpu, target, nll)
                    self._reference_bytes += needed
                kl, baseline_nll = 0.0, nll
            else:
                if key not in self._reference:
                    raise RuntimeError(f"Missing native full-logit reference for {key}")
                p_cpu, baseline_target, baseline_nll = self._reference[key]
                if target != baseline_target:
                    raise RuntimeError("Baseline and candidate target token differ")
                p = p_cpu.to(q.device)
                if p.shape != q.shape:
                    raise RuntimeError("Baseline and candidate vocabulary differ")
                # Padded vocabulary positions can have log-probability -inf.
                # Their zero reference mass contributes exactly zero, not NaN.
                difference = torch.where(torch.isfinite(p), p - q, torch.zeros_like(q))
                kl = float((p.exp() * difference).sum().item())
                if cfg.get("release_reference", False):
                    self._reference.pop(key)
                    self._reference_bytes -= p_cpu.numel() * p_cpu.element_size()
            if not all(math.isfinite(v) for v in (nll, kl, baseline_nll)):
                raise FloatingPointError(f"Non-finite quality metric for {key}")
            by_path[cfg["metrics_path"]].append({
                "group": cfg["group"], "doc_key": cfg["doc_key"], "arm": cfg["arm"],
                "output_position": position, "target_token": target,
                "context_tokens": cfg["prefix_tokens"] + position,
                "prediction_phase": "prefill_next_token" if position == 0 else "decode",
                "batch_rows": logits.shape[0], "persistent_batch_index": index,
                "processor_call": self._call, "processor_pid": os.getpid(),
                "vocab_size": logits.shape[-1], "nll": nll,
                "baseline_nll": baseline_nll, "kl_base_to_policy": kl,
                "reference_cache_bytes_after_row": self._reference_bytes,
            })
            logits[index].fill_(-float("inf"))
            logits[index, target] = 0
        for path, rows in by_path.items():
            with open(path, "a", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        return logits


def _self_test():
    """CPU-only checks for move semantics and PPL aggregation; no GPU/vLLM."""
    from types import SimpleNamespace as NS
    make = lambda params, prompt, output: {"name": params, "output": output} if params else None
    live = []
    states = {0: {"name": "old", "output": []}, 2: {"name": "b", "output": []}}
    update_request_states(states, NS(removed=[0], added=[(0, "a", [], live)],
                                     moved=[(0, 2, "SWAP"), (2, 1, "UNIDIRECTIONAL")]), make)
    assert states[0]["name"] == "b" and states[1]["name"] == "a" and 2 not in states
    live.append(17)
    assert states[1]["output"] == [17]
    update_request_states(states, NS(removed=[], added=[(1, None, [], [])], moved=[]), make)
    assert 1 not in states
    rows = [{"nll": n, "baseline_nll": n, "kl_base_to_policy": 0., "group": "g", "doc_key": str(i)}
            for i, n in enumerate((0., math.log(9.)))]
    summary = quality_summary(rows)
    assert abs(summary["ppl"] - 3.) < 1e-12  # Mean PPL would incorrectly be 5.
    assert summary["documents"] == 2
    print("PASS: persistent-batch remove/add/move, output-list reference, token-weighted PPL")
    try:
        import torch
    except ImportError:
        print("NOT RUN: tensor numerical test (torch unavailable on this host)")
        return
    import tempfile
    with tempfile.TemporaryDirectory(prefix="teacher-force-cpu-") as temporary:
        proc = TeacherForceLogitsProcessor(None, torch.device("cpu"), False)
        baseline_logits = {"a": [[0., 1., -float("inf")], [2., 0., -float("inf")]],
                           "b": [[2., 0., -float("inf")], [0., 1., -float("inf")]]}
        targets = {"a": [0, 1], "b": [1, 0]}
        for arm, order in [("native", ["a", "b"]), ("identity", ["b", "a"]), ("changed", ["a", "b"])]:
            output_lists = {name: [] for name in order}
            path = Path(temporary) / f"{arm}.jsonl"
            added = []
            for index, name in enumerate(order):
                cfg = {"group": "numeric", "doc_key": name, "arm": arm,
                       "role": "baseline" if arm == "native" else "candidate",
                       "forced_tokens": targets[name], "metrics_path": str(path),
                       "prefix_tokens": 1, "expected_batch": 2,
                       "release_reference": arm == "changed"}
                params = NS(extra_args={EXTRA_KEY: cfg}, max_tokens=2, ignore_eos=True)
                added.append((index, params, [9], output_lists[name]))
            proc.update_state(NS(removed=list(proc._requests), added=added, moved=[]))
            for position in range(2):
                logits = torch.tensor([baseline_logits[name][position] for name in order])
                if arm == "changed":
                    logits[:, 0] += .5
                result = proc.apply(logits)
                assert result.argmax(-1).tolist() == [targets[name][position] for name in order]
                for name in order:
                    output_lists[name].append(targets[name][position])
            records = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(records) == 4
            if arm == "identity":
                assert max(abs(r["kl_base_to_policy"]) for r in records) < 1e-7
            elif arm == "changed":
                assert min(r["kl_base_to_policy"] for r in records) > 0
        assert not proc._reference and proc._reference_bytes == 0
    print("PASS: CPU tensor forcing, reordered request-keyed full KL, padded -inf vocabulary, cache release")


if __name__ == "__main__":
    _self_test()
