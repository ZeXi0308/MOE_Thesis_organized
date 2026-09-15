"""Optional same-precall qualification diagnostic; no served-output replacement."""
import json
from pathlib import Path
import time
import torch


def aggregate(partials, dtype, reverse=False):
    ordered = list(reversed(partials)) if reverse else partials
    result = ordered[0].to(dtype).clone()
    for partial in ordered[1:]:
        result.add_(partial.to(dtype))
    return result.to(partials[0].dtype)


def difference(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError("compared partial shape/dtype differs")
    finite = bool(torch.isfinite(a).all() & torch.isfinite(b).all())
    bits_a, bits_b = (t.contiguous().view(torch.uint8).reshape(t.numel(), t.element_size()) for t in (a, b))
    equal_elements = int((bits_a == bits_b).all(dim=1).sum())
    delta, ref = a.float() - b.float(), b.float()
    denominator = float(torch.linalg.vector_norm(ref)) if finite else None
    numerator = float(torch.linalg.vector_norm(delta)) if finite else None
    return dict(allfinite=finite, elements=a.numel(), bit_equal_elements=equal_elements, bit_equal=equal_elements == a.numel(),
        allclose=bool(torch.allclose(a, b, rtol=.01, atol=.01)), maxabs=float(delta.abs().max()) if finite else None,
        relative_l2=numerator / denominator if denominator else (0. if numerator == 0 else None))


class Layer47Replay:
    def __init__(self, outdir):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=False)
        self.partials, self.maps, self.call_id, self.finished = [], [], None, False
        self.result = None

    def gate(self):
        return self.result["gate"] if self.finished else dict(status="FAIL", complete=False)

    def wanted(self, record):
        return record["layer_name"] == "model.layers.47.mlp.experts" and not self.finished

    def retain(self, record, group_map, partial):
        if not self.wanted(record):
            return
        if record["measurement"] or not record["validation_run"]:
            raise ValueError("partial replay is qualification-only")
        if self.call_id not in (None, record["call_id"]):
            raise ValueError("partial replay crossed calls")
        self.call_id = record["call_id"]
        # result aliases the first y and later add_ mutates it: retain BEFORE aggregation.
        self.partials.append(partial.detach().clone())
        self.maps.append(group_map.detach().clone())

    def finish(self, runtime, method, layer, fixed, actual, full_reference, w13, w2, record):
        if not self.wanted(record):
            return
        started = time.perf_counter()
        x, weights, ids = fixed
        state = runtime.layers[id(layer)]["state"]
        groups = [g["required_experts"] for g in record["groups"]]
        active = set(ids.detach().cpu().flatten().tolist())
        flattened = [e for group in groups for e in group]
        if (self.call_id != record["call_id"] or len(self.partials) != len(groups)
                or len(flattened) != len(set(flattened)) or set(flattened) != active):
            raise ValueError("same-call disjoint mask coverage failed")
        masked_partials, masks = [], []
        for group, mapping in zip(groups, self.maps):
            values = mapping.detach().cpu().tolist()
            slots = [slot for slot in values if slot >= 0]
            if (len(values) != state.num_experts or {e for e, s in enumerate(values) if s >= 0} != set(group)
                    or len(set(slots)) != len(slots) or any(s >= state.cap_experts for s in slots)):
                raise ValueError("actual group map is not a disjoint scratch-slot map")
            mask = torch.tensor([e if e in set(group) else -1 for e in range(state.num_experts)], dtype=torch.int32, device=x.device)
            masks.append(mask)
            masked_partials.append(runtime.kernel(hidden_states=x, w1=w13, w2=w2,
                topk_weights=weights, topk_ids=ids, activation=layer.activation,
                quant_config=method.moe_quant_config, apply_router_weight_on_input=layer.apply_router_weight_on_input,
                global_num_experts=state.num_experts, expert_map=mask))
        reconstructed = aggregate(self.partials, actual.dtype)
        if not difference(actual, reconstructed)["bit_equal"]:
            raise ValueError("retained actual partials do not reconstruct the served result")
        grouped_full = aggregate(masked_partials, actual.dtype)
        fp32, reverse = aggregate(self.partials, torch.float32), aggregate(self.partials, actual.dtype, True)
        tensors = dict(x=x, topk_weights=weights, topk_ids=ids, actual=actual, full_reference=full_reference,
                       same_partition_full=grouped_full, actual_sum_fp32_cast=fp32, actual_sum_reverse_bf16=reverse)
        for name, items in (("actual_partial", self.partials), ("actual_group_map", self.maps),
                            ("same_partition_partial", masked_partials), ("full_identity_group_map", masks)):
            tensors.update({f"{name}_{i}": t for i, t in enumerate(items)})
        payload = sum(t.numel() * t.element_size() for t in tensors.values())
        d2h = sum(t.numel() * t.element_size() for t in tensors.values() if t.device.type == "cuda")
        snapshot = {name: tensor.detach().to(device="cpu", copy=True) for name, tensor in tensors.items()}
        with (self.outdir / "layer47.pt").open("xb") as stream:
            torch.save(snapshot, stream)
        diagnostic = dict(scope="QUALIFICATION_ONLY; original tolerance; no quality or performance claim",
            call_id=self.call_id, layer_name=record["layer_name"], context=record["context"], groups=groups,
            actual_vs_full=difference(actual, full_reference), actual_vs_same_partition_full=difference(actual, grouped_full),
            same_partition_full_vs_full=difference(grouped_full, full_reference),
            partial_comparisons=[difference(a, b) for a, b in zip(self.partials, masked_partials)],
            actual_vs_fp32_cast=difference(actual, fp32), actual_vs_reverse_bf16=difference(actual, reverse),
            actual_partial_reconstruction_bit_equal=True, rtol=.01, atol=.01,
            disjoint_complete_group_masks=True, rows=len(x), output_elements=actual.numel(),
            expected_partial_elements=len(groups)*actual.numel(),
            actual_partial_finite_elements=sum(int(torch.isfinite(t).sum()) for t in self.partials),
            reference_partial_finite_elements=sum(int(torch.isfinite(t).sum()) for t in masked_partials),
            kernel_geometry=dict(actual_w13=list(state.scratch_w13.shape), actual_w2=list(state.scratch_w2.shape),
                reference_w13=list(w13.shape), reference_w2=list(w2.shape),
                residual_scope="Scratch/local-expert and full-weight kernel shapes differ; a residual alone is not proof of a paging/map defect"),
            accounting=dict(extra_full_weight_copy_bytes=0, full_weights_reused_from_original_reference=True,
                extra_masked_kernel_calls=len(groups), extra_identity_map_h2d_payload_bytes=sum(m.numel()*m.element_size() for m in masks) if x.device.type == "cuda" else 0,
                retained_partial_and_map_clone_payload_bytes=sum(t.numel()*t.element_size() for t in self.partials+self.maps),
                cpu_snapshot_tensor_payload_bytes=payload, snapshot_d2h_payload_bytes=d2h,
                serialized_file_bytes=(self.outdir / "layer47.pt").stat().st_size,
                finish_host_span_s=time.perf_counter()-started,
                scope="Additional qualification operations; excludes original full-reference weight copies; span is not pure GPU compute"))
        comparisons = diagnostic["partial_comparisons"] + [diagnostic["actual_vs_same_partition_full"]]
        diagnostic["gate"] = dict(complete=True, status="PASS" if all(c["allfinite"] and c["bit_equal"] for c in comparisons)
            and diagnostic["actual_vs_full"]["allfinite"] else "FAIL",
            scope="This captured pre-call only; every partial and final grouped reference must be bit-exact and finite; full48-layer finite checked separately")
        with (self.outdir / "layer47.json").open("x") as stream:
            json.dump(diagnostic, stream, indent=2, allow_nan=False)
        record["numerical_localization"] = dict(path=str(self.outdir), qualification_only=True)
        self.result, self.finished = diagnostic, True
        self.partials.clear(); self.maps.clear()


def install(runtime, outdir):
    if getattr(runtime, "numerical_localization", None) is not None:
        raise ValueError("numerical localization already installed")
    runtime.numerical_localization = Layer47Replay(outdir)
    return runtime.numerical_localization
