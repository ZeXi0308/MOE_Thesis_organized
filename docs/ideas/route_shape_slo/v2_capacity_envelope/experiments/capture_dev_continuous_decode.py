#!/usr/bin/env python3
"""Run the existing producer with an explicit v2 conformance diagnostic.

The canonical producer requires serial and batched decode to choose identical
ordered experts.  The retained RTX 5090 pilot observed that OLMoE can change
that assignment under batch execution even when request/token identity remains
closed. This development-only wrapper preserves the raw routes and requires
exact token/completion parity, but records rather than hides any serial-vs-
batch expert-assignment divergence.  Formal producer/manifests are untouched.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
PRODUCER_PATH = HERE.parents[2] / "bcrd" / "experiments" / "capture_continuous_decode.py"


def _load_producer() -> Any:
    sys.path.insert(0, str(PRODUCER_PATH.parent))
    spec = importlib.util.spec_from_file_location(
        "rce_v2_assignment_audit_producer", PRODUCER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import producer: {PRODUCER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PRODUCER = _load_producer()
_PATCHED = False


def assignment_route_signature(
    batches: Sequence[Mapping[str, Any]], row_index: int
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Return a per-layer expert multiset, independent of top-k slot order."""

    return tuple(
        (
            int(batch["layer"]),
            tuple(
                sorted(
                    int(value)
                    for value in batch["selected_experts"][row_index].tolist()
                )
            ),
        )
        for batch in batches
    )


def run_development_conformance_audit(
    model: object,
    *,
    requests: Sequence[Any],
    request_rows: Mapping[str, Mapping[str, Any]],
    request_ids: Sequence[str],
    eos_token_id: int | None,
    duration_provider: Any | None = None,
) -> dict[str, Any]:
    """Require token parity and measure ordered/multiset route conformance."""

    try:
        import torch
    except ImportError as exc:
        raise PRODUCER.ProtocolError("serial audit requires PyTorch") from exc

    by_id = {item.request_id: item for item in requests}
    audited_steps = 0
    matched_assignment_steps = 0
    matched_assignment_layers = 0
    matched_ordered_layers = 0
    audited_layers = 0
    differences: list[dict[str, Any]] = []
    for request_id in request_ids:
        spec = by_id[request_id]
        expected_steps = list(request_rows[request_id]["steps"])
        with torch.inference_mode():
            prefill, _ = PRODUCER._timed_call(
                model,
                "serial_prefill_audit",
                1,
                duration_provider,
                input_ids=spec.input_ids,
                attention_mask=spec.attention_mask,
                use_cache=True,
                output_router_logits=True,
                return_dict=True,
            )
        cache = getattr(prefill, "past_key_values", None)
        logits = getattr(prefill, "logits", None)
        if cache is None or logits is None:
            raise PRODUCER.ProtocolError("serial audit prefill returned no cache/logits")
        attention_mask = spec.attention_mask
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        for expected in expected_steps:
            if int(next_token.item()) != int(expected["input_token_id"]):
                raise PRODUCER.ProtocolError(
                    f"serial token input mismatch for {request_id}"
                )
            prior_length = PRODUCER._cache_length(cache)
            attention_mask = torch.cat(
                (attention_mask, attention_mask.new_ones((1, 1))), dim=1
            )
            position_ids = attention_mask.long().cumsum(-1)[:, -1:] - 1
            with torch.inference_mode():
                output, _ = PRODUCER._timed_call(
                    model,
                    "serial_decode_audit",
                    1,
                    duration_provider,
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_position=torch.tensor(
                        [prior_length], dtype=torch.long, device=next_token.device
                    ),
                    past_key_values=cache,
                    use_cache=True,
                    output_router_logits=True,
                    return_dict=True,
                )
            cache = getattr(output, "past_key_values", None)
            logits = getattr(output, "logits", None)
            if (
                cache is None
                or logits is None
                or PRODUCER._cache_length(cache) != prior_length + 1
            ):
                raise PRODUCER.ProtocolError(
                    f"serial cache closure failed for {request_id}"
                )
            batches = PRODUCER._native_route_batches(
                output, expected_rows=1, config=getattr(model, "config")
            )
            observed_ordered = tuple(
                (
                    int(batch["layer"]),
                    tuple(
                        int(value)
                        for value in batch["selected_experts"][0].tolist()
                    ),
                )
                for batch in batches
            )
            expected_ordered = tuple(
                (
                    int(layer["layer"]),
                    tuple(int(value) for value in layer["experts"]),
                )
                for layer in expected["route_signature"]
            )
            if [layer for layer, _ in observed_ordered] != [
                layer for layer, _ in expected_ordered
            ]:
                raise PRODUCER.ProtocolError(
                    f"serial route layer identity mismatch for {request_id}"
                )
            step_assignment_match = True
            for (layer, expected_experts), (_, observed_experts) in zip(
                expected_ordered, observed_ordered
            ):
                audited_layers += 1
                if expected_experts == observed_experts:
                    matched_ordered_layers += 1
                assignment_match = sorted(expected_experts) == sorted(observed_experts)
                if assignment_match:
                    matched_assignment_layers += 1
                else:
                    step_assignment_match = False
                    if len(differences) < 16:
                        differences.append(
                            {
                                "request_id": request_id,
                                "decode_step": int(expected["decode_step"]),
                                "layer": layer,
                                "batched_experts": list(expected_experts),
                                "serial_experts": list(observed_experts),
                            }
                        )
            if step_assignment_match:
                matched_assignment_steps += 1
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            if int(next_token.item()) != int(expected["predicted_next_token_id"]):
                raise PRODUCER.ProtocolError(
                    f"serial greedy token mismatch for {request_id}"
                )
            audited_steps += 1
        if expected_steps and eos_token_id is not None:
            expected_stop = request_rows[request_id]["stop_reason"]
            if expected_stop == "eos" and int(next_token.item()) != int(eos_token_id):
                raise PRODUCER.ProtocolError(
                    f"serial EOS stop mismatch for {request_id}"
                )
        elif (
            not expected_steps
            and request_rows[request_id]["stop_reason"] == "eos_before_decode"
        ):
            if eos_token_id is None or int(next_token.item()) != int(eos_token_id):
                raise PRODUCER.ProtocolError(
                    f"serial pre-decode EOS mismatch for {request_id}"
                )

    batch_dependent = matched_assignment_layers != audited_layers
    return {
        "status": (
            "PASS_TOKEN_PARITY_ROUTE_BATCH_DEPENDENT"
            if batch_dependent
            else "PASS"
        ),
        "requests": len(request_ids),
        "steps": audited_steps,
        "layers": audited_layers,
        "token_match_fraction": 1.0,
        "route_identity_match_fraction": (
            matched_assignment_layers / max(1, audited_layers)
        ),
        "route_assignment_step_match_fraction": (
            matched_assignment_steps / max(1, audited_steps)
        ),
        "route_ordered_layer_match_fraction": (
            matched_ordered_layers / max(1, audited_layers)
        ),
        "route_identity_semantics": "per_layer_expert_assignment_multiset",
        "batch_dependent_route_observed": batch_dependent,
        "difference_examples": differences,
        "topk_order_checked": True,
        "gate_weight_checked": False,
        "reference_type": "same-model serial cached-decode conformance diagnostic",
        "scientific_ground_truth": False,
        "development_wrapper": str(Path(__file__).resolve()),
    }


def install_development_assignment_audit() -> None:
    global _PATCHED
    if _PATCHED:
        return
    PRODUCER.run_serial_audit = run_development_conformance_audit
    _PATCHED = True


def _manifest_path(argv: Sequence[str]) -> Path:
    for index, value in enumerate(argv):
        if value == "--workload-manifest" and index + 1 < len(argv):
            return Path(argv[index + 1]).resolve()
        if value.startswith("--workload-manifest="):
            return Path(value.split("=", 1)[1]).resolve()
    raise SystemExit("development wrapper requires --workload-manifest")


def require_development_manifest(argv: Sequence[str]) -> None:
    manifest = json.loads(_manifest_path(argv).read_text(encoding="utf-8"))
    if manifest.get("run_class") != "development":
        raise SystemExit("assignment-order relaxation is development-only")
    marker = manifest.get("route_capacity_envelope")
    if not isinstance(marker, Mapping) or marker.get(
        "serial_route_identity_semantics"
    ) != "per_layer_expert_assignment_multiset":
        raise SystemExit("workload did not freeze the v2 assignment-audit semantics")


def main() -> None:
    require_development_manifest(sys.argv[1:])
    install_development_assignment_audit()
    PRODUCER.main()


if __name__ == "__main__":
    main()
