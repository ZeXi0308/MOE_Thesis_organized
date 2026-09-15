from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from cjc_policy import (
    CJCValidationError,
    LUT_COMPONENT_PROVENANCE,
    LUT_EXPERT_SOURCE,
    LUT_HOST_STAGING_SOURCE,
    LUT_LAUNCH_SOURCE,
    LUT_PACK_SOURCE,
    LUT_REDUCTION_SOURCE,
    LUTPoint,
    PlacementManifest,
    RouteContribution,
    ServiceLUT,
)
from prepare_cjc_calibration import (
    _canonical_json,
    derive_arrival_rate_per_us,
    fit_two_state_mmpp,
    load_arrival_trace,
    measure_ack_components,
)
from merge_cjc_luts import validate_rows
from run_cjc_lut_gpu import select_ids


REVISION = "model@revision"


def _route(request: str, receiver: int) -> RouteContribution:
    return RouteContribution(
        schema_version="cjc-route-v1",
        model_revision=REVISION,
        data_manifest_sha256="d" * 64,
        request_id=request,
        forward_id=f"{request}:forward",
        batch_id="batch-0",
        phase="prefill",
        decode_step=0,
        layer_id=0,
        token_id=f"{request}:token",
        token_position=0,
        topk_slot=0,
        expert_id=0,
        sender_rank=0,
        receiver_rank=receiver,
        valid=True,
        route_weight=1.0,
        route_source="native_model_forward",
        placement_manifest_sha256="p" * 64,
    )


def _formal_lut() -> ServiceLUT:
    return ServiceLUT(
        [
            LUTPoint(
                REVISION,
                0,
                1,
                2.0,
                1.0,
                0.0,
                3.0,
                4.0,
                LUT_COMPONENT_PROVENANCE,
                LUT_EXPERT_SOURCE,
                LUT_PACK_SOURCE,
                LUT_LAUNCH_SOURCE,
                LUT_HOST_STAGING_SOURCE,
                LUT_REDUCTION_SOURCE,
            )
        ],
        formal=True,
    )


class CjcCalibrationTest(unittest.TestCase):
    def test_formal_lut_rejects_ambiguous_single_source(self) -> None:
        with self.assertRaisesRegex(CJCValidationError, "per-component provenance"):
            ServiceLUT(
                [LUTPoint(REVISION, 0, 1, 1, 1, 1, 1, 1, "measured_same_gpu")],
                formal=True,
            )
        self.assertEqual(_formal_lut().lookup(REVISION, 0, 1).launch_us, 0.0)

    def test_mmpp_fit_is_trace_derived_and_non_degenerate(self) -> None:
        intervals = [1.0, 1.2, 8.0, 9.0, 1.1, 1.3, 7.0, 8.5]
        timestamps = [0.0]
        for interval in intervals:
            timestamps.append(timestamps[-1] + interval)
        fit = fit_two_state_mmpp(timestamps)
        self.assertLess(fit["mmpp_low_multiplier"], 1.0)
        self.assertGreater(fit["mmpp_high_multiplier"], 1.0)
        self.assertGreater(fit["mmpp_switch_probability"], 0.0)
        self.assertLess(fit["mmpp_switch_probability"], 1.0)

    def test_arrival_trace_rejects_sealed_or_synthetic_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arrival.json"
            payload = {
                "schema_version": "cjc-arrival-trace-v1",
                "protocol_split": "sealed",
                "source": "synthetic_stress",
                "timestamps_us": list(range(65)),
            }
            import hashlib

            payload["manifest_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(CJCValidationError, "calibration arrivals only"):
                load_arrival_trace(path)

    def test_arrival_rate_uses_bottleneck_receiver_demand(self) -> None:
        placement = PlacementManifest(
            sha256="p" * 64,
            ep_size=8,
            gpus_per_node=4,
            expert_to_sender={(REVISION, 0): 0},
            request_to_receiver={"r0": 0, "r1": 4},
        )
        rate, demands = derive_arrival_rate_per_us(
            [_route("r0", 0), _route("r1", 4)],
            lut=_formal_lut(),
            placement=placement,
            hidden_size=1,
            dtype_bytes=1,
            descriptor_bytes=0,
            alignment_bytes=0,
            link_gbps=1.0,
            target_rho=0.5,
        )
        # Per contribution: 1 + 0 + 3 + 4 us plus 1 byte / 125 bytes/us.
        self.assertAlmostEqual(demands["node0:combine_ingress"], 4.004)
        self.assertAlmostEqual(demands["node1:combine_ingress"], 4.004)
        self.assertAlmostEqual(rate, 0.5 / 4.004)

    def test_ack_accounting_keeps_host_and_wire_sources_separate(self) -> None:
        timing = measure_ack_components(
            model_key="tiny", iterations=20, repeats=2, link_gbps=200.0
        )
        components = timing["components"]
        self.assertEqual(
            components["build_us"]["source"], "measured_same_run_host_monotonic_ns"
        )
        self.assertEqual(components["wire_us"]["source"], "analytic_link")
        self.assertEqual(components["wire_us"]["message_bytes"], 32)

    def test_lut_identity_selection_is_deterministic(self) -> None:
        first = select_ids(range(16), count=4, seed=7, prefix=(REVISION, 2, "expert"))
        second = select_ids(range(16), count=4, seed=7, prefix=(REVISION, 2, "expert"))
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 4)

    def test_merge_rejects_missing_layer_row_point(self) -> None:
        point = {
            "model_revision": REVISION,
            "layer_id": "1",
            "rows": "1",
            "launch_us": "0",
            "source": LUT_COMPONENT_PROVENANCE,
            "expert_source": LUT_EXPERT_SOURCE,
            "pack_source": LUT_PACK_SOURCE,
            "launch_source": LUT_LAUNCH_SOURCE,
            "host_staging_source": LUT_HOST_STAGING_SOURCE,
            "reduction_source": LUT_REDUCTION_SOURCE,
        }
        with self.assertRaisesRegex(CJCValidationError, "incomplete"):
            validate_rows(
                [point],
                revision=REVISION,
                expected_layers=[1],
                expected_row_grid=[1, 2],
            )


if __name__ == "__main__":
    unittest.main()
