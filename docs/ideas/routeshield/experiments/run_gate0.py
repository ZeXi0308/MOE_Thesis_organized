from __future__ import annotations

"""Fail-closed RouteShield Gate-0 readiness/result-contract CLI."""

import argparse
import json
from pathlib import Path

try:
    from .protocol import MetricCell, evaluate_metric_cells, load_config, readiness_report
    from .raw_recompute import recompute_bundle, strict_json_file
    from .schema import ProtocolError
except ImportError:
    from protocol import MetricCell, evaluate_metric_cells, load_config, readiness_report
    from raw_recompute import recompute_bundle, strict_json_file
    from schema import ProtocolError


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--metrics-json")
    source.add_argument("--raw-bundle")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _smoke_cells(config: dict[str, object]) -> list[MetricCell]:
    cells: list[MetricCell] = []
    for model in config["models"]:  # type: ignore[index]
        cells.append(MetricCell(
            model=str(model["key"]),
            load_cell="70pct",
            traffic_class="ADV_TEXT",
            metric_name="REPLAYED_TTFT_P99",
            harm_point=0.25,
            harm_lcb=0.15,
            oracle_gain_point=0.15,
            oracle_gain_lcb=0.08,
            oracle_recovery_lcb=0.55,
            simple_capture_ucb=0.80,
            benign_goodput_loss_ucb=0.03,
            exactness_pass=True,
            queue_stable=True,
            no_drop_or_starvation=True,
            full_request_dag_exact=True,
            legal_action_space=True,
            oracle_exact=True,
        ))
        cells.append(MetricCell(
            model=str(model["key"]),
            load_cell="70pct",
            traffic_class="NAT_PATHOLOGICAL",
            metric_name="REPLAYED_TTFT_P99",
            harm_point=0.0,
            harm_lcb=0.0,
            oracle_gain_point=0.0,
            oracle_gain_lcb=0.0,
            oracle_recovery_lcb=0.0,
            simple_capture_ucb=0.0,
            benign_goodput_loss_ucb=0.03,
            exactness_pass=True,
            queue_stable=True,
            no_drop_or_starvation=True,
            full_request_dag_exact=True,
            legal_action_space=True,
            oracle_exact=True,
        ))
        cells.append(MetricCell(
            model=str(model["key"]),
            load_cell="30pct",
            traffic_class="NAT_BENIGN",
            metric_name="REPLAYED_TTFT_P99",
            harm_point=0.0,
            harm_lcb=0.0,
            oracle_gain_point=0.0,
            oracle_gain_lcb=0.0,
            oracle_recovery_lcb=0.0,
            simple_capture_ucb=0.0,
            benign_goodput_loss_ucb=0.03,
            exactness_pass=True,
            queue_stable=True,
            no_drop_or_starvation=True,
            full_request_dag_exact=True,
            legal_action_space=True,
            oracle_exact=True,
        ))
    return cells


def main() -> None:
    args = _parse_args()
    try:
        config = load_config(args.config)
    except (ProtocolError, OSError, TypeError, ValueError, KeyError) as exc:
        payload = {
            "schema": "routeshield-gate0-output-v1",
            "status": "INVALID_CONFIG",
            "formal_result": False,
            "reason": str(exc),
            "evidence_boundary": "Config load failed; no evidence was evaluated",
        }
    else:
        readiness = readiness_report(config)
        try:
            if args.raw_bundle:
                payload = recompute_bundle(
                    args.raw_bundle,
                    config=config,
                    config_path=args.config,
                    allow_small_fixture=args.smoke,
                )
                payload["readiness"] = readiness
                if (
                    args.smoke
                    and payload.get("status") == "RAW_RECOMPUTE_DIAGNOSTIC_ONLY"
                ):
                    payload["status"] = "RAW_RECOMPUTE_SMOKE_ONLY"
                    payload["evidence_boundary"] = (
                        "Small raw paired-block fixture; hashing and recomputation "
                        "smoke only"
                    )
            elif args.smoke:
                contract_check = evaluate_metric_cells(config, _smoke_cells(config))
                payload = {
                    "schema": "routeshield-gate0-output-v1",
                    "status": "SMOKE_ONLY",
                    "formal_result": False,
                    "contract_branch_preview": contract_check.get("threshold_branch"),
                    "readiness": readiness,
                    "evidence_boundary": (
                        "Deterministic synthetic aggregate-shape fixture; no scientific "
                        "evidence"
                    ),
                }
            elif args.metrics_json:
                raw = strict_json_file(args.metrics_json)
                if not isinstance(raw, dict) or set(raw) != {"cells"}:
                    raise ProtocolError("aggregate metrics root must contain only cells")
                raw_cells = raw["cells"]
                if not isinstance(raw_cells, list):
                    raise ProtocolError("aggregate metrics cells must be a list")
                cells = [MetricCell.from_mapping(row) for row in raw_cells]
                payload = evaluate_metric_cells(config, cells)
                payload["readiness"] = readiness
            elif readiness["status"] != "READY_FOR_FORMAL_GATE0":
                payload = readiness
            else:
                payload = {
                    "schema": "routeshield-gate0-output-v1",
                    "status": "BLOCKED_MISSING_FORMAL_EVIDENCE",
                    "formal_result": False,
                    "reason": "ready config requires a hash-bound --raw-bundle",
                }
        except (ProtocolError, OSError, TypeError, ValueError, KeyError) as exc:
            payload = {
                "schema": "routeshield-gate0-output-v1",
                "status": (
                    "UNSOLVED_EXACT_STATE_LIMIT"
                    if str(exc) == "UNSOLVED_EXACT_STATE_LIMIT"
                    else "INVALID_ARTIFACT"
                ),
                "formal_result": False,
                "reason": str(exc),
                "readiness": readiness,
            }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{payload['status']}: wrote {output}")


if __name__ == "__main__":
    main()
