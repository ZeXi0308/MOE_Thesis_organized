from __future__ import annotations

"""Replay one frozen BCRD instance with an explicit legal assignment."""

import argparse

try:
    from .core import Contribution, ProtocolError, ReplayConfig, ServiceCatalog, read_instances, read_json, simulate_assignment, write_json
except ImportError:
    from core import Contribution, ProtocolError, ReplayConfig, ServiceCatalog, read_instances, read_json, simulate_assignment, write_json


def contributions_from_instance(instance: dict[str, object]) -> list[Contribution]:
    raw = instance.get("contributions")
    if not isinstance(raw, list):
        raise ProtocolError("instance contributions must be a list")
    return [Contribution.from_mapping(item) for item in raw if isinstance(item, dict)]


def replay(
    instance: dict[str, object],
    assignments: list[int],
    catalog: ServiceCatalog,
    *,
    hold_us: float,
    remote_latency_us: float,
    remote_bytes_per_row: int,
) -> dict[str, object]:
    contributions = contributions_from_instance(instance)
    config = ReplayConfig(
        replica_count=int(instance["replica_count"]),
        hold_us=hold_us,
        remote_latency_us=remote_latency_us,
        remote_bytes_per_row=remote_bytes_per_row,
    )
    result = simulate_assignment(contributions, assignments, catalog, config)
    result.update(
        {
            "instance_id": instance["instance_id"],
            "hold_us": hold_us,
            "remote_latency_us": remote_latency_us,
            "assignment": assignments,
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--assignment-json", required=True)
    parser.add_argument("--service-curve", required=True)
    parser.add_argument("--hold-us", type=float, default=0.0)
    parser.add_argument("--remote-latency-us", type=float, default=0.0)
    parser.add_argument("--remote-bytes-per-row", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instances = {row["instance_id"]: row for row in read_instances(args.instances)}
    if args.instance_id not in instances:
        raise SystemExit(f"unknown instance {args.instance_id!r}")
    assignment = read_json(args.assignment_json)
    if not isinstance(assignment, list) or any(not isinstance(value, int) for value in assignment):
        raise ProtocolError("assignment JSON must be a list of integers")
    result = replay(
        instances[args.instance_id],
        assignment,
        ServiceCatalog.from_csv(args.service_curve),
        hold_us=args.hold_us,
        remote_latency_us=args.remote_latency_us,
        remote_bytes_per_row=args.remote_bytes_per_row,
    )
    write_json(args.output, result)
    print(f"wrote replay to {args.output}")


if __name__ == "__main__":
    main()
