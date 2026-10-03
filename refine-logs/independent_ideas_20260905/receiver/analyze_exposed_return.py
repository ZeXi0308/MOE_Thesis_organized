#!/usr/bin/env python3
"""Fixed-order structural conditional bounds, requiring an explicit causal DAG.

Baseline reconstruction establishes compatibility, not causal completeness.
No Nsight/CUPTI exporter is connected by this tool; chronology is never inferred
to be causality and an observed start is never accepted as a release condition.
"""
import argparse
from collections import defaultdict, deque
import json
import math
from pathlib import Path


class NotIdentified(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise NotIdentified(message)


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), "nonfinite/invalid timestamp")
    return float(value)


def analyze(trace):
    if not isinstance(trace, dict):
        trace = {}
    result = dict(status="NOT_IDENTIFIED", scientific_go=False,
        evidence_type="SYNTHETIC_FIXTURE" if trace.get("provenance") == "synthetic_fixture" else "UNVERIFIED_TRACE",
        fixed_order_structural_conditional_bound=None, negative_control=None, reasons=[],
        limitations=["The real serving Nsight/CUPTI exporter is not connected by this implementation.",
            "Reconstruction proves compatibility, not completeness or causal validity of supplied edges/releases.",
            "Durations and resource order are held fixed; altered contention, routing and scheduling are not modeled.",
            "This structural conditional bound is not a measured system Oracle or deployable gain."])
    try:
        require(trace.get("schema") == "ep-return-explicit-dag-v1", "unknown schema; no raw trace adapter")
        synthetic = trace.get("provenance") == "synthetic_fixture"
        require(synthetic or trace.get("provenance") == "natural_ep_serving", "natural EP provenance not supplied")
        require(trace.get("clock_domain") == "unified_nsys_cupti" or synthetic, "rank-local clocks are not a unified timeline")
        coverage = trace["coverage"]
        for key in ("causal_dependencies", "resource_serialization", "exogenous_releases",
                    "request_completions", "all_moe_layers", "message_identity"):
            require(coverage.get(key) is True, f"missing explicit coverage attestation: {key}")
        if not synthetic:
            require(trace["environment"]["gpu_count"] == 8, "8 distinct GPUs required for this existence gate")
            require(trace["environment"]["optimized_backend"] is True, "optimized backend evidence missing")
            require(trace["environment"].get("hardware_backend_compatibility_verified") is True,
                    "hardware/backend version compatibility has not been verified")
            require(all(trace["environment"].get(k) for k in ("gpu_model", "backend_name", "backend_version")),
                    "freeze GPU model and backend name/version before interpreting natural EP input")
            require(bool(trace["environment"].get("transport")), "transport identity missing")
        rows = trace["nodes"]
        nodes = {n["id"]: n for n in rows}
        require(bool(nodes) and len(nodes) == len(rows), "empty/duplicate node identity")
        requests = {r["request_id"]: r for r in trace["requests"]}
        require(bool(requests) and len(requests) == len(trace["requests"]), "empty/duplicate request identity")
        tolerance = number(trace.get("clock_tolerance_us", 0.001))
        require(0 <= tolerance <= 1, "clock tolerance must be between 0 and 1 us")
        parents, children = defaultdict(set), defaultdict(set)
        resource_members, releases, durations = defaultdict(set), {}, {}
        returns = set()
        for node_id, node in nodes.items():
            start, end = number(node["start_us"]), number(node["end_us"])
            require(0 <= start <= end, "invalid node chronology")
            durations[node_id] = end - start
            identities = node["request_ids"]
            require(len(set(identities)) == len(identities) and set(identities) <= set(requests), "unknown/duplicate node request identity")
            release = node["release"]
            releases[node_id] = number(release["time_us"])
            require(releases[node_id] >= 0 and bool(release.get("evidence")), "release requires semantic evidence")
            if release["kind"] == "request_arrival":
                rid = release["request_id"]
                require(rid in requests and rid in identities, "release request identity mismatch")
                require(abs(releases[node_id] - number(requests[rid]["arrival_us"])) <= tolerance, "release is not request arrival")
            else:
                require(release["kind"] == "external_event" and release.get("action_independent") is True,
                        "release must be exogenous; observed start/endogenous wait is not accepted")
            resources = node["resources"]
            require(len(set(resources)) == len(resources), "duplicate resource identity")
            require(bool(resources) or durations[node_id] == 0, "positive service needs explicit serialization resources")
            for resource in resources:
                resource_members[resource].add(node_id)
            require(type(node["return_path"]) is bool, "return_path must be explicit boolean")
            if node["return_path"]:
                require(node["phase"] in ("return_a2a", "receiver_unpack", "receiver_combine"), "unknown return service classification")
                require(bool(identities) and bool(node.get("message_ids")), "return service lacks request/message identity")
                returns.add(node_id)
        require(bool(returns), "no identified return-path service nodes")

        def edge(source, target):
            require(source in nodes and target in nodes and source != target, "invalid dependency identity")
            parents[target].add(source)
            children[source].add(target)

        dependencies = trace["dependency_edges"]
        require(bool(dependencies), "kernel chronology without dependencies is not identified")
        for item in dependencies:
            require(bool(item.get("evidence")), "dependency has no causal evidence")
            edge(item["source"], item["target"])
        seen_resources, resource_edge_count = set(), 0
        for sequence in trace["resource_sequences"]:
            resource, order = sequence["resource_id"], sequence["node_ids"]
            require(resource in resource_members, "unknown serialization resource")
            require(resource not in seen_resources and bool(sequence.get("evidence")), "duplicate/unsupported resource sequence")
            seen_resources.add(resource)
            require(len(set(order)) == len(order) and set(order) == resource_members[resource], "resource sequence omits/adds service identity")
            for source, target in zip(order, order[1:]):
                edge(source, target)
                resource_edge_count += 1
        require(seen_resources == set(resource_members), "resource serialization coverage missing")
        indegree = {n: len(parents[n]) for n in nodes}
        ready, order = deque(n for n in nodes if indegree[n] == 0), []
        while ready:
            current = ready.popleft()
            order.append(current)
            for child in children[current]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        require(len(order) == len(nodes), "cycle in dependency/resource DAG")

        def replay(deleted):
            starts, ends = {}, {}
            for n in order:
                starts[n] = max([releases[n], *[ends[p] for p in parents[n]]])
                ends[n] = starts[n] + (0 if n in deleted else durations[n])
            return starts, ends

        starts, baseline = replay(set())
        residual = max(max(abs(starts[n] - nodes[n]["start_us"]), abs(baseline[n] - nodes[n]["end_us"])) for n in nodes)
        require(residual <= tolerance, f"baseline reconstruction mismatch {residual} us; missing constraints remain unknown")
        ancestors = {}
        for n in order:
            ancestors[n] = set(parents[n])
            for p in parents[n]:
                ancestors[n].update(ancestors[p])
        for rid, request in requests.items():
            arrival = number(request["arrival_us"])
            final = request["completion_node"]
            require(final in nodes and rid in nodes[final]["request_ids"], "request completion identity mismatch")
            require(abs(baseline[final] - number(request["completion_us"])) <= tolerance and baseline[final] > arrival,
                    "request completion timestamp/denominator mismatch")
            tokens = request["tokens"]
            require(bool(tokens) and [t["token_index"] for t in tokens] == list(range(len(tokens))), "token completion coverage incomplete")
            previous = None
            for token in tokens:
                n = token["completion_node"]
                require(n in nodes and rid in nodes[n]["request_ids"], "token completion identity mismatch")
                require(abs(baseline[n] - number(token["completion_us"])) <= tolerance and baseline[n] >= arrival, "token completion timestamp mismatch")
                require(n == final or n in ancestors[final], "token has no causal path to request completion")
                require(any(nodes[p]["release"]["kind"] == "request_arrival" and
                    nodes[p]["release"]["request_id"] == rid for p in ancestors[n] | {n}),
                    "token has no causal request-arrival release")
                require(previous is None or previous in ancestors[n], "token progression causal edge missing")
                previous = n

        def summarize(deleted):
            _, changed = replay(deleted)
            output = []
            for rid, request in requests.items():
                final, arrival = request["completion_node"], request["arrival_us"]
                saving = baseline[final] - changed[final]
                output.append(dict(request_id=rid, baseline_latency_us=baseline[final] - arrival,
                    zeroed_service_latency_us=changed[final] - arrival, completion_shift_us=saving,
                    conditional_fraction=saving / (baseline[final] - arrival),
                    token_completion_shifts_us=[baseline[t["completion_node"]] - changed[t["completion_node"]] for t in request["tokens"]]))
            return dict(deleted_node_ids=sorted(deleted), requests=output,
                accounting="longest-path completion with releases/dependencies/resource edges; no span summation")

        result.update(status="SYNTHETIC_FIXTURE_ONLY" if synthetic else "CONDITIONAL_STRUCTURE_ONLY",
            evidence_type="SYNTHETIC_FIXTURE" if synthetic else "MULTI_GPU_EP_INPUT_NOT_INDEPENDENTLY_CERTIFIED",
            baseline_reconstruction_max_residual_us=residual, dependency_edges=len(dependencies),
            resource_serialization_edges=resource_edge_count,
            fixed_order_structural_conditional_bound=summarize(returns))
        control = set(trace.get("negative_control_node_ids", []))
        require(control <= set(nodes) and not control & returns, "negative control must identify non-return nodes")
        if control:
            result["negative_control"] = summarize(control)
            result["negative_control"]["zero_completion_shift"] = all(r["completion_shift_us"] <= tolerance for r in result["negative_control"]["requests"])
    except (KeyError, TypeError, AttributeError, NotIdentified) as exc:
        result.update(status="NOT_IDENTIFIED", fixed_order_structural_conditional_bound=None, negative_control=None)
        result["reasons"].append(str(exc))
    return result


def report(result):
    lines = [f"# Receiver exposed-return analysis: {result['status']}", "",
             f"Evidence: {result['evidence_type']}; scientific GO: false.", "",
             *result["limitations"], "", *result["reasons"]]
    bound = result["fixed_order_structural_conditional_bound"]
    if bound:
        lines.extend(["", "Fixed-order structural conditional bound; not a measured system Oracle.", "",
            "| Request | Observed latency us | Zero-service latency us | Completion shift us | Conditional fraction |",
            "|---|---|---|---|---|"])
        for row in bound["requests"]:
            lines.append("| " + " | ".join(str(row[k]) for k in ("request_id", "baseline_latency_us",
                "zeroed_service_latency_us", "completion_shift_us", "conditional_fraction")) + " |")
    lines.extend(["", "Current real-trace exporter and natural 8-GPU serving measurement remain UNRUN in this preparation."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(json.loads(args.trace.read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (args.output_dir / "report.md").write_text(report(result))
    return 0 if result["fixed_order_structural_conditional_bound"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
