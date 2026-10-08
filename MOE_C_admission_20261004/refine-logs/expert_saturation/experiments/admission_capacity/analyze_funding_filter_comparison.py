#!/usr/bin/env python3
"""Six actual victim-order runs with a current-funding least-progress arm.

The filtered arm changes only which current eligible victims reach the existing
least-progress ranking.  It is replayed from each policy's own retained
pre-action KV state.  This analyzer does not synthesize a filtered trajectory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys


sys.dont_write_bytecode = True

FILTER_FIELD = "filter_victims_by_funding"
VARIANTS = ("most_output", "least_progress", "least_feasible")
CONTEXT_ANALYZER_SHA256 = "9a1d4b2ff10a9bb2b1f0f48cf2ea05b598f883c8e1a03ae3033dc9cf4fb23c2a"
REUSED_ANALYSIS_SHA256 = {
    "analyze_recovery_holdout_comparison.py": "906ee0f5d32254706168b43ffbb544efed61fb7f1aa28ec6efac93c78960dc36",
    "analyze_apc_rotation.py": "1c728ec89fef52c3e908ba17ce0ee1e51bcdd5e4094793d7b06436f1076b5074",
    "analyze_restore_token_reservation.py": "f5def0228345f7fabdbc883776fa22b923412e36130cbbfedbb793aaf7e28342",
    "analyze_prefix_cache_baseline.py": "590ee20ee3648bb097abf82f29b23ac1e249ca588216bba4b0b13d713d18a473",
    "analyze_completion_headroom.py": "e23a33d1641b725bee12390f20c7d29220bd5f48f73c9888138b21a7a453126e",
}
EXPECTED_CELLS = (
    dict(label="funding-block0-most_output", block=0, variant="most_output",
         victim_order="most_output", filter_victims_by_funding=False),
    dict(label="funding-block0-least_progress", block=0, variant="least_progress",
         victim_order="least_progress", filter_victims_by_funding=False),
    dict(label="funding-block0-least_feasible", block=0, variant="least_feasible",
         victim_order="least_progress", filter_victims_by_funding=True),
    dict(label="funding-block1-least_feasible", block=1, variant="least_feasible",
         victim_order="least_progress", filter_victims_by_funding=True),
    dict(label="funding-block1-least_progress", block=1, variant="least_progress",
         victim_order="least_progress", filter_victims_by_funding=False),
    dict(label="funding-block1-most_output", block=1, variant="most_output",
         victim_order="most_output", filter_victims_by_funding=False),
)
COMPARISON_CONTRACT = (
    ("funding-block0-most_output", "funding-block0-least_progress"),
    ("funding-block0-least_progress", "funding-block0-least_feasible"),
    ("funding-block0-most_output", "funding-block0-least_feasible"),
    ("funding-block1-most_output", "funding-block1-least_progress"),
    ("funding-block1-least_progress", "funding-block1-least_feasible"),
    ("funding-block1-most_output", "funding-block1-least_feasible"),
    ("funding-block0-most_output", "funding-block1-most_output"),
    ("funding-block0-least_progress", "funding-block1-least_progress"),
    ("funding-block0-least_feasible", "funding-block1-least_feasible"),
)
SCOPE = (
    "One reused heterogeneous 32-document workload, one 6656-block physical "
    "pool and two reversed order blocks. All six actual cells must complete "
    "before comparisons. The funding filter uses only each step's retained "
    "before-state owned block counts. Full request costs, policy-specific future "
    "state, output differences and all repeats remain; no quality, significance, "
    "Oracle, independent-workload, strong-filtering-data, GPU-benefit or method-GO claim."
)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def replace_once(source, old, new, description):
    require(source.count(old) == 1, f"{description} source layout changed")
    return source.replace(old, new)


def expected_filter(variant):
    require(variant in VARIANTS, "unknown comparison variant")
    return variant == "least_feasible"


def validate_flag_contract(config, decisions, variant):
    enabled = expected_filter(variant)
    require(FILTER_FIELD in config and config[FILTER_FIELD] is enabled,
            "configured funding-filter flag differs")
    if enabled:
        require(decisions and all(FILTER_FIELD in row and row[FILTER_FIELD] is True
                                  for row in decisions),
                "enabled funding-filter decision flag missing or forged")
    else:
        require(all(FILTER_FIELD not in row for row in decisions),
                "disabled arm recorded a funding-filter decision flag")


def validate_arm_contract(config, raw, decisions, variant):
    order = "most_output" if variant == "most_output" else "least_progress"
    require(config["variant"] == variant and config["cap"] == raw["target_cap"] == 32,
            "arm/cap differs")
    require(config["rotation_victim_order"] == order
            and config["completion_policy"] == "rotate"
            and config["policy_family"] == "strong_simple_rotation",
            "rotation arm identity differs")
    validate_flag_contract(config, decisions, variant)


def filtered_rotation_replay(context, library):
    """Extend the reviewed context replay only at the selector call."""
    source = inspect.getsource(library.rotation.replay)
    excluded = ("q=d['pre_exchange_ownership']", "q['live_owned_blocks']",
                "qualification_receipts+=1")
    removed = {needle: sum(needle in line for line in source.splitlines())
               for needle in excluded}
    require(removed == {excluded[0]: 1, excluded[1]: 2, excluded[2]: 2},
            "context replay exclusion layout changed")
    source = "\n".join(line for line in source.splitlines()
                       if not any(needle in line for needle in excluded))
    source = replace_once(source, "victim_order='most_output'",
                          "victim_order='least_progress'", "victim order")
    source = replace_once(source,
                          "d['effective_victim_order']=='most_output'",
                          "d['effective_victim_order']=='least_progress'",
                          "effective victim order")
    source = replace_once(source,
                          "states[r]['prompt_tokens'],4096,states[r]['output_tokens']",
                          "states[r]['prompt_tokens'],states[r]['prompt_tokens']+1024,states[r]['output_tokens']",
                          "per-request context")
    old_call = ("expected=tracker.decide(k,views,waiting,before['pool']['free_blocks'],"
                "{r:need(r) for r in waiting})")
    new_call = ("released_blocks={r:owned(r) for r in running}\n"
                "            expected=tracker.decide(k,views,waiting,before['pool']['free_blocks'],"
                "{r:need(r) for r in waiting},released_blocks=released_blocks)")
    source = replace_once(source, old_call, new_call, "funding-filter selector call")
    source = replace_once(source, "    require(forced>0, 'INVALID_NO_ACTION')",
                          "    # No-action complete calibration remains retained.",
                          "no-action calibration gate")
    namespace = dict(library.rotation.replay.__globals__,
                     check_release=library.old_release)
    exec(compile(source, __file__ + ":filtered-context-replay", "exec"), namespace)
    replay = namespace["replay"]

    def checked(raw, decisions, selector):
        result = replay(raw, decisions, selector)
        result.update(funding_filter_replayed=True,
                      released_blocks_source="memory_trace.before.requests.block_counts[0]",
                      future_state_reused=False)
        return result

    return checked


def replay_factory(context, library, variant):
    if expected_filter(variant):
        return filtered_rotation_replay(context, library)
    return context.rotation_replay(library, variant)


def make_inspector(context):
    source = inspect.getsource(context.inspect_cell)
    source = replace_once(
        source,
        "        require(cfg['variant']==variant and cfg['cap']==raw['target_cap']==32 and cfg['rotation_victim_order']==('least_progress' if variant=='native' else variant), 'arm/cap differs')",
        "        validate_arm_contract(cfg, raw, decisions, variant)",
        "context arm contract",
    )
    source = replace_once(
        source,
        "        library.rotation_replay=rotation_replay(library, variant)",
        "        library.rotation_replay=replay_factory(context, library, variant)",
        "context replay factory",
    )
    namespace = dict(context.inspect_cell.__globals__,
                     context=context,
                     replay_factory=replay_factory,
                     validate_arm_contract=validate_arm_contract)
    exec(compile(source, __file__ + ":funding-filter-inspector", "exec"), namespace)
    return namespace["inspect_cell"]


def validate_campaign(campaign, metadata):
    cells = campaign["cells"]
    require(cells == metadata["cells"] == list(EXPECTED_CELLS),
            "canonical campaign/metadata cell inventory differs")
    return cells


def import_dependencies(context_path, analysis_library):
    require(sha(context_path) == CONTEXT_ANALYZER_SHA256,
            "reviewed context analyzer changed")
    context = load_module("funding_filter_context_analysis", context_path)
    sys.path.insert(0, str(analysis_library))
    try:
        library = load_module("funding_filter_analysis_library",
                              analysis_library / "analyze_recovery_holdout_comparison.py")
    finally:
        sys.path.pop(0)
    modules = (library, library.rotation, library.token, library.base, library.headroom)
    require({Path(module.__file__).name: sha(Path(module.__file__)) for module in modules}
            == REUSED_ANALYSIS_SHA256, "reviewed analysis dependency changed")
    context.retain_no_action_calibration(library)
    return context, library


def package_source(bundle, metadata):
    prepared = bundle / "preparation/pkg"
    require(prepared.is_dir(), "prepared package is missing")
    require(all(sha(prepared / name) == value
                for name, value in metadata["files_sha256"].items()),
            "prepared package changed")
    readback = bundle / "execution/readback/pkg"
    if readback.exists():
        require(readback.is_dir() and all(sha(readback / name) == value
                    for name, value in metadata["files_sha256"].items()),
                "readback package changed")
    return prepared


def analyze(bundle, context_path, analysis_library, dependencies=None):
    context, library = dependencies or import_dependencies(context_path, analysis_library)
    source = bundle / "preparation/pkg"
    metadata = read(bundle / "preparation/preparation.json")
    source = package_source(bundle, metadata)
    campaign = read(source / "campaign.json")
    specs = validate_campaign(campaign, metadata)
    workload_path = source / "inputs_preparation/prepared/heterogeneous/workload.json"
    input_config_path = source / "inputs_preparation/prepared/heterogeneous/config.json"
    inputs, input_config = read(workload_path), read(input_config_path)
    require(hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
            == input_config["workload_sha256"], "input workload hash differs")
    metrics = library.module("funding_filter_metrics", source / "metrics.py")
    selector = library.module("funding_filter_selector", source / "absence_rotation.py")
    inspector = make_inspector(context)
    cells = [inspector(bundle, spec, library, metadata, source, inputs,
                       input_config, metrics, selector) for spec in specs]
    comparisons = []
    if all(cell["eligible"] for cell in cells):
        for block in (0, 1):
            rows = {cell["variant"]: cell for cell in cells if cell["block"] == block}
            require(set(rows) == set(VARIANTS), "block variant inventory differs")
            comparisons.extend([
                context.compare(rows["most_output"], rows["least_progress"]),
                context.compare(rows["least_progress"], rows["least_feasible"]),
                context.compare(rows["most_output"], rows["least_feasible"]),
            ])
        by_variant = {variant: [cell for cell in cells if cell["variant"] == variant]
                      for variant in VARIANTS}
        for variant in VARIANTS:
            require(len(by_variant[variant]) == 2, "repeat inventory differs")
            first, second = sorted(by_variant[variant], key=lambda row: row["block"])
            comparisons.append(context.compare(first, second))
        require([(row["baseline"], row["action"]) for row in comparisons]
                == list(COMPARISON_CONTRACT), "comparison order differs")
    status = ("MEASUREMENT_ONLY" if comparisons else
              "UNRUN" if all(cell["status"] == "UNRUN" for cell in cells)
              else "INCOMPLETE")
    result = dict(
        status=status,
        scope=SCOPE,
        producer_sha256=sha(Path(__file__)),
        reused_context_analyzer={"path": str(context_path), "sha256": sha(context_path)},
        reused_analysis={str(Path(module.__file__)): sha(Path(module.__file__)) for module in
                         (library, library.rotation, library.token,
                          library.base, library.headroom)},
        prepared_metadata_sha256=sha(bundle / "preparation/preparation.json"),
        prepared_source_sha256=metadata["files_sha256"],
        comparison_contract=[dict(baseline=baseline, action=action)
                             for baseline, action in COMPARISON_CONTRACT],
        cells=cells,
        comparisons=comparisons,
    )
    for cell in cells:
        cell.pop("outputs", None)
    return result


def rejection_message(call):
    try:
        call()
    except (KeyError, TypeError, ValueError) as error:
        return str(error)
    raise ValueError("negative flag contract was accepted")


def legacy_default_replay(bundle, source, context, library):
    selector = library.module("funding_filter_legacy_selector",
                              source / "absence_rotation.py")
    default_calls = []
    original_tracker = selector.AbsenceRotation

    class DefaultPathTracker(original_tracker):
        def decide(self, *args, **kwargs):
            require("released_blocks" not in kwargs,
                    "unfiltered replay passed a funding map")
            default_calls.append(True)
            return super().decide(*args, **kwargs)

    selector.AbsenceRotation = DefaultPathTracker
    results = bundle / "execution/readback/results"
    variants = ("most_output", "least_progress", "least_progress", "most_output")
    labels = ("context-block0-most_output", "context-block0-least_progress",
              "context-block1-least_progress", "context-block1-most_output")
    rows = []
    for label, variant in zip(labels, variants):
        folder = results / label
        raw, decisions = read(folder / "raw.json"), read(folder / "headroom-decisions.json")
        replay = context.rotation_replay(library, variant)(raw, decisions, selector)
        rows.append(dict(label=label, variant=variant, raw_sha256=sha(folder / "raw.json"),
                         decisions_sha256=sha(folder / "headroom-decisions.json"),
                         causal_proposals_checked=replay["causal_proposals_checked"],
                         forced_preemptions=replay["forced_preemptions"],
                         released_blocks_argument="omitted"))
    require(default_calls, "legacy replay did not call the selector")
    return rows


def cpu_checks(bundle, legacy_bundle, context_path, analysis_library):
    context, library = import_dependencies(context_path, analysis_library)
    metadata = read(bundle / "preparation/preparation.json")
    source = package_source(bundle, metadata)
    validate_campaign(read(source / "campaign.json"), metadata)
    require(callable(filtered_rotation_replay(context, library)),
            "filtered context replay adapter did not compile")
    legacy = legacy_default_replay(legacy_bundle, source, context, library)

    valid_filtered = dict(variant="least_feasible", cap=32,
                          rotation_victim_order="least_progress",
                          completion_policy="rotate",
                          policy_family="strong_simple_rotation",
                          filter_victims_by_funding=True)
    valid_least = dict(valid_filtered, variant="least_progress",
                       filter_victims_by_funding=False)
    validate_flag_contract(valid_filtered, [{FILTER_FIELD: True}], "least_feasible")
    validate_flag_contract(valid_least, [{}], "least_progress")
    rejected = {
        "missing_config_flag": rejection_message(
            lambda: validate_flag_contract(
                {key: value for key, value in valid_filtered.items() if key != FILTER_FIELD},
                [{FILTER_FIELD: True}], "least_feasible")),
        "forged_filtered_config_false": rejection_message(
            lambda: validate_flag_contract(dict(valid_filtered, filter_victims_by_funding=False),
                                           [{FILTER_FIELD: True}], "least_feasible")),
        "forged_unfiltered_config_true": rejection_message(
            lambda: validate_flag_contract(dict(valid_least, filter_victims_by_funding=True),
                                           [{}], "least_progress")),
        "missing_filtered_decision_flag": rejection_message(
            lambda: validate_flag_contract(valid_filtered, [{}], "least_feasible")),
        "forged_filtered_decision_false": rejection_message(
            lambda: validate_flag_contract(valid_filtered, [{FILTER_FIELD: False}],
                                           "least_feasible")),
        "flag_present_on_unfiltered_decision": rejection_message(
            lambda: validate_flag_contract(valid_least, [{FILTER_FIELD: False}],
                                           "least_progress")),
    }
    unrun = analyze(bundle, context_path, analysis_library,
                    dependencies=(context, library))
    require(unrun["status"] == "UNRUN" and len(unrun["cells"]) == 6
            and len(unrun["comparisons"]) == 0
            and all(cell["status"] == "UNRUN" for cell in unrun["cells"]),
            "empty six-cell campaign was not retained as UNRUN")
    filtered_specs = [row for row in read(source / "campaign.json")["cells"]
                      if row["variant"] == "least_feasible"]
    filtered_raw = [bundle / "execution/readback/results" / row["label"] / "raw.json"
                    for row in filtered_specs]
    require(not any(path.exists() for path in filtered_raw),
            "actual filtered logs exist; preparation-only CPU status is stale")
    return dict(
        status="PASS_CPU_ONLY",
        scope=("Preparation-time analyzer checks only. Four retained old rotation "
               "trajectories are replayed through the new selector's default path. "
               "No filtered trajectory, GPU result, performance value or strong-filtering "
               "result is simulated."),
        old_actual_rotation_default_replay=legacy,
        filtered_replay_adapter_compiled=True,
        flag_contract=dict(valid_paths_checked=2, rejected=rejected),
        empty_campaign=dict(status=unrun["status"], cells=len(unrun["cells"]),
                            comparisons=len(unrun["comparisons"])),
        actual_filtered_logs="UNRUN",
        dependencies=dict(
            analyzer_sha256=sha(Path(__file__)),
            context_analyzer_sha256=sha(context_path),
            analysis_library_sha256=sha(analysis_library / "analyze_recovery_holdout_comparison.py"),
            reused_analysis_sha256=REUSED_ANALYSIS_SHA256,
            selector_sha256=sha(source / "absence_rotation.py"),
            prepared_metadata_sha256=sha(bundle / "preparation/preparation.json"),
        ),
    )


def write_exclusive(path, value):
    require(not path.exists(), f"refuse output overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--context-analyzer", required=True, type=Path)
    parser.add_argument("--analysis-library", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--cpu-checks-output", type=Path)
    parser.add_argument("--legacy-bundle", type=Path)
    args = parser.parse_args()
    if args.cpu_checks_output is not None:
        require(args.legacy_bundle is not None,
                "--legacy-bundle is required for CPU checks")
        result = cpu_checks(args.bundle, args.legacy_bundle,
                            args.context_analyzer, args.analysis_library)
        write_exclusive(args.cpu_checks_output, result)
        print(json.dumps(dict(status=result["status"],
                              legacy_cells=len(result["old_actual_rotation_default_replay"]),
                              empty_campaign=result["empty_campaign"],
                              actual_filtered_logs=result["actual_filtered_logs"])))
    else:
        require(args.legacy_bundle is None,
                "--legacy-bundle is CPU-check-only")
        result = analyze(args.bundle, args.context_analyzer, args.analysis_library)
        write_exclusive(args.output, result)
        print(json.dumps(dict(status=result["status"],
                              cells=[dict(label=cell["label"], status=cell["status"],
                                          error=cell.get("error")) for cell in result["cells"]],
                              comparisons=len(result["comparisons"]))))


if __name__ == "__main__":
    main()
