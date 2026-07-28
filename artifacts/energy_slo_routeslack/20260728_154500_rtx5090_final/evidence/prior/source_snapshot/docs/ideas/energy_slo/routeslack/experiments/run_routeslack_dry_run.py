from __future__ import annotations

"""Run the RouteSlack protocol pipeline on non-evidence synthetic fixtures."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from typing import Iterable, Sequence

from routeslack_protocol import (
    CompletionSet,
    ContributionIdentity,
    Gate0Evidence,
    IdentityLedger,
    OnlineObservation,
    OracleInput,
    ServiceEnergySurface,
    SurfacePoint,
    evaluate_gate0,
    run_online_policy,
)


BASELINES = (
    "immediate_execution",
    "fixed_row_threshold",
    "fixed_timeout",
    "earliest_deadline_first",
    "least_loaded_replica",
    "min_predicted_finish",
    "lplb_like_token_balancing",
    "two_tier_static_power",
    "min_finish_plus_two_tier_power",
    "route_unaware_batch_kv_phase_energy_controller",
)
ORACLE = "future_known_oracle"
TEST_SUITES = (
    (
        "bcrd",
        ("discover", "-s", "docs/ideas/bcrd/experiments", "-p", "test_*.py"),
    ),
    (
        "routeslack",
        (
            "discover",
            "-s",
            "docs/ideas/energy_slo/routeslack/experiments",
            "-p",
            "test_*.py",
        ),
    ),
    (
        "route-row-contracts",
        (
            "docs.ideas.energy_slo.route_row_fp8.experiments.test_continuous_decode_harness",
            "docs.ideas.energy_slo.route_row_fp8.experiments.test_power_accounting",
        ),
    ),
    (
        "joulequeue",
        (
            "discover",
            "-s",
            "docs/ideas/energy_slo/joulequeue/experiments",
            "-p",
            "test_*.py",
        ),
    ),
)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _write_json(path: Path, value: object) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[object]) -> None:
    _write_text(
        path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args: list[str]) -> str:
    # The interactive PATH may put the git-ai wrapper first.  Provenance must
    # use the platform Git directly because the wrapper currently panics when
    # invoked by Python from a non-ASCII workspace path.
    git_executable = "/usr/bin/git"
    try:
        result = subprocess.run(
            [git_executable, *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as exc:
        return f"UNAVAILABLE: {exc}"
    return result.stdout.rstrip() if result.returncode == 0 else f"FAILED({result.returncode}): {result.stdout.rstrip()}"


def _environment() -> dict[str, object]:
    torch_info: dict[str, object]
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        torch_info = {"error": f"{type(exc).__name__}: {exc}", "cuda_available": False}
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch_info,
        "cuda_driver": None,
        "gpu": None,
        "nvml": None,
        "evidence_label": "[Observed] local environment probe; no GPU measurement",
    }


def _run_unit_tests() -> tuple[dict[str, object], str]:
    results: list[dict[str, object]] = []
    log_parts: list[str] = []
    for suite, arguments in TEST_SUITES:
        command = [
            sys.executable,
            "-B",
            "-m",
            "unittest",
            *arguments,
        ]
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
        output = completed.stdout
        count = None
        for line in output.splitlines():
            if line.startswith("Ran ") and " tests" in line:
                try:
                    count = int(line.split()[1])
                except (IndexError, ValueError):
                    count = None
        results.append(
            {
                "suite": suite,
                "exit_code": completed.returncode,
                "tests": count,
                "passed": completed.returncode == 0,
            }
        )
        log_parts.append(
            f"$ {' '.join(command)}\nexit_code={completed.returncode}\n{output.rstrip()}\n"
        )
    summary = {
        "suites": results,
        "tests": sum(int(row["tests"] or 0) for row in results),
        "failed_suites": sum(not bool(row["passed"]) for row in results),
        "all_passed": all(bool(row["passed"]) for row in results),
    }
    return summary, "\n".join(log_parts)


def _synthetic_contributions() -> tuple[ContributionIdentity, ...]:
    rows: list[ContributionIdentity] = []
    for request_index in range(2):
        for decode_step in range(2):
            for layer in range(2):
                for slot in range(2):
                    rows.append(
                        ContributionIdentity(
                            request_id=f"request-{request_index}",
                            input_event_id=f"document-{request_index}",
                            token_id=100 + decode_step,
                            decode_step=decode_step,
                            layer_id=layer,
                            expert_id=(request_index + decode_step + layer + slot * 3) % 8,
                            topk_slot=slot,
                            source_rank=request_index,
                            target_replica=(request_index + slot) % 2,
                        )
                    )
    return tuple(rows)


def _policy_fixture(seed: int) -> list[dict[str, object]]:
    """Exercise all interfaces with arbitrary cost units, never Joules."""

    rng = random.Random(seed)
    observation = OnlineObservation(now_ns=1_000, queue_depth=4, visible_rows=(1, 2, 4))
    completed = CompletionSet(
        token_keys=frozenset(
            (f"request-{request}", step) for request in range(2) for step in range(2)
        ),
        output_sha256=hashlib.sha256(b"routeslack-dry-run-exact-output").hexdigest(),
    )
    rows: list[dict[str, object]] = []
    for index, name in enumerate(BASELINES):
        decision = run_online_policy(
            lambda visible, offset=index: (visible.queue_depth + offset) % 3,
            observation,
        )
        rows.append(
            {
                "policy": name,
                "input_interface": "OnlineObservation",
                "decision": decision,
                "synthetic_cost_units": round(100.0 - min(index, 8) * 0.75 + rng.random() * 0.0, 6),
                "completed_token_count": len(completed.token_keys),
                "output_sha256": completed.output_sha256,
                "scientific_result_eligible": False,
            }
        )
    oracle_input = OracleInput(observation, future_arrival_ns=(1_100, 1_300))
    rows.append(
        {
            "policy": ORACLE,
            "input_interface": type(oracle_input).__name__,
            "decision": 2,
            "synthetic_cost_units": 90.0,
            "completed_token_count": len(completed.token_keys),
            "output_sha256": completed.output_sha256,
            "scientific_result_eligible": False,
        }
    )
    return rows


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    ratio = position - lower
    return ordered[lower] + ratio * (ordered[upper] - ordered[lower])


def _bootstrap_mean_ci(
    values: list[float], *, seed: int, replicates: int = 2_000
) -> tuple[float, float, float]:
    if not values:
        raise RuntimeError("bootstrap values cannot be empty")
    rng = random.Random(seed)
    means = [
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(replicates)
    ]
    point = sum(values) / len(values)
    return point, _percentile(means, 0.025), _percentile(means, 0.975)


def _host_noop_tax_fixture(
    contributions: tuple[ContributionIdentity, ...],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Measure CPU framework costs only; these are not GPU controller taxes."""

    observation = OnlineObservation(
        now_ns=1_000, queue_depth=4, visible_rows=(1, 2, 4)
    )
    sink: list[object] = []

    def base() -> None:
        _ = 1 + 1

    def instrumentation() -> None:
        _ = contributions[0].request_id

    def route_hook() -> None:
        sink.append(contributions[0])
        sink.clear()

    def logging() -> None:
        _ = json.dumps(contributions[0].as_dict(), sort_keys=True)

    def decision() -> None:
        _ = run_online_policy(lambda _: 0, observation)

    operations = {
        "empty_loop": base,
        "instrumentation": instrumentation,
        "route_hook": route_hook,
        "logging": logging,
        "decision_framework": decision,
    }
    trials = 25
    iterations = 2_000
    raw: list[dict[str, object]] = []
    by_name: dict[str, list[float]] = {}
    for name, operation in operations.items():
        samples: list[float] = []
        for trial in range(trials):
            started = time.perf_counter_ns()
            for _ in range(iterations):
                operation()
            elapsed = time.perf_counter_ns() - started
            value = elapsed / iterations / 1_000.0
            samples.append(value)
            raw.append(
                {
                    "operation": name,
                    "trial": trial,
                    "iterations": iterations,
                    "host_us_per_call": value,
                    "scientific_result_eligible": False,
                }
            )
        by_name[name] = samples
    base_p50 = statistics.median(by_name["empty_loop"])
    summary = {}
    base_samples = by_name["empty_loop"]
    for index, (name, samples) in enumerate(by_name.items()):
        mean, ci_low, ci_high = _bootstrap_mean_ci(
            samples, seed=20260728 + index
        )
        paired_increments = [
            value - base for value, base in zip(samples, base_samples)
        ]
        increment_mean, increment_low, increment_high = _bootstrap_mean_ci(
            paired_increments, seed=20261728 + index
        )
        summary[name] = {
            "host_mean_us_per_call": mean,
            "host_mean_ci95_us_per_call": [ci_low, ci_high],
            "host_p50_us_per_call": statistics.median(samples),
            "host_p99_us_per_call": _percentile(samples, 0.99),
            "p50_increment_over_empty_loop_us": max(
                statistics.median(samples) - base_p50, 0.0
            ),
            "paired_mean_increment_over_empty_loop_us": increment_mean,
            "paired_mean_increment_ci95_us": [increment_low, increment_high],
            "trials": trials,
            "iterations_per_trial": iterations,
        }
    return raw, {
        "scope": "local CPU host-only; no CUDA, NVML, route hook, or GPU energy",
        "independent_unit": "timed outer trial",
        "operations": summary,
    }


def run_dry_run(
    output_dir: Path,
    *,
    seed: int,
    run_tests: bool = False,
    include_files: Sequence[tuple[Path, str]] = (),
    provenance_commands: Sequence[str] = (),
) -> dict[str, object]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty artifact directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in ("raw", "processed", "figures", "logs"):
        (output_dir / child).mkdir(exist_ok=True)

    included_files: list[str] = []
    for source, relative in include_files:
        source = source.resolve()
        target = (output_dir / relative).resolve()
        if output_dir not in target.parents or not source.is_file():
            raise RuntimeError(
                f"invalid included artifact source/target: {source} -> {relative}"
            )
        if target.exists():
            raise RuntimeError(f"included artifact target already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        included_files.append(str(target.relative_to(output_dir)))

    contributions = _synthetic_contributions()
    ledger = IdentityLedger(expected_top_k=2)
    raw_contributions = []
    for stage in ("routed", "dispatched", "executed", "combined"):
        ledger.record(stage, contributions)
        raw_contributions.extend(
            {"stage": stage, **identity.as_dict()} for identity in contributions
        )
    ledger.assert_conserved()

    surface = ServiceEnergySurface(
        (
            SurfacePoint(1, "default", 10.0, 1.0),
            SurfacePoint(4, "default", 20.0, 2.0),
            SurfacePoint(1, "low", 15.0, 0.8),
            SurfacePoint(4, "low", 30.0, 1.6),
        ),
        default_tier="default",
    )
    fallback = surface.lookup(rows=99, tier="unknown")
    policy_rows = _policy_fixture(seed)
    noop_raw, noop_summary = _host_noop_tax_fixture(contributions)

    gate0 = evaluate_gate0(
        Gate0Evidence(
            native_continuous_decode=False,
            kv_advances_one=False,
            route_identity_complete=True,
            latency_window_aligned=False,
            energy_window_aligned=False,
            warmup_excluded=True,
            repeat_denominator_equal=True,
            thermal_state_logged=False,
            matched_completion_set=True,
            output_exactness=True,
            oracle_isolated=True,
        )
    )
    test_summary, test_log = (
        _run_unit_tests()
        if run_tests
        else ({"suites": [], "tests": 0, "failed_suites": 0, "all_passed": None}, "NOT_RUN\n")
    )
    run_status = (
        "DRY_RUN_BLOCKED_TEST_FAILURE"
        if run_tests and not bool(test_summary["all_passed"])
        else "DRY_RUN_COMPLETE"
    )
    summary = {
        "schema": "routeslack-dry-run-v1",
        "status": run_status,
        "formal_result": False,
        "gate0": gate0.status,
        "gate0_open_items": list(gate0.open_items),
        "synthetic_contributions_per_stage": len(contributions),
        "identity_stages": 4,
        "baselines_exercised": list(BASELINES),
        "oracle_exercised": ORACLE,
        "out_of_range_fallback": {
            "status": fallback.status,
            "action_eligible": fallback.action_eligible,
            "tier": fallback.point.tier,
        },
        "physical_energy_samples": 0,
        "physical_latency_samples": 0,
        "confidence_intervals": None,
        "included_development_artifacts": included_files,
        "unit_tests": test_summary,
        "noop_host_tax": noop_summary,
        "evidence_boundary": (
            "Synthetic audit fixture only. It verifies pipeline and fail-closed "
            "invariants, not a model, GPU, energy, latency, SLO, or Oracle result."
        ),
    }

    _write_jsonl(output_dir / "raw/contributions.jsonl", raw_contributions)
    _write_jsonl(output_dir / "raw/policy_results.jsonl", policy_rows)
    _write_jsonl(output_dir / "raw/noop_host_overhead.jsonl", noop_raw)
    _write_json(output_dir / "processed/dry_run_summary.json", summary)
    _write_json(output_dir / "environment.json", _environment())
    _write_text(
        output_dir / "config.yaml",
        "schema: routeslack-dry-run-v1\n"
        f"seed: {seed}\n"
        "formal: false\n"
        "workload: synthetic-audit-fixture\n"
        "models: []\n"
        "dtype: null\n",
    )
    command = (
        f"{shlex.quote(sys.executable)} -B "
        "docs/ideas/energy_slo/routeslack/experiments/run_routeslack_dry_run.py "
        f"--output-dir {shlex.quote(str(output_dir))} --seed {seed}"
        + (" --run-tests" if run_tests else "")
        + "".join(
            " --include-file "
            + shlex.quote(f"{source.resolve()}={relative}")
            for source, relative in include_files
        )
        + "".join(
            " --provenance-command " + shlex.quote(value)
            for value in provenance_commands
        )
    )
    _write_text(
        output_dir / "commands.sh",
        "#!/bin/sh\nset -eu\n"
        + "".join(value.rstrip() + "\n" for value in provenance_commands)
        + command
        + "\n",
    )
    _write_text(output_dir / "git_diff.patch", _git(["diff", "--no-ext-diff"]) + "\n")
    _write_text(
        output_dir / "figures/README.md",
        "# Figures intentionally absent\n\n"
        "No physical samples were collected. Generating the requested scientific "
        "plots from synthetic fixtures would be misleading.\n",
    )
    _write_text(
        output_dir / "logs/dry_run.log",
        f"{run_status}\nFORMAL_RESULT=false\nGATE0=FAIL\n"
        + "OPEN=" + ",".join(gate0.open_items) + "\n",
    )
    _write_text(output_dir / "logs/unit_tests.log", test_log)
    _write_text(
        output_dir / "verdict.md",
        "# Dry-run verdict\n\nMEASUREMENT_ONLY\n\n"
        "Gate 0 is not authorized. This synthetic artifact validates measurement "
        "contracts only and does not test, prove, or disprove a physical hypothesis.\n",
    )

    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    manifest = {
        "schema": "routeslack-artifact-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_result": False,
        "status": summary["status"],
        "gate0": gate0.status,
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_status": _git(["status", "--short"]),
        "command": command,
        "provenance_commands": list(provenance_commands),
        "random_seed": seed,
        "model_revisions": {},
        "files": {
            str(path.relative_to(output_dir)): {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        },
        "blocked": [
            "natural continuous-batching decode producer and serving timeline unavailable",
            "instrumented exactness is not qualified on both frozen formal models",
            "CUDA/NVML GPU unavailable",
            "physical latency/energy/thermal state unavailable",
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument(
        "--include-file",
        action="append",
        default=[],
        metavar="SOURCE=RELATIVE_PATH",
        help="copy a development artifact into the output and hash it in the manifest",
    )
    parser.add_argument(
        "--provenance-command",
        action="append",
        default=[],
        help="command that generated an included development artifact",
    )
    return parser.parse_args()


def _parse_includes(values: Sequence[str]) -> tuple[tuple[Path, str], ...]:
    parsed: list[tuple[Path, str]] = []
    for value in values:
        if "=" not in value:
            raise SystemExit("--include-file must be SOURCE=RELATIVE_PATH")
        source, relative = value.split("=", 1)
        if not source or not relative:
            raise SystemExit("--include-file must be SOURCE=RELATIVE_PATH")
        parsed.append((Path(source), relative))
    return tuple(parsed)


def main() -> None:
    args = parse_args()
    result = run_dry_run(
        args.output_dir,
        seed=args.seed,
        run_tests=args.run_tests,
        include_files=_parse_includes(args.include_file),
        provenance_commands=tuple(args.provenance_command),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
