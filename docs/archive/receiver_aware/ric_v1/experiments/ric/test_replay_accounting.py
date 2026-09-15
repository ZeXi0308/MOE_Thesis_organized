from __future__ import annotations

from dataclasses import replace
import heapq
from pathlib import Path
import math
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from ric import replay as replay_module  # noqa: E402
from ric.accounting import (  # noqa: E402
    RICAccountingError,
    TraceMetrics,
    assert_replay_conservation,
    assert_sham_feedback_cost_equivalence,
    deployment_decomposition,
    empirical_cvar,
    paired_retention_bootstrap,
    paired_trace_bootstrap,
    quantile_type1,
    trace_metrics_from_result,
)
from ric.replay import (  # noqa: E402
    TAX_NON_GRID_RULE,
    TAX_RECORD_COUNT_GRID,
    ContractTaxSurface,
    ReplayConfig,
    action_collapse_matrix,
    action_signature,
    run_replay,
    run_sham_against_reference,
)
from ric.scenario import ReplayWorld, build_complete_fixture_world  # noqa: E402
from ric.schema import RICValidationError, validate_full_background  # noqa: E402
from ric.wire import (  # noqa: E402
    ContractMessage,
    ContractRecord,
    ContractTax,
    encode_contract,
    join_identity_hash_parts,
)


def _exact_tax_surface() -> ContractTaxSurface:
    return ContractTaxSurface(
        points=tuple(
            (
                count,
                ContractTax(
                    state_build_us=0.001 * count,
                    hash_us=0.002 * count,
                    encode_us=0.003 * count,
                    transfer_us=0.004 * count,
                    decode_us=0.005 * count,
                    lookup_us=0.006 * count,
                    apply_us=0.007 * count,
                    policy_lookup_us=0.008 * count,
                ),
            )
            for count in TAX_RECORD_COUNT_GRID
        ),
        source_id="measured-test-surface",
    )


def _small_world(trace_index: int = 0) -> ReplayWorld:
    return build_complete_fixture_world(
        model_key="fixture",
        model_revision="fixture/moe@ric-v1",
        top_k=2,
        num_experts=8,
        trace_index=trace_index,
        seed=202607223001 + trace_index,
        payload_bytes=1024,
        closure_budget_us=200.0,
    )


def _force_same_timestamp_credit_fixture(world: ReplayWorld) -> tuple[ReplayWorld, str]:
    """Make one receiver completion coincide with its remaining sibling release."""

    target_join = sorted(world.joins)[0]
    first, second = world.joins[target_join]
    base = first.contribution.arrival_us
    first_end = (
        (base + first.stage_service.sender_egress_us)
        + first.stage_service.shared_cut_us
        + first.stage_service.receiver_ingress_us
    )
    rebuilt = []
    for index, task in enumerate(world.tasks):
        if task.task_id == first.task_id:
            ready = base
        elif task.task_id == second.task_id:
            ready = first_end
        else:
            # Preserve ready>=arrival while keeping all background tasks in the
            # workload but out of the target instantaneous fixture.
            ready = max(task.contribution.arrival_us, first_end + 1000.0 + index)
        rebuilt.append(replace(task, contribution=replace(task.contribution, ready_us=ready)))
    records = [task.contribution for task in rebuilt]
    audit = validate_full_background(
        records,
        top_k=world.top_k,
        num_experts=world.num_experts,
        ep_size=world.ep_size,
        expected_request_ids=world.expected_request_ids,
        expected_token_blocks_per_request=128,
        expert_to_sender=world.expert_to_sender,
        request_to_receiver=world.request_to_receiver,
        expected_layer_by_request=world.expected_layer_by_request,
        expected_model_revision=world.model_revision,
        score_join_identities=world.scored_joins,
    )
    changed = replace(world, tasks=tuple(rebuilt), full_load_audit=audit)
    return changed, second.task_id


def _delay_all_tasks_past_deadlines(world: ReplayWorld) -> ReplayWorld:
    """Keep the complete load while exposing clock-driven slack transitions."""

    anchor = max(task.contribution.deadline_us for task in world.tasks) + 1000.0
    rebuilt = tuple(
        replace(
            task,
            contribution=replace(
                task.contribution,
                ready_us=anchor + index,
            ),
        )
        for index, task in enumerate(world.tasks)
    )
    audit = validate_full_background(
        [task.contribution for task in rebuilt],
        top_k=world.top_k,
        num_experts=world.num_experts,
        ep_size=world.ep_size,
        expected_request_ids=world.expected_request_ids,
        expected_token_blocks_per_request=128,
        expert_to_sender=world.expert_to_sender,
        request_to_receiver=world.request_to_receiver,
        expected_layer_by_request=world.expected_layer_by_request,
        expected_model_revision=world.model_revision,
        score_join_identities=world.scored_joins,
    )
    return replace(world, tasks=rebuilt, full_load_audit=audit)


class CompleteWorldTests(unittest.TestCase):
    def test_frozen_olmoe_shape_is_full_4x128xtopk(self) -> None:
        world = build_complete_fixture_world(top_k=8, num_experts=64)
        self.assertEqual(len(world.expected_request_ids), 4)
        self.assertEqual(len(world.joins), 512)
        self.assertEqual(len(world.tasks), 4 * 128 * 8)
        self.assertEqual(world.full_load_audit.scored_join_count, 512)
        for siblings in world.joins.values():
            self.assertEqual(len(siblings), 8)
            self.assertEqual(
                len({task.contribution.arrival_us for task in siblings}), 1
            )

    def test_score_mask_never_changes_workload_or_action_trace(self) -> None:
        world = _small_world()
        subset = frozenset(sorted(world.scored_joins)[:64])
        masked = world.with_score_mask(subset)
        self.assertEqual(world.task_fingerprint, masked.task_fingerprint)
        self.assertEqual(world.service_fingerprint, masked.service_fingerprint)
        self.assertEqual(
            world.full_load_audit.resource_demand_fingerprint,
            masked.full_load_audit.resource_demand_fingerprint,
        )
        full_result = run_replay(world, arm="sender_fcfs")
        masked_result = run_replay(masked, arm="sender_fcfs")
        self.assertEqual(action_signature(full_result), action_signature(masked_result))
        self.assertEqual(
            full_result.completion_by_task_us, masked_result.completion_by_task_us
        )
        self.assertEqual(len(masked_result.scored_join_latencies_us), 64)
        self.assertEqual(len(masked_result.all_join_latencies_us), 512)


class ReplayCausalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.world = _small_world()

    def test_global_four_stage_full_drain_and_conservation(self) -> None:
        baseline = run_replay(self.world, arm="sender_fcfs")
        aware = run_replay(self.world, arm="ric_full_zero_delay")
        assert_replay_conservation((baseline, aware))
        self.assertEqual(
            baseline.completed_stage_count,
            3 * len(self.world.tasks) + len(self.world.joins),
        )
        self.assertEqual(baseline.completed_join_count, len(self.world.joins))
        self.assertEqual(
            {record.stage for record in baseline.action_trace},
            {
                "sender_egress",
                "shared_cut",
                "receiver_ingress",
                "receiver_combine",
            },
        )
        for task in self.world.tasks:
            rows = [row for row in baseline.action_trace if row.task_id == task.task_id]
            self.assertEqual([row.stage for row in rows], [
                "sender_egress", "shared_cut", "receiver_ingress"
            ])
            self.assertLessEqual(rows[0].service_end_us, rows[1].service_start_us)
            self.assertLessEqual(rows[1].service_end_us, rows[2].service_start_us)
        combine_rows = [
            row for row in baseline.action_trace if row.stage == "receiver_combine"
        ]
        self.assertEqual(len(combine_rows), len(self.world.joins))
        for row in combine_rows:
            # The replay's public completion ledger is the combine end, never
            # the last contribution-unpack end.
            self.assertIn(row.task_id, baseline.join_completion_us)
            self.assertEqual(
                baseline.join_completion_us[row.task_id], row.service_end_us
            )

    def test_future_release_and_service_cannot_change_current_action(self) -> None:
        by_sender: dict[int, list[object]] = {}
        for task in self.world.tasks:
            by_sender.setdefault(task.identity.sender_rank, []).append(task)
        sender, tasks = max(
            by_sender.items(),
            key=lambda item: max(
                task.contribution.ready_us for task in item[1]
            )
            - min(task.contribution.ready_us for task in item[1]),
        )
        target = max(tasks, key=lambda task: task.contribution.ready_us)
        cutoff = float(target.contribution.ready_us)

        def mutate_future_task(task: object) -> object:
            if task.task_id != target.task_id:
                return task
            stage = replace(
                task.stage_service,
                sender_pack_us=task.stage_service.sender_pack_us * 17.0,
                shared_cut_us=task.stage_service.shared_cut_us * 19.0,
                receiver_unpack_us=task.stage_service.receiver_unpack_us * 23.0,
            )
            return replace(
                task,
                contribution=replace(
                    task.contribution,
                    ready_us=cutoff + 10_000.0,
                    service_us=stage.total_us,
                ),
                stage_service=stage,
            )

        mutated_tasks = tuple(
            mutate_future_task(task) for task in self.world.tasks
        )
        mutated_audit = validate_full_background(
            [task.contribution for task in mutated_tasks],
            top_k=self.world.top_k,
            num_experts=self.world.num_experts,
            ep_size=self.world.ep_size,
            expected_request_ids=self.world.expected_request_ids,
            expected_token_blocks_per_request=128,
            expert_to_sender=self.world.expert_to_sender,
            request_to_receiver=self.world.request_to_receiver,
            expected_layer_by_request=self.world.expected_layer_by_request,
            expected_model_revision=self.world.model_revision,
            score_join_identities=self.world.scored_joins,
        )
        mutated_world = replace(
            self.world,
            tasks=mutated_tasks,
            full_load_audit=mutated_audit,
        )

        def current_sender_prefix(result: object) -> tuple[tuple[str, float], ...]:
            return tuple(
                (row.task_id, row.service_start_us)
                for row in result.action_trace
                if row.stage == "sender_egress"
                and row.resource_id == f"sender:{sender}:egress"
                and row.service_start_us < cutoff
            )

        for arm in (
            "sender_srpt",
            "topology_projected_finish",
            "ric_full_zero_delay",
        ):
            with self.subTest(arm=arm):
                original = current_sender_prefix(run_replay(self.world, arm=arm))
                changed = current_sender_prefix(run_replay(mutated_world, arm=arm))
                self.assertTrue(original)
                self.assertEqual(original, changed)

    def test_incremental_receiver_ledger_matches_scan_scheduler_result(self) -> None:
        incremental = run_replay(self.world, arm="topology_projected_finish")
        with patch.object(
            replay_module._Simulator,
            "_aggregate_for_receiver",
            replay_module._Simulator._aggregate_for_receiver_scan,
        ):
            scan_reference = run_replay(
                self.world, arm="topology_projected_finish"
            )
        self.assertEqual(incremental, scan_reference)

    def test_unsorted_queue_snapshot_matches_sorted_reference_bitwise(self) -> None:
        unsorted_snapshot = run_replay(self.world, arm="sender_srpt")
        with patch.object(
            replay_module._Simulator,
            "_views",
            replay_module._Simulator._views_sorted_reference,
        ):
            sorted_reference = run_replay(self.world, arm="sender_srpt")
        self.assertEqual(unsorted_snapshot, sorted_reference)

    def test_r_view_contains_only_current_sender_ready_joins(self) -> None:
        original = replay_module.validate_r_view
        observations = 0

        def checked(view: object) -> None:
            nonlocal observations
            original(view)
            observations += 1
            ready_joins = {
                task.identity.join_identity for task in view.base.sender.ready_tasks
            }
            exposed = {state.join_identity for state in view.receiver_join_state}
            self.assertTrue(exposed <= ready_joins)
            self.assertTrue(
                all(
                    task.identity.sender_rank == view.base.sender.sender_rank
                    for task in view.base.sender.ready_tasks
                )
            )

        with patch.object(replay_module, "validate_r_view", side_effect=checked):
            run_replay(self.world, arm="ric_full_zero_delay")
        self.assertGreater(observations, 0)

    def test_zero_delay_credit_applies_before_same_timestamp_schedule(self) -> None:
        world, remaining_task_id = _force_same_timestamp_credit_fixture(self.world)
        result = run_replay(world, arm="ric_compressed_zero_delay")
        selected = [
            row
            for row in result.action_trace
            if row.stage == "sender_egress" and row.task_id == remaining_task_id
        ]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].visible_missing_count, 1)

    def test_clock_driven_slack_bucket_transitions_emit_without_arrival(self) -> None:
        world = _delay_all_tasks_past_deadlines(self.world)
        target = world.joins[sorted(world.joins)[0]][0]
        arrival = target.contribution.arrival_us
        budget = target.contribution.deadline_us - arrival
        expected = tuple(arrival + fraction * budget for fraction in (0.5, 0.75, 1.0))
        charged = run_replay(world, arm="ric_wire_charged")
        assert charged.control_plan is not None
        emission_times = tuple(
            event.emission_us for event in charged.control_plan.events
        )
        for transition_us in expected:
            self.assertTrue(
                any(
                    math.isclose(
                        emission_us,
                        transition_us,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    for emission_us in emission_times
                )
            )

    def test_charged_replay_invokes_single_canonical_wire_apply(self) -> None:
        original = replay_module.apply_wire_contract
        calls = 0

        def wrapped(*args: object, **kwargs: object) -> object:
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        with patch.object(replay_module, "apply_wire_contract", side_effect=wrapped):
            result = run_replay(self.world, arm="ric_wire_charged")
        self.assertGreater(calls, 0)
        self.assertEqual(calls, result.contract_messages)
        self.assertIsNotNone(result.control_plan)
        self.assertEqual(calls, len(result.control_plan.events))

    def test_comparison_arms_use_same_bounded_wire_apply(self) -> None:
        original = replay_module.apply_wire_contract
        for arm in ("ric_compressed_zero_delay", "ric_compressed_delayed"):
            calls = 0

            def wrapped(*args: object, **kwargs: object) -> object:
                nonlocal calls
                calls += 1
                return original(*args, **kwargs)

            with patch.object(
                replay_module, "apply_wire_contract", side_effect=wrapped
            ):
                result = run_replay(self.world, arm=arm)
            self.assertGreater(calls, 0)
            self.assertEqual(calls, result.contract_messages)
            self.assertGreater(result.contract_bytes, 0)
            self.assertEqual(sum(result.control_component_us.values()), 0.0)

    def test_standard_drr_spends_sufficient_deficit_on_same_flow(self) -> None:
        sender_to_tasks: dict[int, list[object]] = {}
        for task in self.world.tasks:
            sender_to_tasks.setdefault(task.identity.sender_rank, []).append(task)
        selected = next(
            rows
            for rows in sender_to_tasks.values()
            if len({row.identity.request_id for row in rows}) >= 2
            and max(
                sum(row.identity.request_id == request for row in rows)
                for request in {row.identity.request_id for row in rows}
            )
            >= 2
        )
        sender = selected[0].identity.sender_rank
        first_flow = min({row.identity.request_id for row in selected})
        same = sorted(
            [row for row in selected if row.identity.request_id == first_flow],
            key=lambda row: row.task_id,
        )[:2]
        other = min(
            (row for row in selected if row.identity.request_id != first_flow),
            key=lambda row: row.task_id,
        )
        simulator = replay_module._Simulator(
            self.world,
            "sender_age_service_drr",
            ReplayConfig(drr_quantum_us=1000.0, starvation_us=1e9),
        )
        queue = [(0.0, row.task_id) for row in (*same, other)]
        first = simulator._choose_drr(sender, queue)
        queue = [row for row in queue if row[1] != first]
        second = simulator._choose_drr(sender, queue)
        self.assertEqual(
            simulator.tasks[first].identity.request_id,
            simulator.tasks[second].identity.request_id,
        )

    def test_drr_visit_continuation_exact_residual_and_inactive_reset(self) -> None:
        simulator = replay_module._Simulator(
            self.world,
            "sender_age_service_drr",
            ReplayConfig(drr_quantum_us=10.0, starvation_us=1e9),
        )
        sender = 0
        task_ids = [f"flow-a-{index}" for index in range(5)]
        task_ids.append("flow-b-0")
        simulator.tasks = {
            task_id: SimpleNamespace(
                identity=SimpleNamespace(
                    request_id="flow-a" if task_id.startswith("flow-a") else "flow-b"
                ),
                stage_service=SimpleNamespace(sender_pack_us=3.0),
            )
            for task_id in task_ids
        }
        queue = [(0.0, task_id) for task_id in task_ids[:5]]
        residuals = []
        for _ in range(3):
            chosen = simulator._choose_drr(sender, queue)
            queue = [row for row in queue if row[1] != chosen]
            residuals.append(simulator.drr_deficit_us[(sender, "flow-a")])
        self.assertEqual(residuals, [7.0, 4.0, 1.0])
        self.assertFalse(simulator.drr_visit_continuation[sender])

        # The fourth call starts a new visit and adds Q exactly once: 1+10-3=8.
        chosen = simulator._choose_drr(sender, queue)
        queue = [row for row in queue if row[1] != chosen]
        self.assertEqual(simulator.drr_deficit_us[(sender, "flow-a")], 8.0)
        self.assertTrue(simulator.drr_visit_continuation[sender])

        # Removing flow-a from the active set resets all of its residual credit.
        simulator._choose_drr(sender, [(0.0, "flow-b-0")])
        self.assertEqual(simulator.drr_deficit_us[(sender, "flow-a")], 0.0)
        self.assertFalse(simulator.drr_visit_continuation[sender])

    def test_exact_dynamic_tax_surface_covers_grid_non_grid_and_above16(self) -> None:
        surface = _exact_tax_surface()
        for count in (1, 4, 8, 16, 2, 7, 17, 31, 255):
            self.assertEqual(
                surface.tax_for(count),
                dict(surface.points)[count],
            )
        self.assertEqual(
            surface.non_grid_rule,
            TAX_NON_GRID_RULE,
        )
        self.assertEqual(len(surface.fingerprint), 64)
        with self.assertRaisesRegex(
            RICValidationError, "exact points for counts 1..255"
        ):
            ContractTaxSurface(
                points=tuple(
                    (count, dict(surface.points)[count])
                    for count in (1, 4, 8, 16)
                ),
                source_id="incomplete",
            )

    def test_charged_uses_each_events_actual_record_count_tax(self) -> None:
        surface = _exact_tax_surface()
        result = run_replay(
            self.world,
            arm="ric_wire_charged",
            config=ReplayConfig(contract_tax_surface=surface),
        )
        assert result.control_plan is not None
        self.assertEqual(
            sum(result.contract_record_count_histogram.values()),
            result.contract_messages,
        )
        for component in ContractTax.__dataclass_fields__:
            expected = math.fsum(
                float(getattr(event.tax, component))
                for event in result.control_plan.events
            )
            self.assertAlmostEqual(result.control_component_us[component], expected)
        for event in result.control_plan.events:
            self.assertAlmostEqual(
                event.delivery_us - event.emission_us,
                ReplayConfig().wire_delay_us + event.tax.total_us,
                places=9,
            )
        self.assertAlmostEqual(
            math.fsum(
                result.control_component_us[component]
                for component in ContractTax.__dataclass_fields__
            ),
            math.fsum(event.tax.total_us for event in result.control_plan.events),
        )
        self.assertEqual(
            result.contract_tax_surface_fingerprint, surface.fingerprint
        )
        double_counted = replace(
            result,
            control_component_us={
                **result.control_component_us,
                "end_to_end_apply_us": 1.0,
            },
        )
        with self.assertRaisesRegex(
            RICAccountingError, "additive control component ledger"
        ):
            trace_metrics_from_result(
                double_counted, closure_budget_us=200.0
            )

    def test_sham_rejects_event_tax_or_surface_fingerprint_tampering(self) -> None:
        surface = _exact_tax_surface()
        charged = run_replay(
            self.world,
            arm="ric_wire_charged",
            config=ReplayConfig(contract_tax_surface=surface),
        )
        assert charged.control_plan is not None
        first = charged.control_plan.events[0]
        bad_event = replace(first, tax=ContractTax())
        with self.assertRaisesRegex(
            RICValidationError, "event tax/surface mismatch"
        ):
            replace(
                charged.control_plan,
                events=(bad_event,) + charged.control_plan.events[1:],
            )
        with self.assertRaisesRegex(RICValidationError, "fingerprint mismatch"):
            replace(
                charged.control_plan,
                contract_tax_surface_fingerprint="0" * 64,
            )
        other_surface = replace(surface, source_id="different-measured-artifact")
        with self.assertRaisesRegex(RICValidationError, "tax differs"):
            run_sham_against_reference(
                self.world,
                charged_plan=charged.control_plan,
                config=ReplayConfig(contract_tax_surface=other_surface),
            )

    def test_sham_reuses_exact_charged_cadence_bytes_delay_and_tax(self) -> None:
        charged = run_replay(self.world, arm="ric_wire_charged")
        assert charged.control_plan is not None
        # The sham must decode/apply the exact wire plan, but its scheduler
        # must never construct an RView from the decoded semantic state.
        with patch.object(
            replay_module,
            "validate_r_view",
            side_effect=AssertionError("sham scheduler consumed receiver semantics"),
        ):
            sham = run_sham_against_reference(
                self.world, charged_plan=charged.control_plan
            )
        assert_sham_feedback_cost_equivalence(charged, sham)
        self.assertEqual(charged.control_component_us, sham.control_component_us)
        self.assertEqual(charged.control_plan.events, sham.control_plan.events)

    def test_wire_fault_falls_back_without_refunding_tax(self) -> None:
        faults = {
            (sender, receiver, 1): "malformed"
            for sender in range(self.world.ep_size)
            for receiver in range(self.world.ep_size)
        }
        result = run_replay(
            self.world,
            arm="ric_wire_charged",
            config=ReplayConfig(wire_faults=faults),
        )
        self.assertGreater(sum(result.fault_counts.values()), 0)
        self.assertGreater(result.fallback_decisions, 0)
        self.assertGreater(result.contract_bytes, 0)
        self.assertGreater(result.contract_bytes, result.contract_received_bytes)
        self.assertGreater(sum(result.control_component_us.values()), 0)

    def test_charged_control_tax_uses_explicit_fcfs_host_resources(self) -> None:
        result = run_replay(self.world, arm="ric_wire_charged")
        control = {
            key: value
            for key, value in result.resource_service_demand_us.items()
            if key.startswith("control:")
        }
        self.assertTrue(any(":receiver:" in key for key in control))
        self.assertTrue(any(":sender:" in key for key in control))
        self.assertTrue(all(value > 0 for value in control.values()))
        assert result.control_plan is not None
        standalone = ReplayConfig().wire_delay_us + max(
            event.tax.total_us for event in result.control_plan.events
        )
        self.assertTrue(
            any(
                event.delivery_us - event.emission_us > standalone + 1e-9
                for event in result.control_plan.events
            )
        )

    def test_sender_apply_fcfs_is_ordered_by_transfer_arrival_and_cache_completion(self) -> None:
        surface = _exact_tax_surface()
        simulator = replay_module._Simulator(
            self.world,
            "ric_wire_charged",
            ReplayConfig(contract_tax_surface=surface),
        )
        simulator.events.clear()
        simulator.pending_contracts.clear()

        selected = None
        for sender in range(self.world.ep_size):
            by_receiver = {}
            for join, siblings in self.world.joins.items():
                if any(task.identity.sender_rank == sender for task in siblings):
                    by_receiver.setdefault(join.receiver_rank, join)
            if len(by_receiver) >= 2:
                first_receiver, second_receiver = sorted(by_receiver)[:2]
                selected = (
                    sender,
                    by_receiver[first_receiver],
                    by_receiver[second_receiver],
                )
                break
        self.assertIsNotNone(selected)
        assert selected is not None
        sender, slow_join, fast_join = selected
        tax = surface.tax_for(1)

        def make_transfer(join, emission_us: float):
            key_hash, tag = join_identity_hash_parts(join)
            message = ContractMessage(
                sender_rank=sender,
                receiver_rank=join.receiver_rank,
                epoch=join.epoch,
                sequence=1,
                records=(
                    ContractRecord(
                        join_key_hash64=key_hash,
                        layer_id=join.layer_id,
                        missing_slot_mask=1,
                        identity_tag16=tag,
                        slack_bucket=2,
                        flags=0,
                    ),
                ),
            )
            payload = encode_contract(message)
            return replay_module._ControlTransfer(
                emission_us,
                sender,
                join.receiver_rank,
                join.epoch,
                1,
                1,
                tax,
                payload,
                len(payload),
            )

        slow = make_transfer(slow_join, 0.0)
        fast = make_transfer(fast_join, 1.0)
        # A pre-existing receiver-side encode backlog makes the message emitted
        # first arrive at the shared sender apply host last.
        simulator.receiver_control_available_us[slow_join.receiver_rank] = 100.0
        slow_arrival = simulator._charged_control_transfer_arrival_us(
            now_us=slow.emission_us,
            receiver_rank=slow.receiver_rank,
            tax=slow.tax,
        )
        fast_arrival = simulator._charged_control_transfer_arrival_us(
            now_us=fast.emission_us,
            receiver_rank=fast.receiver_rank,
            tax=fast.tax,
        )
        self.assertLess(fast_arrival, slow_arrival)
        # Insert in emission order; DES must still process transfer-arrival order.
        simulator._push(slow_arrival, 2, "control_transfer_arrival", slow)
        simulator._push(fast_arrival, 2, "control_transfer_arrival", fast)

        applied_receivers = []
        join_by_receiver = {
            slow_join.receiver_rank: slow_join,
            fast_join.receiver_rank: fast_join,
        }
        while simulator.events:
            event = heapq.heappop(simulator.events)
            if event.kind == "wire_delivery":
                delivery = event.payload
                join = join_by_receiver[delivery.receiver_rank]
                # Transfer arrival and sender-host reservation are not cache apply.
                self.assertNotIn(join, simulator._cache_snapshot(sender))
                simulator._process_event(event)
                self.assertIn(join, simulator._cache_snapshot(sender))
                applied_receivers.append(delivery.receiver_rank)
            else:
                self.assertEqual(event.kind, "control_transfer_arrival")
                simulator._process_event(event)
                transfer = event.payload
                self.assertNotIn(
                    join_by_receiver[transfer.receiver_rank],
                    simulator._cache_snapshot(sender),
                )

        self.assertEqual(
            applied_receivers,
            [fast_join.receiver_rank, slow_join.receiver_rank],
        )
        self.assertEqual(
            [event.receiver_rank for event in simulator.emitted_control_events],
            [fast_join.receiver_rank, slow_join.receiver_rank],
        )

    def test_action_collapse_matrix_is_complete_and_symmetric(self) -> None:
        fcfs = run_replay(self.world, arm="sender_fcfs")
        srpt = run_replay(self.world, arm="sender_srpt")
        matrix = action_collapse_matrix((fcfs, srpt))
        self.assertEqual(set(matrix), {"sender_fcfs", "sender_srpt"})
        self.assertTrue(matrix["sender_fcfs"]["sender_fcfs"])
        self.assertEqual(
            matrix["sender_fcfs"]["sender_srpt"],
            matrix["sender_srpt"]["sender_fcfs"],
        )


def _metric_row(
    trace: int,
    arm: str,
    *,
    cvar: float,
    violation: float,
    fingerprint_suffix: str = "",
    workload_seed: int = -1,
) -> TraceMetrics:
    if workload_seed == -1:
        workload_seed = 202607220000 + trace
    return TraceMetrics(
        trace_id=f"trace-{trace:02d}",
        workload_seed=workload_seed,
        model_key="olmoe",
        cell="poisson_rho60",
        arm=arm,
        closure_count=512,
        p50_us=cvar * 0.5,
        p95_us=cvar * 0.8,
        p99_us=cvar * 0.95,
        cvar99_us=cvar,
        violation_rate=violation,
        closure_budget_us=90.0,
        control_bytes_over_payload=0.0,
        stale_rate=0.0,
        fallback_rate=0.0,
        sender_ready_wait_mean_us=1.0,
        sender_ready_wait_p99_us=2.0,
        starvation_count=0,
        full_drain_goodput_per_us=1.0,
        queue_utilization=0.5,
        task_fingerprint=f"tasks-{trace}-{fingerprint_suffix}",
        service_fingerprint=f"service-{trace}-{fingerprint_suffix}",
        contract_tax_surface_fingerprint="tax-surface-fixture",
        score_mask_fingerprint=f"score-{trace}",
        resource_demand_fingerprint=f"resource-{trace}-{fingerprint_suffix}",
    )


class AccountingAndStatisticsTests(unittest.TestCase):
    def test_frozen_quantile_and_fractional_cvar(self) -> None:
        values = tuple(float(value) for value in range(1, 101))
        self.assertEqual(quantile_type1(values, 0.99), 99.0)
        self.assertEqual(empirical_cvar(values, 0.99), 100.0)
        short = (1.0, 2.0, 3.0, 4.0)
        self.assertEqual(empirical_cvar(short, 0.75), 4.0)

    def test_result_reduces_to_one_complete_trace_metric(self) -> None:
        result = run_replay(_small_world(), arm="sender_fcfs")
        row = trace_metrics_from_result(result, closure_budget_us=200.0)
        self.assertEqual(row.closure_count, 512)
        self.assertEqual(row.trace_id, result.trace_id)
        self.assertTrue(math.isfinite(row.cvar99_us))

    def test_paired_bootstrap_resamples_complete_trace_vectors(self) -> None:
        rows = []
        for trace in range(32):
            rows.append(_metric_row(trace, "baseline", cvar=100.0, violation=0.10))
            rows.append(_metric_row(trace, "candidate", cvar=90.0, violation=0.05))
        summary = paired_trace_bootstrap(
            rows,
            baseline_arm="baseline",
            candidate_arm="candidate",
            n_bootstrap=500,
            expected_trace_count=32,
        )
        self.assertAlmostEqual(summary.cvar99_relative_reduction_lcb, 0.10)
        self.assertAlmostEqual(summary.violation_absolute_reduction_lcb, 0.05)

    def test_unpaired_or_duplicate_trace_row_hard_fails(self) -> None:
        rows = [
            _metric_row(0, "baseline", cvar=100.0, violation=0.1),
            _metric_row(0, "candidate", cvar=90.0, violation=0.05),
        ]
        with self.assertRaisesRegex(RICAccountingError, "duplicate"):
            paired_trace_bootstrap(
                rows + [rows[0]],
                baseline_arm="baseline",
                candidate_arm="candidate",
                n_bootstrap=10,
            )
        with self.assertRaisesRegex(RICAccountingError, "unpaired"):
            paired_trace_bootstrap(
                rows[:1],
                baseline_arm="baseline",
                candidate_arm="candidate",
                n_bootstrap=10,
            )

    def test_fingerprint_drift_rejects_paired_bootstrap(self) -> None:
        rows = [
            _metric_row(0, "baseline", cvar=100.0, violation=0.1),
            _metric_row(
                0,
                "candidate",
                cvar=90.0,
                violation=0.05,
                fingerprint_suffix="drift",
            ),
        ]
        with self.assertRaisesRegex(RICAccountingError, "fingerprint"):
            paired_trace_bootstrap(
                rows,
                baseline_arm="baseline",
                candidate_arm="candidate",
                n_bootstrap=10,
            )

    def test_duplicate_seed_does_not_increase_independent_n(self) -> None:
        rows = [
            _metric_row(0, "baseline", cvar=100.0, violation=0.1),
            _metric_row(0, "candidate", cvar=90.0, violation=0.05),
            _metric_row(
                1,
                "baseline",
                cvar=100.0,
                violation=0.1,
                workload_seed=202607220000,
            ),
            _metric_row(
                1,
                "candidate",
                cvar=90.0,
                violation=0.05,
                workload_seed=202607220000,
            ),
        ]
        with self.assertRaisesRegex(RICAccountingError, "duplicate workload seed"):
            paired_trace_bootstrap(
                rows,
                baseline_arm="baseline",
                candidate_arm="candidate",
                n_bootstrap=10,
            )

    def test_retention_fails_closed_on_any_nonpositive_bootstrap_headroom(self) -> None:
        rows = [
            _metric_row(0, "B", cvar=100.0, violation=0.10),
            _metric_row(0, "R0", cvar=70.0, violation=0.04),
            _metric_row(0, "Rwire", cvar=85.0, violation=0.07),
            _metric_row(1, "B", cvar=100.0, violation=0.10),
            _metric_row(1, "R0", cvar=110.0, violation=0.04),
            _metric_row(1, "Rwire", cvar=105.0, violation=0.07),
        ]
        # Overall headroom is positive, but a resample containing only trace 1
        # has negative headroom.  Such draws must not be silently discarded.
        with self.assertRaisesRegex(
            RICAccountingError, "non-positive or near-zero"
        ):
            paired_retention_bootstrap(
                rows,
                baseline_arm="B",
                r0_arm="R0",
                charged_arm="Rwire",
                n_bootstrap=100,
                seed=1,
            )

    def test_retention_and_deployment_accounting(self) -> None:
        rows = []
        for trace in range(32):
            rows.extend(
                [
                    _metric_row(trace, "B", cvar=100.0, violation=0.10),
                    _metric_row(trace, "R0", cvar=80.0, violation=0.04),
                    _metric_row(trace, "Rwire", cvar=90.0, violation=0.07),
                ]
            )
        retention = paired_retention_bootstrap(
            rows,
            baseline_arm="B",
            r0_arm="R0",
            charged_arm="Rwire",
            n_bootstrap=500,
        )
        self.assertAlmostEqual(retention.point, 0.5)
        self.assertAlmostEqual(retention.lcb, 0.5)
        decomposition_rows = {
            "B": _metric_row(0, "B", cvar=100.0, violation=0.10),
            "R0": _metric_row(0, "R0", cvar=80.0, violation=0.04),
            "Rcmp0": _metric_row(0, "Rcmp0", cvar=82.0, violation=0.04),
            "RcmpD": _metric_row(0, "RcmpD", cvar=85.0, violation=0.05),
            "Rwire": _metric_row(0, "Rwire", cvar=90.0, violation=0.07),
        }
        decomposition = deployment_decomposition(
            decomposition_rows,
            baseline_arm="B",
            r0_arm="R0",
            compressed_arm="Rcmp0",
            delayed_arm="RcmpD",
            wire_arm="Rwire",
        )
        self.assertAlmostEqual(decomposition.net_value, 10.0)
        self.assertAlmostEqual(decomposition.residual, 0.0)
        drifted = dict(decomposition_rows)
        drifted["Rwire"] = replace(
            drifted["Rwire"], resource_demand_fingerprint="different-resource"
        )
        with self.assertRaisesRegex(RICAccountingError, "matched trace world"):
            deployment_decomposition(
                drifted,
                baseline_arm="B",
                r0_arm="R0",
                compressed_arm="Rcmp0",
                delayed_arm="RcmpD",
                wire_arm="Rwire",
            )


if __name__ == "__main__":
    unittest.main()
