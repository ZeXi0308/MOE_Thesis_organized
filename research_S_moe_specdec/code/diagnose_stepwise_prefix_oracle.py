#!/usr/bin/env python3
"""Single-step perfect-prefix-information diagnostic on a fixed gamma-7 trace.

python code/diagnose_stepwise_prefix_oracle.py \
  --trajectory runs/olmoe_ngram_matrix_r02/gamma7 \
  --calibration runs/FULLBUDGET/calibration/calibration.json \
  --output runs/derived/stepwise_prefix_r02_gamma7

The trajectory is never advanced under another action. Existing
AdaptiveGammaPolicy/AcceptanceEWMA are replayed using only prior observations;
their choices are not measured online adaptive choices. A separate reference
uses the future prefix revealed by the fixed gamma-7 trajectory. Both are
scored with the SAME budget-7 complete-step calibration table. No score is an
end-to-end latency, speedup, deployable oracle, or theoretical upper bound.

All eight caps must have identified progress and covered costs before reporting
a full-action perfect-prefix score or choice gap. Otherwise unknown caps and a
restricted identified-subset minimum are retained separately. No cost table =>
pending, without computing scores. CPU only; --self-test needs no experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import tempfile
import unittest

from analyze_acceptance_tracking import iter_steps, write_csv
from cost_model import AcceptanceEWMA, AdaptiveGammaPolicy, CalibrationTable
from plot_results import REGIME_FIELDS, _context, _finite, _requests, resolve_regime, route_instrumented


BUDGET = 7
CAPS = tuple(range(BUDGET+1))
BOUNDARIES = [
    "Development diagnostic on the executed fixed-gamma-7 trajectory, not independent held-out policy evaluation.",
    "The perfect-prefix reference uses future accepted prefixes from this same step. It is not implementable online.",
    "Cap changes can change subsequent tokens, proposals, KV state, batching and execution costs. Those changed trajectories are not simulated or verified.",
    "Cost comes from separate budget-7 complete synchronous host-step calibration cells or the existing within-range batch interpolation; it is not this counterfactual step's measured cost.",
    "A score is calibrated step cost divided by mean per-request progress, not measured effective TPOT, end-to-end policy performance or a theoretical bound.",
    "Do not convert mean scores or their gaps into an end-to-end speedup; never plot these as measured policy curves in F3.",
    "No extra draft is invented beyond the observed full offer. Accepted-prefix truncation is used only on identifiable, nonterminal request outputs.",
    "Replayed pre-step choices use the existing estimator and cost policy with a cold prior once per run; they are not actually executed adaptive choices. Unlogged warmup state is not reconstructed.",
    "Steps, requests and ranks share an execution trajectory and are not independent system repetitions. Raw run-level summaries are retained.",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identified_accepts(observation, cap):
    """Identify one request's accepted draft prefix after a hypothetical cap.

    Full acceptance of a SHORT offer does not imply any larger proposal exists.
    Full acceptance of an INCOMPLETE observed prefix does not reveal later ranks.
    A valid observed first rejection determines all later cap outcomes, provided
    native verified==scheduled establishes no ambiguous invalid slot placement.
    """
    if observation.get("terminal") or not observation.get("terminal_known"):
        return None,"terminal_or_unknown_termination"
    emitted=observation.get("emitted")
    accepted=observation.get("accepted")
    if emitted is None or emitted<=0:
        return None,"complete_request_output_unavailable"
    if accepted is not None and emitted != 1+accepted:
        return None,"observed_output_differs_from_one_plus_accepted"
    if cap==0:
        return 0,"zero_cap"
    available=observation.get("available")
    scheduled=observation.get("scheduled")
    verified=observation.get("verified")
    known=[x for x in (available,scheduled,verified,accepted) if x is not None]
    if any(type(x) is not int or x<0 for x in known):
        return None,"invalid_prefix_count"
    if (available is not None and scheduled is not None and scheduled>available
        or accepted is not None and available is not None and accepted>available
        or accepted is not None and verified is not None and accepted>verified
        or verified is not None and scheduled is not None and verified>scheduled):
        return None,"inconsistent_prefix_counts"
    if available==0:
        return 0,"observed_zero_offer"
    if accepted is None:
        return None,"accepted_prefix_unavailable"
    target=min(cap,available) if available is not None else cap
    if accepted>=target:
        return target,"accepted_prefix_covers_capped_offer"
    if verified is None or scheduled is None or verified!=scheduled:
        return None,"verified_prefix_or_invalid_slot_placement_unidentified"
    if accepted<verified:
        return accepted,"observed_first_rejection_before_later_caps"
    if target<=verified:
        return min(accepted,target),"complete_capped_prefix_observed"
    return None,"unobserved_later_prefix_or_offer"


def score_candidates(observations, batch, table):
    candidates=[]
    for cap in CAPS:
        values=[identified_accepts(o,cap) for o in observations]
        missing=Counter(reason for value,reason in values if value is None)
        progress=1+statistics.mean(value for value,_ in values) if values and not missing else None
        cost=None;cost_source=None;cost_error=None
        try:
            # Never replace draft costs using a direct-draft gamma-specific table.
            estimate=table.lookup(batch,cap,proposal_budget=BUDGET)
            cost,cost_source=estimate.total_step_ms,estimate.source
        except KeyError as exc:
            cost_error=str(exc)
        status="identified" if progress is not None and cost is not None else "progress_unidentified" if progress is None else "cost_uncovered"
        candidates.append({"gamma":cap,"proposal_budget":BUDGET,"status":status,
            "mean_progress_tokens_per_request":progress,"calibrated_complete_step_ms":cost,
            "cost_source":cost_source,"cost_coverage_error":cost_error,
            "score_ms_per_mean_progress":cost/progress if progress is not None and cost is not None else None,
            "identified_request_count":len(values)-sum(missing.values()),
            "unidentified_request_reasons":dict(missing),
            "identification_sources":dict(Counter(reason for value,reason in values if value is not None))})
    return candidates


def compare_choices(candidates, replayed_gamma):
    known=[c for c in candidates if c["score_ms_per_mean_progress"] is not None]
    best=min(known,key=lambda c:(c["score_ms_per_mean_progress"],c["gamma"])) if known else None
    complete=len(known)==len(CAPS)
    replayed=next((c for c in known if c["gamma"]==replayed_gamma),None)
    executed=next((c for c in known if c["gamma"]==BUDGET),None)
    minimum=best["score_ms_per_mean_progress"] if best and complete else None
    return {"all_caps_identified_and_cost_covered":complete,
        "identified_subset_best_gamma":best["gamma"] if best else None,
        "identified_subset_min_score":best["score_ms_per_mean_progress"] if best else None,
        "perfect_prefix_info_gamma":best["gamma"] if complete else None,
        "perfect_prefix_info_score":minimum,
        "replayed_pre_step_gamma":replayed_gamma,
        "replayed_choice_expost_score":replayed["score_ms_per_mean_progress"] if replayed else None,
        "executed_cap7_calibrated_score":executed["score_ms_per_mean_progress"] if executed else None,
        "replayed_choice_score_gap":replayed["score_ms_per_mean_progress"]-minimum if replayed and minimum is not None else None,
        "executed_cap7_score_gap":executed["score_ms_per_mean_progress"]-minimum if executed and minimum is not None else None}


def analyze_run(record, source, context, table, emit, alpha=.7, decay=.9, prior_tested_count=1.0):
    if route_instrumented(record,context): raise ValueError("route instrumentation is not eligible")
    if not record.get("all_finished"): raise ValueError("complete performance population required")
    if int(record.get("gamma",-1))!=BUDGET or int(record.get("proposal_budget",-1))!=BUDGET:
        raise ValueError("fixed gamma=proposal_budget=7 trajectory required")
    if record.get("mode") not in ("native","static") or context.get("native_batch_table"):
        raise ValueError("fixed untruncated gamma-7 trajectory required, without a native batch table")
    regime=resolve_regime(record,context)
    for key in REGIME_FIELDS:
        if key not in table.metadata or str(table.metadata[key])!=str(regime[key]):
            raise ValueError(f"cost/trajectory domain mismatch or unavailable metadata: {key}")
    estimator=AcceptanceEWMA(alpha=alpha,decay=decay,alpha_estimator="exposure_ratio",prior_tested_count=prior_tested_count)
    initial=estimator.snapshot()
    policy=AdaptiveGammaPolicy(table,estimator)
    counts=Counter();eligibility=Counter();unknown=Counter();choice_sources=Counter()
    choices={name:Counter() for name in ("replayed_pre_step","perfect_prefix_info","identified_subset")}
    metrics={key:[] for key in ("perfect_prefix_info_score","replayed_choice_expost_score","executed_cap7_calibrated_score","replayed_choice_score_gap","executed_cap7_score_gap")}
    source=str(Path(source).resolve())
    for wave_index,wave,host,step,config,observations,identity_source in iter_steps(record):
        counts["host_steps"]+=1
        if not observations:
            if step.get("decode_batch",0): eligibility["decode_identity_unavailable"]+=1
            continue
        counts["decode_steps"]+=1;counts["decode_request_steps"]+=len(observations)
        counts["terminal_request_steps"]+=sum(o["terminal"] for o in observations)
        counts["terminal_recovered_from_runner"]+=sum(o["terminal_recovered"] for o in observations)
        counts["unknown_terminal_request_steps"]+=sum(not o["terminal_known"] for o in observations)
        eligible=[o for o in observations if o["terminal_known"] and not o["terminal"]]
        next_budget=int(step.get("proposal_budget_next",BUDGET))
        full_budget_known=next_budget==BUDGET and int(step.get("truncated_by_controller",0))==0
        reason=("mixed_or_non_decode_phase" if step.get("phase")!="decode" else
                "terminal_or_unknown_request_in_step" if len(eligible)!=len(observations) else
                "full_budget_exposure_not_established" if not full_budget_known else None)
        if reason:
            eligibility[reason]+=1
        else:
            counts["eligible_pure_decode_steps"]+=1
            counts["eligible_pure_decode_request_steps"]+=len(eligible)
            batch=int(step["decode_batch"])
            # The online policy receives the recorded pre-schedule control batch,
            # never the future accepted prefix. Ex-post scores use actual B.
            control_batch=step.get("control_batch")
            replayed_gamma=None;decision_source=None;decision_error=None;predicted_score=None
            if type(control_batch) is int and control_batch>0:
                try:
                    choice=policy.choose(control_batch,draft_gamma=BUDGET)
                    replayed_gamma=choice.gamma;decision_source=choice.source
                    predicted_score=choice.ms_per_output_token_per_request
                except ValueError as exc:
                    if "no calibrated candidate" not in str(exc): raise
                    replayed_gamma=0;decision_source="uncovered_calibration_fallback"
                    decision_error=str(exc)
            else:
                decision_error="pre-schedule control batch missing or zero; no adaptive choose invocation reconstructed"
            candidates=score_candidates(eligible,batch,table)
            compared=compare_choices(candidates,replayed_gamma)
            for candidate in candidates:
                if candidate["mean_progress_tokens_per_request"] is None:
                    unknown[f"gamma{candidate['gamma']}:progress_unidentified"]+=1
                if candidate["calibrated_complete_step_ms"] is None:
                    unknown[f"gamma{candidate['gamma']}:cost_uncovered"]+=1
            if compared["all_caps_identified_and_cost_covered"]: counts["full_action_comparison_steps"]+=1
            else: counts["partially_identified_or_uncovered_steps"]+=1
            if replayed_gamma is not None: choices["replayed_pre_step"][replayed_gamma]+=1
            if compared["perfect_prefix_info_gamma"] is not None: choices["perfect_prefix_info"][compared["perfect_prefix_info_gamma"]]+=1
            if compared["identified_subset_best_gamma"] is not None: choices["identified_subset"][compared["identified_subset_best_gamma"]]+=1
            if decision_source: choice_sources[decision_source]+=1
            # Every reported score mean uses the same paired comparison cohort.
            # Partial-action scores remain in per-step output, not these means.
            if compared["all_caps_identified_and_cost_covered"] and compared["replayed_choice_score_gap"] is not None:
                counts["paired_replayed_reference_steps"]+=1
                for key in metrics:
                    metrics[key].append(compared[key])
            actual_progress=statistics.mean(o["emitted"] for o in eligible) if all(o["emitted"] is not None for o in eligible) else None
            emit({"source":source,"wave":wave,"wave_index":wave_index,"step":step["step"],
                "actual_decode_batch":batch,"pre_schedule_control_batch":control_batch,
                "request_count":len(eligible),"identity_source":identity_source,
                "observed_host_step_ms":host["host_step_ms"],"observed_gamma7_mean_emitted_tokens":actual_progress,
                "observed_cap7_host_ms_per_mean_progress":host["host_step_ms"]/actual_progress if actual_progress else None,
                "replayed_decision_cost_source":decision_source,"replayed_decision_error":decision_error,
                "replayed_pre_step_predicted_score":predicted_score,
                "replayed_alpha_before":estimator.alpha,
                "replayed_availability_before":estimator.availability_survival(BUDGET),
                "replayed_prefix_survival_before":estimator.prefix_survival(BUDGET),
                "available_drafts": [o["available"] for o in eligible],
                "scheduled_drafts": [o["scheduled"] for o in eligible],
                "verified_drafts": [o["verified"] for o in eligible],
                "accepted_drafts": [o["accepted"] for o in eligible],
                "request_ids":[o["id"] for o in eligible],"candidates":candidates,**compared,
                "evidence_kind":"fixed_trajectory_single_step_diagnostic_not_policy_performance"})
        # Preserve the existing tracking estimator semantics even when a whole
        # step is excluded from scoring: other nonterminal request observations
        # remain valid history. Updates occur strictly after pre-step decisions.
        valid=[o for o in eligible if o["accepted"] is not None and o["verified"] is not None]
        estimator.update([o["accepted"] for o in valid],[o["verified"] for o in valid])
        offered=[o["available"] for o in eligible if o["available"] is not None] if full_budget_known else []
        if offered: estimator.update_availability(offered,BUDGET)
        counts["history_acceptance_request_steps"]+=len(valid)
        counts["history_availability_request_steps"]+=len(offered)
    requests=_requests(record)
    repeat=re.search(r"(?:^|-)r(\d+)(?:-|$)",str(record.get("label","")))
    summary={**regime,"source":source,"source_sha256":digest(source),"run_id":record.get("label"),
        "batch":int(record["batch"]),"repeat":int(repeat.group(1)) if repeat else None,
        "sample_source_indices":sorted(r["source_index"] for r in requests),
        "request_count":len(requests),"output_tokens":record.get("output_tokens"),
        "gamma":BUDGET,"proposal_budget":BUDGET,"counts":dict(counts),"excluded_step_reasons":dict(eligibility),
        "unidentified_cap_counts":dict(unknown),"choice_histograms":{k:dict(v) for k,v in choices.items()},
        "replayed_decision_cost_sources":dict(choice_sources),"initial_replayed_estimator":initial,
        "final_replayed_estimator":estimator.snapshot(),
        "estimator_boundary":"cold prior once per run; existing exposure_ratio estimator, continuous recorded waves; same-step outcomes update only after scoring",
        "score_aggregation_boundary":"equal-weight means over the same steps with all eight caps identified/cost-covered and a scorable replayed choice; not end-to-end or independent-repetition estimates",
        "actual_executed_policy":"fixed cap7; replayed choices were never executed",
        "status":"paired_step_diagnostic_computed" if counts["paired_replayed_reference_steps"] else "no_identified_full_action_pairs",
        "evidence_kind":"development_fixed_trajectory_step_diagnostic"}
    for key,values in metrics.items():
        summary[key+"_count"]=len(values)
        summary[key+"_mean"]=statistics.mean(values) if values else None
        summary[key+"_median"]=statistics.median(values) if values else None
        summary[key+"_max"]=max(values) if values else None
    return summary


def validate_cost_table(path):
    obj=json.loads(Path(path).read_text())
    records=obj.get("records",[])
    if not records: raise ValueError("no measured cost records")
    if any(r.get("proposal_budget")!=BUDGET or r.get("evidence_kind")!="measured_complete_step_cost" or _finite(r.get("total_step_ms")) is None for r in records):
        raise ValueError("every cost record must be a measured budget-7 complete-step cost")
    batches=sorted({int(r["batch"]) for r in records})
    cells={(int(r["batch"]),int(r["gamma"])) for r in records}
    missing=[(b,g) for b in batches for g in CAPS if (b,g) not in cells]
    if missing: raise ValueError(f"incomplete cap grid at calibrated batches: {missing}")
    return CalibrationTable(records,obj.get("metadata",{}))


def write_readme(output,status):
    text="# Fixed-trajectory single-step prefix diagnostic\n\n"
    text+=f"Status: `{status}`. This directory is separate from measured F3 policy comparisons.\n\n"
    text+="For cap g, identify each nonterminal request's accepted prefix only from its actual offer, verified prefix and first rejection. Progress is one target token plus that identified prefix. A short fully accepted offer stays short for all larger caps. Unseen later prefixes remain unidentified.\n\n"
    text+="The score is `C(actual_B, g, proposal_budget=7) / mean_request_progress(g)`. The pre-step replay uses the existing controller and only past observations. The perfect-prefix reference uses future information from the same recorded step. Full-action minima/gaps require all eight actions to be identified and cost-covered; restricted-subset minima are separately named.\n\n"
    text+="`step_diagnostics.jsonl.gz` retains every eligible step's candidate scores and identification/coverage reasons. `run_summary.csv` and `diagnostic.json` retain raw run summaries and choice histograms. Censoring counts include plugin flags and independently recovered final client chunks.\n\n"
    text+="\n".join("- "+boundary for boundary in BOUNDARIES)+"\n"
    (output/"README.md").write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trajectory",nargs="*",default=[])
    parser.add_argument("--calibration")
    parser.add_argument("--output")
    parser.add_argument("--alpha",type=float,default=.7)
    parser.add_argument("--decay",type=float,default=.9)
    parser.add_argument("--prior-tested-count",type=float,default=1.0)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test: self_test();return
    if not args.output: parser.error("--output is required")
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if (output/"step_diagnostics.jsonl.gz").exists():
        raise FileExistsError("choose a fresh output directory to preserve previously computed diagnostics")
    sources=[]
    for value in args.trajectory:
        path=Path(value)
        sources.extend(sorted(path.rglob("perf-*.json")) if path.is_dir() else [path] if path.is_file() else [])
    sources=sorted(set(p.resolve() for p in sources))
    report={"schema_version":1,"status":"pending_cost_calibration","input_trajectory_files":[str(p) for p in sources],
        "requested_trajectory_paths":args.trajectory,"calibration":args.calibration,
        "replayed_estimator":{"alpha":args.alpha,"decay":args.decay,"alpha_estimator":"exposure_ratio","prior_tested_count":args.prior_tested_count},
        "boundaries":BOUNDARIES,"analysis_code_sha256":digest(__file__),"runs":[],"exclusions":[]}
    if args.calibration and Path(args.calibration).is_file():
        try:
            table=validate_cost_table(args.calibration)
            report.update(calibration_sha256=digest(args.calibration),calibration_metadata=table.metadata)
        except (ValueError,KeyError,TypeError) as exc:
            report["cost_validation_error"]=str(exc);table=None
        if table is not None:
            report["status"]="pending_trajectory" if not sources else "development_diagnostic_computed"
            with gzip.open(output/"step_diagnostics.jsonl.gz","wt") as stream:
                for source in sources:
                    try:
                        record=json.loads(source.read_text())
                        # A later alignment failure must not leave a rejected
                        # run's earlier scores in the accepted step artifact.
                        with tempfile.TemporaryFile(mode="w+t") as staged:
                            emit=lambda row:staged.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
                            summary=analyze_run(record,source,_context(source),table,emit,args.alpha,args.decay,args.prior_tested_count)
                            staged.seek(0)
                            for line in staged: stream.write(line)
                        report["runs"].append(summary)
                    except (ValueError,KeyError,TypeError) as exc:
                        report["exclusions"].append({"source":str(source),"reason":str(exc)})
            if sources and not report["runs"]: report["status"]="no_eligible_trajectory"
            elif report["exclusions"]: report["status"]="partial_development_diagnostic"
            elif report["runs"] and not any(r["status"]=="paired_step_diagnostic_computed" for r in report["runs"]):
                report["status"]="no_identified_full_action_pairs"
    write_csv(output/"run_summary.csv",report["runs"])
    (output/"diagnostic.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    write_readme(output,report["status"])
    print(json.dumps({"status":report["status"],"run_records":len(report["runs"]),"output":str(output)}))


def self_test():
    def observation(offer=7,scheduled=7,verified=7,accepted=2):
        return {"available":offer,"scheduled":scheduled,"verified":verified,"accepted":accepted,
                "terminal":False,"terminal_known":True,"emitted":1+accepted}
    class PrefixTests(unittest.TestCase):
        def test_short_offer_and_zero_offer_do_not_invent_drafts(self):
            o=observation(offer=2,scheduled=2,verified=2,accepted=2)
            self.assertEqual([identified_accepts(o,g)[0] for g in CAPS],[0,1,2,2,2,2,2,2])
            zero=observation(offer=0,scheduled=0,verified=0,accepted=0)
            self.assertEqual([identified_accepts(zero,g)[0] for g in CAPS],[0]*8)
        def test_unseen_prefix_is_unknown_but_first_reject_is_informative(self):
            o=observation(scheduled=2,verified=2,accepted=2)
            self.assertEqual(identified_accepts(o,2)[0],2)
            self.assertIsNone(identified_accepts(o,3)[0])
            o=observation(scheduled=2,verified=2,accepted=1)
            self.assertEqual(identified_accepts(o,7)[0],1)
        def test_censor_output_and_invalid_slots_do_not_become_prefix_truth(self):
            o=observation();o["terminal"]=True
            self.assertIsNone(identified_accepts(o,1)[0])
            o=observation();o["emitted"]=2
            self.assertIsNone(identified_accepts(o,1)[0])
            o=observation(scheduled=7,verified=3,accepted=2)
            self.assertIsNone(identified_accepts(o,7)[0])
            self.assertEqual(identified_accepts(o,2)[0],2)
        def test_perfect_prefix_can_select_less_than_fixed_or_prior(self):
            table=CalibrationTable([{"batch":1,"gamma":g,"proposal_budget":7,"total_step_ms":1+.2*g} for g in CAPS])
            candidate=score_candidates([observation(accepted=1)],1,table)
            compared=compare_choices(candidate,7)
            self.assertEqual(compared["perfect_prefix_info_gamma"],1)
            self.assertGreater(compared["replayed_choice_score_gap"],0)
            candidate[3]["score_ms_per_mean_progress"]=None
            incomplete=compare_choices(candidate,7)
            self.assertIsNone(incomplete["perfect_prefix_info_score"])
            self.assertIsNone(incomplete["replayed_choice_score_gap"])
            self.assertIsNotNone(incomplete["identified_subset_min_score"])
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(PrefixTests)
    if not unittest.TextTestRunner().run(suite).wasSuccessful(): raise SystemExit(1)


if __name__=="__main__": main()
