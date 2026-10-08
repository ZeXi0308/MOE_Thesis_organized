"""Small host-ITL feedback; every input comes from a completed engine step."""
from __future__ import annotations

import math
from statistics import median


def apply_nonpreemptive_limit(scheduler, target, maximum):
    if type(target) is not int or not 1 <= target <= maximum:
        raise ValueError("target must be within the unchanged engine capacity")
    active = len(scheduler.running)
    if active > maximum:
        raise ValueError("running set exceeds the unchanged engine capacity")
    limit = max(target, active)
    scheduler.max_num_running_reqs = limit
    return limit


class AdmissionFeedback:
    def __init__(self, config, maximum):
        self.policy = config.get("policy", "static")
        self.initial = config["cap"]
        self.caps = list(config.get("feedback_caps", [8, 12, 16, 32]))
        self.slo = config["tpot_slo_s"]
        if self.policy not in ("feedback", "shadow", "single_down", "single_shadow"):
            raise ValueError("feedback helper requires feedback, shadow, single_down or single_shadow policy")
        if (not self.caps or any(type(c) is not int or not 1 <= c <= maximum for c in self.caps)
                or self.caps != sorted(set(self.caps)) or self.initial not in self.caps):
            raise ValueError("feedback caps must be increasing engine-valid integers including initial cap")
        if not math.isfinite(self.slo) or self.slo <= 0:
            raise ValueError("feedback requires a finite positive TPOT SLO")
        self.single = self.policy in ("single_down", "single_shadow")
        self.single_down_target = config.get("single_down_target", 16)
        if self.single and (type(self.single_down_target) is not int
                or self.single_down_target not in self.caps
                or self.single_down_target >= self.initial):
            raise ValueError("single_down_target must be a feedback cap below the initial cap")
        self.single_fired = False
        self.intent_target = self.initial
        self.completed_steps = self.last_change_step = 0
        self.observations, self.decisions, self.actions = [], [], []

    def observe(self, request_itls, *, received_s, available_s):
        self.completed_steps += 1
        if not request_itls:
            return
        if (not math.isfinite(received_s) or not math.isfinite(available_s)
                or available_s < received_s
                or any(not math.isfinite(v) or v < 0 for v in request_itls.values())):
            raise ValueError("invalid completed host ITL observation")
        if self.observations and received_s < self.observations[-1]["received_s"]:
            raise ValueError("host ITL observations moved backwards")
        self.observations.append(dict(completed_step=self.completed_steps, received_s=received_s,
            available_s=available_s, request_itls_s=dict(request_itls),
            step_median_itl_s=median(request_itls.values())))

    def decide(self, *, decision_start_s, active, waiting):
        window = self.observations[-4:]
        if window and window[-1]["available_s"] > decision_start_s:
            raise ValueError("feedback decision used an unavailable future observation")
        previous = self.intent_target
        cooldown = self.completed_steps - self.last_change_step
        recent = median(o["step_median_itl_s"] for o in window) if len(window) == 4 else None
        reason = "insufficient_history"
        if recent is not None:
            reason = "cooldown" if cooldown < 4 else "hold"
            if self.single:
                if self.single_fired:
                    reason = "single_action_latched"
                elif cooldown >= 4 and recent > self.slo:
                    self.intent_target = self.single_down_target
                    self.single_fired = True
                    reason = "itl_above_slo"
            elif cooldown >= 4:
                index = self.caps.index(previous)
                if recent > self.slo and index > 0:
                    self.intent_target = self.caps[index - 1]
                    reason = "itl_above_slo"
                elif recent < 0.8 * self.slo and waiting > 0 and active == previous and index + 1 < len(self.caps):
                    self.intent_target = self.caps[index + 1]
                    reason = "headroom_and_waiting"
                elif active > previous:
                    reason = "natural_drain"
        changed = self.intent_target != previous
        if changed:
            self.last_change_step = self.completed_steps
        applies = self.policy in ("feedback", "single_down")
        target = self.intent_target if applies else self.initial
        row = dict(decision_index=len(self.decisions), policy=self.policy,
            completed_steps=self.completed_steps, decision_start_s=decision_start_s,
            signal_cutoff_s=window[-1]["received_s"] if window else None,
            signal_available_s=window[-1]["available_s"] if window else None,
            window_completed_steps=[o["completed_step"] for o in window],
            recent_step_median_itl_s=recent, cooldown_steps=cooldown,
            active_before=active, waiting_before=waiting, intent_before=previous,
            intent_target=self.intent_target, target_cap=target, intent_changed=changed,
            applied_change=changed and applies, reason=reason)
        self.decisions.append(row)
        if changed:
            self.actions.append(row)
        return row
