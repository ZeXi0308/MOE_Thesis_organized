#!/usr/bin/env python3
"""Describe context before each arm's earliest actual recovery-repeat action."""
from pathlib import Path


def adapted_source():
    path = Path(__file__).resolve().parent.parent/'recovery_fit/summarize_context.py'
    text = path.read_text()
    old = """        decisions = cell['recovery_fit_actions'].get('rows', [])
        if len(decisions) != 1:
            cells.append(dict(cell=directory.parent.name, status='NO_UNIQUE_DECISION'))
            continue
        stamp = decisions[0]['decision_s']"""
    new = """        decisions = [row for row in cell['recovery_repeat_actions'].get('rows', [])
                     if row.get('actual_queue_change') is True]
        if not decisions:
            cells.append(dict(cell=directory.parent.name, mode=cell['mode'],
                status='NO_ACTUAL_ACTION', actual_action_count=0,
                decision_anchor='No actual queue change; shadow observations are not interventions'))
            continue
        first_action = min(decisions, key=lambda row: (row['decision_s'], row.get('event_index', 0)))
        stamp = first_action['decision_s']"""
    changes = {
        old: new,
        "            status='ANALYZED', planned=cell['planned'], raw_requests=len(requests),":
        "            status='ANALYZED', planned=cell['planned'], raw_requests=len(requests),\n"
        "            actual_action_count=len(decisions), first_action_event_index=first_action.get('event_index'),\n"
        "            decision_anchor='Earliest actual queue-changing event decision_s; host observation before its mutation',",
        "        semantics='Host/client observations. Closed pre-decision gaps only; a gap crossing the decision is excluded. '":
        "        semantics='Host/client observations. Each arm uses the earliest actual queue-changing event decision_s, '\n"
        "        'the host observation immediately before that mutation, even when later actions occur. '\n"
        "        'Shadow opportunities do not define an intervention time. No-action arms are marked NO_ACTUAL_ACTION. '\n"
        "        'Closed pre-decision gaps only; a gap crossing the decision is excluded. '",
    }
    for old, new in changes.items():
        if text.count(old) != 1:
            raise RuntimeError('Frozen context adaptation boundary changed: '+old[:90])
        text = text.replace(old, new)
    return text


_namespace = dict(__name__='recovery_repeat_context', __file__=str(__file__))
exec(compile(adapted_source(), str(__file__)+'[fit-context]', 'exec'), _namespace)
analyze = _namespace['analyze']
main = _namespace['main']


if __name__ == '__main__':
    main()
