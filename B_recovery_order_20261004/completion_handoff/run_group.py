#!/usr/bin/env python3
"""Pinned normal-capacity controller, with a fixed completion-handoff ABBA plan."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT_SHA = '85278d8c2e90f9ccd7d37b8003781258d0d6f226ad3397285c3658dc3b98f101'


def adapted_source():
    path = ROOT.parent/'normal_capacity/run_group.py'; payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != PARENT_SHA:
        raise RuntimeError('Normal-capacity controller source changed')
    text = payload.decode()
    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise RuntimeError(f'Controller adaptation boundary changed: {old[:80]}')
        text = text.replace(old, new)
    replace("        tasks = [(cap, 'native') for cap in p['baseline_caps']]\n        if p['baseline_caps'] != [64, 192, 256] or p['engine_max_num_seqs'] != 256:\n            raise ValueError('Only the three predeclared operating points are allowed')\n        kv_bytes = 'auto'",
        "        if p['sequence'] != ['native', 'after_sample', 'after_sample', 'native'] or p['cap'] != 256 or p['engine_max_num_seqs'] != 256 or p['fixed_gpu_kv_bytes'] != 77242302464:\n            raise ValueError('Requires the fixed cap256 completion-handoff ABBA plan')\n        tasks = [(256, mode) for mode in p['sequence']]\n        kv_bytes = p['fixed_gpu_kv_bytes']\n        receipt['fixed_gpu_kv_bytes'] = kv_bytes")
    replace('A_RECOVERY_LEASE_MODE=\'off\', B_RECOVERY_ORDER=mode,',
        "A_RECOVERY_LEASE_MODE='off', B_RECOVERY_ORDER='native', B_COMPLETION_FINALIZE=mode,")
    replace("str(ROOT/'inputs'/f'cap{cap}')", "str(ROOT.parent/'normal_capacity/inputs'/f'cap{cap}')")
    replace("for f in [*sorted(ROOT.glob('*.py')), *sorted((ROOT.parent/'pkg').glob('*.py'))]",
        "for f in [*sorted(ROOT.glob('*.py')), *sorted((ROOT.parent/'normal_capacity').glob('*.py')), *sorted((ROOT.parent/'pkg').glob('*.py'))]")
    start = "            events = json.loads((cell/'output/recovery-order.json').read_text())['events']"
    end = "                receipt['simple_age_merged_with_native'] = age_equal"
    if text.count(start) != 1 or text.count(end) != 1:
        raise RuntimeError('Conditional STORE-order experiment boundary changed')
    replace(text[text.index(start):text.index(end)+len(end)],
        "            receipt['cells'][-1]['completion_finalize_mode'] = mode")
    return text


def main():
    namespace = dict(__name__='completion_handoff_controller', __file__=str(__file__))
    exec(compile(adapted_source(), str(__file__)+'[normal_capacity]', 'exec'), namespace)
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
