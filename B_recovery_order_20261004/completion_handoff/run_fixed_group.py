#!/usr/bin/env python3
"""Equal-output ABBA followup; immutable EOS drivers remain the execution base."""
import hashlib
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PINS = {
    'completion_handoff/run_group.py': '8b96f1ab9420f325b2e7471ee5dec20a283b5cddb6bb44eb38eb6400a15138a3',
    'completion_handoff/run_cell.py': 'd56ee158bfb0fcaa80ca4f6de07c15df2c0e79976db872a861ff42122c3be4a2',
    'completion_handoff/completion_finalize.py': '6ec52ce72cef642abb9ec63fe0ca5a549160b70d9d70559659222a53bd1c3e49',
    'normal_capacity/run_cell.py': '9f92617d44bbf8cda6992261129c35ff844a14a569dc8f259ed2bdedaa2388bb',
    'pkg/run_recovery_cadence.py': '7046411c3b19a4d023004ea873f807aac0719aa882246649e671ea9e5d073be9',
    'pkg/request_measurement.py': '1b322be02505381dedb0aaf47f58513dfef61d60bf98e8d9590e18e3012534e4',
}


def verify_sources():
    for name, expected in PINS.items():
        if hashlib.sha256((ROOT.parent/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Executed EOS source changed: '+name)


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('Fixed-output adaptation boundary changed: '+old[:90])
    return text.replace(old, new)


def fixed_termination_source(text):
    text = replace_once(text, "output_tokens=config['output_tokens'], output_mode='eos',",
        "output_tokens=config['output_tokens'], output_mode='fixed',")
    text = replace_once(text, '            ignore_eos=False, min_tokens=0,',
        '            ignore_eos=True, min_tokens=0,')
    text = replace_once(text, '        config, workload = load_inputs(args.inputs)',
        "        config, workload = load_inputs(args.inputs)\n"
        "        if (config['requests'] != 256 or config['output_tokens'] != 1024\n"
        "                or config.get('output_tokens_by_request') or config.get('ignore_eos') is not True\n"
        "                or config.get('min_tokens') != 0 or config.get('output_mode') != 'fixed'):\n"
        "            raise ValueError('Requires the controlled256 fixed1024 input contract')")
    wrapper_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    text = replace_once(text, "        config['recovery_lease_mode']=lease_mode",
        f"        config['fixed_output_wrapper_sha256']={wrapper_sha!r}\n"
        "        config['controlled_output_tokens_per_arm']=262144\n"
        "        config['controlled_scope']='Fixed1024 ignoring EOS; same natural input prefixes and arrivals; no natural-EOS or quality claim'\n"
        "        config['recovery_lease_mode']=lease_mode")
    return replace_once(text, 'shlex.join([sys.executable, *sys.argv])',
        "shlex.join([sys.executable, sys.argv[0], '--cell', *sys.argv[1:]])")


def adapted_source():
    verify_sources()
    spec = importlib.util.spec_from_file_location('executed_completion_controller', ROOT/'run_group.py')
    parent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent)
    text = parent.adapted_source()
    text = replace_once(text, "str(ROOT.parent/'normal_capacity/inputs'/f'cap{cap}')",
        "str(Path(p['inputs_dir'])/f'cap{cap}')")
    return replace_once(text, "str(ROOT/'run_cell.py'), '--inputs'",
        "str(ROOT/'run_fixed_group.py'), '--cell', '--inputs'")


def main():
    verify_sources()
    if len(sys.argv) > 1 and sys.argv[1] == '--cell':
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        text = (ROOT/'run_cell.py').read_text()
        text = replace_once(text, '        text = original()',
            '        text = fixed_termination_source(original())')
        namespace = dict(__name__='fixed_completion_cell', __file__=str(ROOT/'run_cell.py'),
                         fixed_termination_source=fixed_termination_source)
        exec(compile(text, str(ROOT/'run_cell.py')+'[fixed-output]', 'exec'), namespace)
    else:
        namespace = dict(__name__='fixed_completion_controller', __file__=str(ROOT/'run_group.py'))
        exec(compile(adapted_source(), str(ROOT/'run_group.py')+'[fixed-inputs]', 'exec'), namespace)
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
