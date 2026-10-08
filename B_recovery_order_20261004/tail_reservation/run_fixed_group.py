#!/usr/bin/env python3
"""Controlled fixed1024 native/full-tail ABBA under the unchanged whole-group lock."""
from pathlib import Path

from run_fixed_cell import load_fixed

ROOT = Path(__file__).resolve().parent


def main():
    fixed = load_fixed()

    def fixed_group_source(text):
        text = fixed.replace_once(text,
            "str(ROOT.parent/'normal_capacity/inputs'/f'cap{cap}')",
            "str(Path(p['inputs_dir'])/f'cap{cap}')")
        return fixed.replace_once(text, "str(ROOT/'run_cell.py'), '--inputs'",
            "str(ROOT/'run_fixed_cell.py'), '--inputs'")

    path = ROOT/'run_group.py'
    text = fixed.replace_once(path.read_text(), '    text = parent.adapted_source()',
        '    text = fixed_group_source(parent.adapted_source())')
    text = fixed.replace_once(text, '        EXTRA_SOURCE_PATHS=[path,',
        '        EXTRA_SOURCE_PATHS=[*FIXED_EXTRA_SOURCES, path,')
    namespace = dict(__name__='fixed_tail_group', __file__=str(path),
        fixed_group_source=fixed_group_source,
        FIXED_EXTRA_SOURCES=[ROOT.parent/'completion_handoff/run_fixed_group.py'])
    exec(compile(text, str(path)+'[fixed1024]', 'exec'), namespace)
    return namespace['main']()


if __name__ == '__main__':
    raise SystemExit(main())
