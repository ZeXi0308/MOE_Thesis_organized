#!/usr/bin/env python3
"""Run one frozen cell; archive its data locally before invoking the next cell."""
import argparse
from run_campaign import CELL_BY_LABEL, run_cell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cell', choices=list(CELL_BY_LABEL))
    args = parser.parse_args()
    return run_cell(args.cell)


if __name__ == '__main__':
    raise SystemExit(main())
