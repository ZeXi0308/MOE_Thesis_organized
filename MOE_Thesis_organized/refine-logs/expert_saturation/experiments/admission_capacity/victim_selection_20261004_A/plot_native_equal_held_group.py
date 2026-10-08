"""Render the single-intervention probe with the existing all-request plots."""
import argparse
import json
from pathlib import Path
from plot_native_group import draw

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for path in draw(json.loads(args.analysis.read_text()), args.output, 'Equal-held-once'):
        print(path)
