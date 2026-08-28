from __future__ import annotations

"""Merge per-model/per-layer curve CSVs while rejecting duplicate coordinates."""

import argparse
import csv
from pathlib import Path

try:
    from .benchmark_expert_service_curve import CURVE_COLUMNS
    from .core import ProtocolError, sha256_file, write_json
except ImportError:
    from benchmark_expert_service_curve import CURVE_COLUMNS
    from core import ProtocolError, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    coordinates = set()
    for raw_path in args.input:
        with Path(raw_path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != CURVE_COLUMNS:
                raise ProtocolError(f"{raw_path}: unexpected service-curve schema")
            for row in reader:
                key = (row["model"], int(row["layer"]), int(row["rows"]))
                if key in coordinates:
                    raise ProtocolError(f"duplicate service-curve coordinate {key}")
                coordinates.add(key)
                rows.append(row)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CURVE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["model"], int(row["layer"]), int(row["rows"]))))
    write_json(
        target.with_suffix(".meta.json"),
        {
            "schema": "bcrd-service-curve-merge-v1",
            "inputs": {path: sha256_file(path) for path in args.input},
            "rows": len(rows),
            "output_sha256": sha256_file(target),
        },
    )
    print(f"merged {len(rows)} curve points into {target}")


if __name__ == "__main__":
    main()
