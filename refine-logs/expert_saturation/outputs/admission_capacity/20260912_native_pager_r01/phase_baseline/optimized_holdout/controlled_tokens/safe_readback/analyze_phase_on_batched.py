"""Phase-policy comparison with the same optimized pager in every arm."""
import argparse
import hashlib
import json
from pathlib import Path

from analyze_phase_prefill import evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args = parser.parse_args()
    config = json.loads((args.results/"config.json").read_text())
    result = evaluate(args.results,config)
    result["scope"] = config["holdout_scope"]
    for name,cell in result["cells"].items():
        if cell["status"] != "COMPLETED":
            continue
        raw = json.loads((args.results/name/"result.json").read_text())
        for key in ("map_initial","map_measurement","map_validation"):
            cell[key] = raw[key]
        if "token_control" in raw:
            control = raw["token_control"]
            expected = [control["token_id"]] * 8
            actual = raw["requests"][cell["measured"]["new_ids"][0]]["output_token_ids"]
            cell["controlled_output"] = dict(valid=actual == expected,
                expected_output_sha256=hashlib.sha256(json.dumps(expected).encode()).hexdigest(),
                scope=control["scope"], primer={k:v for k,v in raw["token_control_primer"].items()
                                               if k != "output_token_ids"})
            if actual != expected:
                cell["status"] = result["status"] = "INVALID_COMPARISON"
        if (cell["map_measurement"]["mode"] != "batched"
                or cell["map_validation"]["status"] != "PASS"
                or cell["map_measurement"]["pinned_map_bytes"] != 4096):
            cell["status"] = result["status"] = "INVALID_COMPARISON"
    with args.out.open("x") as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
        handle.write("\n")
