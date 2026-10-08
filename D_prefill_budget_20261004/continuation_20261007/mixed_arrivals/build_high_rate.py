"""One development pressure point: preserve all work, scale arrivals by 0.8."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    original = json.loads((ROOT / "workload_more_decode.json").read_text())
    rows = [dict(r, arrival_s=r["arrival_s"] * 0.8) for r in original]
    assert len(rows) == 256
    assert all({k: v for k, v in a.items() if k != "arrival_s"}
               == {k: v for k, v in b.items() if k != "arrival_s"}
               for a, b in zip(original, rows))
    (ROOT / "workload_high_rate.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print("256 requests; nominal arrival window 32s; last arrival", rows[-1]["arrival_s"])
    print("prompt", sum(len(r["prompt_token_ids"]) for r in rows),
          "fixed output", sum(r["max_tokens"] for r in rows))


if __name__ == "__main__":
    main()
