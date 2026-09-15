"""Extract existing selector reason counts; retain previous residual outputs."""
import json
from pathlib import Path

here = Path(__file__).resolve().parent
source = here.parent / "readback/results/diag-on/selective-store.json"
decisions = json.loads(source.read_text())["selector_decisions"]
absence = [d for d in decisions if d["reason"].startswith("absence ")]
cooldown = [d for d in decisions if d["reason"] == "swap cooldown"]
result = {
    "source": str(source),
    "explicit_absence_block_count": len(absence),
    "explicit_absence_step_min": min(d["step"] for d in absence),
    "explicit_absence_step_max": max(d["step"] for d in absence),
    "swap_cooldown_block_count": len(cooldown),
}
with (here / "gate_counts.json").open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print(json.dumps(result, indent=2))
