# experiments/shared

跨 idea 公共库与通用仿真。idea 专用脚本在 `docs/ideas/*/experiments/` 或 `docs/99_archive/killed_ideas/*/scripts/`，通过文件头 bootstrap 自动把本目录加入 `sys.path`。

主要模块：`capture_moe.py`, `modeling.py`, `policies.py`, `fake_quant.py`, `metrics.py`, `prompts.py`, `paths.py`，以及 `run_ep_congestion_sim.py` / `run_tbt_*.py` 等。
