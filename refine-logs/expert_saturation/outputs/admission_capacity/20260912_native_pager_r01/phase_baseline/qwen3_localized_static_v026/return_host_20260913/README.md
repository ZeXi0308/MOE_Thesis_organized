原 r01 的实测状态为 **INTERRUPTED_DURING_LOADING_NO_TERMINAL_RECEIPT**。返回端点为 `connect.weste.seetacloud.com:23478`。[inspection01.json](inspection01.json) 确认原 worker 89182 与原 stage producer 均已不存在；[reconciliation.json](reconciliation.json) 记录 8 个完整分片完成全 SHA 校验并消费、9,889 个源 tensor 已消费，最后事件是第 9 片 `DOWNLOAD`。数值资格化结果为 0，性能格为 0；退出码与中断原因未知，原 `launch.json` 的 `RUNNING` 状态保持原样。

[readback_manifest.json](readback_manifest.json) 记录生产进程不存在、归档前后文件稳定的 48 文件回读，归档 SHA256 为 `f401766874b1e5888b105742c713b4b0c40c95ca3389d0e3fe199eb406870474`。原件保留于 [readback_original_r01/](readback_original_r01/)，此次中断不构成科学 NO-GO。

[environment01.json](environment01.json) 核对 Python 3.12.3、Torch 2.11.0+cu130、vLLM 0.26.0，已安装关键源码、Torch memory 与 WiSP SHA 匹配。主机为 Xeon Gold 6459C，CPU 0–7 可用；GPU UUID 为 `GPU-0a66cc34-b091-2000-ba7b-e576b6d3d7d6`，显存 32,607 MiB，driver 595.71.05，cgroup 92 GiB。该环境采样曾记录其他 GPU worker；启动时仍须通过父脚本的实际占用检查。

[attempt02/](attempt02/) 当前为 **CPU_PREPARED_UNRUN**，未在本文宣称上传或启动。[protocol.json](attempt02/protocol.json) 与 [revision.json](attempt02/revision.json) 固定整体 21,600 秒、每 episode 600 秒、原 92 GiB/84 GiB 内存边界；17 个 runtime 源码和模型、输入、静态 ABBA 参数保持不变。临时分片使用独占新建的系统盘目录 `/root/qwen3-localized-static-v026-r02-shard-workspace`；结果与 loader receipt 仍在 `/root/autodl-tmp/qwen3-localized-static-v026-r02`。两盘分别要求可用空间大于 6 GiB 并留痕。只改变临时分片介质，全部加载与 IO 成本继续计入；未加入 cache collector，未删除旧第 9 片或其他数据。

现有 [analyze_qwen_localized_static.py](../analyze_qwen_localized_static.py) 接受显式 `--launch-dir`，不要求 workspace 位于结果目录内。以下命令仅供 r02 完成并按 `attempt02/readback_r01/{results,launch}` 布局回读后使用；在本目录执行，当前尚未运行：

```bash
python3 ../analyze_qwen_localized_static.py --input-dir attempt02/readback_r01/results/comparison --launch-dir attempt02/readback_r01/launch --out attempt02/summary.json
```

原 r01 缺少数值及性能结果，不对其运行完整性能分析器。当前结论只关闭旧任务状态核对与新启动的 CPU 准备，科学结果仍待实际运行。


## 启动后的实测更新

[attempt02/start_verified.json](attempt02/start_verified.json) 确认包7f54419f…已上传，monitor13262/parent13263/worker13275存活；三次GPU空闲检查通过。状态RUNNING_LOADING，第1/16片下载，0数值/性能结果。上述CPU_PREPARED_UNRUN保留为启动前准备状态；最新现场读数在[watch_attempt01.jsonl](attempt02/watch_attempt01.jsonl)。
