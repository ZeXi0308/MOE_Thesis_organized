# G64 剩余三点固定网格：CPU 准备，GPU 未执行

首块 `native_full_native / eager / ltr_t30_q10` 仍须先完成原始归档与独立审计。本文件只准备后续 `ltr_t30_q1 / ltr_t200_q1 / ltr_t200_q10`，不表示它们已经运行或首块有效。四个 LTR 点全部完成且通过逐格资源/请求审计之前，不能选择 G64 开发点，不能把 H128 称为盲测或确认。

## 已冻结执行身份

- 新计划：[G64_PERF_REMAINING_GRID_PLAN_R02_20260930.json](G64_PERF_REMAINING_GRID_PLAN_R02_20260930.json)，SHA-256 `01d6164d1561968de356ef65fec1143967dd23b13f2e4940e2da8076bd9e1b93`。新 session 与三个新 output 均以 `r02-20260930` 命名，不复用首块的任何目录。
- 继续使用已上传的 `candidate_g64_perf_r01`，manifest SHA-256 `2755945122e3c346ca7e7e5ceda0d4668a8e1d3c90ba5c8d9d7b6ab3e89e0eb9`。其 `run.sh` 已冻结这三个点的 T/Q 参数，并在包根目录为每个 arm 建立一次性 `launch-once-*` 标记；新包与新策略源码均不需要。
- 继续使用 [serial_group_g64_perf_existing_model.py](serial_group_g64_perf_existing_model.py)，SHA-256 `b899cad3c36789ed7eaffdeac052e93bd20d0b37434a9001c0563e8587115073`。新计划只传三个未执行 arm。控制器在共同锁下核 C 来源与 A 私有模型、离线解析、GPU UUID/空闲、每格墙钟、前后 GPU 进程和归档逐文件读回；任何失败停止组内后续格。
- 新组自限总墙钟 4200 s，每格最多 900 s，主机上限沿用 90 GiB cgroup（`96636764160` 字节），单 RTX 5090。这个限制独立于首块，没有自动重试。

## 首块提供的必需输入

1. 完整复制并逐文件校验的首块 session `moe-a-g64-perf-session-r01-20260930`。它的 `receipt.json` 必须是 `CELLS_COMPLETE`，三格 exit 0、非超时、归档 `VERIFIED`、每格结束 GPU 计算进程为空，模型收据完整。
2. 使用 [audit_g64_perf_three_arm.py](audit_g64_perf_three_arm.py) 从该 session 重算的 **新**审计 JSON 及其独立 SHA-256。审计状态必须为 `PILOT_THREE_ARM_COMPLETE_PARTIAL_TQ_GRID`，64/64 请求各格完成，`calibration_status=NOT_SELECTABLE_PARTIAL_GRID`。即使 T30/Q10 未达到 eager 的 97% 输出率与 105% 平均完成门，也仍须按冻结计划测完其他三个点；首块无效则停止。
3. 执行者复核首块的实际输出长度、EOS/上限分布、来源和资源限制。完成者真实 flow、各臂实际输出率、每请求生成间隔及固定 goodput 前沿均保留；不要因观察首块后改变四点网格。

## 只读启动门

[gate_g64_perf_remaining_grid_r02.py](gate_g64_perf_remaining_grid_r02.py) 是 CPU-only。它核首块审计 SHA、收据 SHA、三个 archive 的每个文件 SHA、64/64、相同机器/模型/运行时/锁/host 合同，以及新计划与首块的路径/arm 不相交。`--recompute-pilot-audit` 在本地从原归档重算完整审计。远端上传同一审计 JSON 后再使用 `--remote-state-check`，额外核首块三枚已消耗标记、剩余三枚不存在、新 session/output 不存在及共同锁 inode。脚本只打印 `READY_CPU_ONLY`；它不会保留 GPU 锁，随后控制器仍须非阻塞抢锁及重新查 GPU。

本地命令模板（先填入首块审计 JSON 的实际路径与 SHA）：

```sh
python3 -B gate_g64_perf_remaining_grid_r02.py \
  --pilot-session /ABS/LOCAL/moe-a-g64-perf-session-r01-20260930 \
  --pilot-audit /ABS/LOCAL/G64_PILOT_AUDIT.json \
  --expected-pilot-audit-sha256 AUDIT_SHA256 \
  --remaining-plan G64_PERF_REMAINING_GRID_PLAN_R02_20260930.json \
  --expected-remaining-plan-sha256 01d6164d1561968de356ef65fec1143967dd23b13f2e4940e2da8076bd9e1b93 \
  --controller serial_group_g64_perf_existing_model.py \
  --package candidate_g64_perf_r01 \
  --recompute-pilot-audit
```

远端只读门使用相同参数，把 plan、controller、package、pilot session、audit 换成远端绝对路径，并以 `--remote-state-check` 替换 `--recompute-pilot-audit`。它通过后，执行者单独核上传后的 gate/plan/audit 字节，现场检查 GPU 与共同协调记录，再以原 `serial_group_g64_perf_existing_model.py` 和新 plan SHA 启动一次组运行。运行时不得复用已存在的 `r02` session/output；若门或组失败，保留原始证据并另订新身份，不在原目录补跑。

三格回读后，应再做与首块同等级的逐格原始审计，并将四个 LTR 点在同一 G64 cohort 与 eager 参照下比较：仅请求完整、实际输出率 ≥ eager 97%、平均已完成 flow ≤ eager 105% 的点合格；合格点按全请求最大生成返回间隔最小，平手依次按输出率、平均完成、低 T、低 Q。无合格点记 `NO_QUALIFYING_POINT`。`native_full_native` 单列完整系统参照；保留输出/EOS 差异和 peer 恶化，不把单次开发网格写成统计或生产 SLO 结论。

CPU 单元验证：`PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_gate_g64_perf_remaining_grid_r02`，5 项通过。它不执行 vLLM 或 GPU。
