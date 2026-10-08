# 固定迟来 burst：准入动作窗口诊断

状态（2026-10-07 更新）：六臂全部COMPLETE；PID25837/工具会话21411已退出0并释放共同整卡锁。两遍恢复信号新增动作均0，原始结果及完整分析、图表均已保存，见 [结果报告](WESTB_PRESSURE_BURST_RESULT.md)。本profile根据已见稳态恢复时间设计，仍是探索诊断，不是独立测试集或自然生产获胜场景。

唯一负载因素是外部到达序列。保持 `pressure_population` 的 384 条唯一文章、dev/test 交替顺序、原 request IDs、完整 prompt tokens、模型/BF16/4096 上下文、自然 EOS/1024 输出上限、64 GiB usable GPU KV、16 GiB host KV 和 native 256 槽位。固定策略顺序为 `fixed, kv, recovery, recovery, kv, fixed`；固定臂沿用已完成 steady 开发的选中 cap，KV/recovery 上限仍 256，KV 水位 3277 块、恢复阈值 2/清零 0、100 ms 采样和 10 s 等待豁免均不变。没有修改 `admission.py`。

对原顺序的零基索引 i，六臂共享同一预先确定的到达数组：

- `0 <= i < 256`：`arrival_s[i] = 0.1*i`，首段从 0.0 到 25.5 s。
- `256 <= i < 384`：`arrival_s[i] = 55.0`，最后 128 条同刻到达。

这个数组不依赖任一臂的运行状态、未来 EOS 或真实输出长度。25.5–55.0 s 的空档是公开的负载定义；不是暂停已有到达源或把已到达请求等待藏在引擎外。仍由现有 `measure_episode` 在到期时提交请求，计时从固定 external arrival 开始，客户端提交 lag、scheduler waiting 都计入 TTFT/完成时间；观察及超时口径仍覆盖全部 384 条，不能只统计已接纳者。

只读检查确认 `request_measurement.py:84–111` 在还有未来请求时不会因当前引擎排空而退出；重复的 55.0 s 到达被依次提交后才执行下一 `engine.step()`，原始 external arrival 仍传给每条请求。因此逻辑同刻到达不等于提交调用同时完成，实际提交 lag 保留并计入结果。

动机：既有 steady 运行中恢复约在 28–35 s 开始，但实际新 gate 在恢复期间少见，见到时常已超过 10 s，且 KV 已低于水位。55 s 的迟来请求旨在检验是否能出现年龄尚未豁免的实际 gate；这不是保证恢复与新请求重叠，也不是服务收益预测。

## 开发 selection 与 burst 协议

`--profile pro-pressure-burst-test` 必须显式提供一个已完成 `pro-pressure-dev` 的 `selection.json`，并保留同目录 `status.json`。验证输入文件 hash、384 人口、开发 caps/scores 为 128/192/256、选中 cap、原 steady 0.1 s trace/workload hash、COMPLETE 状态及其 selected cap。一律读取原 selection，不修改它，不在 burst 上重新选优。steady 最佳 cap 迁移到 burst，不声称它是 burst 最优固定 cap。

协议分别记录实际 burst trace/workload、`selection_development` 的原 steady trace/workload、`selection_source` 原文件 hash/内容，以及 `selection_completion_source`。没有把两种轨迹写成相同开发 trace。正常 `pro-pressure-dev` / `pro-pressure-test` 仍保留原 0.1 s 序列和原 hash：

| 对象 | SHA256 |
|---|---|
| `pro_inputs.json` | `6b6be96f8fec6e92b6e59390bab0fe414aeafdaab9539000d88050f8bbf22d25` |
| steady trace | `34513450387479b4257f7a01ea9042056667466b4ade5e0c137aa1f0fadb8632` |
| steady workload | `7723a0b4e88ba647da79e658cd596e00337075ea761d8692c47c494ee018533d` |
| burst trace | `4abe6838ec23fc6ae32f8bd44e0e00f5881c716e6fc4618762ddc5d73faa87d3` |
| burst workload | `2a8daed05751d2d2a682b4814e88d6659fb5ad0c3f0fe0a1d4c089d4819f724d` |

CPU 已核对 384 个 arrival、原 ID/token/顺序完全相同、首尾时刻、steady 对象未被修改、旧 pressure_population/subset 输出不变；用已完成 westb `pro-pressure-dev-r02` selection 验证通过。对输入/profile/population/caps/scores/选中 cap/gap/两个 hash/完成状态的 11 个错误或不匹配例子均拒绝。`run.py --help` 与 `git diff --check` 通过；没有运行 GPU。

实际执行命令（该目录已完成，重复实验须换新输出目录）：

```sh
bash /root/moe-c-admission-20261007-v4/launch.sh /root/moe-c-admission-20261007-v4/pro-pressure-burst-test-r01 \
  --profile pro-pressure-burst-test \
  --selection /root/moe-c-admission-20261007-v3/pro-pressure-dev-r03/selection.json \
  --wait-lock-seconds 3600
```

selection 来自已 COMPLETE 的上述 v3/dev-r03，选中cap128。六臂已全部完成，具体状态见 RUNSTATE.json。复用主机共同排他锁，整组六臂串行；未改写冻结v3目录。

## 解释边界

只在此完全相同 burst trace 内比较三策略及反序重复，报告所有到达请求的原始 TTFT/完成时间/最大生成间隔、完整联合 SLO 面、吞吐、排空、失败/超时/未完成和自然输出量。不要把 steady 与 burst 的延迟差当成因果提速。固定 baseline 是明确的 steady-cap 迁移；若出现收益，后续仍需检验有竞争力的 burst 固定基线与独立负载。

先检查实际 gate 的 `changed_by_recovery`、唯一请求数，以及 KV/年龄/cap 条件。若没有恢复独有动作，承认此动作窗口诊断仍未建立增量决策价值；若有动作但完整结果无增益，也如实保留负结果。不能以恢复与新请求在时间上重叠，或仅 decode 停顿改善，替代完整服务收益。

本组冻结payload SHA256为 `ffa769938d3e2545ef5f17cf268948cba401ee07aee3bbb9a55c1b3862e6e702`；run.py为`f88d1264a5003054d9150a3c22849cf3aaf3da92c317b43253c047a86279da09`，admission.py为`c9dda825dade091c5673139f472980f927aa2d72c6fb9d4e52f80d2af37d2ae5`。完整原始包为`runs/westb-pressure-burst-test-r01.tar.gz`，SHA256 `27cfe1fc28081f565882636f8f8b62ca15485c65c56feb04ca3ea4ad582b7466`。远端和本地raw均保留，未覆盖旧包。
