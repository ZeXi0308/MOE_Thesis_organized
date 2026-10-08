# westb-pressure-test-r02：优化观测版稳态六臂

**恢复信号的增量价值仍未成立。** 六臂各384次到达，合计2,304个请求全部完成；失败、拒绝、超时、未完成均为0。两个recovery运行实际改变的gate决定均为0。相对同cap的KV-only，recovery在正序块的20点joint goodput为0胜20负，反序块为19胜1负，平均flow由+0.682%翻为−0.596%。这些差异不能归因于没有实际执行的恢复追加门控。

本轮是冻结v3：来源`fc7f0a1`、远端`/root/moe-c-admission-20261007-v3/pro-pressure-test-r02`、GPU `94203fc3-1021-3a9c-a367-cff792479616`。`run.py` SHA256为`484275bc507dfd54f1abc9d10a56ef0038daa2d0d1cda4c85dadc7bf0e46e422`，`admission.py`为`c9dda825dade091c5673139f472980f927aa2d72c6fb9d4e52f80d2af37d2ae5`。PID22245已退出0；81个已提取文件逐一与归档核验字节数和SHA256，无重提取或覆盖。归档SHA256为`47f205b79ac1a520185213c9792aeadab5ca796f973260726d15f275b495f113`，见[raw-index](analysis/westb-pressure-test-r02.raw-index.json)。

每臂使用相同384请求人口、相同0.1 s到达轨迹，窗口38.3 s。fixed使用同版本dev-r03预选的cap128，KV和recovery使用cap256；**准入并发上限不是相同的**。native上限256、模型、GPU/host预算、victim规则、恢复执行和服务量子相同。block1执行fixed→KV→recovery，block2执行recovery→KV→fixed。开发与测试共用人口/轨迹，仍是探索性验证；不合并v2/旧GPU结果，不将请求或20个阈值点当作独立运行重复。

| 执行臂 | Admission cap | 完成/到达 | TTFT mean / p95 (s) | 完整flow mean / p95 (s) | 最大生成间隔 p95 / max (s) | Token/s | 排空 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00 fixed | 128 | 384/384 | 29.733 / 66.451 | 71.640 / 104.335 | 0.184 / 0.184 | 2717.69 | 102.041 |
| 01 KV | 256 | 384/384 | 16.905 / 45.131 | 73.844 / 95.604 | 0.193 / 16.581 | 2945.64 | 91.755 |
| 02 recovery | 256 | 384/384 | 16.993 / 45.472 | 74.348 / 96.370 | 0.195 / 17.489 | 2932.27 | 92.454 |
| 03 recovery | 256 | 384/384 | 16.660 / 44.882 | 73.418 / 95.434 | 0.193 / 17.798 | 2936.13 | 91.553 |
| 04 KV | 256 | 384/384 | 16.806 / 45.613 | 73.858 / 96.046 | 0.196 / 17.908 | 2923.00 | 92.315 |
| 05 fixed | 128 | 384/384 | 30.832 / 67.766 | 73.420 / 106.289 | 0.198 / 0.198 | 2687.29 | 103.890 |

所有TTFT/flow从外部到达起算。服务分母按执行序为140.341196、130.055510、130.753795、129.853036、130.615334、142.190200 s，包含到达窗口后的完整排空；表中排空从最后到达算起。完整分布与逐请求数据见[九臂主分析](analysis/westb-pressure-all-r02.json)。KV256相对fixed128降低TTFT p95约32%，但平均flow增加3.1%/0.6%，并引入16–18 s的最大生成停顿；不能只展示吞吐和TTFT改善。

六臂TTFT和完整flow的全人口CDF按两个顺序块分开展示，图中标明fixed128、KV256与recovery256：[SVG](analysis/westb-pressure-test-r02-cdf.svg)、[PNG](analysis/westb-pressure-test-r02-cdf.png)。每条曲线对应单次运行的384请求分布，不是运行级重复或置信区间。

全部20点预声明探索性联合SLO为TTFT 2/5/10/20/40 s × maxgap 0.25/0.5/1/2 s，flow≤120 s。以下每格按**fixed128 / KV256 / recovery256**显示联合达标请求数，请求分母各384；goodput以各自完整服务时间为分母。不选阈值作为事后应用SLO。

| Block1 TTFT | gap≤0.25 s | gap≤0.5 s | gap≤1 s | gap≤2 s |
|---|---:|---:|---:|---:|
| ≤2 s | 132 / 213 / 213 | 132 / 213 / 213 | 132 / 213 / 213 | 132 / 213 / 213 |
| ≤5 s | 132 / 218 / 218 | 132 / 218 / 218 | 132 / 218 / 218 | 132 / 218 / 218 |
| ≤10 s | 132 / 218 / 218 | 132 / 218 / 218 | 132 / 218 / 218 | 132 / 218 / 218 |
| ≤20 s | 136 / 220 / 219 | 136 / 220 / 219 | 136 / 221 / 220 | 136 / 221 / 220 |
| ≤40 s | 267 / 279 / 268 | 267 / 284 / 273 | 267 / 286 / 275 | 267 / 286 / 275 |

| Block2 TTFT | gap≤0.25 s | gap≤0.5 s | gap≤1 s | gap≤2 s |
|---|---:|---:|---:|---:|
| ≤2 s | 130 / 214 / 214 | 130 / 214 / 214 | 130 / 214 / 214 | 130 / 214 / 214 |
| ≤5 s | 130 / 220 / 219 | 130 / 220 / 219 | 130 / 220 / 220 | 130 / 220 / 220 |
| ≤10 s | 130 / 220 / 219 | 130 / 220 / 219 | 130 / 220 / 220 | 130 / 220 / 220 |
| ≤20 s | 132 / 221 / 220 | 132 / 222 / 220 | 132 / 222 / 221 | 132 / 222 / 223 |
| ≤40 s | 263 / 268 / 278 | 263 / 275 / 284 | 263 / 275 / 285 | 263 / 275 / 287 |

| 比较 | Block1 goodput胜/负/平 | Block2 goodput胜/负/平 | Block1达标率胜/负/平 | Block2达标率胜/负/平 |
|---|---:|---:|---:|---:|
| KV 对 fixed | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 |
| recovery 对 fixed | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 | 20 / 0 / 0 |
| recovery 对 KV | 0 / 20 / 0 | 19 / 1 / 0 | 0 / 8 / 12 | 5 / 7 / 8 |

相对fixed128的20点优势包含更高准入上限和KV门控的联合变化，不能算恢复状态信号贡献。关键同cap对照rec/KV的TTFT p95差异为+0.757%/−1.601%，flow p95为+0.801%/−0.637%，吞吐为−0.454%/+0.449%，方向随块翻转。反序块goodput多数点胜出也不等于达标率全面提高，两种口径分别保留。

| 执行臂 | 总输出token | 自然stop / length | Preempt | 步首最大恢复集合 | cap / KV重复defer | controller wall / 服务占比 | controller CPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00 fixed | 381,404 | 13 / 371 | 0 | 0 | 191,898 / 0 | 1.420 s / 1.012% | 1.459 s |
| 01 KV | 383,097 | 11 / 373 | 42 | 9 | 0 / 7,784 | 0.264 s / 0.203% | 0.263 s |
| 02 recovery | 383,406 | 10 / 374 | 48 | 9 | 0 / 7,689 | 0.266 s / 0.203% | 0.264 s |
| 03 recovery | 381,265 | 13 / 371 | 40 | 10 | 0 / 7,564 | 0.266 s / 0.205% | 0.265 s |
| 04 KV | 381,788 | 13 / 371 | 39 | 10 | 0 / 7,630 | 0.268 s / 0.205% | 0.266 s |
| 05 fixed | 382,107 | 12 / 372 | 0 | 0 | 196,470 / 0 | 1.589 s / 1.118% | 1.633 s |

natural EOS开启、min_tokens=0、max output=1024。rec对KV的总输出变化为+309（+0.0807%）/−523（−0.1370%），输出序列不同153/384、105/384；KV对fixed变化为+1,693（+0.4439%）/−319（−0.0835%），序列不同217/384、210/384。同策略跨块fixed/KV/rec的序列也分别有205/138/113个请求不同。没有任务质量结论，不把自然输出差异当等工作提速。全部输出差异和比较见[comparison](analysis/westb-pressure-test-r02-comparison.json)。控制开销已包含在服务计时中；重复defer不是拒绝请求，也未暂停到达源。

**恢复状态口径：** `Gate.begin`在调度步开始刷新`recovery_since`；`Gate.defer`记录的`recovery_count`只是该步首集合大小，并不在gate动作瞬间重新扫描。latch按约100 ms采样更新；free和active则在gate评估时读取。不能把这些字段视为完全同时采样的连续状态，或把latch=true等同于动作瞬间有非零真实积压。未被原生等待循环评估到的队尾请求也不在gate日志中。

| 臂 | 首次非零步首backlog snapshot (s) | 首次latch snapshot (s) | latch gate评估/唯一请求 | latch且age<10、KV/cap允许的评估/请求 | 恢复增量改变 |
|---|---:|---:|---:|---:|---:|
| 00 fixed | 无 | 无 | 0 / 0 | 0 / 0 | 0 |
| 01 KV | 33.879 | 34.398 | 5 / 5 | 0 / 0 | 0 |
| 02 recovery | 33.696 | 34.348 | 8 / 6 | 0 / 0 | 0 |
| 03 recovery | 34.168 | 34.429 | 0 / 0 | 0 / 0 | 0 |
| 04 KV | 34.331 | 35.243 | 3 / 2 | 0 / 0 | 0 |
| 05 fixed | 无 | 无 | 0 / 0 | 0 / 0 | 0 |

时刻相对外部到达原点，仅为首次观测边界，不能推断连续起止或漏观测时长。rec02的8条latch=true记录如下；每条均`bypass=true`、`kv_only_denied=false`、`denied=false`：

| 时间 (s) | age (s) | free | active | 记录的步首recovery_count |
|---|---:|---:|---:|---:|
| 60.691076 | 35.491 | 204 | 199 | 2 |
| 60.829741 | 35.530 | 161 | 199 | 1 |
| 61.699827 | 36.200 | 121 | 199 | 2 |
| 61.770163 | 36.170 | 35 | 200 | 2 |
| 61.836516 | 36.237 | 18 | 200 | 0 |
| 62.981270 | 37.081 | 180 | 199 | 2 |
| 63.119724 | 37.120 | 10 | 200 | 1 |
| 63.187671 | 37.188 | 2 | 200 | 0 |

全部年龄超过10 s，因此KV与recovery gate都实际豁免；free低于3277并不表示本次KV gate也阻挡。两个步首count=0而latch=true的记录说明采样清除滞后。rec03没有latch=true的新请求gate评估。所有请求在48.3 s达到绝对年龄10 s，但恢复在到达窗口内已经出现，不能把零机会解释为“恢复出现得太晚”。零机会只覆盖实际经过gate的记录，不能保证未执行的native full-ISL allocation会成功。完整事件、首次许可和等待见[decision summary](analysis/westb-pressure-test-r02-decisions.json)。FIFO首次许可顺序相同，实际许可时间和多数批次不同。

同版本三档开发固定参照保留在九臂主分析和comparison中，不只与selected128比较：

| 测试臂 | 对dev-r03 cap128 goodput胜/负/平 | 对cap192 | 对cap256 |
|---|---:|---:|---:|
| 01 KV | 20 / 0 / 0 | 20 / 0 / 0 | 18 / 2 / 0 |
| 02 recovery | 20 / 0 / 0 | 17 / 3 / 0 | 13 / 7 / 0 |
| 03 recovery | 20 / 0 / 0 | 20 / 0 / 0 | 18 / 2 / 0 |
| 04 KV | 20 / 0 / 0 | 17 / 3 / 0 | 18 / 2 / 0 |

这些是同版本开发参照，非反序配对/独立确认，不能替代同cap的rec/KV结果。原始launch log第61行记录首测量臂`test-00-fixed`在18:04:48发生`fused_moe_kernel`推理期JIT警告；没有可隔离的时长，故仅记录存在性，不推测延迟贡献、不减去成本。后续固定臂也单独保留，没有用它覆盖首臂。

结论范围为**NATIVE_SERVING、v3稳态探索**：优化后KV256在固定20点上稳定优于mean-flow选中的fixed128，但有平均flow/长停顿等代价，且比较包含并发上限变化；恢复追加门控仍零动作、相对KV结果翻转，不支持恢复感知机制收益。该门控的当前最小实现可由KV-only替代；若继续研究恢复信息，须先证明存在未超龄且可执行的新请求决策窗口。突发实验、独立测试集和任务质量不属于本报告，不用尚未完成的结果提前扩展结论。

复算命令（本目录执行；已有输出禁止覆盖，复算需新文件名）：

```sh
python3 -B analyze.py runs/westb-20261007/pro-pressure-dev-r03/dev-cap{128,192,256} runs/westb-20261007/pro-pressure-test-r02/test-{00-fixed,01-kv,02-recovery,03-recovery,04-kv,05-fixed} --output analysis/westb-pressure-all-r02.json
python3 -B decision_summary.py runs/westb-20261007/pro-pressure-test-r02/test-{00-fixed,01-kv,02-recovery,03-recovery,04-kv,05-fixed} --output analysis/westb-pressure-test-r02-decisions.json
python3 -B compare.py analysis/westb-pressure-all-r02.json --output analysis/westb-pressure-test-r02-comparison.json
```

绘图复现命令（在本工作区根目录运行，文件存在时会拒绝覆盖）：既有脚本只在本次渲染的内存中按实际配置补上cap标签，没有修改数据或脚本。

```sh
python3 - <<'PY'
import importlib.util, json, sys
from pathlib import Path
root = Path('research/C_admission_20261004')
analysis = root/'analysis/westb-pressure-all-r02.json'
source = json.loads(analysis.read_text())
caps = {mode: {c['admission']['configuration']['cap'] for c in source['cells']
    if Path(c['cell']).name.startswith('test-')
    and c['admission']['configuration']['mode'] == mode}
    for mode in ('fixed', 'kv', 'recovery')}
assert caps == {'fixed': {128}, 'kv': {256}, 'recovery': {256}}, caps
spec = importlib.util.spec_from_file_location('c_comparison_plot', root/'plot_comparison.py')
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)
names = {'fixed': 'Fixed', 'kv': 'KV', 'recovery': 'Recovery'}
plot.POLICIES = tuple((mode, f'{names[mode]} (cap {next(iter(caps[mode]))})', color, style)
    for mode, label, color, style in plot.POLICIES)
sys.argv = ['plot_comparison.py', '--analysis', str(analysis),
    '--output', str(root/'analysis/westb-pressure-test-r02-cdf.svg'), '--png']
plot.main()
PY
```
