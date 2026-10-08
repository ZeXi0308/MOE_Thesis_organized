# 实际准入机会与一次有界延后

本轮在既有负结果之后继续，最终目标仍是有真实因果证据的轻量准入原则、独立确认及英文投稿包；当前远未完成。原始恢复数量迟滞规则保留为历史失败机制，不重跑其六臂。

唯一当前问题：原生分配、完整在途上限和KV规则已经允许的新prefill，在真实恢复尚未结束时短暂延后，是否有完整服务收益？先排除计数遗漏A，再区分动作太晚B和进展时机C。当前不改变任务、生成上限、到达过程或GPU容量。

## 复用与必要修正

旧active统计running+remote，遗漏暂停/恢复等待。新观察器维护成功进入RUNNING/remote的唯一admitted集合，只有完成才离开；抢占不删，同一步新准入立即加入。CPU从旧两条恢复轨迹重建的首次计算后live峰值为221/220，尚未看到旧gate在256上限处漏记；这是有边界误差的诊断，不是准确动作时状态，不预报修正的收益。

旧10s规则是原型防饥饿豁免，没有应用SLO授权依据。首个诊断仍保留KV门控的原10s年龄豁免；新的延后仅约束控制器自己增加的阻挡窗口。二者分开记录。所有总等待仍从固定external arrival计时，原生安全容量永不豁免。

观察放在原生waiting allocate_slots之前，局部chunk、prefix/remote、lookahead、full-ISL及reservation已确定。仅在当前pinned单卡OLMoE full-attention配置下，复用原生只读块数试算；不能分配后释放假装试算。真实allocate结果与预测在线核对。恢复等待、远端在传、传输已完成待执行、已恢复执行分别记录，gate字段不使用步首快照代替。

## 固定运行与唯一动作

沿用BF16 OLMoE、64GiB usable GPU KV、16GiB host KV、4096上下文、256原生槽位、1024服务量子。384个既有唯一输入及0.1*i外部到达、自然EOS/1024上限不变，所有压力前缀请求计入结果。没有独立测试集主张。

对照采用先前有竞争力的KV256配置，补正确全生命周期计数，固定水位3277页。此处不重新扫参，不将它称为最优fixed，也不将旧running+remote cap当作完整在途cap。

先跑 `pro-opportunity-scan`：一个完整baseline臂，门控探针关闭，仅记录第一个合法机会及实际native分配结果。若存在合法机会，再启动单独整组 `pro-opportunity-probe`：baseline / delay / delay / baseline。两臂共同修正计数和观测，唯一区别是一次额外延后。

合法机会：从未开始的新请求，native-fit、完整cap和KV检查均允许，动作时仍有真实未完成恢复。baseline记录相同触发谓词的首个影子事件；delay仅第一次事件开启100ms窗口，250ms为硬门控截止上限。超过截止不再由探针拒绝，实际开始仍可能受原生步长或容量限制，不承诺总等待或完成时间有界。没有满足条件则记录NO_OPPORTUNITY，绝不强行触发。

首次hold后本步后续新请求也hold以保持FIFO；已有计算、恢复、remote请求一律继续原生流程。被hold的队列finally恢复。不能break整个waiting遍历而阻止旧请求恢复；native-fit=false仍保留原生allocation失败break语义。

## 测量与解释

每次shadow/action保存live逻辑状态与当前原生资源局部参数。相同输入重新执行，不做GPU快照；必须比较动作前请求ID、已开始集合、computed/output计数及资源状态，不宣称严格同状态分叉。后续batch/route/KV自然演化，不能离线拼接。

沿用全部到达口径、240s共同观察上限和完整排空；报告全部失败/拒绝/超时/未完成、TTFT/flow/maxgap/TPOT、请求/token吞吐和排空。没有应用SLO，原始分布为主、完整预声明20点面仅作探索。新旧请求分组及旧请求恢复进展不能替代全人口指标；输出量/结束原因同时报告。

首个scan没有机会时先从时序定位B，不直接换阈值。只有因果证据指向生命周期异质性不足时，才单独改变少量合法生成上限，各臂相同EOS与预算。若探针有效，再做去恢复信号的相同控制结构对照；收益不稳定或不抵消新请求等待，就保留负结果。一次失败不结束总论文目标，但也不据此扩张无界搜索。

## 资源与执行

唯一共享锁 `/root/autodl-tmp/moe-research-gpu.lock`；2026-10-07现场身份 `2304:15049831297`。新profile只打开既有文件，身份不符停止；整组串行、结束即释放。当前同一PRO6000 UUID已核对，其他会话持锁时仅CPU准备，至多一个C有界候锁进程。

首个v5扫描r01取得共同锁后，上一持锁进程尚在退出，安全检查在GPU初始化前终止；没有测量请求。记录保留于 `analysis/westb-opportunity-scan-r01-failed.raw-index.json`。v6仅增加15s有界GPU进程清空等待：仍持同一锁、只读轮询，任何计算进程存在时不初始化GPU，超时即失败。策略、输入和探针幅度均不变。

重试命令（必须新输出目录，实际执行状态见RUNSTATE）：

```sh
bash /root/moe-c-admission-20261007-v6/launch.sh /root/moe-c-admission-20261007-v6/opportunity-scan-r02 --profile pro-opportunity-scan --wait-lock-seconds 3600
# 仅scan确认机会后，另组取得同一锁：
bash /root/moe-c-admission-20261007-v6/launch.sh /root/moe-c-admission-20261007-v6/opportunity-probe-r01 --profile pro-opportunity-probe --wait-lock-seconds 3600
```

状态由RUNSTATE.json和实际进程共同核对，不能将准备写成已执行。
