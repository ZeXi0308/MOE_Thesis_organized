# M0 correctness boundary and hard-cap packing component challenge

## Verdict

保留 M0 为受限原生实验基线；本轮没有证据支持把其容量公式单独命名为新的调度算法。同 FIFO、相同合法域、相同保守 token 边界、**新/partial-prefill 请求始终保留 full-bound 矩形**时，独立逐轮 hard-cap global packing 与 M0 动作相同。这是组件挑战，既不是 CacheOPT 官方复现，也不证明 CacheOPT 原文 pairwise embedding 或 WAIT 与 M0 全部等价。允许新候选自身在未来退休的 general packing 是不同 profile，可能更紧，但须另证 prefill/decode 兑现条件。

唯一新实验为本机 CPU 检查；没有连接远程、运行 GPU、修改原实现或原始证据。前置域与失败条件见 `PROTOCOL.md`。证据上限 STRUCTURAL，不是性能、完整 runtime correctness 或质量证据。

## Source reality

初读 worktree `/private/tmp/moe-research-c-20260929-v2`，HEAD `f5b27506fc6aa1612e81b20abca2bddc5497b47b`，未跟踪 fresh 分析 JSON/PNG/SVG。中途观察共享树为 HEAD `93fb7725bf3a7f642dacf31e6e5234e648a509e2` 并有三个未跟踪 temporal-cap-donor 脚本；最终只读快照（2026-10-01 09:17:45 UTC）已为 `df508795ef7f7287e90cc1d7cb3130bbd0e21491`，工作树clean。初始HEAD读取早于协议文件创建时间09:09:25 UTC，但起始精确时钟未捕获，不伪造时间戳。并发版本变化不归因为本轮；本审查未改源代码，也没有审查这些并发新增脚本。初、中、末快照及固定source哈希见 `source_provenance.json`。

实际审查目录 `refine-logs/expert_saturation/experiments/admission_capacity/20260929_c_baseline_delivery`：

- `C_NATIVE_RETIREMENT_ADMISSION_V1.py`: SHA256 `261b8f2697042f75130203019e8722c424cca3e71d82ac309c78f29ec1bc8eb6`。
- `C_NATIVE_MAX_BOUND_ADMISSION.py`: `b6a601d1edc05804bee70d7f6ecdb2f00e1dd017b2280163df2db88250fb50ee`。
- `C_NATIVE_RETIREMENT_FRESH_CELL_V1.py`: `90c0065e8f004f98c636c1453caacaf678033144b96e9e74a708943e8b13703b`。

`/private/tmp/moe-c-native-source-20261001` 中 scheduler、KV manager、worker 哈希分别为 `2ed2a550…`、`3f4af8d2…`、`81b7627f…`，逐项吻合 fresh 两格运行资格回执中的 pinned native 源码。不能用 8 月主目录状态或早期 checkpoint 代替此版本。

## Capacity, execution and native ownership

设 B=16，纯 decoder 有 P prompt tokens、O 已输出 tokens、M 硬 cap、C=P+O−1 已计算 tokens，R=M−O>0。M0 定义

`E_D = max_{t in {0} union {R_i}} Σ_i ceil((P_i+O_i+t)/B) · 1[t<=R_i]`。

退休之间存活集合固定且各项单调不减，因此最大值在 0 或某个退休端点；必须在端点求和时保留 `t=R_i` 的该请求，不能提前扣掉最后一轮分配。新请求及仍在 prefill 的已有承诺使用 `b_j=ceil((P_j+M_j)/B)`。实际 M0 只在**全部 resident 都为纯 decoder**时刷新证书/准入，否则禁止新增并保留旧 protected 集。

注意上式是保守边界，不是 native 精确物理峰值。同步 non-spec 下第 k 次后续 forward（1<=k<=R）分配到 `C+k=P+O+k−1`，生成第 M 个输出后 stop/free，所以最后真实已计算 token 是 `P+M−1`。M0 端点用 `P+M`，多留一个 token；某些 block 边界上多一块。例如 P=15,O=1,M=2 时最后分配16 tokens=1块，M0端点17 tokens=2块。该差别不会使 M0 低估；不能把它写成 exact physical peak。

当前 native 账目：pool 4097 总块含1个 null block，资格合同已得4096 usable（8,592,031,744总KV bytes）；不可再次减 null。无 prefix cache、KV connector/offload、spec/lookahead、watermark、deferred-free；`reserved_blocks` 只在 async KV load 分支由 `_inflight_prefill_reserved_blocks()` 填充，此限定域为0。`scheduler_reserve_full_isl=true` 是 full input/current-sequence admission feasibility check，不是额外物理分配，更不是额外 hard-cap 常驻 reservation。适配器的 `active` full bounds 是逻辑上界，不能再从物理 free blocks 中扣第二次。

原生源码接口：`Scheduler.schedule()` running loop（约469–620）先调度旧 resident；新请求只在后续 waiting loop 添加；`KVCacheManager.allocate_slots()` 在 computed+new_tokens 上分配；`_update_request_with_output()` append token 后 check_stop；`update_from_output()` stopped 分支调用 `_free_request()`→`_free_blocks()`→`kv_cache_manager.free()`。适配器只包装 `s.schedule` / `s._free_request`，临时改 `s.max_num_running_reqs` 以准入所选 FIFO 前缀，native 仍拥有分配、采样和释放。

可兑现域：FCFS、32槽、1024 token预算、同步每请求每步最多1输出、PP=DP=1，实际资格为单卡/full-attention；protected decoder 是 running 前缀且 eligible。旧 decoder最多32个，前缀先占1 token各一个，新prefill无法抢走该额度。源码 post-schedule 核对每个protected恰好schedule 1，下一次pre-schedule核对输出增1或已经真实free。EOS提前释放只降低未来占用。metadata/phase/queue检查、无抢占检查均必须保留。

适配器本身没有完整独立验证所有“single GPU”配置字段；例如没有直接断言 tensor_parallel_size=1，单卡边界由运行资格/engine配置共同成立，不能把构造器错误信息当普适保证。

## Failure handling and minimal counterexamples

当前 `MaxBoundAdmission.fail()` 只追加 violation 并 raise RuntimeError。安装 context 退出只恢复 monkeypatch；cell 异常保存 INCOMPLETE，finally shutdown engine。**没有 freeze-new-admission + obligation-preserving safe drain**。正常 EOS/length 才能通过包装的 free；取消产生 FINISHED_ABORTED 即使 native 已释放也会触发违反，因此取消属于未支持边界。

借用未来容量之后，切回 full-bound 不会偿还已有承诺。可实现的未来 DRAIN 模式必须：停止风险准入；保留原 protected 集和服务顺序，直到其 cap/EOS/free；每步旧义务加最小prefill额度<=token budget；保留 allocator/native 检查；若旧义务本身无法兑现，不能宣称安全drain，应中止并把受影响请求计失败。该模式本轮 **UNIMPLEMENTED/UNRUN**。

| 最小反例 | 击穿的错误保证 |
|---|---|
| 1个protected decoder + 1个需prefill的resident，token budget=1，同时要求各至少1 token | memory bound可行不代表progress可行；两义务需要2 tokens |
| B=16、容量4块：A(P17,O1,M2)、B(P1,O1,M33)有E=3块，可再准入1块请求；A一直被排除，B继续到3块时A仍占2块 | 原先E成立依赖A按轮退休，冻结/重新分类丢失旧承诺后，即使停止准入也可需5>4块 |
| 一个长prefill插入protected前且吃完预算 | FIFO队列顺序不自动保证decoder服务，必须保持running prefix或显式预算 |
| M2容量6，旧占用4在轮2释放，队头H需5；后继B需2可立即fit且活到轮3 | 普通first-fit会把H原轮2预约推迟：H+B=7>6；aging只调优先级不能保证预约 |
| M2剩余KV足够，但后继prefill先耗尽预约轮token预算 | memory预约不等于head prefill服务预约；轮次预约又不等于墙钟TTFT保证 |

前两组是结构反例，非新增GPU失败事件。early EOS和合法native取消可使真实内存更少，但取消回调兼容性仍是当前实现缺口。

## CPU and existing-log results

| 比较 | 状态数 | 峰值差异 | 准入差异 |
|---|---:|---:|---:|
| 0–2 decoder ordered小状态穷举 | 2,971 | 0 | 0 |
| 固定seed随机合法状态 | 1,000 | 0 | 0 |
| fresh retirement_1 all-decode | 5,998 | 0 | 0 |
| fresh retirement_2 all-decode | 5,992 | 0 | 0 |

生成状态调用实际 M0 `schedule()`，native部分为CPU stub，故不验证native allocator；fresh部分对照实际存下的peak/selected，FIFO相同所以前缀长度相同即前缀相同。全部12,345 fresh调用都成功恢复resident/queue，零对齐失败。逐状态数据在 `state_comparison.jsonl`，结果在 `result.json`。保守M0高于精确computed-token模型：生成状态1,643次（最多7块），两个fresh分别3,804/4,059次（最多4/5块）；不等于可以直接更紧后获得性能收益。

M1现有日志已经足以做最便宜的结构判断：mixed且waiting为155/156调用；D-envelope+P-fullbound+head fit只有77/80调用，涉及49/52个head。这些调用D均仍为running prefix，旧protected未丢，32槽与1024预算可容纳D及至少1个prefill token。然而这些可fit状态总调度间隔仅1.717/1.769秒，占有等待的调度时间2.38%/2.47%；距下一实际all-decode状态中位22.96/22.55ms、最大77.71/67.57ms。不能把49/52个head当独立重复，也不能据此说M1能省相同时间；改变准入后后续state会改变。当前证据不支持优先实现M1。

M2实际M0全decode状态中head被容量阻挡4,613/4,590调用；其中有至少一个后继能fit为2,820/3,527调用，涉及25/25个head。但日志没有head预约兑现方案，也不证明可安全回填；已失败first-fit不可作为可兑现性证据。M2仍为备线，先做EASY/保守回填直接对照，不能立即重跑无条件first-fit。

现成决策计时仅为pre-schedule adapter部分：两格p50 100.92/98.95微秒，p95 129.94/123.91微秒，p99 170.50/157.55微秒，总0.5895/0.5726秒，对应episode85.75/83.90秒（约0.69%/0.68%）。这不含native wrapper其余日志/同步总成本；CPU进程总时长和额外内存尚无合格分解，不先做事件缓存优化。

## Reproduce and next decision

已有输出只读；重跑需新输出目录，例如：

```sh
mkdir -p /private/tmp/c-m0-packing-rerun
python3 C_research_artifacts/20261001/research_design_v2/mechanism_probe/probe.py --output /private/tmp/c-m0-packing-rerun/result.json
```

脚本依赖上述已检查worktree和本地fresh raw，在import前只读校验M0/base/cell三个固定SHA，不匹配即SOURCE_MISMATCH退出；禁止import写bytecode缓存，拒绝覆盖已有result/trace。新增guard已用 `--help` 验证在当前来源上通过，不重复运行已完成的CPU比较。`log_context.py`是已有日志的补充描述，不改变实测策略；其输出保持一次写入。

本轮最弱链路不再是端点算法实现，而是 **同信息同profile强简单规则已覆盖M0动作，且退化输出压力下服务收益能否迁移至健康自然任务**。只保留M0为Primary基线；下一最小有辨别力的原生对照应是健康任务native/full-bound FIFO/M0资格小块，在新的GPU授权和冻结质量/采样合同后执行。无独立runtime约束或系统规律残差则放弃“新算法”叙事。M1机会短，M2安全预约未证，不并行开发控制器。

探针限制补注：PROTOCOL将随机组简写为near-capacity；实际脚本按0–31个resident生成后仅过滤E<=4096，覆盖宽占用域，未要求1,000个样本全部接近容量。结果应读作“1,000个固定seed合法随机状态”，不声称压力边界样本数。上述A/B停顿反例是抽象证书状态，不声称已由当前M0完整到达历史证明其可达；它只展示进度假设为何不能从安全论证删除。
