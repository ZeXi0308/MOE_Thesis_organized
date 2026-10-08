# 串行运行合同与当前可执行命令

> 当前：Q1/Q10 r03 已完整运行且固定探索判据失败。保护范围诊断也已完成；当前为 PROTECTION_YIELD_TRIPLET_PLAN_R01_20261001.json 三格，已上传，首次公共锁忙未运行；下方旧状态为历史。

状态（2026-10-01）：`H1_B1_METRIC_PASS_NO_DIRECT_B2_GPU_DEFERRED`。首off/on各128/128、27原件哈希通过；on输出率为off的108.40%、mean flow91.08%、maxgap3.176比3.699s，数值达标但on实际direct=0，不能归因H1动作。审计SHA `cde40968…790fb962`；444.636s退出GPU EMPTY。首个B2尝试exec17071公共锁EAGAIN/exit75，未GPU初始化。先本地分析零动作边界，再自然断点重试原冻结B2，不自动轮询锁。以下均为历史。

**2026-09-30 连通性更新：** 在用户要求下，经允许的网络环境已完成一次密码认证 SSH 和只读机器查询；本地受限沙箱的 DNS 错误不能再解释为该远端不可达。现场见 1 张 RTX 5090（UUID `GPU-3fc910c2-bf65-5273-e6b5-6c0d8b6ce03e`）、当前 cgroup `memory.max=96636764160` 字节，查询时无 compute process。详见[只读收据](REMOTE_READ_ONLY_20260930.json)。以下 2026-09-29 关于“尚未连接/DNS 无记录”的叙述是历史检查；GPU/host 使用许可及总时间/费用预算仍未明确，所有 GPU 格保持 `UNRUN`。
后续只读环境检查发现预期共同锁不存在，默认 Python 未安装 vLLM，默认 OLMoE 缓存路径不存在；其它位置未穷举。故须先明确运行时/模型准备的授权和来源，再现场重核精确源码、固定 revision 与锁；不能将当前 SSH 连通误当成资格包可启动。

**2026-09-29 执行前增补：** [H1 token 预算反例](H1_TOKEN_BUDGET_ADDENDUM.md)取消了“物理 direct 必然不延迟 target 首新输出”的无条件预测；target 实际调度量、首新输出、victim/peer 得失必须在 off/on 对照中记录。已执行 G 诊断 54 次 READY commit 的即时 KV 直恢必要条件全部不成立，不能用该轨迹预估 H1 收益；G/H 轻量事件仍只有 `0..816` 的机会界。新主机名本地 DNS 无记录，尚无 SSH 认证或现场状态；历史 r02 shell 固定旧 UUID/Python/锁，若机器不同须单独接受 successor 包。详见最新 [台账](../RESULT_LEDGER.md) 和共享 `research_coord/STATE.md`。

## 源码与 payload 前置门

共享仓库是 sparse partial clone，接受包原先有 32 个 tracked 路径、仅 4 个工作树文件。现已在**隔离**目录 `/private/tmp/moe-recovery-source-20260929-r01` 从同一远端 HEAD `76d6d888de42081c63cd440a8a67623161d8181f` 恢复 r02、H r02、G 诊断、旧 commit 重检报告和 `20260914_recovery_progress_model_r01/input.json`；r02 manifest **30/30**、H manifest **35/35** 文件逐一 SHA-256 匹配。原 tar 归档未取得，不能重新计算已接受 `6888c318…dbe7ec4c9e73` 的归档字节 SHA。G/H 的请求级 `raw.json` 未跟踪且本地无副本；G 详细诊断的逐 commit `selective-store.json` 已恢复并核对，54 次 READY 全部 `free<need`。远端新机器是否持有环境/模型/接受包仍未知。

另一协作进程已交付新的 `ltr_fair_policy.py`/`ltr_fair_native.py` 与 25 项 CPU 通过的测试，以及新 `ltr_baseline_calibration_20260929.json`；root 只读复核通过，尚未把它接受为 GPU 包或修改旧 r02。当前原 r02 可作为单格资格包候选，`check_lifecycle.py` 在隔离目录配当前共同 harness 复跑 `PASS_CPU_ONLY`；运行时实际模型、worker、原生 load 与 EOS 仍待 GPU。新的纯 CPU 动作模型 `20260929_model_action_value/` 8 项测试通过，但未接 native 快照；本轮 H1 只用一个 default-off gate，不在热路径叠加第二个模型/controller。共同 `staged_store_rotation.py` 与已接受 F/G/H 包的 adapter 不是同一字节；root 已将 C 针对 G/H 精确字节的补丁集成到新 [H1 候选包](candidate_h1/README.md)，旧接受包不变。候选包仍须现场核用户资源范围及 GPU 生命周期资格，不能直接当已接受性能组。

共享仓库的本地无网络门禁（仍会失败，只用于说明 sparse 状态）：

```sh
GIT_NO_LAZY_FETCH=1 git cat-file -e HEAD:refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_ltr_style_component_r02/manifest.json
GIT_NO_LAZY_FETCH=1 git cat-file -e HEAD:refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_ltr_style_component_r02/pkg/run.sh
```

当前两项在共享仓库仍失败；隔离检出已补齐并逐文件验证。r02 的原 `pkg/run.sh` 固定旧 GPU UUID `GPU-4015b79d…`、旧 Python 与缓存路径，**只有新机器实际逐项一致**才可复制这 30 个原件和 manifest 原样运行，并现场再验 SHA。若新机器 UUID/路径不同，先把仅环境入口的改动单独封为新 SHA/新包身份，复核 CPU 和科学合同后再执行；不能将它称为原 r02 字节包的运行，也不能用本轮修改过的共同源码替换其 adapter。模型缓存和输入须现场核对，不下载大型模型。仓库中外部进程新增的 `ltr_fair_*`、`20260929_model_action_value/` 等尚未合并文件保持原样，未核之前不参与固定对照。
环境入口候选已独立放在 [candidate_ltr_r02_env](candidate_ltr_r02_env/README.md)：原 r02 的 25 个 `pkg/` 文件仅 `run.sh` 一处变化，24 个科学运行文件逐字不变；新 manifest 26/26 校验与 shell 语法通过。这只是 CPU 候选和新包身份，不能称为原 r02 字节包的执行；用户授权与现场机器/源码/模型/锁/物理预算核对仍是启动前门禁。该入口仍只跑 G64/T30/Q10 一格，不能接 H128 性能或 fair LTR successor 控制器。
外部交付的 [fair LTR successor overlay](candidate_ltr_fair_overlay/README.md)虽有 CPU runner 接线证据，其策略与已接受 r02 不同，当前**不在执行序列**；本轮只保留既定 H1 一个新机制。r02 单格资格及同输入强基线的独立包门禁不能由该 overlay 的测试替代。

## 同资源比较顺序

1. 在用户授权范围内做只读新机器现场检查：GPU UUID/进程/显存、两卡或指定卡的隔离、共同 `/root/autodl-tmp/moe-research-gpu.lock`、cgroup/process-tree host 限额、模型/runtime/source hash 和 r02 payload 是否可得。任何资源忙碌或未知、预算不符、源漂移均 `ABORT` 并保留状态。检查空闲不等于占用许可；整组前台锁必须覆盖初始化、换格与归档。
   两个候选包的 `run.sh` 只保护单格；整组由 root 的 [串行会话封装](SERIAL_GROUP.md)持同一 fd 9、总墙钟及格间/归档收据。发生超时或异常，先在锁内尽力核 compute process，再现场复核所有 GPU/worker/host 状态；未证实空闲则停止，不推进下一格。该封装目前仅支持 G64 资格候选和 H128 H1 候选；新 H128 LTR 性能包冻结后另行扩展身份白名单与复核。
2. 若新机器与已接受 r02 的固定 GPU UUID/路径/源码合同一致，则只执行其原 G64/T30/Q10、180s native 生命周期诊断；否则先为 r02 的环境入口生成独立包身份，保持 LTR adapter 与工作负载逐字不变并复核后执行同一单格资格。若 pinned vLLM 后端或模型实质不兼容，则 `ABORT`，不能用另一组件冒充 r02。真实 store/load/flush、量子跨首新输出、EOS、槽位和归属均要现场验证。资格失败为 `INCOMPLETE`，不做性能排名。不得把 CPU fixture 写成 GPU 通过。
3. 接受**新**同机器性能包之前冻结一档自然到达 cohort、到达序列、OLMoE/tokenizer/revision、EOS 允许/cap1024、GPU 4096 usable blocks（若新卡物理不符则重订合同）、host KV 16 GiB 与独立物理限额、最大 episode 180s、warmup/服务/drain 边界。先串行 native full/no-extra-rotation，再 selected/eager，随后兼容 LTR-style 的事前 G 开发校准 T∈{30,200} × Q∈{1,10} 至多四点；合法点以同窗口 eager 的实际 token 率≥97%、平均完成≤105%选择最小最大 gap。没有合格点就保留负结果，不扩网格。G/H 已见过，不能充当 H1 盲测。
   [LTR 同输入只读审查](ltr_audit/REPORT.md)已确认原 r02 是固定 G64 单格诊断，H128 性能必须新包身份；其三个 LTR 策略模块可保持原字节，输入、安全上界、runner 和环境入口另封并做静态/CPU 复核。G64 原生生命周期资格未通过前，不把 H128 LTR 候选列入有效强基线。
4. 固定所选强基线和 H1 代码后，在**同一** selected/eager 后端做 `commit_recheck=False/True` 反序配对，先一代表性压力点。只有动作与资源门禁通过，才增无压力负控、自然 EOS/异构长度和持续 host 周转；不能临时加压制造事件。完整成本包含在线检查、保存/搬运/重算、真实输出和 drain；失败/拒绝/未完成全留在 cohort。
5. 对共同短时域只比相同的后续四个正调度调用，计未兑现恢复及窗口尾 KV/队列债务；完整请求独立演进才作主结论。新输入/到达序列确认要在参数冻结后另行接受，旧 G/H 不标盲测。

开发阶段**全臂共同主评价** goodput 前沿固定为 `D_F={10,20,30,40}s` × `D_G={1,2,4,8,12}s`，无生产 SLO 含义。分母为各臂相同定义的 capture 时长；仅完成且有新输出、TTFT 与每请求最大生成间隔达标者计入分子。输出 token 率和带 180s 未完成罚时的 mean flow 与原生参考并报。另一协作进程 LTR 比较器中的三点阈值只作其 CPU 开发诊断，正式同臂比较统一用本目录固定前沿。旧 eager/current 相对 3%/5% 合同仍只解释旧对照，不转换为相对 native 的达标判据。

取得同一完整 cohort 的两份 `raw.json` 后，根目录可直接运行请求级分析；路径换为当次**新** raw，不覆盖旧结果：

```sh
python3 refine-logs/expert_saturation/experiments/admission_capacity/20260929_commit_recheck/evaluate_goodput.py \
  --reference /path/to/native/raw.json \
  --candidate /path/to/h1-on/raw.json \
  --expected-requests 128 --timeout-s 180 \
  --output /path/to/new/analysis/goodput-native-vs-h1.json
```

同理分别给 eager、LTR-style、H1-off/on 做配对。`evaluate_goodput.py` 校验请求数、身份、到达和 token 时间，输出全阈值计数、吞吐、完成/失败/未完成、请求等权 gap 分布及逐请求差异。它**不能**验证资源/源码公平或质量；这些必须由同包 config、runtime receipt、物理监测及输出/EOS差异共同判断。新机器 GPU 数、host 硬预算、准确源包和运行时长批准前，远端执行命令保持 `UNRUN`，不构造一个假装可启动的脚本。


## 2026-10-01 当前唯一探索：在轮转前选择可直接恢复的替代者

H1两个反序配对完成，共同停顿判据失败，停止原固定目标提交重检的性能扩展。新的自然证据为prepare步78/267存在数值fit的其它暂停请求，其中40步符合原30步缺席条件；不更改输入、到达、资源或阈值。

主假说：仅在原eager已提出rotation时，从其它实际可接纳且已等待至少30步的PREEMPTED请求中选择最长等待者，可避免该次prepare/store/victim并返回新输出，同时完整请求停顿收益没有被其它请求代价抵消。原目标被延后是必须计入的代价。实际门禁失败沿用原proposal；H1 commit_recheck在两臂均off。

只执行一个off/on原生探索配对，900s每格、3200s总墙钟，共同锁/90GiB cgroup及已有GPU不变。两臂都用轻量performance记录和同一新包，on才开fit_first_resume。先回答实际改选→原生准入→后续输出是否成立，同时保留全部128请求的服务指标、逐请求得失、goodput和输出差异。未完成/错误立即保留并停后续格；零实际动作即不扩性能。预定探索有利方向仍为on/off输出率≥97%、mean flow≤105%、最大gap更低，且不能隐去原目标等待和peer恶化；一个配对不作确认或新颖性结论。规则属于容量感知的强简单基线。

计划`FIT_FIRST_PAIR_PLAN_R01_20261001.json`；复用旧串行锁/归档实现，仅一个小wrapper固定新包、顺序、开关和输出路径，不新增审计框架。

已有诊断40个年龄合格的数值fit时刻覆盖38个原目标、28个替代目标（相关时刻，不是独立重复）。原目标缺席步数中位67.5、替代者55；完整历史需求中位186.5/129块。改选系统性偏向更小历史，原目标等待代价需显式报告；这些数值不作新的选择阈值。见`A_FIT_FIRST_DISPLACEMENT_DESCRIPTION_R01_20261001.json`。


## 2026-10-01 容量合格victim简单基线

Fit-first首个原生配对47实际动作/256完整请求，但maxgap3.795→12.186s，当前规则停止扩展。旧H1诊断中511/511次prepare资金不足拒绝可精确重现，且另一个同样eligible的victim均足够。新唯一问题：保留最长等待target、已有eligible规则及most-output排序，在排序前过滤free+held不足者，能否消除该类拒绝并改善完整请求停顿？这是强化简单基线，不包装算法新意。

一个off/on原生探索配对：同H128/同资源/同保存后端，两臂fit-first与H1全部off；capacity_victim默认off，on仅增加即时资金资格筛选，无可行victim时noop。choice不算实际动作，原prepare、提交检查及真实native preemption成立后才计capacity_victim_commits。每格900s、组3200s、同公共锁与90GiB cgroup；首格错误停止，不自动重复。全部128请求、输出差异、逐请求TTFT/flow/gap及固定goodput保留；有利探索方向仍为率≥97%、flow≤105%、maxgap下降，单块不作确认或新颖性结论。不改变阈值/seed/输入以挽救原fit-first结果。


## 2026-10-01 同策略 A/A：正式服务中的缓存不对称

相同 capacity=true/q1 两格256/256完成；first/second率1440.868/1455.358 token/s、mean flow39.689/36.666s、maxgap2.694/3.537s。第二格flow低7.62%但尾部更差；62输出序列不同、0终止原因不同。同策略也有相当幅度差异，单个策略配对不能直接归因，且不能据一次A/A估计总体方差或扣除其它配对效应。结果见`A_CAPACITY_IDENTICAL_PAIR_RESULT_R01_20261001.json`。

阶段定位`A_CAPACITY_IDENTICAL_PHASES_R01_20261001.json`：第一个token的最早内容分歧在1.176/0.844s，早于两格首次原生抢占6.383/6.457s；0–5s均26到达、2完成、0抢占，第二格已多978token。缓存元数据`A_CAPACITY_IDENTICAL_CACHE_PHASES_R01_20261001.json`显示first正式计时内64个Triton文件写入（2,867,515B），second为0；这与早期分歧重叠，尚非全部差异的因果解释。

唯一下一项：同策略暖缓存A/A，用已完成缓存的两个独立副本作相同起点，保持候选、输入及资源不变。检验正式阶段是否仍写编译缓存、0–5s输出差距及完整分布是否收敛；一组仍非统计确认。条件q10仅CPU就绪，未在GPU运行；本轮不以额外审计扩展门槛。


## 2026-10-01 Q1/Q10 原生保护探索失败及唯一定位实验

完整结果 `A_CAPACITY_PROTECTION_PAIR_RESULT_R03_20261001.json`：同新GPU、同H128、capacity=true，两格128/128完成；Q1/Q10输出123519/124468，率1491.394/1415.183 token/s（94.890%），mean flow37.340/38.116s（102.080%），maxgap2.243/2.529s；原定rate≥97%且maxgap下降未通过。60输出序列、1终止原因不同；flow51改善77恶化、gap54改善74恶化、TTFT27改善101恶化。实际206次延长均产出10个新token、8次未来增长不足退回首输出保护；全抢占375→263、短服务1–2输出后再次抢占22→11，局部现象改善未转化为完整服务收益。停止该Q10实现的性能扩展，不扫q、不自动追加反序块。

`A_CAPACITY_PROTECTION_EXPOSURE_R03_20261001.json`：206个不重叠延长区间共29.661s，其间其它请求仍输出42945token；不是独占GPU或可直接相加的损失。137个延长episode释放后仍有后续抢占，69个直至完成没有。最早内容分歧在1.587s，早于首次抢占；正式缓存写入16/0，因此不能作精确单动作归因或等工作量比较。`A_CAPACITY_PROTECTION_PRINTED_TRANSFERS_R03_20261001.json`：各7个完整打印区间的store39.548/35.316GB、load123.619/85.656GB，CUDA copy时间不等于请求暴露等待，首尾未覆盖仍unknown。

新的唯一问题：延长保护不仅保留目标未来增长，还沿用全局非目标waiting-loop break；是否实际阻挡了当时数值足够的其它等待请求？下一单格保持同Q10策略，只记录实际命中该break、held peers、空闲与目标增长/等待头需求及调度预算，避免全栈诊断。若没有真实命中的可容纳状态，不基于结构猜测放开准入；若存在，再检验一个保护范围更窄的实际动作。该诊断不与旧格做性能排名，也不将数值fit等同原生准入。完整贡献与新输入确认仍缺，未READY。


## 2026-10-01 Q10 全局保护范围：实际队列阻挡成立

同Q10稀疏原生诊断128/128完整完成，168.058s，exit0/GPU EMPTY。1755个延长步全部命中waiting break；174步涉及24个队首、22个保护目标，R0/无transfer job/无skipped queue/pending pushFalse，free足够队首完整历史及目标未来增长，余量5–354块。另123步有peer hold，共1029请求步。首个反例0008645需185块、free235、目标growth0，仍被保护分支挡住；数值可容纳不是原生准入证明。见`A_PROTECTION_SCOPE_DIAG_RESULT_R01_20261001.md`及完整rawsession。

唯一下一项为实际waiting边界的条件提前释放，未知/不足保留Q10；用Q1/条件释放Q10/普通Q10三格同时保留强简单参照和消融。真实原生接纳、首新输出、原目标及peer直到完成均入账。仍为简单基线延伸，无独立贡献主张，不扫q/seed，不把诊断时间与旧性能比较。计划`PROTECTION_YIELD_TRIPLET_PLAN_R01_20261001.json`尚未冻结/上传。


### 条件提前释放三格已固定、首次资源申请延后

`candidate_protection_yield_r01`在实际waiting-loop gate尝试提前释放：复用已有 `_direct_resume_reason` 的状态/slot/真实所有权检查，并要求R0、无注册transfer/未知push、full-history+目标future-growth有资金；其它情况保留原break。记录`protection_yield_to_ready_head`与`yield_head_admission`（包括NO_NATIVE_ADMISSION）。释放不因未接纳而回滚，不承诺10输出。四个实际closure CPU例通过，只是实现检查。

同包Q1/q10_yield/普通Q10顺序，计划`PROTECTION_YIELD_TRIPLET_PLAN_R01_20261001.json` SHA85560f9904fc29d1dde5098c5e9456b3e2e1ecc61a7ec34458713662c34b3022，包manifest b77bdf94bcea2b98fbb7e191a379ba199ad8fd782f50b7b1b26af521ab757503。已上传解压，exec38476首次flock EAGAIN/exit75，无测量session或A GPU作业。下次仅运行既有controller，不重解压，不改策略/输入。GPU未运行，不能声称接纳或性能收益。


## 2026-10-01 条件提前释放三格完成：实际动作成立，尾部判据失败

Q1 / 条件释放Q10 / 普通Q10三格均128/128完成，组445.759s、全部exit0/GPU EMPTY。率1419.121 / 1413.993 / 1404.383 token/s，mean flow37.704 / 38.153 / 38.041s，每请求最大gap的p95为1.772 / 2.857 / 1.924s，cohort maxgap2.038 / 3.656 / 2.313s。条件释放相对Q1率99.639%、meanflow101.189%，但停顿下降判据失败；不能因与普通Q10吞吐略高而替换Q1强参照。

25次提前释放均为真实ASYNC_LOAD_ADMITTED，并有后续新输出及最终完成；admission→新输出中位46.5ms。原保护对象全部完成，10个动作之后原对象还有后续抢占（相关事件，不是因果增量）。总抢占332 / 266 / 249，短1–2输出再次抢占18 / 10 / 12；局部减少没有转成尾部收益。条件释放相对Q1/普通Q10有57/65输出序列不同、均0终止原因差异，输出总数124464 / 124445 / 124464，不能称等工作量加速。

主结果`A_PROTECTION_YIELD_TRIPLET_RESULT_R01_20261001.json`；raw为`moe-a-protection-yield-session-r01-20261001/`。已打印7个完整区间store39.781 / 33.643 / 34.152GB、load115.402 / 87.323 / 85.788GB，首尾未知，CUDAcopy时间不等于暴露请求等待，见`A_PROTECTION_YIELD_PRINTED_TRANSFERS_R01_20261001.json`。停止此条件释放规则的性能扩展，不扫q/seed或自动追加反序块。下一步只从现有raw定位最差停顿的抢占→准入→新输出和原对象/peer代价，再决定有依据的实际动作；全论文仍未READY。


### 最差停顿定位及工作稿

`A_PROTECTION_YIELD_TRIPLET_TARGET_PEER_R01_20261001.json`：最长六个gap共同落在73.97–78.49s，抢占后到记录准入前占各gap的98.36–98.66%，准入到输出48.3–55.5ms。起始抢占五个匹配强制victim交换，一个跟随普通Q10保护释放；联合区间13个保护开始、12个十输出释放、0提前yield。不能严格把整个前段归为单一排队原因或作相同前态因果结论。停止扩展Q10/提前释放，下一唯一可检验线索：一次victim回收后的空闲是否足以同时恢复原target和另一个实际等待者，以及该余量是否在首target新输出前消失；先使用已有H1完整诊断，不新跑审计格。

工作稿已编译 `paper_a/output/pdf/A_recovery_working_draft_20261001.pdf`（10页，保留NOT SUBMISSION READY），主源码`paper_a/draft.tex`，重建`bash paper_a/build.sh`。已检查全页排版及最新图表，无未解析引用/越界。PDF只是当前证据的可读版本，不改变独立贡献/新数据确认缺口。


## 2026-10-01 新唯一探索：原目标首输出后的一个资金合格 follow-up

已见H1真实诊断中267次强制commit有93次数值足够原target+另一PREEMPTED请求，85次到原target首输出时另一个等待者仍可容纳；更大的FCFS头挡在前面。该证据不含完整native jobs/R/streaming slot门禁，不能直接认定可接纳。余量多数仍在，暂不引入并发加载或多目标保护。

具体动作：固定capacity/Q1，原最长等待target照常恢复并返回首个新输出；仅在这次正常强制rotation的首输出释放边界，至多再选择一个达到原30步缺席门槛、当前完整历史可容纳且真实_direct_resume_reason/已知R0/无jobs/明确无pendingpush通过的等待者，按最长缺席顺序尝试。它使用现有单目标首输出保护及native恢复，不额外抢占；它自己的首输出不能再次触发follow-up。UNKNOWN/不足回退Q1。与旧fit-first的区别是原target先被服务，且不允许连续绕过；仍须记录当时被越过的最长等待者和原victim的代价。这只是更强的简单基线，尚无独立贡献。

只做off/on相邻原生pilot，同包同H128、Q1/capacity=true、其余开关off，同已完成yieldtriplet第一格cache的两个独立副本，900s/格、3200s/组、90GiB、原共用非阻塞锁。需全部128请求完成、非零实际followup接纳→新输出→完成、原target先输出且followup步无额外抢占，并满足rate≥97%/meanflow≤105%/maxgap下降相对同组Q1；全20点goodput、逐请求得失和序列/终止差异一并保留。无动作或失败则保留边界，不调阈值/seed/每轮followup数量救援。

草稿 `candidate_spare_followup_r01`（实施中）、`run_spare_followup_pair_r01.py`、`SPARE_FOLLOWUP_PAIR_PLAN_R01_20261001.json` 和 `analyze_spare_followup_pair_r01.py`（等事件字段对齐）；尚未冻结、上传或GPU运行。


### 有界 follow-up 已固定，首次 GPU 申请延后

Root完成新私有包，仅三文件改动：Q1原target首输出后，仅当当前PREEMPTED队首完整历史超过free时，按既有年龄规则选择一个真实门禁合格的额外等待者；复用原生单目标恢复，origin保证每正常rotation最多一个且不链式触发。7个实际closure CPU用例通过（含off、未知R、pending、资金不足、无head阻挡、真实scheduled回执/不链式和NO_NATIVE_ADMISSION留痕），不作GPU证据。

包25文件manifest SHA `cc171a26feeb57c719b73d3788d8531a0c3ab5f295457d594a7670301f3cc6de`；controller `run_spare_followup_pair_r01.py` SHA `547c85aebc8fcacd3bca4bcb91ce7be88058425791e569aa9f54e1c141d9ad6a`；plan `SPARE_FOLLOWUP_PAIR_PLAN_R01_20261001.json` SHA `2343699e5d5d43bb604dbdc5c8425242e12514fb14e2ecc816d902c14c21b735`。两臂同包off/on，900s每格/3200s组，与既有90GiB及共同锁；首阶段cache由已完成yieldtriplet第一格独立复制。

3,428,009B tar已上传/解压 `/root/moe-a-spare-followup-stage-r01-20261001`。exec31795首次controller退出75/flock EAGAIN，GPU_DEFERRED、无A测量启动。分析脚本 `analyze_spare_followup_pair_r01.py` 已对齐事件字段并通过语法/CLI检查，等待真实原件；不造输出。下个自然断点先查session/精确controller进程，若不存在只启动既有包；不要再次解压、快速轮询或改q/seed。主研究未完成，工作稿PDF不等于可投稿。


## Spare-followup reverse-order replication, frozen before r02 results

The unchanged candidate r01 produced 24 native chains and passed every criterion in first off/on block (256 complete). Execute one on/off block with the same source warm cache, inputs, arrivals, parameters and resource limits; see SPARE_FOLLOWUP_PAIR_PLAN_R02_20261001.json. Both blocks must independently pass complete cohorts, real follow-up output/completion, primary first, no recursion/extra same-step preemption, rate >=97%, mean flow <=105%, lower maximum gap. Retain all actor costs and output changes. No pooled rescue, parameter change or third rescue block. This is replication on seen input, not untouched-input confirmation or novelty.
