# 复现与模块使用入口（当前实现）

当前结果入口为 `RESULTS.md`。本文先给可执行入口，后半部分保留历史执行记录；历史段落中的 PID、排队状态和“尚未完成”描述只反映当时状态，不能作为现在重新提交的依据。

## 版本与运行范围

已测 r09/r10 和现有 Qwen native 校准使用 `code-r12`（SHA256 `38f8f9c3eed04842d8b19ce02e3cae20d41ac961e4f54f774dabecc476f89c88`）。待跑六组 OLMoE 前沿改用 `code-r13`，归档 `runs/code-r13.tar.gz`（44,411B，SHA256 `458ea0aca8c5e6d23b4548ec84ff9b0ac3f2467e73df294b4a7ec1b923ae3e37`），逐文件SHA见 `configs/code-r13-sha.json`。新版本包含owner去重、reentry与租约到期桶处理，已上传且通过CPU检查，尚无该版本GPU结果。不能混合不同版本形成一条策略曲线。

模块仅安装到单个 scheduler 实例，不修改共享 vLLM 安装。`install()` 验证 Scheduler、KVCacheManager、offload scheduler 源文件 SHA，而非只检查版本字符串；要求 pinned 同步 FCFS、单 full-attention KV group、16-token block、关闭 prefix caching/speculative decoding、单设备上下文。异步 KV LOAD 仍受支持，`AsyncScheduler` 不在当前范围。支持条件以模块 `install()` 的实际校验为准。运行环境参数以各格 `engine_args.json`、`environment.json` 为准。

## 安装、输出通知与卸载

`horizon_sched.py` 自身只依赖 Python 标准库；把选定快照的目录加入模块搜索路径后，在创建引擎、预热及清空预热状态之后安装。下面两个函数可嵌入本课题使用的同进程 `LLMEngine` 驱动：

```python
import time
from horizon_sched import Config, install

def attach_horizon(engine):
    scheduler = engine.engine_core.engine_core.scheduler
    return install(scheduler, Config(
        H_a=64, H_r=32, lease_victim=True,
        stall_order=True, estimate_Hr=False,
    ))

def notify_output(adapter, internal_request_id, new_token_count, received_at):
    if new_token_count > 0:
        adapter.receipt(internal_request_id, received_at)
```

每次捕获新增输出时，用同进程的 `time.perf_counter()` 获取 `received_at`，传入 scheduler 使用的内部 request ID；不能传文档ID、相对实验起点时间或 Unix 时间。记录时刻应与延迟观测一致，不能把回调处理后的另一个时间当成原输出时刻。该回调提供 stall 排序与 EWMA 的观测信号；缺失 receipt 时排序按到达时间回退并计数，不能声称执行了完整 stall/EWMA 行为。当前 runner 的实际接线在 `code/run_cell.py`，使用 `origin + received` 还原同一 monotonic clock 时刻。

卸载入口是 `adapter.uninstall()`，返回统计字典（可序列化为 `horizon.json`），重复卸载返回同一统计状态。它恢复被替换的实例方法并清空租约等内部状态；它不取消请求、不释放引擎、不代替 offload drain 或 engine shutdown。完整驱动应先结束/处理请求并 drain，再逆序移除后来安装的 observer/hook，最后卸载 adapter；若后来的组件仍占用同一 hook，卸载会报错，避免覆盖其他组件。

配置中的 `H_a=0` 关闭新请求视野门控；固定视野模式（`estimate_Hr=False`）下 `H_r=0` 关闭恢复视野门控/新租约。完整关闭恢复保护的已测消融同时设置 `lease_victim=False`；`stall_order=False` 单独关闭恢复前缀重排。`estimate_Hr=True` 使用输出回调估计恢复步数，当前不具备全局无抖动保证。每步维护仍包含线性账本重建与 victim/前缀操作，不能称为已达到 O(1) 的完整实现。

复杂度口径：`_build()` 遍历 RUNNING 与在途请求的并集，且 `_row()` 逐个检查这些请求持有的物理块，因此单次重建为 O(请求数 + 持有块数)。`begin()` 和恢复 reentry 的 `allocated()` 均可使账本失效；同一 waiting 阶段不保证只重建一次。只有已缓存总量的门控查询为 O(1)。这也适用于冻结的 code-r12 执行路径；不修改历史归档中的原始说明字段。

code-r13 包括完整 LOAD 的 reentry 快速路径：缓存已验证的非运行在途 owner，若历史 token 数和持有块数未变且无剩余历史块需求，转为 RUNNING 时仅增加 n，复用原有增长需求。分配变化、重复回调、异步重入仍使账本失效；失败 ACK 在原生处理后使账本失效。`check_native_cpu.py` 将该路径与完整重建的 demand/R/I/n/reserved 对照，并实际调用安装后的 ACK hook。此为 CPU 验证，尚无真机开销或服务收益证据，完整 O(1) 要求仍未满足。

## 完整实验与本地重建

新版 F3 在同版本 W2 十臂结果完整后选择 cap 与 H，并在 W1/W3/W2-2λ 上固定使用。`figures/F3/prepare_transfer.py` 接受 `--w2-plan <W2配置>`、`--code-snapshot code-r13`、`--cap <已选cap>`、`--ha <已选H>`、`--reason <根据W2完整目标说明取舍>`、`--session <新session名称>`、`--output <新配置路径>`，生成两臂 W2-2λ 配置。它复用 F2 的结果核对，保存选择点及源配置/指标/收据 SHA，只改变到达率和 session；不覆盖已有配置、不提交 GPU 工作。W1/W3 从现有前沿复用同一组固定参数，不能分别重新调优。W2 选择点不能称独立确认；不同预算分别选择。当前未测数据拒绝分支已实际检查，完整生成及后续新版 F3 绘图仍待数据；旧 `F3/plot.py` 只绘历史版本。

修正版 F5 按单组完整十臂配置生成，示例：`python3 -B research_A_horizon/figures/F5/plot.py --plan research_A_horizon/configs/olmoe24-w2-corrected-frontier.json --code-snapshot code-r13`（需要 matplotlib）。输出 `figures/F5/<session>-cost.{json,pdf,png}`；缺格只生成缺失状态 JSON。复用 F2 的模型/输入/代码版本/执行配置核对，使用本组实际 cap 标签及指标 SHA，不与旧版 F5 合并。预留分位数仍是 KNOWN 门控尝试样本，不是时间均匀样本或置信区间；CPU 时间是包含观测成本的整个调度器均值，不是隔离的 hook 增量。该入口的完整数据绘图与视觉检查待组运行后完成。

GPU 实验通过冻结快照的 `run_group.py --plan <新plan.json> --wait-lock-seconds 21600` 执行。plan 中的 session 必须是新目录；不要直接重跑已提交配置。该入口设置运行环境、整组申请公共锁、逐格检查磁盘和 GPU 占用并在首个失败后停止。`run_cell.py` 已包含安装、输出回调、观察器、卸载及 shutdown 的完整接线，可作为上面 API 的可运行参考。

当前接续为 `code/continue_matrix_r13.py`（PID43621），按 `configs/matrix-r13-sha.json` 用 `code-r13` 执行六组前沿。它等待原 Qwen native 校准34980完成并归档；旧协调器34979已结束，勿恢复。`code/archive_matrix.py --qwen-session qwen-calibration-r11b --socket /private/tmp/horizon-matrix-ssh`（PID60690）继续负责下载、核验、分析和确认。均已提交，先核对进程与收据，勿重提。公共锁候锁时间与实际GPU持锁预算分开；累计执行上限仍约12小时。新矩阵F2/F5重绘须显式指定 `--code-snapshot code-r13`，历史示例保留原版本。

从已归档数据重建单组指标（工作区根目录）：

```sh
python3 -B research_A_horizon/code/analyze.py --session research_A_horizon/runs/<session>
```

每个图表的具体命令见下文对应 F1–F6/T1–T3 段落。旧 F2/F3/F5 使用原账本快照；修正版十臂面板使用 `figures/F2/plot_corrected.py`，同模型六子图使用 `figures/F2/plot_matrix.py`。数据缺失时不生成结果图。论文仍为工作草稿，完整矩阵及最终版式尚未验收。

## 历史执行记录（保留复现来源，旧状态不作为当前操作指令）


在工作区根目录运行三个 CPU 数学检查：

```sh
python3 -B research_A_horizon/code/check_cpu.py
```

该命令只检查数学模型。已完成三项上 GPU 前检查的入口为：

```sh
python3 -B research_A_horizon/code/check_native_cpu.py
```

它执行实际改造后的 pinned victim 分支与回滚，并使用 B 真实 KNOWN 快照；完整 engine 行为仍需真机验证。

首格运行配置为 `configs/w2-48-native-calibration-r01.json`，远端已提交。查看现有 receipt 和进程后再决定后续动作，不重复提交同 session。新组入口为 `code/run_group.py --plan <plan.json> --wait-lock-seconds 1800`。

完整结果取回后：`python3 -B research_A_horizon/code/analyze.py --session research_A_horizon/runs/<session>`。缺 raw 的失败格会明确输出 performance=null，未完成格不纳入有效前沿。

F1 历史动机图已生成并目视检查；本地现有 matplotlib 环境下可重绘：

```sh
MPLCONFIGDIR=/private/tmp/horizon-matplotlib XDG_CACHE_HOME=/private/tmp/horizon-cache /private/tmp/format_sched_plot_venv/bin/python -B research_A_horizon/figures/F1/plot.py
```

其他有 matplotlib 的 Python 也可运行同脚本。`figures/F1/data.json` 保存提取数据及两份历史来源 SHA；panel(c) 与 native 时间线来自不同运行，不拼成同一因果链。

LaTeX 骨架：`paper/main.tex`，已通过 pdflatex 编译；尚非完整论文，未做最终版式验收。F1–F6/T1–T3 的数据、逐图脚本和真机命令尚未产生，不使用占位数值绘制结果。

```sh
pdflatex -interaction=nonstopmode -halt-on-error -output-directory /private/tmp research_A_horizon/paper/main.tex
```

远端复用 B `plan-r03.json` 的 `library_path` 与 `runtime-overlay`。公共锁按整组持有，组前数据盘 ≥10 GiB；新 session 放数据盘。每组释放锁后压缩 raw，下载核对，再清理已归档可再生 cache。旧组原始结果与其他会话文件不覆盖。

有效自然EOS结果：`runs/w2-48-native-calibration-r02`（λ10）与 `runs/w2-48-calibration-policy-r03`（λ20的native/HORIZON64/32）。对应 `.raw.tar.gz` 保留完整原始数据；metrics.json由analyze.py生成。r03确定48GiB/W2的λ20；r04为相同工作点native三次重复，仍在运行。

基线源码来源及SHA见 `code/vendor_c/sources.json`。full-budget/MC使用原C installer；MC当前是同步scheduled-prefill适配，不是async_mc.py的v25异步完整系统。基线适配待真机验证。

F2当前数据由r03及r04完整运行产生，缺失臂记录于 `figures/F2/data.json`，逐运行全指标表为 `figures/F2/T1-runs.tsv`。r05/r06取回并analyze后运行同一命令自动加入数据：

```sh
MPLCONFIGDIR=/private/tmp/horizon-matplotlib XDG_CACHE_HOME=/private/tmp/horizon-cache /private/tmp/format_sched_plot_venv/bin/python -B research_A_horizon/figures/F2/plot.py
```

r04三次native重复已经完整归档并核验，局部波动描述见该session的variability.json。r05基线、r06前沿扫描已提交，先检查现有进程/receipt，不重复启动。W1/W3 builder已支持family参数；W1全短、W3同W2文档长度而长组先到达。新负载尚未生成，不把脚本存在当实验完成。

T2历史表重建：`python3 -B research_A_horizon/figures/T2/build.py`，来源B原文节名/行号及指标SHA见data.json。

r06已终态STOPPED：前三个cap及Ha16完成，Ha256/1024未运行；完整归档已核验。仅在数据盘≥10GiB后，用code-r05运行r06b两格续跑计划。r07独立code-r07含broad公共预热和mc-load适配，计划三格尚未提交。两种预热协议不能直接合并成同策略重复。

CPU负载构造命令（远端已有输出，不要覆盖或重复执行）：`/root/miniconda3/bin/python -B build_workload-families.py --source <configs/w2.json中的source_path> --model <model_path> --family W1 --output configs/w1.json`，W3同理。复用原plan的LD_LIBRARY_PATH、runtime-overlay，设置HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE/PYTHONNOUSERSITE。输入已取回configs/w1.json、w2.json、w3.json；W3按document_id验证与W2提示词完全一致，仅顺序改变。

### 扩容后的续跑

r06b：`configs/w2-48-frontier-r06b.json`（仅缺失Ha256/1024，code-r05）；r07：`configs/w2-48-warmup-mc-r07.json`（code-r07，公共broad预热，native/full-budget/mc-load）。两组已提交，共用GPU锁；不要重复运行。磁盘实测200GiB、可用158GiB。待终态后补归档hash。

r06b已完成并本地核验：8,997,116B，SHA `96ea9e17abe91db31abd6d40608db4b95e191201c8a60d1ee4c23bf4bdb7a26a`。analyze.py重建两格metrics；F2重绘为13点、无缺失臂，T1同步。r07仍排队。

第二模型下载已启动PID3373：`HF_HUB_DISABLE_XET=1 python -B prepare_qwen.py --root /root/autodl-tmp/research_A_horizon/models --revision b968826d9c46dd6066d109eabc6255188de91218`。先核对进程与models/qwen3-8b-download.json，勿重复提交。完成后使用记录的model_path构造输入，builder加`--source-model <OLMoE snapshot> --disable-thinking`。Qwen运行前仍需新代码快照和真机验证。

F5重绘：`MPLCONFIGDIR=/private/tmp/horizon-matplotlib XDG_CACHE_HOME=/private/tmp/horizon-cache /private/tmp/format_sched_plot_venv/bin/python -B research_A_horizon/figures/F5/plot.py`。输入F2/data.json及其源metrics哈希，门控采样与整调度器口径见F5/data.json。r08计划configs/load-transfer-r08.json，执行code-r05/run_group.py；Qwen下载PID3373在r08终态后SIGCONT。

r07 raw已核验，SHA `8d3b58a6b93a4e80947fd29804d5b13fd9823181101c4f88e22ca67e2120e8d1`；analyze.py输出mc_load_activity。r08 controller4274实际存活，勿重提。

T3重建：`python3 -B research_A_horizon/figures/T3/build.py`，九项源链接、版本与定位随data.json保存，table.tex已载入论文。24GiB下一校准配置w2-24-calibration-r09.json未提交，r08仍运行。

F3重绘命令同F2，脚本改为figures/F3/plot.py；当前2点/6缺失，r08归档分析后重绘。消融预备计划configs/w2-48-ablations-r10.json（未提交），代码runs/code-r10.tar.gz，SHA 66b6a3af8f95671e2288ae7f30a6958a3bcdd55762e5f8fdf2d903f7cd00490c。关闭lease同时设置hr=0和lease_victim=0，关闭admission为ha=0。

恢复链关联已集成analyze.py，调用code/recovery_chain.py；r03已重分析，旧trace缺ACK保持null。原始归档不变。稀疏ACK/再次分配日志只存在当前本地源码，code-r10快照仍为修改前版本。

当前可用新快照code-r11，归档SHA `4e2604b0613a7e71029da8eefd3c9a440342fcc5fd2ad6cdabb54aeb6139637a`。包含ACK/再次分配记录、恢复链analyzer、按document_id固定采样种子。code-r10历史快照不变。

r08已归档核验（SHA `f8b0decc6fd0d63b53115b0406998095ad586df138e7c35782586eb8e12d301f`），analyze.py后重绘F3得到8点、0缺失。Qwen3373已恢复，启动新GPU组前检查下载状态以避免混杂。

Qwen下载改为fetch_qwen_mirror.py --root /root/autodl-tmp/research_A_horizon/models，PID9620；3373已终止，旧收据保留。输入metadata为qwen3-hf-weights.json及qwen3-modelscope-source.json，全部权重SHA相同。完成后用prepare_qwen.py原固定revision补齐snapshot。

全运行T1：在工作区根目录执行`python3 research_A_horizon/figures/T1/build.py`，输出`figures/T1/{data.json,all-runs.tsv}`。当前25行/24完整；失败保留，r07下载重叠与broad预热单独标注，不池化。旧`F2/T1-runs.tsv`仅覆盖初始前沿，本入口覆盖全部已分析运行。

镜像下载更正：首次启动未复用HF随机后缀partial；新下载已超过旧进度，继续PID9620，最终校验SHA后再清理旧partial。模型未就绪前不提交Qwen格。

F4：`python3 research_A_horizon/figures/F4/plot.py`。数据来自`runs/w2-48-ablations-r10/metrics.json`及同组receipt/inputs；严格核对`configs/w2-48-ablations-r10.json`。完整数据前只更新missing清单，不绘制结果。完整后输出F4.pdf/F4.png，主图四个组件臂，敏感性与MC保留在data.json。

当前Instruct恢复链图：`MPLCONFIGDIR=/private/tmp/horizon-matplotlib XDG_CACHE_HOME=/private/tmp/horizon-cache /private/tmp/format_sched_plot_venv/bin/python -B research_A_horizon/figures/F1/recovery.py`。输入r03/metrics.json，源SHA与逐事件样本在recovery-data.json；与F1历史固定输出动机图分开。论文已加入该图并完成修改页视觉检查。

Qwen资产后续控制器：`python -B finish_qwen_assets.py --root /root/autodl-tmp/research_A_horizon --code /root/autodl-tmp/research_A_horizon/code-r11 --downloader-pid 9620`，当前PID12459。先核实该进程与models/qwen-assets.json，不重复提交。仅CPU准备：镜像完成→固定revision完整验证→三类输入→W2/W3 token多重集核对。脚本SHA256 d601003003f3e133fb00494846fc017e1657b992dd33a81d458ebf18873af068。

修正版code-r12已上传独立目录并核验：43,579B，SHA256 `38f8f9c3eed04842d8b19ce02e3cae20d41ac961e4f54f774dabecc476f89c88`。包含在途LOAD增长跨账本重建保留的修复；旧code-r11不改动。`python3 -B research_A_horizon/code/check_native_cpu.py`与check_cpu.py通过。r09/r10尚未启动，后续GPU应使用code-r12并记录该版本；Qwen资产进程12459仍使用code-r11的未变化tokenizer构建路径。

r09/r10接续：`bash configs/launch-r09-r10.sh`已在远端由nohup启动PID12998，日志runs/queue-r09-r10.log。等待资产12459终态并检查COMPLETE，顺序执行code-r12/run_group.py；两组任一失败停止。不要再次执行该脚本；先检查进程及既有session。T1/all-runs.tsv新增code_snapshot追踪实际执行目录。

Qwen首轮校准计划已在远端configs/qwen-calibration-r11.json，尚未运行。两格分别48GiB λ10/24GiB λ5；仅在资产校验完成后用code-r12/run_group.py执行，并按两预算各自的抢占占比继续校准，不能将初始λ当冻结压力点。

T1排版补充材料：先运行`python3 -B research_A_horizon/figures/T1/build.py`，再运行`python3 -B research_A_horizon/figures/T1/latex.py`，随后对`research_A_horizon/paper/supplement.tex`运行两次pdflatex。当前生成五张表/六页，25格全部保留，空缺为破折号；未池化运行。PDF位于paper/supplement.pdf，六页已渲染目视核对。新增实验后需重新生成表与PDF。

F2初版选择固定为r03/r04/r05/r06/r06b，脚本已核对W2、OLMoE固定revision、相同prompt token SHA、legacy warmup及code/code-r05目录。data.json明确ledger_version=original_before_inflight_growth_fix；code-r12不得自动并入旧曲线。F2/F5已重建，仍13个原始点，不改指标或历史判决。

下载尾段已切换为`finish_qwen_ranges.py --root /root/autodl-tmp/research_A_horizon/models --asset-pid 12459`，进程18225。旧9620已终止并留收据，12459暂挂起等待该脚本的终态恢复，12998仍等待12459。先核实这三个进程与modelscope-download.json；不要再启动9620对应旧命令。分段完整SHA校验后才发布blob。

Qwen资产已完成：runs/qwen-assets.tar.gz（1,627,010B，SHA256 `1de5b7308490c70e6d20faa812b5847691e92033abf5ef2df3597318a9acf535`）已取回核验；configs/qwen-w1.json、qwen-w2.json、qwen-w3.json及models下对应收据已本地解包，输入SHA逐一匹配qwen-assets.json。下载18225/资产12459已结束；r09 PID19899等待公共锁，12998将在其成功后执行r10。

Qwen校准接续队列PID20361，日志runs/queue-qwen-r11.log；脚本configs/launch-qwen-r11.sh等待既有12998并检查r09/r10成功，不重复提交。输入长度统计与六份源SHA见configs/workload-summary.json。

校准分析按预算隔离；以下命令只分析本地已归档数据，不启动GPU：

```sh
python3 -B research_A_horizon/code/calibrate.py --kv-gib 24 research_A_horizon/runs/load-transfer-r08
python3 -B research_A_horizon/code/calibrate.py --kv-gib 48 research_A_horizon/runs/w2-48-native-calibration-r02 research_A_horizon/runs/w2-48-calibration-policy-r03
```

r09归档并分析后，将其目录加到第一条命令。Qwen r11归档并分析后，对其目录分别执行`--kv-gib 24`和`--kv-gib 48`；不混用OLMoE数据。脚本核对模型revision、prompt token SHA与KV预算；按cell名称关联指标，未测格单列，重复session拒绝，不能以缺失指标代替零抢占。

扩展前沿配置生成示例（仅生成；输出文件已存在时拒绝覆盖）：

```sh
python3 -B research_A_horizon/code/build_frontier_plan.py \
  --calibration-session research_A_horizon/runs/w2-48-calibration-policy-r03 \
  --calibration-cell 0 --inputs research_A_horizon/configs/w2.json \
  --remote-inputs /root/autodl-tmp/research_A_horizon/configs/w2.json \
  --session /root/autodl-tmp/research_A_horizon/runs/olmoe48-w2-corrected-frontier \
  --caps 128 160 192 --mc-policy mc-load \
  --output research_A_horizon/configs/olmoe48-w2-corrected-frontier.json
```

W1/W3替换输入与session名，保留同一W2校准来源和cap；Qwen与24GiB必须使用各自通过5–15%门槛的校准格。三个olmoe48配置已本地生成但未提交；启动前先检查r09/r10/r11既有队列及冻结的实际代码版本。MC LOAD为组件适配，不能标注为完整MC-SF。

r09已归档：runs/w2-24-calibration-r09.raw.tar.gz，4,208,763B，SHA256 `6515045ac0d015828d90a41b3b80a919ccf853bd3f804dfc79c3531432380d29`。`python3 -B research_A_horizon/code/analyze.py --session research_A_horizon/runs/w2-24-calibration-r09`重建metrics；按24GiB校准命令加入该目录得到λ5/高档λ10。code-r12首次ACK观测在28/28恢复episode有效。24GiB三类负载前沿配置已本地生成，未提交。

F1完整ACK链：`MPLCONFIGDIR=/private/tmp/horizon-matplotlib XDG_CACHE_HOME=/private/tmp/horizon-cache /private/tmp/format_sched_plot_venv/bin/python -B research_A_horizon/figures/F1/complete_chain.py`；输入r09 metrics，输出complete-chain-data.json/PDF/PNG，六个不重叠阶段逐episode核对总和。此图仅描述native24GiB λ5的28个episode，不与旧48GiB策略episode拼接。

r10完整归档：runs/w2-48-ablations-r10.raw.tar.gz，36,025,684B，SHA256 `2c19d831df625ee498e31742a93b477124bd2c49f551b4441c20c583cadc4d0e`。先`python3 -B research_A_horizon/code/analyze.py --session research_A_horizon/runs/w2-48-ablations-r10`，再用绘图venv执行`research_A_horizon/figures/F4/plot.py`；8格数据进入F4/data.json，前4格组成组件图。Qwen r11控制器29079已启动，不重复提交。

OLMoE24前沿接续PID30031：configs/launch-olmoe24-frontier.sh，日志runs/queue-olmoe24-frontier.log，等待Qwen29079成功结束后执行code-r12/run_group.py与olmoe24-w2-corrected-frontier.json。不重复启动；先核对进程和目标session。T1已纳入34格，supplement.pdf重新生成并完成六页检查。

r10固定prompt类别分解：`python3 -B research_A_horizon/figures/F4/request_classes.py`，读取八格raw与metrics，核对外生请求身份及分类人口和总计，输出F4/request-classes.json/tsv。后验描述，不将请求数当运行级重复。

修正版十臂前沿重绘：用绘图venv执行`research_A_horizon/figures/F2/plot_corrected.py --plan research_A_horizon/configs/olmoe24-w2-corrected-frontier.json --code-snapshot code-r12`。先用analyze.py生成该session metrics。脚本核对十臂计划、执行snapshot、模型/采样/输出配置、prompt tokens与完整结束；每组独立输出F2/<session>.json/PDF/PNG，不修改原F2.pdf。缺失时仅输出未测状态，不生成占位图。同入口适用于后续各模型/预算/负载的十臂配置。

### OLMoE矩阵接续（已提交，勿重复执行）

远端PID31718运行 `code/continue_olmoe_matrix.py`，执行器仍为 `code-r12/run_group.py`。脚本与五组配置SHA在 `configs/olmoe-matrix-continuation-sha.json`。接续顺序及门槛见RESULTS最新节；每次先查询存活PID、日志与实际receipt，不按本段重提。归档下载并在本地重新计算SHA256与字节数后，将这两个实际匹配值写入对应远端session的 `local-archive-verified.json`，格式为 `{"sha256":"<verified hash>","bytes":<verified bytes>}`。未经本地核验不得写确认文件。确认前可按原规则清理已核验组的可再生cache与已归档raw；下一组仍会重新检查磁盘门槛。

本地自动归档入口为 `python3 -B code/archive_matrix.py --socket /private/tmp/horizon-matrix-ssh`（当前54246已运行，不重复启动）。需要预先认证的SSH复用连接；脚本不持有密码、不提交GPU任务，仅处理源码中显式列出的七个session。终态STOPPED或任一传输/校验/分析错误会停止；核实原进程终态和错误后才能人工接续。已下载并核验的归档不会重复下载，解包不会覆盖不同已有证据。远端清理仅在本地核验和分析成功后执行，保留配置、收据、日志和压缩归档。

### 修正版F2六子图

待六组数据全部本地归档并生成metrics后运行：

```sh
MPLCONFIGDIR=/private/tmp/horizon-matplotlib XDG_CACHE_HOME=/private/tmp/horizon-cache /private/tmp/format_sched_plot_venv/bin/python figures/F2/plot_matrix.py --plans configs/olmoe48-w1-corrected-frontier.json configs/olmoe48-w2-corrected-frontier.json configs/olmoe48-w3-corrected-frontier.json configs/olmoe24-w1-corrected-frontier.json configs/olmoe24-w2-corrected-frontier.json configs/olmoe24-w3-corrected-frontier.json --output figures/F2/olmoe-corrected-matrix
```

在research_A_horizon目录运行。数据不齐时仅输出缺失状态JSON；完整时输出PDF/PNG，随后必须目视检查。Qwen六组配置就绪后可用同入口输出到figures/F6，不能替换为OLMoE数据。

### r11候锁失败后的有效接续入口

旧r11为零格候锁超时，归档保留，旧队列与归档进程均已退出。当前远端34979运行 `code/continue_matrix_r11b.py`，子控制器34980执行 `code-r12/run_group.py --plan configs/qwen-calibration-r11b.json --wait-lock-seconds 21600`；后续按configs/matrix-r11b-sha.json顺序逐组运行，必须先完成本地归档确认。当前本地60690运行 `code/archive_matrix.py --socket /private/tmp/horizon-matrix-ssh --qwen-session qwen-calibration-r11b`。这些均已提交，不能直接再次执行。新session仅改变候锁窗口及输出路径，没有改变测量配置。

新版 F3 使用 `python figures/F3/plot_corrected.py --w1-plan <W1计划> --w2-plan <W2计划> --high-plan <prepare_transfer生成的双倍到达率计划> --w3-plan <W3计划> --code-snapshot code-r13 --output <新输出基名>`。读取选择记录并验证来源SHA、同模型/预算/代码版本、固定策略参数和双倍到达率，输出JSON；只有四种条件数据齐全且前沿组完成才输出PDF和PNG。W2仍是选择数据，各点为单次探索测量。PDF/PNG导出已用标明合成数据的临时预览完成版式检查；真实数据读取及最终图像核验尚待实测，不能用旧版本图替代。
