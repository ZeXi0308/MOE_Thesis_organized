# 本轮结果与决策（2026-09-29）

**Verdict：`CPU_COMPONENT_CHECKED / GPU_UNRUN / METHOD_UNPROVEN`。** 主问题仍 OPEN。H1 是提交时跳过原计划 victim 的单自由度简单规则，尚未证明自然域中存在同时满足 KV、序列槽、connector 和块归属的实例，更未证明完整请求收益或 MoE 特有性。

**2026-09-30 更新：** 用户要求尝试的新 SSH 地址已在允许的网络环境完成认证；[只读收据](REMOTE_READ_ONLY_20260930.json)记录单张 RTX 5090 与 90 GiB cgroup host 硬限的现场快照。此前本地沙箱 DNS 失败仅限该沙箱。未获 GPU/host 占用及总时间/费用的明确范围，未上传或运行任何 GPU 格；下文 2026-09-29 的“未连接”仅是当时状态。

## 实际完成

- 建立 [统一任务/接口](README.md) 和 [最小模型](MODEL.md)：唯一假说、online state/action、B/C 独占交付、原生/强简单/LTR-style/新机制的串行顺序与停止条件。root 独占 `CURRENT_EXPERIMENT.json` 和共享 `RESULT_LEDGER.md`；B/C 无 GPU 或主台账写权。
- B 的 [只读证据](b/EVIDENCE.md) 与 `recompute_checkout.py` 重算旧 D859/E1399 `free−need=98/171` 块。隔离检出恢复的旧 `selected/full.json` 还记录 running 26/22，上限 32，因此保存 CPU 快照有 6/10 个空闲序列槽；两组各 25 个 commit 仅各 1 个余额动作差异。又恢复并逐事件对齐 G 详细诊断的 2650 个 schedule 快照：54 个 READY commit **全部 `free<need`，差 2–193 块**，该轨迹 H1 的 KV 直接资助机会精确为 `0/54`。G/H 轻量格其余 126+690 次应用无逐 commit 资源快照，合法 H1 机会界由 `0..870` 收紧为 **`0..816`**。旧 D/E 原始 selective-store/raw、现场全块归属和完整 connector 生命周期仍未验证，不能定为已执行 direct。旧 victim 之后 gap 0.112/0.261 秒且非各轨迹最大值，不能扣成净收益。
- C 交付 `c/commit_recheck.patch`，root 复核清洁基底 SHA `d1002357…e63697` 后顺序应用到共同 `staged_store_rotation.py`，补强流式请求占槽门禁后的 SHA `8dc692e0…a50be31`。`install(...,commit_recheck=False)` 保留旧目标/victim选择和提交路径，另与 on 臂共同增加请求身份和未知 victim job 的安全取消；on 在旧 `commit_reason=READY` 后即时验 KV、**含 streaming 保留的空闲序列槽**、队列、target pending 和物理块独占。通过则同一目标不强制抢占原 victim，失败回退旧 commit，过期取消。目标首新输出/终止仍由反馈解除保护。已接受 F/G/H 包的 adapter 是另一版本；C 另交 `c/accepted_gh_commit_recheck.patch`，root 仅应用于独立候选包。两处集成字节均与从准确基底应用最新补丁的结果相同。
- 共同和 G/H 精确版各 4 项定向 CPU 测试通过，覆盖 direct/default-off、普通或 streaming 占满槽/字段未知、缺块回退、共享或部分 KV、在途 load/store、状态过期、EOS/终止及取消。共同源码和 `evaluate_goodput.py` 编译通过。`evaluate_goodput.py` 的合成两请求 smoke 通过，确认失败请求留在分母且未完成 flow 罚时；**未用它分析真实 raw**。
- 静态包兼容门禁重新逐文件验证已接受 H r02 与 H1 候选各自 manifest，H128 `workload.json` 文件 SHA 均为 `dd8ac656…`、规范化 SHA 均为 `606f71fd…`，模型、资源配置与六项共同后端及四个 warmup 文件字节相同，结果 `COMPATIBLE_INPUT_AND_BACKEND`。此检查不验证运行机器、原生生命周期或动作收益；LTR r02 仍是 G64 输入，不能直接与 H128 排性能。
- 另一协作进程新增 LTR-style 共同后端 CPU 基线文件及纯 CPU 动作模型，root 未改其未合并文件；只读复核分别 25/8 项测试通过。LTR-style 可作为新的独立包候选，不能代替已接受 r02 的原生诊断或称完整 LTR。B 模型当前不接 runtime 快照；运行时以已集成的单一 C gate 为唯一提交裁决，避免叠加控制器。其三点 goodput 诊断阈值不用于正式全臂比较，正式阈值按本目录 [串行合同](RUN_PLAN.md) 统一。
- [LTR 同输入审查](ltr_audit/REPORT.md)核定已接受 r02 只包含 G64/T30/Q10 诊断；H128 输入能由其 loader 在 CPU 上核身份与逻辑 SHA，底层测量/warmup 字节兼容。H128 性能须在 G64 实际原生生命周期资格后另建新包：LTR 三个策略模块可保持原字节，输入、安全上界、performance runner、资源入口与 manifest 需独立冻结。当前仅有审查，未构建/运行 H128 LTR 包。

本地 sparse partial clone 最初使两个旧检查缺 `20260914_recovery_progress_model_r01/input.json`，在读取前态时停止。随后从同一远端 HEAD `76d6d888…` 恢复到隔离临时检出；r02 的 manifest 所列 **30/30 文件 SHA-256 均匹配**，H 已接受包 **35/35** 匹配。原 tar 包未取得，故归档字节 SHA 只保留历史接受记录。使用准确 r02 包及当前共同 CPU harness 重跑 `check_lifecycle.py` 得 `PASS_CPU_ONLY`，包括原生调度循环和 adapter closure 模拟；没有实际 load/tensor/GPU/EOS。旧 `check_staged_rotation_closures.py` 的 AST fixture 补入默认关闭标志后，在隔离测试树用恢复的 `input.json` 重跑通过五个旧分支。`check_staged_save_contract.py` 还依赖未跟踪 D6 `raw.json`，仍未重跑；G/H 请求级 raw 亦缺席。新 SSH 入口尚未连接，GPU/host 当前状态 UNKNOWN，用户允许的 GPU/host/时长或费用预算仍待明确；没有上传、初始化、GPU 请求或后台作业。

为保留 r02 的 G64 单格诊断而移除旧机器写死的入口，root 新建 [环境入口候选](candidate_ltr_r02_env/README.md)。源 30/30 文件复验，候选的 25 个 `pkg/` 文件仅 `run.sh` 改动，其余 24 个含 runner/LTR 策略/输入和共同后端逐字不变；新 26/26 manifest、shell 语法与 Python 源码编译通过。入口要求明确 GPU UUID、同一共享锁、单格墙钟、离线模型和当前进程所属的有限 cgroup v2 host 硬限，仍须新机器现场验证与总预算批准。它不是原接受 archive 的运行，更不是 H128 LTR 性能包；GPU 仍 `UNRUN`。

root 又在两个冻结候选包**之外**加入 [整组串行执行封装](SERIAL_GROUP.md)：只接受当前两项固定 manifest 身份，单前台格、共同 fd 9 锁、显式批准总墙钟、格间 GPU compute process 检查与输出复制/SHA readback。四项本地 CPU 替身检查通过：正常单格归档；归档含 symlink 时 fail-closed、仍记录退出前 GPU 检查并释放本地锁；指向候选包内部的父目录 symlink 被拒绝；超时子进程收到有界终止。它仍不能自证用户授权、GPU 完全空闲、模型 revision 或异常 worker 已全部退出；这些要现场核。没有实际会话计划、连接或 GPU 运行，不增加科研结果。

从仓库根可复核本轮 CPU 结果：

```sh
python3 refine-logs/expert_saturation/experiments/admission_capacity/20260929_commit_recheck/b/recompute_checkout.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v refine-logs/expert_saturation/experiments/admission_capacity/20260929_commit_recheck/c/test_commit_recheck.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s refine-logs/expert_saturation/experiments/admission_capacity/20260929_model_action_value -p 'test_*.py' -v
```

LTR-style 25 项测试从 `refine-logs/expert_saturation/experiments/admission_capacity` 运行 `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_ltr_fair_policy test_ltr_fair_native test_ltr_baseline_compare`。后一组及动作模型属于未合并的另一进程文件；root 只读测试，未将其作为已接受 GPU 包。

## 判别与下一步

H1 在 CPU fixture 中做出了旧 selected/eager 不做的动作：在同一已接受恢复目标上，commit 当刻若当前 KV、序列槽和可见 connector 状态均允许，`V={planned victim}` 改成 `V=∅`。信息来自 scheduler 当前 running/waiting、block pool/owner、connector transfer jobs 与版本，不包含未来 EOS、真实剩余输出或动作后的 route。`DIRECT_READY` 只是原生准入的前置筛选；原生 running 步增长、调度 token 预算或 connector lookup 延期仍可能使 target 没有同次进展，且同次首新输出不保证早于 off。31 个 running / 1024 token / 994 token 目标的 [CPU 条件反例](H1_TOKEN_BUDGET_ADDENDUM.md)已使这一无条件预测撤销。C 的[异步收据补丁](c/INTEGRATION.md)已由 root 仅集成到候选 adapter：正调度 token 或匹配的 target load job 加实际物理块才增加 `direct_commits`；缺收据 fail-fast。新旧各 4 项定向 CPU 测试与候选 25/25 manifest 通过。该计数只代表原生**准入**，不代表 load 完成或首新输出。它**可能**保留 victim 的连续输出并避免后续加载/重算；旧 prepare 已登记的 store、batch 变化和未来 peer KV 压力必须实际计费，不能宣称已省时。

最强系统参照仍是原生 full/no-extra-rotation 的效率与多数请求 gap；selected/eager 是停顿优先简单参照；兼容 LTR-style 的原包 CPU 生命周期已复核，但 GPU 生命周期及公平校准均未运行。没有共同前态物理同一性的动作 Oracle；旧 D/E 是 KV 足额且有序列槽的保存 CPU 快照，尚缺真实 direct 反馈。G 详细诊断提供一个明确无动作的自然负控。C 对 G/H 精确字节的补丁已由 root 集成到独立 [H1 候选包](candidate_h1/README.md)：只变 adapter、单格 runner 和 shell，保留 21 个原 pkg 文件，25 文件候选 manifest 全验；精确后端和异步收据合计 8 项 CPU fixture、runner 输入/CLI smoke 通过。H 输入已见过，只用于开发，不能充当盲确认。证据上限是 `CPU_COMPONENT`，关键 GPU 和资源授权仍 `UNRUN`，不是机制或问题级 NO-GO。

唯一下一实验：在用户明确机器 GPU、host 和时间/费用范围且准确 payload 或新包身份可核后，由 root 独占现场资源，先跑一格 LTR-style native 生命周期诊断；通过后按 [RUN_PLAN.md](RUN_PLAN.md) 串行同资源 native、eager、校准 LTR-style、H1 off/on。若自然域无合法 direct，停于无动作空间；若强简单规则或 LTR 已覆盖增量，吸收修正为基线。不扩 cooldown/window/headroom/growth predictor，也不同时运行其它 controller。
