# 新主机迁移补充记录

当前：已按用户指示切回weste:23478，同一六项全部完成并回传，192/192请求；MEASUREMENT_ONLY。最新结果见[RESULTS_WESTE_ADDENDUM.md](RESULTS_WESTE_ADDENDUM.md)，旧迁移/暂存经过保留。

初次记录时间：2026-09-13T07:23:12.971421+00:00。状态：`HOST_MIGRATION / ENVIRONMENT_UNVERIFIED / SIX_CELLS_UNRUN`。

本轮执行者已使用用户提供的新主机成功登录，正在恢复运行环境；尚未启动 GPU 实验。此次本地补充只记录迁移和资格边界，没有执行远端命令。

## 主机变化

| 项目 | 本轮记录 |
|---|---|
| 新端点 | `root@connect.westc.seetacloud.com:53036` |
| GPU | RTX 5090；`GPU-70fa1c0a-77d4-c14a-9daf-7e685874eef9` |
| NVIDIA driver | `595.71.05` |
| 基础 Python | `3.12.3`；`/root/miniconda3/bin/python` |
| 主机资源提示 | 25 CPU allocation / 90 GB banner；初始裸数据盘可用约 50 GB |
| 初始 GPU 进程 | 主机检查记录为空；每格初始化及正式测量前仍由冻结 runner 检查 |
| 旧端点 | `connect.weste.seetacloud.com:23478` 当前 SSH connection closed |

CPU allocation/banner 与 `torch.get_num_threads()` 是不同字段；新环境实际线程数尚未验收，不能由 25 CPU allocation 推导。新主机状态入口见 [bootstrap 状态](bootstrap-westc-53036/STATUS.json)。上述硬件与登录信息来自本轮执行者报告及该状态记录；本补充没有独立远端复测。

## 冻结包与历史阻塞

本地重新计算确认 `preparation/execution.tar.gz` SHA256 仍为：

`fd1342a6aea8f2cc886a0a6957d4059dcc2d3ffad1797b0428fb23d8be162b4c`

冻结 source 文件及输入文件亦与 `preparation/status.json` 中记录的 SHA256 一致。本补充没有修改冻结 package、source、raw 或原 REPORT；六格顺序及单一 `rotation_victim_order` 干预保持冻结。

[UPLOAD_REVIEW_RECORD.json](UPLOAD_REVIEW_RECORD.json) 中 `BLOCKED_APPROVAL` 是此前对旧目的地、新六项包上传执行尝试的历史拒绝记录，当时 uploaded=false、gpu_executions=0。该记录保留；它描述旧尝试及旧目标，不代表新主机已有上传、环境验收或实验结果。新主机迁移按本轮用户指示推进环境恢复，当前没有六格 GPU 执行证据。

## 新环境验收仍未完成

目标恢复为 Torch `2.11.0+cu130`、vLLM `0.26.0`、Transformers `5.15.1`，OLMoE 模型与 tokenizer revision 固定为 `6d84c48581ece794365f2b8e9cfb043c68ade9c5`。环境安装或下载完成均不能替代下列实际 preflight 与 KV 检查。

当前执行 driver `experiments/admission_capacity/run_frozen_kv_remote.py` 第 41 行固定使用 `/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python`；第 54–70 行检查指定 snapshot 的每个模型 shard 内容 SHA256 等于其 resolved blob 文件名、给定 GPU UUID 存在、无计算进程以及上述三个库的精确版本。`/root/miniconda3/bin/python` 仅作为新虚拟环境的基础解释器。

以下模型摘要来自旧八格已通过的 preflight，供同一固定 revision 的新下载核对；新主机尚未验证这些 bytes/hash：

| 模型 shard | bytes | SHA256 |
|---|---:|---|
| model-00001-of-00003.safetensors | 4997744872 | `5e3cff7e367794685c241169072c940d200918617d5e2813f1c387dff52d845e` |
| model-00002-of-00003.safetensors | 4997235176 | `15ef5c730ee3cfed7199498788cd2faf337203fc74b529625e7502cdd759f4a7` |
| model-00003-of-00003.safetensors | 3843741912 | `a9abac4ac1b55c9adabac721a02fa39971f103eea9a65c310972b1246de76e04` |

旧摘要来源：[victim-order preflight](../20260913_rotation_victim_order_r01/execution/preflight.json)。当前 driver 是按 snapshot index 和 resolved blob 文件名校验，源码没有硬编码上表三个摘要。

冻结 runner 另要求单张可见 GPU/vLLM0.26（`preparation/source/run_probe.py:98`），scheduler.py 精确源码 SHA256 `2ed2a550b6558b2495eda845a97ae38bcf0225027b9e25fbf00fc3880c1d3941`（`rotation_native.py:17,77–81`），实际 KV 16,089,350,144 bytes、7,671 usable blocks 与 full reservation（`run_probe.py:110–123`）。这些在新主机均尚未完成验收。CPU 线程及库版本会在每格 environment.json 记录；driver、CPU allocation 与 GPU UUID 的迁移事实由本补充及后续执行记录保留。

## 比较边界

六格都在新主机、同一恢复后的环境上按冻结顺序重新执行：block0 A/C/B，block1 B/C/A。A=least_progress，C=first_most_then_least，B=most_output。只有六格实际完成、资格成立后，才进行同 block 三臂配对和同角色跨 block 观察。

旧卡八格用于提出此次问题和核对固定输入/模型身份，不作为新主机 A/B/C 的直接性能基线；新旧 GPU UUID、driver、CPU配置及主机环境差异不计为排序收益。当前新六格仍为 `UNRUN`，新环境仍为 `UNVERIFIED`。

## 本轮独立环境检查补充

2026-09-13 UTC 07:24–07:26，新主机实际导入得到 Torch 2.11.0+cu130、vLLM 0.26.0、Transformers 5.15.1；scheduler.py SHA 与上文冻结值相同。模型分片仍由已有下载 PID2091 获取，本会话没有启动第二次安装或下载；本会话 bootstrap 因目标 venv 已存在而在创建子进程前停止。

宿主可见208逻辑CPU、affinity 0–207；cgroup cpu.max=2500000/100000（25核配额），memory.max=96,636,764,160 bytes；PyTorch默认 intra/inter-op 均104线程。冻结 launcher 不新增线程配置。上述默认值不代替各实际cell的环境记录；结果须在同新主机内比较。GPU检查为2MiB占用、无计算进程。模型内容、实际KV和GPU执行仍待完成。当前状态见 bootstrap-westc-53036/STATUS.json。

## 当前接续状态（2026-09-13T07:58:25.792515+00:00）

**BLOCKED_APPROVAL / 六项 UNRUN。** 固定软件版本、scheduler摘要、三个模型shard与旧实验的字节数/SHA256均相同；tokenizer可离线加载。见[完整预检](bootstrap-westc-53036/model_and_runtime_preflight.json)和[就绪/清理记录](bootstrap-westc-53036/readiness_and_cleanup.json)。最后一次检查GPU已空闲；其后上传执行在本地进程启动前被自动审批拒绝，具体理由见[本次新端点拒绝记录](UPLOAD_REVIEW_RECORD_WESTC_53036.json)。六项execution/analysis目录不存在，上传和GPU执行均0。旧端点拒绝记录保持原样。

第二个公开模型分片采用只读前缀复用和独立分段补齐；一次分段超时保留，针对性修复后整片SHA与旧实验一致，再以不覆盖方式补入blob和snapshot链接。原模型下载和外部GPU任务均未终止；本会话冗余临时文件已清理，完整模型缓存和失败/修复记录保留。额外tokenizer检查曾误要求长度等于模型padding后的词表50304；修正为实际50280个token的ID范围0–50279位于模型嵌入范围内，未修改模型或tokenizer。密码未写文件。

CPU路径分析补充见[入口和边界](diagnostics/README.md)：按source ID连接首次有效选择、恢复链、早四/末二完成变化及width1/2/4账本。一遍旧V八格兼容核对逐项一致；F仍0性能配对，C只有CPU选择夹具验证。

| 固定报告项 | 本轮结果 |
|---|---|
| Verdict | BLOCKED_APPROVAL / UNRUN；主问题OPEN |
| Evidence type | 新主机CPU/模型文件预检；旧GPU数据的CPU兼容核对 |
| What was measured | 固定软件/源码/模型一致性，GPU占用变化，分析入口兼容性 |
| What was not measured | 新主机六项请求性能、实际KV资格及C动作效果 |
| Strongest baseline | 同新主机重新执行A least_progress和B most_output；旧卡结果不替代 |
| Oracle/headroom status | 未新增Oracle或暂停硬界 |
| Claim ceiling | 迁移底座已准备；新C性能未知 |
| Failure category | 自动审批不接受新包/新目的地范围；非科学失败 |
| Resurrection condition | 原问题未判死；本具体上传执行获准且GPU重新预检通过后接续 |
| One next smallest experiment | 冻结A/C/B、B/C/A六项，共192次正式请求，逐格回传 |

本轮尚不能回答“首次交换是否足以保留吞吐改善并减少早完成请求损害”；所需的六项实验未运行。没有启动后台等待或自动执行任务。

接续命令如下。控制连接如已过期，先重新建立用户提供的SSH连接；不要将密码写入命令文件。

```bash
python3 -B refine-logs/expert_saturation/experiments/admission_capacity/run_frozen_kv_remote.py \
  --source refine-logs/expert_saturation/outputs/admission_capacity/20260913_rotation_first_swap_r01/preparation \
  --output refine-logs/expert_saturation/outputs/admission_capacity/20260913_rotation_first_swap_r01/execution \
  --host root@connect.westc.seetacloud.com --port 53036 \
  --control-path /private/tmp/moe-westc-53036.sock \
  --gpu-uuid GPU-70fa1c0a-77d4-c14a-9daf-7e685874eef9 \
  --remote-dir /root/autodl-tmp/moe-rotation-first-swap-20260913-r01 \
  --labels cohort0-block0-least_progress cohort0-block0-first_most_then_least cohort0-block0-most_output \
           cohort0-block1-most_output cohort0-block1-first_most_then_least cohort0-block1-least_progress
```

该命令当前因上述自动审批拒绝而尚未执行；不能换路径、拆命令或复用远端输入绕过。

## 用户明确授权后的上传（2026-09-13）

用户已明确允许本六项包上传至connect.westc.seetacloud.com:53036并执行。自动审批接受本次暂存上传，driver退出0；[execution/execution.json](execution/execution.json)为STAGED、cells为空，远端目录为`/root/autodl-tmp/moe-rotation-first-swap-20260913-r01`，archive SHA256与冻结包一致。历史拒绝记录保留，不再是当前阻塞。

现有GPU进程PID5127属于另一Qwen3 r02初始化任务，未干预。当前启动了一次有界15分钟的只读空闲监测；查询失败停止，GPU空闲才调用同参数`--resume-staged`，不自动重跑已开始的cell，最终状态以[监测记录](wait_then_resume_status.json)与execution记录为准。

外部上传驱动增加`--stage-only/--resume-staged`，不修改冻结实验包或GPU代码：暂存允许忙卡，接续重查软件/模型/GPU、同一主机/路径/顺序、local/remote archive与解包文件一致性；已有cell活动时拒绝重跑。12组mocked orchestration检查通过，真实本次stage已成功；检查范围见[driver检查](driver_staging_checks/checks.json)。

现有接续命令应保留前文全部参数并增加`--resume-staged`；不要对已存在execution目录再运行默认上传模式。结果形成后使用既有主分析入口与diagnostics/diagnose_paths.py。

## 本轮结束时的准确状态（2026-09-13T08:33:42.214616+00:00）

上传完成，冻结archive SHA与远端再次核对一致；[post_wait_remote_check.json](post_wait_remote_check.json)确认远端results为空。15分钟有界监测的30次GPU检查均遇PID5127，占用约3650MiB，未调用六项执行入口。监测退出2并结束，未终止外部Qwen3任务；本轮没有后台自动等待或执行。执行记录保持STAGED以便安全接续，实验实际为0/6。当前明确授权仍覆盖这份包及六项执行，后续无需重复询问授权。

[直接接续命令](RESUME_COMMAND.sh)保留同一source/output/host/port/GPU/远端目录与顺序，只使用`--resume-staged`；启动时重新验证包与GPU状态。不要改为默认上传模式，也不要重复执行任何已启动cell。

| 固定报告项 | 本轮结束状态 |
|---|---|
| Verdict | BLOCKED_GPU / STAGED；六项UNRUN，主问题OPEN |
| Evidence type | 真实SSH/SCP上传与内容核验；CPU编排检查；GPU占用观察 |
| What was measured | 原包上传成功及哈希一致；固定模型/软件资格；30次忙卡观察 |
| What was not measured | 六项完整请求、实际KV资格、C首次交换作用及性能 |
| Strongest baseline | 计划同新主机重跑A least_progress、B most_output；均未执行 |
| Oracle/headroom status | 没有新增Oracle或硬界 |
| Claim ceiling | 操作就绪与资源阻塞；不能推导新机制效果 |
| Failure category | 外部GPU占用；不是审批阻塞或科学NO-GO |
| Resurrection condition | GPU释放并通过冻结runner的再次预检 |
| One next smallest experiment | 从已上传包接续同一六项，不改输入、参数或顺序 |

“首次交换是否足以保留吞吐改善并减少早完成请求损害”仍未被本六项回答。上传或CPU检查不替代GPU实测。

## 用户指示切回 weste:23478（2026-09-13T09:43:02.707856+00:00）

用户已给出该端点的新连接凭据并要求切换，授权继续同一份六项包。新连接成功；GPU为`GPU-0a66cc34-b091-2000-ba7b-e576b6d3d7d6`，RTX5090、driver595.71.05，检查时2MiB且无计算进程。Torch2.11.0+cu130、vLLM0.26.0、Transformers5.15.1在位；OLMoE固定revision三个模型文件字节数相符，内容哈希及实际KV资格由执行预检核实。数据盘可用11GB。

包SHA仍为`fd1342a6aea8f2cc886a0a6957d4059dcc2d3ffad1797b0428fb23d8be162b4c`。全新本地`execution02_weste_23478`接收该主机的六项；远端同名目标目录此前不存在。旧`execution`仍是westc暂存0项，保持原样；没有终止或修改其Qwen3进程。固定A/C/B、B/C/A、输入与参数不变，六项全部在weste执行后才能比较。

命令见[RUN_WESTE_COMMAND.sh](RUN_WESTE_COMMAND.sh)。这是一次新的目标主机执行，使用全新输出目录，不对westc的STAGED记录使用resume。

## weste执行完成（2026-09-13T10:07:42.283855+00:00）

execution02_weste_23478/execution.json为COMPLETE，六项均READ_BACK/returncode0，192/192请求、196608新输出；主分析和路径分析均MEASUREMENT_ONLY、六项资格成立。完整性fresh same-family/provisional复核PASS，P0/P1=0、独立数值差异0。完整请求权衡、尾段模型和下一项见[结果补充](RESULTS_WESTE_ADDENDUM.md)。本次driver和分析进程已结束，无后台GPU任务；结束时短暂观察到其他PID6513，其后ps检查已不存在，未做干预。原始raw、冻结包、旧westc暂存记录均保留。
