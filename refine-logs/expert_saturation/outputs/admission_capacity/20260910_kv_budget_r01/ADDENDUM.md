# 20260910 新端点迁移：科学方案不变，硬件重新资格化

本目录是独立 attempt `20260910_kv_budget_r01`。本地迁移未连接远端、未读取或保存凭据，
未上传、未启动 GPU 进程。科学结论 UNRUN。旧 r01 的资源失败与 r02 的未执行状态
仅为来源历史；旧 execution/resource 状态、日志、分析和 raw 未复制为本次证据。

继承来源为 [20260908_kv_budget_r02](../20260908_kv_budget_r02/COMMANDS.md)。
七份科学 Python 模块、两份来源 patch、short/long 的全部输入与配置逐字继承。
执行归档与 SHA256 文件逐字继承；归档内不含 SSH 端点或 attempt 远端根路径，
因此无需重打包。归档内两份文档保留原日期历史，本目录文档记录本次执行条件。
本地 stage/driver 只改端点、根目录与日期标签；分析脚本使用动态 HERE，逐字继承即可
读取本目录，仍依赖已有 native-preemption 分析 helper，不复制历史测量结果。

新目标为 `root@connect.weste.seetacloud.com:11155`；远端根为
`/root/autodl-tmp/moe-kv-budget-20260910-r01`。执行者应记录本次实际 GPU 型号、UUID、
显存、driver/CUDA 与 Python/PyTorch/Transformers/vLLM 版本及计算进程隔离状态。
目标硬件为 RTX 5090，软件预期为 vLLM 0.26；旧 GPU 身份和资格不自动迁移。
暂定 Python 为 `/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python`。
冻结 launcher 使用 GPU 0，并以离线方式从 `/root/autodl-tmp/hf-cache/hub` 读取模型：
`allenai/OLMoE-1B-7B-0924`，模型与 tokenizer revision 均为
`6d84c48581ece794365f2b8e9cfb043c68ade9c5`，BF16。
若 GPU、Python、版本或模型缓存路径不同，由执行者先记录并处理环境差异，
不得把旧资格写为通过，也不得静默修改冻结模型、输入或引擎配置。

固定四项 95→90→90→95、完整输入、三次暖机、原生抢占恢复和请求会计均不变。
每项仍有现成 live KV 资格检查；0.95 未达到足量驻留条件则保留记录并停止，测量 UNRUN。
0.90 记录实际池容量，不能假设在新硬件一定复现旧抢占。全部 cell 使用同一本次 GPU，
每个 cell 完整回传后才继续。新硬件差异限制与旧 campaign 的跨运行比较；本次主比较
始终是本次同 GPU 的 0.95 与 0.90。
# 新实例现场状态与上传边界

2026-09-10 已连接用户新提供的 weste:11155：RTX 5090，32607 MiB，
UUID `GPU-e4f548ee-bb0c-664b-b2ce-3a006789e189`，驱动 595.71.05；初查无 GPU 计算进程。
现场存在安装 vLLM 0.26.0 / Transformers 5.15.1 的 PID 1739，随后检查仍存活；
固定模型 revision 的配置与 tokenizer 已出现，权重仍在下载。软件栈和完整权重尚未合格。
只读检查保留在 `SETUP_OBSERVATIONS.jsonl`。未读取安装日志内容。

两次上传请求均在本地自动审批阶段被拒绝，stage 脚本未执行；未上传或启动本次 GPU 实验。
已核对冻结包 15 项内容、归档 SHA256 和不含所提供凭据；审批仍要求当前对话明确授权
将这份代码与输入传至指定端点，已向用户提出一次具体授权请求。原始科学状态保持 UNRUN。

后续：用户已在当前对话明确授权上传、四项运行及全部回传。stage 已完成，
`UPLOAD_20260910.json` 保存远端归档校验，SHA256 与原冻结包一致。
再次检查时，原安装 PID 1739 已不存在且所需包未安装；新建 `setup_runtime.py`
仅修复专用虚拟环境，安装固定 vLLM 0.26.0、Transformers 5.15.1、PyTorch 2.11.0。
新安装命令、PID 和日志单独保留，不修改归档内科学代码。权重下载与运行资格仍待完成，
四项 GPU 测量尚未启动；此前上传审批阻塞已解除。

原PyPI安装尝试实际存活并缓慢推进，约10分钟仍在依赖元数据阶段。对公开索引的小范围
测速及精确wheel可用性核对后，只终止了本次自建安装PID4284；原launcher记录EXITED/-15。
命令及全部日志已归档回传为 `setup-runtime-first.tar.gz`，校验记录在同名readback JSON。
后续使用相同三个版本的镜像安装，launcher4833/child4834；现有模型下载PID2025保持不变。

已启动本地后台runner21199，等待镜像安装成功、完整权重SHA256及实际GPU资格；之后自动
运行已授权的四项对照与逐项回传，再分析一次。它只检查已有下载句柄，不启动替代下载，
失败则停止诊断。该启动是执行安排，当前没有GPU测量结果；状态和进程已现场核实为WAITING。
