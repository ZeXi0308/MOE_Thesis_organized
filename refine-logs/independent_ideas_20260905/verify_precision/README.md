# Verify Precision: matched-budget selector qualification

状态：代码准备；真实实验 `UNRUN`。旧 H2 的离线 masking 无效。兄弟工作树
`毕设论文资料-longrun-B-resurrection-b141` 的纠正运行确实推进了独立 KV，
但 16 步 reactive 平均 served-high 为 4.5，fixed V 为 4.0，仍是预算不公平；
其权威裁决为 `INCONCLUSIVE_RUNTIME`。来源 HEAD、verdict/runner/config/threshold
文件 SHA 及 12 个历史 calibration/evaluation document hashes 记录在
`source_lock.json`，运行代码不依赖兄弟目录。没有复制或修改旧 artifact。

唯一下一实验：在 8 个新的完整 WikiText-103 articles 上，以相同 64-token prompt、
32-token teacher-forced continuation 比较 fixed-period、budget-capped reactive、
budget-matched random，反转顺序重复两轮。只沿用旧冻结阈值
`0.04084732383489609`，不重调阈值；32 步是明确的新运行域。

每策略每步都从自己的 canonical KV 做 H/L 分叉并提交所选 cache，下一步真实
重新执行。每策略恰好 4 served-high、32 physical-high + 32 physical-low calls。
fixed 每 8 步 serve high；random 在运行前抽取 4 步；reactive 先按阈值选择，
预算用完后停止 high，剩余步数等于剩余预算时补足。`forced_high_steps`、
`threshold_selected_high_steps`、`threshold_requested_high_steps` 和
`budget_blocked_high_steps` 单独记录，防止强制补足掩盖选择器行为。
这些数字描述实际 KV 历史，不能推断未执行的“不补足”反事实轨迹。

复用当前仓库 `triage_runtime.py::execute_same_state_step/clone_cache/
PreparedInt4ExpertBackend`，新增适配层只负责预算和账本。保留每步 raw JSONL，
异常不会覆盖已写数据。完整策略 wall time 包含双执行、KV clone、KL、决策、
提交和 journal；每个 H/L forward 单独计时。共同 prefill 与 all-high reference
成本独立记录，不冒充在线推理成本。相同 calls 不代表相同时间或能耗。

数据由已有 article parser 离线加载；以固定 offset/seed 取样，重复、旧文档或
长度不足直接失败，不替换成有利样本。freshness 只覆盖冻结的纠正 pilot、
calibration 和使用同文档的 repeat；不声称完成全仓历史数据排除。

```bash
.venv/bin/python refine-logs/independent_ideas_20260905/verify_precision/run_qualification.py \
  --prepare-only --split test --offset 0 --output-dir /tmp/verify-budget-prepare-unique

# 已有独占 CUDA GPU 上，复用准备目录；新输出目录不能已存在。
.venv/bin/python refine-logs/independent_ideas_20260905/verify_precision/run_qualification.py \
  --prepared-dir /tmp/verify-budget-prepare-unique --output-dir /tmp/verify-budget-run-unique

.venv/bin/python refine-logs/independent_ideas_20260905/verify_precision/analyze_qualification.py \
  /tmp/verify-budget-run-unique
.venv/bin/python -m unittest discover \
  -s refine-logs/independent_ideas_20260905/verify_precision -p 'test_*.py'
```

`--prepared-dir` 核验输入文件 SHA 与执行源码 SHA，直接复用冻结 token IDs 和参数，
不重新选文档或 tokenize。GPU 机器的软件版本可能不同，须另行预检；输入准备机的
版本保留在 config 中。汇总器读取全部三臂×文档×repeat，缺 run 为 PARTIAL，
预算或身份错误为 INVALID；每文档先平均重复，再给 median/正负计数，pooled 使用
ratio of sums，零分母返回 null，不加 epsilon。

当前简单基线为预声明 phase=0 的等预算 fixed-period，同时保留等预算 random。
最强 periodic phase 尚未通过独立 calibration 选择；`new_budget_oracle=UNRUN`，
旧 16 步 Oracle 不适用于新 32 步动作空间。因此正结果最多是 provisional selector
signal，不能称超越最强简单策略。正结果要求 reactive
在全部文档上保留原始 accumulated-KL、与两基线的配对差值及重复稳定性，再判断
是否有可重复的选择增量；大量 forced fill 必须一起报告。负结果只关闭当前
逐步阈值选择器；不能判死 precision verification 家族。

这不是旧 H4 escalation 的直接复现，也不测试 native INT4、部署收益、task quality、
TTFT/TPOT、serving 或 EP。低精度仍是常驻 INT4 QDQ 权重经 BF16 `F.linear` 执行；
所有策略每步都双执行，因此只是 matched-budget double-shadow 选择资格化。
CUDA 缺失时真实实验保持 UNRUN，CPU 仅用于预处理及 4 个定向工程测试。

本轮可直接复用的输入在 [prepared_r01](prepared_r01/COMPLETE.json)：8 个 articles，
每个 64 prompt tokens + 32 continuation tokens，2 repeats。已验证 prepared 输入逐字节
复用及 no-CUDA 返回 UNRUN；另通过 tiny OLMoE 的真实 CPU DynamicCache/QDQ 调用检查。
后者使用随机初始化小模型，只验证 API/状态推进，不是 pretrained 数值或性能结果。

```bash
.venv/bin/python refine-logs/independent_ideas_20260905/verify_precision/run_qualification.py \
  --prepared-dir refine-logs/independent_ideas_20260905/verify_precision/prepared_r01 \
  --output-dir /tmp/verify-budget-gpu-r01
```
