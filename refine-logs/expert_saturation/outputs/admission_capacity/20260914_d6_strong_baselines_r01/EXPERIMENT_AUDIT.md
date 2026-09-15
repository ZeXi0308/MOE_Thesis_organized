# 定向实验审阅

Date: 2026-09-14
Auditor: /root/d6_result_audit, GPT-5.6-Sol ultra, fresh read-only
review_independence: same-family
acceptance_status: provisional
Overall: PASS；P0/P1=0

范围：本目录协议、STATUS、REPORT、分析JSON、8格metrics/config/status/qualification/environment、运行日志、运行与策略模块，以及E/analyze_d6_strong_baselines.py、analyze_pressure_review.py、analyze_call_progress.py、analyze_completion_distribution.py。仅本轮A-F完整性检查，不复审历史、不开GPU、不创建.aris。审阅者独立读取文件；执行者提供路径和检查项。

## 审阅原答

定向审计结论：PASS（same-family provisional）。无 P0/P1。

- A GT provenance — PASS：这是实机性能测量，无任务 GT。请求来自固定 WikiText 数据集与哈希身份（config.json:17-37）；延迟、TTFT、TPOT、ITL均由实际时间戳推导（metrics.py:85-125）。跨臂输出相同数仅作诊断，报告明确不等同质量验证（REPORT.md:41）。
- B denominator/accounting — PASS：吞吐分母为首个到达到真实观察结束，平均完成为32个请求延迟算术均值，max-ITL覆盖全部请求（metrics.py:136-155）；百分比仅以指定基线原值为分母（analyze_d6_strong_baselines.py:62-69）。未发现自最大值/均值归一化或重复计时。8格完成面积恒等式独立重算误差均 <1e-10（定义见 analyze_completion_distribution.py:16-24）。
- C existence/number/status — PASS：8个结果目录及所需文件均存在；每格 COMPLETE/32，campaign各仅一次执行且 exit=0（readback/campaign.log:1-17）。主表8行、六组百分比/ITL差、逐请求快慢计数均从各 metrics.json 独立重算完全一致；256条 raw 输出定向复核得到六对24/22/27/24/22/27，匹配 analysis.json:705-969。results.tar.gz实算 SHA256 与 STATUS.json:137 一致。DECISIONS.md:1-17、资格文件 measurement_status=NOT_YET_RUN 是运行前快照；终态由各格 status.json:2-7 与 campaign 判定。STATUS.json:140 的 audit=IN_PROGRESS 在本审计返回前属实，封账时应更新即可。
- D actually-called metrics — PASS：主运行实际调用 summarize_episode_requests 并写出 metrics（run_probe.py:13,181-186）；call-progress被主分析调用（analyze_pressure_review.py:8,38）；completion分析生成现存产物（analyze_completion_distribution.py:13-25）。未发现幽灵指标。
- E scope/baseline/causal cutoff — PASS：native、headroom、least_progress、most_output按两组反序实跑；策略只读当前运行/等待/KV/输出进度状态（absence_rotation.py:133-205；rotation_native.py:157-176）。报告明确 APC d6未测、无holdout/显著性/质量/Oracle/方法GO（REPORT.md:39-53），范围语言未越证据。
- F classification — PASS：NATIVE_SERVING，限定为单RTX5090、原生 vLLM 0.26 进程内实际执行，调度器有受控补丁；不是外部服务、多GPU或质量证据（run_probe.py:97-116；REPORT.md:3,51）。

执行者封账：收到本答后更新顶层audit状态；所有raw、初始化资格快照及执行前协议保留，报告补明时态层级。审阅行号对应审阅时文件。
