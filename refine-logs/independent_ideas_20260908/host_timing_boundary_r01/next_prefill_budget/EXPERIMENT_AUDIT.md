# 独立完整性审查

结论：**WARN / provisional，P0=0、P1=0**。Fresh GPT-5.6-Sol ultra，
review_independence=same-family。审查在仓库小型探索预算内收尾，没有创建.aris。

| 检查 | 裁决 | 已验证证据与边界 |
|---|---|---|
| A 输入/参考 | WARN | `prepare_inputs.py:25-53`、`prepared/mixed/config.json:34-65`记录来源、hash和前缀；没有model-output GT。原Arrow本轮未独立重读，完整dataset provenance为NOT_VERIFIED。 |
| B 指标/会计 | WARN | `runtime/metrics.py:85-155`计入到达/完成及失败，所有16份metrics和8个测量比较从raw精确重算。相对量分母为1024原始时间。`queue_s`是host submission lag，不是native scheduler queue。 |
| C 实际执行/归档 | PASS | 两个`readback/results/*-process.json:1-15`均exit0；16个raw全部COMPLETE、16请求×128输出/时间戳，两个40文件归档逐字节相同且SHA正确；正式共128请求、2384steps。 |
| D 执行路径 | PASS | `run_prefill_budget.py:35-48,110-135`真实设置scheduler预算并采集；`runtime/native_capture.py:72-130`执行预算/既有decode/抢占/KV保护；复制的feedback路径未执行，也未冒充已测。 |
| E 公平性/范围 | WARN | block内共享engine/cap/编译容量/KV池，反序与共同warmup正确；block间4429/4477 KV块不称容量相同。all_short预算不绑定，mixed动作真实绑定。输出轨迹不相同，不能称same-state或token-equivalent speedup，不能外推统计稳健尾部收益。 |
| F 类型 | PASS | real_system_timing_measurement / no_task_ground_truth / native_in_process_gpu_execution / request_level_exploratory_intervention。 |

原reviewer对`analysis/interference.json`的完整对象重建留下未定位差异，未提供具体diff，
并明确这不影响已精确重算的核心TTFT/TPOT/ITL/wall。
随后仅对此差异进行一次独立定位：`/root/current_scope_check`采用时间边界切段、bisect和
math.fsum，未调用原实现；1024个请求桶值、160个group均值、8个残差字段全部精确相等，
最大绝对差异0，metadata与顺序也未发现差异。最大守恒残差1.734723475976807e-18秒。
复算源码保留为`verify_interference.py`；没有修改原raw或分析。未复现原未指明的对象差异，
无需数值修正；时间桶仍只作描述性诊断，不是GPU kernel成本或可回收税。

可支持：本次固定自然token输入、单5090/OLMoE/vLLM0.26/cap8下，256相对1024在两个
反序block均增加平均TTFT、TPOT、完成墙钟，同时部分描述性ITL分位数下降；没有通过
跳过既有decode、抢占或KV adjustment获得表面收益。

不支持：整体方法GO、Oracle或全局最优；native queue或纯GPU加速；质量/token一致性；
生产HTTP、多模型/多卡和跨workload泛化；statistically robust tail improvement。
远端原件在执行者操作中保留；reviewer因只读本地范围未亲自远程核验。
