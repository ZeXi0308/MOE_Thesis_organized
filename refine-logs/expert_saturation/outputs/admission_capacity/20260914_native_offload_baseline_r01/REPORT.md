# Native默认prompt-offload四格结果

Verdict: MEASUREMENT_ONLY。外部KV恢复真实生效，但当前默认后端未形成请求级净收益；不判死offload问题或完整decode KV恢复。

四格COMPLETE/exit0，共128请求、131072输出token。固定旧d6、6656 usable blocks、APC off、原生FCFS、16GiB host KV配置；两臂不安装rotation/headroom适配器。warmup pending传输排空与connector cache reset均通过。GPU已释放。完整raw/warmup/log归档SHA de651231d24eef0d9ee826d4daab9979414a345af50f9f2c13011bad1bb3b51d。

| block/arm | 平均完成 | wall | 最大ITL | 重算位置 |
|---|---:|---:|---:|---:|
|0 off|20.866s|27.020s|14.167s|21426|
|0 on|25.838s|33.383s|18.104s|2994|
|1 on|22.499s|28.701s|14.173s|2994|
|1 off|21.079s|27.289s|14.477s|21426|

on/off平均完成+23.83%/+6.73%，wall+23.55%/+5.17%，maxITL+27.79%/−2.10%。第一on显著更慢，不能丢弃或以第二on替换；无统计显著性、非劣或质量结论。两on动作计数一致不等于环境/输出/执行路径完全相同。

## 动作证据与范围修正

安装源码readback/native-offload-base.py:510–512默认offload_prompt_only=True，scheduler按该开关截断到prompt长度。本次正确名称是默认prompt-offload基线，不是完整decode KV swapping。

两个on主测均实际store12,884,901,888 bytes（12GiB）、load2,415,919,104 bytes（2.25GiB）。6次自然抢占不变，加载字节/131072=18432个KV位置，与21426−2994完全相等，约86.03%重算位置被替代。743次正lookup是等待中重复查询，lookup_matched_sum不能当唯一复用token。off的load/store皆0。剩余decode重算仍存在。

所有arm参数storage13,838,323,712bytes、KV storage13,960,740,864bytes的比较需读各memory快照；on使用cross-layer cache（kv storage_count=1），不是原单层布局。16GiB是host配置容量，当前快照没有完整host实际驻留/峰值测量，不能把预算当实测占用。后续集成应补主机RSS/缓存驻留及元数据，不声称已完整计费该部分。post-request drain本组0calls，相关时间保留。

## 首轮成本定位

engine.step、其中scheduler span、调用外时间按真实时间戳形成互斥账本；逐调用确认scheduler包含于engine.step。

| arm | scheduler | engine其余 | 调用外 | 最大单调用 |
|---|---:|---:|---:|---:|
|0 off|0.870s|25.665s|0.485s|0.148s|
|0 on|2.512s|29.823s|1.048s|1.168s|
|1 on|1.496s|26.661s|0.544s|0.146s|
|1 off|0.935s|25.773s|0.581s|0.109s|

第二block净wall增加1.411s，其中scheduler+0.560s、engine其余+0.888s、调用外−0.037s。第一block还有更大漂移/尖峰，根因未确认。不能把engine其余全部归GPU或把同步传输span相加当暴露成本。on完成传输统计load约44.5/43.8ms、store234.5/238.1ms；这些是后端时间，不是可直接相减的完整请求耗时。

## 边界与下一动作

Evidence type: NATIVE_SERVING in-process四格；小样本旧cohort探索。
Strongest baseline: 同资源原生off，现成OffloadingConnector默认prompt-only作为待定位强基线。已有rotation/headroom不并入本次配对。
Measured: actual loads/stores、恢复工作量减少、完整请求、固定GPU KV、可见计时分解、全失败保留（本组0）。
Unmeasured: host实际驻留峰值、自由EOS/质量、full-decode offload、rotation兼容、scheduler内部与kernel逐项税、跨cohort。
Oracle/headroom: 局部KV传输空间已测；当前完整请求没有净正。没有全动作Oracle。
Failure category: 外部恢复生效，但backend/布局/调度等整体成本抵消收益；第一on还存在未定位运行漂移。
Claim ceiling: 默认prompt-offload在当前固定域未胜出，不能扩写为CPU KV offload普遍无效。
Checks: 4×COMPLETE/32×1024、有效KV块数和connector类型、清缓存返回、加载字节与重算总差、SHA、互斥时间会计。仅执行者定向检查，未做新增fresh独立审计。
Reopen: 同状态可重复的backend税定位及消除，或完整decode保留有额外可测空间。
One next smallest experiment: 先对同一默认on后端定位build_connector_meta/lookup/update与engine剩余时间的增量，保持原输入和动作；未定位前不扩大offload范围或直接解除rotation guard。

直接回答研究问题：KV传输比重算局部便宜、并且确实少算，不足以保证平均完成更快。当前优先减少和定位实际恢复后端的执行税，研究主问题保持不变。
