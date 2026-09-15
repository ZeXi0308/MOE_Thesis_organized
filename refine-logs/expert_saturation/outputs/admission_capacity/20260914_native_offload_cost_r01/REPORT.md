# 默认offload有限开销观测

Verdict: MEASUREMENT_ONLY。connector路径有约1.1–1.2s可见开销，但主时间仍在未细分engine；没有足够依据直接优化某个Python热点。观测开关自身也有不可忽略的配对耗时差。

四格全部COMPLETE/exit0，128请求/131072输出，GPU已释放。回读SHA6dd504b78d98f047bc3851eafaa9208353f913aa20fcfd488b0aea9b0afaafb9。四格均native prompt-only offload on/16GiB/APC off/d6，区别仅cost-profile开关；不要将profile-off写成offload-off。

| block/profile | wall | mean completion |
|---|---:|---:|
|0 off|31.985s|24.490s|
|0 on|32.694s|25.287s|
|1 on|35.548s|27.618s|
|1 off|33.865s|25.969s|

两组profile-on/off wall+2.217%/+4.968%。不能把这些差全部归因观测：两profile-on本身wall差8.728%，缺少更多受控重复；这也不是总体噪声底。所有原件保留，不删慢格、不给旧结果事后“去噪”。

## 身份与会计

四格1866步完整signature相同，每对32/32输出序列相同。load均2,415,919,104 bytes、store12,884,901,888 bytes、重算2994位置。固定6656 usable blocks、主测前cache reset均通过。

每个profile-on记录17569条span，1866个engine.step根，各步骤exclusive wall/thread和等于根inclusive，最大会计误差0。CPU嵌套检查之外，真实native接口与卸载均正常执行。

| 互斥方法项 | block0 | block1 |
|---|---:|---:|
|查询matched tokens|0.240s|0.255s|
|更新request/offload状态|0.510s|0.331s|
|构建保存任务|0.250s|0.277s|
|build_connector_meta自身|0.052s|0.062s|
|处理抢占/启动传输/准备保存/回收完成合计|0.091s|0.108s|
|处理worker返回|0.056s|0.078s|
|上述connector合计|1.200s|1.112s|
|engine.step其他部分|30.435s|33.205s|

这些是主线程时间区间，不是可移除成本；异步传输并未全部包含在worker调用span里。根thread时间31.632/34.309s接近wall31.635/34.316s，不能直接解释为Python瓶颈，运行库/驱动轮询与等待也可能消耗CPU。其他线程、GPU kernel活动尚未分离，engine residual不能叫GPU耗时。

## 研究解释

此前on/off共同前1026步已有执行税，本轮定位出部分持续connector工作，但未解释大部分engine时间或数秒运行间变化。若只删除一个0.25–0.51s方法，也不能预报完整系统会获得同额节省。offload减少重算是真实结果，净性能失败的物理机制仍未闭合。

Evidence type: NATIVE_SERVING原生状态+有限profiling，小样本旧cohort。
Strongest baseline: 同backend/profile-off；上一组offload-off比较保持原裁决。
Measured: signature/output/transfer一致性、profile配对wall、主线程互斥会计、完整请求。
Unmeasured: GPU kernel/驱动等待/CPU派发分解、硬件频率影响、host真实驻留峰值、full-decode或rotation兼容。
Oracle/headroom: 无新Oracle或机制收益；不能以选中span之和当可消除上界。
Claim ceiling: 定位部分成本并识别观测/运行波动；无方法GO、生产或泛化主张。
Failure category: 现有观测范围不足以做物理归因，不是offload问题死亡。
Checks: SHA、4格32×1024、6656块、reset、字节、调度与文本一致、嵌套守恒通过；执行者定向检查，无新增独立审计。
Reopen/continue: 在相同decode工作段得到CPU派发/同步等待/GPU活动的可区分证据。
One next smallest experiment: 使用既有共同前缀，限定32个纯decode调用做CPU/CUDA活动轨迹，不再增加全程host方法埋点；结果用于源定位，不将受profiler扰动的完整wall当策略收益。运行前冻结选段与输出，确认段的实际逻辑执行相同。
