# 同执行路径时间差：四项受控重复

PREPARED_UNRUN。前轮已观察同一most_output逻辑路径的一次恢复调用时间差；原因未定。
问题：保持输入、策略与公共预热，长调用是否再次出现，native/most完整吞吐方向是否重复？
只改变执行批次；复用已经观察过的cohort2，不声称新holdout或新运行域。
OLMoE BF16/vLLM0.26/RTX5090；KV16089350144bytes、7671usable blocks；32×3072输入/1024输出、50ms steady。
顺序：block0 native, most_output；block1 most_output, native。四项均独立引擎，原三个公共预热不变。
策略、runtime、采集、模型、输入、token预算、动作阈值全部复用父包；不清缓存，不追加压力预热，不预设冷编译原因。
整组连续交接，排在GPU_COORDINATION现有F/X及A-review之后；每格边界/初始化/测量前查占用，忙或查询失败ABORT。
已有gpu_state记录功率、温度与SM时钟。保留环境、全部调用、失败和中断；不自动重跑、不替换canonical。
全四项同资源、同输入、真实动作及请求账本合格后，按block比较native→most；同角色重复只作描述。
主指标为完整吞吐与最大ITL，同时报告平均完成/TTFT/逐请求损益、调度、重算与收尾成本。
wall=scheduler_inclusive+engine_non_schedule+outside_engine_calls；decision在scheduler内。含重算调用不是纯GPU税。
诊断按第一次成功forced交换对应请求的恢复末次调用对齐；只有逻辑轨迹相同才比较同step编号。
所有长调用保留在主结果；不减去前轮0.734617秒差值，不事后挑选快调用或仅报告有利block。
若同路径长调用重现，再加能区分host等待/设备执行的最小观测；若未重现，只记本重复未见，不宣称根因已找到。
若轨迹改变，先定位第一次实际schedule/action差；若完整收益变号，保留权衡，不扫描阈值。
固定只跑四项，不按结果继续追加。当前问题OPEN，结果上限NATIVE_SERVING/MEASUREMENT_ONLY。
最近邻系统、headroom跨域、业务SLO、质量和Oracle仍未补测；参考SLO全部通过不等于长暂停SLO成立。
