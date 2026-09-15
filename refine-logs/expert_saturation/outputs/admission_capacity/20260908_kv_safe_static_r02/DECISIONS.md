# Live KV 块预算推导的安全静态 cap：执行前冻结

研究问题：在已暴露 KV 容量约束的同一长上下文运行域，按真实 KV 布局推导的安全
静态 cap 能否比已完成的 cap16 改善完整请求结果？本实验只比较两种静态准入上限，
不实现动态控制器，也不把普通 KV 容量限制当作 MoE 新颖性。

继承 `20260906_native_memory_pressure_r01` 的32篇真实文章、文档身份和输入token：
主测量全部为3072输入、固定1024输出、50ms到达。保持 pinned BF16 OLMoE、
vLLM0.26、engine32、context4096、token预算1024、显存预算0.90、chunked prefill、
FCFS、同步执行、无prefix缓存。共同三次暖机与原实验逐项相同，完整暖机raw均保留。
原实验的输入、raw、报告和执行包不修改。

## 唯一公式与适用条件

每个新引擎初始化后、暖机与测量前，直接读取其 KV cache group/spec、block_size、
实际 block_pool 和可用块。只有一个完整 attention group，所有层共用该组block table，
没有 sliding-window/chunked-attention、prefix共享、EAGLE、speculative/lookahead、
跨rank context parallel，以及其它未核实的布局语义时才允许计算。

```text
per_request_reserved_blocks = ceil((3072 + 1024) / live_block_size)
safe_cap = min(32, floor(live_usable_blocks / per_request_reserved_blocks))
```

`live_usable_blocks` 必须来自初始化后的真实池，排除实际保留的 null block，并核对
空池free与可用块相等。固定总输出长度使按完整长度预留退化为常数cap；该公式不使用
当前或未来运行结果挑选数值，不根据某次峰值少用几个块就提高cap。
保留最大完整长度是保守上界，不能把该cap称为实测最优或动态Oracle。

baseline16与safe两arm均做相同布局资格检查并记录live计算。若safe<=16、输入边界
变化、池不是空池或布局不在已核实范围，写明 `UNRUN / QUALIFICATION_FAILED`，
不执行测量，不回退到猜测cap。计算细节随每cell的 `safe-cap-qualification.json` 留存。

## 四个独立进程与判读

固定执行顺序：`repeat0-baseline16`、`repeat0-safe`、`repeat1-safe`、
`repeat1-baseline16`。每cell新建引擎；初始化资源差异和各自算出的safe_cap全部报告。
若safe cap随引擎变化，两次safe不能声称同一数值档位复现，只能作为同一公式的结果。

主要报告全部请求的完成状态、TTFT、平均TPOT、完成时间、队列和实际活动宽度，
以及KV占用、分配失败、首次抢占保护。5s/0.2s仅为参考SLO；不应用短文本9ms阈值。
请求时间包含采集/调度成本；不把局部阶段再加到完整请求分母。

若safe不能无抢占地完成，首先定位公式或运行语义缺口，停止提高cap；保护退出保留为
容量边界，不与完整episode吞吐比较。若两次完整比较都没有实质收益，则停止该静态
替换；若有净收益，结论仅为普通容量配置改善，下一步再判断是否存在MoE动作余量。
出现变号仅保留不稳定结论，不按结果调整公式或追加同配置campaign。

## 执行和回传

每次只运行 `run_one_cell.py LABEL`，进程退出后回传全部cell目录、stdout、stderr与
退出记录，核对本地完整后才启动下一cell。所有尝试保留；原始文件拒绝覆盖。
资格计算源码及执行前核实的vLLM接口记录随新包保存；未得到live资格通过前GPU测量
状态为 `UNRUN`。本次准备没有启动GPU。
