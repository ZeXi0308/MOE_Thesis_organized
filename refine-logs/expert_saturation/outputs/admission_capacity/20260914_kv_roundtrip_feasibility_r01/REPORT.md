# A：KV 往返替代重算的成本资格

Verdict: OPEN / CPU_FEASIBILITY_ONLY。字节与生命周期已核算；D2H、分页打包/恢复及native集成尚未测量，无方法GO。

## 新问题与冻结事实

停止轮转只能交换平均完成与停顿，单次victim选择预测空间很小。当前问题是在原固定KV容量、原计算语义下，能否用额外主机暂存降低恢复成本。主机容量和传输不能免费；改变重算为恢复后未来调度必须独立推进，不能将旧轨迹删掉重算直接算系统收益。

从两次已核验有效KV快照得到每token全16层131072 bytes（128KiB），真实FLASH_ATTN shape=(6657,16,16,256)，stride=(65536,256,4096,1)。按d6强基线两个least raw的39次实际抢占逐个核对状态、后续首次调度和首次新输出：

- 两block均单向有效字节17,992,515,584，即17.99GB；对应往返35.99GB。
- 按旧重算轨迹保存到首次新输出的生命周期估计，峰值host暂存3,446,800,384 bytes，即3.45GB。这不是新swap策略的已验证峰值保证。
- 首次自然抢占3273个已计算位置，有效KV428,998,656 bytes；按205整块搬运429,916,160 bytes。
- 该请求恢复调用窗口0.11027/0.12934s，包含其他请求的同时decode和host成本，不能全当可替代成本。冻结模型token项约0.03813s。

由模型token系数11.65064us/position得到无打包/调度税、两方向带宽相等时，2×131072/B = 11.65064us，B约22.50GB/s。这个数是经验模型的等成本点，不是物理定律、实测纯重算成本或安全上界。

## 现有传输证据与缺口

只读复用20260912_expert_union_measured_r01/s3/h2d/h2d.json：768MiB连续pinned H2D中位56.27GB/s，best peak56.44GB/s。该记录没有本次双向非连续KV布局、当前GPU UUID及并发成本资格；不能给D2H赋同样速度。当前不以它作GO裁决。

## 已准备的唯一探针

probe.py：同捕获shape/stride、16层完整pool、205/207/256块随机位置；分别计量index_select+同步D2H、H2D+index_copy。预分配pinned host和单层GPU staging；全块padding计费。一个warmup加7次实测全部保留，计时外破坏选中块并逐元素验证恢复为原始按block/layer区分的内容。

整组fcntl共同锁，GPU占用查询失败/忙碌即ABORT，初始化前检查。输出禁止覆盖、异常原件保留。不包含模型加载、native hook、并发decode或完整请求，不能成为系统性能结果。额外GPU staging与pinned buffer预算在后续集成时必须计入。

目前只有Python语法检查；GPU探针UNRUN、上传0，不声称CUDA执行或正确性已通过。拟在已冻结prefix四格之后重新现场核验，不能插入其他会话组间隙。

Evidence type: 原生轨迹字节/恢复对齐 + CPU模型成本资格。
Strongest baseline: 同资源least-progress原生recompute；既有headroom是另一个准入机制，不等于同状态KV swapping。
Oracle/headroom: 未验证双向物理成本，也无action-conditioned请求收益。
Claim ceiling: 当前可准备测试，不足以证明KV swapping净收益或新颖性。
Failure category: 物理成本与集成成本未测，非NO-GO。
Reopen/continue: 实测含打包往返明显低于可替代恢复成本后，才做最小native同前态恢复分支；否则保留窄域成本边界。
One next smallest experiment: 协调窗口后执行上述独立双向布局探针，保留所有重复，随后与同状态真实恢复成本比较。
