# LTR容量选择组件：fit-scan 与 rank-prefix

PREPARED_UNRUN。固定研究问题仍为同KV预算下的已有请求暂停与完整服务量。

当前HEAD c4ae4f5daaea929e1a4358862296d832f1cc67ab、共享dirty；已读权威入口、RESULT_LEDGER、GPU_COORDINATION、LTR四格实测/审计、最长gap与近邻生命周期定位。原子任务是排除已发现的资源后端接入差异，恢复义务方案暂停，不同时运行第二控制器。

唯一问题：先遇到当前不可容纳的高优先级候选时，停止装入后续候选，是否减少本后端的短恢复/重复重算，并改变全体请求的暂停—完成代价？

最弱链：在已观察step650，fit-scan跳过较早需226/219块请求（剩余210块），恢复只需206块的尾请求；该段后来未到新输出又被抢占。prefix在同当步状态不会开始这段恢复，但其真实后续未运行，不能将该静态判断当节省或完成收益。

实验固定fit_scan/rank_prefix/rank_prefix/fit_scan四个独立新引擎，两者都启用LTR200/10。仍为d6旧32文档、3072输入/1024输出、50ms steady、1024调度token、实际KV13,960,740,864 bytes/6656块、APCoff、OLMoE BF16/vLLM0.26。只改变不可行候选后的continue或break；失败trial不会提交victim、预留或预算改变。原生分配器、实际抢占、完整历史需求不变，未选择resident保持KV。

这是公开LTR中的rank-prefix容量选择组件在共同FCFS/recompute后端上的移植，不是完整预测器/CPU SWAP复现；公开代码还有批量预留、回退和其它资源会计，不能把本组说成逐行等价。原始源码缓存git blob与commit tree已核对，commit13bbf6ff3dab661791d41362551b089e5f77c91c、scheduler blob1a56ad6fad8610a8153e263cc81716a5dd39ffdd。

主结果：全体最大ITL、完整wall/吞吐，同时逐请求TTFT、平均完成、输出差异、全部失败、重算、0/1/2新输出即再次抢占的恢复段。epoch内JIT计入wall并保留，不用小幅变号宣布稳态收益；各策略完整未来独立执行。时间与块守恒沿用已验证账本，decision是scheduler子项。

支持/停止：若prefix明显消除已观察浪费，先分析强简单规则的完整收益和代价，不增加保护；若依然有未完成恢复、只改等待分配或无净效果，只针对真实残留决定下一干预。原生接口/资格失败停止并保留，不写科学NO-GO，不扫threshold/quantum。

证据上限NATIVE_SERVING进程内/MEASUREMENT_ONLY；没有业务SLO、质量、独立新颖性、Oracle、稳态显著性或完整LTR主张。新cohort/异构/突发/原生及最强轮转同组基线仍是后续确认范围。

原组件方唯一CPU实现/封包，root唯一上传/执行/回读，原方上传0/GPU0已明确确认。包SHA1fea4a612af77a0062163cec33566765c2df25ade4021407756ec43b1734ef73，16文件、11项CPU检查由原方完成。本会话再次校验原archive及每个成员SHA，复用已完成四格的执行/锁/回读代码，仅更新独立remote目录。

CPU准备完成，尚未上传/GPU；B r02已完整释放，启动前现场核验。每格及初始化/换格整组共同flock，无后台候卡、无杀进程、無自动覆盖或重跑。唯一下一动作是首次执行这四格。
