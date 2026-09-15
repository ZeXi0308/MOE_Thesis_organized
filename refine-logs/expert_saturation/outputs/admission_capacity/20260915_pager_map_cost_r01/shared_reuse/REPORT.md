# 强基线X的映射复用与两个同步边界

2026-09-15。**否决完整内容相同才跳过上传的快路径；异步候选改为同时处理两张map。** 这是上一轮group-map成本假说在实际强基线上的接续，不是新的paging方向。GPU未执行，旧frequency/guard及共享池主结果不重跑。

接续：[条件调用入口与前态重建](probe_preparation/REPORT.md)已完成CPU准备。发现不复位重复旧plan会读错18个active expert，改为每次从相同完整前态进入原执行路径；性能baseline采用未修改oneshot，候选包装成本计入。GPU仍UNRUN。

## 新证据：完整map内容基本不复用

从已有logical P/V两个文档集合中只取四个X运行，读取原始plan的完整map；不读取性能优劣来选调用、不重新计算旧主表。每格measurement开始建立冷的逐层shadow，第一次调用必须上传，此后仅比较已经发生的同层map。完整execution map包含384项，local persistent map包含64项；比较只使用当前plan和此前shadow，不用未来信息。

| 文档组/运行 | layer calls | 可比较的同层转换 | execution map完全相同 | local map完全相同 | 可跳过上传/原两张上传 |
|---|---:|---:|---:|---:|---:|
| P/1_X | 896 | 880 | 0 | 5 | 5/1792 |
| P/4_X | 896 | 880 | 0 | 5 | 5/1792 |
| V/1_X | 912 | 896 | 0 | 6 | 6/1824 |
| V/4_X | 912 | 896 | 0 | 7 | 7/1824 |

合计3,552次可比较转换中execution map均变化；7,232次原map上传中仅23次满足完整内容复用，占0.318%，条件少传5,888字节。所有命中都在decode阶段。分阶段、行数、变化条目数和每个事件全部保留在 [analysis.json](analysis.json) 与 [events.json](events.json)。有限轨迹计数不是完整策略的反事实性能；即便时间变化导致未来调度不同，本分析也不替代实际执行。

这否定本批数据上的“相同完整map大量重复上传”假说，故不实现完整内容缓存分支。不据此否定只验证当前active项、其它map表示或其它运行域；这些不属于本轮候选，不再逐个增加模块。

## 源码给出的更重要修正

当前X的 [oneshot源码](../../20260912_wisp_olmoe_r01/logical_alignment_validation_r01/source/run_shared_pool_pager.py)136–167行按同一stream执行：

```text
权重D2D/H2D提交
→ CPU list构造64项CUDA local map（blocking）
→ D2D复制到state.expert_map_device
→ CPU list构造384项CUDA execution map（blocking）
→ fused kernel提交
```

Python先求值第一个`torch.tensor(...device)`再调用外层`.copy_()`，所以第一次materialization已经等待完此前权重队列。**只异步化第二张execution map会保留真正承接权重等待的第一处barrier。** 原Qwen单map模型不能直接套用到这个修改；这排除了一个会错误评价异步策略的实施方案。

两张map不能简单合并为同一内容。local map是logical expert 0..63到private local slot 0..20；当前X kernel不用它，但普通adapter、reset与cache metadata会读写，必须保持一致。execution map则把active expert映射到当前层private绝对pool slot或shared slot336..383，非active以及ID64..383为−1；kernel使用整个384-slot pool、`global_num_experts=384`。其他层共享权重pool，不拥有本层local map。

此前模型的`W`应取**该提交边界之前**的pending work。对仅改第二张的负控，第一处barrier之后的W不再包含权重等待；对两张一起改，才有机会让host的kernel提交与原权重等待重叠。解析反例保留在 [qualification.json](qualification.json)：同样H=.2、额外staging=.08ms，第二张前没有剩余工作时可慢.07ms；两张都异步且原pending work=5ms时简化模型省.2ms。数值是显式解析输入，非实测/预测收益；现有host计时不能直接替代W或H。

**上一轮H上限的条件也需明确。** 它假定共同map准备时间已经抵消、额外staging非负且host提交时间不变；而当前组件还取消了每次临时map分配，准备成本可能更小。不能将H当真实实现的普遍收益上界。新增`publication_timeline`分别保留两方案的每段host准备、device copy和blocking位置：先推进host准备，再按同stream排copy，仅在实际barrier处让host等待，最后计算kernel入队/启动。显式反例中准备时间.30→.02ms、M=.01ms、H=.10ms、W=0时，收益为.29ms，超过H；这只是模型反例，不是GPU结果。见 [model_preparation_addendum.json](model_preparation_addendum.json)。后续探针必须分别测准备、复制与提交，不能只验证拟合的H上限。

## 实现变化与资格边界

[shared_map_staging.py](../../../../experiments/admission_capacity/shared_map_staging.py)复用既有`PartialMapBank`，只替换X原来的两处map publication，保留CPU slot/LRU更新、D2D/H2D权重顺序、普通dispatch锁/事件、pool大小、global expert域、kernel输入与输出路径。

提供三个明确模式：`blocking`逐句沿用原两次构造；`execution_only`保留第一次barrier、仅异步第二张，作为因果负控；`all_async`把两张都从预分配pinned buffer异步提交。local map仍保留H2D临时map→原state map的D2D复制，避免把删除这次复制混入主要处理。每层独立两张staging map，16层共host pinned/GPU各28,672字节，所有臂都分配。没有用完整内容缓存快路径。

host buffer在DMA读完前不可覆写；device buffer由同stream内已排队的kernel消费后才能覆写。组件保留必要的DMA完成事件等待，**不承诺所有host等待归零**。当前候选限制为固定consumer stream，切换stream拒绝；现有X的ordered dispatch必须保留，未来需要跨stream资格时应扩展事件契约，不能取消拒绝。reset/普通ensure可以更新原state map；本候选每次都重新发布，不缓存它们的旧内容，避免shadow失效错误。

已在CPU执行五项定向检查，覆盖两轮不同map的延迟DMA消费/重用、原state map与execution map值相同、三个模式blocking factory次数4/2/0、非法stream/shape/fullstage拒绝，以及原解析式和显式准备成本的模型边界。exact X oneshot源码的两点安装编译通过。这里的队列是CPU语义夹具，不是CUDA数值、kernel时间或生产正确性证明；GPU数值与成本、真实进程/请求状态演进仍未验证。

## 当前唯一实验建议（替代上一轮两臂建议）

问题：同时取消两处map factory阻塞是否提前真实kernel启动、缩短完整调用，还是只移动等待。候选/基线是上述三个模式；预期`execution_only`无法隐藏权重等待；`all_async`的净收益取决于准备成本变化、提前提交所覆盖的设备空档及buffer复用等待，全命中时可能无益或更差。

为避免挑有利时刻，已从P/1_X提取首个measurement有加载调用（call112/layer0，32行、38miss）和首个全命中调用（call983/layer7，1行、0miss）；见 [probe_cases.json](probe_cases.json)。选择未使用时间标签。文件保存实际route/plan，不包含隐藏态或权重；若单层资格使用生成张量，必须明确是合成数值夹具，只能形成`LOCAL_KERNEL`或条件调用证据。

建议在一个主方接受的单卡窗口内，两个状态分别执行固定三模式及反序，记录同map/权重结果、完整调用墙钟与少量device事件（复制完成、kernel开始/结束）。各模式使用相同观察成本；设备事件不可与host区间直接相加。按OLMoE实际几何，384槽共享pool为4,831,838,208字节，单层CPU master为805,306,368字节，再显式计入map、输入和workspace；不需重载Qwen或启动完整KV服务。该资源是静态张量预算，运行时分配/HWM还须现场记录。

入口组件为`shared_map_staging.allocate/install`，三模式与实例输入已具体化；benchmark driver尚未封包，执行身份尚未接受，**没有GPU排队、上传或进程**。局部完整调用无空间则停止此map-staging实现、保留同步边界规律；有空间才提议在当前强无保护底座验证完整请求成本。不得把“host_map下降”单独作为通过条件，不增加第三个缓存机制。

复算从共享仓库根目录运行，输出用新路径（原件不覆盖）：

```sh
python3 refine-logs/expert_saturation/experiments/admission_capacity/analyze_shared_map_reuse.py --campaign refine-logs/expert_saturation/outputs/admission_capacity/20260912_wisp_olmoe_r01/logical_alignment_performance_r01 --campaign refine-logs/expert_saturation/outputs/admission_capacity/20260912_wisp_olmoe_r01/logical_alignment_validation_r01 --out-dir /tmp/shared_map_reuse_recheck
cd refine-logs/expert_saturation/experiments/admission_capacity
python3 -m unittest test_partial_map_staging -v
```

本轮交付是可复用的零覆盖边界、强基线下两个barrier的模型修正及CPU资格组件。它使下一实验有区分力；主问题和本支线完整服务收益尚未完成，不以这些检查代替长期目标达成。
