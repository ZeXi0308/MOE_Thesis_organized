# 专家映射的同步边界：把等待记在哪里，与能省多少不同

2026-09-15，专家分页与执行组织长期支线。**本轮把“未知控制税”收紧为一个具体同步边界，并交付可检验的最小搬运组件。** 当前没有新增GPU结果，也没有复活cap24频率保护。新的问题是：group-map上传中的host阻塞，是否延迟了kernel提交；解除阻塞能填补多少实际设备空档，而非只让等待换一个位置。

接续增补：[强基线X的复用与两个同步边界](shared_reuse/REPORT.md)已完成。完整内容复用无足够覆盖，候选改为X两张map共同异步提交，保留仅改第二张的负控；该处实验建议替代本页最初两臂建议。本页H收益上限仅适用于共同准备成本抵消的简化模型；新增显式准备/分配成本时序，不能将该上限外推到真实组件。

## 职责、快照与假说

负责专家分页/执行组织成本；资产是既有frequency/guard/替换债务、共享池X/fullstage/logical及Qwen r04。当前HEAD de64dae5，独立worktree为 `/private/tmp/moe-paging-cost-20260915`，共享原件只读。已核对PAPER_ARGUMENT、CURRENT_EXPERIMENT、RESULT_LEDGER、GPU_COORDINATION及既有权威入口；主线自然域native-full保存资格由原owner执行，本支线不重复其结果或争用GPU。

观察：Qwen少搬运/少组未稳定兑现完整请求收益，原记录仍未分开控制与传输。假说H：`host_map_ms`的大部分变化来自同stream前序工作等待，按map大小/CPU循环宽度定价会失准。替代解释是map自身构造成本随组宽变化、或仅有运行时间漂移。最小区分工作：锁定现有调用的映射尺寸，按相同行数/组宽比较命中与缺失，用前两格拟合、后两格检验，再核对实际调用路径。探索性时间切分不当作独立确认集。

## 新增证据

仅复用Qwen r04四格的7,584次measurement layer call、8,656个group，不纳入预热/资格，不增加实验样本数。每组CPU→GPU group map均为128个int32，即512字节；组宽只改变写入多少有效slot，不改变map长度。四格专家组仍使用完整行输入与固定scratch，当前required/miss均在本层router之后、ensure之前可知。

| 实际格 | host apply总计 ms | 其中host map ms | map占apply |
|---|---:|---:|---:|
| static32前 | 5476.724 | 1658.878 | 30.290% |
| static16前 | 5730.829 | 1725.758 | 30.114% |
| static16后 | 5203.032 | 1764.571 | 33.914% |
| static32后 | 5583.793 | 1633.709 | 29.258% |

这是Python wall区间的互斥嵌套分账：`apply = route + Σensure + Σmap + Σsum + residual`。residual含kernel调用、Python计划和记账，不叫纯kernel时间；任何CUDA span都未相加或扣除。图层/组级原值见 [analysis](analysis/analysis.json)、[groups](analysis/groups.json)、[layers](analysis/layers.json)。

对每层首组拟合非负线性模型；训练前两格3792调用，检验后两格3792调用，未随机打散相邻调用。首组避免把同层前一组kernel也混入解释；后两格仍是同engine、同文档，跨负载泛化未验证。

| 模型及合法输入 | 后两格MAE ms | 后两格WAPE |
|---|---:|---:|
| 常数＋当前组required数量 | 0.515 | 73.276% |
| 上式＋当前组miss数量 | 0.109 | 15.450% |

第二个模型拟合系数为约0.13026 ms/miss，另外两系数落在非负边界0；这是受该采样域影响的预测关系，不是每次搬运的纯硬件成本。CUDA观测时间、未来route/EOS、请求身份、step及cell身份都没有进入预测。即便第二模型误差较低，也不能用它预测其它策略未执行的未来轨迹或完整请求收益。

相同行数/required的条件反例也保留：后两格 `rows=1, required=8` 时，295个全命中首组map中位0.03042ms，233个有缺失首组为0.10258ms。map尺寸和循环宽度相同而时间不同，削弱仅由映射尺寸解释的模型；状态仍未随机干预，不能从此差值直接估计可回收等待。全分层均保留，没有只输出这个有利例子。

## 源码定位及被排除的解释

Qwen冻结 [wisp_expert_groups.py](../20260912_native_pager_r01/phase_baseline/qwen3_new_gpu_20260913/attempt04/wisp_expert_groups.py)的238–244行先构造CPU list，再执行 `torch.tensor(mapping, dtype=int32, device=x.device)`，计时到返回。前序权重与persistent resident map在当前compute stream异步提交。PyTorch 2.11.0的 [tensor_new.cpp](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/utils/tensor_new.cpp)对CPU数据构造后的device转换显式使用`non_blocking=false`；其 [官方传输说明](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)说明该路径等待stream。因而group-map操作会吸收前序尚未完成的搬运，后续组还可能吸收上一组kernel等待。

这排除了将本字段当作纯CPU map构造税的解释。仍未测得纯DMA/实际kernel空档，不能把1.63–1.76秒称为可节省时间。第二块static16比32的map区间反而多130.862ms，而apply短380.761ms，故这一边界也不能单独解释该块完整性能翻号。主表和旧裁决保持不变。

当前较强共享池X的 [执行源码](../20260912_wisp_olmoe_r01/logical_alignment_validation_r01/source/run_shared_pool_pager.py)155–156行也在权重复制后用CPU list构造device map。这里只确认同类同步路径仍在，未把Qwen的毫秒外推给X；旧batched persistent-map修复已继承，不再次包装为新结果。

## 一个对选择有用的最小模型

将共同map构造结束定义为时间0。`W`是同stream既有工作的剩余时间，`M`是map DMA时间，`H`是从可执行的host提交开始到kernel入队的CPU时间，`C`是异步staging额外CPU成本。两方案使用相同权重、map内容、执行顺序和kernel，公共kernel时长抵消：

```text
blocking kernel start = W + M + H
async kernel start    = max(max(W, C) + M, C + H)
saving                = min(H, W + min(M, H) - C)
```

因此取消host阻塞并不消除GPU的权重依赖；其本局部最大收益受`H`限制，不能按`W`整段扣除。选择也不是始终async：仅当上述saving为正才选择它。举例的成本均为解析输入而非测量：M=.01、H=.20、C=.08ms时，W=0选择blocking（async慢.07ms），W=.20选择async（快.13ms），W=5也仅快.20ms。见 [qualification.json](qualification.json)。256个非负网格组合验证闭式与时序递推一致；这是模型自洽检查，不是实测收益。

现有`host_map`不能直接当W，host residual不能直接当H。需要一次最小device时间线才能校准这些量。该模型已改变动作设计：只移动map提交的host同步，不碰victim、保护、容量或专家集合，也不把字节不变视为无动作价值。

## 已实现的最小组件及唯一下一实验

[partial_map_staging.py](../../../experiments/admission_capacity/partial_map_staging.py)预分配每层每组pinned host map与device map，只替换group-map materialization。相同stream保留map复制→kernel消费→下次覆盖的顺序；host buffer重用前等待此前DMA完成，跨stream直接拒绝。Qwen48层×3组的bank占host pinned和GPU各73728字节，两臂都持有相同bank。模式关闭时仍走原blocking构造，权重/LRU/分组/kernel参数不变。原exact apply函数的安装编译已CPU检查，DMA延迟消费、buffer重用及非法stream行为通过定向检查；**真实CUDA数值、资源轨迹和完整服务均UNRUN**。没有改共享runtime或已执行封包。

唯一建议是先做小型同stream CUDA资格/成本探针，而非重载整个Qwen：用既有记录覆盖的全命中首组与需加载首组两类状态，保持相同map/权重/kernel形状，分别交错blocking/async/async/blocking。核对实际输出/map内容及完整调用墙钟，并以少量device事件区分DMA完成、kernel启动与结束；观察成本两臂相同。目标是检验上述H上限和是否只是等待搬家；不把它当请求级方法结果。

资源建议：一张已授权5090、一个受主方协调的有限窗口；单层scratch/CPU master按选择的现有模型几何明确计费，不改主线KV实例或占用另一GPU制造隔离假设。执行身份和版本尚未由主方接受，未排队、上传或启动。若局部完整调用无空间，就保留该成本规律并停止此staging实现；若空间可复现，再由主方决定是否在当前强无保护X底座验证完整服务。当前只有这一项候选，没有为多个负载另建冻结包。

## 接续与复算

论文可用边界：观察成本可能承接前序队列等待；减少这个host指标不等于消除设备依赖。当前证据为请求运行的调用级条件分析、源码语义及CPU模型/组件资格，无新方法GO。旧frequency停止边界仍有效；“必须等待主方先证明全部成本空间才可思考”的旧流程约束被最新自主研究指令替代，本支线自行推进这个具体假说。

以下从共享仓库根目录执行（Python需NumPy，所有输出用新路径）：

```sh
python3 refine-logs/expert_saturation/experiments/admission_capacity/analyze_pager_map_cost.py --input-dir refine-logs/expert_saturation/outputs/admission_capacity/20260912_native_pager_r01/phase_baseline/qwen3_new_gpu_20260913/attempt04/readback_r01/results/comparison --out-dir /tmp/pager_map_cost_recheck
cd refine-logs/expert_saturation/experiments/admission_capacity
python3 -m unittest test_partial_map_staging -v
```

本地独立worktree实际运行使用已安装NumPy的Python3.14；系统Python缺NumPy的首次尝试在导入阶段退出，未生成结果、未安装环境。原始Qwen数据和所有旧结果未覆盖，模型fit/test切分与全部分层在同一analysis保留。
