# 等一拍何时会增加恢复资金？

**有条件的结论：没有成员变化或外部释放时，单靠其他请求继续decode不会增加同一victim/target的可恢复资金。** 这不是“总应该立即恢复”：推迟可能改变host保存/传输成本，完成/原生抢占可能释放容量，策略资格也会变。本笔只判容量变化，不判完整服务价值。

定义当前free为F、victim独占可释放块为Bv、target当前历史需求为T。prepare期间每个运行请求新增gi块，无其他分配/释放，target不变，则：

```text
fund_now = F + Bv − T
F_next = F − Σgi
Bv_next = Bv + gv
fund_next = fund_now − Σ[i≠v] gi <= fund_now
```

victim自己的增长被随后释放抵消；它不是额外资金来源。若存在释放R和新分配A，还需显式加R−A；不得用真实未来EOS填R。候选集合固定时，对每个victim成立，取最大值也不会因纯增长改善。该表达只适用于独占/明确可释放块，不适用于共享引用或未完成传输源。

## 原生prepare→commit检查

从两组各25prepare之前的snapshot，只读当前computed、held、free、target历史；预测每个running下一decode跨块需求。下一快照只用于评价，不作为模型输入。原件/主表未修改。

| 组 | 预测与实际资金余量完全一致 | 缺失 | 预测余量严格下降 | 当前可行→预测下一拍不可行 |
|---|---:|---:|---:|---:|
| selected D | 24/25 | 0 | 19 | 0 |
| native full E | 24/25 | 0 | 20 | 0 |

两个误差全部保留，没有通过扩大模型或引入未来事件凑50/50：

- D prepare858：预测137，实际331，差194块。peer0005122在prepare调用中被native抢占，释放194块；原方segments记录computed3093/output87、preemptions3→4。这不是EOS或目标恢复释放，是另一请求承担了驱逐代价。
- E prepare1398：预测181，实际355，差174块。peer0002134之前已output1023/max1024，持174块，下一拍退场。声明上限在已经临近末token时提供可检验的完成条件，但不允许把所有请求的max_tokens当真实EOS预告；本模型未使用它，误差作为已知边界保留。

其余48个前态均运行集合不变、目标状态不变、所有running恰推进一位置；模型资金余量逐项吻合。它们仍来自两个相关episode，不是48个独立实验。两次观察资金增加均有明确资源释放；不将mandatory prepare对照冒充实际执行的“可选delay”干预。

CPU还构造了最小合法反例：F=1，victim持1块，target需2块，即时刚好足够；一个peer下一步需1块，victim不增长。等待该步后资金差变−1。以及反例对照：只有victim跨块时，先分配后释放相抵，资金不增加。玩具状态只证明关系的可能性，不声称自然负载发生率。

## 研究决定

纯“等待能攒余量”的假说在固定集合/无释放条件下被否定；延后动作必须明确依赖可见的释放条件或成本收益，不能以当前容量紧张自动推出等待更安全。

本批全部50prepare均没有被一拍模型从可行改判不可行，commit已有真实重新检查。故不为这些结果新增预留/提前拒绝guard；它目前帮助解释条件，尚未改变已执行动作。也不能用这点判死带有真实完成/外部释放状态的延后机制。

下一只复用既定selected/full轻量对照，确定保存成本是否提供值得建模的动作差异；不再为提高这些轨迹匹配精度补完整重放，不扫描horizon。若正常运行出现即将完成/额外驱逐改变可恢复资金的决策实例，再用该实例区分“有证据的释放等待”和“纯增长等待”。

复算在仓库根目录（输出新文件）：

```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_native_full_gate_r01/joint_growth_transfer/check_prepare_margin.py --selective refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_native_full_gate_r01/execution_weste_26862/readback/results/diagnostic-native-full/selective-store.json --output /private/tmp/prepare-full-new.json
```

D换为原selected组diagnostic-current/selective-store.json。模型prepare_margin.py同目录；输出prepare_selected.json/prepare_full.json保留全部行和偏差。证据是STRUCTURAL + 既有原生前后态检查，非秒级预测、性能GO或新GPU结果。
