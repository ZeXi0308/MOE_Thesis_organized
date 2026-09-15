# DEPA-MoE

当前状态：`DEVELOPMENT_ONLY_NOT_SCIENTIFIC`。

DEPA（Deadline- and Expert-Pressure-Aware Admission）探索连续到达、异构 deadline 下的 admission、iteration batch composition 与 expert-work release。当前目录只实现了 CPU 因果回放、request ledger、小规模 exact Oracle 和三道串行 Gate；没有正式 5090 continuous-decode 数据，也没有多卡 EP/TPOT/P99 结果。

在[当前权威主线](../../current/README.md)中，DEPA 不是已选定主机制。它只有在共同现象 Gate 表明 broader expert-pressure exposed share 跨模型成立、且 BCRD-specific fragmentation 不能更小地解释空间时，才进入 action-space 收缩和 prior-art 复审。

- [实验代码与运行边界](experiments/README.md)
- [冻结开发配置](experiments/configs/depa_v1.json)

禁止把 development fixture 的 Gate 1/2 PASS 写成科学结果；该夹具固定 `scientific_result_eligible=false`。
