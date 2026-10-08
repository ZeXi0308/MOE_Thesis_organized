# 第一轮观测结论

**MEASUREMENT_ONLY。等待窗口存在；复杂依赖顺序selector没有足够新增空间。已转入真实加载/计算重叠优化对照。**

OLMoE-Instruct、RTX5090、native vLLM eager pager，两臂各16条完整提示、每条强制16输出token，均完成256token，无preemption。3840行是request-token-layer观测，不是3840个独立请求。

| 观测 | cap24 | cap64负控 |
|---|---:|---:|
| 全部top-k早于最后一组完成 | 531/3840（13.83%） | 0/3840 |
| 提前行ready→层尾中位窗口 | 3.601ms | 不适用 |
| 纯decode提前行 | 227/1488（15.26%） | 0/1488 |
| 纯decode提前行窗口中位 | 1.385ms | 不适用 |
| 实际expert加载字节 | 219,194,327,040 | 100,663,296 |
| 完整有界episode wall | 6.588s | 1.292s |

cap64只用于结构负控，容量不同，以上wall差不能称策略加速。cap24的ensure区间总4.372s，包含host提交间隙，不能当纯PCIe耗时；局部ready窗口互相重叠，不能相加成请求节省。两臂13/16个完整16-token前缀一致，未证明数值/质量等价。均按length停止，不是自然EOS质量评估。首轮为人为限制缓存，不是模型自然装不下；存在measurement期间首次JIT，不能用于精确性能结论。

固定当前单层、所有resident贡献先完成、K=24冷加载预算的结构对比：high-load 2685行、oldest 3749行、最简单补齐greedy3771行、精确oracle3777行。pure decode的greedy与oracle均1481行。greedy已取得99.84%的oracle行数，剩余仅6行且全部mixed/prefill；此计数并非时延或请求收益。

裁决只停止当前小模型/当前单层结构目标的复杂selector搜索，不判死整个MoE调度族。下一最小实验为同容量真实生成：原大组串行、小组串行、小组copy/compute流水，见上级OPTIMIZATION_EXPERIMENT.md。若流水只有相对小组串行的局部收益，仍比原版慢，则判当前实现无净优化价值。

数据：request_readiness.json、pager/calls.jsonl和raw.json；口径修正版metrics_scoped.json；结构精确上界prefix_oracle.json。全部原始文件保留。
