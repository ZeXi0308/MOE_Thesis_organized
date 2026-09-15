# 完成释放先于保存上报，复用仍受原生flush约束

本方负责保存语义与成本。本轮针对C提出的“推进某请求到声明cap以获得容量”的条件模型，独立核对完成到块释放的真实边界，不实现其allocator、不重复主性能表或GPU组。

**结论改变：不能假设正常完成后必须等全部store完成才能归还GPU块，也不能把归还块视为无执行依赖的容量。** 当前同版本native offload正常完成时立即走core free；后续若复用涉及未完成保存的源块，沿原生`jobs_to_flush`等待。它提供的是可分配容量与带潜在同步成本的复用路径，而非延期持块直到`finished_sending`。

## 实际原件

复用native-full单格全部64个完成请求，以终止output的精确`received_s`连接同一完成engine call，再取其后首次scheduler内存快照。**63个有后续快照的请求均已不在core request表、持块为0；最后1个没有后续快照，保留未观察边界。** 没有将观察不到的free函数时间补造为零延迟。

其中6个请求的store完成上报晚于其终止输出：5个是下一schedule为已完成请求新建的store，另1个已在最后计算step建好。前5个中的4个与该schedule的真实flush job相连；剩余1个没有此flush记录。以下时间均是同一诊断的host相对时刻，只呈现先后次序。

| 请求 | 引擎终止输出s | 首次观察0持块s | 较晚上报store job | finished时建job | flush step |
|---|---:|---:|---:|---|---:|
| 1582 | 25.744081 | 25.744498 | 2186 | 是 | 1496 |
| 5161 | 47.068964 | 47.069356 | 3911 | 是 | 2477 |
| 5679 | 47.700633 | 47.700902 | 3955 | 是 | 无记录 |
| 5885 | 47.897354 | 47.897651 | 3970 | 否 | 无记录 |
| 6758 | 48.030127 | 48.030346 | 3979 | 是 | 2530 |
| 7115 | 48.641911 | 48.642021 | 4010 | 是 | 2577 |

实例1582：最后schedule尚持235块，终止输出在25.744081s；下一schedule入口已持0块。该步随后才为其最终完整chunk234注册store2186，源GPU块1257，并加入flush；worker完成上报在25.772242s。这个先后关系反驳“必须等store完成上报才free”。它没有测出flush净暴露时间，不能把25.772242减终止时刻当成可以消除的等待。

## 同hash原生源码解释

源码分别嵌入[core快照](../../20260914_load_ready_contract_r01/native_source.json)与[offload快照](../../20260914_kv_roundtrip_feasibility_r01/native_offload_source.json)；相关文件hash与本组冻结runtime记录一致。

1. Core `update_from_output`判定停止后调用`_free_request`（core scheduler 1722–1726、1810–1815）。
2. Connector `request_finished`注册pending store涉及的源块并保留最终hash，但返回`False, None`（offload scheduler 1282–1308）。
3. Core因此走`_free_blocks → _free_request_blocks → kv_cache_manager.free`；正常完成的最后模型步骤已处理（core 2207–2258）。异步在途写入/load中止的延期释放分支不能与此正常完成路径混用。
4. 下一schedule如重新分配pending store源块，加入`jobs_to_flush`；该步新建finished store若源块已被分配，也加入flush（offload scheduler 1113–1163）。
5. Worker先提交所需store，再`wait(jobs_to_flush)`（worker 281–303）。Store完成经`completed_jobs`清理注册；这个worker不给store发`finished_sending`（worker 328–363、offload scheduler 1215–1251）。

完成上报不是精确DMA终点。当前快照也没有外层model-runner与底层wait的完整执行计时；因此只支持依赖关系和实际注册观察，不支持净同步耗时、字节正确性或性能归因。

## 可供共同模型使用的最小状态关系

必须保留三个不同集合：当前请求持有的GPU块、块池实际可分配块、待保存作业仍引用的源块。前两者可按原生free变化；第三者可与可分配集合重叠。新分配集合A如果与某pending job的源块相交，不能据此判容量不可行，而应保留原生flush依赖。最终finished chunk还可能在该schedule新建，故“完成时pending=0”不充分保证无flush。

```
容量可行性：使用当前原生free和真实分配结果
复用执行依赖：由该步原生metadata生成的jobs_to_flush承接
成本：相关保存提交/等待与混合执行路径，不能先假定为0或把job时长相加
```

这不是另写pager或提前调用manager探测。C的条件cap路径可以在真实完成/free发生后使用新增容量，不能提前按未来EOS贷出；接入时沿原生metadata/worker fence，不额外把所有finished store块扣成held，也不跳过flush。A在commit时读到的实际free可以用于当前容量重检，但不意味着相应恢复调用没有执行成本。

**支持实例**是上述1582的完成→0持块→新store/flush。**反例边界**是5679的finished store没有本次flush记录：不能对每次finished store统一加一份同步税；是否暴露取决于实际源块复用与运行时依赖。

本结论修正共同状态接口，不宣称完成优先策略有收益，也不另建GPU实验组。下一接入若被主方采用，只需验证实际完成后的分配继续使用原生flush路径；完整收益仍以已在运行的selected/full强底座比较为前提。

## 复算

脚本已实际运行，保留全部64请求与6个晚store、每个连接时刻和来源hash。输出新路径，原件不改：

```bash
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_native_full_gate_r01/finish_release/analyze.py \
  --cell refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_native_full_gate_r01/execution_weste_26862/readback/results/diagnostic-native-full \
  --scope-analysis refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_native_full_gate_r01/execution_weste_26862/analysis.json \
  --output /tmp/native-finish-release.json
```

证据层级：NATIVE_INPROCESS_DIAGNOSTIC派生时序＋同版本源码；MEASUREMENT_ONLY。没有新的GPU样本、服务排名、逐函数free/flush耗时或性能Oracle。最后请求的无后续快照是测量边界，不是释放失败。
