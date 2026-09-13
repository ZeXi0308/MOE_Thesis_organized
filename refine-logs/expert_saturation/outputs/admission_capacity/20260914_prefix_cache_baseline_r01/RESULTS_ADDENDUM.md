# 原生 APC 强基线实测补记

Verdict：`MEASUREMENT_ONLY`。本组回答了“原生前缀缓存是否已解除当前长暂停”：**没有；它减少了真正重复执行，但两个被抢占请求的恢复开始步骤未变。** 此结论只适用于当前 cohort2、固定 KV、无跨请求共享首块的自然文章输入，不是 APC 的普遍 NO-GO。

Evidence type：`NATIVE_SERVING`，原生 vLLM 进程内完整请求。RTX 5090，OLMoE-1B-7B BF16，vLLM 0.26.0，32请求×3072输入/1024输出，50ms到达；每臂实际 KV 16,089,350,144 bytes / 7,671可用块。off/on/on/off四个独立引擎全部完成并回读，128/128请求、131072输出，无失败。相同预热请求后两臂均排空并成功reset缓存，测量前引用/hash为空、7671块全部可用；每臂独立执行，原生臂均未安装轮转或headroom。

| 执行顺序 | wall (s) | 请求/s | 平均完成 (s) | 最大 ITL (s) | 真正重复位置 |
|---|---:|---:|---:|---:|---:|
| cohort2-block0-apc_off | 23.938615 | 1.336752 | 21.491376 | 4.663436 | 7685 |
| cohort2-block0-apc_on | 23.820034 | 1.343407 | 21.367483 | 4.685997 | 6677 |
| cohort2-block1-apc_on | 23.739310 | 1.347975 | 21.285463 | 4.708893 | 6677 |
| cohort2-block1-apc_off | 23.748766 | 1.347438 | 21.271313 | 4.686814 | 7685 |

开启 APC 相对关闭 APC，两block吞吐分别 +0.497819% / +0.039834%，平均完成 −0.576479% / +0.066523%；最大ITL分别增加22.560/22.079ms。这是同输入的两次相关描述性比较，没有显著性、非劣或稳健净收益主张。逐请求TTFT、完成、ITL与输出序列对照保存在 [analysis.json](analysis/analysis.json)，没有按结果删请求或删repeat。

What was measured：每格1348次完整调用、2次自然抢占。APC-on两次均在step1026为request0016821成功复用1008个token位置，首次准入复用为0；完整真实执行区间并集重算量由7685降至6677。全执行位置138725→137717，新prefill98304、新decode32736不变。两victim的首次恢复步骤仍为1030/1026。最长暂停request0017069的恢复前等待约4.53–4.57秒，恢复调用跨度约0.115–0.118秒；缓存命中发生在另一个victim，未提前最长暂停请求的恢复服务。

空闲且有hash的缓存块仍属于free池，重新touch它们消耗可用块；缓存复用减少计算，不保证减少完整历史重新驻留所需的空闲容量。这里观察到的命中与恢复步骤支持该解释，但没有枚举跨请求共享、不同缓存保留状态或改变恢复优先级的反事实。

What was not measured：跨请求共享前缀、长持续服务、异构长度、第二模型/运行域、外部HTTP服务、业务SLO、任务质量及全动作Oracle。捕获与边界检查成本保留在wall；warmup/模型启动不计入该既定测量窗。恢复跨度可重叠、不得相加为wall或纯重算GPU节省；token位置不是GPU时间。

Strongest baseline：本次已覆盖原生APC开/关；轮转在APC开启下尚为UNRUN，旧关闭APC轮转不能直接与本组跨运行拼接宣称收益。Oracle/headroom status：没有完整动作空间上界；这里只关闭“原生APC已消除该暂停”的解释。

Claim ceiling：原生缓存可复用被抢占请求自身历史，当前单cohort减少重算但未改变恢复开始步骤；仍是measurement，不是独立方法贡献。Failure category：不是运行或APC机制失败，当前恢复等待瓶颈仍暴露。Resurrection condition：跨请求共享或不同缓存保留/容量域需要新的真实执行，不能外推本结果。

One next smallest experiment：保持既有most_output轮转配置，资格化APC下真实释放会计后，与APC-on native做native/most/most/native四格；不扫阈值、不改问题。先CPU准备，GPU排在已登记A d6及B编译定位整组之后。

原始 [execution.json](execution/execution.json)、四格gpu_results及冻结包保留；冻结包SHA256 `6266a0dff1411cbf5403beb8ccab18918325ed912b2054e68985e286302290fd`。复算见 [ANALYZE_COMMAND.sh](ANALYZE_COMMAND.sh)。本补记不改原PREPARED_UNRUN准备记录或raw；独立有限审阅结果另列audit，未提前宣称通过。
