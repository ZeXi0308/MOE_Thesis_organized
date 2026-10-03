# X 同臂时间差：先出现 host 包络差，再跨到达边界

新增问题是：两次测量均无新增编译时，第二次 X 的 +243.390 ms 落在哪里？输入只读 [attempt01/results](attempt01/results)，机器可读计算见 [OBSERVED_COST_LOCALIZATION.json](OBSERVED_COST_LOCALIZATION.json)。只解释本次实际轨迹，不制造共同未来。

| 互斥 host 区间 | 第二次−第一次 ms |
|---|---:|
| copy/ensure | −74.082367 |
| apply 内除 ensure 以外 | +154.432898 |
| engine 内、apply 外 | +162.160110 |
| engine 外 | +0.879163 |
| capture 合计 | +243.389804 |

apply 内其余可再拆为 route回读 +13.419833、planner +8.389873、未细分 apply +132.623192 ms；未细分项含 kernel 的 host 调用和记账，不是 GPU compute time。每 engine 调用对应一 scheduler step 与16个不同层的测量记录，rows与调度token数匹配，host残差均非负。CUDA load span 的 −74.385664 ms 独立列示，不能再次放入上述加减。

源 [run_shared_pool_pager.py](attempt01/source/run_shared_pool_pager.py:107) 的 load event包络含D2D、H2D和map更新；F还含hit-stage/writeback D2D，且包络可包含提交空隙。它不是纯H2D、PCIe wire或纯kernel时间。D2D少约54%不能按比例外推关键路径收益。

**前缀资格与首次分叉。** 两次测量初始slots/LRU/clock相同，与各自上次warmup写回的完整slot/map/LRU一致；前3步的source request身份/进度、规范化行顺序、48个layer call的完整route和plan/copy身份均相同。hidden/router/KV张量字节没有核验，不称完整数值前态相等。

这个可观察前缀的 engine 总时长已增加44.792728 ms：ensure +5.935825、非ensure apply +19.498989、engine其余 +19.357914。step0 layer0接近一致，layer13/14的apply为8.605→10.028、8.531→12.986ms；该步engine其余另增9.505ms。两次step1 layer11均有19.826/20.869ms的planner包络，但没有GC或线程计时，原因未验证。

step2结束于0.495831/0.540861s，分居第三请求0.5s到达边界两侧。因此step3实际请求数2/3、rows64/96；后续各自推进，最终57/56steps、912/896层调用。新增编译/handle初始化均为0；不能把后面的字节、工作量及时间差当固定轨迹反事实。

**唯一下一实验。** 保持同16文档、原始计划到达、X资源及r02编译覆盖，用X observer off/on/on/off四新引擎复用已有CPU计数与GC观察，最小补记engine、planner和native kernel host包络。先区分线程实际CPU、GC重合与等待/未细分部分；所有观测及完整进程成本计入，不新增GPU同步。源码接入口当前仅CPU准备，GPU `UNRUN`。若这些信号仍不能定位，保留未知；不将改变到达相位后的有利数字作为性能GO。

JSON中的50ms延迟是分析者备选建议，未执行；执行者选定的下一问题是上述先出现成本的来源。现有数据回答：额外时长主要位于非ensure apply与engine其余部分，且可观察前缀已变慢，随后才改变实际到达批次；尚未定位GC、驱动、host或GPU根因。
