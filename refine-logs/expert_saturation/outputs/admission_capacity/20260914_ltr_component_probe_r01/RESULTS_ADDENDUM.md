# LTR等待提权组件：首次原生四格实测

Verdict：`MEASUREMENT_ONLY`，固定KV暂停/服务量问题仍`OPEN`。四格完整结束且逐步核对通过；提权实际返回新输出，整组吞吐、平均完成和最大ITL没有一致改善。原`REPORT.md`是运行前协议，其UNRUN文字保留为历史状态。

Evidence type：`NATIVE_SERVING`进程内，OLMoE-1B-7B BF16、vLLM0.26.0、单RTX5090、APC off、KV13,960,740,864 bytes/6656可用块、32个旧cohort文档、每个3072输入/1024输出、50ms steady到达。boost off/on/on/off各用独立新引擎；128/128请求、131072输出，无失败或剔除。

What was measured：共同的priority-packing/current-history-reservation/native-recompute资源后端，只有LTR等待提权是否参与排序不同；固定threshold200、quantum10。下表均从请求观测起点计费，包含真实调度、观测、重算与引擎成本；初始化及三次同配置warmup在原件中单独保留，不属于已预热请求episode的wall。

| 格 | wall秒 | 请求/秒 | 平均完成秒 | 最大ITL秒 | 抢占数 | 真正重复计算位置 |
|---|---:|---:|---:|---:|---:|---:|
| block0 off | 27.544777 | 1.161745 | 21.458610 | 4.691627 | 20 | 69176 |
| block0 on | 28.160975 | 1.136324 | 22.067661 | 4.928370 | 22 | 75742 |
| block1 on | 27.668794 | 1.156538 | 21.601234 | 4.867612 | 22 | 75742 |
| block1 off | 28.558400 | 1.120511 | 22.332265 | 4.904622 | 20 | 69176 |

on相对同block off：吞吐−2.188%/+3.215%，平均完成+2.838%/−3.273%，最大ITL+236.743ms/−37.011ms。block0全部32个请求完成更慢，block1全部32个更快；每请求最大ITL变差的分别8/32、3/32。全部逐请求变化保存在`analysis/analysis.json`，不挑有利block或事后SLO。

两个on实际调度路径相同，两个off也相同；各策略内部输出逐请求32/32相同。off两次wall差1.013623秒，on差0.492181秒，不能把一对的符号或两次重复当稳定性能/噪声界。on/off每对29/32请求输出相同；0000628、0002733、0003259分别从零基输出位置639、919、725开始不同。各策略沿自己的真实状态独立生成；这是固定输出数量的性能测量，未作质量或逐token一致性保证，也未定位浮点差异来源。

两次on均只有一个有效提权epoch：请求0003640在step609–618连续获调度10次，其中3次纯重算、1次重算并返回首新token、之后6次返回新token，合计7新输出。首次新输出分别在epoch开始后115.693ms/108.400ms返回；没有“10次量子耗尽仍未返回任何新输出”的事件。off的对应计数仅是诊断，其10次调度分散在step647–927、输出3个；不能把它称为实际优先保护。

真实重复位置on比off多6566（75742−69176）；两者fresh prefill98304、fresh decode32736不变。调度调用数off1864/on1862；held resident-call数off14/on110。释放块与实际free变化、计划/实际token分配、每步完整历史预留、held状态不变以及200/10计数全部逐步核对。调度总时间四格分别1.069582/1.118257/1.049153/1.227537秒，decision为其中0.240140/0.256911/0.236252/0.283891秒，不能再次相加。

What was not measured：未知EOS、异构长度、APC开启、第二cohort/模型、网络客户端时间、业务SLO、任务质量、完整LTR长度预测器或CPU SWAP系统、UniBoost整体策略。最长gap事件的具体形成原因正在另作只读CPU定位，不能先归因给量子不足。

Strongest baseline：本组off是同一自定义资源后端去除提权，**不是原生vLLM**；旧d6 native/headroom/least/most只作背景，不拼历史最有利数字排名。该组用于判别已有提权组件在本后端的作用，未声称超越整个近邻系统。LTR的提权/量子来自其已核对公开组件；本文没有新增组件级新颖性主张。

Oracle/headroom status：没有全动作Oracle或净收益上界。本次已观察提权返回新输出，但完整请求收益仍未稳定；不能据此否定恢复调度问题。

Claim ceiling：一个受控资源域、一个cohort的真实执行及组件效果测量；无方法GO、生产稳定性或质量主张。两次on中的量子均实际提供服务，故本样本不支持“固定10次量子完全被重算吃掉”的解释。

Failure category：当前200/10组件未表现一致的全体请求改善；同路径计时波动与跨请求代价均需保留。不是接口失败、未运行或整个LTR/恢复机制的NO-GO。

Resurrection condition：本组件并未判死。只有新的明确失败事件、代表性负载或可检验成本解释才支持下一改动；不因本组变号扫描threshold/quantum。

One next smallest experiment：只读定位两个on中0003571的最长无输出区间，核对其中实际重算、再抢占和idle计数重置。先回答为何已提供有效服务的提权仍留下约4.9秒全局最大暂停，再确定唯一后续干预。

原件：`execution/readback/{pkg,results,group-status.json,campaign.log}`；执行器`execute.py`；分析入口`../../../experiments/admission_capacity/analyze_ltr_component_probe.py`。回传archive SHA256 `4ad8920c9b26f51d9a281c418eeebcbaea183ab983c83575a1b7a3f8de3930f4`；原冻结包SHA256 `12716f4f05586a7e5c7abb2ab3e1db591a61300d4f1ac264f08bf2898aaee502`。GPU于1789324160.649现场检查后释放给B compile-domain r02，无追加GPU组。

本次直接回答：等待提权在当前后端确实转化成连续新输出，但不足以证明固定KV下全体暂停与完整服务量同时改善；缺口已收窄到残留暂停的跨步恢复过程。
