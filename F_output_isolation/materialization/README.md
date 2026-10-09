# F：输出物化时机与粒度

本阶段沿用[科研基准](../../AGENTS.md)，只探索输出物化动作。C1与先前排序、缓存结果保持原样；冻结协议及预算见[protocol.md](protocol.md)。**最终决定：结束当前实现分支及F独立论文投入，保留回放器。** 28次冻结CPU运行已完成，候选尚未证明独立方法价值并伴随集中退化；不追加动态timer、优先级或GPU验证。

当前入口：[一页判决](verdict.md)、[主结果表及全部重复与副作用](results.md)、[逐请求机器可读分析](summary.json)。high相对同释放规则静态方案，候选轻P99降低16.66%/12.90%，完整server+client CPU却增加0.60%/0.13%，重P99增加4.44/4.82ms；medium成本与交付收益方向不一致。局部工作移到timer解释了大幅OP CPU下降，不能解释为消除了同等工作。保留这一交付取舍，不宣称整个问题不存在。

**解释修正：成本与交付要求分开。** API轻重只能提供成本线索，不能推出谁应优先、谁能多等；75ms不是logprobs用户的服务约定。总CPU不下降不自动否定调度或隔离价值，总体轻流P99下降也不自动证明贡献。未来重启须有依据的服务需求、可合法调整的等待范围、强简单方法留下的缺口，以及能改变动作选择的新状态；没有精确SLO时评价可达取舍，不能临时发明重请求的等待容忍度。本次只修正文档解释，不改变冻结协议、实现和数据，不追加实验。

以下记录已完成阶段的原主张：研究对象是共享CPU frontend的文本流与top-20 logprobs流，假设通过合并可选物化消除重复封套/调用，不推迟生成状态或丢失内容。原主指标为轻请求客户端交付P99，同时审阅完整CPU与每请求代价；该探索指标不构成服务权益依据。

| 假设 | 支持与竞争解释 | 本阶段区分方式 |
| --- | --- | --- |
| 问题 | 存在逐输出固定成本；但top-k成本可能主要是不可省的逐条目工作 | 原生interval1/2/4及早期配额门控的完整CPU、chunk/bytes和逐请求代价 |
| 模型 | 能合并的是响应对象、封套、交接和调用；不能直接省掉逐token解码、停止检测与完整top-k本体 | 同源输入与同sample position总量；逐步状态、完整语义相同，实际物化调用和总CPU一起判断 |
| 方法 | L减少外层sample materialization调用；竞争解释是仅延后相同工作且额外复制/累计抵消节省 | L/S配对计入全部CPU，检验本候选的工作减少主张；隔离价值另需服务取舍证据，不以CPU下降为必要条件 |

可观测状态仅有当前已接收token数、原始payload字节、最老pending时间及API类别；不用未来轨迹。动作是保持生成状态更新、在对象构造前返回None，或把全部未交付内容交给原生构造器。L另保存原始logprobs数组，直到同样的静态输出点再按原顺序物化；cumulative_logprob每步仍更新。内存与copy是必要副作用。跨批consumer/事件循环可以交错，75ms timer不是客户端硬时限。

## 实现与最小复现

- `gate.py`：强简单门控与唯一L候选，所有新增状态按请求保存，finish/abort清理timer与引用。
- `server.py`：沿用原生OutputProcessor、tokenizer、chat generator、Pydantic、FastAPI/uvicorn，仅替换引擎执行器；固定输出handler单条配额，无priority/cache。
- `run.py`：原独立producer与HTTP客户端，全部语义校验在正式客户端CPU窗口之外；chunk可合并，token仍按所属chunk可见时刻计延迟。
- `semantic_check.py`：生成状态、Unicode、stop string、engine STOP、直接processor abort和无后续token的deadline检查，不是性能证据。HTTP取消链未接入和覆盖，不据此声称客户端取消正确性。

在原服务器`/root/F_output_isolation`（vLLM0.26.0/Python3.12，依赖见父目录environment.json；仅加载已有tokenizer）：

```sh
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/root/miniconda3/bin/python materialization/semantic_check.py \
  --output materialization/semantic_check.result.json
/root/miniconda3/bin/python materialization/run.py \
  --trace inputs/high.json --output materialization/reproduce_high
/root/miniconda3/bin/python materialization/run.py \
  --trace inputs/medium.json --output materialization/reproduce_medium
```

输出目录必须不存在。默认14run正反序及75ms/4token/1024B在源文件与protocol.md冻结；每个server重新启动。默认frontend/client/producer为核2/4/6，全部arm同资源。`execute.py`用于本次正式顺序执行high和medium并记录源码、已安装运行时身份及执行日志，不覆盖已有结果。强制CUDA不可见，不获取或使用GPU锁。

已保存结果可在本地仅用Python标准库复算：

```sh
python3 F_output_isolation/materialization/analyze.py
```

该命令核验两组各14run、账目时间顺序、完整语义摘要、最终状态与源时序，再生成`results.md`和`summary.json`。保存的原件位于`high/`、`medium/`。`semantic_check.result.json`为一次通过的七臂语义结果；`execution_hashes.json`绑定10项执行源码/输入，`runtime_identity.json`绑定安装版12项相关源码和版本，`execution.json`记录两组exit 0，`download_verification.json`记录远端归档SHA256与本地一致、执行源码匹配及下载文件哈希。正式性能执行总墙钟约14.2分钟，GPU使用为零。

原生两个interval基线直接设置每个RequestState的原生stream_interval。T/B/S在同一原生make_request_output之前门控，避免把“先构造再丢弃”当弱基线。B预算是8B/sample ID加原生logprob数组nbytes，不是已序列化JSON字节；无需先做昂贵物化才能决策。公共轻量计数也计入完整CPU，不能把本阶段native1与旧阶段绝对时间直接做配对。

## 测量边界

CPU包括测量窗口内状态更新、全部物化、timer、raw合并、序列化与控制；另报clientCPU及两者总和。初始化、输入渲染和预构建engine数组不在此窗口，engine IPC省略；不能称完整真实服务CPU。OutputProcessor局部CPU不含timer回调物化，不可代替完整serverCPU。峰RSS含预建回放数据；client为同一driver的累计高水位。`pending_raw_bytes_peak`是逻辑积压预算而非实际内存。固定产出下排空吞吐不表示最大容量。

所有生成token、top-k条目、finish/stop、usage与DONE完整保留；只改变合法chunk边界及重复封套字节。原始客户端正文在摘要完成后释放，保存原始时间记录及语义摘要，离线不声称重新逐字核对已释放正文。原GPU未开logprobs、合成候选高重复、固定回放批内顺序与共享主机限制仍在。完整CPU收益若仅来自推迟heavy交付，需作为取舍报告，不能称消除了条目工作。
