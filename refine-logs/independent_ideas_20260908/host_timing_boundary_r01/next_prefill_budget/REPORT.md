# 固定cap8的prefill预算：尾部间隔变短，但完整请求代价增加

2026-09-08。本线程独立实验；未接管其他进程的admission ladder、单步宽度或KV容量实验。

**回答：在本次长短混合自然文本域，将每步预算从1024降至256，能够缩短部分尾部token间隔，
但没有改善平均TPOT、TTFT或整批完成时间。** 两个相反测量顺序的引擎均表现出明显TTFT与
完成时间代价，因此不能把较短的尾部间隔写成完整请求收益或方法GO。

## 实际执行与数据

- RTX5090单卡，OLMoE-1B-7B-0924固定revision，BF16，vLLM0.26.0 / PyTorch2.11.0+cu130。
- 固定引擎cap8、编译token容量1024；仅在完全排空后改变scheduler实际预算256/1024。
- 相同16篇自然文章：all_short全128输入；mixed交替128/2048输入。每条固定128输出token，
  到达间隔50ms。两个新引擎采用相同4个warmup，测量顺序完全反转。
- **8个测量episode、8个warmup全部完成；128次正式请求执行、128次预热请求执行。
  两个进程退出码均为0。** 正式数据有2384个scheduler步骤，未将步骤当独立统计样本。
- 全部原始请求、输出token、时间戳、scheduler记录、环境、命令、日志、退出码和执行代码包
  已回传本地。两个归档逐文件字节核对、JSON读取均通过；远端原件保留。

数据入口为 `readback/results/{forward,reverse}/`；传输记录为
`TRANSFER-forward.json`、`TRANSFER-reverse.json`。压缩包合计2,351,288字节，SHA256分别为：

```
forward 390235e59dcb9508b70fcec292434f96eb2da0e3ea1c09e9755461940c1a6ac1
reverse a8c584bd9a5289174ccaf9e52510338a42a7197ecb87127867618a7cf2e8bcb6
```

## 完整请求结果

以下均为**256相对1024**，两引擎分别报告，不择优、不合并抹平差异。

| mixed指标 | forward | reverse |
|---|---:|---:|
| 全体平均TTFT | 440.917→582.253ms（+32.06%） | 431.131→580.145ms（+34.56%） |
| 全体平均TPOT | 8.710→8.952ms（+2.77%） | 8.641→8.849ms（+2.40%） |
| 整批完成墙钟 | 2.572→2.804s（+9.05%） | 2.548→2.780s（+9.14%） |
| 短输入请求ITL p95 | 15.694→13.250ms（−15.57%） | 14.619→13.404ms（−8.31%） |
| 长输入请求ITL p99 | 20.842→13.897ms（−33.32%） | 20.679→13.769ms（−33.41%） |
| 全体ITL p99 | 21.950→20.938ms（−4.61%） | 20.712→14.041ms（−32.21%） |

ITL分位数是该组实际token间隔的描述性合并分布，不能视作成千上万个独立实验。
全体p99改善幅度不稳定，不能只引用reverse的32%。5s/0.2s只是冻结的参考SLO，
本次没有据这些数据重新选择门槛，也未声称生产SLO-goodput提高。

## 动作确实生效，以及代价在哪里

两个引擎的mixed域中，1024档都有16步实际超过256；256档都有68步触及预算上限。
既有decode跳过、抢占、KV调整均为0，输入prefill token总量守恒，接纳上限始终8。
含prefill和decode的混合步骤均由27增至71；每条请求在已被调度decode时经历的混合步骤
均值由8.875增至22.6875。不是通过暂停已有decode降低表面延迟。

all_short四个测量cell的实际每步token最大值都是135，两档均无观察到预算绑定。
其整批完成时间变化只有+0.39%/+0.51%，平均TPOT变化+0.55%/+0.09%；TTFT小幅变号。
这个负控支持mixed效应与预算暴露有关，也显示非绑定条件并非完全没有运行波动。

复用互斥host时间分解后，mixed平均TPOT中，含prefill步骤的poststamp分量由
1.239/1.218ms增至2.412/2.356ms；pure-decode分量由7.289/7.257ms降至6.363/6.327ms。
各桶之和严格还原每请求TTFT/TPOT（最大浮点残差小于2e−18秒）。
这说明更多decode时间落在混合步骤中，与更碎的prefill干扰相符；
**步骤标签也随动作变化，不能把pure-decode桶下降解释为kernel加速，或把任一桶当可直接回收的税。**
poststamp包含wrapper尾部、引擎执行和返回；不是纯GPU时间。

## 结论边界

这是单模型、单卡、原生in-process runtime的真实调度动作对照。每个episode独立推进
KV、batch、输出和完成轨迹，没有复用另一策略的未来route。相同输入、种子和精度
不保证输出token逐个相同：mixed两引擎均只有4/16条完整序列跨预算相同；
all_short对应6/16和8/16。结果是各策略实际自由生成的固定长度执行成本，
没有任务质量、token一致性或数值conformance保证。

两个引擎KV块数为4429和4477；同引擎内两档共享KV池，均无抢占。
因此主比较在每个引擎内完成，不把两个引擎宣称为容量逐字节相同。
运行环境、初始JIT及设备探测日志都随原始数据保留。warmup用于初始化，单独归档，
不计入正式请求分母；正式墙钟从该episode起点持续到请求排空，包含提交、排队和采集。

降低分块预算产生ITL/TTFT权衡已有先例，不能以本结果主张新动作或MoE专属残差。
[vLLM官方分块prefill说明](https://docs.vllm.ai/en/v0.26.0/configuration/optimization/#chunked-prefill)

## 裁决与唯一下一步

| 字段 | 结论 |
|---|---|
| Verdict | MEASUREMENT_ONLY；256静态档在本域产生尾部间隔与TTFT/完成时间权衡，无整体方法GO |
| Evidence type | REQUEST_LEVEL / NATIVE_IN_PROCESS_GPU_INTERVENTION |
| What was measured | 8测量+8预热；真实预算、既有decode推进、TTFT/TPOT/ITL、整批墙钟、互斥时间分量 |
| What was not measured | 动态策略、Oracle、任务质量、专家信号残差、HTTP服务、第二模型或多卡EP |
| Strongest baseline | 同引擎、同输入的1024静态预算；最高已测档，不称全局最优 |
| Oracle/headroom | 未执行；尾部权衡不等于存在可免费取得的净收益 |
| Claim ceiling | 当前自然长短混合域中，缩小prefill块可减小部分尾部间隔，却增加暴露次数和请求代价 |
| Failure category | 简单动作的完整请求成本；不是无效实验或整个prefill调度家族NO-GO |
| Resurrection condition | 若目标改为明确的逐token间隔约束且能承受已测TTFT代价，256可能是可用配置 |
| One next experiment | 在相同mixed域内，用唯一中间静态预算512对照1024并反序重复，检查简单静态配置能否缓和已测权衡；不新增预测器、不扫阈值 |

下一实验仍UNRUN。本轮已回答问题：**小prefill块并不自动改善完整请求；本域中更少的
单次尾部停顿伴随更多混合步骤，TTFT与总完成时间付出了明确代价。**

重算主表使用 `analyze_results.py`，全部逐请求结果在 `analysis/summary.json`；
时间分解脚本为 `decompose_interference.py`，结果为 `analysis/interference.json`。
输出序列对照见 `analysis/output_parity.json`。重算时使用新目录，原始raw不修改。

独立完整性审查：fresh GPT-5.6-Sol ultra给出WARN/provisional，P0=0、P1=0。
核心指标精确重算；原未定位的时间桶对象差异经一次独立切段复算未复现，1024个请求桶值
及全部均值最大差异0。来源与样本/轨迹边界仍保留，见`EXPERIMENT_AUDIT.md`。
