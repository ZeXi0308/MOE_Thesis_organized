# H: independent-input cadence result

**Verdict: MEASUREMENT_ONLY；预声明的H迁移合同通过。** 在128篇未用于选择的新完整文章、25.4s有限到达和同GPU/host预算下，selected/eager相对selected/current两配对最大生成gap下降13.60%/3.12%，实际输出tokens/s变化+5.439%/+0.0728%，平均完成−6.263%/+0.492%。两对均满足事前冻结的吞吐损失≤3%、平均完成增幅≤5%且最大gap降低，主分析状态为`STABLE_TRADEOFF_CONFIRMED_IN_H`。这表示两个执行块都满足本次研究代价合同，不是统计显著性、业务SLO或一般保证。

与选择数据G相比，最大gap优势明显缩小；第二配对的最大gap仅少0.1213s。H不是全部指标或全部请求受益：第二配对93/128请求完成更慢，99/128请求TTFT更大。原生完整保存、无额外轮转参考仍有更高服务效率和更好的多数请求gap，保留三臂取舍，不能从current配对胜出宣称战胜所有baseline。

六格768/768测量请求全部完成，复用同128篇新文章；不是768篇独立文章。另有396个固定应用warmup请求。输入按原source-order完整文章规则和固定256–3072token资格选择，排除224篇旧train文章，实际460–3064prompt tokens、247295输入tokens；到达0.2s，EOS允许、min_tokens0、cap1024。G用于选择机制及3%/5%预算，H在任何结果前冻结验证；未按EOS、动作或性能筛文或选择repeat。

证据层级：**NATIVE_SERVING_INPROCESS_HOST_MEASUREMENT / REQUEST_LEVEL**。current/eager仅global cooldown20→0，其余selected保存、victim/target、guards、保护、prepare/commit/cancel和份额不变。native_full_native同时改变保存范围及是否添加轮转，是同资源系统参考。精确load/store/重算和恢复启动时间在六轻量格中NOT_MEASURED，不能由gap改善补成内部启动因果证据。G原生组合资格复用，无新增诊断。

## 完整服务及预算

| 格 | 输出tokens / length-stop | capture wall(s) | 输出tokens/s | 请求/s | 平均完成(s) | TTFT(s) | 最大gap(s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| block0-native_full_native | 124445 / 121-7 | 75.74465 | 1642.954 | 1.68989 | 31.12858 | 16.47230 | 11.83480 |
| block0-current | 125069 / 122-6 | 83.72648 | 1493.781 | 1.52879 | 35.72539 | 18.52676 | 3.39782 |
| block0-eager | 124528 / 121-7 | 79.06388 | 1575.030 | 1.61894 | 33.48791 | 17.00314 | 2.93585 |
| block1-eager | 124509 / 121-7 | 79.11934 | 1573.686 | 1.61781 | 33.40167 | 16.97164 | 3.75963 |
| block1-current | 125069 / 122-6 | 79.53303 | 1572.542 | 1.60939 | 33.23802 | 17.26446 | 3.88088 |
| block1-native_full_native | 124470 / 121-7 | 76.46646 | 1627.773 | 1.67394 | 31.48679 | 16.69635 | 12.01122 |

| eager相对current | 配对0 | 配对1 |
|---|---:|---:|
| 最大gap | −0.46197s / −13.596% | −0.12126s / −3.124% |
| 实际输出吞吐 | +5.439% | +0.0728% |
| 请求吞吐 | +5.897% | +0.5229% |
| 平均完成 | −2.23748s / −6.263% | +0.16364s / +0.492% |
| 平均TTFT | −1.52362s / −8.224% | −0.29282s / −1.696% |
| 输出数量 | −541 / −0.433% | −560 / −0.448% |
| 预声明代价及gap要求 | PASS | PASS |

127/128和126/128请求具有相同输出长度，两配对均127/128具有相同stop类型；只有65/128和73/128条输出序列完全相同。文档12425在current两格均到1024上限，eager分别在483/408输出自然停止；配对1文档20167还从current50变成eager106。因此输出变化必须保留，不能把墙钟差当作等工作量加速，亦未证明任务质量等价。实际tokens/s已经使用各臂真实输出量，但不消除输出变化对整个后续轨迹的影响。

## 逐请求分布和原生系统参考

每请求最大gap的等权分布如下，单位秒，以(n−1)q线性插值。全部128请求有定义的gap，未出现0/1输出请求；原协议仍保留无定义情况，不插值同批返回的token。

| 格 | median | p90 | p95 | max |
|---|---:|---:|---:|---:|
| block0-native_full_native | 0.02686 | 1.28166 | 6.15819 | 11.83480 |
| block0-current | 1.70211 | 2.83833 | 3.11817 | 3.39782 |
| block0-eager | 1.23748 | 1.98041 | 2.54599 | 2.93585 |
| block1-eager | 1.09064 | 2.17447 | 2.54953 | 3.75963 |
| block1-current | 1.46940 | 2.75381 | 2.96842 | 3.88088 |
| block1-native_full_native | 0.02705 | 1.63698 | 5.76942 | 12.01122 |

eager/current逐请求gap改善/恶化为87/41、77/51，差值中位数−0.24261s/−0.03520s；最大单请求恶化仍为+2.60527s/+3.36710s。完成改善/恶化为113/15、35/93，TTFT为120/8、29/99。第二配对TTFT均值降低但多数请求更慢，不能用均值替代分布。当前简单eager规则把更多请求的长间隔压低，同时保留个体和整体完成代价。

| 原生参考相对其他臂 | 最大gap变化 | 输出吞吐变化 | 平均完成变化 | 平均TTFT变化 |
|---|---:|---:|---:|---:|
| block0: native−current | +248.31% | +9.986% | -12.867% | -11.089% |
| block0: native−eager | +303.11% | +4.313% | -7.045% | -3.122% |
| block1: native−current | +209.50% | +3.512% | -5.269% | -3.291% |
| block1: native−eager | +219.48% | +3.437% | -5.733% | -1.622% |

原生两格最大gap为11.8348/12.0112s，明显高于eager2.9359/3.7596s；但相对eager有107/106条请求gap更小、21/22条更大，median约27ms，而eager为1.24/1.09s。原生同时有+4.313%/+3.437%输出吞吐、−7.045%/−5.733%平均完成优势。它是必须保留的更集中长停顿与多数请求效率的权衡参照，不能描述为被全面击败。

原生相对eager分别少83/39输出，stop数同为7，但两文档输出长度仍不同；相对current少624/599输出。原生与eager同长度126/128，同序列62/128和72/128。系统差异同时含完整保存与取消额外轮转，不归因cooldown。预声明3%/5%仅用于eager/current；不事后把它改成另一个baseline门槛或忽略native前沿。

## 稀疏恢复、EOS及host边界

| 格 | 实际抢占 | 主动轮转 | 1–2输出后再抢占段 | 结束有效host KV blocks/GiB | drain(μs) | VmHWM(GiB) |
|---|---:|---:|---:|---:|---:|---:|
| block0-native_full_native | 41 | N/A | 1 | 8192 / 16.0 | 8.091 | 18.53054 |
| block0-current | 156 | 120 | 5 | 8192 / 16.0 | 16.624 | 18.53667 |
| block0-eager | 258 | 221 | 5 | 8192 / 16.0 | 9.875 | 18.53518 |
| block1-eager | 275 | 242 | 8 | 8192 / 16.0 | 8.932 | 18.53513 |
| block1-current | 143 | 107 | 3 | 8192 / 16.0 | 11.484 | 18.54821 |
| block1-native_full_native | 41 | N/A | 1 | 8192 / 16.0 | 10.705 | 18.53054 |

零新输出后再次抢占的段六格均为0。短服务段在eager仍出现，不能因最大gap降低就声称已消除反复抢占，也不能由这些段推断可删除的load/recompute成本。每格最大gap都跨越该请求的一次实际抢占及其后新输出，但这些是不同轨迹中的不同请求/状态，不是同一动作前态的分叉对照。

最大gap区间依次为：原生0文档18987的50.50458→62.33938s；current0文档16293的52.99385→56.39167s；eager0文档14859的51.60754→54.54339s；eager1文档18675的70.04923→73.80885s；current1文档17914的62.14442→66.02530s；原生1文档18840的50.85363→62.86485s。ID仅供离线定位，不成为在线选择特征。

**恢复期自然EOS仍未覆盖。** 每格只有最早两个stop发生在该格首次抢占之前，其余4或5个stop发生于已有其他请求抢占之后；但全部stop请求自身的成功抢占记录都是0。四个有rotation的格均无`target_terminal`事件；原生无该策略，相关字段N/A。自然停止与其他请求恢复同场出现，不等于被恢复请求自然终止，也没有新增load/terminal资格证据。

六格实际GPU KV均8,592,031,744bytes（4096usable+1null、2MiB/block），host唯一分配均17,179,869,184bytes/8192blocks。测量前有效host KV均0，request-end和drain-end均8192有效块/16GiB，pending store entries及scheduler store/load/ack均0。H覆盖了所有臂**host末态达到容量**的情形；边界快照不提供完整占用轨迹或逐key替换事件，不能据此声称host-history turnover已经测得。

VmHWM为18.5305–18.5482GiB，含初始化与warmup历史。共享cgroup memory.peak缺失为UNKNOWN；父上限197,568,495,616bytes、swap0不是进程独立host硬限制。KV有效字节是同一预分配buffer中的内容，不能与RSS或cgroup视图相加。

capture+drain wall依次75.74465/83.72650/79.06389/79.11935/79.53305/76.46647s，实际输出率1642.9542/1493.7804/1575.0301/1573.6858/1572.5413/1627.7723tokens/s。drain为8.09–16.62μs，但完成路径上的必要最终store/control已在capture wall中；不能说保存成本为零。该加总排除间隔中的host快照/序列化，不重叠累加内部传输。

六格engine初始化20.595–21.407s，应用warmup0.846–0.892s，进程墙钟105.931–115.120s；每格66个应用warmup请求均保留。原生初始化内部graph/kernel预热也包含在初始化成本。旧r01另付出一次21.268sengine初始化、30.302s组时间，0应用warmup/0测量请求；该迁移断言失败保留，不混入r02六格服务分母。

## 判定、剩余边界与接续

本轮新增证据把“G上选择的eager是否能迁移到独立完整文章和更长有限到达”从未测推进为**两个配对均满足预声明合同**，并观察到优势变小、native多数请求/完整效率优势，以及所有臂host末态满容量。它支持在此输入域保留eager作为停顿优先的简单参照，不要求所有指标同向；不支持新增控制器、普适支配、充分验证的方法或独立新颖性。

完整动作Oracle、任务质量、恢复期EOS、host-key生命周期替换、其他模型/APC/offload后端、稳定无限到达及生产SLO仍未测。r01是人数资格迁移实现错误，非资源或机制NO-GO；r02成功不抹掉其原件/成本。模型方独立核验预声明late64预测，读取相同per_request，不在此重复该次要分析，也不能替代全128主目标。

唯一后续研究判断交root：固定这条简单参照及其三臂取舍，先补齐已识别的、与原生保存兼容的最近邻动作比较入口；没有新的可区分残差前不扫描第三cooldown、不加保护窗口。当前包无追加GPU、诊断、重复或新策略，未声称已有性能Oracle/方法GO。

## 原件与复算

r02接受包SHA `b7360986bc5519b7dfbe99d3c2c511b7766acf2d885c854adf7f6c91aa6d520f`，仅修复旧safe_static人数断言64→128，输入/策略/指标不变。完整失败前件见`../20260915_natural_cadence_holdout_r01/RESULTS.md`，其包与raw均未覆盖。

唯一controller29201/shell29202、前台session72036于1789420535.5771453至1789421203.8267627执行六格，全部COMPLETE、exit0；整组668.24962s。原件归档`execution_weste_26862/natural-cadence-holdout-20260915-r02.readback.tar.gz`，SHA `33e88e55150134bc772772a1df75fa3260af40fec83473ecf9ae3d677461e40a`，19,995,548bytes；220文件及35接受payload全部本地核SHA。双GPU空闲、controller/shell退出，共同flock于2026-09-14T21:27:40.122180+00:00释放。唯一执行/回读/主分析均由`/root/prepare_start_contrast`完成，无后台任务。

主分析为`execution_weste_26862/analysis.json`，逐请求及系统差值完整保留；`report_tables.json`及包外`summarize_report.py`仅整理这些既定指标、已有EOS/host/稀疏原件，未新增观测。原生无rotation时辅助表的空terminal列表表示无该观察器，并非测得0事件。root的paper_view及模型预测消费主分析，不代替原件或再次计数。

仓库根目录复算，使用尚不存在的输出路径：

```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_cadence_holdout_r02/analyze_cadence.py \
  --results refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_cadence_holdout_r02/execution_weste_26862/readback/results \
  --output /private/tmp/natural-cadence-holdout-r02-analysis.json
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_cadence_holdout_r02/summarize_report.py \
  --analysis /private/tmp/natural-cadence-holdout-r02-analysis.json \
  --results refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_cadence_holdout_r02/execution_weste_26862/readback/results \
  --output /private/tmp/natural-cadence-holdout-r02-report-tables.json
```
