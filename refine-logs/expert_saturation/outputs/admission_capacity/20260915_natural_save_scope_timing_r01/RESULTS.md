# Natural save scope: complete-request result

**Verdict: MEASUREMENT_ONLY；四格、256/256请求全部完成。当前运行域不支持把 native_full 升级为整体更强的长停顿基线。** 相对 selected，native_full 在两配对中的最大生成间隔分别增加13.19%和63.96%；多数请求的间隔下降，但尾部分布变差。输出吞吐、平均完成和TTFT的方向在两配对间翻转。这里形成的是完整服务取舍，不能声称 selected 在所有指标上支配 native_full，也不能把原生完整保存判为普遍无效。

唯一执行、归档和主分析：`/root/prepare_start_contrast`。接受包SHA256：`8f9f0f3414feba5ca5f9d62fe668a3623a059b2defb8aecdfa1db5fcb1759de7`。实际顺序为 selected/full/full/selected，没有诊断追加、重跑或替换。26862前台controller21125、shell21126；开始1789415325.9303913，终态1789415599.5051074。两配对冻结分析均为COMPLETE、无分析合同错误。

证据层级是 **NATIVE_SERVING_INPROCESS_HOST_MEASUREMENT / REQUEST_LEVEL**：原生vLLM0.26执行、相同自定义current调度、同一稀疏抢占观测；不是生产网络流量。64篇完整自然文章按0.2s进入，`ignore_eos=False/min_tokens=0`，1024只是配置cap。唯一因素是保存范围：selected沿用准备阶段的victim完整前缀限制；native_full不覆盖原生`_calc_num_offloadable_tokens`且`offload_prompt_only=False`。共同victim、保护、准备/提交、原生份额及30/20/30 guards均保持接受版本。原生路径资格复用D/E，不用其诊断墙钟作本组性能比较。

| 实际格 | 完成/输出tokens | length / stop | capture wall(s) | 输出tokens/s | 请求/s | 平均完成(s) | 平均TTFT(s) | 最大生成gap(s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| block0-selected | 64 / 59102 | 57 / 7 | 36.52676 | 1618.05 | 1.75214 | 18.49190 | 4.24731 | 2.61064 |
| block0-native_full | 64 / 58617 | 57 / 7 | 35.90265 | 1632.67 | 1.78260 | 18.13623 | 4.22427 | 2.95497 |
| block1-native_full | 64 / 57680 | 56 / 8 | 36.73067 | 1570.35 | 1.74241 | 18.37529 | 4.27649 | 3.83465 |
| block1-selected | 64 / 58617 | 57 / 7 | 36.01343 | 1627.64 | 1.77711 | 18.10505 | 4.19754 | 2.33883 |

| native_full 相对 selected | 配对0 | 配对1 |
|---|---:|---:|
| 最大gap | +0.34434s / +13.19% | +1.49582s / +63.96% |
| 输出吞吐 | +0.90% | −3.52% |
| 请求吞吐 | +1.74% | −1.95% |
| 平均完成 | −0.35567s / −1.92% | +0.27024s / +1.49% |
| 平均TTFT | −0.02304s / −0.54% | +0.07894s / +1.88% |
| 输出量 | −485 / −0.82% | −937 / −1.60% |

两配对分别只有42/64、43/64条输出token序列完全相同；各有62/64条输出数量相同。实际终止/序列受到各自执行轨迹影响，因此不能把较短墙钟或完成时间称为同等生成工作量的加速。所有输出、stop原因和不利配对保留。这里没有追加质量判定。

EOS边界按现有稀疏raw核对：四格所有`stop`请求自身均没有成功抢占记录，`target_terminal`事件均为0。全组首次成功抢占时刻依次8.19895/8.66727/7.98640/8.53030s；此前已完成的stop数为6/7/8/7。唯一晚于该时刻的stop是block0-selected文章1582，于10.40705s结束，但它自身没有被抢占。因此本组仍**没有覆盖被抢占/恢复请求的自然EOS或保护期间terminal释放**；也没有测精确load/恢复启动相对EOS的时序。全组其他请求开始抢占后发生EOS，不等于EOS发生在该请求恢复之后。

每请求最大生成间隔的分布如下（秒；对64个请求等权，分位数按排序后`(n−1)q`线性插值，属于既定逐请求分布的描述，不是新增SLO）：

| 格 | median | p90 | p95 | max |
|---|---:|---:|---:|---:|
| block0-selected | 0.03249 | 1.17881 | 1.91023 | 2.61064 |
| block0-native_full | 0.02601 | 1.42898 | 2.49453 | 2.95497 |
| block1-native_full | 0.02693 | 1.44996 | 3.56250 | 3.83465 |
| block1-selected | 0.46286 | 1.21501 | 1.76055 | 2.33883 |

配对0中34条gap下降、30条上升；配对1中36条下降、28条上升，均无完全相等。配对差值的中位数分别−0.000042s、−0.20666s，最大单请求恶化+2.55030s、+2.67983s。完成时间改善/恶化请求数分别54/10和21/43；TTFT分别41/23和15/49。不能只呈现平均或只呈现最坏请求。

稀疏记录保留了实际`_preempt_request`调用及返回输出。四格成功抢占数依次28/28/28/27，策略主动轮转20/19/18/19；新增1–2个输出后再次抢占的短服务段依次 **2/3/0/0**，零新输出后再次抢占均为0。故“更完整保存消除短服务”没有稳定支持，也不能仅靠一次诊断中为0就增加保护窗口。短段只证明已交付服务很短；未测具体load/recompute成本，不能据此算可删除时间。

四格的最大gap都跨越同一请求的实际抢占与其后首个新输出。前三格最坏请求为文章0748，间隔分别13.75260→16.36324、14.11306→17.06804、14.35794→18.19259s；最后一格为文章0913，13.22972→15.56856s。抢占入口分别紧随区间左端约0.55–0.74ms。这个观察将剩余长停顿定位在“被抢占至重新交付新输出”的路径，但**不区分可执行启动等待、原生pending/传输、重算与后续计算分配**；不把整个间隔当可删除上界，不依赖请求ID设计在线策略。

四格实测GPU KV tensor均为8,592,031,744bytes，即4096usable blocks加1null；16tokens/block、2MiB/block。唯一host KV分配均为17,179,869,184bytes（16GiB/8192blocks），两臂相同。warmup及缓存重置后有效host KV均为0。

| 格 | 请求结束有效host KV | 请求结束pending store/load/ack | post-request drain | 进程VmHWM(GiB) | 初始化 / warmup(s) |
|---|---:|---|---:|---:|---:|
| block0-selected | 1396blocks / 2.72656GiB | 0 / 0 / 0 | 8.535μs | 18.33611 | 20.95473 / 0.86965 |
| block0-native_full | 8192blocks / 16GiB | 0 / 0 / 0 | 10.217μs | 18.35596 | 21.10184 / 0.85617 |
| block1-native_full | 8192blocks / 16GiB | 0 / 0 / 0 | 9.551μs | 18.33383 | 21.19380 / 0.87147 |
| block1-selected | 2499blocks / 4.88086GiB | 0 / 0 / 0 | 10.553μs | 18.37499 | 21.07751 / 0.85491 |

drain前后有效KV相同且pending为0；最终store/原生控制成本已在capture路径中，不因post-request drain很短就记为无成本。capture+drain依次36.52677/35.90266/36.73068/36.01344s，对应输出率1618.0463/1632.6646/1570.3494/1627.6423tokens/s。这个非重叠加总排除其间host快照/序列化；不再叠加内部传输时间。66条warmup/格及初始化、缓存清理、关闭成本另有raw/timing留痕，完整进程墙钟约66.06–67.23s/格。

VmHWM包含该进程初始化/warmup历史，不能叫测量episode的精确峰值。host有效KV只有边界快照，selected的过程中峰值未知；full在结束时达到已分配的16GiB上限。共享父cgroup上限197,568,495,616bytes、swap上限0，`memory.peak`不存在，保留UNKNOWN；不是每进程独立硬限制。KV、RSS及cgroup占用重叠，不能相加。详细KV jobs、传输次数、重算及恢复启动时间本组均NOT_MEASURED。

本轮新增证据改变的是**强基线选择和短服务解释**：完整原生保存具有真实有效缓存覆盖，却没有在这个固定资源/current运行域降低最大停顿；稀疏短服务也不是稳定消失或稳定恶化的现象。未测Oracle或动作反事实，尚无可兑现headroom上界。结论限制在一个模型、一个受控0.2s到达点、两个配对；不外推其他负载、自然EOS压力或其他offload后端及APC开启域。没有性能比较失效、资源越界或未完成请求；主要限制是取舍、输出轨迹变化与缺少细分因果计时。下一问题由root在该证据上统一选择，本执行方不自动补诊断、重复或扫描。

原件：[唯一回读](execution_weste_26862/readback/)、[主分析](execution_weste_26862/analysis.json)、[归档核验/释放](execution_weste_26862/readback-verification.json)。唯一归档SHA256为`be54df2b3c781e3d17e793179b31763b603122c275aba557887be827762f0c59`，6,567,737bytes；156个归档文件及29个接受payload全部核SHA。controller/shell已退出，两GPU无计算进程，共同flock于1789415641.8005402释放；完成本地回读后执行方明确释放整组GPU窗口。原件与接受包未改。

从仓库根目录复算，输出须使用尚不存在的新文件（冻结analyzer拒绝覆盖）：

```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_save_scope_timing_r01/analyze_scope_timing.py \
  --results refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_save_scope_timing_r01/execution_weste_26862/readback/results \
  --output /private/tmp/natural-save-scope-timing-reanalysis.json
```

逐请求差值及短段直接位于`performance_comparisons[].request_deltas`和`cells[].sparse_recovery`；分布由`cells[].requests.requests[].max_engine_return_gap_s`按上文公式得出。最大区间直接取各格raw中相邻`token_times_s`最大差，并与同请求`preemption_events.method_entered_s`对齐。EOS边界用`stop_reason=stop`请求的`completion_s`比较全部成功抢占的最早`method_entered_s`，同时按request_id核对该请求自身的抢占记录；这些是观察性定位，没有新增实验或未来信息策略。
