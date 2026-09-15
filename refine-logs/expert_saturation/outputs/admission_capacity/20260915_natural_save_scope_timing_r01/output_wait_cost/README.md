# 直接抢占事件的累计输出等待

状态：**CPU_VERIFIED / 四格直接事件分解已完成**，见[实际结果](RESULTS.md)。本方只补充累计等待会计；执行与主性能分析仍归CURRENT唯一负责人，冻结包不改。没有新的GPU组或诊断回读。

已有cooldown对照表明最大间隔下降时，累计间隔可能上升；旧补充分析依赖诊断事件向相同轻量输出轨迹投影。本组已用共同稀疏事件直接观测抢占，因此新增`partition(raw)`只使用同一次运行的`preemption_events`和实际返回token时间。它不重复主分析的恢复段计数，而把每个输出间隔恰好计费一次，并保留该间隔中的抢占次数。

完整请求的守恒式为：

```
完成延迟 = 首输出前等待 + 含成功抢占的生成间隔
         + 其余生成间隔 + 最后输出到完成的尾部
```

含抢占的间隔包括抢占前后执行、排队和可能的恢复，不能称纯恢复税。成功调用`_preempt_request`也不能证明已经执行load/recompute。跨请求累计是相互重叠的request-seconds，不是episode墙钟。

失败请求以观察终点截尾，保留未返回engine call中的成功抢占和失败方法事件；不拿未闭合尾段冒充完整生成间隔。首输出前抢占单列，未输出请求不伪造TTFT，未来尚未到达请求保留但观察时间为0。同批返回多个token只使用引擎返回边界，不插值内部ITL，不重置实际最后新输出时间。

本轮3项定向CPU检查已通过：

- 同一2秒间隔中两次抢占仅计2秒，保留次数2；同时处理多token返回与1秒完成尾部，拒绝错配last-output。
- engine call未返回、失败抢占、无输出等待和未来未到达请求全部保留，完整服务比较标False。
- 首输出前抢占与零输出终止保留不同状态，不伪造生成间隔。

从仓库根运行：

```bash
python3 -m unittest discover \
  -s refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_save_scope_timing_r01/output_wait_cost \
  -p test_partition.py -v

python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_save_scope_timing_r01/output_wait_cost/partition.py \
  --raw PATH_TO_CANONICAL_CELL_RAW_JSON \
  --output /tmp/direct-output-wait.json
```

原预定问题已用canonical四格回答：累计含抢占间隔、其余生成间隔、首输出等待各如何变化，代价是否集中于相同请求？结果和复算见[RESULTS.md](RESULTS.md)。主表/回读未重复；不同臂无需相同实际输出或KV轨迹，输出量变化仍由主评价保留。本组件本身不证明保存范围或调度器的性能收益。
