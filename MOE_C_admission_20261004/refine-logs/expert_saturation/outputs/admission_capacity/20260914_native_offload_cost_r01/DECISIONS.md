# 默认prompt-offload执行税定位

原四格发现：所有输出一致，同臂执行序列一致；两组on/off首次调度分叉均1026，之前engine额外3.258/1.594s，其中scheduler1.331/0.518s。不能归因于之后恢复改变工作量，也不证明物理布局一致。第一on全程漂移保留，不能只扣几个尖峰宣称收益。

本组固定原native on/16GiB/APC off/d6/旧请求及所有warmup、reset、load/store观测不变。仅cost-profile=0/1/1/0，四个新引擎。作用对象是观测，不是策略；保持默认prompt-only，不启用full-decode。排当前首输出恢复四格之后，启动前重新核验。

主问题：额外成本主要位于每步connector调度、worker传输协调，还是尚未覆盖的engine执行？

9个connector方法加engine.step的有限主线程perf_counter/thread_time观测，嵌套项互斥计费。无CUDA同步/事件、不profile全模型、不更改返回或异常。其他线程不计；thread_time不是GPU利用率或所有CPU线程总耗时。engine剩余项仍不能直接称GPU税。CPU嵌套会计/返回/异常/恢复检查通过，安装源码方法名已核对，实际native接线UNRUN。

保存所有重复/失败/编译与主机缓存/请求、实际KV和transfer事件；主测后drain仍记录。用原生成序列和调度signature比较profile off/on是否改变执行，再比较完整wall与completion评估观测代价。若profile改变路径，保留结果但不把两臂差异全部当纯观测开销。

解读保持MEASUREMENT_ONLY：不同方法的inclusive时间不相加；只加exclusive项并核对每个engine.step守恒。前1026步仅在新raw确认前缀后使用，不硬编码为必然相同；用实际新轨迹定位。无新controller，无方法GO。

下一唯一动作由结果决定：可消除的CPU重复工作则最小修正并同后端对照；若主要剩余在engine执行，才做针对布局/worker的最小对照。不因一个profiler结果判死offload。
