# 原生offload调用级成本定位

Verdict: MEASUREMENT_ONLY / NATIVE_HOST_PROFILE_DIAGNOSTIC。单格COMPLETE/32请求，冻结500..531签名通过，32条完整输出与旧on一致；GPU已释放。发现观测器自身是需要消融的成本路径，仍未证明offload的不可消除成本或净收益。

cProfile主线程32调用scope合计975.870ms，不可与原Torch profiler时长直接比较。32次torch.Event.synchronize累计478.4ms，与GPU工作重叠；不能当可移除主机税。

观察到15次完整GC配对、0未配对：12次gen0、2次gen1、1次gen2；gen2在调用530持续68.947ms，collected=0/uncollectable=0。其他GC均短于4.2ms。其量级与此前另一次运行66.556ms同步后尖峰相近，但不同运行、不同调用、不同profiler，不能回填前次GC归因。观测改变分配与GC时机。

一个具体调用链：memory_telemetry.state 64次累计107.4ms → request_state 2050次累计104.0ms → kv_cache_manager.get_blocks 2050次累计91.0ms。installed kv_cache_coordinator.py:359–366将各single_type_manager.req_to_blocks条目包装为tuple，再由观测器仅提取len。genexpr self 77.2ms可能包含执行期间触发的GC/运行停顿，不能直接声称生成器运算本身花了77ms。以上链条重叠，不可相加；更不能称107ms全是offload新增税，因为此单格没有profile-off对照。

scheduler.schedule本体累计148.7ms；native_capture.schedule累计274.0ms，其中memory_telemetry.schedule累计258.2ms。数据采集是完整请求路径一部分，必须单独检验；尚未将成本归给某个connector算法。cProfile inclusive计数与上下文/递归相互重叠，部分函数累计大于所选scope，排名仅供定位，不构成互斥会计。

唯一下一最小实验：同native-off/on底座，对逐步KV观测做等价低分配实现消融——直接从已存在的manager.req_to_blocks取长度，保持输出字段/采样频率/前后时刻不变，避免为只读块数重建包装对象。先CPU核验逐状态值一致，再无cProfile/Torch profiler的真实交错重复，判断采集改动是否影响完整请求成本；不禁用GC，不删不利请求或原始观测。若无收益，不继续救该微优化。该变化只优化观察器，不包装为新的调度机制。

边界：旧d6受控固定长度/单5090/native进程内运行；未证明第二workload、free generation质量、full-decode offload、轮转connector兼容或多卡。最强对照仍是此前相同资源native offload关闭的真实四格；本格无Oracle或方法GO。失败类别从笼统offload主机税收窄为观测成本与runtime成本尚待因果分离。

原件归档SHA256 ee1c2c925fa62ede8c379b318d06890f7920a1a97f1666347c62318b414215df；analysis.json由analyze_python_window.py生成，pstats及函数完整调用者保留。本次执行者定向核验，非fresh独立审计。
