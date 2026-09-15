# Native完整增量保存：下一单格资格的兼容性

**结论：源码层可做一个最小保存范围开关；尚未取得native-full GPU资格。直接移除selected限制仍会触发一处不适用的诊断断言，不能原样启动。** D的64请求、0.2s到达、EOS允许/cap1024、4096usable GPU块、16GiBhost、current调度及保护语义全部固定。旧`20260914_native_offload_baseline_r01`是prompt-only，不充当此处full基线。

**最小开关。** 在新版本的`pkg/staged_store_rotation.py`增加`store_scope=selected|native_full`：selected保留`limited()`；native_full不覆盖`cs._calc_num_offloadable_tokens`，使用原始绑定方法。D已经设置`offload_prompt_only=False`。原生`_build_store_jobs()`对本步scheduled及finished请求，根据本步计划完成的token范围/当时已知终止状态、完整chunk及原生保存进度产生增量job；不是所有KV永久驻留，也不读取未来EOS。保留同一次prepare→commit、victim/target排序、30/20/30/.90/8及token份额，不趁机删除prepare一步。

**确定需要调整的检查。** `native_store_delta.py::inspect_store_delta()`只接受`plan.source_blocks`，即prepare之前的完整prefix。native-full合法保存可包含本步新完成的块，例如prepare时computed≡15(mod16)，本步decode后多出一个完整块；继续调用会报`Store outside prepared complete prefix`。该selected专用检查只能在selected分支使用；full分支记录原生job的request、注册、source/key覆盖及完成证据，覆盖范围按原生本步语义解释，不能再截回plan边界，否则仍是受限保存。

**pending/flush/保护兼容依据。** 同版本原生`build_connector_meta()`把所有被抢占请求的pending store及被重用块相关job并入`jobs_to_flush`；worker先提交延期store再wait，正常store延期至下一engine步启动。当前adapter只检查victim pending集合是flush集合的子集，不屏蔽其它job，因而无需改flush或metadata。保护只保留GPU增长空间及等待load完成，不限制其它请求的原生store；已有mixed-prefill/skipped队列取消与保护断言继续保留。如果full保存引出的新load/queue状态触发这些断言，应保留为兼容性失败，不能为该基线单独放宽共同调度规则。

**唯一待资格确认项。** 下一`diag-native-full`记录全范围增量store、host容量/有效条目、load及剩余重算、抢占/块重用flush、终止后的final-store与drain。更多store、host淘汰或必要传输成本是开关的真实后果；不能漏计，也不能据源码称它们已被D验证。初始化和warmup保持一致并reset connector；完整请求与后续drain分别记录。本单格仅判路径与共同调度是否兼容，不与D诊断耗时直接宣称性能胜负。

源码依据：`pkg/staged_store_rotation.py`、`pkg/native_store_delta.py`、`pkg/native_offload_observer.py`；原生快照为`refine-logs/expert_saturation/outputs/admission_capacity/20260914_kv_roundtrip_feasibility_r01/native_offload_source.json`中的offloading scheduler/worker。其scheduler、worker、config文本SHA均与D的`pkg/runtime_source_hashes.json`一致。本次只读定位并新增此说明；未改接受包、未造新包、未运行测试或GPU。
