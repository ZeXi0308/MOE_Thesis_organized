# commit重检：独立adapter副本已接入，尚未GPU资格化

新增参数 `recheck_commit_funding=False`，仅位于本目录候选副本；共享源码、原资格包和已接受轻量包未修改。默认关闭，分支仅在开启后执行额外资金检查。补丁针对本E冻结adapter，不能不核对版本就套入其他源。

READY且直接资金足够时改用direct_resume阶段：保留victim及其仍在使用的块；建立原target/output_start与队列提升；保留recovery_guard。旧victim有in-flight load仍拒绝。不强制释放源块，不伪造victim preempt、flush、applied_rotations或tracker rotation。原已消耗的cooldown不回退，prepare保存成本不扣除。完成本次schedule后清除plan；后续沿原首新输出解除保护逻辑。

资金不足或默认关闭继续原swap，包括原生preempt通知、pending store flush和实际rotation计数。已有commit拒绝条件优先，不以free足够绕过身份、状态、ownership与open-population条件。

检查执行的是候选文件中实际begin/schedule函数体，AST仅将closure nonlocal改为测试namespace global。native与connector由CPU替身提供；验证5种分支以及计划/计数清理。它没有运行完整install、pinned native AST变换或真实异步传输，因此状态为CPU_COMPONENT_CHECKED_GPU_UNRUN，不称native serving验证。

复算（仓库根目录）：

```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260915_natural_native_full_gate_r01/commit_recheck/test_commit_adapter.py
```

CPU_CHECK.json记录来源版本与未验证部分；候选完整文件、commit_disposition.py与统一diff同目录。无新controller、运行包或GPU组。

下一资格必须覆盖真实target加载/正常decode、victim保留、guard期间无额外抢占、metadata和末态drain；随后看后续更多running带来的完整服务代价。按已提出的最小建议，与主方选定保存底座进行同状态或真实独立执行对照，不抢占正在进行的selected/full组。


## 运行入口接入（本轮补充）

run_commit_recheck_candidate.py为E原诊断runner的独立副本；统一补丁commit_recheck_runner.patch新增--commit-recheck，默认关闭。配置写recheck_commit_funding、传给真实adapter并检查返回开关一致；environment source_sha256增加commit_disposition.py，避免机制代码未入执行记录。CLI --help实际执行通过，候选源码compile通过；未加载模型或启动GPU。

执行组被接受后，才在唯一新目录复用E包并应用两份补丁（保持正式文件名run_streaming_recovery.py/staged_store_rotation.py），加入commit_disposition.py。不能直接运行当前异名候选文件并误以为它会加载候选adapter，不能覆盖旧E包。届时运行入口为：

```sh
python run_streaming_recovery.py --inputs inputs --warmup-inputs warmups --output-dir ../results/diagnostic-recheck --commit-recheck
```

这是一条待在接受后的独立包内执行的命令，当前GPU_UNRUN，不是已经可在Mac上运行的模型验证。

唯一建议预算：一格64请求native-full诊断资格，沿E输入、0.2s、EOS、4096usable块、16GiB host和30/20/30。先确认direct_resume实际触发、target真实新输出、同次提交没有victim preempt/forced计数、pending store/load按native生命周期结束，保留全部后续peer/抢占结果。零触发记录NO_ACTION，不循环改压力或挑step。此格不与旧E诊断墙钟算收益，不替代后续同底座完整服务比较；不在已接受四格窗口插队。
