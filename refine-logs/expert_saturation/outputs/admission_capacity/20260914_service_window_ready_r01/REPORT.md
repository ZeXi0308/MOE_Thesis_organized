# 窗口状态接口：占块 / computed 与可执行前缀分开

Verdict: **STRUCTURAL_INTERFACE_CORRECTION**。原生异步恢复的数值计数不足以决定“下一步能否产生新输出”；最小窗口状态需保留原生load阻塞状态。修正已落入模型、8个定向测试通过，没有新增GPU执行或完整服务收益。

本轮HEAD仍de64dae5，主工作区包含其他会话修改；只在自有`/private/tmp/moe-window-main-20260914`修改窗口模型/测试，集成前确认两份源仍一致。已读最新共享台账、GPU协调、四格streaming结论和恢复底座新结果。streaming的低暴露结果保持，不调整其输入或压力。

## 新增问题与真实反例

唯一问题：**已有KV块数、上下文和native computed计数，是否足以说明目标能够执行恢复后的剩余输入？**

复用单次保存原件`20260914_selective_store_once_r01/readback/results/selective-on`，提取request3571第一次load完成边界：

| 字段 | 第331步调度前 | 第332步调度前 |
|---|---:|---:|
| native computed | 3296 | 3296 |
| 当前history | 3307 | 3307 |
| 已持有块 | 206 | 206 |
| 已输出token | 235 | 235 |
| 本步实际调度位置 | 0 | 11 |

job116的worker完成记录10257734.882203728处于331调度结束10257734.863563443与332调度开始10257734.884458791之间。第332步的11位置分为10重算位置和1新decode位置。第331步已经占块并填写computed，却不能把3296直接当作可执行前缀。

第二次load也保留：job117在1040调度后完成，1041首次执行84位置（83重算、1decode）；本例前后数值状态不同，不伪称第二个“相同计数”反例。两事件均见[counterexample.json](counterexample.json)。

依据已保存的安装版本源码`20260914_load_ready_contract_r01/native_source.json`，scheduler.py:978–998在load未完成时已分配块、填写num_computed_tokens并阻塞请求；2622–2642接收worker通知；2586–2601在后续调度中由原生promotion解除阻塞。raw memory snapshot没有request.status字段，所以阻塞边界是**源码合同加实际完成通知**推得，不能冒称直接采到了该状态。事件时间只用于既有轨迹定位，不作为未执行策略的未来通知。

## 最小改动

`service_window_model.Request`加入`waiting_for_remote_kv`，默认False，原重算域行为保持。

- 等待原生load时，gpu_blocks保持实际持有量；computed_tokens只能填写当前可执行的有效前缀，不能复制尚未ready的native计数。错误导入会明确拒绝。
- 已在等待load的target返回`readiness_ok=False / WAIT_FOR_NATIVE_LOAD_PROMOTION`，不出具可执行资格；这表示等待原生状态推进，不是未来窗口不可行。
- 正在等待load的peer不能被动作profile计为新输出；未参与输出的等待peer仍持块、输出年龄继续增长。
- 只有原生调度器处理当前已可见的完成通知、完成promotion并校验前缀后，调用方才能更新有效computed并清除阻塞。模型不自己推进通知或根据预计DMA时长清除标志。
- 尚未开始的计划恢复仍可进行条件窗口分析，但其first-output路径必须包含load完成通知及下一次调度边界。时间来自合法的动作profile；本次没有构造新时延预测。

测试包含真实计数形状3296/3307/206及明确标为合成的路径时间：pending不出具执行资格，ready后恢复源可用，但`C_remaining=None`时仍不给摊销资格。另检查pending计数误导入及等待peer不能产生新输出。原6个窗口/混合服务/增量KV/重叠/沉没成本检查继续通过，共8个；不做覆盖率工程。

这是模型输入合同修正，**尚未接入在线native调度器**；没有修改原生pager、删除connector保护或更改已冻结GPU包。

## 成本仍不能数值化

共享新结果已经说明：单次保存少6591重算token，但完整调用仅1869→1867；少5次含重算调用同时多3次纯decode，135输出转移。完整轮转的least/most重算调用分别仍产4226/4739新输出，most虽重算更多，总调用却更少。这要求保留混合服务分母。

因此不把4–5s含重算调用时间、逻辑传输字节或1/2/4步条件情景填入窗口`C_remaining`。模型原本已支持动作独立路径、peer输出和未知边际成本；本轮不改成本算术。保存收益需与同恢复排序、同底座关保存比较；不能把排序分支的33调用差归给保存。

## 结束合同

- **Evidence type / measured:** 既有native scheduler/worker事件的只读定位与结构模型回归；无新性能测量。
- **Not measured:** load时延预测、动作条件成本界、在线窗口排序、完整延迟—效率收益、第二模型服务结果。
- **Strongest baseline:** native/most冻结基线保持；本次无新策略比较。
- **Oracle/headroom:** 未知。ready正确性是必要条件，不证明窗口值得执行。
- **Claim ceiling:** 当前接口不能把占块或native computed升级为ready；没有新系统方法主张。
- **Failure category:** 输入状态语义不完整已修正；科学收益仍未验证。
- **Reopen/continue condition:** 当前可见ready状态正确进入接口，且强简单策略后仍存在可执行动作的完整服务残差；不得复用未来完成序列做候选Oracle。
- **One next GPU experiment:** 继续原已暂存的funding-filter六格，区分当前资源资格修正能否改善完整服务权衡；仅赢least未赢most则吸收为基线修正。它仍GPU_UNRUN，现有Qwen整组终态未获本会话现场确认，不能启动副本。

本轮只读连接检查返回已授权共享ControlPath不存在；没有尝试其他凭据来源、没有远端启动、没有后台候卡。现有Qwen状态文件不是实时存活证据，不能据此宣称verified wait或已释放。

直接回答问题：**数值KV状态不足以认定恢复后可执行；必须区分原生尚未ready的已占块状态。补齐这个边界消除了一个输入误判，但没有证明服务窗口改善了完整请求权衡。**

命令：`python3 extract.py`（只读原件，输出以exclusive create保留）；在experiments/admission_capacity执行`python3 -m unittest test_service_window_model`，8/8 PASS。无新GPU/审计矩阵。
