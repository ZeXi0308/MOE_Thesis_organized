# TriageAudit-MoE v2 严格 Code Review

日期：2026-07-23  
范围：Level 0/1 代码、Level 2 runner 与 GPU 前 tiny smoke  
结果：**CPU REVIEW PASS / TINY SMOKE ENGINEERING PASS / FORMAL MECHANISM PROBE APPROVED**

## 1. 代码执行链路与实验逻辑

1. `triage_manifest.py` 只做 LF 换行规范化，按 text SHA-256 去重、排除历史 hash，再生成 calibration/sealed 零交集 manifest。
2. `run_triage_audit_gpu.py --mode calibration` 对每个模型的同一批 32 篇文档执行 BF16 prefill，收集固定 8 项 router summary + prefill NLL；decode 以 always-low canonical KV 前进，每步从该 canonical KV 做 non-aliased BF16/INT4 same-state fork，仅保留 low branch，产生 document-CVaR90 discrepancy label。
3. `finalize_calibration.py` 只接受 `split=calibration` 行，要求两模型各 32 篇、document hash 集合完全一致；执行 frozen ridge、2,000 次 document bootstrap assignment stability，并锁定每模型 P90 audit threshold。H0 任一失败时 sealed 不可运行。
4. `run_triage_audit_gpu.py --mode sealed` 先只收集 prefill features，使用 frozen predictor 分配 `{2,4,8}`；hash arm 仅对 triage 的 exact period multiset 做 document-hash 重排。每篇文档只读共享一份独立 BF16 reference logits，8 个 candidate arms 各有 fresh prefill 和独立 canonical KV。
5. `triage_executor.py` 对 audit step 计两次 candidate forward；非 audit step 只计 served candidate，另一路仅计 diagnostic cost。dangerous label 始终来自本 arm、本 step、同 pre-step KV 的 BF16/INT4 discrepancy，不反馈给非 audit 决策。
6. `analyze_triage_results.py` 先验证两模型、8 arms、64 篇文档、period/phase、AuditState replay、dangerous 重算、step/document metrics 和 forward/clone counters 全部闭合，然后才进入 5,000 次 paired-document bootstrap + Holm。

## 2. 已确认正确的关键实现

- **Exact budget control**：`budget_matched_hash_periods` 保留完全相同的 period multiset；runner 和 analyzer 双重断言 histogram 闭合。
- **Common phase**：所有 periodic arms 仅使用 `sha256("audit-phase-v2"|document|period) mod period`；policy name 不进入 hash。
- **Same-state semantics**：每次 fork 都 deep-copy cache，检查原 cache/high/low 的 tensor storage 不 alias，两 branch cache length 必须同时 `+1`。
- **Unique canonical KV**：只有 served branch 进入下一 step；未 served branch 没有任何持久引用。非 diagnostic 路径可用单 branch 直接前进。
- **Diagnostic non-interference**：CPU fixture 已验证 diagnostic on/off 的 action trace、每步 served-logit SHA-256 和 final-cache SHA-256 完全一致；tiny smoke 会在真实 OLMoE cache 上再验证一次。
- **INT4 intervention**：仅 expert FFN linear 使用准备一次的 per-output-channel symmetric RTN `[-7,7]` dequantized BF16 weight proxy；router/attention/LM head 不变。runner 检查 expected linear count、backend activation count、expert proxy 真实命中和非零 discrepancy。
- **Hook equivalence**：tiny smoke 必须通过 native-original vs patched-full 的 max-abs logit error `<=0.02` 且 token KL `<=1e-7`；route recorder 必须每个 MoE layer 恰好命中一次。
- **Cost ledger**：candidate high/low、diagnostic high/low、physical high/low、audit、clone、served high/low 分列记录；正式证据不使用 wall time。
- **Statistics**：H1/H2/H4 使用逐文档 paired effect 中位数；H3 在每次 document bootstrap 内重建 policy-level mean quality-cost envelope，不使用 outcome-aware per-document baseline oracle；p-value 使用 null-centered one-sided bootstrap，再做 Holm。
- **Leakage gates**：calibration finalizer 拒绝任何非 calibration 行；sealed runner 必须读到两模型 H0 PASS 且 config-bound 的 lock。
- **Reproducibility/artifacts**：model/tokenizer revision、seed、BF16、TF32 off、deterministic algorithms、CUBLAS workspace、config/source/protocol/data hash、fsync JSONL resume 和 no-overwrite 都有明确实现。

当前审查绑定值：

- source aggregate SHA-256: `05023e8a7bcc8632461663440a1952cf72da330e1637fe2f19b1cfedc8dbccc1`
- config SHA-256: `286ec92484f394389a94a5419a6ebba6ba674ab114f85058a83b7a8660f0a563`

## 3. 潜在 bug、偏差与混杂因素

### 已在本轮修复

1. calibration 曾传入 `inf` threshold，会被 executor 的 finite check 拒绝；已改为 always-low 不读取的 finite `0.0`。
2. 原 executor 无法关闭 diagnostic，因而不能证明非干扰；已新增 single-action path 与 logits/cache fingerprint。
3. H3 曾对每篇文档事后选 baseline，会构造不可部署 oracle；已改为 policy-level Pareto envelope。
4. analyzer 曾在缺失 always/full-shadow 或损坏 step log 时仍可进入 bootstrap；已增加 8-arm 完整性与逐 step 重放。
5. 显存超 guard 但未 OOM 曾会被记为 `INVALID_RUN`；已改为 `BLOCKED_MEMORY`，不混同科学 NO-GO。
6. smoke 的主行曾可能在 invariance check 前落盘；已改为所有 hook/intervention/invariance check 通过后才原子追加一行。
7. 首次 tiny smoke 在模型构造前发现 `transformers==4.53.3` 不接受通用 `dtype=` loader 参数；没有执行 forward，也没有采信输出。已改用该冻结环境支持的 `torch_dtype=`，并新增 mock loader 回归测试，检查 pinned revision、offline、本地 BF16 和 CUDA eval 契约。

### 仍存在，但 tiny smoke 正是对应的快速证伪

1. Hugging Face 实际 `DynamicCache`/legacy tuple 是否可 deepcopy、是否存在隐藏 storage alias，CPU fake cache 不能代替真实验证。
2. 当前 remote transformers 版本下两模型的 MoE class 形状、expert linear count 与 shared patch 兼容性只能在真实模型上确认。
3. OLMoE BF16 model + 完整 prepared INT4 proxy + prefill + dual cache fork 的真实 peak reserved memory 无法由 CPU 推导。

### 正式 calibration 前阻断项的关闭证据

1. source-bound `memory_certificate.json` 已生成并经独立 artifact audit：peak reserved `24.9648 GiB`，低于 `29.8567 GiB` guard；patched-full logit/KL error 均为 0，INT4 discrepancy 非零，diagnostic invariance PASS。
2. 已生成 `gpu_run_approved=MECHANISM PROBE ONLY` 的 source/config-bound `approval_v2.json`。该 approval 不批准 sealed；calibration H0 失败时 sealed 仍由 runner fail-closed。

数据阻断项已于本轮关闭：已从 pinned revision 生成 28,911 篇 train articles，保守排除历史 shuffled 前 1,000 篇后有 27,904 篇双 tokenizer `>=96` tokens 的候选；冻结 32 calibration + 64 sealed，两 split 及历史 registry 均零交集，sealed manifest 权限为 `0600`。

本试验仍有 teacher forcing、W4A16 dequantized BF16 quality proxy、单 GPU、无 native INT4 kernel、无 continuous batching 的明确边界；即使 Gate M GO，也不能直接声称 TPOT/P99/能耗收益。

## 4. 必须修改项与建议修改项

### tiny smoke 前 MUST FIX

- 无。上述代码阻断项已修复，source/config hash 已重算，CPU 回归通过。

### formal calibration 前 MUST FIX

- 无。tiny smoke 与 source-bound approval 已关闭上述阻断项。任何 source/config 变更都会使 certificate/approval 自动失效，并要求重跑。

### SHOULD FIX / 后续扩展

- Gate M 若 GO，Level 3 必须新建 native INT4 kernel/serving 协议，不允许在当前 runner 上追加 speed 宣称。
- 对真实 cache clone/switch 的延迟与 HBM 记账应在 Level 3 独立完成，不混入 Gate M 主指标。

## 5. 最小 CPU / 小样本 dry run 结果

执行：

```text
PYTHONWARNINGS=error PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -v -p 'test_*.py'
python -m py_compile *.py
run_triage_audit_gpu.py --help
finalize_calibration.py --help
analyze_triage_results.py --help
```

结果：

- **44/44 tests PASS**；warning-as-error 下无 warning；其中新增 loader API 契约回归测试；
- 全部 Python 文件 `py_compile` PASS；
- 三个 CLI entrypoint 均可正常解析；
- CPU dry 覆盖 exact period multiset、common phase、AuditState/lockout counters、INT4 prepare-once/restore、cache non-alias/length +1、diagnostic on/off action+logit+cache invariance、calibration lock、median bootstrap/Holm、8-arm raw replay 和 no-overwrite/resume。

CPU dry 不能证明真实 Hugging Face cache、model hook 或 5090 显存可行，因此不以 CPU PASS 取代 tiny smoke。

## 6. GPU 准入结论

**GPU Run Approved: MECHANISM PROBE ONLY**

允许对冻结的 32-document calibration split 执行两模型机制验证。批准边界仍是单 GPU、teacher-forced、W4A16 dequantized BF16 quality proxy；不得解释为 native INT4、continuous batching、TPOT/P99 或能耗收益。

**Formal calibration GPU Run Approved: YES**  
**Sealed GPU Run Approved: NO**

只有 calibration 完整结束、H0 PASS 并生成 source/config/data-bound calibration lock 后，才重新审查 sealed 准入。
