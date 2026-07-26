# TriageAudit v2 GPU tiny smoke 状态

日期：2026-07-23  
状态：**ENGINEERING_PASS / SCIENTIFIC_RESULT=NULL / FORMAL MECHANISM PROBE APPROVED**

## 审查绑定

- source aggregate SHA-256: `05023e8a7bcc8632461663440a1952cf72da330e1637fe2f19b1cfedc8dbccc1`
- config SHA-256: `286ec92484f394389a94a5419a6ebba6ba674ab114f85058a83b7a8660f0a563`
- OLMoE model tree SHA-256: `909942c554a983a7929a8e2cf7d3326b007ed4c694f3a28c825c1f846e86267e`
- OLMoE revision: `6d84c48581ece794365f2b8e9cfb043c68ade9c5`
- GPU: `NVIDIA GeForce RTX 5090`, 33,668,988,928 bytes

## 执行历史

第一次尝试在模型构造前因 `transformers==4.53.3` 不接受 `dtype=` 而终止。该目录被永久记录为 `INVALID_RUN`，`scientific_result=null`；没有 forward，也没有采信任何输出。修复为 `torch_dtype=` 后新增 loader 契约测试，远端 warning-as-error CPU review 为 44/44 PASS，源码哈希随之重新冻结。

第二次尝试仅执行已批准的 OLMoE 1-document、prompt=8、decode=2 engineering smoke，并通过独立 artifact audit：

- native-original vs patched-full max-abs logit error: `0.0`；
- native-original vs patched-full token KL: `0.0`；
- same-state BF16/INT4 discrepancy: `0.0287669431`, `0.0282509960`，证明低精度 expert proxy 确实生效；
- diagnostic-off action/logit/final-KV invariance: PASS；
- peak allocated: `24.9289 GiB`；
- peak reserved: `24.9648 GiB`；
- allowed peak: `29.8567 GiB`；memory gate PASS。

成功 smoke artifact SHA-256：

- `status.json`: `f01cbaad974f1938709255c434f5ee5bd59095ffd36424db2f28750f01b2834e`
- `memory_certificate.json`: `499a2c1fdb5542a0c892a5832a03bd46e3ca16c756745aa070d69dfc666c5784`
- `raw_results.jsonl`: `b2499aef9afc991a95600f1e95baebce3dc9cddd206002b7c221c3d199a4c03a`
- `source_manifest.json`: `667a77c81ebaac089614ea0e3901a527db8afdca13c6c22ed96a202d23eddb62`

## 准入结论

**GPU Run Approved: MECHANISM PROBE ONLY**

允许进入 calibration；它仍是单 GPU、teacher-forced、W4A16 dequantized BF16 quality proxy 的机制验证，不是 native INT4、continuous batching、TPOT/P99 或能耗证据。

Sealed 仍未批准。只有 calibration H0、数据/arm 完整性和 calibration lock 全部通过后，才允许读取 sealed manifest。
