# TriageAudit v2 数据冻结报告

状态：**DATA PREPARATION PASS / NO GPU RESULT / SEALED NOT OPENED FOR OUTCOME ANALYSIS**  
日期：2026-07-23

## 冻结输入

- dataset: `wikitext/wikitext-103-raw-v1`
- revision: `b08601e04326c79dfdd32d625aee71d232d685c3`
- split: `train`
- dataset fingerprint: `f697edc678db8d0f`
- article parser: 与 `experiments/shared/prompts.py` 相同的顶层 title/article 聚合语义
- model tokenizers: OLMoE/LLM-jp 的 frozen revision
- minimum length: 两 tokenizer 都至少 96 tokens（prompt 64 + teacher-forced decode input 32）

## 历史排除

历史 WikiText-103 article 实验共用 `Random(20260720).shuffle`。已知使用/候选区间的最大终点为 728；本轮保守排除 shuffle 后前 1,000 篇 article 的 canonical text SHA-256。

- parsed articles: 28,911
- historical hashes excluded: 1,000
- dual-tokenizer-valid unused candidates: 27,904

## 冻结结果

- calibration: 32 unique documents
- sealed: 64 unique documents
- calibration/sealed overlap: 0
- selected/historical overlap: 0
- selected minimum OLMoE tokens after truncation check: 96
- selected minimum LLM-jp tokens after truncation check: 96
- sealed manifest permission: `0600`

## Artifact hashes

- `calibration_manifest.jsonl`: `4ddcc822cfe1c7cbf2ccdc6dd0d90d62ce120bd5d95afb2f697d9a90e22338bf`
- `sealed_manifest.jsonl`: `54b48b59bec487bbd151ecad10a9e90d12b04f0ef474067aff8ce5ef93db7c88`
- `historical_hash_registry.json`: `9b7405ae8898e86419b09f47a89f35c1bef252b4b9e3d2b1a07e70dd3ed96373`
- `provenance.json`: `51c04dc8be15f2247864a8c292438805d10eba997fa9850253182966aae2838b`

工作区路径：`docs/ideas/B_verify_precision/triage_audit/data_v2_2026-07-23/`

该报告只证明数据冻结与隔离闭合，不包含任何模型 outcome 或科学结论。
