# cohort3 限定完整性复核

Reviewer: `/root/cohort3_integrity`, GPT-5.6-Sol；same-family / provisional。

**WARN，无 P0/P1。** 四格结果、输入身份、计时分母、真实执行及资源对照支持报告的限定描述性结论。范围仍为同模型、固定长度/压力、预先使用过的不重叠文档集合，每臂两次交错执行。复核不证明性能显著性、质量或泛化。

复核实际确认：所有性能格32请求/32768输出token；输入与原服务集合在文档ID、文档SHA、prompt-token SHA均无交集；到达一致。两对各26/32文本相同，且同样六请求不同；同臂跨重复各32/32文本一致。只记录此事实，不把它升级为正确性或数值因果定位。

两个限定事项：

- 审阅的限定文件列表不含归档本体/23项验证回执，因此归档SHA与23文件一致性的声明不是本次审阅独立验证的内容；执行方既有核验与独立复核分开。
- `policy_hooks_modified=false` 来自测量helper，而调用方已装入staged scheduler adapter。它只能说明helper未额外修改policy hook，不能说明stock scheduler未经修改。说明见 [ADDENDUM.md](ADDENDUM.md)。

逐项结果见 [EXPERIMENT_AUDIT.json](EXPERIMENT_AUDIT.json)。原报告、raw、执行包均保留；不因这些限制重跑、不扩大审计。主线统一对照由原负责人完成，本线复核至此结束。
