# 文章来源边界补充：实际输入前缀未改变

2026-09-11，在只读检查下一批自然未截断输入时，发现旧文章重建器将正文中的表格注释误识别为顶级标题。本说明保留原始输入、metadata、代码和raw，不改写其历史内容。

来源为固定Wikitext-103-raw-v1 train shard0，revision `b08601e04326c79dfdd32d625aee71d232d685c3`，Arrow SHA256 `0c22278eac8186f5dd6060fcf84e4bcb4aa2266c51c77a279169764814f9a99e`。本地离线读取确认Arrow与三个pinned OLMoE tokenizer文件均匹配既有hash。

旧实现位于 [prepare_inputs.py第59行](../../expert_saturation/outputs/admission_capacity/20260906_native_memory_pressure_r01/inputs_preparation/prepare_inputs.py#L59)，使用 `re.fullmatch(r'= [^=].*? =', text.strip())`。具体原始行如下，行号为dataset的零基索引：

```text
377  ' Note : Pos \n'
378  ' = Position ; GP = \n'
379  ' Games played in ; G \n'
380  ' = Goals ; A = \n'
```

378、380、382、388、390、392、394共七行是同一冰球赛季文章中的表格注释，前后不是空行，却均满足旧标题正则。如果直接复用该规则挑选短的完整文章，会把这些片段作为独立文档；不能靠增加最小token长度隐藏这个错误。

对本轮既有16条source_requests逐条检查标题、起止行，并重新分词对应完整文章，结果如下：

| 检查项 | 结果 |
|---|---|
| 16个来源开始行是否为真实文章标题 | 全部是，未选中上述七个伪标题 |
| 旧来源字符串hash与声明是否一致 | 16/16一致 |
| 结束边界发生变化 | 仅第4条，`memory-train-article-0000316` |
| 第4条文章 | `2011 – 12 Columbus Blue Jackets season` |
| 第4条旧范围 / 完整范围 | `[316,378)` / `[316,406)` |
| 第4条旧声明全文token数 / 重建全文token数 | 3821 / 4515 |
| 重新分词完整文章后的实际128/2048输入前缀 | 16/16逐token完全一致 |

因此，准确表述是“来自16个不同文章的固定长度前缀”。第4条metadata中的完整来源范围及原始全文token数不准确；其余15条边界未改变。本轮实际执行token IDs、请求身份、性能测量与策略对照没有因此变化。这次核对仅覆盖既有16条和下一批候选所在前2500行，不是全数据集语义边界审计。

下一批输入应采用完整原始行标题匹配 `r' = [^=\n]+ = \n'`，同时要求标题前后原始行为空；从标题到下一个有效标题逐字串接原row，不添加分隔符，不接受分片末尾未闭合文章。这一结构修正排除了上述反例，再按固定源顺序排除本轮16个document IDs，选择16篇全文不超过3968 tokens的文章。只读候选分词已可完成；正式workload及native1024/FCFS GPU诊断尚未生成或运行。
