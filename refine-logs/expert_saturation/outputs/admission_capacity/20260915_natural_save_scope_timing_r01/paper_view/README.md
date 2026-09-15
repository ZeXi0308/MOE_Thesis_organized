# 保存范围的论文图

[PDF](saving_tradeoff.pdf) / [SVG](saving_tradeoff.svg) / [PNG](saving_tradeoff.png)。从已完成的[唯一主分析](../execution_weste_26862/analysis.json)生成，未增加实验或重新执行主分析；绘图入口为 `python3 paper_view/plot.py`，也可从仓库根使用完整脚本路径。

前两图展示每对全部64请求的最大引擎返回间隔经验分布。第三图并列实际输出吞吐、全局最大间隔和平均完成（标注），箭头从selected指向full，圆/三角表示配对0/1。两对均保留，未选SLO或误差界。输出量/内容不同，图不表示等工作量加速；同一输入的两对也不是两个独立负载。

固定原生vLLM/OLMoE、4096 usable GPU块＋null、16GiB host分配和current调度，仅改变保存范围。图支持的判断是full没有成为本域长停顿更强基线，不能据此宣称selected支配所有指标或所有后端。全部原件、主分析与主报告不变。
