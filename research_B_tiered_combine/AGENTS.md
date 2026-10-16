# 两 lane combine
继承 ../AGENTS.md。当前任务全文：configs/task_brief.txt。
当前事实、资源状态及下一动作：RESULTS.md。
只修改本目录和远端 /root/autodl-tmp/research_B_tiered_combine；不修改其他课题。
远端实验使用既有 /root/autodl-tmp/moe-research-gpu.lock，并现场核实 GPU/进程。
每个完成的实验回传 runs/ 到本地，执行 code/pull_results.sh；失败结果也保留。
不得在文件中保存 SSH 密码。源码和配置在本地维护，再部署；模型权重不当作实验数据重复备份。
