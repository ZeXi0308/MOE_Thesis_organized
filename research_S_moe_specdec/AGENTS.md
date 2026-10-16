# MoE speculative decoding

继承 ../AGENTS.md；本课题明确任务见 configs/task_brief.txt，最新进度见 RESULTS.md。
只修改本目录与服务器 /root/autodl-tmp/research_S_moe_specdec，不改其他课题或共享 vLLM 安装。
GPU 使用 /root/autodl-tmp/moe-research-gpu.lock，整组持锁并核对现场占用；预算约 20 GPU 小时。
源码在本地维护，再部署；失败与原始输出保留。SSH 凭据不得写入文件。
先完成可开关机制，再进行评价。当前效果未达预期时写明同一路线内的建设性下一步。
模型预测和测量分开；logical expert working-set bytes 不标成实测 HBM traffic。
