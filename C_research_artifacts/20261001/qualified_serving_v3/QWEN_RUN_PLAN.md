# Qwen 固定资格运行（执行前冻结）

本组只问：已有同一组 GSM8K 16 题，在第二种 instruction 模型、自身聊天模板、自然 EOS 下，是否达到开发用健康筛查？它不是调度性能比较，也不是官方 GSM8K 评测或质量等价证明。

- 模型：`Qwen/Qwen2.5-1.5B-Instruct`，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。官方固定 revision 的 metadata/tokenizer 已下载；权重 LFS pointer 给出 3,087,467,144 bytes 和 SHA256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`。权重传输使用固定 revision 镜像 HTTPS，完整散列通过才能初始化引擎。
- 输入：原资格输入 `20261001_c_instruct_gsm8k_inputs_v1/workload.json` 的全部 test0–15，复用八示例 `chat_user_content`，Qwen 自身 template 后接 `Answer:`，重新分词，无截断或跳题。gold 只作事后评分。
- 配置：native vLLM 0.26.0、BF16、seed20260905、greedy、max1024/min0、无文本 stop、自然 EOS；APC 开启、同步、FCFS、maxseq32、batch1024、context4096、KV 8,589,934,592 bytes。每次启动两次短 warmup 后清空 APC，再让16题全部 t=0 到达，完整 drain。
- 资源：用户新提供的45495服务器；GPU `GPU-3fc910c2-bf65-5273-e6b5-6c0d8b6ce03e`。已有锁 inode `2304:25841682495`，只打开、不创建。首次占用与 GPU 初始化前均核空闲；忙则退出，不杀其他任务。它与旧机 OLMoE 结果不构成性能配对。
- 权重只暂存在本任务独有 `/dev/shm/c-qwen-qualified-20261001-v1`，模型总大小另加24GiB容器内存余量；保留已验证权重便于接续，失败 partial 不冒充模型。其他证据写入本任务新目录，不覆写旧结果。
- 边界：下载总限900秒、child限960秒；native child总限1200秒、各生成循环900秒。父进程仅可终止它自己新建的超时进程组。所有失败保留且不自动重试实验。下载失败没有模型质量结论。
- 筛查判据见 `quality_protocol.md`：16/16完成、0空白、自然EOS至少12/16、数值正确且EOS至少8/16、重复后缀至多1/16；完整性单独核查。门未通过就停止以此域主张健康调度收益，不通过换题/seed/cap救结果。

运行只复用已有范围下载器，新增独立 cell 是为了去掉旧脚本绑定的模型/GPU依赖，同时保持真实 native 采样、APC重置和失败回执；没有实现控制器或性能矩阵。这里的代码和CPU检查不计GPU结果。唯一后继是先分析全部16题，再决定是否开展已声明的摘要资格任务。

远端命令在新部署目录下：

```sh
OMP_NUM_THREADS=25 MKL_NUM_THREADS=25 PYTHONDONTWRITEBYTECODE=1 \
  /root/autodl-tmp/c-vllm-v026-venv/bin/python -u health_launch.py
```

执行前所有源、输入、协议、模型清单散列记录于 `freeze.json`；结果写 `qwen-health-v1/`。本计划记录的是协议，不是任务已完成。
