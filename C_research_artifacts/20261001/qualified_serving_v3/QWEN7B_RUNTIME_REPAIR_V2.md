# 7B 资格启动环境修复 v2

Q0 v1 在 engine_init 阶段退出，未生成 measured-outputs。原生错误为 `RuntimeError: FlashInfer requires GPUs with sm75 or higher`；栈位于引擎 dummy sampler 初始化进入 FlashInfer JIT。父子进程已退出，GPU 回到2MiB且无计算进程。失败原始包为 `qwen7b_q0_init_failed_v1_20261001.tar.gz`，SHA256 `434259c1d255305c6d8a1c80da877c579cadae452f5a435711987be4c5084c01`。这是初始化无效单元，不能计为数学质量失败或完整请求实验。

成功1.5B Q0、Q1和serial日志均明确使用 `VLLM_USE_FLASHINFER_SAMPLER=0`；OLMoE启动器也显式设置。本次启动继承的环境漏传该变量。成功Q0也出现同一SM12.x/CUDA警告，因此警告本身不证明需要升级CUDA。现场 `vllm/v1/sample/ops/topk_topp_sampler.py` SHA256为 `ad9406a08a9bfcc84f182dab4522920f73605d4999191ee8f0dbb1479d946506`；其显式env=0分支在选择FlashInfer sampler之前返回False。

唯一修复是在新 `qwen7b_launch_v2.py` 中显式设置该环境变量并记录到回执，恢复已成功使用的采样后端设置。未安装软件、升级CUDA或修改共享venv。原v1脚本、冻结清单、模型权重、失败结果保持原样；v2只开放Q0/Q1，两者都继承相同env。输入、模型revision、greedy/seed、cap、32seq、1024token budget、4096context、8GiB KV、APC、原生cell及分析器均沿用原冻结文件。没有新质量输出可供选参，也不放松任何质量门。

v2执行依赖新增冻结于 `qwen7b_freeze_v2.json`。输出为新 `qwen7b-q0-v2`，只有其自身Q0完整且通过原数学门，才允许独立 `qwen7b-q1-v2`。命令为既有venv Python执行 `qwen7b_launch_v2.py --run-q0`，通过后执行 `--run-q1`。同一既有共享锁、GPU隔离、超时与仅本任务子进程回收规则不变。Q1通过前128容量单元仍UNRUN。
