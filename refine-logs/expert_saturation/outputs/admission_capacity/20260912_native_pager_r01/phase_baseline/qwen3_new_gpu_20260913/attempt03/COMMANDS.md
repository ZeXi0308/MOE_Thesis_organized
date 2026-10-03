# CPU准备与未来单次接续命令

本次只执行CPU检查，未SSH/GPU/下载。CPU复现（测试会创建并终止自己的dummy进程组）：

```sh
python3 cpu_check.py
```

原detached launcher接续命令如下。仅由父线程在既有队列前组完整释放、现场核对后执行；此文件不是启动回执。新stage/results若存在即拒绝，禁止覆盖。

```sh
mkdir /root/autodl-tmp/qwen3-new-gpu-localized-static-v026-launch-r03/detached_receipts
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python /root/autodl-tmp/qwen3-new-gpu-localized-static-v026-launch-r03/detached_launch.py --package-dir /root/autodl-tmp/qwen3-new-gpu-localized-static-v026-launch-r03 --python /root/autodl-tmp/expert-saturation/vllm-0.26/bin/python --entry-sha256 1b89fe56927ac9917895e45ac3cf3a60ed3b959166204a154d3035147dee49b8 --protocol-sha256 5ce5d1ad5b310791edfddf5e1077a46b608c5fe463e693675ca7d85252e5f71e --receipt-root /root/autodl-tmp/qwen3-new-gpu-localized-static-v026-launch-r03/detached_receipts
```

GPU运行命令由run_remote.command_for构造，CPU AST检查与原r02逐参数相同，仅独占stage/results/workspace路径变化。监控回读launch.json、detached_receipts/*/receipt.jsonl、run.log、hardware.jsonl、results/loader_receipt.jsonl以及comparison的数值gate/四格原件。最终必须区分EXITED/exit_code、数值gate、4格完成和原始终态缺失。
