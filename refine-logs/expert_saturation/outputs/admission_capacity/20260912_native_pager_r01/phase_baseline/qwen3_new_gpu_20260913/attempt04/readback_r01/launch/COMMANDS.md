# r04 可核对命令（尚未执行）

CPU本地复现：`python3 cpu_check.py`。测试仅创建/回收自身dummy进程组，无GPU或网络。

父线程在前序整组释放、输入成员哈希及现场状态核对后解包至全新stage，再执行原detached接口：

```sh
mkdir /root/autodl-tmp/qwen3-new-gpu-localized-static-v026-launch-r04/detached_receipts
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python /root/autodl-tmp/qwen3-new-gpu-localized-static-v026-launch-r04/detached_launch.py --package-dir /root/autodl-tmp/qwen3-new-gpu-localized-static-v026-launch-r04 --python /root/autodl-tmp/expert-saturation/vllm-0.26/bin/python --entry-sha256 2bfdd99699896e92a68f5330519ead1a657cef0f1592ad48f317b3442ee9354c --protocol-sha256 7be5717a525392bd7a2f81fcccf2ad708d44e10ea35bb89f8427b2bee0a3d826 --receipt-root /root/autodl-tmp/qwen3-new-gpu-localized-static-v026-launch-r04/detached_receipts
```

父进程的唯一worker命令由run_remote.command_for构造，原参数完全相同，`--output`指向r04/comparison，`--loader-workspace`为`/root/qwen-streaming-shards-20260914-r04`，receipt仍在r04结果目录。父进程负责workspace独占新建，不预先创建它。

回读stage的launch.json及.tmp、run.log、hardware.jsonl、detached_receipts；结果的loader_receipt.jsonl、comparison数值gate和四格原件；失败时另保留根盘workspace中的本次partial。stage/data归档不能宣称包含另盘模型partial，必须单列其路径/大小与保留状态。所有启动/加载/资格/退出成本保留。
