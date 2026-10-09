# 直接编码分支：冻结版本与最小复现

本说明保存已有实现的复现条件，不是本轮执行计划。本轮没有重跑CPU服务、编译或GPU；当前定位见[冻结入口](README.md)。

## 保存的版本

版本名为`F-direct-20261009-frozen`。该工作区没有Git提交身份，用源码内容确定版本，不捏造commit：

| 身份 | SHA-256 |
|---|---|
| 全部源码及必要父依赖的内容版本 | `144b8a4bb90c966b4c048e521a2066ad9edd4d8d010732be197c404806a7733f` |
| 冻结前完整本地归档 | `7e5d74044aa97fdce99fdb7cde99a0465c7c82344f5a9ba1c4313f9d2e56b055` |
| 冻结前source_manifest.json | `8fca5fd91b84085c2a789a14252549a57f2a21fd5e6b4caf21a6a73c804ac4d2` |

[归档](direct_encoding_snapshot.tar.gz)含42个本地文件：直接编码目录全部源码、文档、成功/失败记录及当时缓存，另含父目录`native_adapter.py`、`bootstrap.py`、`environment.json`。[preservation.json](preservation.json)列逐文件SHA、大小及源码版本算法。归档创建前核对旧清单18个源码/文档和5个证据项全部相符，创建后核对归档每个成员。当前入口文档会标记冻结，原清单保持不动；需要逐字还原冻结前文档时读取归档。

核心实现分工：C++ CPU数组直编码及token静态片段；异步前端桥和显式context回退；每请求CUDA槽/event的GPU组件；首次候选D2H前的采样接线。早期同步GPU组件及`bench_real.py`也保存，但不代表修正后的异步服务路径。代码不再修改。

## 运行条件

原记录环境为Linux x86_64、Python 3.12.3、vLLM 0.26.0（`gffd46bfab`）、Torch 2.11.0、transformers 5.17.0、tokenizers 0.23.2、pydantic 2.13.5；依据保存的[父环境快照](../environment.json)，不是本轮重新采集环境。CPU encoder由C++17编译器动态构建，使用NumPy和ctypes；NumPy及g++精确版本未保存，不补造版本号。GPU历史编译使用nvcc 12.8、sm_120，完整命令见[编译记录](../direct_encoding_20261009/gpu_async_build.json)。

此目录不是独立Python包。`dependency_check.py`和`native_semantics.py`从父目录导入原生适配器与bootstrap；后者仅在进程内处理原环境升级留下的孤儿模块，不修改安装包。只复制直接编码子目录不足以复现。

原生CPU检查只加载tokenizer，不加载模型权重；默认路径是`/root/autodl-tmp/moe-research-20261002/model`，两个脚本均支持`--model`指定同一tokenizer。tokenizer资产、模型权重、已安装Python/CUDA工具链及远端生成的`.so`未打包；本地不存在这些`.build`产物。已保存token表的元信息和内容摘要可辅助核对，但不能替代未归档的tokenizer资产/文件版本。原生检查的词表为50281项；尚未执行的模型接线按模型配置50304准备，不能混用这两个边界。

## 最小CPU复现命令（仅供复现，不在本轮执行）

在具备上述环境的原F目录或单独解压的副本中，保留父子目录结构。若使用归档，可在F目录内新建空目录解压；不要覆盖原始证据。

```sh
set -e
cd /root/F_output_isolation
mkdir direct_encoding_20261009/reproduce_cpu
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cu13/lib
export TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES='' /root/miniconda3/bin/python \
  direct_encoding_20261009/dependency_check.py \
  --output direct_encoding_20261009/reproduce_cpu/dependency.json
CUDA_VISIBLE_DEVICES='' /root/miniconda3/bin/python \
  direct_encoding_20261009/native_semantics.py \
  --output direct_encoding_20261009/reproduce_cpu/semantics.json
```

`mkdir`应在目录已存在时停止并换用新位置，避免覆盖旧结果。两个输出均应为`ok:true`、`gpu_used:false`、`performance_claim:false`。完整语义检查应覆盖已保存8个用例的文本、按序候选、数值、bytes、终止、usage、DONE及逐步生成状态；依赖检查应允许plain和先就绪请求独立完成。生成块数可因原生collector合并不同，不以此计算性能收益。受控gate或Future的时间值、初始化时间不作为延迟或吞吐基准。

这些是CPU正确性检查，未冻结性能核预算，也不据其时间字段比较CPU/GPU。旧性能回放的frontend/client/producer三核配置只属于旧路线，不能称新分支已在该预算下完成性能对照。本轮不重跑以上检查，只核对已保存结果。

## 待验证问题与不可执行的旧计划

| 未知事项 | 已保存到哪里 / 目前缺什么 |
|---|---|
| 真实top-20及自然批次 | `capture_real.py`保存预定64请求、16个heavy、两波到达和96/192输出上限；两次历史日志均锁忙，未生成capture目录。不是已经采到后丢失的数据 |
| GPU正确性及设备独立完成 | `gpu_async_encoder.*`编译记录存在，未构造CUDA encoder、运行kernel或完成真实依赖检查 |
| 首次D2H前的原生接线 | `sampling_branch.py`保存单进程/单GPU、sync scheduling、无spec decode、普通text/top-20合同；只有代码和静态接口检查，没有已验证模型/HTTP共跑驱动 |
| 强CPU与GPU的完整成本 | 尚无compact D2H+CPU编码与GPU编码+expanded D2H的对照；D2D快照、padding、raw回退、每请求launch/DMA、轮询和host收尾均须计入 |
| 同资源的服务取舍 | 新分支没有客户端尾延迟、重流损失、饥饿、CPU/内存、吞吐及下一步推理干扰证据；CPU模式目前共享GPU固定池，不是CPU最小内存配置 |
| 可形成独立贡献的应用问题 | 尚未建立重要交付需求、简单方法的剩余缺口与可区别于已有工作的贡献 |

冻结前的协议、GPU命令、参数与输入计划保留在归档中，用于解释代码；它们不再是排队任务。本轮不启动GPU、不候锁、不增加执行位置选择器或其他方法。资源可用也不自动重启；重新投入必须先满足上表最后一项及用户重新确定的范围。
