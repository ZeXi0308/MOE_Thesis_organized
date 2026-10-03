# 2026-09-08：逐 cell 执行与及时回传

本补充只改变进程编排和日志留存。原 `DECISIONS.md`、`REPORT.md`、`ADDENDUM.md`
和 `execution.tar.gz` 均保留，科学配置未变。新包为 `execution-20260908.tar.gz`；
旧包 SHA256 为 `62f968b639dd1ec8951c96df20d36681039e386e3d116615226bce4048bb2175`。

`run_probe.py`、`memory_telemetry.py`、`native_capture.py`、`metrics.py` 与两组输入的
配置/工作负载均逐字节等于旧包。每进程的共同暖机仍为 short32/cap16、short32/cap32、
long2/cap2，输出均为16 tokens；测量仍为32请求、128/3072输入、1024输出、50ms到达，
固定cap16/32，engine32、context4096、token预算1024、显存预算0.90。

`run_probe.py` 原本已把每次暖机完整结果写入 `warmup-0.json`、`warmup-1.json`、
`warmup-2.json`，然后才判断是否失败。成功和失败/容量保护返回均保留完整请求、
scheduler、输出事件和已采集内存数据，因此此次无需修改暖机或运行实现。

## 唯一执行入口与顺序

在新的远端目录解包。使用既有 GPU Python，每次只调用一个 cell：

```bash
/root/autodl-tmp/expert-saturation/vllm-0.26/bin/python -u run_one_cell.py repeat0-short-cap16
```

其余 cell 依次为：

```text
repeat0-long-cap16
repeat0-short-cap32
repeat0-long-cap32
repeat1-long-cap32
repeat1-short-cap32
repeat1-long-cap16
repeat1-short-cap16
```

`run_campaign.py --list` 只列出完整冻结顺序；`run_campaign.py --cell LABEL`
与 `run_one_cell.py LABEL` 等价。没有自动连续执行八个 cell 的入口。
每次调用启动一个全新 `run_probe.py` 进程，结束后立即返回，不启动下一 cell。

输出保存在 `results/`：

- `LABEL/`：测量 raw、完整 warmup raw、配置、环境、内存快照、指标与最终状态。
- `LABEL.stdout.log`、`LABEL.stderr.log`：分开的标准输出和错误输出。
- `LABEL-execution.json`：实际命令、目录、启动器/子进程PID、启动器源码哈希、
  起止时间、返回码或启动/等待异常。等待异常不代表子进程已停止，需按记录PID核实。

已有 cell 目录、退出记录或日志均拒绝覆盖。每个 cell 完成或触发容量保护后，
先回传其全部目录、两份日志和退出记录，校验归档并读取核心 JSON；确认本地完整后
才按冻结顺序启动下一个。回传前不删除远端数据。非预期失败先检查，不自动重跑。
`CAPACITY_BOUNDARY_STOP` 仍为保留全部部分轨迹的边界结果，不作为完整吞吐比较。

## 本地针对性核验

源码语法检查通过；复用 `load_inputs` 核验32对输入、token哈希、相同文档/到达及
短前缀与长前缀一致。八项顺序与原包一致；除启动器之外九个原包文件字节未变。

在临时目录用无GPU的假子进程分别返回0和7：两种返回码均正确保留，stdout/stderr
分开可读，一次只运行指定 cell；重复标签被拒绝且旧数据字节不变。夹具已移除。
这些检查只验证启动与留存，不提供GPU性能或容量证据。此次准备没有连接远端或启动GPU。
