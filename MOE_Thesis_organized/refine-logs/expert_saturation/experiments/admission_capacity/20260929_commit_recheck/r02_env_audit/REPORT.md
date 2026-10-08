# r02 G64 单格诊断：新机器环境入口审查

范围：只读审查已接受的 `20260915_natural_ltr_style_component_r02`、当前 H1 候选的 `pkg/run.sh` 和本轮 `RUN_PLAN.md`；无 SSH、GPU、controller 或科学结果。原 r02 清单 30/30 文件在隔离检出中逐项 SHA-256 相符；`manifest.json` SHA-256 为 `ce03adb36e244db2a5bc6c7924295a9f155648bfc33ce789f4be71c151231dd1`。原 `package.tar.gz` 不在本地，无法复核 `package.sha256` 记录的归档字节 `6888c318…4c9e73`。

## 必须保持的实验合同

- 这仍是**一格资格诊断**：G 的 64 个原始 token-ID 输入、0.2 s 固定到达、自然 EOS、cap 1024、最长 180 s *measurement capture*、LTR-style T30/Q10；不是 H128 性能比较，也不是完整 LTR。
- `run_ltr_style.py`、`ltr_style_native.py`、`ltr_style_selected.py`、`recovery_service_components.py`、共同 adapter/计量代码、`inputs/`、`warmups/`、`runtime_source_hashes.json` 均保持原字节。OLMoE revision/tokenizer revision `6d84c485…`, vLLM 0.26 pinned source，4096 usable + 1 null GPU KV 块、16 GiB/8192 native host 块、cap32/1024 batch tokens、APC off、full-history reservation、selected native saving均不因机器变化而调整。原 runner 会核实实际 GPU/host KV 分配及模型、输入身份；这些是现场运行时门，不是静态已通过的资格。
- 原 `pkg/run.sh` 固定旧 UUID `GPU-4015b79d…d0e5`、Python/cache 路径和 `/root/autodl-tmp/moe-research-gpu.lock`。若新机器这些值完全相同，原 30 文件与 manifest 可逐字复制后现场再验。若有任何差异，只在**新目录和新包身份**中改环境入口、更新 manifest/归档 SHA 并复核；不能原地改已接受 r02、沿用其包 SHA，或借机改策略/工作负载。`controller.py` 的 `launch-once` 和每格 `results/` 均需全新 staging，旧失败与旧接受包保留。

## 新环境入口的 fail-closed 门

1. **授权与占用。** 启动前取得用户明确的机器入口、GPU UUID/数量、host 内存硬上限及本格和总组时长/费用范围。现场只读核 UUID、所授权 GPU/隔离卡的 compute 进程和显存占用、同机 controller/process tree、`GPU_COORDINATION.md` 的整组交接；任何占用或查询不明则 ABORT。`CUDA_VISIBLE_DEVICES` 必须精确指向获授权 UUID，runner 的 `torch.cuda.device_count()==1` 还不足以证明它映射到正确物理卡。旧 runner `gpu_state()` 查询所有卡的 compute 进程，可作为保守二次门，但它没有单独验证授权 UUID 或非 compute 显存占用。
2. **共同锁。** 继续用同一宿主机上其它研究组使用的 `/root/autodl-tmp/moe-research-gpu.lock`（如该机实际共同路径另有定义，先统一所有包），`flock -n`，至少从预飞行经初始化、三次 warmup、测量、drain 到关闭保持同一 FD；锁忙立即 ABORT，不后台候卡、不自动重试。旧 `pkg/run.sh` 在 runner 退出即释放锁，`controller.py` 随后才写最终 group status；若要求归档也在整组排他窗口，后继入口须设置**单一**外层锁所有者，不能在内外层分别打开同一锁自冲突。H1 入口允许调用者传任意 `H1_LOCK_PATH`，因此 root 必须把它固定到**相同的实际锁文件**；两个不同路径的锁不能证明 controller 串行。协调记录和 advisory lock 均不能替代现场进程检查。旧 `controller.py` 在 shell 取锁前先写 `launch-once`；忙锁导致该 staging 已消耗，不能删 marker 原地重启。
3. **host 与模型。** 要有可读的进程树 cgroup v2 `memory.max`，值为有限正整数且不超过获批 host bytes，并确认 runner 及其全部子进程受同一限额；仅 `offload_gib=16`、shell 变量、`ulimit` 或物理机器总内存不能充当硬限。原 runner 会再核实际 16 GiB host KV/8192 块；总进程内存还含模型和其他 host 开销。只用现有离线 HF cache，固定 model/tokenizer revision，`HF_HUB_OFFLINE=1`，缺缓存就 ABORT，不下载或更换模型。
4. **源码与期限。** 用固定 Python/vLLM 环境，先验新包 manifest 30/30（修改入口后应为 30/30 新 SHA）及原 `preflight.py` 的 8 个 vLLM 源码哈希；runner 另核 scheduler/KV 管理器两个固定哈希并记录版本、更多源码哈希。源码不一致则 ABORT，不用当前工作树 adapter 替换。180 s 仅是测量 capture 上限；engine 初始化、三次各可达 120 s 的 warmup、drain、shutdown另占时间。环境入口须用**覆盖整格**的获批 wall 上限（含超时终止自己的进程树），记录实际耗时与预算扣减；超时或强制终止只记 `INCOMPLETE` 并保留部分 raw/log/status，不自动重启或进入性能组。

## 审查结论

原包的静态 payload 完整；它在新机器上仍为 `GPU_UNRUN`。先建立仅环境入口变化的新身份，再由 root 按上述门现场资格运行；只有原生 selected store/load/flush、ready scheduling、首个新输出、量子跨首输出、EOS 和资源账均成立，才允许把后续同输入强基线性能包列入比较。新入口不应把 H1 `run.sh` 的可配置锁路径直接当作已统一的共同锁，也不应把旧 `controller.py` 的 `COMPLETE` 外层状态代替每格 native 证据。
