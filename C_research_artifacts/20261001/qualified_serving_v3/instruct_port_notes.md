# Instruct资格迁移核对（只读）

已读PLAN、V2 CELL/LAUNCHER/FREEZE/DOWNLOAD、运输修订及checkpoint；worktree HEAD=a3c1bda5，未跟踪download测试。三份V2源码与freeze哈希一致。旧checkpoint末尾仍是10:53的cap消融；较新的Instruct修订记载V1在下载期取消、已reap/清stage，**没有生成结果**。V2是否已启动须根任务查旧机回执，不能从计划推断完成或重启。

**最小迁移包**：freeze的全部remote_files、FREEZE自身、两个PF辅助模块，以及`20260930_c_sustained_baselines_dev_v1/pkg/{memory_telemetry,run_recovery_cadence}.py`。后两者不在V2 freeze中，V3应补哈希；不需复制旧raw。固定Instruct revision=7f1c97f440f06ce36705e4f2b843edb5925f4498，GSM8K前16题/eight-shot/greedy/cap1024/EOS/APC和warmup保持不变。

**硬编码变更仅在独立新包**：PF launcher的BASE/PARENT/PYTHON/HF_HOME/LOCK/LOCK_INODE/GPU_UUID；PF cell的GPU_UUID；Instruct launcher的ROOT/STAGE/CELL/FREEZE版本路径。新机现场登记既有共享锁，不能沿用旧inode或替换锁。修改后冻结新SHA，原freeze不覆盖。EXPECTED_RUNTIME包含Python完整build字符串、torch2.11.0+cu130、CUDA13.0、vLLM0.26.0、transformers5.15.1、25线程和8份vLLM源码hash；普通pip同版本不保证相同，差异需独立资格，不能删除检查。

**资源**：模型13,841,132,070B；tmpfs可用≥14,914,873,894B，cgroup剩余≥41,758,419,494B且memory.max为有限值；下载中留20GiB。磁盘>256MiB。KV=8,592,031,744B（4097块含null）；另需BF16权重与运行时显存，现场实测，不能仅按KV选卡。锁内单GPU、无其他compute进程、显存占用≤64MiB；原32GiB卡通过不代表新卡自动通过。

**入口**：V2 launcher无参数；CELL参数如下，必须由持锁launcher启动，不能绕过：

```sh
python C_INSTRUCT_GSM8K_CELL_V2.py --inputs INPUTS --output NEW_ROOT/native --parent PARENT --model-dir OWNED_EMPTY_STAGE --model-manifest C_INSTRUCT_MODEL_MANIFEST_V1.json
python C_INSTRUCT_GSM8K_ANALYZE_V1.py --inputs-dir INPUTS --run-dir NEW_ROOT/native --metadata-dir 20261001_c_instruct_model_metadata_v1 --output NEW_ANALYSIS.json
```

新机最终SSH使用`ssh -p 45495 root@connect.westd.seetacloud.com`；新launcher尚未实现，不能给假运行命令。

下载8路32MiB/段，单段最多3次、全局1200s，HTTPS206/range/长度/整文件SHA任一错即停；总child1560s。忙锁返回75不排队。16题完成、tokenizer/EOS/APC-reset/drain均合格才评质量；正确答案伴自然EOS与长度变化才进入服务小对照。错误、重复或撞cap主导则保留负结果，不删题调prompt救结果。仅清自己stage且先确认child回收及GPU空闲；运行失败≠方法NO-GO。本核对未联网、未GPU、未改共享树。
