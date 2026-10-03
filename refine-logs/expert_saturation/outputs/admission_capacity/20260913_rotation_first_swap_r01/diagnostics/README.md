# First-swap CPU path/tail diagnostics

入口 `diagnose_paths.py` 仅做本地只读分析；不启动 GPU、上传或修改冻结包。当前 F 六项未形成合格主分析，`prepared_gate/diagnostics.json` 为 `UNRUN_OR_UNQUALIFIED`，没有性能配对。

weste:23478 的六项主分析使用 `F/analysis02_weste_23478`，执行数据使用 `F/execution02_weste_23478`；旧 westc 的 `F/execution` 保留。从仓库根目录，在该六项主分析完成后执行：

```sh
python3 refine-logs/expert_saturation/outputs/admission_capacity/20260913_rotation_first_swap_r01/diagnostics/diagnose_paths.py \
  --bundle refine-logs/expert_saturation/outputs/admission_capacity/20260913_rotation_first_swap_r01 \
  --run-dir refine-logs/expert_saturation/outputs/admission_capacity/20260913_rotation_first_swap_r01/execution02_weste_23478 \
  --primary refine-logs/expert_saturation/outputs/admission_capacity/20260913_rotation_first_swap_r01/analysis02_weste_23478/analysis.json \
  --output-dir refine-logs/expert_saturation/outputs/admission_capacity/20260913_rotation_first_swap_r01/diagnostics/measured02_weste_23478
```

`--run-dir` 可省略以兼容旧入口，默认仍是 `bundle/execution`；指定它会同时切换执行状态、raw哈希校验和数据读取路径。`--output-dir` 必须不存在；入口不会创建主分析的 `F/analysis`。主分析缺失、部分格或任何未合格状态均只输出门禁状态，不生成性能比较。合格后再次核对manifest、执行顺序/回传状态和主分析保存的所有raw/decision哈希，避免消费旧资格下已变化的输入。

roles、cohort、block及配对来自manifest与主分析，没有八格/两cohort假设。C从每条decision的 `effective_victim_order` 选择排序；成功交换计数及切换规则复用主分析验收。每次选择的资格候选、真实victim/恢复对象、动作前记录状态和sourceID都保留。原生抢占到恢复/首个新输出的路径直接保留主分析已核对的 `recovery_accounting`。

输出覆盖每格完整请求完成次序、每请求完成时间/step、纯decode每种宽度的调用数及互斥engine host时长、width1/2/4入口已产生输出数量和实际请求集合。配对显示首次选择差异，固定以同block A的最早四请求、最后两请求作身份锚点；同时显示另两臂是否仍由同两请求收尾。跨臂关联使用真实sourceID，候选同分按原生请求ID排序，不要求后续状态一致；输出计数不含重算。批宽桶变化与早/晚完成转移是实际路径，不是可相加的独立因果收益或硬上界。

## 已验证范围

`run_dir_checks/checks.json` 保留此次路径参数修改的 5 项 CPU 检查：旧 CLI 默认、显式 `--run-dir`、默认目录未完成、显式目录缺失、显式目录为 STAGED，全部通过。检查只使用内存中的资格元数据和临时执行状态，不读取 raw、不重算旧 V、不生成 measured 结果；既有 manifest/raw 哈希门禁代码保留，本次未重新执行合格数据路径。

`v_compatibility/` 是对旧V已完成八格A/B的一次兼容核对，实验ID仍为 `20260913_rotation_victim_order_r01`，不能算新F结果。兼容运行命令与上面相同，将 `--bundle`、`--primary` 指向V，省略 `--run-dir` 或将其指向 `V/execution`，将输出放到新的diagnostics子目录。

`compatibility_checks.json` / `check_compatibility.py` 保留以下CPU检查：

- V全部forced对象/step、恢复对齐/最长暂停/重算与旧 `analysis/paths.json` 一致。
- V全宽度调用、width2 host时间/请求集合/入口output计数及全部逐请求完成值与旧 `analysis/tail_paths.json` 精确一致；4个配对width2都是87→8。
- F缺失合格主分析以及部分campaign不产生性能比较。
- 一个微型CPU状态夹具验证C按记录的most→least模式选候选；不代表C的原生GPU执行。

没有验证新C的GPU结果，也没有重新运行或修改V原件、F冻结GPU source或raw。`check_compatibility.py` 对已有兼容结果作核对，采用独占写入，保留已有检查记录。
