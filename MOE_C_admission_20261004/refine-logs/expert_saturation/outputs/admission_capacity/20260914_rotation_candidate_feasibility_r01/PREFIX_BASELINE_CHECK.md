# Prefix caching 基线核对

2026-09-14 CPU 只读检查：在 `refine-logs/expert_saturation/outputs/admission_capacity/` 下枚举全部 `engine_args.json`，237 个文件全部记录 `enable_prefix_caching: false`；其中 218 个同目录 `status.json` 标记 `COMPLETE`。其余为 BLOCKED 2、UNRUN 1、CAPACITY_BOUNDARY_STOP 4、FAILED 7、INCOMPLETE 3、INITIALIZING 2。未找到记录 true 的引擎配置。

这是本地已回读配置的范围检查，文件数不是独立运行数；无法排除尚未回读、其它目录或未保存配置的运行。当前 cohort2 八格及本诊断三格均关闭缓存。因此默认缓存路径仍是强基线/泛化缺口，不能写成已经比较过。

当前 `experiments/admission_capacity/rotation_native.py:58` 在安装时拒绝 `manager.enable_caching`，`:70` 要求 `KVCacheCoordinatorNoPrefixCache` 和单一 FullAttentionManager。不是仅修改 flag 就能得到合法同机制对照：共享缓存情况下，block-table 长度不必等于可立即释放的唯一块数，本诊断的 `free + victim_owned` 也不直接迁移。原生缓存开启本身和轮转适配后的同预算比较，应分别资格化；本轮未实现、未运行。

复查命令（仅读取小配置）：

```python
from collections import Counter
import json
from pathlib import Path
root = Path('refine-logs/expert_saturation/outputs/admission_capacity')
counts, status_counts, enabled = Counter(), Counter(), []
for path in root.rglob('engine_args.json'):
    args = json.loads(path.read_text())
    flag = args.get('enable_prefix_caching', 'MISSING')
    counts[str(flag)] += 1
    if flag is True:
        enabled.append(str(path))
    status = path.with_name('status.json')
    if status.exists():
        status_counts[str(flag), json.loads(status.read_text()).get('status', 'MISSING')] += 1
print(counts, status_counts, enabled)
```

唯一当前 GPU 接续仍是原冻结的四项运行时重复；本核对不提交另一组实验，也不因未测缓存基线否定已有关闭缓存条件下的测量。
