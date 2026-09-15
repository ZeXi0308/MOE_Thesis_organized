# BLOCKED_RESOURCE_BUSY — 服务窗口主会话

前三个本会话连续等待turn均现场核验原Qwen r04：23:28的ps确认6954/6967及原命令；1789399836.583与1789399952.236的/proc读取确认同命令、starttime_ticks分别1030824445/1030825491。第三次fcntl非阻塞检查共同`/root/autodl-tmp/moe-research-gpu.lock`仍busy。不是陈旧状态文件或连接超时，也没有判定Qwen失败。

HEAD仍de64dae5，共享dirty修改保留。资源等待前已完成：恢复前KV增长证书、声明cap模型修正、原生恢复事件成本映射、轻量请求计时入口及3项CPU行为测试；重复保存runner的失败raw保留补丁在独立worktree实际应用并通过2项异常fixture。现有分析不能代替完整重复保存/搬运/同步GPU执行，继续增加检查不会关闭该不确定性。

重复保存原方已暂存21文件包6f412e39…f2c3、无新启动。主树/远端冻结包尚未吸收`failure_retention/preserve_capture.patch`；修正版独立runner SHA为cb406646…4172，原方须以明确新包版本接续并保留旧包，勿沿用旧SHA。测量入口尚未接入GPU runner，不能写GPU已资格或性能收益。

恢复条件：B确认原Qwen整组终态，然后独立核验同一receipt、PID身份、GPU与共同锁，按既定A repeated-staged→原funding单victim顺序接续。A先验证重复保存off/on的实际生命周期和完整成本，再判断是否需要独立性能测量；主会话复用原件，不启动重复组。host配置16GiB与实测host用量、最后请求返回与资源排空须分别记账。

研究状态仍OPEN；完整延迟—效率收益未证明。goal标记blocked只停止资源依赖下的自动空转，不是科学NO-GO或目标完成。没有创建后台候卡器、终止他人进程、改动父cgroup或启动付费资源。
