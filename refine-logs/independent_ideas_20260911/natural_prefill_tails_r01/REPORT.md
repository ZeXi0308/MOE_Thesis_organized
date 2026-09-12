# 自然长度 prefill 基线准备与执行阻塞

## Verdict

`UNRUN`。本轮完成了可执行输入、runner、分析及冻结包，属于准备进展；没有GPU测量，尚未回答自然长度下tiny tail是否出现。不能把连接或权限失败写成科学负结果。

## Evidence type

CPU离线输入处理和执行计划检查。已有GPU分块仅是上一轮的历史事实；本轮目标证据是单卡原生in-process请求级基线。

## What was measured

实际重建并分词16篇完整文章，总29103 tokens，长度733–3011；固定源顺序、排除上一轮16个IDs、无截断。来源边界、原文和token hash、长度及到达序列检查通过。每个引擎1预热+1正式的两块计划均prepare-only退出0，source/input hash一致。

分析脚本通过AST及旧真实native raw的最小检查：16请求prefill守恒，单块请求不被记作tiny tail。此检查不产生本轮性能数据。共享runtime未修改；runner相对上一轮新增30行，输入脚本123行，分析以已有身份/计时/会计逻辑裁剪，没有增加控制器。

运行包已实际创建：16文件、168533字节，SHA256 `92f78162aa1160fcd6c376c3beb463f33135ad610a0695fc51f70f924560ac41`。两次上传调用均被权限工具拒绝创建进程，原因为自动审批超时；包含一次明确允许的重试。之后两次只读SSH观察均为退出255、`Connection closed by 116.172.94.204 port 11155`。保留 `UPLOAD-approval-timeouts.json` 和 `REMOTE-observation.json`，无GPU进程启动记录、无raw、无回传包。

## What was not measured

自然输入的实际prefill分块、排队、TTFT、TPOT、maxITL及完成时间均未运行。16个独立文章和计划32次正式请求执行不是实测样本量。未验证远端当前目录或GPU占用，SSH观察失败不替代运行状态证据。

## Strongest baseline

本轮只有native1024/FCFS/cap8/单请求阈值0，测量自然输入。没有待测策略或相对加速比；旧固定长度cohort只能作历史背景，不能当同输入对照。

## Oracle / headroom status

未测。尾块大小及其所在整步时间也不会被直接转换为可移除成本。

## Claim ceiling

只支持“冻结实验已准备并通过必要CPU检查，GPU执行尚未开始”。保留上一轮单请求512限额无稳定完整收益的结论，不自动复活该策略。

## Failure category

执行基础设施阻塞：上传前的自动审批超时，以及后续SSH连接被关闭。既不是实验逻辑无效，也不是调度机制被证伪。主机是否停止或更换尚待确认。

## Resurrection / resume condition

指定SSH端点可用且上传/执行权限通过。恢复后先观察本轮独立目录和进程，确认是否已有原件；只有无已有执行的证据才启动冻结计划，不因观察超时自动重启。

## One next smallest experiment

执行这两块原生自然长度基线，不改变配置。每块结束先保留、归档、回传全部预热/正式raw及退出状态，核对后再执行下一块。tiny tail预先定义为至少2个实际prefill块且末块1–32 tokens，同时报告全请求和多块请求两个分母。
