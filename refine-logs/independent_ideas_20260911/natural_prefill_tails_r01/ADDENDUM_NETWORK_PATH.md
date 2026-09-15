# SSH连接路径复核

本次继续执行时，目标文件的研究正文未变化，仓库HEAD仍为`2a37765fe522b1d74609a686f1d327ede7619a50`，冻结输入和运行包保持原样。上一轮完成准备工作；本轮没有GPU执行，新增的是连接故障定位证据。

1. 再次使用目标文件指定入口只读检查，仍返回退出255、`Connection closed by 116.172.94.204 port 11155`，见 `REMOTE-observation-continuation-02.json`。
2. 同一地址可以建立TCP连接，但限量读取的认证前响应是`HTTP/1.1 502 Bad Gateway`及企业网络错误页，没有SSH协议标识，见 `SSH-banner-check.json`、`SSH-preauth-response.json`。没有发送SSH凭据或远端命令。
3. 本机只读路由查询显示该地址经`utun4`接口、网关`192.168.255.10`转发；未发现已启用的系统代理开关，见 `LOCAL-network-observation.json`。这不足以确认故障具体发生在哪一跳，但说明需要检查连接路径，不能仅根据Connection closed认定GPU主机关机或密码失效。

没有修改VPN、代理或路由，没有重传执行包，没有启动或重启任何实验。上一轮两次上传前自动审批超时仍保留在原记录中，本轮未重新测试上传权限。

恢复条件是得到可正常完成SSH握手的已授权入口。恢复后先检查本轮目录、上传包和进程；确认没有已有执行再继续冻结计划，逐块运行并回传。原有SSH与实验授权保持有效，当前缺少的是可用连接路径。自然输入的tiny tail问题仍为`UNRUN`，不产生新性能结论。

第三个连续目标回合再次只读复核，SSH仍退出255并返回Connection closed，见`REMOTE-observation-continuation-03.json`。本轮没有新的GPU或科研结果，未重新测试上传权限，也未重复改变任何网络配置。已按连续三轮同一阻塞规则将长期目标标记为blocked；冻结实验保留UNRUN，待可用SSH路径恢复后继续。
