# 分析就绪

原件逐步scheduler签名、memory非时钟字段、完整输出比较；全32请求mean/wall/maxITL/recompute/preemptions均保留。旧真实raw自比、时钟偏移不误判、KV字段扰动能检出。四个缺失pair均UNRUN，不产生零值收益。transfer字节及实际容量仍需GPU回读后单独核验。

remote /root/kv-observer-cost-20260914-r01，原SHA b4f51933…0ed894和17项核对通过，未启动driver。

传输核验已加入：只比较worker实际load/store总字节及sizes多重集，不累加lookup matched，允许异步回报分组/时钟不同。旧两on均复算load2415919104/store12884901888字节且sizes一致，off为0；新8格仍UNRUN。
