# F 直接编码分支：冻结保存

**2026-10-09：按用户决定冻结新方法探索，定位为工程候选与测量案例。** 保留全部实现和既有证据；本轮没有新实验、GPU、候锁、外部评论或PR。只有重要应用需求和可区别于已有工作的贡献依据才能重新投入，资源空闲不自动重启。

- [全F证据状态表](../freeze_20261009/evidence.md)：旧候选性能结果、新CPU检查、未完成GPU/端到端验证分开。
- [最小复现、代码版本、运行条件与待验证项](../freeze_20261009/reproduction.md)。
- [完整冻结快照](../freeze_20261009/direct_encoding_snapshot.tar.gz)与[逐文件保全清单](../freeze_20261009/preservation.json)。
- [本分支已有结果](results.md)、[CPU完整语义记录](async_semantics_final.result.json)、[依赖检查](dependency_check_v2.result.json)、[CUDA编译记录](gpu_async_build.json)。
- [旧物化路线逐请求得失案例](../freeze_20261009/request_case.md)与[成果边界](../freeze_20261009/boundaries.md)。该案例不是本分支的性能结果。

CPU桥已删除初版自行引入的整批编码屏障；构造用例支持完整语义与请求独立完成。GPU异步组件可编译，但CUDA执行、首次D2H模型接线、真实候选、完整CPU/GPU成本及客户端收益均未验证。编译和CPU检查不构成性能证据，旧路线失败也不是本分支失败。

源码、历史成功/失败JSON与日志、原source_manifest保持原样。冻结前的README/verdict/protocol已逐字保存在快照；当前文档更新只用于标记冻结。旧source_manifest仍对应冻结前文档，当前源码身份以保全清单核对。协议中的GPU运行计划仅为历史记录，本轮不执行。
