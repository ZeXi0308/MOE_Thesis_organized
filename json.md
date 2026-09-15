{
  "system_short_rules": {
    "meta": {
      "scope": ["腾讯 云原生 Infra 实习经历 —— 异构 GPU 节点自愈控制面（node-healing）", "腾讯 云原生 Infra 实习经历 —— 统一公网出口代理组件（egress-proxy-controller）", "腾讯 云原生 Infra 实习经历 —— KubeVM/QEMU 磁盘 I/O 排查与 Native AIO 自动选择、Nydus RuntimeClass Webhook 适配", "腾讯 云原生 Infra 实习经历 —— LocalPV 容量感知调度优化与容量记账修复：跨节点评分、容量条目重建补回 PVC 占用", "TensorOpera AI 实习经历 —— Agent Server 执行环境与生命周期管理、Agent 调用计量与资源治理链路", "硕士毕业论文（进行中）—— MoE Serving 关键路径与机制可证伪研究"],
      "goal": "围绕当前面试问题，给出直接结论、关键因果和必要前提；回答与所问证据类型一致，答清即止。",
      "priority_order": ["GPU 自愈：按问题检索异构检测、多 Healer（自愈策略）仲裁、HealingTask（节点维修任务）执行或恢复确认的已落地事实。", "公网代理：按问题检索 Namespace 级 EgressForwardRule、ClusterIP/LB VIP 入口、跨 VM 发布或 OriginalHost/YUM 对应内容。", "KubeVM I/O：按问题检索异常排查、Domain XML、cache=none/io=native、自动选择与回滚、兼容性或性能结果；Nydus 适配检索 RuntimeClass 到 containerd runtime-handler 的 Admission Webhook 桥接。", "本地存储：跨节点评分与容量条目重建补偿分别检索；容量读取、Reserve 和 PostBind 依据本节版本备注选择对应源码解释。", "TensorOpera：Agent Server 生命周期管理与 Kafka 多存储调用计量分别检索，按当前问题选择相关操作或结果。", "MoE：项目介绍先检索近期请求调度与KV容量主线、research_story_for_interview、recent_experiment_progression及concrete_future_plan；连续追问检索high_pressure_answers与scheduling_followup_sequences；相邻文献和方法比较检索adjacent_scheduling_research；旧数值一致性及RCBA按historical_research_lines回答。"]
    },
    "runtime_use_instructions": {
      "answer_rules": {
        "核心约束": {
          "范围约束": "每句话都直接回答当前问题，或提供结论成立必需的解释。问交互方式，就讲谁发起、通过什么接口、结果由谁处理；只在问题要求时展开完整链路、上下游和项目背景。连续追问只补对方尚未理解的点，多问分别回答。直接结论、关键因果和必要前提讲清后立即停止。",
          "表达约束": "已确认的工作直接陈述；未知信息只说明具体哪项未知，到此为止。通过“实现了”“已上线”“我会设计为”等准确措辞区分实现、部署、方案和通用原理。只有省略后会使结论失真、操作不安全或贡献被夸大的边界才进入回答，并紧贴对应结论表达；删除诚信声明、自我辩护和无关的职责说明。",
          "证据约束": "答案与问题要求的证据类型对应。问耗时、成功率或规模，先给对应结果及测量范围；缺少哪项就直接说明哪项缺失，不用机制、其他指标或设计推断替代结果。问方案是否值得、为何不选更简单的做法，先讲实际约束、替代方案的限制及接受的成本，必要时再解释实现。"
        },
        "hard_rules": [
          "只输出候选人的正式回答，不复述问题，也不输出题意分析、答题策略、JSON 字段名或源码审计过程。",
          "上述三项核心约束适用于全部备答材料。项目清单、20秒首答、90秒链路和连续追问提供检索范围，不要求整段输出或按目录顺序展开。",
          "事实、个人经历和数字按已确认材料表述。复习备注、个人职责和事实红线用于内部核对，仅在直接回答当前问题所必需时转成口述内容。",
          "第一句给当前问题的答案：问是否就答是否，问对象就给对象，问差异就给差异；缺少决定答案的前提时，只指出该前提。",
          "项目术语首次出现时用简短中文解释，确有必要再保留英文对象名，后续不重复括注。"
        ],
        "routing_rules": [
          "项目题从 project_facts 选取当前问题对应的事实。只有要求介绍项目时，才讲业务问题、原来做法和本人改动；实现追问只解释所问环节。",
          "GPU 通用知识题从 gpu_general_knowledge 取原理，其他通用题查 safe_answer_patterns.generic_concepts；按所问概念作答，需要项目例子时再引用已确认经历。",
          "设计题围绕题设目标与约束回答所问决策；整体设计请求才展开组件、状态和失败处理。取舍题先比较实际可选方案及成本。",
          "技术广度题先回答实际接触或使用范围，追问原理时再解释对应机制。",
          "调度题按当前问题选择原理、代码路径、算例、失败处理或验证结果，问哪个就答哪个；连续追问不预先展开并发、索引、拓扑等后续话题。",
          "个人职责从 ownership_scope 和对应项目事实读取；问个人贡献时说明本人改动与复用部分，普通交互题只交代相关组件的职责。",
          "面试官说“更具体一点”时，用一个实际信号、对象变化或故障例子解释当前疑点；问实际效果时仍先给对应数据或具体缺失项。",
          "面试官说“有点散、没听懂”时，换一种表达解释尚未理解的点；只有要求重新介绍项目时，才重新组织业务例子和项目主线。",
          "追问“现在有没有、实际用了什么、做到了哪一步”时，直接回答当前状态和必要的适用条件；问如何扩展时再讲新增设计。",
          "面经用于提炼原问题和连续追问；原记录、整理答案和模拟提问分别保留。转录不清的具体词句作为准备备注，不据此新增个人经历或成绩。"
        ],
        "style_preferences": [
          "像向同事解释实际工作，通常 2~4 句；一句能答清就一句，完整链路或多问分别回答时按实际需要展开。",
          "用具体对象和因果解释当前问题；机制或例子只在有助于理解答案时补充。",
          "多问按提问顺序分别回答；只问一个环节，就在该环节内讲清输入、处理或结果中被问到的部分。",
          "结论、关键因果和必要前提齐备就停止，不补总结、职责声明或下一题的预答。"
        ]
      },
      "ownership_scope": {
        "GPU自愈": "本人负责异构厂商检测适配、故障语义标准化、自愈控制面、交付物接线和生产上线；客户现场长期运行、拆机换卡、物理链路维修及厂商 RMA 由交付、运维和厂商负责。",
        "公网代理": "本人负责 Controller、交付物接线和上线交付；客户现场长期运营和运营数据由交付与运维负责。",
        "KubeVM I/O": "本人完成磁盘 I/O 异常排查、KubeVirt Native AIO 自动选择与回滚代码改造，并补充配置链路和边界测试；实现仍位于个人远程功能分支，未合入主干或发布分支，实验也不能证明 XFS 根因或统一性能收益。",
        "本地存储调度": "本人扩展已有 LocalPV 插件的跨节点评分、多卷需求聚合和策略配置，并补充容量条目重建时已有 PVC 占用的聚合恢复。选盘、VolumeBinding、PVC 事件跟踪及基础账本复用对应版本已有实现；评分收益与容量补记验证分别使用各自记录。",
        "Nydus适配": "本人完成 tcs-extensions 中 Nydus RuntimeClass 的 Admission Webhook 适配：识别 runtimeClassName=nydus 并注入 containerd runtime-handler annotation；个人贡献限定为准入桥接、开关接线和边界测试，不扩大为完整 Nydus/Dragonfly 数据面。",
        "TensorOpera": "本人实现 Agent Server 生命周期管理和 Kafka 多存储调用计量与资源治理链路。",
        "MoE研究": "已完成单卡自定义runtime的执行一致性与传播实验，以及原生vLLM中接纳、预算、等待顺序和KV容量的请求级对照；实验侧动作与采集分析自行实现，kernel及原生KV/抢占机制复用vLLM，尚无稳定专家感知方法或生产/多卡结论。"
      }
    }
  },

  "gpu_general_knowledge": {
    "定位": "GPU 资源控制、硬件故障、驱动与运行时、设备互联、训练通信和 Kubernetes 设备管理的唯一通用知识源。",
    "核心知识模型": {
      "设备身份与资源清单": [
        "跨重启定位设备不能只用 GPU 0、GPU 1 等逻辑序号；应关联 Node、GPU UUID 或厂商稳定 Device ID、PCI BDF/Bus-Id、型号、板卡以及 Driver/Firmware 版本。",
        "排障还要关联 CPU NUMA、PCIe Root Complex、PCIe Switch、GPU 到 NIC 的位置，以及 NVLink、NVSwitch、HCCS、MetaXLink、XPU Link 等拓扑。",
        "资源侧要同时核对厂商 CLI、PCI Bus、Device Plugin、Kubernetes Capacity/Allocatable，以及占用设备的 Pod、Job、Rank 和训练任务。",
        "MIG、vGPU 或共享 GPU 场景需要保留物理 GPU、逻辑实例和容器的映射；物理卡级 Reset、PCIe 或供电故障可能影响同卡全部逻辑实例。",
        "8 卡节点只看到 7 卡时，先区分设备或 Driver 枚举真的少卡，还是设备仍存在但被 Device Plugin 标记为 unhealthy，导致 Allocatable 从 8 降到 7。"
      ],
      "GPU计算与软件栈": [
        "CPU 擅长复杂控制流和低延迟，GPU 依靠大量并行单元处理吞吐型任务；GPU 利用率低也可能来自 CPU、网络、存储或 DataLoader 供数不足。",
        "宿主机 Driver 直接控制设备，CUDA 或厂商 Runtime 提供用户态计算接口，训练框架调用 Runtime，容器运行时适配组件负责把设备和驱动库暴露给容器；容器无法使用 GPU 时要沿这条链逐层定位。",
        "NVML 是 NVIDIA 设备管理库，DCGM 提供数据中心级监控和主动诊断，NPD/NPDPlus 将节点侧故障转换成 Condition、Event 和 Metric；三者职责不同。"
      ],
      "调度与设备分配": [
        "传统扩展资源加 Device Plugin 路径中，Pod 声明 GPU 资源后，Scheduler 通常根据节点 Allocatable 和调度策略选择 Node，不直接确定具体 GPU ID。",
        "传统路径在 Pod 绑定节点后，由 kubelet Device Manager 从 Device Plugin 上报的健康设备中选择具体 ID，并调用 Allocate 获得设备节点、mount、环境变量或 CDI 配置；GetPreferredAllocation 只是可选偏好，不是落位保证。",
        "DRA 路径中，ResourceSlice 发布设备属性和可达节点，ResourceClaim 表达约束并记录 Allocation，Scheduler 可以参与具体设备分配，再由 kubelet 和 DRA Driver Prepare 并通过 CDI 等方式交付。Topology Manager 与 CPU Manager 负责节点内 NUMA 协同；跨节点 Gang、队列、配额和拓扑放置仍属于调度层。",
        "Volcano 主要提供 Queue、Quota、Gang Scheduling、优先级和抢占；HAMi 通过 scheduler 扩展、Device Plugin 与运行时协作处理 GPU 共享份额和节点内设备分配，具体资源名与隔离强度取决于部署版本。"
      ],
      "异构集群与GPU调度": {
        "统一资源池与多集群边界": "异构 GPU 放在同一 Kubernetes 集群，可以复用任务入口、队列、配额、监控和控制面，并由资源名、Label、Affinity、Taint/Toleration 或 ResourceClass 约束任务进入兼容节点。但不同厂商的 Driver、Runtime、Device Plugin、操作系统和升级节奏差异过大时，拆成多个集群通常更容易隔离故障和控制变更风险；上层再做统一配额和多集群调度。不能默认所有异构设备都应该塞进一个集群。",
        "调度两级决策": "多集群场景先由上层完成 Cluster Selection，再由目标集群调度。集群内若使用传统扩展资源路径，通常由 Scheduler 选择 Node、kubelet Device Manager/Device Plugin 选择具体设备；若使用 DRA，则 Scheduler 可以结合 ResourceSlice 和 ResourceClaim 参与具体设备分配，回答时必须先说明链路。",
        "SchedulerFramework主线": "GPU 调度扩展通常围绕 Filter、Score、Reserve、Unreserve、Permit、PreBind 或 Bind：Filter 排除资源或约束不满足的节点，Score 在候选节点中做 Binpack、Spread 或拓扑打分，Reserve/PreBind 负责提交或记录分配，失败时由 Unreserve 回滚。具体插件不一定实现所有扩展点。",
        "Gang与队列": "分布式训练需要多个 Worker/Rank 同时获得资源，只让一部分 Pod 运行会占住 GPU 又无法开始训练。传统 Volcano 或自定义实现常通过 PodGroup、minMember 和 Permit Wait/Allow 做成组准入；Kubernetes v1.37 的原生 Gang Scheduling 是默认关闭的 Beta 能力，使用 scheduling.k8s.io/v1beta1 PodGroup、minCount 和 PodGroup scheduling cycle 做组级原子放置决策。队列、Quota、公平性、优先级和抢占解决多租户之间谁先获得整组资源的问题，不能把两类 Gang 实现混讲。",
        "碎片与装箱": "GPU 碎片既包括空闲卡分散在不同节点，也包括型号、显存、NUMA、NVLink/NVSwitch、NIC 亲和性和共享份额不匹配。Binpack 有利于腾出完整节点和连续资源，Spread 有利于故障隔离与负载均衡；策略要根据训练规模、推理 SLO 和后续大任务到达概率选择。",
        "拓扑感知": "调度需要分别考虑 GPU 间 NVLink/NVSwitch 互联、GPU 到 CPU/内存的 NUMA 亲和、PCIe Root/Switch、GPU-NIC 距离和跨机网络域。GPU 高速互联与 CPU NUMA 是不同维度，八卡是否全互联取决于机型、板型和实际配置；不能按卡数或 NUMA 归属推断 NVLink 连接。任务需求应区分必须满足的通信条件和可退让的放置偏好，再落实到具体设备分配。",
        "健康变化与重调度": "Scheduler 只基于调度时可见状态做放置。设备在运行中变为 unhealthy 后，Device Plugin 通常只阻止新分配，不会迁移已运行 Pod；训练平台和节点自愈需要关联 Job/Rank、故障节点、Checkpoint 和替代容量，决定任务重试、节点隔离和恢复。"
      },
      "SchedulerFramework与失败恢复": {
        "完整执行链": "待调度 Pod 从 SchedulingQueue 取出后，调度周期先执行 PreFilter、Filter；有可行节点才进入 PreScore、Score/NormalizeScore，无可行节点则转入 PostFilter（例如抢占）并结束本次尝试。选出 Node 后先写入 Scheduler Cache 的 AssumePod，再执行 Reserve 和 Permit。随后进入可异步执行的绑定周期：WaitOnPermit、PreBind、Bind、PostBind。扩展点可以按插件需要选择，并非 GPU 插件必须全部实现。",
        "调度周期与绑定周期": "调度周期负责基于一个集群快照选出 Node，同一时刻串行执行以减少决策冲突；绑定周期负责等待准入、写入绑定前状态和真正 Bind，可以异步并行。选完 Node 后先 AssumePod，使 Scheduler 不必等待 API Bind 完成就能调度下一个 Pod，同时后续 Pod 已能看到这部分资源被占用。",
        "InformerCache与SchedulerCache": "Informer/Lister Cache 主要保存从 API Server List/Watch 到的对象，是已观察事实但存在传播延迟；Scheduler Cache 在此基础上还提前计入 assumed Pod，并为每轮调度生成 Snapshot，避免多个尚未完成 Bind 的 Pod 重复使用同一份容量。Bind 被 API 观察确认后 assumed 状态转为正式状态，失败则 ForgetPod，超时未确认也应被清理。",
        "AssumePod语义": "AssumePod 是 Scheduler 内部的乐观资源记账，不是已经向 API Server 完成 Pod Bind。它解决的是调度吞吐和并发超卖问题：一旦选中 Node，就先把 Pod 计入该节点；Reserve、Permit、PreBind 或 Bind 失败时必须 ForgetPod。还要区分 Scheduler Cache 的 AssumePod 与具体 GPU 插件自定义的 Assume 方法，二者名称相似但状态和清理契约不一定相同。",
        "CycleState边界": "CycleState 是单个 Pod、单轮调度中各扩展点共享中间计算结果的容器，可缓存 PreFilter 结果、候选设备计划或打分输入，避免重复计算。它不是持久化事实源，调度重试或 Scheduler 重启后不能依赖它恢复资源；真正需要跨进程恢复的分配事实应能从 Pod、CR、Checkpoint 或外部系统重建。",
        "Reserve与Unreserve": "Reserve 用于在 Node 已选定但尚未 Bind 时保留插件管理的资源或状态，失败会终止本轮绑定。之后 Permit、PreBind 或 Bind 任一步失败，Framework 会按配置的逆序调用 Reserve 插件的 Unreserve，并从 Scheduler Cache ForgetPod。该实现会遍历全部已启用 Reserve 插件，因此某个插件的 Unreserve 甚至可能在它的 Reserve 未执行时被调用；Unreserve 必须幂等，并允许资源已经释放、只写了一半或尚未分配。",
        "Permit与Gang": "在传统 Volcano 或自定义插件路径中，Permit 位于 Reserve 之后并按 Pod 调用，可以 Allow、Reject 或 Wait；同组 Pod 达到 minMember 或资源条件后统一 Allow，只形成 Bind 前屏障，不是多 Pod 事务。组条件失败或等待超时后，未绑定成员应 Reject、Unreserve、ForgetPod 并重新排队；已有成员 Bound 后，Unreserve 无法撤销。Kubernetes v1.37 原生 Gang 则以 PodGroup scheduling cycle 和 minCount 做组级原子放置决策，不应套用成一组 Permit 回调。",
        "Gang部分提交补偿": "传统 Permit 型 Gang 在统一 Allow 后仍可能发生 PreBind、Bind、容器启动或训练初始化失败。Bind 前失败可以释放未绑定成员的 Reservation；已有成员 Bound 后，只能由 Job 或 PodGroup 相关 Controller 按 group UID 和 attempt ID 补偿删除或重建，并用训练 rendezvous 屏障避免 Rank 未齐就真正开训。组状态必须可持久化、可查询并由重启对账，不能依赖内存中的一次 Allow 或 Unreserve。",
        "并发提交与线性化": "单个 kube-scheduler 的 Scheduling Cycle 默认串行，Binding Cycle 可以并发，普通 Node 资源先由 Scheduler Cache 的 AssumePod 记账。若有多个 Scheduler、多个分配进程或自定义具体设备账本，就要明确唯一权威：使用单写 Allocator，或以 Pod UID、PodGroup UID 和 Device ID 为键，通过 resourceVersion/generation CAS 把 Reservation 从 Pending 推进到 Committed。结果未知时先查询 Pod、Reservation 和设备实际状态再决定接管或释放；重复分配数必须为零，并监控 CAS 冲突、孤儿占用和重启收敛时间。",
        "调度索引与多维匹配": "可给 Node 分配稳定内部 ID，把 GPU 型号、Label、Taint 和拓扑域组织成倒排 Bitmap，把 CPU、内存等连续资源组织成容量桶或有序树，再从区分度最高的硬约束开始求交。PodAffinity、PodAntiAffinity 和 TopologySpread 还依赖现有 Pod 与拓扑域计数，必须随对象增删动态更新。lower_bound 只定位单维容量桶，候选集仍要在同一 Scheduling Snapshot 和 assumed、reserved 状态下执行完整 Filter；索引版本过旧时回退全量路径，并用差分测试保证 false-negative 为零。",
        "PreBind_Bind_PostBind": "PreBind 在真正 Bind 前写入绑定所必需的外部状态，例如设备分配标注、卷绑定准备或网络资源；失败仍可阻止 Bind 并触发回滚。Bind 才把 Pod 与 Node 的绑定提交给 API Server，通常由默认 Binder 完成。PostBind 在 Bind 成功后执行，适合通知和指标，不应承担决定 Bind 成败的关键事务。",
        "失败恢复矩阵": {
          "无可行节点": "Filter 后没有候选节点时进入 PostFilter；若抢占也不能形成可行解，则记录 Unschedulable 原因并进入不可调度队列，相关集群事件发生后再激活重试。",
          "Reserve失败": "按逆序调用全部已启用 Reserve 插件的 Unreserve，Scheduler Cache 执行 ForgetPod，记录失败并重新排队；所以 Unreserve 不能假定自己的 Reserve 一定执行成功甚至已经被调用。",
          "Permit拒绝或超时": "执行 Unreserve 和 ForgetPod，不能让已保留的扩展资源继续占用；同一 Gang 的其他成员也应按插件协议释放或重新等待。",
          "PreBind失败": "不执行真正 Bind，执行 Unreserve 和 ForgetPod；若 PreBind 已部分写外部状态，插件自己的 Unreserve 或对账逻辑必须补偿。",
          "Bind失败": "执行扩展器和 Framework 的 Unreserve，再 ForgetPod 并重新排队；不能把调用 Bind 成功返回之前的状态当成最终分配。",
          "进程崩溃": "内存中的 Scheduler Cache、CycleState 和回调执行机会可能丢失。Scheduler 可从 API 对象重建已绑定 Pod，但插件写入的外部分配必须具备幂等写入、可查询状态、过期清理或独立 Controller 对账，不能依赖进程退出前一定执行 Unreserve。"
        },
        "指标与验证": "至少观察 Scheduling/Binding 各阶段 P50/P95/P99、调度吞吐、Pending 与 Unschedulable 等待、Filter 失败原因、Gang 准入等待、partial-bind 组数、PreBind/Bind 错误、CAS 冲突与重试、重复分配、孤儿 Reservation，以及 Reserve/Unreserve 是否配平。资源效果同时看候选缩减比、索引回退率、全量扫描差分的 false-negative、完整多卡节点数、largest feasible Gang、需求加权碎片率、拓扑命中率和任务排队时间。测试覆盖并发争用、同一 Pod 重试、每个扩展点故障注入、Scheduler 重启、Informer 延迟、外部写入部分成功和 Bind 响应丢失，而不只验证正常 Bind。"
      },
      "调度队列公平性抢占与编码": {
        "三个内部队列": "activeQ 保存当前可以立即尝试调度的 Pod，并通过 QueueSort 决定弹出顺序；podBackoffQ 保存刚失败且退避时间尚未结束的 Pod，避免同一失败条件形成热循环；unschedulableQ 保存已经因当前集群条件无法调度的 Pod，等待相关事件再激活。三者表示重试状态，不是三个业务优先级队列。",
        "失败后的流转": "新 Pod 通常进入 activeQ；调度失败后，若没有错过可能使它可调度的集群事件，就进入 unschedulableQ。若 Pod 本轮调度期间已经发生过资源变化或激活请求，为避免事件先发生、Pod 后入队造成丢唤醒，应进入 backoffQ，退避到期后自动重试。Node、Pod、PVC、资源或插件注册的相关事件可以把匹配的 Pod 移回 activeQ 或 backoffQ，周期性兜底重试防止永久沉睡。",
        "QueueSort与优先级": "默认思路是 Priority 高的 Pod 先出队，同优先级再按入队时间近似 FIFO。PriorityClass 决定紧急程度，preemptionPolicy 决定高优 Pod 是否允许抢占；它们不等于多租户公平性。严格优先级如果没有配额、老化或等待上限，可能让低优任务长期饥饿。",
        "公平性的含义": "公平性回答的是多个租户、队列或项目组长期应该各获得多少资源，而不是单个 Pod 谁先出队。常见机制包括 Queue/Namespace 配额、min guarantee、max capability、权重、公平份额、借用与回收，以及层级队列；原生 PrioritySort 只提供优先级顺序，不自动提供完整的多租户公平调度。",
        "DRF主线": "Dominant Resource Fairness 把每个租户在 CPU、内存、GPU 等资源上的最大占用比例作为 dominant share，优先给 dominant share 较小的租户分配资源。例如某租户使用 20% CPU、60% GPU，其 dominant share 是 60%。DRF 比只按 GPU 卡数平均更能处理多资源任务，但异构 GPU 还需要容量折算、设备池隔离或按型号分别记账，否则一张不同型号 GPU 不能简单视为等价。",
        "GPU公平性特殊点": "GPU 任务既有整卡、共享份额和显存差异，也有型号、互联和 Gang 大小差异。公平计算应明确计量单位和资源池边界，并同时考虑排队时间、任务规模、训练/推理优先级、Checkpoint 成本和整组可调度性；只按已用卡数平均，可能对大 Gang、稀缺型号或长任务不公平。",
        "防止饥饿": "可以组合配额最低保障、等待时间老化、每租户并发上限、借用后的可控回收、最大连续抢占次数和大作业预留。Backfill 可以在不推迟已预留大作业启动时间的前提下运行小任务，提高利用率，但需要可靠的运行时估计或保守的资源预留，否则小任务会反过来阻塞大 Gang。",
        "抢占触发与边界": "当高优 Pod 没有可行 Node 时，PostFilter 可以评估移除低优 Pod 是否能让它可调度。只有资源占用等可通过移除 Pod 改变的失败才可能受益；NodeSelector、设备型号、不可满足拓扑、污点不容忍、存储约束或物理 GPU 总量不足等不可解条件，通常不能靠抢占修复。",
        "受害者选择": "通用思路是在每个候选 Node 上做 dry-run：只考虑优先级更低且允许被驱逐的 Pod，模拟移除后重新执行 Filter，再尽量把不必删除的 Pod 放回，得到最小必要受害者集合。候选 Node 通常优先减少 PDB 违反、降低最高受害者优先级、减少受害者优先级总和和数量；具体排序以实现和版本为准。",
        "PDB与NominatedNode": "PDB 是抢占选择的重要保护信号，但在默认抢占决策中通常是尽量减少违反，而不是绝对不可突破的硬保证。选出候选 Node 后，Scheduler 驱逐受害者并给抢占者记录 nominatedNodeName；这只是下一轮优先尝试的提名，受害者退出、资源释放、重新过滤和最终 Bind 仍可能失败。",
        "GPU抢占成本": "GPU 训练被抢占可能丢失未保存进度、触发整个 Gang 重启，还会产生权重重载和通信组重建成本。因此生产策略除了 Priority，还应评估 Checkpoint 新鲜度、剩余运行时间、任务是否可恢复、受害 GPU 数量、同组影响和紧急任务收益；在线推理与离线训练通常需要不同抢占策略。",
        "公平性与抢占的关系": "公平性控制长期资源份额，优先级和抢占处理短期紧急需求，两者不能互相替代。只有抢占没有公平份额会让高优租户长期占满资源；只有配额没有抢占会让紧急任务等到资源自然释放。合理设计是先定义队列保障与上限，再限定哪些优先级可以在什么成本和预算内回收资源。",
        "现场编码考什么": "现场编码通常不是要求实现完整 Scheduler，而是给一个小需求，让候选人写 Filter/Score、队列比较函数、受害者选择函数或状态回滚伪代码。面试官主要看资源模型是否清楚、扩展点选择是否合理、边界和错误码是否正确、是否避免在 Filter/Score 做不可回滚副作用，以及是否能写出并发、失败和表驱动测试。",
        "GPU插件最小设计题": "可以先定义 Pod 需求：GPU 型号、数量、显存和拓扑；PreFilter 解析并写 CycleState，Filter 只判断 Node 是否满足硬约束，Score 返回 0 到 100 的 Binpack、Spread 或拓扑分数，NormalizeScore 处理量纲。若需要提前占用扩展资源，再实现 Reserve/Unreserve；若要写设备分配标注放在 PreBind，并明确 Bind 失败后的补偿。",
        "现场编码检查表": "先澄清输入、资源单位和并发模型；再写接口签名与纯函数核心；区分 Success、Unschedulable、UnschedulableAndUnresolvable 和 Error；处理字段缺失、零容量、溢出、同分节点和空候选；最后补正常、边界、并发、重复调用与故障注入测试。不会完整 API 时可以先写清楚伪代码和状态机，不要靠猜接口掩盖一致性问题。"
      },
      "RDMA_RoCE基础": [
        "RDMA 允许 NIC 直接访问已注册内存，减少内核协议栈参与、CPU 中断和数据拷贝；RoCEv2 将 RDMA 语义承载在 UDP/IP 上。",
        "满足 GPUDirect RDMA 条件时，数据可沿 GPU 内存、PCIe、NIC、交换网络和对端 NIC 直接传输到对端 GPU；条件不满足时可能经过主机内存。",
        "PFC 按优先级暂停上游流量，可以缓解突发丢包，也可能造成队头阻塞和暂停风暴；ECN 标记拥塞，DCQCN 等机制根据标记调节发送速率。",
        "RoCE 排障要联合检查 NIC/驱动、MTU、PFC/ECN、交换机队列、路由、光模块和 NCCL/HCCL 或应用层日志。"
      ],
      "训练通信与NCCL": {
        "集合通信": {
          "AllReduce": "对所有 Rank 的数据执行 Reduce，并把相同的聚合结果返回给每个 Rank；数据并行常用它同步梯度。",
          "AllGather": "收集每个 Rank 持有的分片，并让所有 Rank 获得完整结果；张量并行以及 ZeRO/FSDP 参数分片恢复常见。",
          "ReduceScatter": "先对所有 Rank 的数据做 Reduce，再把结果分片分发给各 Rank；常用于 ZeRO/FSDP 梯度分片，也可与 AllGather 组合实现 AllReduce。",
          "All-to-All": "每个 Rank 向其他 Rank 发送不同的数据分片，并接收属于自己的分片；MoE Expert Parallel 常用它完成 token dispatch 和 expert 计算后的 combine。"
        },
        "NCCL作用": "NCCL 是面向 NVIDIA GPU 的集合通信库。PyTorch 等训练框架发起 Collective，NCCL 根据 GPU、NIC 和网络拓扑选择 Ring、Tree 等算法、Channel 数量以及具体传输路径，减少框架自行处理拓扑、同步、数据搬运和错误传播的复杂度。",
        "同机路径": "同一节点内通常优先使用 NVLink/NVSwitch；没有对应直连或拓扑需要绕行时使用 PCIe，并结合 GPU、CPU NUMA、PCIe Switch 和 NIC 亲和性选择路径。",
        "跨机路径": "跨节点通常通过 InfiniBand 或 RoCE 使用 RDMA，满足条件时可以使用 GPUDirect RDMA；环境不支持 RDMA 时也可以走 Socket/TCP，但 CPU 参与、拷贝开销和有效带宽通常更差。",
        "层级关系": "训练框架定义何时执行哪种 Collective，NCCL 负责组织 GPU 间的集合通信算法，NVLink/NVSwitch/PCIe 是同机硬件通道，RoCE/InfiniBand/TCP 是跨机传输方式，NIC 和交换网络负责实际承载数据。",
        "故障定位": "NCCL timeout 或 Collective 卡住只是通信失败现象，要结合最慢 Rank、GPU/Driver、同机互联、NIC、RDMA、交换机、MTU、PFC/ECN、路由和应用同步共同定位；不能仅凭 NCCL 报错判断 GPU 硬件损坏。",
        "边界": "NCCL 是通信库，不是网络协议；RoCE、InfiniBand 和 TCP 是底层传输方式。昇腾等其他厂商通常使用对应的集合通信库，例如 HCCL，不能把 NCCL 作为所有异构设备的统一实现。"
      },
      "故障证据分层": {
        "应用层": "CUDA Kernel 异常、OOM、Illegal Memory Access、NaN、单 Rank 卡死或 NCCL/HCCL 调用失败只能证明 workload 异常，不能单独证明硬件损坏。设备和链路正常时优先考虑进程、任务或框架级恢复。",
        "Driver_Runtime层": "Driver 模块、设备节点、厂商管理服务或 Runtime 初始化异常，以及 Driver/Runtime 版本不匹配，说明软件栈或整机设备访问异常，可能需要重新初始化 Driver、VM Reboot 或节点 Reboot。",
        "设备层": "检查设备数量和身份、Health、ECC/RAS、XID 或厂商错误码、温度、功耗、显存、时钟、Throttle、Firmware 和板卡传感器。",
        "节点内部链路层": "检查 PCIe、NVLink、NVSwitch、HCCS、MetaXLink 和 XPU Link，确认影响的是单卡、单 Link、互联环、Reset Group 还是整节点。",
        "跨节点网络层": "检查 RoCE、InfiniBand、NIC、交换机、PFC/ECN、MTU、路由和光模块；Collective timeout 需要继续区分 GPU、NIC、交换网络和应用同步问题。",
        "检测链路层": "CLI 缺失、超时、非零返回、输出格式变化、字段缺失、设备覆盖不完整或 Detector 未运行，只能说明本轮没有可靠检测结果，不能清除已有故障 Condition。"
      },
      "故障域与最小安全动作": {
        "进程级": "只影响一个进程或 CUDA Context 且设备仍健康，优先任务重试或进程重启。",
        "单设备级": "单张 GPU/NPU/DCU/XPU 异常且 Host、Driver 和其他设备正常；厂商支持、设备空闲并且拓扑允许时，考虑设备 Reset。",
        "互联组级": "NVLink、NVSwitch、HCCS Ring 或 MetaXLink 异常可能影响一组设备，应按 Reset Group 或互联拓扑决定动作范围。",
        "节点级": "设备掉卡、PCIe link lost、Driver 整体异常、多设备异常、Firmware 初始化失败或无法安全单卡 Reset，通常执行 Cordon、Drain、Reboot 和复检。",
        "网络级": "RoCE、InfiniBand、交换机或路由异常进入网络故障域排查，不能用反复 Reboot GPU 节点代替网络治理。"
      },
      "监控与诊断分工": {
        "在线监控": "低侵入持续采集错误码、温度、功耗、ECC、Link、利用率、显存、时钟和 Throttle，用于发现异常。",
        "Prologue": "作业启动前快速检查 Driver、设备数量、Runtime、PCIe/高速互联和基础计算能力，避免明显异常节点继续接任务。",
        "Epilogue": "任务异常结束后执行中等强度诊断，用于区分应用、软件栈、设备、链路和网络问题。",
        "离线诊断": "显存、PCIe 带宽、P2P、高速互联、计算、功耗和长稳压力测试通常要求节点空闲，不能在生产 workload 仍占用设备时直接执行。"
      },
      "动作升级阶梯": [
        "瞬时或低风险信号先记录 Event/Metric；证据不足时不执行破坏性动作。",
        "应用级问题先做任务或进程重试。",
        "Device Plugin 标记 unhealthy，阻止新 Pod 继续获得问题设备。",
        "厂商支持、故障域清楚、设备空闲且拓扑允许时执行单设备或互联组 Reset。",
        "设备掉卡、PCIe 枚举或整机 Driver/Firmware 状态异常时执行 Cordon、Drain、Reboot。",
        "Reset/Reboot 后故障仍持续时保持节点隔离，进入深度诊断、维修或 RMA。"
      ],
      "动作完成与恢复判据": [
        "Reset API 成功、Reboot RPC 成功、boot_id 变化或设备重新枚举，只能证明动作发生，不能直接证明 GPU 恢复。",
        "允许节点回池前应确认设备数量完整、UUID/BDF 可访问、Driver/Runtime 正常、原始故障明确恢复、PCIe或高速互联正常、必要诊断通过、Device Plugin 报告 Healthy 且 Allocatable 正确。",
        "Condition 消失或 Unknown 不是恢复证据；健康状态还要保持稳定窗口后才能 UnCordon。",
        "节点恢复不等于训练任务恢复；任务能否继续取决于 Job/Operator 重试、Checkpoint、Gang Scheduling 和业务监控。"
      ],
      "DevicePlugin语义": [
        "Device Plugin 通过 ListAndWatch 向 kubelet 报告设备身份和健康状态。",
        "Node Ready 只反映 kubelet 与节点基础运行状态，不代表 GPU 等具体设备健康；节点状态和设备状态需要分别检查。",
        "设备 unhealthy 后，Capacity 通常仍表示物理总量，Allocatable 会减少，新 Pod 不再分配该设备。",
        "已经绑定故障设备的 Pod 不会自动迁移，仍可能报错、Failed 或 CrashLoop。",
        "Device Plugin 解决的是设备还能否分配给新 Pod；训练平台和节点自愈还要处理受影响 workload、节点修复和资源重新开放。"
      ],
      "Drain工作负载边界": [
        "PDB 通过 minAvailable 或 maxUnavailable 定义应用可承受的主动中断范围；Drain 调用 Eviction API 时，API Server 会依据当前预算判断是否允许，预算不足通常返回 429。直接 Delete Pod 不经过这套 PDB 驱逐保护，正常维护不能在 Eviction 被拒绝后自动降级为 Delete。",
        "DaemonSet Pod 通常在 Drain 时被忽略；StatefulSet 迁移可能受存储挂载和启动顺序影响；使用 emptyDir 或本地盘的 Pod 还要评估数据丢失风险。",
        "Drain 完成表示相应 Pod 的驱逐流程完成；替代 Pod 是否创建、何时可运行由上层控制器和调度器负责。它不保证训练已保存 Checkpoint、整组任务已恢复或业务重新达到 SLO。"
      ],
      "自动化安全边界": [
        "Condition=True 持续达到 duration 后才创建维修任务，过滤瞬时抖动。",
        "Condition=False 需要稳定窗口后才允许资源回池。",
        "限制同时维修节点数、每小时和每天次数、单节点冷却时间以及最大 Reset/Reboot 次数。",
        "Reset/Reboot 返回未知时先观察设备或主机实际状态，不能因 RPC 超时盲目重做。",
        "执行显存、计算、互联或功耗压力诊断前必须摘流并确认相关设备空闲。",
        "厂商故障矩阵按硬件代际、Driver、Runtime、Firmware、Fabric Manager、Device Plugin、虚拟化模式和互联拓扑版本化；未知输出格式和设备覆盖不完整时 fail-closed。"
      ],
      "GPU互联与CPU_NUMA的区别": [
        "NVLink/NVSwitch 描述 GPU 间的通信连接，CPU NUMA 描述 CPU、内存及设备 PCIe 上行的本地性；同一 NUMA 不保证 GPU 间存在 NVLink，不同 NUMA 也不意味着 GPU 间一定没有 NVLink/NVSwitch 通路。",
        "具有 NVSwitch 的八卡系统可以让八张 GPU 高速互通，同时其 PCIe 和 CPU NUMA 亲和分布在不同域；其他板型可能只有成对或部分互联，必须查询具体硬件。",
        "nvidia-smi topo -m 要分别读 GPU-GPU 连接、GPU-NIC 距离、CPU Affinity 与 NUMA Affinity。连接标识说明拓扑，实际带宽和应用表现还要结合 P2P 或集合通信验证。",
        "拓扑不佳是否允许启动取决于任务契约：硬约束无法满足时等待或拒绝，软偏好则允许在其他可行设备组合上运行并明确性能取舍。"
      ]
    },
    "典型故障与处置": {
      "检测链路失败": "CLI 不存在、超时、非零退出、输出截断、解析失败或设备覆盖不完整时保留旧状态，记录命令、耗时、stderr 和返回码；只有命令真实完成、覆盖完整且检查项明确正常才能恢复对应状态。",
      "Driver_Runtime异常": "Driver 模块、设备节点、管理服务或 Runtime 初始化异常时先判断是否整机受影响并阻止新任务进入；可按厂商能力重新初始化 Driver 或执行 VM/节点 Reboot，恢复后验证软件栈、设备枚举和基础计算。",
      "掉卡_PCIe_FallenOffBus": "预期 8 卡只能识别 7 卡、原 Bus-Id 消失或日志出现 PCIe link lost 时，先对齐 UUID/BDF、PCI Bus、厂商 CLI 和 Device Plugin；受控 Reboot 后仍缺卡就保持隔离并排查板卡、riser、插槽、主板、供电、Firmware 和 BMC 日志。",
      "ECC_RAS": "区分 Correctable/Uncorrectable、Contained/Uncontained、历史累计与新增趋势，以及 Page Offlining、Retirement、Row Remap Pending/Failure；偶发可纠正错误通常不直接重启，不可纠正或持续增长需要依据厂商建议选择应用重试、Reset、Reboot、离线内存诊断或硬件处理。",
      "XID_厂商错误码": "XID 可能来自应用、Driver、PCIe、显存、GPU Engine、互联、Firmware 或硬件，不能统一为任意 XID 就 Reboot；先映射成 DeviceLost、PCIeLinkLost、UncorrectableECC、DriverError、InterconnectError、ThermalCritical 等稳定语义，再由策略选动作。",
      "同机高速互联": "NVLink/NVSwitch、HCCS、MetaXLink 或 XPU Link 异常要确定单 Link、单 GPU、互联环、Reset Group、Fabric 或通信库故障；恢复要验证 Link Up、Speed/Width、P2P 和 Collective。",
      "温度功耗降频": "区分正常 Power Limit、风道制冷、风扇、供电、传感器和 GPU 本体问题；可告警、降载或摘除节点，恢复要看温度、时钟和 Throttle Reason 在压力状态下稳定。",
      "无显式错误的性能退化": "PCIe/互联带宽、时钟、算力或 Step Time 异常时先排除 workload、CPU 数据处理、存储、DataLoader、网络、通信库、Batch Size、软件版本和 NUMA 放置；摘流后再做带宽、显存、计算和功耗诊断，不能仅因利用率低直接 Reboot。",
      "RoCE_InfiniBand": "Collective 卡住、跨节点带宽下降或延迟增大时关联 NIC、交换机端口、路由、MTU、PFC/ECN、光模块、Job 和 NCCL/HCCL 日志；停止故障网络域接新任务并排查网络，不能用 GPU Reboot 代替。"
    },
    "厂商基础矩阵": {
      "NVIDIA": "主要信号和工具包括 Kernel Xid/SXid、NVML、nvidia-smi、DCGM Health/Diagnostics、Fabric Manager、NVSwitch 日志和 nvidia-bug-report。定位设备同时使用 UUID/GUID 与 PCI BDF；动作可能是无需处理、应用重启、GPU Reset、节点 Drain/Reboot 或硬件诊断。ECC 需结合 containment、Page Offlining、Row Remap 和 Recovery Action。",
      "昇腾": "主要使用 npu-smi、Device Health、Error Code/Information、HCCS、PCIe、RoCE、Ascend Device Plugin、MindCluster Fault Events 和 FaultDiag。Health 的 UNKNOWN 不能当作健康；HCCS Ring 会影响 Reset 范围，设备健康、上报资源、调度可用数量、任务重调度和 HCCS/RoCE 状态需要一致。",
      "昆仑芯": "主要检查 xpu_smi、Driver/XPU-RT 版本、Bus-Id、Volatile Uncorrectable ECC、温度、功耗、显存、利用率和 XPU Link。CLI 无输出可能是 Driver、设备节点、容器映射或命令环境问题；XPU Link 与单卡计算故障应分开建模，动作能力与版本绑定。",
      "沐曦": "主要使用 mx-smi 和 MX-DCM，分别检查 Overall/GPU Health、Error/Fatal、Hardware/Memory Exception、PCIe、MetaXLink、显存、算力、功耗和压力诊断。Overall Healthy 不代表每个诊断维度都通过；主动诊断前要确认节点空闲。",
      "海光": "主要使用 hy-smi、Driver 日志、设备 BDF、XID、Info、设备数量和 PCIe 状态。示例 `XID: 160, info: PCIe link lost` 应保留 BDF 和原始信息，映射为稳定的 PCIeLinkLost Condition；达到策略阈值后执行 Cordon、Drain、Reboot 和原 Condition 复检，不把海光 XID 数字套到其他厂商。"
    },
    "标准排障Runbook": [
      "保护现场：在 Reset/Reboot 前保存时间、Node、设备身份、Job/Pod/Rank、软件和固件版本、Kernel/厂商/通信日志及近期变更。",
      "确认清单：对比预期设备数、厂商 CLI、PCI Bus、Device Plugin、Capacity/Allocatable、MIG/vGPU 实例和占用 Pod。",
      "分类证据：判断属于 Detector、应用、Driver/Runtime、设备、PCIe、同机互联、跨节点网络还是机房环境。",
      "确定故障域：确认影响一个进程、一张卡、Reset Group、互联环、整节点还是网络域。",
      "阻止扩大：按范围标记 Device unhealthy、Cordon、Drain 或停止故障网络域调度；Drain 遵守 PDB。",
      "先低侵入后高侵入：先日志、Health、软件栈、枚举和 Link 查询，节点空闲后再做显存、带宽、P2P、计算、功耗和长稳测试。",
      "执行最小安全动作：任务重试、进程重启、设备 Reset、互联组 Reset、VM/节点 Reboot，最后才是隔离和硬件处理；结果未知先查状态。",
      "验证恢复：重新检查设备枚举和身份、Driver/Runtime、原 Condition、PCIe/高速互联、必要诊断和错误复发情况。",
      "重新发布：Device Plugin Healthy、Allocatable 与实际设备一致并保持稳定后再 UnCordon。",
      "复盘沉淀：记录根因、有效动作、复发、维修/RMA 结果，并更新 Condition、动作模板、阈值和厂商版本矩阵。"
    ],
    "高频通用问答": {
      "做GPU资源控制要懂硬件到什么程度": "要懂设备如何发现和分配、每类信号能证明什么、故障域有多大、应采取哪一级动作，以及哪些恢复证据允许资源重新回池。目标不是独立判断每个板级物理根因，而是基于可靠证据做安全处置。",
      "八卡只能看到七卡怎么查": "先比较厂商 CLI、PCI Bus、Device Plugin 和 Capacity/Allocatable，区分真实枚举少卡与仅被标记 unhealthy。若 PCI Bus 和厂商工具都少卡，再按 UUID/BDF 查 Kernel/Driver 日志；Reboot 后仍缺卡则保持隔离并排查 PCIe、riser、供电、主板和卡。",
      "怎么区分软故障和硬故障": "先判断证据层和影响范围，再看受控恢复后是否复发。应用 Context、Driver 状态和 Runtime 初始化更偏软件；掉卡、不可纠正 ECC 或 PCIe/互联错误在 Reset/Reboot 后持续存在，更偏硬件或物理链路。",
      "XID能直接说明硬件坏了吗": "不能。XID 是定位和动作决策入口，可能来自应用、Driver、PCIe、显存、Engine 或硬件；需要结合设备身份、持续性、遥测和诊断映射成稳定故障语义。",
      "什么时候任务重试_Reset_Reboot": "应用级且设备健康先重试；单设备异常且厂商、空闲状态和拓扑允许时 Reset；掉卡、PCIe link lost、Driver 整体异常或无法安全单卡 Reset 时，执行节点级 Cordon、Drain 和 Reboot。",
      "DevicePlugin标记unhealthy后会怎样": "kubelet 减少对应资源 Allocatable，新 Pod 不再分配该设备，但 Capacity 不一定变化，已有 Pod 不会自动迁移；后续任务处理和节点修复由训练平台和自愈系统负责。",
      "为什么Reboot成功不算恢复": "它只证明主机动作发生。还要验证设备完整枚举、Driver/Runtime、原始 Condition、关键链路、Device Plugin 和 Allocatable，并在稳定窗口后才回池。",
      "为什么不能无限Reboot": "故障多次重启仍复发时继续自动恢复的收益很低，还会造成任务反复失败和重启风暴；应设置次数、并发、小时/天额度和冷却时间，达到门槛后保持隔离并转深度诊断或维修。",
      "PCIe_NVLink_HCCS_RoCE分别在哪一层": "PCIe 是 Host 到加速卡的基础连接；NVLink、NVSwitch、HCCS、MetaXLink 属于节点内 Scale-Up 互联；RoCE/InfiniBand 属于跨节点 Scale-Out 网络。三类故障的影响范围和处置路径不同。",
      "不同厂商真正能统一什么": "统一的是故障控制契约：检测完整性、稳定 Condition、故障域、持续时间、动作策略和恢复复检；不能统一的是 CLI、错误码、互联拓扑、Reset 能力和诊断命令，这些保留在厂商适配层。",
      "什么时候进入RMA": "单个错误码不足以决定 RMA。受控 Reset/Reboot 后仍持续、反复掉卡、ECC/RAS 达到厂商退化门槛、离线诊断失败、链路错误持续或无法稳定通过压力测试时，保留身份、日志和动作记录后进入维修或厂商 RMA。",
      "Drain完成后训练是否恢复": "Drain 负责疏散节点上的相应 Pod，遵守 PDB 的常规路径通过 Eviction；替代 Pod 是否重建由上层控制器决定。训练还要满足自身的重试、Checkpoint 和整组恢复条件，PDB 不能证明这些条件已具备，节点回池也不能替代业务恢复验证。",
      "为什么需要异构GPU集群": "业务同时存在大模型训练、推理、数据处理和开发调试，不同任务对算力、显存、互联、成本和软件生态的要求不同；再叠加硬件供给和国产化需求，平台需要管理多种设备。异构集群的目标不是把设备差异抹掉，而是统一任务入口、资源模型、配额和运维，再把厂商差异留在调度约束、Driver、Runtime 和 Device Plugin 适配层。",
      "为什么放同一集群而不是每种GPU一个集群": "同一集群可以共享控制面、队列、配额、任务入口和监控，减少资源孤岛，但前提是 Driver、Runtime、Device Plugin 和节点镜像能够安全隔离，Scheduler 也能保证任务只进入兼容节点。若厂商软件栈冲突、升级节奏不同或需要隔离故障域，就应拆成多个集群，再由上层平台统一管理；这不是固定的一选一答案。",
      "Scheduler会选择具体哪张GPU吗": "要区分两条资源链路。传统 Device Plugin 扩展资源模式下，Scheduler 主要根据资源数量和约束选择 Node，具体设备 ID 通常由 kubelet Device Manager 在节点侧分配；GetPreferredAllocation 只是优选建议，不是最终保证。DRA 模式下，Scheduler 会根据 ResourceClaim 和 ResourceSlice 为 Claim 分配匹配设备并持久化结果，再选择能够访问这些设备的 Node。定制 GPU Scheduler 也可以提前选择设备，但必须说明设备计划写在哪里、如何预留、怎样交给 kubelet 落实以及失败后如何回收。",
      "GPU调度插件通常在哪些阶段工作": "先在 Filter 排除 GPU 型号、数量、显存、拓扑或共享份额不满足的节点，再在 Score 对剩余节点做 Binpack、Spread 或拓扑打分。需要维护扩展分配状态时，可在 Reserve/PreBind 提交占用，并在失败时通过 Unreserve 回滚；是否使用 Permit/Bind 取决于 Gang 和设备绑定方案，不能默认每个插件都实现全部扩展点。",
      "为什么分布式训练需要GangScheduling": "一个训练 Job 的多个 Rank 往往必须同时运行才能进入 Collective；只调度一部分 Pod 会占住 GPU，却无法正常训练，并可能与其他任务形成资源死锁。Gang Scheduling 在资源足以满足最小成员数时成组放行，不足时整体等待或回退，通常还要结合 Queue、Quota 和优先级。",
      "Binpack和Spread怎么选": "Binpack 尽量把任务压到较少节点，便于释放完整空闲节点、降低碎片，对同机高速互联友好；Spread 把任务分散开，有利于故障域隔离和负载均衡。大模型训练通常更重视拓扑连续性和整组资源，在线推理还要考虑副本故障隔离与 SLO，不能只按剩余 GPU 数量打分。",
      "GPU资源碎片是什么_怎么治理": "碎片不只是总卡数够但分散在多台机器，还包括型号、显存、共享份额、NUMA、NVLink/NVSwitch 和 GPU-NIC 亲和性不匹配。治理手段包括按资源形态分队列、拓扑感知 Binpack、整机优先、延迟小任务、任务重排或弹性伸缩；迁移和抢占是否可用取决于任务能否 Checkpoint 和恢复。",
      "怎么做GPU拓扑感知调度": "先确认实际机型及 GPU、CPU NUMA、PCIe 和 NIC 拓扑，再把任务需求拆成硬约束与软偏好。Filter 排除没有可行设备组合的节点，Score 比较通信路径和剩余碎片；硬约束不满足才等待，软偏好允许在其他可行组合上运行。节点选定后还要落实具体设备，GetPreferredAllocation 只是节点侧优选建议，不能保证 Scheduler 指定的 GPU ID。需要严格设备计划时，要使用有预留、执行确认和失败回收的定制协议，或通过 DRA 记录并交付分配；NUMA 亲和也需要节点侧资源管理配合。",
      "队列Quota优先级和抢占分别解决什么": "Queue 把任务放进可治理的等待域，Quota 限制或保障租户可用资源，优先级决定资源竞争顺序，抢占在高优任务无法调度时回收低优任务资源。GPU 任务抢占成本很高，必须考虑 Checkpoint、重启耗时和成组资源是否真的能够释放，不能只删除几个低优 Pod 就宣称抢占成功。",
      "运行中的GPU故障后Scheduler会自动迁移任务吗": "不会。设备变为 unhealthy 后主要影响后续 Allocatable 和新 Pod 分配，已绑定设备的训练进程仍可能失败或卡住。需要训练 Operator、Job Controller 或平台根据故障 Rank、Checkpoint 和替代容量决定整组任务重试，再由节点自愈隔离和修复故障节点。",
      "单集群和多集群GPU调度怎么分工": "多集群控制面先根据集群容量、GPU 型号、地域、队列、数据位置和故障域选择目标集群；进入目标集群后，再由本地 Scheduler 选择 Node，最后由 kubelet/Device Plugin 分配设备。Karmada 更偏多集群资源传播和 Cluster Selection，Volcano 更偏集群内批任务、Queue 和 Gang，两者解决的层级不同。",
      "数据和存储为什么也会影响GPU调度": "GPU 空闲不代表任务可以高效运行，模型权重、数据集和 Checkpoint 若跨 Region 或远端读取，可能让大量 GPU 等待 I/O。调度应把数据位置、存储带宽、缓存命中、跨域网络成本和预热时间一起考虑，否则资源利用率看似提高，实际 Step Time 或推理延迟反而变差。",
      "一个Pod的完整调度链怎么讲": "Pod 先进入 SchedulingQueue；调度周期先执行 PreFilter、Filter，有候选节点才进入 PreScore、Score/NormalizeScore，无可行节点则进入 PostFilter 并结束本次尝试。成功选出 Node 后先 AssumePod，再执行 Reserve 和 Permit；随后异步绑定周期执行 WaitOnPermit、PreBind、Bind 和 PostBind。Reserve、Permit、PreBind 或 Bind 失败时，要执行 Unreserve、ForgetPod 并按失败原因重新排队。",
      "为什么调度周期和绑定周期要分开": "选 Node 的调度周期依赖共享集群快照，串行化更容易维持一致决策；真正 Bind 可能等待 Permit、API 写入或外部系统，若也阻塞调度主循环会显著降低吞吐。因此选中后先 Assume，再让绑定周期异步执行；代价是必须设计完整的失败回滚和 assumed Pod 过期清理。",
      "SchedulerCache和InformerCache有什么区别": "Informer Cache 是通过 List/Watch 观察到的 API 对象缓存；Scheduler Cache 除已观察对象外，还会提前计入已选 Node 但尚未确认 Bind 的 assumed Pod，并基于它生成调度 Snapshot。前者解决读 API 的效率，后者还解决调度决策期间的乐观记账和资源超卖。",
      "AssumePod是不是已经绑定成功": "不是。AssumePod 只是 Scheduler 内存中先把 Pod 计入目标 Node，真正绑定还要经过 Reserve、Permit、PreBind 和 Bind。后续失败必须 ForgetPod；如果 Bind 成功，Informer 观察到正式 Pod 后再完成 assumed 到 confirmed 的衔接。",
      "CycleState是干什么的": "CycleState 保存一个 Pod 在当前调度周期各插件共享的临时结果，例如预计算约束、候选设备计划和打分输入，避免扩展点之间重复计算。它随本轮调度生命周期结束，不能作为跨重试、跨进程的资源真相；需要恢复的状态必须外置并可对账。",
      "Reserve和PreBind有什么区别": "Reserve 更早，在 Node 选定后为插件资源做临时保留，让后续 Permit 等待期间其他 Pod 不会抢走；PreBind 更晚，在 API Bind 前完成绑定必需的持久化或外部写入。插件可以只实现其中一部分，所以要看具体代码，不能背成所有 GPU 插件都在 Reserve 扣资源。",
      "为什么Unreserve必须幂等": "失败可能发生在 Reserve、Permit、PreBind 或 Bind 的不同位置，Framework 会按逆序遍历全部已启用 Reserve 插件，某个 Unreserve 甚至可能在自己的 Reserve 未执行时被调用。资源因此可能已释放、只写了一半或根本没写；Unreserve 要允许重复调用和空状态，按实际状态做最小补偿，不能再次释放导致负数或覆盖新分配。",
      "Reserve_Permit_PreBind_Bind失败分别怎么处理": "Reserve 失败后按逆序调用全部已启用 Reserve 插件的 Unreserve 并 ForgetPod；Permit 拒绝或超时、PreBind 失败、Bind 失败也都会执行 Unreserve 和 ForgetPod，再记录原因并重新排队。区别在于失败越靠后，外部状态越可能已部分写入，插件越需要补偿和独立对账。",
      "Permit怎么实现GangScheduling": "这是传统 Volcano 或自定义插件的一种做法：Permit 让单个 Pod 进入带超时的 Wait，同一 PodGroup 达到 minMember 且组级资源可行后再逐个 Allow。条件失败时要 Reject 仍在等待的成员，并让尚未 Bind 的 Pod 执行 Unreserve、ForgetPod 和重试。Allow 只是 Bind 前屏障；如果随后只绑定了部分成员，Unreserve 撤不回已经 Bound 的 Pod，还要靠持久化组状态、补偿终止或整组重排和独立对账恢复。Kubernetes v1.37 原生 Gang 使用 minCount 和 PodGroup scheduling cycle 做组级放置决策，不应直接套用这套 Permit 叙述。",
      "Scheduler重启后怎么避免GPU资源泄漏": "内存 Assume 和 CycleState 可由调度器重建，但 GPU 插件若已经写 Pod 标注、CR 或外部分配，就不能只依赖 Unreserve 回调。应使用 Pod UID 等幂等键，让写入可查询、释放可重复，并由启动扫描或独立 Controller 对齐 Pod、Node、设备分配和实际运行状态，清理孤儿占用。",
      "怎么测试一个GPU调度插件": "先做 Filter/Score 的资源、型号和拓扑边界，再覆盖同一 Pod 重试、多 Pod 并发争用、Reserve/Permit/PreBind/Bind 逐点故障注入、Unreserve 重复调用、Scheduler 重启、Informer 延迟和外部写入部分成功。最后验证 API 对象、Scheduler Cache、插件账本和真实设备占用最终一致，并观察排队时延、调度吞吐、碎片和拓扑命中，而不只看 Pod 是否 Running。",
      "activeQ_backoffQ_unschedulableQ分别是什么": "activeQ 放现在可以立即尝试的 Pod，Scheduler 只从这里 Pop；backoffQ 放刚失败且退避尚未结束的 Pod，避免失败热循环；unschedulableQ 放已证明在当前状态下不可调度的 Pod，等待相关集群事件激活。它们是三种重试状态，不是三个租户队列。",
      "调度失败后Pod为什么有时进backoffQ有时进unschedulableQ": "通常不可调度 Pod 进入 unschedulableQ 等事件；但若它被取出调度期间已经发生过一次可能让它成功的资源变化，事件扫描时它还不在 unschedulableQ，直接放回会丢失唤醒。此时先进入 backoffQ，退避结束自动重试，用调度周期号封住这个竞态窗口。",
      "什么事件会重新激活不可调度Pod": "要看失败插件注册了哪些 ClusterEvent，常见包括 Node 新增或资源/Label/Taint 更新、已绑定 Pod 删除或变化、PVC/PV/StorageClass 变化，以及 Pod 自身调度约束更新。事件只激活可能受益的 Pod，仍在退避期的先进入 backoffQ；另有超时兜底避免因漏事件永久等待。",
      "PrioritySort等于公平调度吗": "不等于。PrioritySort 解决当前 activeQ 中谁先尝试，通常是高 Priority 先、同优先级按时间；公平调度解决多个租户或队列长期各获得多少资源，需要配额、权重、最低保障、上限、借用和回收等机制。只有严格优先级可能让低优任务长期饥饿。",
      "DRF是什么_为什么适合多资源调度": "DRF 把每个租户在各类资源上的占用比例取最大值作为 dominant share，并优先给 dominant share 更小的租户。例如使用 20% CPU 和 60% GPU，主导份额就是 60%。它能避免只按 CPU 或 GPU 单维平均，但异构 GPU 仍需先明确型号资源池或容量折算，不能默认所有卡等价。",
      "异构GPU怎么计算公平份额": "先按型号、显存、互联或能力建立可替代资源池，再决定按卡数、标准化算力、显存份额还是成本权重计量；同时给稀缺型号设置队列保障和上限。没有可靠归一化时，宁可分池记账，也不要把不同卡强行折成一个看似精确的数字。",
      "怎么避免低优任务或大Gang一直饿死": "低优任务可采用等待时间老化、最低配额和最大等待保护；大 Gang 可使用整组资源预留、并发限制与安全 Backfill。借用空闲配额可以提高利用率，但要有明确回收条件、抢占预算和冷却时间，防止资源反复抖动。",
      "Scheduler抢占的完整流程是什么": "高优 Pod 无可行 Node 后进入 PostFilter；先排除抢占也无法解决的节点，再在候选节点副本上模拟移除低优 Pod并重新 Filter，得到可行受害者集合。随后选择代价较小的 Node、驱逐受害者并写 nominatedNodeName；等资源真正释放后，抢占者还要重新调度和 Bind。",
      "哪些不可调度问题不能靠抢占解决": "NodeSelector 或设备型号不匹配、Taint 不容忍、硬性亲和/反亲和、存储拓扑冲突、网络或许可证条件缺失、集群物理总量不足，以及节点上没有更低优先级受害者时，删除现有 Pod 也不会形成可行解。抢占只适合释放资源能够改变结果的约束。",
      "PDB能绝对阻止Scheduler抢占吗": "不能把它说成绝对保证。默认抢占会优先选择较少违反 PDB 的候选和受害者，但当只有违反 PDB 的方案可行时仍可能选择它；而且 PDB 主要约束自愿中断，不覆盖所有故障和删除场景。回答时应说尽量保护，不说绝不会违反。",
      "nominatedNodeName代表抢占成功了吗": "不代表。它表示 Scheduler 为该 Pod 提名了下一轮优先尝试的 Node，并让其他调度决策考虑这份预期资源；受害者可能仍在优雅退出，节点状态可能变化，重新 Filter、Reserve、Permit、PreBind 或 Bind 仍可能失败。",
      "现场让写GPU调度插件该怎么答": "先澄清资源模型和硬约束，再用 PreFilter 解析 Pod 需求并放入 CycleState，Filter 判断型号、数量、显存和拓扑是否满足，Score 对候选 Node 做 0 到 100 的 Binpack、Spread 或拓扑打分。只有确实需要临时占用扩展资源才实现 Reserve/Unreserve，需要在 Bind 前写设备信息才用 PreBind；最后补缺字段、并发争用、重复回滚和阶段失败测试。",
      "Gang达到minMember后部分PodBind失败怎么办_Permit等于整组原子吗": "先确认集群版本和实现：传统 Volcano 或自定义 Permit 是逐 Pod 的 Reserve、Wait 和 Allow，形成的只是 Bind 前屏障，不能说成多 Pod 事务；Kubernetes v1.37 原生 PodGroup 则用 minCount 和组调度周期做原子放置决策，不能混讲。以下补偿针对传统 Permit 路径：按 PodGroup UID、Pod UID 和 attempt ID 持久化 Planning、Reserved、Admitted、Abort 等状态，使重启后可以对账。Bind 前失败时整组 Reject 并执行 Unreserve、ForgetPod；已有成员 Bound 后，Unreserve 无法撤销，只能由 Job 或 PodGroup 相关 Controller 补偿删除或重建，并用训练启动屏障避免 Rank 未齐就开训。Bind 响应丢失时先读取 Pod UID、nodeName 和 Reservation 决定接管还是释放，不能直接重试。至少监控组准入 P95/P99、partial-bind 组数、达到 minMember Ready 的成功率、回滚时长和孤儿 Reservation 最大年龄。",
      "多维资源索引怎么设计_怎么证明没有漏掉可行节点": "我会给 Node 分配稳定内部 ID，把 GPU 型号、Label、Taint 和拓扑域组织成倒排 Bitmap，把 CPU、内存等连续资源组织成容量桶或有序树，再从区分度最高的硬约束开始求交。NodeAffinity 可以主要依赖 Node 属性索引，但 PodAffinity、PodAntiAffinity 和 TopologySpread 还依赖现有 Pod 与拓扑域计数，必须随 Pod 增删动态更新。lower_bound 只能定位单维容量桶，完整多维匹配并不是 O(log N)，索引得到的小候选集仍要基于同一 Scheduling Snapshot 和 assumed、reserved 状态执行完整 Filter。若 List/Watch 延迟、乱序更新、删除 tombstone 或设备健康突变导致索引版本过旧，就回退标准 Filter，并通过版本校验和周期全量重建修复。验证时把索引结果与全量扫描做差分测试，要求 false-negative 为零，同时观察候选缩减比、索引更新延迟、回退率、调度 P95/P99、内存和 GC 开销。",
      "Gang和普通Pod并发抢同一GPU_最终确认的线性化点在哪里": "单个 kube-scheduler 的 Scheduling Cycle 默认串行，Binding Cycle 可以并发，普通 Node 资源先由 Scheduler Cache 的 AssumePod 记账，不能先假设标准路径存在多个并发选点线程。多个 Scheduler、多个进程或具体 GPU 设备账本仍要明确唯一权威，采用单写 Allocator 或基于 resourceVersion、generation 的 Reservation CAS，并用 Pod UID、PodGroup UID 和 Device ID 标识占用。跨多个 Node 的 Gang 没有 Kubernetes 多对象事务，可以按固定顺序获取每个设备或节点的 Hold，竞争者把 Hold 视为不可用，全部成功后再把组状态切成 Committed，任一步失败则按状态幂等释放。CAS 成功后在 Bind 前崩溃、Bind 响应丢失或设备健康突变时，应先读取 Pod、Reservation 和设备实际状态，再决定接管、释放或重排。至少监控 CAS 冲突与重试率、重复设备分配数、孤儿 Hold 数及最大年龄、提交与回滚时延和重启收敛时间。",
      "Gang并发1_普通任务并发5_怎么主动避让又不浪费资源": "先在队列准入层判断大 Gang 是否进入资源预留窗口；尚未形成可行整组时，只按稀缺 GPU 型号、整机和拓扑形状对关键节点做 Score 降权，避免过早硬锁资源。整组方案可行并创建有效 Reservation 后，普通任务才不得破坏这批关键容量，并优先 Binpack 到其他碎片节点。尚未兑现的保留量可以借给非 Gang 做 Backfill，但只接纳能在大任务启动期限前结束，或能够安全 Checkpoint、抢占和恢复的任务。Reservation 要有 TTL 和取消条件，并结合等待老化、Queue Quota、借用回收、抢占预算与冷却时间；固定保留某个 GPU 百分比不能当通用答案。验证时同时观察 Gang 启动 P95/P99、非 Gang slowdown、reserved-idle GPU-hours、Backfill 超期率、饥饿与抢占抖动次数和总体利用率。",
      "GPU碎片率怎么定义_如何证明Binpack确实有效": "GPU 碎片不能只用 freeGPU/totalGPU 表示，而应定义为总量看似足够，但因节点、型号、显存、CPU/NUMA、NVLink 或 GPU-NIC 形状无法组成当前队列需求的不可用容量。至少统计完整多卡节点数、largest feasible Gang、按任务 Shape 计算的 stranded GPU，以及按排队需求加权的 schedulability。放置时比较调度前后的 fragmentation delta，让小任务优先填已有碎片，同时保护稀缺型号、整机和连续拓扑域；Backfill 必须服从大任务的预约截止时间。若 GPU-only Binpack 造成 CPU、内存或 NIC 瓶颈、热节点和故障域集中，或迁移丢失的训练进度大于缩短的等待时间，就应降低装箱权重或放弃 Defrag。验证不能只看利用率，还要通过历史负载回放或对照实验比较因碎片 Pending 的任务数、完整节点数、最大可行 Gang、任务排队时间、SLO，以及迁移损失的 GPU-seconds 和 Checkpoint 成本。",
      "Node选好后具体GPU_CPU_NUMA_NIC由谁落位_怎么保证计划没有被改掉": "要先区分设备分配链路：传统扩展资源加 Device Plugin 模式下，Scheduler 主要选择 Node，具体 GPU ID 由 kubelet Device Manager 决定，GetPreferredAllocation 只是可选偏好，不能把拓扑 Score 说成设备落位保证。严格 CPU 绑核还需要 Guaranteed Pod 的整数 CPU 请求、CPU Manager static 策略以及合适的 Topology Manager restricted 或 single-numa-node 策略。若控制面必须选择并持久化具体 GPU 或 NIC，可以使用 DRA，由 ResourceSlice 暴露设备属性和容量、ResourceClaim 表达约束并记录 Allocation，再由 kubelet 和节点 Driver Prepare、通过 CDI 等方式交付；也可以明确采用厂商自定义 Scheduler、Reservation CR 或 Annotation 与 Device Plugin 的闭环。拓扑信息过期、设备在 Schedule 到 Prepare 之间变为 unhealthy、CPU 与 GPU 没有共同 NUMA hint 或实际设备与计划不一致时，严格任务应失败释放并重新排队，不能静默降级。验证时对齐计划 ID、ResourceClaim 或 Annotation、容器实际 CDI 和设备节点、cpuset，并观察同 NUMA、同 NVLink 域和 GPU-NIC 亲和命中率、降级率、Prepare 失败率、调度额外延迟及 NCCL 或 P2P 带宽。",
      "八卡GPU全互联是否等于同一个CPU_NUMA": "不等于。NVSwitch 可以让八张 GPU 高速互通，同时它们到 CPU 的 PCIe 上行和 NUMA 亲和仍分布在不同域。先确认机型，再分别查看 nvidia-smi topo -m 的 GPU 间连接与 CPU/NUMA Affinity；卡间通信、CPU 供数和 GPU-NIC 路径要分别判断。",
      "虚拟机GPU直通场景怎么做拓扑感知调度": "先根据宿主机的物理 GPU 互联、PCIe、NUMA 和 NIC 位置选择宿主及具体设备，再由虚拟化层落实 GPU/NIC 直通、vCPU 绑核和内存 NUMA 绑定。Guest 看到的虚拟 NUMA 不能直接证明宿主物理亲和正确，需要核对两层映射。多 GPU 的 P2P 或 NVSwitch 能力还取决于直通方式、Fabric 分区和虚拟化配置，最后在 Guest 内验证实际设备、通信可达性和带宽。",
      "多卡任务凑够卡但拓扑不好_一定要Pending吗": "先看拓扑是硬约束还是偏好。如果任务必须使用某个互联域或最低通信能力，没有满足条件的设备集合就继续等待；如果只是希望更快，可以在满足硬约束的组合中选择得分更高的方案，必要时接受性能退让。不能只看卡数，也不能不问任务要求就一律 Pending。",
      "GPU_Pod从创建到运行相比普通Pod多了什么": "主要多了 GPU 资源发布与约束、具体设备分配和运行时设备注入。传统 Device Plugin 路径中，插件通过 ListAndWatch 向 kubelet 报告设备，kubelet 更新节点扩展资源，Pod 用 nvidia.com/gpu 等资源名声明需求；型号和拓扑还需要标签、亲和性或调度扩展支持。Scheduler 选 Node 后，kubelet Device Manager 选择健康设备 ID 并调用 Allocate，再由运行时根据返回的设备、挂载、环境变量或 CDI 信息交付给容器。只请求一张 GPU 不会让默认调度器自动理解卡间拓扑。",
      "Gang整组可行方案怎么搜_举一个贪心失败后回溯的例子": "我会先基于同一份资源快照生成候选节点，把已有占用、assumed Pod 和有效 Reservation 都扣掉，再优先放候选最少的 Pod，候选数相同时先放需求大的。搜索过程中只扣减临时容量，后续成员放不下就撤销试放并换位置。例如一个 minMember=3、三个成员都必须启动的任务，N1 剩 6 核、24 GiB、3 卡，N2 剩 8 核、32 GiB、4 卡；P1 要 6 核、24 GiB、3 卡，P2/P3 各要 4 核、16 GiB、2 卡，其他硬约束均满足。如果因为软偏好先放 P1→N2，再放 P2→N1，两个节点都只剩 2 核、8 GiB、1 卡，P3 就放不下；撤销 P2 和 P1 后改成 P1→N1、P2/P3→N2，整组恰好可行。这说明逐个 Pod 贪心失败不代表整组无解。实际搜索要设时间或次数预算，耗尽只表示本轮没找到方案；找到完整方案后，再通过权威资源账本校验和预约，遇到并发冲突就释放本轮占用并重新规划。传统 Permit 只是绑定前屏障，已有成员 Bind 成功后的失败还需要 Controller 补偿，不能把搜索成功说成整组原子绑定。",
      "八卡申请两卡怎么剪枝和打分_举一个CPU_NIC与碎片联合选择的例子": "先按健康、占用、型号等条件缩小设备集合，再过滤任务要求的互联、CPU NUMA 和 NIC 条件，最后给可行组合打分，兼顾通信代价与后续任务能否利用剩余资源。举一个假设拓扑：八卡中 3 号已占用，可用的高速互联组合来自 0/1/2、4/5、6/7 三组，组间不满足本任务的互联要求；三组分别靠近不同 CPU NUMA 域，可用 CPU 和 RDMA VF 分别为 8 核加 1 个 VF、4 核加 1 个 VF、2 核且无 VF。当前任务要求两卡、4 核和 1 个 VF 位于同一 CPU NUMA 域。八选二原本有 28 种，排除占用卡后剩 21 种，按互联过滤后剩 0/1、0/2、1/2、4/5、6/7 五种，再排除 CPU 和 VF 不足的 6/7，得到四种可行组合。在通信条件相当时，我会选 4/5，保留 0/1/2 及其 CPU、VF 给后续需要三卡、6 核和 1 个 VF 的任务；如果拆掉三卡组，总空闲卡数即使够，也可能凑不出后续任务需要的形状。这个例子解释的是碎片代价，实际权重还要结合任务需求和通信性能。真实八卡机必须查询拓扑，不能默认按上述方式分组，GPU 互联和 CPU NUMA 也要分别判断。最终选定的 GPU、CPU 和 NIC 还要由设备分配协议及节点侧资源管理落实；GetPreferredAllocation 只是偏好，严格设备选择需要 DRA 或自定义分配闭环，落位失败时应释放并重排。"
    },
    "通用知识红线": [
      "不要说 GPU 故障绝大部分通过 Reboot 都能修复；只能说部分 Driver、设备状态机和枚举异常可能通过 Reset/Reboot 恢复。",
      "不要仅凭 XID、ECC 非零、CLI 报错或 GPU 利用率低判断硬件损坏。",
      "不要说 Device Plugin 标记 unhealthy 后会自动迁移已有 Pod。",
      "不要使用逻辑 GPU index 作为跨重启稳定设备身份。",
      "Condition 消失、Unknown、CLI 空输出或 Detector 重启都不能作为恢复证据。",
      "不要在 workload 占用设备时直接 Reset 或运行高强度主动诊断。",
      "单设备、互联域、节点和网络故障不能统一使用一个动作模板。",
      "厂商支持某种 Reset 是通用能力信息，不自动代表当前项目实现了该动作。",
      "Reboot 请求成功、主机完成重启和 GPU 恢复是三层证据。",
      "节点 UnCordon 不代表原训练任务已经恢复。",
      "不要把统一异构资源池说成固定单集群；同集群还是多集群取决于软件栈兼容、故障域、升级和运维边界。",
      "不要一概而论说原生 Scheduler 不选择具体设备：传统 Device Plugin 扩展资源路径通常只选 Node，再由 kubelet Device Manager 分配设备；DRA 路径则由 Scheduler 为 ResourceClaim 分配匹配设备并持久化结果。",
      "不要把 Volcano 说成多集群调度控制面，也不要把 Karmada 说成节点内 GPU 分配器。",
      "GPU 调度、拓扑打分和共享隔离可以作为岗位通用知识回答，但当前没有证据时不能说成本人已在生产实现。",
      "不要把 Scheduler Cache 的 AssumePod 说成 API Bind 已成功；它只是绑定前的乐观记账，失败后还要 ForgetPod。",
      "不要背成所有 GPU 插件都在 Reserve 阶段扣减资源；Filter、Score、Reserve 和 PreBind 的职责要根据通用契约和具体方案说明。",
      "不要把传统 Permit 放行或 Reserve/Unreserve 说成 Gang 的组级原子事务；这些扩展点按 Pod 执行，统一 Allow 后仍可能部分 Bind，Unreserve 也不能撤销已经 Bound 的成员。Kubernetes v1.37 原生 PodGroup Gang 是另一套组调度周期语义。",
      "不要把单维容量索引的 lower_bound 和 O(log N) 说成完整多维调度复杂度；候选返回、索引求交、精确 Filter、Gang 匹配和提交校验都有额外成本。",
      "不要把 Node 选择或拓扑打分成功说成具体 GPU、CPU NUMA 和 NIC 已经落位；还要说明设备分配协议、kubelet 执行和实际结果验证。",
      "不要承诺所有失败都会自动回滚或进程崩溃后零泄漏；外部写入需要幂等、过期清理和独立对账。",
      "不要把 activeQ、backoffQ 和 unschedulableQ 说成三个租户优先级队列；它们表示立即尝试、限速重试和事件等待三种状态。",
      "不要把 PriorityClass 或 PrioritySort 等同于多租户公平性；公平性还需要配额、权重、保障、上限、借用和回收。",
      "不要说高优 Pod 一定能通过抢占调度成功；不可解约束、无低优受害者、资源释放延迟和后续绑定失败都会使抢占无效。",
      "不要说 PDB 能绝对阻止 Scheduler 抢占，也不要说 nominatedNodeName 等于已经预留或绑定成功。",
      "调度队列、公平性、抢占和插件编码作为通用知识回答；当前简历没有 GPU Scheduler 实现经历，不能把设计题或源码理解讲成本人项目。",
      "不要把 GPU 的 NVLink/NVSwitch 全互联等同于同一个 CPU NUMA，也不要把八卡节点默认说成全互联或两个四卡互联域。",
      "不要把拓扑偏好全部当硬约束；只有必须满足的条件不可行时才等待或拒绝，较差的通信路径是否可接受由任务契约决定。",
      "不要把 Guest NUMA 当作宿主物理拓扑，也不要把 VM GPU 拓扑通用方案归入本人 KubeVM 磁盘 I/O 改造经历。"
    ]
  },

  "project_facts": [
    {
      "company": "腾讯",
      "role": "云原生 Infra 实习生",
      "period": "2026-04 ~ 至今",
      "focus": "TCS/TKE 云原生基础设施：异构 GPU 节点自愈控制面、统一公网出口代理组件、KubeVM/QEMU 磁盘 I/O 排查与 Native AIO 自动选择、Nydus RuntimeClass Admission Webhook 适配、本地存储感知调度优化",
      "terminology_translation": {
        "TCS节点": "腾讯内部云平台管理的 Kubernetes 工作负载节点。",
        "unatgw": "平台已有的公网 NAT 出口能力。",
        "COS-CGI": "对象存储相关的公网访问服务。",
        "OriginalHost": "为了兼容已有业务的域名访问方式，让客户端继续使用原公网域名；ClusterIP 入口通过 service-id 导向 Service，LoadBalancer 入口通过受管 DNS A 记录导向 VIP。",
        "NPDPlus": "部署在节点侧的故障检测组件，负责把厂商工具和驱动日志中的异常统一上报为节点状态、事件和监控指标。",
        "NodeCondition": "Kubernetes 节点状态条件；在本项目中用于持续表达某类 GPU 故障是否存在，True 表示故障仍在，明确为 False 才能作为恢复证据。",
        "NodeHealer": "节点自愈策略对象；描述适用节点、故障场景、优先级、限流规则和动作模板引用，本身不直接执行维修动作。",
        "MachineHealingTemplate": "机器自愈动作模板；定义经过审核的节点隔离、工作负载驱逐、重启和解除隔离步骤。",
        "HealingTask": "节点维修任务对象；持久化记录一次自愈的目标节点、触发故障、执行步骤和当前阶段，便于控制器重启后继续执行。",
        "Cordon": "隔离节点，阻止普通工作负载继续调度到故障节点；本项目通过自愈专用 NoSchedule taint 实现。",
        "Drain": "通过 Eviction 驱逐节点上允许中断的 Pod，并遵守 PDB；替代 Pod 的创建和调度由上层控制器与 Scheduler 负责，Drain 本身不搬迁进程或训练状态。",
        "UnCordon": "在获得明确恢复证据后解除节点隔离，重新允许工作负载调度。",
        "KubeVM": "基于 KubeVirt 管理的虚拟机产品形态；Kubernetes 负责声明和调度 VMI，virt-launcher 内的 QEMU 执行实际虚拟机磁盘 I/O。",
        "NativeAIO": "QEMU 的宿主机 I/O 后端模式 io=native，使用 Linux Native AIO；它只决定 QEMU 如何提交宿主 I/O，不等同于 cache=none、iothread、磁盘扇区大小或持久化保证。",
        "cache=none": "QEMU 磁盘缓存模式，通常通过 O_DIRECT 绕过宿主机页缓存；Native AIO 的自动选择以最终 cache=none 为必要条件，但二者是独立配置。",
        "RuntimeClass": "Kubernetes 用于为 Pod 选择容器运行时处理器的资源；Pod 通过 spec.runtimeClassName 引用，Admission Webhook 不能替代节点侧 RuntimeClass 和运行时配置。",
        "NydusRuntimeHandlerAnnotation": "io.containerd.cri.runtime-handler=nydus；它是 containerd 1.7 runtime-level snapshotter 模式下给 PullImage 补充运行时上下文的实验性桥接信号，不是 RuntimeClass 的替代品，也不是 Nydus 已生效的证明。"
      },
      "project_background": {
        "one_sentence": "两项主要生产交付是异构 GPU 节点自愈和统一公网出口代理；此外，我还排查了 KubeVM 磁盘 I/O 异常，并在 KubeVirt 功能分支扩展 QEMU Native AIO 的自动选择、参数传递和回滚链路，同时在 tcs-extensions 补充 Nydus RuntimeClass 到 containerd runtime-handler 的 Admission Webhook 适配。调度方面，我还在既有本地存储插件上补充跨节点容量评分、分散与装箱策略及多卷配置处理。"
      },
      "platform_interview": {
        "高压追问首答": {
          "你们支撑的是什么业务_什么平台": "我们支撑的是腾讯云 TCS 私有云里的 Kubernetes 基础设施平台，最终服务企业客户，上层承载各类云产品和业务系统。我所在的方向偏底层 Infra，主要把节点、网络和 GPU 等基础设施能力接入 Kubernetes 管控体系。我负责的具体项目是异构 GPU 节点自愈和受控公网出口，分别属于节点管理和网络控制面。",
          "TCS对Kubernetes做了哪些产品化封装": "底层仍然是 Kubernetes，但对用户提供的是产品化的集群与节点管理、工作负载、网络、存储、异构 GPU、监控和运维能力。用户通过页面或 API 配置需求，平台再把它转换成 Kubernetes 资源和控制器行为。例如 GPU 自愈，用户只需要配置哪些节点启用自愈、处理哪些故障以及限流策略，不需要自己编写 Controller，后端会生成相应策略并自动执行修复。",
          "用户能直接拿到Kubernetes_API或kubeconfig吗": "TCS 整体的账号、kubeconfig 和权限交付不是我直接负责的范围。通常用户可以通过平台 API，或者在授权范围内使用 Kubernetes API，但具体开放程度由平台 RBAC 和租户权限决定。我负责的公网代理组件里可以确认的是：用户只能在授权 Namespace 声明策略，Controller 的 ServiceAccount 才有权限维护相应的 Service、Endpoints 等受管资源。",
          "介绍一下你做的两个组件": "第一个是异构 GPU 节点自愈。节点侧把 NVIDIA、昇腾、昆仑芯等不同厂商的 GPU 故障统一上报为 NodeCondition；控制器根据 NodeHealer 策略创建 HealingTask，执行 Cordon、Drain 和必要的重启，最后重新检查原始故障状态，确认恢复后再把节点放回集群。第二个是统一公网出口代理。它让业务通过 EgressForwardRule 声明允许访问的固定公网目标；Controller 创建 Service、管理代理 VM 后端，并自动发布 Nginx 配置，把原来依赖人工 SSH 维护多台代理 VM 的过程自动化。",
          "你还做过KubeVM存储相关工作吗": "做过一次 KubeVM 数据盘 I/O 异常排查。我先对比 KubeVM 与 CVM 的 launcher Domain XML，拆开 cache、I/O backend、iothread 和 blocksize 等变量，再在 KubeVirt 中扩展 Native AIO 自动选择：最终为 cache=none、磁盘源为 file/device、且用户未显式指定 I/O 模式时选择 io=native，并通过集群级负向开关保留旧行为。代码和测试在我的远程功能分支，但目前没有合入主干或发布分支。",
          "Nydus的RuntimeClass适配是你做的吗": "是，这段适配逻辑是我补充的。它解决的是 containerd 1.7 runtime-level snapshotter 的接口时序差：RuntimeClass handler 标准上在 RunPodSandbox 时传递，而 PullImageRequest 没有独立运行时信息。Pod 在 CREATE/UPDATE 时若已设置 runtimeClassName=nydus，Webhook 会幂等注入或纠正 io.containerd.cri.runtime-handler=nydus，让 containerd 能在拉镜像阶段选择该 runtime 绑定的 Nydus snapshotter。我的工作边界是准入桥接、开关接线和边界测试，不创建 RuntimeClass，也不部署 snapshotter、nydusd、Dragonfly 或节点 containerd 配置，更不能单凭注入证明 Lazyload。",
          "更偏控制层还是调度层": "两类都有，主要经历仍在 Kubernetes 控制面，包括 GPU 节点自愈和公网代理配置发布；调度方面，我在已有本地存储插件中实现了跨节点容量评分，支持分散与装箱、多卷需求和按 StorageClass 配置。前两项改变节点可用状态或网络入口，本地存储这项则直接参与 Scheduler 的节点选择；成组训练调度和 GPU 拓扑联合优化没有算作这次实现。"
        }
      },
      "supplementary_personal_implementations": {
        "Nydus_RuntimeClass_Admission_Webhook": {
          "模块定性": "这不是 Nydus 镜像加速器本身，而是 Kubernetes RuntimeClass 意图到 containerd 1.7 PullImage snapshotter 选择之间的准入桥接器：业务已经显式选择 runtimeClassName=nydus 时，Webhook 再补齐 io.containerd.cri.runtime-handler=nydus。真正的数据面仍由 Nydus 格式镜像、nydus-snapshotter、nydusd、containerd 节点配置以及可选的 Dragonfly P2P 链路完成。",
          "项目实现状态": "该 Webhook 适配代码已进入 tcs-extensions 的 master 和 release/tcs2.3.5.1；是否部署到具体集群以及节点侧 Nydus 是否实际生效，需要按运行态链路另行确认。",
          "镜像加速背景": {
            "默认OCI路径的问题": "普通 OCI 镜像冷启动通常要先解析 manifest、下载所需 layer 并完成解压展开，镜像越大、节点弹性扩容越集中，Pod 越容易把时间耗在镜像准备阶段；大量节点同时从 Registry 回源，还会放大源站带宽和并发压力。",
            "Nydus解决什么": "Nydus 把 RAFS 文件系统元数据与数据 blob 分离，先取得启动所需的 bootstrap/元数据，再按文件访问以 chunk 粒度取数据，并可配合预取和本地缓存。因此它解决的是不必等待完整镜像下载与展开即可准备 rootfs；收益应以实际冷启动和读取链路验证，不能只因设置了 RuntimeClass 就宣称加速。",
            "Dragonfly解决什么": "Dragonfly 解决的是大规模分发与回源压力：dfdaemon 作为节点侧 Peer/代理，Scheduler 为下载任务选择 Peer 或 Seed Peer，数据按 piece 在节点间复用，未命中时再回源。Nydus 与 Dragonfly 可以组合成按需取 chunk 加 P2P 分发，但二者是不同层次，Nydus Webhook 本身不包含 Dragonfly 逻辑。",
            "chunk与piece边界": "Nydus chunk 是 RAFS 中文件数据的寻址和校验单位，Dragonfly piece 是 P2P 传输与缓存调度单位。一次 Nydus Range 读取可映射到一个或多个 Dragonfly piece，二者不是固定一一对应，不能在面试中混称。",
            "Lazyload代价": "Lazyload 是把一部分网络与解压 I/O 从启动前移到容器运行期首读：它能缩短启动关键路径，但未缓存文件第一次访问可能出现延迟抖动，后端不可用时也可能在运行期暴露读取失败；如果业务最终读取几乎整个镜像，总下载量收益会收窄。prefetch 是启动速度、首读稳定性和提前流量之间的权衡。",
            "为什么选择性启用": "通过 runtimeClassName=nydus 显式 opt-in，可以让普通 Pod 继续使用默认 runc/overlayfs，把兼容性和故障影响面限制在选择 Nydus 的工作负载；前提是 RuntimeClass 的调度约束或节点全覆盖能保证 Pod 落到具备 Nydus 能力的节点。"
          },
          "四个概念不要混淆": {
            "RuntimeClass": "Kubernetes 集群级资源。Pod 用 spec.runtimeClassName 引用它，kubelet 查到其中的 handler，并在 RunPodSandboxRequest 中把 runtime handler 传给 CRI；它主要表达运行时选择，不会自动向 Pod 注入任意 containerd 私有 annotation。",
            "runtime_handler_annotation": "Pod annotation io.containerd.cri.runtime-handler=nydus 会随 Pod annotations 进入 CRI PodSandboxConfig。containerd 1.7 的 PullImage 请求本身没有独立 runtime 信息时，会读取这个实验性 annotation，再按 runtime 配置覆盖默认 snapshotter。",
            "runtime_handler不等于更换OCI执行器": "runtime handler 选择的是 containerd 中一套命名 CRI 配置；默认 runtime 和 nydus runtime 都可以继续使用 io.containerd.runc.v2，Nydus 路径的关键差异是该 handler 绑定了不同 snapshotter，而不是一定换成另一种容器执行器。",
            "Nydus数据面": "containerd 的 runtimes.nydus 需要绑定 nydus snapshotter，proxy_plugins.nydus 需要指向可用 socket，同时启用 snapshot annotations，并避免过早丢弃 unpack 所需内容；nydus-snapshotter 再启动或管理 nydusd，为转换后的 Nydus 镜像提供 RAFS/FUSE 或相应后端的按需读取。",
            "Dragonfly分发面": "可选的 dfdaemon/Seed Peer/Scheduler/Manager 链路负责 piece 级 P2P 分发和回源治理，不负责 Kubernetes RuntimeClass 选择，也不能替代 Nydus snapshotter。"
          },
          "为什么仅有RuntimeClass还不够": {
            "时序差异": "RuntimeClass handler 的标准传递点是 RunPodSandbox；镜像拉取通常发生在创建容器之前。目标 containerd 1.7 实现明确说明 PullImageRequest 没有携带运行时信息，所以 runtime-level snapshotter 选择需要从 PodSandboxConfig annotation 取 runtime handler。",
            "缺少桥接的后果": "Pod 的 Sandbox 后续可能按 nydus handler 创建，但镜像拉取和 unpack 仍可能使用默认 snapshotter，结果可能是退回普通完整拉取、选错镜像准备路径，或因下游配置组合而失败；不能把结果一律说成某一种固定报错。",
            "Webhook价值": "在 Pod 持久化并进入 kubelet 前，将业务已经表达的 runtimeClassName=nydus 转换成 containerd 1.7 能在 PullImage 阶段消费的信号，避免每个业务 YAML 手工维护同一 annotation，并用确定性收敛保证二者一致。"
          },
          "端到端数据与控制链": [
            "镜像构建侧：将 OCI 镜像转换或构建为可被 Nydus 识别的镜像产物，并校验 bootstrap、manifest 和 blob 引用；Webhook 不做镜像转换。",
            "集群准备侧：创建 RuntimeClass nydus，并在目标节点注册同名 containerd runtime handler、nydus snapshotter proxy plugin 和可用 socket；按目标版本设置 disable_snapshot_annotations=false，通常还要保留 discard_unpacked_layers=false；若并非所有节点覆盖，还要用 RuntimeClass.scheduling 或等价节点约束防止错调度。",
            "业务声明侧：Pod 显式设置 spec.runtimeClassName=nydus；当前模块不会替普通 Pod 自动选择 Nydus。",
            "准入侧：API Server 将 Pod CREATE/UPDATE 交给 tcs-admission-webhook；Nydus 子 Handler 判断开关、operation 和 runtimeClassName，随后新增或纠正 io.containerd.cri.runtime-handler=nydus。",
            "Patch侧：通用 Admission 框架对修改前后的 typed Pod 序列化并生成 JSONPatch，API Server 应用后保存最终 Pod。",
            "kubelet与CRI侧：kubelet 将 Pod annotations 复制到 PodSandboxConfig，同时由 RuntimeClass 得到 RunPodSandbox 的 handler；拉镜像时把 sandbox config 交给 CRI。",
            "containerd_1_7侧：containerd 从 PodSandboxConfig 读取 runtime-handler annotation，解析 runtimes.nydus，并为这次 PullImage 选择该 runtime 绑定的 nydus snapshotter，而普通 Pod 仍走默认 snapshotter。",
            "运行侧：nydus-snapshotter/nydusd 挂载并按访问读取镜像数据；若配置 Dragonfly backend，chunk/piece 可优先从本地缓存、Peer 或 Seed Peer 获得，未命中再按配置回源。"
          ],
          "代码调用链": {
            "配置默认值": "NewDefaultConfiguration 将 EnablePatchNydusRuntimeHandler 设为 true；PodMutatingConfig 持有该字段。",
            "启动参数": "--pod-mutating-enable-patch-nydus-runtime-handler 绑定同一布尔值，ApplyTo 写入运行配置；Helm values 默认 true，Deployment 模板显式传参。",
            "注册时机": "NewHandler 在进程启动时复制配置并调用 getPodHandleFuncs 构造一次处理器列表；开关开启时把 Nydus Handler 放在共享 Pod handler 链最后。开关改变需要重启或滚动 Webhook，不是运行时热更新。",
            "准入入口": "正式 .tad 交付模板将 core/v1 Pods 的 CREATE/UPDATE 指向 /admission/mutating/core/v1/pods；通用框架解码当前对象，UPDATE 还解码 oldObject，深拷贝 original 后串行执行所有 Pod handlers。",
            "核心函数": "handlePatchNydusRuntimeHandler 是纯内存收敛逻辑，不查 API、不访问 Informer、没有外部 I/O；它只看最终 Pod，oldPod 未参与判断。",
            "响应生成": "所有 handler 成功后，通用响应层比较 original/current 生成 JSONPatch 并返回 Allowed=true；任一前置 handler 报错会提前终止，Nydus Handler 不再执行。"
          },
          "分支决策表": [
            {
              "条件": "功能开关关闭",
              "结果": "启动时不注册 Nydus Handler；函数内部仍保留一次冗余开关检查。"
            },
            {
              "条件": "非 CREATE/UPDATE",
              "结果": "正式 .tad rules 正常不会转发；若绕过规则直接进入通用 HTTP Handler，外层 operation 校验会拒绝，而核心函数自身也直接跳过。"
            },
            {
              "条件": "runtimeClassName 为空或不等于 nydus",
              "结果": "完全不修改；即使非 Nydus Pod 手工带了该 annotation，也不会清除或拒绝。"
            },
            {
              "条件": "runtimeClassName=nydus 且 annotations=nil",
              "结果": "初始化 map，再写入 io.containerd.cri.runtime-handler=nydus。"
            },
            {
              "条件": "runtimeClassName=nydus，其他 annotations 已存在但目标 key 不存在",
              "结果": "保留其他键，只新增目标键。"
            },
            {
              "条件": "目标 annotation 已经等于 nydus",
              "结果": "立即返回；Nydus 子模块不产生新 diff，属于目标状态幂等。共享 Webhook 的其他 Handler 仍可能产生 Patch。"
            },
            {
              "条件": "目标 annotation 为空、为 runc 或其他错误值",
              "结果": "强制覆盖为 nydus，不拒绝 Pod；这是以 RuntimeClass 为权威的纠正策略，没有单 Pod opt-out。"
            },
            {
              "条件": "UPDATE 中 annotation 被删除或改错，最终 runtimeClassName 仍为 nydus",
              "结果": "重新补齐；逻辑不比较 old/new diff，只对最终对象收敛。"
            },
            {
              "条件": "Pod 带 infra.tce.io/tcs-admission-webhook-exclude 标签",
              "结果": "MutatingWebhookConfiguration 的 objectSelector 让它绕过整个 Pod Webhook，因此不会注入 Nydus annotation，也会跳过其他 Pod mutations。"
            }
          ],
          "幂等与Patch精确口径": {
            "幂等含义": "正确值不重写，缺失或错误值收敛到唯一目标值；它保证的是 Nydus 子模块的目标状态幂等，不能声称整个共享 Webhook 响应一定为空 Patch。",
            "理论Patch": "annotations 为空时通常增加 annotations 对象；map 已有而 key 缺失时增加目标 key；值错误时替换 value；值正确时 Nydus 部分无 diff。具体 JSON Pointer/op 由通用 jsonpatch 库根据序列化结果生成。",
            "当前测试边界": "现有单测只检查函数执行后内存 Pod 中 annotation 的最终值，没有断言 AdmissionReview 到 AdmissionResponse 的最终 JSONPatch opcode、path 或空 Patch。"
          },
          "失败语义与影响面": {
            "Nydus函数本身": "没有下游调用，所有当前分支都返回 nil，因此小函数自身是低失败面的确定性内存变换。",
            "共享框架失败": "当前 Pod 或 UPDATE oldObject 解码失败、前置 Pod Handler 报错、对象序列化或 JSONPatch 生成失败都会返回 Allowed=false；Nydus Handler 位于链尾，前序错误时它不会执行。",
            "failurePolicy": "部署模板对 Pod Webhook 配置 failurePolicy=Fail。Webhook 服务不可达、TLS/超时或框架报错时，所有未被 objectSelector 排除的 Pod CREATE/UPDATE 都可能被 API Server 阻断，blast radius 不只 Nydus Pod。",
            "下游能力缺失": "RuntimeClass 不存在或 handler 未知、节点没有配置 runtimes.nydus、snapshotter socket 不可用、镜像没有 Nydus 产物、Registry/鉴权/P2P backend 异常，Webhook 都不会预检；准入可能成功，失败会在调度、PullImage、RunPodSandbox、CreateContainer 或运行时读取阶段暴露。",
            "运行期首读失败": "Lazyload Pod 可能已经启动，但尚未缓存的文件第一次被访问时仍依赖 backend；Registry、dfdaemon、Peer/Seed Peer 或网络异常会造成首读抖动或失败，是否能回源取决于实际 fallback 配置。",
            "启动回环": "nydus-snapshotter、dfdaemon 等加速基础组件自身不能只依赖尚未就绪的 Nydus/P2P 路径；否则节点重启或首次安装时可能形成组件起不来、加速链也无法就绪的 bootstrap 回环。",
            "关闭开关": "只影响 Webhook 重启后的新准入请求，不会清理已存在 Pod 的 annotation，也不会回滚节点侧 containerd/Nydus 配置。"
          },
          "隐式契约与版本边界": {
            "名称硬编码": "实现假设 RuntimeClass 资源名、RuntimeClass.handler、containerd runtimes 键和 annotation 值都叫 nydus。代码不会读取 RuntimeClass 对象的 handler；若资源名为 nydus 而 handler 实际为 nydus-handler，该实现会写错，因此部署清单必须保持同名契约。",
            "只做单向补齐": "它以 runtimeClassName=nydus 为触发条件，却不清理非 Nydus Pod 上手工填写的 runtime-handler annotation，也不在理论上的 RuntimeClass 移除时清理旧 annotation；这是窄范围补齐器，不是双向一致性校验器。",
            "Webhook顺序假设": "部署清单未配置 reinvocationPolicy。若另一个更晚执行的 Webhook 才设置 runtimeClassName=nydus，本 Webhook 默认不会因此重新执行，可能漏注入；可靠前提是 runtimeClassName 在进入本 Webhook 前已经存在。",
            "containerd版本": "上游 Nydus 文档把 runtime-level snapshotter 模式限定为 containerd >=1.7，并要求 sandbox spec 带 runtime-handler annotation；本模块源码也明确写 containerd 1.7。它不能被泛化成所有 containerd 版本都必须使用的方案。",
            "与1_6设计稿的冲突": "containerd 官方发布说明把 CRI Runtime Specific Snapshotter 的首次发布明确列为 v1.7，上游 Nydus 文档也要求 >=1.7；因此 stock containerd 1.6.9 不能直接支撑这段 Webhook 的选择逻辑。本地总体设计稿把目标基线写成 1.6.9，同时示例保留 1.7 annotation，属于必须闭环的版本冲突；只有内部 fork 明确 backport 且节点 config dump、源码和运行态实验均能证明时，才可作为例外。",
            "交付模板差异": "正式 .tad/applications/tcs-admission-webhook/templates/mutating.yaml 为 Pod 注册 CREATE+UPDATE；仓库 legacy charts/tcs-admission-webhook/templates/cert.yaml 的 Pod 规则只注册 CREATE。源码函数支持 UPDATE，不等于所有历史交付形态都会收到 UPDATE，回答时必须先确认现场使用哪套 Chart。",
            "不要混淆两种选择": "runtime-handler annotation 解决的是一次 PullImage 该用哪个 runtime-specific snapshotter；Nydus variant 选择解决的是同一 image index 中选 OCI 还是 Nydus manifest。二者相关但不是同一个机制，镜像转换/manifest matcher 问题不能由 Webhook 单独解决。",
            "演进边界": "更新的 CRI RuntimeClass-aware image pull 正在把 runtime handler 变成显式字段；在采用该能力的新版本栈里，这类实验性 annotation 桥接可能不再需要，所以回答必须绑定目标 containerd/CRI 版本。"
          },
          "怎么确认真正生效": {
            "第一层_准入意图": "检查最终 Pod 同时具有 spec.runtimeClassName=nydus 和 io.containerd.cri.runtime-handler=nydus，确认准入桥接已经完成。",
            "第二层_配置前置": "检查 RuntimeClass nydus 的 handler、调度约束，节点 containerd config dump 中 runtimes.nydus 到 snapshotter 的绑定、proxy_plugins.nydus socket，以及 nydus-snapshotter/nydusd 健康状态。",
            "第三层_实际数据面": "从目标容器反查 containerd 实际 Snapshotter，应看到 nydus，并核对 nydus snapshot 列表；再检查 fuse.nydusd 或目标实现对应的真实挂载、nydusd/snapshotter 日志与读取指标。若宣称 Dragonfly P2P，还要另外证明 dfdaemon、Scheduler/Peer/Seed Peer 命中与回源路径。只有 Pod annotation 不能证明 Lazyload。",
            "第四层_加速效果": "在相同节点、相同镜像、冷缓存/热缓存明确的条件下比较镜像准备和 Pod 启动关键时间，并同时观察实际回源字节、P2P/本地缓存命中、首读延迟和运行期 I/O；固定百分比收益必须来自这类受控实验。"
          },
          "个人职责_本人确认": "本人完成该 RuntimeClass/Admission Webhook 适配，因此个人经历可以写补充或实现这段逻辑。",
          "20秒首答": "大镜像冷启动要等待下载和展开，Nydus 用按需加载缩短镜像准备路径，而 Dragonfly 可进一步用 P2P 降低集中回源。我的增量不是实现整套加速栈，而是在 tcs-admission-webhook 中补齐 containerd 1.7 适配：当 Pod 已选择 runtimeClassName=nydus 时，幂等注入或纠正 io.containerd.cri.runtime-handler=nydus，让 PullImage 能选择该 runtime 绑定的 Nydus snapshotter。",
          "90秒展开": "背景上，Nydus 把镜像文件系统元数据和数据 chunk 分离，启动先拿必要元数据，文件真正被访问时再取数据；Dragonfly 则把这些数据按 piece 在 Peer 间复用，减少 Registry 回源。控制面上，RuntimeClass 的 handler 标准上是在 RunPodSandbox 时传给 CRI，但目标 containerd 1.7 的 PullImage 请求没有独立 runtime 信息，runtime-level snapshotter 选择需要从 PodSandboxConfig 的 io.containerd.cri.runtime-handler annotation 取值。我实现了受开关控制的 Pod Mutating Handler，源码支持 CREATE/UPDATE，且仅在 runtimeClassName 精确为 nydus 时处理：annotations 为空就初始化，缺失就新增，错误就纠正，正确就 no-op，再由通用框架生成 JSONPatch。开关从代码默认值、CLI 到 Helm values 和 Deployment 完整接通。边界上，它不创建 RuntimeClass、不装 snapshotter/nydusd、不改 containerd、不转换镜像，也不验证实际 Lazyload；真正验收要继续看 containerd 实际 Snapshotter、FUSE/nydusd 和冷启动及回源指标。",
          "高压追问": {
            "为什么RuntimeClass不够": "RuntimeClass handler 的标准传递点是 RunPodSandbox，而目标 containerd 1.7 的 PullImageRequest 没有独立运行时信息。containerd 因此从 PodSandboxConfig 的实验性 annotation 取 runtime handler，再选择该 runtime 绑定的 snapshotter；Webhook 补的是这段时序和接口信息差。",
            "为什么不直接让业务写annotation": "业务已经通过 RuntimeClass 表达一次运行时意图，再要求手填 containerd 私有键会形成两个可漂移的配置源。Webhook 以 RuntimeClass 为权威统一补齐和纠正，降低接入成本；代价是硬编码同名契约和共享 Webhook 的影响面。",
            "错值为什么覆盖而不是拒绝": "当前策略把 runtimeClassName=nydus 视为权威意图，错误 annotation 会导致拉镜像 snapshotter 与 Sandbox runtime 不一致，所以直接收敛到 nydus。它没有设计单 Pod opt-out；如需允许自定义 handler，应把映射配置化而不是继续硬编码。",
            "怎么证明真的加速": "第一步只证明 Pod 被注入；第二步确认 RuntimeClass、containerd runtime 和 snapshotter socket；第三步从容器反查 Snapshotter 并看到真实 Nydus 挂载/读取；最后在同镜像同节点的受控冷热缓存实验里比较启动时间和回源字节。只看 annotation 不能证明。",
            "Webhook挂了会怎样": "部署是 failurePolicy=Fail，服务不可达、TLS/超时或通用框架错误会阻断所有未排除 Pod 的 CREATE/UPDATE，不只 Nydus Pod。Nydus 小函数本身没有 I/O、总是返回 nil，但它复用共享 Pod webhook，必须按集群级关键依赖做高可用和监控。",
            "为什么UPDATE也处理": "UPDATE 与 CREATE 共用最终状态收敛逻辑；只要最终 Pod 仍是 nydus，annotation 被删除或改错就会补回。它不使用 oldPod，也不是事件 diff 逻辑。实际 runtimeClassName 通常在调度后不可改，UPDATE 只保持对象元数据一致性，不会把已经运行的容器重新拉镜像或重新挂载到 Nydus；而且 legacy Chart 没有为 Pod 注册 UPDATE。",
            "1_6和1_7怎么回答": "这段 Webhook 源码和上游 runtime-level snapshotter 文档都绑定 containerd 1.7。另一份总体设计稿的目标版本写 1.6.9，这是需要现场确认的版本矛盾；在没有节点版本、config dump 和运行态证据前，我不会说同一逻辑已在 1.6.9 生效。",
            "模块还能怎么增强": "把 RuntimeClass 名到 handler 的映射配置化或动态解析，并根据 Webhook 顺序选择 reinvocation；如果节点覆盖不完整，再通过 RuntimeClass.scheduling 绑定 Nydus 能力标签，避免 Pod 落到不支持该运行时的节点。"
          },
          "绝对红线": [
            "不要把 Webhook 注入成功说成 Nydus 已安装、镜像已转换、Lazyload 已发生或性能已经提升。",
            "不要说 Webhook 自动给普通 Pod 选择 RuntimeClass；它只处理已经显式设置 runtimeClassName=nydus 的 Pod。",
            "不要把 RuntimeClass handler、containerd runtime-handler annotation、snapshotter、Nydus manifest variant 和 Dragonfly P2P 当成同一个概念。",
            "不要把 containerd 1.7 的 annotation 方案无条件套到 1.6.9 或更新 CRI；必须绑定实际版本验证。",
            "不要说幂等时整个共享 Webhook 一定返回空 Patch；只能说 Nydus 子模块不贡献 diff。",
            "不要说功能开关能热更新或回滚已有 Pod；Handler 列表启动时构造，变更需滚动 Webhook，且不清理存量 annotation。"
          ],
          "面试安全说法": "我在 tcs-admission-webhook 中实现了面向 containerd 1.7 runtime-level snapshotter 的 Nydus RuntimeClass 适配：对已指定 runtimeClassName=nydus 的 Pod，源码及正式 TAD 交付支持在 CREATE/UPDATE 准入时幂等注入或纠正 io.containerd.cri.runtime-handler=nydus，并接通默认值、CLI、Helm 开关及分支测试；legacy Chart 只注册 CREATE。它只负责 Kubernetes 运行时意图到 CRI PullImage snapshotter 选择的桥接；RuntimeClass、节点 containerd、nydus-snapshotter/nydusd、镜像转换、Dragonfly 分发和实际性能验收均属于下游前置或独立链路。"
        }
      },
      "resume_bullets": [
        {
          "bullet_title": "设计并落地异构 GPU 节点自愈页面化控制面",
          "项目背景": {
            "业务问题": "随着 GPU 集群规模扩大，节点故障处理逐渐成为运维瓶颈。NVIDIA、海光、昇腾、昆仑芯、沐曦等不同厂商的健康检测方式和故障表现不同，人工登录节点定位和恢复效率低，也难以保证不同故障场景采用一致的处理标准；同时，单节点故障还可能影响训练任务运行。",
            "建设目标": "面向私有云中的异构 GPU 节点，把已经确认可自动处理的故障接入统一的检测、策略准入、隔离维修和恢复确认流程。直接目标是减少标准故障处置对人工的依赖，并在健康证据充分后自动恢复可用 GPU 容量；是否允许中断节点上的业务，要由对应业务与平台的恢复能力决定。"
          },
          "业务视角": {
            "直接用户与最终受益者": "直接使用者是 TCS 私有云的平台运维和 GPU 集群管理员，他们按厂商配置节点范围、故障场景、限流和通知。最终受益的是使用 GPU 资源的业务团队：故障处理可以按统一流程推进，维修确认后可用容量自动回池。节点上的业务是否可安全中断，必须结合具体训练或推理任务判断，不能仅凭 GPU 集群或 Kubernetes 部署方式推断其有无状态。",
            "典型业务场景": "典型场景是 GPU 节点持续出现设备丢失、PCIe 链路或驱动异常，并且该故障已有审核过的维修策略。管理员先确认对应业务能够承受中断及恢复，再将节点和故障场景纳入自动维修范围；运行时通过持续时间、维修额度及 Eviction/PDB 控制动作，执行隔离、驱逐、必要的重启和原故障复检。硬件故障未消失或 Drain 受阻时停止并转人工。训练任务还要与上层的任务重试、Checkpoint 和整组恢复机制配合，当前节点控制器不自行判断训练状态是否已安全保存。",
            "为什么需要平台化": "平台化的价值是把检测结果、处理策略、限流、执行步骤和恢复标准统一起来。运维配置的是故障场景与安全边界，Controller 按同一套规则执行；破坏性动作仍由受控模板和容量限制约束，避免把自动化变成无条件批量重启。",
            "一次业务使用流程": "管理员在页面选择节点范围、故障类型、持续时间、限流和通知策略。故障满足策略后，系统生成维修任务，隔离节点并通过 Eviction 驱逐允许中断的 Pod，再执行必要的重启和健康复检；替代 Pod 是否重建由上层工作负载控制器负责。处理过程写入历史并发送通知，证据不足或动作失败时由运维介入。",
            "业务价值": "核心收益是把新增厂商和原先未接通的平台环节纳入既有自愈框架，扩大可自动处置的范围，并减少这些场景对人工串联操作的依赖。厂商适配让昇腾、昆仑芯、沐曦与已有 NVIDIA、海光场景进入统一控制面；平台配置和任务记录减少跨厂商维护及故障跟踪成本；限流、重启防重和原始故障复检控制重复维修与过早回池风险。资源侧的直接价值是让修复后的 GPU 容量自动回到可用池，收益大小再用实际不可用时长和人工介入次数评价。",
            "业务成功标准": "一次自愈成功不能只看重启命令返回，而要看到原始 GPU 故障明确恢复、节点解除隔离并重新具备承载工作负载的条件，同时维修任务和历史记录完整。训练或推理任务是否恢复到业务 SLO 仍由其工作负载控制器和业务监控确认，不能由 HealingTask 状态替代。",
            "量化口径": "问接入范围：在已有 NVIDIA、海光检测基础上补齐昇腾、昆仑芯、沐曦适配，形成五类厂商场景的统一配置和处置链路。问处置耗时、恢复成功率或节省人力比例：当前没有统一的前后对照统计，直接说明对应数据缺失。问怎样衡量效果：比较同类故障的人工介入次数、首次明确故障到恢复调度资格的耗时、不可用 GPU 容量随时间的累计损失，以及复发和误触发情况。",
            "实际业务范围": "项目面向 TCS 私有云的异构 GPU 节点，负责故障隔离、受控维修和健康确认后回池。训练任务有模型参数、优化器和训练进度等状态，即使通过 Job 重建 Pod、通过 Checkpoint 恢复，也仍然是有状态的任务；训练平台中的 API 或 Controller 实例可以无状态，这是另一个层次。训练恢复由上层训练系统负责；推理服务则还要考虑副本容量、预热和请求状态，不能笼统承诺节点重启后业务无损。",
            "规模口径": "已确认的是私有云部署、实际接触规模相对有限；当前资料没有可确认的最大节点数。面试只使用本人确认过的环境规模，5000 节点属于扩容设计题，不能写成已部署规模或已完成的压测结果。"
          },
          "一句话": "我把异构 GPU 节点中已确认可处理的故障接入受控自动维修，减少逐节点人工操作，并在原故障明确恢复后让 GPU 容量重新回池。",
          "简历推荐写法": "负责异构 GPU 节点自愈平台化闭环并完成生产交付：复用 NVIDIA/海光检测与 node-healing 已有动作执行能力，补齐昇腾、昆仑芯、沐曦检测及统一 NodeCondition 契约，完善 NodeHealer 限流、HealingTask 可恢复执行与恢复复检；打通 tcs-platform 四个 YunAPI、页面配置与历史查询、tcs-extensions 阶段通知并纳入 TCS Addon/TAD 交付。",
          "20秒首答": "这是面向 TCS 私有云的异构 GPU 节点自愈能力。我在已有框架上补齐厂商适配和平台控制链，把故障确认、受控维修和恢复回池串成统一流程。核心收益是减少人工串联操作，让修复后的 GPU 容量自动恢复可用，同时控制重复重启和故障未清就回池的风险；上层任务的状态恢复由对应业务系统负责。",
          "项目状态": "已完成从五厂商异构检测、页面配置、任务执行、恢复判定、历史通知到 Addon/TAD 的端到端实现，并由我完成组件实现、交付物接线和最终上线交付。",
          "证据与口径分层": {
            "正式实现与交付": "五厂商检测适配、统一 Condition 契约、NodeHealer/HealingTask 控制链、页面 YunAPI、历史通知和 Addon/TAD 已正式实现并完成最终生产交付。",
            "亲手验证": "可以讲厂商资料与输出样本、配置、自动化测试、日志回放和联调验收中，原始信号如何映射、任务如何推进以及恢复如何判定。"
          },
          "个人贡献边界": {
            "复用部分": "NPDPlus 检测框架、NVIDIA/海光既有检测能力，以及 node-healing 已有的 NodeHealer、MachineHealingTemplate、HealingTask、limiter 和 Cordon/Drain/Reboot 动作执行能力。",
            "本人增量": "异构厂商检测与 Condition 契约、页面配置模型和四个 YunAPI、多 Healer selector/仲裁与单节点单活、恢复判定和失败隔离处理、TCS SSH+boot_id 重启防重、HealingTask 阶段通知、Addon/TAD 跨组件接线。",
            "前端边界": "项目闭环包含配置页面，前端同学负责页面实现；我负责后端 YunAPI、配置模型、NodeHealer 期望状态、节点启用关系和历史/通知契约。"
          },
          "角色与状态所有权": {
            "检测层": "NPDPlus 部署在节点侧，消费不同厂商的 SDK、管理工具和驱动日志，统一输出 NodeCondition（Kubernetes 节点状态条件）、Event 和 Metric，让上层控制面不直接依赖具体厂商工具；它只报告故障事实，不决定是否重启。",
            "策略与派单层": "NodeHealer（节点自愈策略）表达节点范围、故障场景、优先级、限流和恢复模板引用，把运维处理规则从代码中抽离成声明式配置；控制器消费 Condition，完成多 Healer 仲裁并创建 HealingTask。",
            "动作模板层": "MachineHealingTemplate（机器自愈动作模板）定义经过审核的 Cordon（节点隔离）、Drain（工作负载安全驱逐）、Reboot（节点重启）、UnCordon（解除隔离）等恢复动作，页面或普通用户不能随意组合破坏性操作。",
            "执行层": "HealingTask（节点维修任务）记录一次具体故障处理的节点、故障、动作和执行阶段。节点恢复可能持续较长时间，因此任务状态会持久化，Controller 重启后仍能从原来的进度继续执行隔离、Drain、可选重启和安全解封。",
            "平台配置层": "配置页面与 YunAPI 管理模板入口、候选节点、场景开关、限流、通知和历史，只决定期望配置，不直接判断硬件健康。",
            "历史与通知": "历史页面和通知组件读取 HealingTask 状态，但不参与健康判定；即使消息服务出现故障，也不会阻塞主自愈流程。"
          },
          "正常链路_90秒": {
            "链路讲述原则": "问完整自愈流程时，按检测、决策、执行和恢复说明；问其中一步就只讲该步。页面配置、交付和历史通知按对应问题选用。",
            "一句话主线": "NPDPlus 把异构 GPU 故障统一成 NodeCondition → Node Controller 完成多 Healer 匹配、单任务仲裁和安全限流 → HealingTask Controller 隔离、迁移并修复节点 → 原始 Condition 明确为 False 且稳定后才解封。",
            "首答_四阶段": [
              "1. 检测事实：NPDPlus 只负责把当前硬件故障持续维护为 NodeCondition=True，Event 和 Metric 用于解释与观测。",
              "2. 决策派单：Node Controller 检查 selector、场景开关和 duration，从多个 `(NodeHealer, Condition)` 候选中稳定选一个，再通过 limiter 与 NDB 创建 HealingTask。",
              "3. 执行修复：HealingTask Controller 先 Cordon、再遵守 PDB 做 Drain，随后按模板执行可选 Reboot；TCS 路径用持久化旧 boot_id 防止结果未知时重复重启。",
              "4. 恢复闭环：重启完成不等于 GPU 恢复，只有触发任务的原始 Condition 存在、明确为 False 且稳定至少一分钟，才 UnCordon 并完成任务；证据不足就保持隔离。"
            ],
            "完整链路_8阶段": [
              "1. 故障检测：节点侧 NPDPlus 调用厂商管理工具并读取驱动日志，把需要持续表达的故障写成 NodeCondition=True；某个可恢复 Condition 只有在对应检查真实执行、设备覆盖完整且未再观察到故障时，才允许改为 False。",
              "2. 策略匹配：Node Controller 遍历 selector 命中的全部 NodeHealer；只有 Condition 类型匹配、场景已启用且基于 LastTransitionTime 计算的持续时间达到 duration，才形成候选。",
              "3. 仲裁与互斥：先检查该节点是否已经有还没有结束的 HealingTask；如果有，就继续原来的任务，不再创建新任务。只有该节点不存在尚未结束的任务时，才按 priority 选择最高优先级，再按 HealerName、ConditionType 稳定排序，每轮只选一个候选。",
              "4. 执行前检查：只有最终选中的策略进入所属 Healer 的 limiter，检查并发、小时、天、节点冷却和 NDB；其他策略不占额度。最高优先级策略未通过检查时等待重试，不改为执行第二名。",
              "5. 记录本次维修内容：Controller 从 MachineHealingTemplate 展开受控步骤并创建 HealingTask，把 nodeName、来源 Healer、触发 Condition、动作和参数写进任务；真正执行前，再检查同一节点是否还有其他没有结束的任务。",
              "6. 隔离与迁移：先添加自愈专用 NoSchedule taint，再按 Kubernetes Eviction 语义 Drain 并遵守 PDB；Pod 的重建和调度由工作负载控制器与 Scheduler 完成。",
              "7. 修复与动作确认：按模板执行可选 Reboot。CVM 路径观察实例操作；TCS 路径先持久化旧 boot_id，再通过受控 SSH 下发 reboot，SSH 断开后观察 boot_id 变化而不盲目重发。",
              "8. 恢复与解封：Node Controller 检查 HealingTask 中记录的原始 Condition，要求它仍然存在、明确为 False 且稳定至少一分钟，随后写 PhaseRecovered；执行器完成 UnCordon 后进入 PhaseCompleted，其他仍为 True 的故障再参加下一轮仲裁。"
            ],
            "失败后怎么处理": "任何已经进入隔离阶段却没有获得明确恢复证据的任务，默认保持节点隔离并转人工；Controller 重启后会读取还没有结束的 HealingTask、节点和云实例的当前状态以及已经占用的 limiter 额度，从原来的进度继续处理，不会从第一步重新执行所有动作。"
          },
          "失败与补偿闭环": {
            "配置保存部分失败": "ApplyNodeHealer 会先保存 NodeHealer，再核对多台节点的启用关系，跨对象没有事务。任何一步失败都向调用方返回错误，不把部分成功包装成成功；调用方用同一份完整节点选择集重试时，接口会重新比较哪些关系需要增加或移除，直到实际配置与用户提交的配置一致。它依赖同一请求可以安全重试，不依赖回滚，也不宣称原子提交。",
            "检测证据不足": "管理工具缺失、字段解析失败、只检查到部分 GPU 或检测器未运行都不能当作健康；只有本轮检查真实完成并覆盖目标设备，才允许把可恢复 Condition 改为 False。",
            "未达持续时间": "继续等待剩余 duration，避免一次采样抖动直接触发 Drain 或重启。",
            "限流或NDB拒绝": "最高优先级候选延迟一分钟后重试，不绕过它执行低优先级候选；NDB 无法确认剩余容量时停止继续维修，避免检测系统异常反而触发批量操作。",
            "重复任务": "Node Controller 创建任务前会查询同一节点是否已有还没有结束的任务，HealingTask Controller 执行动作前还会再检查一次；已有任务就继续原来的流程或等待它结束，不启动第二条维修流程。前一次检查避免正常派单产生重复任务，后一次检查挡住绕过 Node Controller 创建的任务并发操作节点；List→Create 提供前后两层保护，严格原子唯一还需要唯一锁对象或 resourceVersion CAS。",
            "创建HealingTask失败": "释放刚占用的并发额度，小时/天窗口保留已发生的维修尝试，避免靠重复失败绕过累计限额。",
            "TCS重启结果未知": "SSH 在 reboot 后断开可能表示命令已经执行，不能立即重发。系统先持久化旧 boot_id，再观察新值；只有 boot_id 变化才确认重启。",
            "Controller重启": "Controller 重启后，以还没有结束的 HealingTask 记录为恢复依据，找回原始 Condition、这个任务已经占用的并发额度，并根据任务时间恢复仍处于小时、天和节点冷却窗口内的限流记录；先从原来的进度继续已有任务，再处理新的候选。",
            "动作或恢复超时": "节点仍无健康证据时默认保持隔离；如果节点随后恢复且模板明确允许 UnCordon，系统才自动解除隔离，否则转人工处理。",
            "通知失败": "通知独立于主状态机做有限重试和去重；可能重复也可能最终漏发，但不会回滚或阻塞自愈主链。"
          },
          "高压追问首答": {
            "这个能力直接给谁用_最终服务谁": "直接给 TCS 私有云的平台运维和 GPU 集群管理员使用，他们配置节点范围、允许自动处理的故障、维修额度和通知。最终帮助 GPU 业务团队减少人工处置环节，并在维修确认后重新获得可用容量；如果节点承载训练任务，中断与恢复还必须和上层训练系统配合。",
            "为什么有告警还不够_一定要做自愈": "告警之后，故障确认、节点隔离、工作负载驱逐、修复和恢复复查仍需要有人逐步执行，标准故障也会卡在人工接手和操作衔接上。自愈把已有明确策略、业务能够承受中断的场景自动串起来，减少这部分人工依赖，并用统一恢复标准决定何时回池。硬件需要更换、故障性质不明或业务无法迁移时，告警和人工处置仍然是正确路径，不应强行自动重启。",
            "这个项目的业务价值怎么衡量_有没有数据": "核心收益是让已纳管的可恢复 GPU 故障进入受控自动闭环，减少逐节点人工操作，并在健康确认后归还可用容量。具体完成的变化是：基于已有 NVIDIA、海光检测补齐昇腾、昆仑芯、沐曦适配，把这些厂商的配置、维修任务、恢复判定和历史通知接到统一控制面；符合策略的任务自动推进，异常或证据不足时停在明确阶段。效果主要看同类故障的人工介入次数、恢复调度资格的耗时，以及不可用 GPU 容量的累计损失。厂商覆盖和流程能力是已完成的结果，统一 MTTR 改善和人力节省百分比目前没有前后对照统计。",
            "自动自愈会不会反而扩大训练任务损失": "会。训练任务有模型参数、优化器和训练进度等状态，驱逐一个仍能部分运行的节点可能让整组任务失败，并丢失最近一次 Checkpoint 之后的进度。故障持续时间和维修额度只控制节点动作，PDB 只约束 Eviction 引起的允许中断数量，都不能证明训练已经保存状态。启用自动 Drain/Reboot 前，业务方与训练平台必须明确任务重试、Checkpoint 和整组恢复条件；当前节点自愈不提供这些训练协作能力，不能以节点修复成功替代训练恢复。",
            "你个人到底做了什么": "我复用了 NPDPlus、NVIDIA/海光检测和 node-healing 已有的 CRD 与动作执行能力；我的增量是昇腾/昆仑芯/沐曦适配与统一 Condition、多 Healer 仲裁、恢复判定与失败处理、TCS boot_id 重启防重、四个 YunAPI、通知以及 Addon/TAD 接线。配置页面由前端同学实现，我负责后端和配置契约。",
            "你负责整个healing_engine还是平台编排控制面": "不是从零实现整个 healing engine。既有 node-healing 已经提供 NodeHealer、MachineHealingTemplate、HealingTask、limiter，以及 Cordon、Drain、Reboot、UnCordon 的基本执行框架；我负责的是异构 GPU 场景的平台化增量，包括厂商检测接入与统一 NodeCondition 契约、多 Healer 处理和恢复分支增强、TCS SSH 重启与 boot_id 防重、页面所需的 YunAPI、历史通知以及 Addon/TAD 交付。回答时要把复用的执行底座和本人完成的增量分开，不把整个框架说成从零自研。",
            "请讲一次完整的GPU自愈过程": "我讲一个交付联调里覆盖过的海光 DCU PCIe 链路丢失案例。输入是一条内核日志样本：`hycu0000:3b:00.0 ... XID: 160 ... info:PCIe link lost`。检测器先把它规范化为 `DCU-Xid 160, Device 0000:3b:00.0, Info PCIe link lost`，再由规则映射为 `HygonDCUPCIeLinkLost=True`，Reason 是 `DCUPCIeLinkLostDetected`，Message 保留 PCIe link lost 的故障语义。Node Controller 会确认 Condition 持续到配置的 duration、节点没有尚未结束的 HealingTask，并通过策略限流后创建任务。HealingTask 先 Cordon，再按 Eviction/PDB 语义 Drain，随后按模板执行节点 Reboot。重启只代表动作完成，恢复判据是原始 `HygonDCUPCIeLinkLost` 明确变为 False 并稳定至少一分钟，之后才 UnCordon；如果 Condition 仍为 True、消失或变成 Unknown，就继续保持隔离并进入失败和人工处理。",
            "为什么选择Kubernetes_CRD和Controller_而不是自己做任务系统": "因为故障事实和主要动作都位于 Kubernetes：输入是 NodeCondition，隔离和 Drain 要操作 Node、Pod 与 PDB，恢复后还要重新开放调度。CRD 提供持久化的 Spec/Status，Controller 提供事件驱动、乐观并发和持续调和，使任务在进程重启后仍能从原来的进度继续。它不会自动提供跨对象事务或业务唯一性，所以单节点互斥、外部动作防重和限流仍由项目自己设计。",
            "为什么不用旧NodeRemediation链路": "旧模型更偏一个场景一组 Remediation/Operation/Job，多厂商多故障会让对象数量、页面拼装和限流归属变复杂。新模型让一个 NodeHealer 聚合某厂商的节点范围、多个 Condition 和独立限流，MachineHealingTemplate 管动作，HealingTask 记录每次维修的执行内容和恢复状态。",
            "为什么策略模板任务要拆成三个CR": "因为它们分别解决策略、动作定义和执行实例三个问题。NodeHealer 描述哪些节点和故障适用、优先级、限流及引用哪个模板；MachineHealingTemplate 保存经过审核的 Cordon、Drain、Reboot、UnCordon 动作；HealingTask 则记录某台节点这一次维修的目标、来源 Condition、动作步骤和执行状态。拆开后模板可以复用，策略变更不会改写正在执行的任务，每次维修也能留下独立历史。",
            "为什么一定要有HealingTask_Controller直接收到Condition后处理不行吗": "不行，节点自愈是跨多轮调和的长流程，关键进度必须持久化。假设已经完成隔离和 Drain，重启请求发出后 Controller 崩溃；如果状态只在内存里，重启后就可能再次 Drain 或再次 Reboot。HealingTask 保存本次维修的不可变快照和执行进度，Controller 恢复后先读取未完成任务，再核对节点 taint、Pod、云实例或 boot_id 以及原始 Condition 的实际状态，然后决定继续等待还是推进，而不是从头重跑。",
            "HealingTask的reconcile主流程是什么": "HealingTask Controller 的 reconcile 会读取所有尚未结束的任务，并确保对应的执行 handler 已经启动；handler 按任务中记录的步骤顺序执行 Cordon、Drain、可选 Reboot，随后等待健康复检，同时持续写入阶段和步骤结果。这个实现会由 handler 连续推进多个动作。进程重启后会重新启动这些任务的执行流程，因此普通步骤要能重复检查当前状态并继续完成目标；Reboot 这类破坏性动作则要先查询实例操作或比较 boot_id，避免没有确认上次结果就再次执行。",
            "HealingTask状态机为什么不能只有Pending_Running_Success_Failed": "因为节点自愈存在“动作已经执行但健康尚未确认”和“健康已确认但节点尚未解封”两类关键中间状态。任务需要记录原始 Condition、步骤结果以及 Processing、Recovering、Recovered、Completed/Failed 等阶段，才能区分是在执行动作、等待复检，还是等待 UnCordon。进程重启后还要结合任务 Spec 和外部实际状态，从原来的进度继续处理，不能只凭一个 Success 字段跳过动作。",
            "自愈系统怎么保证幂等": "这里不追求跨 Kubernetes、云 API 和 SSH 的 exactly-once，而是用 at-least-once reconcile 加实际状态校验。隔离前看自愈专用 NoSchedule taint 是否已经存在，Drain 看需要驱逐的目标 Pod 是否已经退出并继续遵守 PDB，CVM 重启查询实例操作，TCS 重启比较任务里持久化的旧 boot_id，恢复则检查原始 Condition 是否明确 False 并稳定。也就是说，是否推进取决于当前事实，而不是只相信上一次 RPC 的返回值。",
            "HealingTask状态和实际节点状态冲突时信谁": "不能简单回答“实际状态优先”。HealingTask Spec 记录本次维修必须达到的目标和要执行的动作；Node、Pod、云实例、boot_id、Condition 等外部状态用于判断动作是否真正完成；Status 记录 Controller 已观察到的流程进度。发生冲突时，以 Spec 判断目标是什么，以外部状态判断目前做到哪一步，再修正 Status 或继续完成剩余动作，不能因为 Status 写成成功就忽略 GPU Condition 仍为 True。",
            "怎么发现节点故障_Controller自己轮询硬件吗": "节点上的 NPDPlus 负责采集厂商工具、SDK 或驱动日志，并通过 Kubernetes API 把持续故障写到 Node 的 status.conditions。Node Controller 通过 Informer 观察 Node 状态，再结合策略和持续时间决定是否创建 HealingTask。原始硬件指标不会直接推给维修 Controller，厂商检测与节点维修在这层状态契约上解耦。",
            "XID故障是不是看到就重启": "不是。日志中的 XID 先由检测层按故障语义映射为 Event 或持续 Condition；即使形成 Condition，也不等于立刻重启。NodeHealer 还会依据场景开关、持续时间、优先级、动作模板、限流和 NDB 决定是否维修。XID 是故障输入，Condition 是持续事实，NodeHealer 才是维修策略。",
            "所以你们主要就是重启对吧": "对，节点重启是我们纳管场景中的主要修复动作之一。我是在已有检测和执行框架上，补齐异构厂商适配，以及策略接入、重启防重和恢复判定这些环节。首先不同故障不能统一按重启处理：比如高温先考虑移走负载、观察温度和检查散热，未必需要重启；PCIe 链路丢失则要检查设备能否被系统识别，可以按策略尝试重启，仍找不到设备就保持隔离、转人工。落到控制面，管理员选择节点和允许自动处理的故障，动作由预置模板确定；Controller 再检查持续时间、是否已有维修任务、并发和维修额度。执行时先隔离再 Drain，Drain 没完成就不能继续 Reboot。TCS 下发重启时，SSH 断开可能是失败，也可能是机器已开始重启，所以我通过持久化旧 boot_id 并观察新值来确认，避免盲目重复下发。最后，主机重启过不等于 GPU 修好了，本次任务触发的原始 Condition 必须明确变成 False 并稳定至少一分钟才能回池；故障仍在或检查结果缺失、未知时继续隔离。这个组件恢复节点的可用状态，训练任务重试和 Checkpoint 恢复仍由上层训练系统负责。",
            "五个厂商都是你从零写的吗": "NVIDIA 和海光复用了既有 NPDPlus 能力；我补齐昇腾、昆仑芯、沐曦适配，并把五家输出统一到 Condition/Event/Metric 契约，再接入同一套派单、恢复、页面配置和交付链。",
            "这个项目让我对GPU异常深入到了哪一层": "我深入到的是“信号采集→故障语义→处置分级→恢复判据”这一层。采集侧要理解 npu-smi、xpu-smi、hy-smi、mx-smi、NVML 和驱动日志分别能证明什么；语义侧要区分设备丢失、健康异常、ECC、PCIe、HCCS/MetaXLink、温度、显存以及驱动/固件故障；处置侧要决定它是只记录 Event、形成持续 Condition，还是进入 Cordon、Drain、Reboot 或失败后保持隔离；恢复侧要求原始 Condition 明确 False 并稳定。更底层的示波器分析、PCIe 信号完整性、板卡供电和芯片失效分析属于现场硬件与厂商诊断范围。",
            "故障语义映射是不是简单的错误码翻译": "不是。一个可用于自动处置的映射至少要回答五个问题：原始证据来自主动查询还是日志事件；影响的是单卡、互联还是整机；它是瞬时事件还是需要持续维护的状态；检测失败和设备故障怎样区分；什么证据可以把 Condition 从 True 恢复为 False。之后还要让同一个 ConditionType 同时出现在检测配置、NodeHealer 和 MachineHealingTemplate 中，避免检测、策略和动作模板各说各话。",
            "具体看什么指标_沐曦_NVIDIA和海光有哪些不同": "我们通过厂商接口、命令行工具和驱动日志获取 GPU 状态，但不同厂商提供的数据格式和错误含义不同，所以需要分别适配。比如同样检查温度，英伟达这条路径调用 NVML，直接拿到温度数值和调用结果；沐曦的工具可能返回同一张卡的多个测温点，适配材料里会比较这些读数，取最高温度；海光的工具也有多个测温点，但现有实现优先读取芯片结温，也就是 junction，没有这个字段再读取 edge。所以适配不仅是换一个命令，还要明确应该读取哪个字段、这个数值代表什么，之后才能按配置判断是否高温。再比如驱动错误码，英伟达的 XID 79 表示通过 PCIe 已经无法访问 GPU，海光材料中的 XID 160 表示 PCIe 链路丢失；检测器需要按各自的日志格式提取设备地址和错误编号，再按各自的规则分类。完成判断以后，检测层再通过 NodeCondition 告诉上层这个节点出现了哪类故障，上层自愈系统根据状态和策略处理，不用再理解各家工具的输出和错误码。",
            "用一个例子讲清楚_NVML信号怎么变成NodeCondition": "以我们复用的 NVIDIA 高温检测为例，检测器先调用 NVML 读取温度，确认接口成功后，再按阈值和时间窗口判断。假设配置阈值为 85℃、窗口为一分钟，GPU 0 原先没有高温故障，窗口内依次成功采样为 [80, 89, 90]℃；代码要求至少两次采样，且达到或超过阈值的样本严格过半，所以第三次达到 2/3 时判为持续高温，不要求先等满一分钟。随后生成检测器内部的故障原因 GPUTempError，匹配 reason=GPUTempError、condition=GPUTempError 的 permanent 规则，并通过 Kubernetes API 写入 Node.status.conditions：Type 为 GPUTempError，Status 为 True，Reason 为 GPUTempError，Message 保留 GPU 编号、温度和阈值。这里 True 表示高温故障成立，permanent 表示维护持续状态，不代表硬件永久损坏；读取失败不能当作高温或恢复正常的证据。Kubernetes Event 与 Condition 是两类上报输出，不需要先生成 Event 再转成 Condition；上层 NodeHealer 再按策略决定是否维修。",
            "详细讲讲你怎么做异构GPU故障语义标准化_举几个例子": "我做的不是把厂商错误码机械换个名字，而是先理解原始信号能证明什么，再把它转换成上层可以稳定消费的故障语义。整个过程分四步：第一步确定数据来源，是周期调用厂商 CLI/SDK 得到当前状态，还是从 messages、kmsg、驱动日志里捕获一次错误事件；第二步确认设备身份和检查完成度，例如设备数、Device ID、Chip ID、Bus-Id 是否完整，命令是否成功、是否覆盖所有卡，避免把命令失败误判成设备故障；第三步把厂商字段映射成稳定的 Condition Type、Reason 和 Message，并区分只记录一次的 Event 与需要持续维护的 Condition；第四步让同一个 ConditionType 同时出现在 NPDPlus、NodeHealer 和 MachineHealingTemplate 中，由策略层决定持续时间、优先级、限流和动作。比如海光驱动日志出现 `XID: 160, info:PCIe link lost`，检测器会规范化设备 BDF 和错误信息，映射成 `HygonDCUPCIeLinkLost=True`，Reason 是 `DCUPCIeLinkLostDetected`。这类 PCIe 链路丢失不是应用重试能解决的，策略可以执行 Cordon、Drain 和节点 Reboot；重启后同一个 Condition 明确变成 False 并稳定一分钟才解除隔离，仍为 True、缺失或 Unknown 就保持隔离转人工。昇腾的例子是先通过 `npu-smi info` 和映射信息建立 NPU ID、Chip ID、Bus-Id 基线：设备数减少或 Bus-Id 消失映射为 `AscendNPUDeviceLost`；`npu-smi info -t ecc` 的不可纠正 ECC 新增映射为 `AscendNPUEccUncorrectable`；HCCS lane down 则单独映射为 `AscendNPUHCCSError`，不能把设备丢失、ECC 和互联故障混成一个“健康异常”。昆仑芯会读取 `xpu-smi` 设备表、`Volatile Uncorr. ECC`、Memory-Usage、Driver Version 和 XPU-RT Version：Bus-Id 消失是 `KunlunxinXPUDeviceLost`，不可纠正 ECC 新增是 `KunlunxinXPUUncorrectableEcc`，驱动或 runtime 不可读则是 `KunlunxinXPUDriverError`；CLI 空输出本身只能说明检测证据不足，不能直接宣布设备健康或硬件损坏。沐曦优先解析 `mx-smi -j` 的结构化设备列表，再分别检查 ECC、PCIe、MMIO 和 MetaXLink：`No available devices` 或设备 ID 消失映射为 `MetaxGPUDeviceLost`，PCIe fatal/AER、MMIO 异常和 MetaXLink down 使用不同 Condition，方便后续选择不同模板和保留准确的排障信息。处置上也不是所有 Condition 都自动重启：瞬时且不需要维护当前状态的信号只产生 Event；需要观察但风险不确定的场景可以形成 Condition、设置 duration 或默认关闭 Healing；设备丢失、严重 PCIe/驱动异常等经过模板评审的场景才允许隔离、Drain 和可选 Reboot；动作后仍异常、检测证据不足或超过重试上限就保持节点隔离并转人工。Condition 支持判断故障征象、风险等级和控制面动作，板卡、插槽、主板或供电等物理根因由现场诊断继续确认。",
            "你能判断哪些故障可自动恢复吗": "判断的是哪些故障适合自动尝试处置，不是提前保证一定修好。依据厂商资料、故障持续性、影响范围和检测结果区分：高温先考虑移走负载、观察温度和检查散热，仓库的高温示例模板就没有 Reboot；设备丢失、需要重新初始化的设备或驱动异常，可以在策略允许时隔离、Drain 后尝试重启，TEP 的设备丢失示例包含 Reboot。少量可纠正错误和短暂警告通常先记录观察；不可纠正 ECC 要按厂商建议和模板评估隔离、重启等恢复措施；确认属于应用自身的问题交给应用方处理，不能直接升级成整机重启。故障类型与动作之间的对应关系由实际启用的策略和模板决定，不能把参考手册里的所有人工处置都说成组件已自动执行。重启后仍异常、检查证据不足或超过重试上限，就保持隔离并转人工；需要断电重上电、换卡或修复物理散热的问题，超出了 TCS 操作系统软重启路径的修复能力。",
            "故障语义标准化和硬件RCA有什么区别": "故障语义标准化回答“哪个节点或设备出现了哪类可操作异常，以及控制面下一步怎么安全处置”；硬件 RCA 要进一步回答“为什么发生，是芯片、显存、PCIe root port、riser、主板、供电、固件还是环境导致”。前者是我这个组件的核心工作，能够做到设备/链路/错误类型级定位；后者通常需要更多现场日志、拓扑、AER、厂商黑匣子信息和物理检查，不应仅凭 NodeCondition 下结论。",
            "讲一个具体厂商适配_不要只列XID_ECC_PCIe": "海光 DCU 的例子最具体：驱动样本 `hycu0000:3b:00.0 ... XID: 160 ... info:PCIe link lost` 会被解析成 `DCU-Xid 160, Device 0000:3b:00.0, Info PCIe link lost`，再命中规则写入 `HygonDCUPCIeLinkLost=True`，Reason 为 `DCUPCIeLinkLostDetected`，Message 为 `Hygon DCU PCIe link lost`。上层不再依赖 hy-smi 或 XID 编码，只消费这个统一 Condition。自愈完成后也不会只看 reboot 成功，而是要求同一个 Condition 明确 False 并稳定一分钟。",
            "昇腾适配具体看什么": "昇腾适配不是只判断 `npu-smi` 命令成功。设备丢失要比较 `npu-smi info` 和映射信息里的 NPU ID、Chip ID、Bus-Id 及设备数量，映射成 `AscendNPUDeviceLost`；健康、温度、ECC、HCCS、PCIe、驱动/固件分别使用独立查询和日志证据，映射为不同 Condition。比如 HCCS lane down 不能归到普通高温，驱动不可用也不能伪装成某张卡丢失。上层 NodeHealer 只消费统一 Condition，不解析厂商字段。",
            "昆仑芯适配具体看什么": "昆仑芯通过 `xpu_smi` 或 `xpu-smi` 读取设备表和状态。设备条目数减少、Bus-Id 消失或不可访问映射为 `KunlunxinXPUDeviceLost`；`Volatile Uncorr. ECC` 出现新增不可纠正错误映射为 `KunlunxinXPUUncorrectableEcc`；Driver Version 或 XPU-RT Version 不可读归到 `KunlunxinXPUDriverError`。关键是先确认设备枚举完整和检查执行成功，再解释具体字段，不能把空输出当健康。",
            "沐曦适配具体看什么": "沐曦优先使用 `mx-smi -j` 获取结构化设备列表，再按需要调用 temperature、memory、ECC、PCIe、MMIO 和 MetaXLink 查询。`No available devices` 或设备 ID 消失映射为 `MetaxGPUDeviceLost`；PCIe fatal/AER 事件映射为 `MetaxGPUPCIeError`；MMIO 与 MetaXLink 分别使用独立 Condition，避免所有链路问题都归成一个笼统的 PCIe 错误。结构化 JSON 解析还要校验字段存在、类型和设备覆盖数。",
            "三个新增厂商适配真正共用的框架是什么": "共用的不是错误码，而是检测契约：先完成设备枚举，再执行各检查项；每个检查项记录是否真正执行和覆盖了哪些设备；原始字段映射成稳定的 Condition Type、Reason 和 Message；只有检查完成且无异常才允许恢复为 False。厂商差异留在命令、解析器和错误映射层，NodeHealer、HealingTask、限流与恢复状态机不感知 npu-smi、xpu-smi 或 mx-smi。",
            "Condition的Type_Reason_Message和时间怎么设计": "Type 要稳定表达上层策略真正关心的故障语义，例如 `HygonDCUPCIeLinkLost`；Reason 表达本次转换原因，例如 `DCUPCIeLinkLostDetected`；Message 保存可排障但不适合参与机器判断的说明。LastTransitionTime 由状态实际变化驱动，NodeHealer 用它计算故障持续时间，恢复侧也用它计算连续健康时间。不要把设备地址等高基数字段塞进 Type；这些信息更适合 Message、Event 或 Metric 标签。",
            "怎么区分厂商工具失败_驱动失败_设备真实故障": "先分证据层。命令不存在、超时或返回码异常，说明检测链路失败；驱动模块、设备节点或管理服务不可用，说明驱动/运行时异常；命令成功且枚举到设备后，温度、RAS、PCIe、设备数量等字段异常，才是设备健康证据。三类问题应映射成不同 Condition 或检测状态，不能把 CLI 没输出直接解释成设备健康，也不能把检测器自身故障误报成某张卡硬件损坏。",
            "检测命令超时或卡死怎么办": "厂商检测命令已统一通过 `CommandContext` 设置超时；超时后终止子进程，限制输出大小，记录命令、设备索引、耗时和 stderr，并把本轮检查标记为未完成。检测链路失败和设备故障分开上报，旧故障 Condition 不会因为一次命令超时被误清为 False。",
            "故障消失后Condition怎么恢复": "恢复不是简单地“这一轮没匹配到日志”。对应检查必须真实运行、覆盖预期设备且没有检测错误，才能把可恢复 Condition 写成 False；Condition 缺失或 Unknown 都不是正向健康证据。Node Controller 还要求这个 False 状态从 LastTransitionTime 起稳定至少一分钟，避免刚恢复又抖回 True 时提前解除隔离。",
            "如何避免瞬时抖动触发自愈": "入口使用 Condition 的 LastTransitionTime 和 NodeHealer 配置的 duration，只有 True 持续达到阈值才创建 HealingTask；恢复端再要求 False 稳定一分钟。这样触发和恢复两侧都有时间门槛。对计数型 ECC 还应比较增量和趋势，而不是看到历史累计值非零就重启。",
            "为什么XID160选择Reboot而不是任务重试或GPU_Reset": "XID 160 表示的是 PCIe link lost，故障域已经超出单个训练进程。任务重试不会让丢失的 PCIe 设备重新枚举；当前 node-healing 动作集合也只有 Cordon、Drain、Reboot、UnCordon，没有 GPU Reset。因此模板选择节点级隔离和可选 Reboot更符合现有能力边界。如果重启后仍然掉卡，就不再用重复重启掩盖硬件问题，而是保持隔离交给运维检查插槽、供电、主板或卡本身。",
            "Reboot失败多少次后转人工": "次数由 MachineRebootConfig 的 `MaxRebootTimes` 和重试间隔控制，不写死在回答里。每次重试前先确认上一次动作是否已经发生；达到配置上限或健康检查超时后，HealingTask 进入失败收口，节点继续保持隔离，系统自动创建维修单并带上节点、厂商、故障 Condition、已执行步骤和失败原因，再交给交付、运维或厂商继续处理。",
            "Drain以后训练任务会自动恢复吗": "Drain 本身不保证训练恢复。它通过 Eviction 驱逐 Pod，是否创建替代 Pod 由上层控制器决定；分布式训练是否整组重试、能否从 Checkpoint 继续，还取决于训练框架和任务配置。我的组件负责节点隔离、受控维修和健康后回池；Drain 受阻时不能直接进入 Reboot，节点恢复也不作为训练恢复的证明。",
            "Condition_Event_Metric分别解决什么问题": "NodeCondition 记录节点当前的持续状态，供策略派单、防抖和恢复确认使用；Event 记录发生过的事件，用于排障；Metric 是监控时间序列，用来观察趋势。Condition 和 Event 通过 Kubernetes API 写入相应对象，Metric 通常由监控系统抓取，不是每次采样都写入 etcd。自愈 Controller 主要根据 Condition 与策略决定动作。",
            "为什么用NodeCondition而不是Event或单独的故障CRD": "NodeCondition 表达的是节点当前是否仍处于某种故障状态，Controller 可以围绕它持续调和，并使用 Status 和 LastTransitionTime 做触发、防抖与恢复确认。Event 更适合记录某一时刻发生过什么，可能过期、聚合或重复，不能作为解除隔离的可靠依据。当前修复单位是节点，如果再创建一套故障 CRD，就要额外维护它与 NodeCondition 的创建、更新、删除和一致性；因此当前用 NodeCondition 保存故障事实，用 Event/Metric 保存解释与观测，用 HealingTask 保存长流程执行状态。若未来要表达每张 GPU 的独立故障实例、同类型多次故障和完整诊断证据，再引入独立 Fault CRD 会更合适。",
            "permanent是不是永远不能恢复": "permanent 表示故障需要持续维护为 Condition；能否自动改为 False 由具体恢复策略决定。瞬时或可复检故障在主动检查通过后可写 False，永久硬件故障则保持 True 并转人工。",
            "厂商CLI没输出会不会被当健康": "不能把没输出当作健康。比如沐曦适配要先确定设备集合；假设枚举出了八张卡，温度查询只返回一张卡的正常结果，就不能认为整台机器的温度检查通过。每个检查项还要核对目标设备和必需字段是否都成功解析，命令返回成功也不代表每张卡都检查成功。命令缺失、解析失败、设备覆盖不完整或检测器没有运行，都只能说明本轮没有足够证据；只有与该 Condition 对应的检查实际完成、覆盖满足要求且结果正常时，才允许将可恢复的故障状态明确写回 False。",
            "为什么Reboot成功还不算恢复": "云 API 成功只说明请求被接受，DescribeInstances 成功也只说明主机操作完成；GPU 可能仍掉卡、Driver 未恢复或检测器未运行。必须等触发任务的 Condition 明确为 False 且保持稳定，才能确认 GPU 已经恢复。",
            "多Healer会不会同时修一台节点": "不会。多个 Healer 只是提出不同处理方案，不会同时执行。Controller 先检查该节点是否已经有还没有结束的 HealingTask；有就继续原任务，不再派新任务。该节点没有尚未结束的任务时，才收集满足条件的 `(NodeHealer, Condition)`，按 priority 选择最高优先级，再按 HealerName、ConditionType 保证每次排序结果一致。每轮只有一个胜者进入所属 Healer 的 limiter，其他方案不占额度；当前任务结束后，仍为 True 的故障才重新参加下一轮。",
            "同一节点同时出现多个故障怎么处理": "比如 node-01 同时上报健康异常和 PCIe 错误，Node Controller 会把已启用、Condition=True 且持续时间达到 duration 的故障加入候选；先按 nodeName 检查是否已有未结束的 HealingTask，有就等待当前任务。没有现有任务时，按 priority 选择最高优先级，同优先级再按 HealerName 和 ConditionType 固定排序，选中的策略通过 limiter 与 NDB 检查后才创建任务。假设健康异常优先级更高，本轮就只为它创建一张 HealingTask，PCIe 故障先等待；HealingTask Controller 真正执行节点动作前还会按 nodeName 再检查一次。当前任务结束后重新读取 Condition，其他仍为 True 的故障重新参加仲裁。这两次检查用于避免同一节点被重复维修，但 List→判断不是 API Server 上的原子唯一约束，priority 也不是分布式锁。",
            "怎么避免同一节点创建重复HealingTask": "互斥对象是 nodeName。以 node-01 为例，Node Controller 创建任务前先查它是否已有尚未结束的 HealingTask；有就不再派新任务。真正执行 Cordon、Drain、Reboot 前，HealingTask Controller 再检查一次，避免其他入口创建的任务同时操作 node-01。priority 决定多个故障先处理哪个，按 nodeName 的两次检查提供前后两层保护；严格原子唯一需要唯一锁对象或 resourceVersion CAS。",
            "为什么多个故障不合并到同一张HealingTask": "因为不同故障的触发原因和恢复证据可能不同。假设一张任务同时包含 GPU 掉卡和 PCIe 错误，执行一次 Reboot 后只有其中一个 Condition 恢复，就很难判断整张任务是否已经成功以及能否解除隔离。当前设计让一张 HealingTask 对应本轮选中的故障和动作，完成后重新读取节点状态；其他 Condition 仍为 True 时再进入下一轮。这样牺牲少量处理并行度，换取明确的故障归因、恢复判断和审计记录。",
            "Detector重启或原Condition消失怎么办": "Condition 消失或变成 Unknown 都不能证明恢复，因为可能只是 Detector 尚未启动、检查失败或状态链路中断。HealingTask 继续等待恢复并保持节点隔离，直到检测层重新给出明确的 False；超过恢复窗口仍没有明确的恢复证据，就转失败和人工处理，不能因为“没有再看到错误”自动解除隔离。",
            "四个YunAPI分别做什么": "ListMachineHealingTemplates 返回厂商入口、预置模板和默认 Healer；ListNodeHealers 返回当前 Healer 配置、候选节点和已选节点；ApplyNodeHealer 保存完整策略，并按用户提交的完整节点列表核对启用关系；ListHealingTasks 查询执行历史。平台层负责配置和查询，不直接执行自愈，也不判断 GPU 是否健康。",
            "同优先级为什么还要按名称排序": "固定排序可以保证 Controller 重启或重复 Reconcile 时仍选出同一个候选，避免任务和 limiter 的归属发生变化。",
            "最高优先级被限流为什么不跑第二名": "那会绕过优先级和容量保护。正确行为是让最高优先级候选一分钟后重试；当前任务结束后，其他仍为 True 的故障再重新参加仲裁。",
            "多Healer怎么兼容旧链路": "通过开关兼容原有单归属模式；异构生产链路启用 selector 后，一个节点可以被多个 Healer 匹配，但最终仍是多候选、单胜者、单任务。",
            "TCS重启如何防止重复执行": "TCS 通过 SSH 下发操作系统重启，但 SSH 断开既可能是命令失败，也可能是机器已经开始重启，不能只凭返回错误就重发。我在重启前读取并持久化旧 boot_id，持久化失败就拒绝下发；若已有本次重启的证据，就先继续确认结果，不重复发命令。下发后遇到 SSH 传输错误时保留旧 boot_id，把结果视为未知，后续继续观察；新旧 boot_id 变化才确认主机发生了重启，Controller 重启后也能读取持久化证据继续处理。这个判断只确认主机重启，GPU 是否恢复仍要复查本次任务的原始 Condition。",
            "Reboot超时后怎么办": "超时后不能直接重新下发命令。CVM 路径继续查询实例操作和主机状态；TCS 路径在 SSH 断开后先把结果视为未知，持续比较 boot_id。只要没有拿到动作完成和健康恢复的明确证据，节点就保持隔离；超过任务的动作或恢复期限后转失败或人工处理，不能盲目再次 reboot。",
            "Controller重启后如何继续任务和恢复限流记录": "Controller 启动时先读取所有还没有结束的 HealingTask，并根据任务记录恢复原始 Condition、已经占用的并发额度，以及仍处于小时、天和冷却窗口内的维修记录，然后从原来的步骤继续处理，再接收新的候选。Controller 调和是 at-least-once，不是 exactly-once，因此关键步骤必须能安全重试，并通过外部状态判断动作是否已经完成。",
            "多个Controller实例如何避免重复执行": "通用方案是用 Leader Election 让多个副本中同一时刻只有一个 active Controller，但选主不能单独保证外部 Reboot 只发生一次：旧 Leader 可能已经发出请求但尚未记录结果，新 Leader 仍要读取 HealingTask、CVM 操作状态或 TCS boot_id 判断动作是否已经发生。当前核对到的 node-healing 部署清单是单副本并关闭 Leader Election，代码只提供可选开关，因此面试时要把通用多副本设计和当前部署事实分开，不能说成已经通过多副本选主生产运行。",
            "为什么要先Cordon再Drain": "先隔离是为了阻止不容忍隔离污点的新 Pod 调度到故障节点，再 Drain 驱逐允许中断的已有 Pod。这个项目通过添加自愈专用 NoSchedule taint 隔离，Drain 遵守 PDB；如果反过来执行，旧 Pod 尚未退出就可能有新 Pod 进入。替代 Pod 的重建和调度由上层控制器与 Scheduler 负责。",
            "Drain成功Reboot失败是否解封": "默认保持隔离。只有节点随后恢复、本次任务触发的 Condition 明确为 False，且模板允许 UnCordon 时，系统才会解除 taint。",
            "Drain被PDB阻塞怎么办": "Drain 走 Eviction API，API Server 会依据 PDB 的 minAvailable 或 maxUnavailable 和当前 disruptionsAllowed 判断是否允许；预算不足时通常返回 429，自愈任务保持节点隔离，等待副本恢复或预算释放后重试。达到任务超时后进入失败或人工处理，不能为了完成维修自动退化成直接 Delete，因为直接 Delete 会绕过 PDB。确需强制处理时只能走单独授权、限范围和全审计的 break-glass 路径，不能隐藏在普通自愈重试里。",
            "直接Delete会绕过PDB_怎么补充类似保护": "这是增强设计，不是当前项目已经实现的能力。PDB 只在 Eviction API 路径执行原生保护，直接 Delete 不会自动获得同等预算检查；因此普通场景仍统一走 Eviction，并通过 RBAC 限制业务方直接 Delete。确实需要受管 Delete 时，我会让入口先向独立预算控制器申请以 Pod UID 和工作负载 UID 为键的 DisruptionReservation，由单一账本通过 resourceVersion CAS 原子计入正在中断的副本；Validating Webhook 只放行持有有效 Reservation 的请求，不能只读取一次 disruptionsAllowed 后放行，否则仍有并发超额竞态。Delete 失败时释放预约，结果未知时先观察 Pod UID、deletionTimestamp 和替代副本状态，替代副本 Ready 后归还额度，超时对账清理孤儿预约。节点彻底失联或硬件风险继续扩大时保留独立权限、明确原因和全审计的管理员 break-glass 路径；正常预算服务异常时默认拒绝。",
            "跨应用多故障等级如何选择最小维修动作": "这是对现有节点级 Cordon、Drain、Reboot 和 NDB/PDB 的增强设计，不冒充当前实现。先根据证据与故障域确定最低必要动作：低风险信号继续观察，进程问题先任务重试，单卡且厂商支持时做设备隔离或 Reset，只有 Driver、PCIe、互联组或整机故障才升级到 Cordon、Drain 和 Reboot。然后枚举节点上的所有应用，分别检查健康副本与 PDB、本地状态、训练 Checkpoint 和 Gang 影响、推理容量与 SLO；任何一个应用不满足安全门槛，都不能用其他应用的富余副本抵消。执行前把节点池容量和全部受影响应用的中断额度汇总成 MaintenanceReservation，由唯一预算分配器通过 resourceVersion CAS 一次性从 Pending 推进到 Granted，拿不到完整额度就等待，不能先占一部分再操作。获得额度后仍按 Cordon、Eviction Drain、修复、复检执行；结果未知时观察实际状态，失败后默认保持节点隔离，只有替代副本 Ready、故障恢复且对账完成后才释放预算和 UnCordon。",
            "怎么判断一次自愈真正成功": "项目层的成功结论是故障节点恢复了可用资格。Cordon、Drain、Reboot 完成只是动作结束，还要确认本次 HealingTask 记录的原始 Condition 仍存在、明确为 False 且稳定至少一分钟，再完成 UnCordon，任务才能结束。Node Ready=True 或重启命令成功都不能单独证明 GPU 恢复；业务侧是否恢复请求或训练进度，还要由工作负载状态和业务指标确认。",
            "节点恢复过程中再次故障怎么办": "只要原始 Condition 再次变成 True，当前任务就不能进入 Recovered，节点继续保持隔离。如果它随后再次变成 False，LastTransitionTime 会更新，连续健康时间需要重新计算；不能沿用上一次短暂健康的计时。如果原故障已稳定恢复但另一条 Condition 仍为 True，则先让当前 HealingTask 完成，再重新仲裁下一条故障，避免两条维修流程同时操作同一节点。",
            "NodeReady为True为什么还不能认为GPU恢复": "Node Ready 主要说明 kubelet 和节点基础链路可用，不能证明 GPU 数量、驱动或具体故障已经恢复；一台节点完全可能 Ready=True，但某张卡仍异常。因此 GPU 自愈会检查 HealingTask 中记录的触发 Condition：它必须存在、明确为 False，并稳定至少一分钟。",
            "PhaseRecovered为什么还不算任务结束": "它只是 Node Controller 确认健康证据已经满足的信号，此时节点可能仍带 taint；HealingTask Controller 完成 UnCordon 后才能写 PhaseCompleted，任务才真正结束。否则会出现任务显示完成但节点仍未重新开放调度的问题。",
            "NPDPlus_NDB和PDB分别是什么": "NPDPlus 是节点侧的故障检测组件，负责调用厂商工具、读取驱动日志并输出 NodeCondition；NDB 是 NodeDisruptionBudget，在节点维修入口按 selector、MaxUnavailable/MinAvailable、CPU/内存比例和可调度节点数判断节点池还能否下线一台机器；PDB 是 Kubernetes PodDisruptionBudget，在 Drain 阶段保护应用 Pod 副本可用性。三者分别属于检测层、节点容量准入层和 Pod 驱逐保护层；没有 NDP 这个容量预算名称。",
            "为什么TEP里写了NPDPlus却没单独写NDB": "这份 TEP 的主线是异构故障检测、NodeCondition 标准化和自愈产品接入，所以组件章节重点写了 NPDPlus；NDB 属于 node-healing 内部的维修准入能力，最终交付代码已经通过 `HealingPolicy.nodeDisruptionBudget` 接入 limiter。它不是另一个检测组件：Controller 在创建 HealingTask 前检查节点池的不可用数、可调度节点数以及 CPU、内存容量比例；预算不足或检查异常时都不启动新维修。",
            "NDB和PDB有什么区别": "NodeDisruptionBudget 在节点维修入口判断还能不能再下线一台节点；PodDisruptionBudget 在 Drain 阶段保护 Pod 副本可用性，作用层级不同。",
            "怎么避免故障风暴触发大面积维修": "入口先用 duration 过滤瞬时抖动，再按每个 Healer 独立检查并发、小时、天和节点冷却额度，同时用 NodeDisruptionBudget 约束节点池还能下线多少节点。NDB 无法确认剩余容量时拒绝启动新维修，最高优先级候选被限流时只延迟重试，不通过执行次优候选绕开保护。",
            "这个项目最难的部分是什么": "最难的是让结果可能暂时无法确认的外部操作能够安全重试，并控制批量维修的影响范围。核心有三点：用 HealingTask 持久化跨轮状态；用 taint、Pod、实例操作、boot_id 和 Condition 等实际状态判断每一步是否已经完成；用稳定仲裁、限流、NDB/PDB 和失败后保留隔离来避免同时影响过多节点。",
            "页面保存策略和节点选择能原子吗": "不能。一个策略对象加多台节点的启用关系没有跨对象事务；接口每次都拿用户提交的完整节点列表重新核对，部分失败后继续重试，直到实际选择和用户提交的一致。",
            "为什么页面不能编辑CordonDrainReboot步骤": "这是权限和安全边界。页面只管节点、场景开关、priority、限流和通知；破坏性动作由 Addon 预置并评审过的 MachineHealingTemplate 管理，避免用户任意拼步骤。",
            "通知能保证不重不漏吗": "不能。阶段和渠道标记可以减少重复，但发送成功后若记录发送结果失败，仍可能再次发送；有限重试全部失败后也可能漏发。系统只能尽量送达，消息服务故障不能阻塞主自愈链路。",
            "NoSchedule是不是绝对封锁": "不是，带匹配 toleration 的 Pod 仍可能调度。它是自愈专用隔离标记，Drain 和集群 toleration 策略共同决定实际隔离强度。",
            "为什么不用DevicePlugin健康状态直接自愈": "Device Plugin 报告设备不健康后，主要影响后续设备分配，不能完成已有 Pod 迁移、节点级动作、限流和恢复复检，因此不能替代 NodeCondition→HealingTask 控制链。",
            "8卡节点只坏1张卡_为什么还可能Cordon和Drain整台节点": "当前自愈链路的执行单位是节点，是否进入 Cordon、Drain 和 Reboot 由故障类型和动作模板决定。Device Plugin 把单卡标记 unhealthy 可以阻止它继续分配给新 Pod，却不会迁移已在使用它的训练进程。驱动、PCIe、GPU 互联或节点级故障共享更大故障域时，整机隔离和重启更符合安全边界；单设备摘除是另一类设计，当前项目没有实现 GPU 热迁移或热更换。",
            "怎么定义项目闭环": "配置部分包括模板、YunAPI、NodeHealer 和节点启用标签；运行部分从检测、仲裁、限流、创建任务、执行动作一直到 Condition 复检和重新开放调度；产品部分包括历史、通知和 Addon 交付。三部分都读取 HealingTask 的任务记录，但历史和通知不能反过来决定节点是否健康。",
            "两个用户同时修改同一个NodeHealer怎么办": "ApplyNodeHealer 会先读取当前对象，再带着当前 resourceVersion 执行 Update。并发修改时，先成功的写入会推进 resourceVersion，后一个旧版本写入应收到 Conflict，调用方需重新读取后再提交，不能静默覆盖。但 NodeHealer 更新和多节点标签对账仍不是一个原子事务，后半段失败仍要用完整期望状态重试。",
            "NodeHealer或模板更新会不会改变正在执行的HealingTask": "不应该改变。创建 HealingTask 时，Controller 已经把本次触发的 Condition、动作步骤、Cordon、Drain、Reboot、UnCordon 配置和超时参数拷贝到 HealingTask Spec。后续 NodeHealer 或 MachineHealingTemplate 的修改只影响新创建的任务，在途任务继续按自己的 Spec 执行，否则会出现任务执行一半语义突变。",
            "取消某些节点自愈时标签如何收敛": "ApplyNodeHealer 接收的是完整 SelectedNodeNames，而不是单次增删事件。平台会把当前已打标节点与这份完整集合比较，对新选节点加标签，对已取消节点移除标签。中途失败会返回错误；重试同一份完整列表时再次计算差集，直到实际标签收敛。这是幂等对账，不是跨多个 Node 的原子更新。",
            "HealingTask历史查询和长期保留怎么做": "ListHealingTasks 支持按厂商、节点、状态和时间范围过滤分页，页面查询依据是持久化 HealingTask，而不是 Controller 内存。任务已配置 TTL 和 GC：执行中任务保留用于重启恢复，进入完成或失败等终态后，按保留期供历史查询，到期后由 GC 清理，避免 HealingTask 长期累积增加 API Server 和 etcd 负担。",
            "是在训练平台还是推理平台使用": "我负责的是 TCS 私有云里 GPU Kubernetes 集群的节点自愈控制面，直接服务平台运维。它不负责训练编排或推理请求调度；节点承载训练时，任务状态由训练框架和平台保存、恢复，承载推理时则要保护副本容量、预热和请求处理。项目能确认的职责是隔离故障节点、执行受控维修并在健康后回池，不能据此把上层训练任务归类为无状态。",
            "具体看什么信号才能判定GPU故障": "先看设备是否完整枚举、驱动是否可用，以及不可纠正 ECC 和设备互联异常。比如海光驱动日志出现 PCIe link lost，会被映射成 PCIe 链路故障 Condition，持续达到策略阈值后才进入维修；它证明的是链路故障征象，不能直接认定板卡损坏。GPU 利用率低、一次 CLI 超时或输出不完整，都不足以单独触发节点重启。",
            "NVIDIA和沐曦的采集适配具体差在哪": "主要差在信号来源、设备标识和字段含义。NVIDIA 侧结合 NVML、nvidia-smi 和驱动 XID 等信号；我新增的沐曦适配优先解析 mx-smi -j 的设备列表，按设备 ID 校验覆盖，再分别检查 ECC、PCIe、MMIO 和 MetaXLink。适配层将这些厂商信号映射成约定的 Condition 类型与状态，策略和维修流程消费这个契约，不把不同厂商的错误码当成同一语义。",
            "扩到5000节点这条自愈链路哪里需要加固": "我会先测两段压力：检测上报到 API Server 的写入，以及故障事件到维修任务的排队时间。大量节点同时上报时，优先减少无变化的 Condition 写入和重复 Event；Controller 检查是否存在全量扫描、慢调用占住 worker 或持续重试，再针对性做索引、队列隔离和限速。并发度不能只往上加，还要受全局及节点池维修预算约束，防止故障风暴一次摘走大量容量。验证看 API 请求延迟与 429、队列深度、调和耗时和同时维修节点数，不能只凭 5000 这个数量认定某处一定是瓶颈。",
            "NodeController具体在哪些地方访问APIServer": "读取侧通过 List/Watch 建立并更新 Node、NodeHealer 和 HealingTask 的 Informer 缓存，调和时很多查询从缓存读取，不是每次都直连 API Server。写入侧会创建 HealingTask；任务控制器执行隔离时更新 Node，Drain 时调用 Pod Eviction API，并把任务阶段写回 status。NodeCondition 是 Node.status.conditions 中的字段，不是一个单独的 Kubernetes 资源。",
            "这个项目从零到一还是优化_具体改善了什么": "是在已有 node-healing 基础上完成异构场景的增量落地。原来已有策略、任务和隔离重启等动作，NVIDIA、海光也有检测基础；我补齐昇腾、昆仑芯、沐曦适配，以及多策略协调、恢复判定、平台配置和交付接线。具体改善是让新增厂商的标准故障进入统一处置流程，运维可以集中配置节点和场景、查看任务进度，节点也按一致的健康标准回池。收益落在厂商接入范围、人工处置环节和恢复安全性上，不把已有执行引擎说成从零重写。",
            "举一个适合自动自愈的场景_收益具体落在哪里": "可以用已有联调覆盖的海光 DCU PCIe 链路丢失说明流程。检测信号映射为持续故障 Condition，节点启用对应策略、持续时间和维修额度满足后创建任务；在业务已允许中断、Drain 能通过的前提下，系统隔离节点、按 PDB 执行驱逐，再按模板重启，原始故障明确恢复并稳定后才回池。流程由持久化任务推进，运维通过状态和历史跟踪，在失败阶段接手。这个例子说明自动处置和容量回池的机制；我的增量是把新增厂商接入这套流程并补齐平台控制链。节点若承载训练，保存进度和恢复任务还需要上层协作；重启后故障仍在则保留隔离并转人工。",
            "如果追问收益数字_恢复耗时和GPU容量怎么计算": "我会按厂商、故障类型和工作负载分组，比较相同口径的自动处置与人工处置样本。恢复耗时从首次明确故障算到恢复调度资格，包含持续时间等待、排队、Drain、重启和复检，不能只取 HealingTask 的执行时间；没有恢复的事件也要单列，不能从样本中删除。人工收益看每次故障实际需要接手的次数和操作时长，资源损失用不可用 GPU 数量对时间积分：整台八卡节点退出资源池半小时，就是 4 GPU·小时的容量不可用，但这是计算示例。改善值来自与可比基线的差异，统计中要计入误触发、额外隔离等损失；容量回池不等于 GPU 利用率或训练吞吐自动提高。",
            "训练有Checkpoint就算无状态吗": "不算。训练任务需要维护模型参数、优化器状态、训练进度，以及按需恢复的数据迭代位置和随机数状态。Checkpoint 把这些状态保存到持久化存储，使新的训练进程能够从某个进度恢复；训练 Pod 可以替换，只表示计算进程可重建，不代表训练没有状态。训练平台的 API 或 Controller 可以把状态放在 Kubernetes 对象或数据库中，使实例逻辑上无状态，但它们管理的训练任务仍有状态。因此讨论节点自愈时，应说明训练是否允许中断、从哪里恢复以及是否需要整组重启。"
          },
          "绝对红线": [
            "不要把多 Healer 说成多个 Healer 同时修一台节点；多个 Healer 可以同时提出候选，但每轮只选一个，而且同一节点同一时刻只有一个还没有结束的任务。",
            "不要把 Reboot 请求成功、主机重启完成和 GPU 恢复混成一件事；它们分别是请求、动作、目标状态三层证据。",
            "不要说跨 Kubernetes、云 API、SSH 和消息服务实现了 exactly-once，也不要承诺通知具备严格的 at-least-once；准确口径是关键步骤可以安全重试，破坏性动作通过外部状态防重，通知只做有限重试。",
            "不要把 PDB、节点维修预算或 Drain 成功说成训练可以安全中断的充分条件；它们不提供 Checkpoint 协调、任务重试或整组恢复。",
            "不要把训练平台应确认的中断准入、Checkpoint 或恢复协议写成本项目已有检查；这些属于接入训练场景需要单独实现和验证的协作能力。",
            "不要说 Cordon 后故障节点立刻不再承载任务；它主要阻止不容忍该隔离条件的新调度，已有 Pod、容忍污点的 Pod 和 Drain 阻塞需要分别说明。"
          ],
          "implementation_details": {
            "对象模型": [
              "NodeHealer 表达一个厂商节点范围、故障场景、优先级、限流和模板引用，是可变的策略对象。",
              "MachineHealingTemplate 由 Addon 预置并审核，定义不同 Condition 对应的 Cordon、Drain、Reboot、UnCordon 动作，避免页面任意拼装破坏性步骤。",
              "HealingTask 是一次维修的持久化记录，里面写明目标节点、来源 Healer、触发 Condition 和动作步骤；策略后续变化不会改写正在执行的任务。"
            ],
            "异构检测与故障语义": {
              "厂商适配": "复用 NVIDIA/海光检测能力，补齐昇腾、昆仑芯、沐曦适配；底层分别消费厂商 SDK、管理工具和驱动日志，下游统一只认 NodeCondition、Event 和 Metric。",
              "统一维度": "覆盖设备丢失、健康状态、温度、ECC、PCIe/互联、驱动与固件等故障维度；需要持续维护的故障写 Condition，瞬时或仅用于观察的异常写 Event。",
              "健康门槛": "只有对应检查在本轮真实执行、相关设备覆盖满足要求、检测过程无错误且没有再观察到故障时，才允许把可恢复 Condition 写为 False。命令失败、字段缺失、检测器未运行或 Condition 消失都不能证明节点已经恢复。",
              "XID与ECC": "XID 要按故障严重度区分应用可重试、需要复检和需要隔离重启的场景；ECC 要区分可纠正与不可纠正错误并关注增量和趋势，不能看到任意非零值就重启整机。",
              "海光XID160映射": {
                "原始输入": "`hycu0000:3b:00.0 ... XID: 160 ... info:PCIe link lost`。",
                "规范化结果": "`DCU-Xid 160, Device 0000:3b:00.0, Info PCIe link lost`。",
                "统一状态": "Condition Type=`HygonDCUPCIeLinkLost`，Reason=`DCUPCIeLinkLostDetected`，Message=`Hygon DCU PCIe link lost`。",
                "策略含义": "这是持续故障 Condition，不等于检测器直接执行重启；NodeHealer 决定 duration、限流和动作模板，HealingTask 执行节点级恢复。",
                "恢复含义": "同一 Condition 必须明确 False 且稳定一分钟，缺失或 Unknown 不能作为恢复证据。"
              },
              "检测链路失败分类": {
                "工具失败": "命令不存在、超时、非零退出或输出格式无法解析，表示本轮检测未完成。",
                "驱动失败": "驱动模块、设备节点或管理服务不可用，应形成独立的驱动/运行时异常语义。",
                "设备故障": "命令成功并完成设备枚举后，温度、RAS、PCIe、设备数量或健康字段异常，才构成设备级健康证据。",
                "共同安全原则": "任何检测失败都不能把已有故障 Condition 清成 False；检测链路异常与设备真实故障不能共用一个模糊 Reason。"
              },
              "命令超时防护": "各厂商 provider 统一通过 CommandContext 执行管理命令，配置逐命令超时、子进程终止和输出大小限制，并记录命令、设备和超时原因。超时属于检测链路失败，不会被误判为设备健康或用来清除已有故障 Condition。"
            },
            "多Healer仲裁": [
              "一个节点可以同时被多个 NodeHealer 的 selector 命中，但多个 Healer 只是提出候选，不会同时修一台节点。",
              "候选必须满足 Condition=True、场景已启用、故障持续时间达标，并且当前没有尚未结束的 HealingTask；再按 priority 选择最高优先级，同优先级时按 HealerName、ConditionType 固定排序。",
              "每轮只选择一个候选并创建任务。固定排序保证 Controller 重启或重复调和时仍选出同一个结果；当前任务结束后，仍为 True 的其他故障再参加下一轮。",
              "旧的单归属模式通过开关保留兼容；异构生产链路启用 selector 后仍保持多候选、单胜者、单任务。"
            ],
            "限流与影响范围控制": {
              "四维限流": "priority 决定先修谁，duration 过滤抖动，并发、小时、天和节点冷却控制维修速度；候选被限流后延迟重试，不绕过最高优先级去执行次优候选。",
              "NDB与PDB": "NodeDisruptionBudget 在节点维修入口判断节点池还能否再下线一台，检查异常时拒绝启动新的维修；PodDisruptionBudget 在 Drain 阶段保护应用副本，两者作用层级不同。",
              "额度回滚": "创建 HealingTask 失败会释放刚占用的并发名额；已经发生的维修尝试仍保留在时间窗口中，避免反复失败绕过累计限制。",
              "单节点互斥": "创建任务前和真正操作节点前，都检查该节点是否已有还没有结束的任务，保证多 Healer 场景下一台节点同一时刻只有一个维修流程。"
            },
            "HealingTask执行与恢复": {
              "Cordon": "给节点加自愈专用 NoSchedule taint，阻止普通工作负载继续调度；它不是绝对封锁，带匹配 toleration 的 Pod 仍可能进入。",
              "Drain": "通过 Kubernetes 驱逐语义移除可迁移 Pod，并遵守 PDB；工作负载是否重建、重建到哪里由 Deployment、StatefulSet 等控制器和调度器负责。DaemonSet、本地盘、静态 Pod 或 PDB 阻塞都可能让 Drain 等待或失败，不能把“发起 Drain”说成“业务已经迁移成功”。",
              "Reboot": "腾讯云路径调用 CVM 重启并查询实例操作状态；TCS 路径通过受控 SSH 下发重启，以 boot_id 变化确认主机确实发生过重启。",
              "Recover": "重启完成只代表修复动作已经执行。Node Controller 继续检查 HealingTask 中记录的原始 Condition，要求它仍然存在、明确为 False 且稳定一分钟，才标记 Recovered；随后执行 UnCordon 并完成任务。",
              "Failure": "如果流程已经执行过 Cordon，而后续动作失败、恢复超时或健康证据不足，默认保留自愈 taint，避免故障节点重新承载业务；如果失败发生在 Cordon 之前，节点保持原有调度状态。节点随后恢复且模板包含 UnCordon 时才自动解除隔离，否则转人工。"
            },
            "TCS重启防重与Controller恢复": {
              "结果未知": "SSH 在 reboot 后断开可能表示命令已经执行，不能立即重发。系统先持久化旧 boot_id，再观察新值；只有 boot_id 变化才确认重启，证据无法持久化时不下发破坏性命令。",
              "进程重启": "Controller 重启后根据 HealingTask 记录继续处理任务。启动时先恢复还没有结束的任务、原始 Condition、已经占用的并发额度，以及仍处于小时、天和冷却窗口内的维修记录，再接收新的候选。",
              "一致性语义": "跨 Kubernetes、云 API 和 SSH 不具备 exactly-once 语义，因此通过持久化任务、可安全重试的步骤和外部状态观测，做到普通请求允许重试，重启等破坏性动作不盲目重复。"
            },
            "页面配置与YunAPI": {
              "接口分工": "ListMachineHealingTemplates 提供厂商入口、预置模板和默认 Healer 名称；ListNodeHealers 返回当前 Healer 配置、候选节点和已选节点；ApplyNodeHealer 保存完整策略并对账节点启用关系；ListHealingTasks 查询执行历史。候选节点来自模板 metadata 中的 selector，由 ListNodeHealers 聚合返回。",
              "配置如何保持一致": "页面只编辑节点范围、故障场景、优先级、限流和通知；破坏性动作由预置模板控制。保存时同时更新策略对象和节点启用关系，跨对象没有事务，因此每次都按用户提交的完整配置核对，失败后继续重试直到一致。",
              "协作边界": "前端负责页面实现，我负责 YunAPI、配置模型、模板元数据、节点启用关系、历史通知契约和端到端联调。"
            },
            "历史与通知": {
              "历史": "HealingTask 记录每次维修的触发故障、动作阶段、结果和耗时，是页面历史、Controller 恢复和问题排查的共同依据。",
              "通知": "通知组件单独监听任务阶段变化，按配置向不同渠道发送消息并记录去重标记；一个渠道发送失败不会阻断其他渠道，也不会回滚主自愈流程。",
              "投递边界": "通知采用有限重试和去重，但不承诺 exactly-once 或严格的 at-least-once；发送成功后若记录结果失败，仍可能重复，超过重试上限也可能漏发。"
            }
          }
        },
        {
          "bullet_title": "设计并实现统一公网出口代理组件",
          "项目背景": {
            "业务动因": "TCE 私有化环境的公网访问需求持续增加：COS 回源需要在 Underlay 机器上解析公网域名，Pod 内的 NTP 拨测程序需要访问公网一级源，容器化 YUM 需要同步软件源，短信和邮件组件也需要访问公网消息网关。原有方案部分依赖实施文档，部分需要交付人员临时设计，缺少统一管理。VPC 侧已有的 Underlay 出网组件会修改主机网卡配置和默认路由，与 TCE 底座的节点网络管理存在冲突，因此由独立 Proxy VM 承载这部分出口能力。本项目将分散的接入、规则配置和健康管理统一成声明式控制面。",
            "为什么使用独立Proxy_VM": "独立 Proxy VM 把公网路由和 NAT 相关变化隔离在 TCE 工作节点之外，不需要修改 TCE 节点的主机网卡或默认路由，也不会把第三方 Underlay 出网组件的网络生命周期混入 TCE 底座管理。Pod 或节点只访问 Kubernetes Service/LoadBalancer 暴露的受管入口，真正的 Nginx 转发和 NATGW 出公网发生在 Proxy VM 一侧。代价是需要额外管理代理 VM 的配置一致性、健康准入、扩缩容和故障摘流，这正是 EgressForwardRule Controller 解决的问题。",
            "已有三类出公网链路": [
              "第一类是已有 unatgw 出口的 COS-CGI、部分 NTP 和虚拟机 YUM 等 Underlay 业务，缺少的是统一公网域名解析；DNS 规则补齐 RegionDNS 到公网 resolver 的白名单转发，拿到公网 IP 后业务仍通过自身 unatgw 出网。Pod 内访问公网一级 NTP 源的拨测是另一项需求，还需要业务报文的代理转发。",
              "第二类是短信、邮件等业务。它们需要单独创建代理 VM，再由运维人工 SSH 到机器上维护 Nginx 配置。",
              "第三类是容器化 YUM。它已经有一套 service-id + 专属代理 VM 的出公网链路，但只服务 YUM 场景。"
            ],
            "核心问题": "业务接入新的固定公网目标仍依赖人工准备和配置代理 VM，多台机器容易配置不一致，各需求分别建设代理也造成资源冗余，后续运维职责与问题归属不清。组件把临时方案沉淀为统一能力：用规则声明接入目标，自动发布配置并维护健康后端，让交付和后续变更有一致的入口与状态。独立代理 VM 保留已有公网出口，避免修改 TCE 工作节点的网络配置。",
            "我的工作": "我把固定公网目标统一成声明式出口控制面：授权方在自己的 Namespace 创建 EgressForwardRule，选择 ClusterIP 或 LoadBalancer 入口，Controller 按规则独立管理 Service、VM 配置、revision 和健康后端；底层复用 Nginx 和 unatgw。TCP OriginalHost 根据入口类型通过 service-id 或受管 A 记录保留原域名，DNS 规则则为已有出网能力的客户端补齐白名单解析。",
            "方案定位": {
              "EgressForwardRule": "Namespace 级的出口策略 API，用户在所属 Namespace 声明固定公网目标、协议和 ClusterIP 或 LoadBalancer 入口。",
              "Controller": "按 Namespace/name 为每条 EgressForwardRule 独立调和，在规则所属 Namespace 创建 selectorless Service、自管 Endpoints 和 EndpointSlice，并生成该规则自己的 conf、ACL 和 revision。Controller 逐 VM 发布当前规则，根据共享运行状态和该规则的检查结果维护后端；LoadBalancer 入口还需等待 VIP。全局策略或 Product 节点变化会让全部规则重新入队，但各自收敛，Controller 不转发业务报文。",
              "selectorless Service": "代理 VM 不属于 Kubernetes Pod，无法使用普通 Service selector，因此通过无 selector Service 和受控 Endpoints 把集群外 VM 纳入 Kubernetes 服务发现。",
              "Nginx": "负责实际的代理转发。",
              "unatgw": "负责最终公网出口 SNAT；本组件不重新实现公网出口能力，而是把受控流量导向平台已有的 NAT 出口。"
            }
          },
          "业务视角": {
            "直接用户与最终受益者": "直接使用者主要是平台运维、云产品交付人员和需要访问固定公网依赖的业务组件负责人。他们负责申请或创建规则；YUM、短信、邮件等业务 Pod 是规则入口的实际消费者。最终客户不需要理解代理 VM 和 Nginx 的维护细节，只需要按产品约定使用受管入口。",
            "典型业务场景": "具体有四类需求：COS 回源的 Underlay 机器需要公网域名解析，以前由交付侧临时搭 DNS Server；Pod 内的 NTP 拨测程序要访问公网一级源，受限于 Pod 的出网能力；YUM 容器化后需要通过专门的 rsync 代理同步公网软件源；短信、邮件组件需要访问固定公网消息网关。它们的目标和端口相对明确，适合逐条授权并共享代理资源。其中 DNS 需求只补解析，Pod 内 NTP 拨测还需要向指定 NTP 目标转发业务报文。",
            "一次业务使用流程": "授权方为固定公网目标创建 EgressForwardRule，选择协议和 ClusterIP 或 LoadBalancer 入口。HTTP/TCP/UDP 业务使用 status.entrypoints 返回的 Service 地址，TCP OriginalHost 也可以保留原域名；DNS 场景则由客户端先询问 RegionDNS，命中白名单 zone 才转入对应 DNS relay。Controller 分别维护每条规则的配置、健康后端、更新和删除流程。",
            "为什么只允许固定目标": "企业私有云的目标是满足必要的公网依赖，同时控制出口范围。把一个规则绑定到一个逻辑目标后，审批、白名单、配置发布、健康状态和审计记录都能对应到明确对象；如果允许客户端运行时选择任意地址，就会退化成通用代理，难以满足最小授权和问题定位要求。",
            "业务价值": "项目把按文档手工配置代理、解析和转发规则的流程改成规则驱动的自动交付，单次配置耗时从约 4 小时降到 10 分钟以内；由每个需求单独新增 KubeVM，转为多条规则共享 Proxy VM，相比原分散方案减少约 70% 的代理机资源使用。统一控制面与滚动发布减少多台 VM 配置不一致造成的访问异常，来源准入和固定目标约束降低误访问与代理滥用风险。扩容时新增 VM 自动同步现有规则，通过配置与健康检查后加入对应后端，交付人员不再逐条手工配置。",
            "业务成功标准": "控制面创建成功只表示规则已受理。真正成功要看到至少一个合格代理后端、入口发布完成，并由真实客户端通过受管入口完成目标协议事务；发生后端故障时，还要能够把问题 VM 摘流并让健康后端继续服务，或在全部不可用时留在受管入口内失败。",
            "量化口径": "单次配置耗时由约 4 小时降至 10 分钟以内，衡量的是代理、解析和转发规则的配置交付效率；代理机资源使用减少约 70%，比较的是多个需求分别新增 KubeVM 与共享 Proxy VM 的方案。资源收益限定在代理机范围，不能换算成整个私有云资源节省、CPU 利用率或现金成本。配置一致性、安全约束和扩容自动纳管是其他收益；故障 MTTR、可用率和业务成功率没有对应统计，不附加数字。",
            "能力边界": "这个组件服务固定目标的受控公网访问，不是全网透明流量治理、通用 Forward Proxy 或 NATGW 产品，也不单独阻断集群已有的直连公网路径。任意互联网访问和强制防绕行需要独立的网络与安全方案。"
          },
          "一句话": "我把固定公网目标的代理接入和多台 VM 配置维护做成统一控制面，让业务提交规则后自动获得可用入口。",
          "简历推荐写法": "设计并实现 Namespace 级 EgressForwardRule，支持 ClusterIP 和 LoadBalancer VIP 两类入口，以 selectorless Service 和受控 Endpoints/EndpointSlice 管理外部代理 VM；架构采用每 Rule 独立配置、ACL 和 revision，结合规则级容量检查、逐 VM 滚动发布及 Builder 本地锁控制变更影响。项目将单次配置耗时从约 4 小时缩短至 10 分钟以内，并通过共享 Proxy VM 减少约 70% 的代理机资源使用。",
          "20秒首答": "业务访问短信、邮件、软件源等固定公网服务时，原来需要运维逐台配置代理 VM，多台机器容易配置不一致。我做了一层统一控制面，让业务提交出口规则后，自动创建访问入口、发布配置，并只让检查通过的 VM 接流量。底层继续复用已有代理 VM 和公网 NAT 出口。",
          "项目状态": "原有 Controller、Product/Builder、TAD、平台服务发现以及 DNS/HTTP/TCP/UDP、容器化 YUM OriginalHost 链路已有生产交付；本次逐 Rule 发布改造的实现与上线状态待确认。",
          "配置与发布粒度": {
            "规则生命周期单位": "EgressForwardRule，也就是 Namespace 内的一条 CR；新增、修改和删除都从这条规则的生命周期进入。",
            "配置编译单位": "单条 EgressForwardRule。Controller 为当前 Rule 独立生成 nginx conf、ACL 和 manifest，并根据实际下发物料计算自己的 desiredRevision；不再把全部规则合并为一份配置包。",
            "发布执行单位": "当前 Rule × Proxy VM。同一 Rule 的目标 revision 按 VM 逐台执行预检、Drain、Apply、Doctor 和 Rejoin；摘流、失败恢复和后端状态都限定到该 Rule。",
            "流量准入单位": "规则 × Proxy VM；每条规则自己的 Endpoints 表达哪些 VM 已通过该规则的 listener、ACL 和健康检查，可以承载该规则流量。",
            "批量变更": "多条 Rule 各自入队、各自收敛。全局策略或 Product 节点列表变化时把所有 Rule 重新入队，每条规则只更新受影响的物料和后端；同一 VM 的共享 Nginx 切换与 reload 由 Builder 本地短锁保护。",
            "固定总结": "Rule 是生命周期和配置编译单位，Rule 与 VM 的组合是发布及流量准入单位；共享 VM 上的 Nginx 临界操作由本地锁串行保护。"
          },
          "个人实现与复用边界": {
            "我实现和交付的": "EgressForwardRule CRD、Controller 主链路、规则校验、Service/Endpoints/EndpointSlice 管理，以及代理 VM 配置生成和滚动发布；同时负责 Controller、Product/Builder、TAD 等交付物的接线与上线交付。",
            "我复用的": "Kubernetes Service/kube-proxy、现网代理 VM、Nginx 转发引擎和底层 NATGW；我的实现位于规则控制、配置发布和健康后端管理层。"
          },
          "正常链路_90秒": {
            "链路讲述原则": "问完整访问路径时，跟着一条请求说明；问 Controller 与 VM 如何交互时，只讲调用接口、远端执行和结果处理。业务背景、配置发布和 OriginalHost/YUM 按对应问题选用。",
            "一句话总览": "Namespace 级 EgressForwardRule 声明固定目标和入口 → Controller 为当前规则创建 Service 并发布独立 conf/ACL/revision → 通过共享项与该规则 Doctor 检查的 VM 才加入对应后端 → HTTP/TCP/UDP 请求经 Service、VM Nginx、unatgw 到公网；DNS 查询则先经 RegionDNS 分流。",
            "回答顺序": [
              "问背景或整体介绍时，用短信、邮件或软件源中的一个例子说明人工接入与多 VM 配置问题。",
              "问业务请求路径时，用一条请求说明 Pod → Service → 代理 VM 上的 Nginx → NATGW → 固定公网目标。",
              "问入口创建、配置发布或健康后端时，直接解释 Controller 在所问环节的操作。",
              "问发布失败、LB 入口或 YUM 原域名时，直接选取对应分支及必要条件。"
            ],
            "第一条_总体Controller调和链": [
              "1. 输入：用户或上层平台在所属 Namespace 创建 EgressForwardRule，声明协议、端口、入口类型和唯一固定公网目标；该组件不接受任意 CONNECT、SOCKS 或运行时动态目的地址。",
              "2. 安全校验：按协议校验字段和端口，规范化 FQDN，检查解析结果、目的地址白名单、规则冲突及资源所有权；地址不符合策略时拒绝更新，已生效规则因策略变化变得不安全时立即关闭该规则入口。",
              "3. 稳定入口：在规则所属 Namespace 创建 selectorless Service、Controller 自管 Endpoints 和受管 EndpointSlice。serviceType 默认为 ClusterIP；选择 LoadBalancer 时，平台 LB Controller 分配可路由 VIP。两类 Service 共用同一套健康代理 VM 后端管理。",
              "4. 配置生成：先从当前规则的受管 Service 确定稳定 targetPort，再把该规则的 nginx conf、ACL、已校验 upstream IP 和物料格式版本规范化，计算独立 desiredRevision；仅改变平台资源且不改变 VM 物料时不触发 reload。",
              "5. 配置发布：远端动作前持久化当前规则的发布门禁，逐 VM 执行 Validate、按需 Probe、该规则容量检查、Drain、Apply 和 Doctor。只摘除并恢复当前规则的后端；不同规则可独立推进，同一 VM 的共享 nginx 变更由 Builder 本地锁保护。",
              "6. 入口开放与状态：ClusterIP 取 Service 的 ClusterIP，LoadBalancer 则等待 status 中的单个 IPv4 VIP；VIP 尚未分配时继续调和，不发布空地址。OriginalHost+ClusterIP 在后端可用后发布 service-id，OriginalHost+LoadBalancer 则发布指向 VIP 的受管 DNS A 记录，最后回写 Configured/Available。"
            ],
            "第二条_VM配置发布链": [
              "1. 为当前 Rule 准备独立 conf、ACL、manifest 和 desiredRevision；任何远端发布动作前，先持久化对应 generation 的 desiredRevision 与 Configured=False/RollingOut，建立连续修改门禁。",
              "2. 对目标 VM 核验 SSH 身份、Builder/runtime 和共享基础状态，Validate 当前 Rule 候选物料；新增规则、目标更新、FQDN IP 变化或新增 VM 时执行适用协议的 Probe，普通 UDP 首版没有通用 Probe。",
              "3. 更新已有 Rule 前，确认摘掉当前 VM 后本 Rule 仍至少有一个可服务后端；容量不足就暂停当前 Rule，其他 Rule 独立推进。新规则没有存量流量，可以逐 VM 收敛后逐步开放。",
              "4. 只从当前 Rule 的 Endpoints/RS 摘掉这台 VM，并等待入口传播，避免新流量进入待更新 listener；这不等于所有存量连接已经结束，也不摘除其他健康 Rule 的后端。",
              "5. Builder 校验当前 Rule 的可信回滚物料，在 VM 本地短锁内切换该 Rule conf/ACL，执行 nginx -t、reload 和实际物料检查；失败只恢复该 Rule 的上一确认版本。",
              "6. 共享 Nginx/unatgw 等基础检查和当前 Rule 的 revision/listener/ACL 均通过后，把 VM 加回本 Rule 后端，再更新下一台；共享项不可信才整机隔离，单 Rule 失败不连带其他 Rule。",
              "7. Apply 失败或响应丢失时查询本 Rule current-revision 与 Doctor，目标版本可信且健康则继续，旧版本已恢复且健康则允许接回本 Rule；结果未知保持摘流并继续确认。当前 Rule 中途失败停止后续滚动，已成功 VM 保留新版，待本轮收敛或完成回滚后才允许下一次修改。"
            ],
            "第三条_真实请求数据链": [
              "1. 集群内 Pod 通常访问 Service ClusterIP；节点或集群外 Underlay VM 需要可路由入口时访问平台分配的 LB VIP。每条规则只选择一种 serviceType。",
              "2. ClusterIP 路径由集群 Service 转发面选择 Endpoints；LB 路径由平台内网负载均衡器把 VIP 流量导向同一份受管后端。",
              "3. 通过检查的代理 VM 上，Nginx listener 按固定目标规则把流量转发到已解析并校验的公网 IP 和端口。",
              "4. HTTP/TCP/UDP 代理流量通过代理 VM 的 unatgw 出公网；DNS 场景中，客户端先经 RegionDNS、DNS relay 和代理 VM 查询公网 resolver，拿到公网 IP 后，已具备 unatgw 的 Underlay 客户端直接通过自身出口访问业务目标。"
            ],
            "第四条_OriginalHost_YUM链": "用户继续访问原公网 rsync FQDN。ClusterIP 规则在后端就绪后发布 service-id，把原域名导向 Service ClusterIP；LoadBalancer 规则等待 VIP 后，在预先授权的 DNS zone 下发布指向 VIP 的叶子 A 记录。两条路径进入 Service 后都经健康代理 VM、Nginx 和 NATGW 到达公网仓库。",
            "失败后怎么处理": "某条规则在一台 VM 更新失败时停止该规则后续滚动，已经成功的 VM 保持新版本；失败 VM 恢复该规则的上一确认版本并通过 Doctor 后，可以按旧版本重新接流。该规则状态不明时保持该 VM 摘流，其他规则独立运行；只有共享 nginx 或 unatgw 等基础状态不可信才整机隔离。全部后端不可用时让访问在受管入口失败，VIP 变化则走受控 DNS 撤销与保留等待。",
            "个人贡献边界": "我主要实现 EgressForwardRule CRD、Controller、Service/Endpoints/EndpointSlice 管理，以及 VM 配置生成和滚动发布；Service/kube-proxy、代理 VM、Nginx 和 NATGW 均为复用能力。",
            "状态边界": "Available 表示当前规则至少有一个合格后端；LoadBalancer 还要求 VIP 已就绪。Configured 表示 Product 目标 VM 对该规则全部达到 desiredRevision、Doctor 通过，且相关平台资源已收敛；两者都是控制面观测，不能代替真实业务请求。"
          },
          "容器化YUM_OriginalHost_ClusterIP闭环": {
            "一句话主线": "原公网 FQDN 不变 → service-id 让集群侧把该域名导向受管 Service → Service 只选择健康代理 VM → Nginx 按固定目标转发 → NATGW SNAT 出公网。",
            "推荐口述版": "这条链路的目标是让容器化 YUM 用户只填原公网 rsync 域名，不改 URL。上层平台把这个域名生成一条 TCP+OriginalHost 规则；Controller 先校验 FQDN 和端口，创建 selectorless Service，把完整 listener、upstream 和 ACL 配置滚动发布到代理 VM，并只把 Doctor 通过的 VM 放进 Endpoints。至少一个后端可用后，Controller 才在 Service 上发布 service-id；仓外的平台服务发现消费这个契约，把原域名在集群侧解析到 Service ClusterIP。客户端流量随后按原域名 → Service → 健康代理 VM → Nginx → NATGW → 公网仓库前进。这样用户入口不变，但访问被收进统一的白名单、发布和健康隔离链路。",
            "记住这四步": ["用户入口不变", "Controller 先准备 Service 和 VM", "service-id 最后开放域名入口", "真实流量走 Service-VM-Nginx-NATGW"],
            "用户体验": "用户继续配置和使用原始公网 rsync FQDN/URL，不需要改成代理 Service 名、ClusterIP 或 VM 地址。准确说是集群侧服务发现接管该域名的解析结果，不是 Controller 修改 rsync URL、改 /etc/hosts 或拦截任意 DNS。",
            "规则契约": "上层平台生成 TCP + OriginalHost 的 EgressForwardRule，单一 upstream 必须是 FQDN，servicePort 必须等于 upstream.port；YUM/rsync 常见链路是原域名:873。端口必须相等，因为 service-id 能把目标地址映射到 Service，却不能改变客户端已经选择的目标端口。",
            "控制面闭环": "Controller 完成当前 TCP 规则的安全校验，独立生成并逐 VM 发布该 Rule 的 listener、upstream、ACL 和 revision；至少一个 VM 通过 Doctor 并进入受管后端后，再在 selectorless Service 上发布 infra.tce.io/service-id，host 仍是原公网域名。",
            "服务发现闭环": "仓库外的平台 service-id 消费器/RegionDNS 同步路径读取 Service 注解，把原公网域名在集群客户端路径解析到该 Service ClusterIP。Controller 只负责发布契约，不负责外部消费者实现。",
            "数据面闭环": "原域名:端口→Service ClusterIP:servicePort→kube-proxy/eBPF 按 Endpoints/EndpointSlice 选择代理 VM IP:targetPort→VM Nginx stream 转发到已解析并校验的公网 IP:upstream.port→Underlay NATGW 做 SNAT 后出公网。Service 负责稳定入口和后端选择，真正的七层/四层代理是 Nginx，公网出口是 unatgw。",
            "与DNSRecord的区别": "OriginalHost+ClusterIP 不创建 DNSRecord，它在 Service 上发布 service-id。OriginalHost+LoadBalancer 另行创建指向 VIP 的受管 A 记录；protocol=DNS 的 DNSRecord 表达 zone 转发，两种 DNSRecord 用途不同。",
            "初次发布与故障语义": "首次发布必须等 listener 和至少一个后端已经可以提供服务后才发 service-id，避免先把域名导向空 Service。已经发布后若代理后端临时全空，保留合法 service-id、清空 Endpoints，让访问在受管入口处直接失败；若删除注解，客户端可能重新解析到真实公网 IP，反而绕过代理和出口策略。",
            "漂移与删除": "OriginalHost+ClusterIP 使用独立的身份标识；service-id 被篡改或发生冲突时先关闭入口，再拒绝继续发布。一个原域名只能归属一条规则。删除时先清 Endpoints/Slice，再随 Service 删除撤销 service-id，最后摘 finalizer。",
            "生产验收": "ClusterIP 路径已从真实客户端验证原 FQDN 解析到受管 Service ClusterIP，并完成流量经代理 VM/NATGW 的真实 rsync 事务。"
          },
          "OriginalHost_LoadBalancer_VIP闭环": {
            "一句话主线": "平台 LB 先为 selectorless LoadBalancer Service 分配 VIP，Controller 再在已授权的 DNS zone 下发布指向该 VIP 的叶子 A 记录，客户端继续使用原域名。",
            "发布前提": "规则必须是 TCP+OriginalHost+LoadBalancer，每个 targetCluster 中恰好有一个覆盖该 FQDN、标记 approved 的外部权威 zone，不能是 forward zone，也不能由本 Controller 管理。同名或通配 A/AAAA/CNAME 冲突、VIP 与 upstream IP 重合都要拒绝；通过授权校验、后端可用且唯一 IPv4 VIP 稳定后，才发布绑定 Rule UID 与 Service UID 的 leaf A 记录，TTL 为 30 秒。",
            "VIP等待": "Service 已创建但尚未获得唯一可用 IPv4 VIP 时，规则保持 Available=False/WaitingForServiceAddress，不发布空入口，也不阻止其他 Rule 独立收敛。已经发布的 VIP 或 Service 身份发生变化时，进入受控撤销和等待流程。",
            "VIP变更": "VIP 或 Service UID 变化时，先清空本规则后端并撤销旧 A 记录，保留原 Service/VIP 至少 120 秒加该 VIP 曾观测到的最大 TTL，期间为 WaitingForDNSWithdrawal，完成后才能释放旧入口并发布新地址。RegionDNS 没有撤销 ACK，这个等待只是工程安全界，还依赖 LB 在等待期保留 VIP，并由现场验证实际传播时间。",
            "生产验收": "LoadBalancer 入口已完成生产数据面验收：Service 获得可路由 IPv4 VIP 后，Controller 发布受管 DNS A 记录并维护健康代理 VM 后端，真实客户端通过 VIP 完成目标协议事务。验收同时覆盖 VIP 分配等待、DNS 前置校验、VIP 变更发布以及 Endpoints 健康后端收敛。"
          },
          "implementation_details": {
            "为什么用无selector Service+受控Endpoints而不是ExternalName": "代理 VM 在集群外没有 Pod 对象，K8s 原生 selector 选不中它们。无 selector 的 ClusterIP 或 LoadBalancer Service 提供稳定入口，Controller 依据每台 VM 对当前 Rule 的配置版本和检查结果显式维护 Endpoints。ExternalName 只做 DNS CNAME，不参与转发，也无法按 Rule 的 Doctor 结果管理多个代理后端。",
            "EndpointSlice维护者": "selectorless Service 没有 Pod selector，内置 Controller 不会凭 VM label 自动发现集群外后端。公网代理 Controller 显式维护受管 Endpoints 和 EndpointSlice，给受管 Endpoints 标记 skip-mirror，避免自动镜像与主动维护冲突。Endpoints 和 EndpointSlice 的 controller ownerReference 都指向同 Namespace 的 Service，并保留 Rule UID 等受管元数据；共享环境和当前 Rule 的检查均通过，才发布该 VM 为本 Rule 后端。",
            "CRD设计": {
              "为什么需要Namespace级资源": "出口规则属于具体租户和业务，Namespace 承载资源归属、RBAC 和生命周期管理。Service 的 controller ownerReference 指向同 Namespace 的 Rule，Endpoints 与 EndpointSlice 都归属该 Service。代理 VM 池、监听端口池和跨 Namespace 的域名冲突仍需统一协调，但每条 Rule 独立生成并发布自己的配置。",
              "Service/Endpoints为何跟随Rule所属Namespace": "Service、Endpoints 和 EndpointSlice 都是 namespaced 资源，跟随 Rule 创建可以保留租户资源边界。Controller 校验 Namespace、Rule UID、Service UID、标签、注解和 ownerReference，不接管归属不匹配的同名对象。集群级 DNSRecord 不能通过跨 scope ownerReference 归属 Rule，因此用 managed-by 和 owner rule UID、Namespace、name 建立归属并显式清理。",
              "跨Namespace冲突范围": "ExplicitService 的 HTTP/TCP/UDP 入口身份包含 Namespace，不同租户可以使用相同目标；DNS zone 和 OriginalHost FQDN 会进入共享服务发现平面，仍按集群范围判断冲突。listener targetPort 则由 Controller 从集群共享端口池按 Rule UID 分配，直到所属 Service 确认删除后才释放。",
              "Upstreams强制min=max=1的原因": "spec.upstreams 只允许一个逻辑公网目标，使白名单、状态和失败归因都对应明确对象。一个 FQDN 可以解析出多个通过安全校验的 IP，并作为同一目标写入本 Rule 的配置；多个业务目标则分别创建多条规则。DNS upstream 必须是固定公网 resolver IP，HTTP/TCP/UDP 按各协议校验 FQDN 或 IP，OriginalHost 要求 FQDN。",
              "accessMode为何单独设计": "省略 accessMode 等价 ExplicitService，客户端直接访问生成的 Service；OriginalHost 仅给需要保留原公网域名的 TCP 客户端使用，ClusterIP 通过 service-id 发布原 FQDN，LoadBalancer 通过受管 A 记录发布 VIP，并要求单一 FQDN 和 servicePort 等于 upstream.port。protocol、accessMode、servicePort、serviceType 决定入口形态，禁止原地修改；dnsZone、httpHost 和 upstreams 可以在通过协议校验、冲突检查及同 Rule 发布门禁后更新，不能把目标域名一概说成不可变。",
              "字段校验如何分层": "API Server 的 OpenAPI Schema 校验基础类型、枚举和端口范围；支持 CEL 的版本可进一步校验组合和不可变字段。TCE Kubernetes 1.22 的交付物不携带 CEL，由 Validating Webhook 补充跨字段、不可变字段及连续更新门禁，Controller 再兜底目标规范化、地址安全和跨规则冲突。FQDN 统一转小写并去掉末尾点，只接受 ASCII 域名；非法解析结果不能进入新配置。"
            },
            "四类协议的共同点与差异": {
              "共同发布方式": "DNS/HTTP/TCP/UDP 共用规则校验、版本计算、逐 VM 发布、Doctor 和后端准入流程，但每条 Rule 都有自己的 conf、ACL、revision 和状态；更新一个协议的一条规则，不会合并重发其他规则。",
              "DNS": "DNS 规则的 Service 端口固定为 53，唯一 upstream 为公网 DNS Server IP:53，nginx stream 承担 TCP/UDP DNS 转发。固定 resolver IP 避免解析上游时依赖同一转发链，并把地址校验、ACL 和探测绑定到明确目标。DNS Probe 发送真实查询，收到包括 NXDOMAIN 在内的合法响应即可判为可达；不使用 httpHost，也不提供多 resolver 自动故障切换。",
              "HTTP": "HTTP 规则是固定目标反向代理，不是 HTTP CONNECT。客户端使用 status.entrypoints 返回的 Service 地址发送明文 HTTP，servicePort 省略时默认 80；本组件不为 HTTP 目标注册 CoreDNS 或 RegionDNS 解析。Nginx 连接已校验并写入当前 Rule 配置的公网 IP，以与 upstream host 规范化后一致的 httpHost 设置 Host、TLS SNI 和证书校验目标，并启用可信 CA 校验。需要客户端保持端到端 TLS 时使用显式授权的固定目标 TCP 规则。",
              "TCP/UDP": "都要求显式 servicePort。ExplicitService 可由 Service 端口映射到独立 upstream.port；OriginalHost 只支持 TCP，要求 FQDN 且 servicePort 等于 upstream.port，因为域名注册只替换目标地址，不能改变客户端选择的端口。普通 UDP 协议异构，首版不提供通用 UDP Probe，不能把配置检查说成上层协议可达验证。"
            },
            "Controller并发模型": "每条 Rule 使用独立 workqueue key、配置和状态调谐；全局策略或 Product 节点列表变化时将全部 Rule 重新入队，各自收敛，不使用 Controller 级全局串行锁。同一 VM 上只有一个共享 Nginx，Builder 在文件和 ACL 切换、nginx -t、reload 及失败恢复的短临界区内加本地锁。不同 Rule 可以独立推进，但这些外部动作不构成跨 VM 或 Kubernetes 资源的原子事务。",
            "一次Rule调谐的执行顺序": "先处理删除或安全停用，关闭本 Rule 入口；正常调谐时校验目标、冲突和资源归属，准备 Service 并读取已分配 targetPort，再生成本 Rule conf、ACL 和 desiredRevision。远端发布动作前持久化当前 generation 的 desiredRevision 及 Configured=False/RollingOut 门禁；随后逐 VM 预检、检查容量、摘流、Apply、Doctor 并维护本 Rule 后端。新规则在首台后端与入口就绪后发布所需域名入口，已有规则滚动完成后按需更新 DNSRecord，最后回写后端观测和 Conditions。",
            "分层抽象": {
              "RemoteBuilder": "Controller 到代理 VM 的受限远程操作边界，按 Rule UID 执行配置校验、apply、delete、current-revision 和 doctor，并提供 resolve、probe 与 list-rules 等只读能力。Builder 单次 apply 在 VM 本地保护当前 Rule 的物料切换和失败恢复；它不负责让多台 VM、Endpoints 和 DNSRecord 同时提交。",
              "forced-command安全边界": "生产环境通过 SSH forced-command wrapper 访问 VM：公钥认证成功后命令不直接交给 shell，而是由 authorized_keys 里的 command= 强制启动 wrapper，原始命令通过 SSH_ORIGINAL_COMMAND 传入，同时禁止端口/Agent/X11 转发和 TTY。wrapper 校验命令、参数、request ID、revision 和上传内容，只调用固定路径的 builder 程序，不执行任意 shell——这是当前 SSH 架构下防止私钥泄露升级为任意命令执行的安全边界。",
              "DataPlane": "当前 Rule 的发布计划：生成独立候选物料和 revision，安排目标 VM 预检、容量检查、逐台更新和恢复，不再维护全部规则共用的全局 revision。",
              "BackendWriter": "统一维护当前 Rule 的 Endpoints/RS 和 EndpointSlice 投影，执行摘流、恢复与重新加入；只有共享环境及当前 Rule 检查均通过的 VM 才能接流，不能因其他 Rule 成功而替它开放。"
            },
            "为什么改为每条Rule独立revision": "全局配置包使一条规则变化也重发全部规则，扩大摘流、校验和失败恢复的影响面。现在每条 Rule 独立生成完整 conf、ACL 和 manifest，desiredRevision 覆盖实际下发的规范化配置、targetPort、已校验 upstream IP 及物料格式版本；逐 VM 只更新当前 Rule。单 Rule 仍保留确定性版本、完整回滚物料和最终一致性盘点，因此局部更新不等于依赖易丢失的操作日志，也不承诺跨 VM 原子切换。",
            "确定性渲染": "先恢复当前 Rule 的 targetPort，再规范化域名、IP、CIDR 和协议参数，对去重后的地址及 ACL 条目采用稳定排序，避免直接用 Go map 遍历顺序输出物料。SHA-256 计算的是本 Rule 实际下发的规范化 conf、ACL、监听端口、upstream IP 和格式版本，相同物料必得相同 revision；仅影响 Service/DNSRecord 而未改变 VM 物料的字段更新，不触发无意义 reload。",
            "同一Rule连续修改怎么避免版本叠加": "尚未开始同步任何 VM 时允许修改，Controller 放弃未下发候选并按最新 generation 重算；首次 apply/doctor 结果尚不确定时暂拒修改；首次失败且没有任何 VM 使用新 revision 接流时允许用户修正。已有 VM 完成新版本检查并准备接流后，锁定本轮 Spec，直到全部收敛或回滚完成，避免旧、中、新三个版本叠加。Controller 在远端动作前先持久化 desiredRevision 和 RollingOut 门禁，Publish 返回后写成功或失败结果；Webhook 只限制当前 Rule，不阻止其他 Rule 更新。",
            "targetPort不可变": "targetPort 是本 Rule 在代理 VM 上的稳定 listener 端口，写入受管 Service 并作为端口占用的唯一事实源；Controller 重启后 list 全部受管 Service 重建分配表，同 Rule 生命周期内不换端口。Service、Endpoints 和 VM 物料必须使用同一分配结果。全局 listenerPortRange 与 Product 安装参数需使用同一现场值，当前不自动跨 Application 同步；Builder 对比候选端口池与 VM 已安装值，不一致拒绝 apply。端口池在组件生命周期内不可变，换池需要维护窗口关闭规则并重装数据面。",
            "单台VM滚动步骤": "先确认 SSH 身份、Builder/runtime 和共享环境，再对当前 Rule 执行 Validate 及按场景需要的 Probe；预检不通过，不触碰该 VM 的旧规则配置。更新已有 Rule 时，确保移除当前 VM 后本 Rule 仍至少有一个可服务后端，才从本 Rule 的 Endpoints/RS 摘流并等待入口传播；随后 Builder 在本地短锁内切换本 Rule conf/ACL、nginx -t 和 reload。共享项和本 Rule revision/listener/ACL 的 Doctor 均通过后才加回当前 Rule，再滚动下一台；固定等待不能证明全部存量连接已结束。",
            "容量下限与幂等恢复": "容量下限保护当前 Rule 的计划内滚动，摘掉当前 VM 会让本 Rule 无可服务后端时暂停；已明确不安全的后端仍需隔离。某 VM 发布失败就停止该 Rule 后续滚动，已成功 VM 保留新版本；失败 VM 恢复旧物料并通过 Doctor 后，可以旧 revision 加回本 Rule。Apply 失败或响应丢失时先查本 Rule current-revision 和 Doctor：目标版本可信且健康则继续，旧版本可信且健康则恢复本 Rule 后端，无法确认时保持摘流；回滚必须有经验证的旧物料锚点，不能仅凭旧 revision 字符串盲目执行。",
            "故障分级隔离": "接入预检确认 SSH 身份和 Builder/runtime；Doctor 的共享项检查 nginx 进程、基础 ACL 和 unatgw，明确不可信时从该 VM 的所有 Rule 后端摘除。规则项按 Rule UID 检查 revision、listener 和 ACL，仅本 Rule 不符合时只摘它自己的后端。暂时 SSH 观测失败与有效检查明确失败要分开处理，身份指纹不符不能静默接受。Doctor 默认每 60 秒巡检本地运行状态，不持续 Probe 公网 upstream，也不能证明真实业务请求成功。",
            "两类源地址范围分别管什么": "regionDNSSourceCIDRs 限制 DNS listener 实际看到的来源，serviceForwardSourceCIDRs 限制 HTTP/TCP/UDP 经 Service 或 LB 转发后在 VM 上呈现的来源；两者是 listener 入向 ACL，应按现场真实转发源配置，不是公网目标白名单，也不是 Kubernetes loadBalancerSourceRanges。allowedEgressCIDRs 才限制 upstream 目的地址；来源与目的范围目前都按全局策略配置，不能把 Namespace/RBAC 当成 VM 报文级隔离。",
            "规则删除的finalizer流程": "先清空本 Rule 的 Endpoints/RS 和 EndpointSlice 后端并等待入口传播，再删除其拥有的 DNSRecord，逐 VM 按 Rule UID 删除 conf、ACL 和 manifest，并在本地锁内校验、reload 和检查；全部必要清理完成后再删 Service/Endpoints 并移除 finalizer。OriginalHost+ClusterIP 的 service-id 随 Service 清理；OriginalHost+LoadBalancer 撤销 A 记录后须保留 Service/VIP 至少 120 秒加该 VIP 曾观测的最大 TTL，等待结束才释放。远端删除失败保留 finalizer 重试。安全校验失败则立即关闭本 Rule 入口并清理物料，CR 保留供修正，不当作用户删除。",
            "落地场景举例": [
              "消息网关短信/邮件：以前需要人工建代理机手动维护 nginx 配置，现在运维创建 HTTP/TCP 规则声明目标域名和端口，Controller 建独立 Service 和独立 nginx listener，短信/邮件 Pod 直接访问该规则专属 Service。",
              "容器化 YUM rsync：上层把用户的原公网 FQDN/873 转成 TCP+OriginalHost 规则；ClusterIP 用 service-id 导向 Service，LoadBalancer 用受管 A 记录导向 VIP，用户 URL 均不变，后续共用健康代理 VM、Nginx 和 unatgw 出公网链路。"
            ],
            "失败恢复需要保留哪些物料": "每条 Rule 至少保留当前和上一已确认成功的不可变 revision 目录，目录内包含 conf、ACL plan 和 manifest，不额外维护 .bak。更新前读回旧物料并校验摘要及 current 指向，一致时才能作为可信回滚锚点；候选 conf/manifest 写入后逐项读回比较，ACL 激活后与实际 nftables 规则结构化比对。只有运行检查确认后提交 appliedRevision，失败恢复当前 Rule，不能改写其他 Rule 的物料。",
            "同revision为什么还要检查和修复": "revision 相同只说明版本标识一致，不证明文件、ACL 或进程状态没有漂移。Controller 仍查询当前 Rule 的 current-revision 和 Doctor；物料及运行检查均通过时只修复 Service、Endpoints/RS、DNSRecord 等平台资源，不重复 reload；同 revision 但运行状态异常时先关闭本 Rule 在该 VM 的流量，再幂等重放并检查后恢复。",
            "逐Rule更新后如何发现历史漂移和孤儿配置": "Controller 周期读取 Kubernetes Rule 集合，并通过 Builder list-rules 盘点 VM 上所有受管 manifest 和实际配置。缺失物料补发，revision 或实际摘要不符时修复，已无对应 Rule 的物料按安全删除流程清理；Rule 文件名使用 r_<ruleUID>.conf，manifest 记录 Namespace、name 和 UID，避免同名 Rule 删除重建后误认旧配置。增量写入缩小变更范围，周期全量核对保证最终一致性。",
            "状态字段怎样表达可用性和收敛": "每条 Rule 用 Available 表达当前是否有合格后端且入口就绪，LoadBalancer 还要求 VIP 可用；Configured 要求 Product 节点列表非空、全部目标 VM 的本 Rule appliedRevision 达到 desiredRevision、Doctor 通过且相关 Kubernetes 资源收敛。observedGeneration 表示处理到的 Spec，appliedGeneration 在该代完全收敛后推进。backendStatuses 按 IP 记录本 Rule appliedRevision、serving、lastStep、reason 和 message，不另建 phase；状态只是观测结果，恢复仍需查询 VM 真实物料及检查结果，Available 也不能替代业务 E2E。",
            "Product节点列表和SSH身份分别从哪里恢复": "代理节点成员的唯一来源是 TAD 根据 Product Application Component nodes 渲染的 tce/product-tcs-egressproxy-config 中 proxy-nodes.conf，Controller 只读且不经 local.json 中转。它表示交付声明，不证明 Product 已安装；新 VM 要通过 SSH 身份、Builder/Nginx、端口池、unatgw 及本 Rule apply/doctor 后才接流。tce/egress-proxy-endpoints 只保存 IP 到 SSH hostKey 的 TOFU 固定结果，不决定成员。节点配置缺失或为空时 Controller 继续运行、后端为空并报告 WaitingForProxyEndpoints；配置恢复后各 Rule 重新入队收敛。",
            "代理VM扩缩容和Product升级怎么配合": "新增 IP 先进入 Product 节点列表，Controller 将全部 Rule 入队，待 Product 安装完成后逐 Rule 补齐新 VM 配置并检查加回；新节点失败不触发已有健康 VM 重发或摘流。缩容或升级先从 Product 列表移除 IP，Controller 从所有 Rule 后端摘除，集群侧交付任务只读确认摘流和传播后才远端卸载、升级或销毁。Product 负责 Compose 常驻 Nginx 的启动与安装，Controller 只通过 Builder 增删规则和 reload；runtime 升级还需同步宿主 Controller 期望版本，升级 VM 恢复原有全部规则后再滚动下一台。"
          },
          "高压追问首答": {
            "所有公网流量都会经过这个组件吗": "不会。HTTP/TCP/UDP 的受管代理请求要进入对应规则的 Service，再由代理 VM 转发；DNS 规则只补齐解析链路，客户端先问 RegionDNS，命中白名单 zone 后才经 DNS relay 到公网 resolver，已具备 unatgw 的 Underlay 业务拿到公网 IP 后从自身出口访问目标。组件统一的是这些场景的规则管理，不透明接管整个集群的所有公网流量。",
            "这个组件的直接用户是谁": "平台最终服务企业客户，但这个组件的直接使用方主要是平台运维人员，以及运行在 TCS 集群里的上层云产品和业务组件。例如 YUM、短信、邮件等业务需要访问固定公网依赖时，由授权方创建规则，业务 Pod 再通过对应的受管入口访问。",
            "为什么企业私有云需要受控公网出口": "私有云不能因为少量外部依赖就给所有工作负载开放任意公网，但 YUM、短信、邮件等业务确实需要访问固定目标。受控公网出口是在保留网络隔离的前提下，为这些必要依赖提供明确入口和固定白名单，同时把规则变更、代理配置和故障处理统一管理起来。",
            "为什么只支持固定目标_不做任意公网代理": "固定目标符合这类业务的最小授权需求：一个规则对应一个公网依赖，审批、配置、健康和审计都有明确对象。通用 CONNECT 或 SOCKS 允许客户端临时选择任意目标，会扩大出口权限，也让安全校验和故障归因更困难，因此不属于这个组件的定位。",
            "这个项目的业务价值怎么衡量_有没有数据": "主要看交付效率和代理资源复用。以前按文档手工配置代理、解析和转发规则，单次配置大约需要 4 小时，自动交付后降到 10 分钟以内；以前每个需求往往单独新增 KubeVM，现在多条规则共享 Proxy VM，相比原方案减少约 70% 的代理机资源使用。另外，统一管理和滚动发布减少了配置不一致导致的访问异常，来源准入与目标约束控制出口范围，新 VM 也能自动同步现有规则。两个数字分别对应配置耗时和代理资源，其他收益不附加没有统计过的提升比例。",
            "客户需要访问任意公网地址怎么办": "这超出了固定目标出口组件的范围，需要走单独的网络与安全评审，选择通用代理、独立网关或其他受控上网方案。不能为了满足任意地址访问而把 EgressForwardRule 放宽成动态目标，否则会破坏这套能力的最小授权和审计边界。",
            "Pod访问GitHub会自动经过代理吗": "不会，需要先配置规则，而且应用必须进入受管入口。例如要允许访问 github.com:443，先创建对应的 EgressForwardRule，声明目标、协议和端口。Controller 完成目标校验、Service 创建和代理 VM 配置发布后，业务通过这个入口访问，流量才会经过代理。",
            "DNS请求也必须经过Service吗": "本组件纳管的 DNS 查询先到 RegionDNS，只有命中显式 dnsZone 的请求才转给该规则的 DNS relay Service，再经代理 VM nginx stream 到一个公网 resolver。普通客户端不能直接访问 relay，否则会绕过 zone 白名单；限制依据是代理 VM 实际看到的 RegionDNS 转发来源。它不转发所有 DNS 查询，也不替换原有内网解析。",
            "没有代理VM_Pod就不能出公网吗": "对我们纳管的交付场景来说，这些 Pod 没有获准的默认公网路径，需要借助已经准备好 unatgw 的代理 VM 访问公网。不过这是部署环境的网络前提，不是这个 Controller 单独实现的阻断能力；如果要求强制所有相关流量必须经过代理，还需要路由、防火墙或 NetworkPolicy 等外部网络策略配合。",
            "代理VM在集群内还是集群外_可以跨集群共享吗": "代理 VM 位于集群外的 Underlay 网络，但由当前集群的公网代理 Controller 纳管。Controller 为本集群的规则创建 Service，并把对应 VM 写入后端。当前可以把它定义为集群级控制面加集群外代理 VM 池；源码没有提供多个 Controller 安全共享同一 VM 池的协议，所以我不会把它扩展成跨集群共享架构。",
            "正向代理和反向代理有什么区别": "正向代理代表客户端访问外部服务，服务端主要看到代理；反向代理代表后端服务接收客户端请求，客户端不需要知道真实后端。判断重点是代理代表哪一侧，而不只是客户端是否知道代理存在。",
            "先简单介绍一下项目背景和原方案问题": "比如业务要接入一个短信公网接口，原来要找运维准备代理 VM，再逐台修改 Nginx 配置，后续修改和删除规则也需要人工处理。我的组件把它变成提交一条出口规则，由 Controller 自动准备入口、发布配置并维护健康后端。底层继续使用已有的代理 VM 和公网 NAT 出口。",
            "原来三类出公网链路分别是什么_为什么要统一": "一类是已有 unatgw 出口、但缺少公网解析的 COS-CGI、部分 NTP 和虚拟机 YUM 等 Underlay 业务；一类是要人工创建代理机、维护 nginx 的短信邮件业务；还有一类是容器化 YUM 自己维护的 rsync 代理和专属 VM。统一规则声明、交付和维护，可以共用代理 VM 并减少人工配置；DNS 场景仍保留客户端自己的出网路径。另有 Pod 内访问公网一级 NTP 源的拨测需求，它还缺业务出网链路，不能归入只补 DNS 的场景。",
            "这是不是通用ForwardProxy": "不是。EgressForwardRule 创建时已经固定目标、协议和端口，Controller 会把目标解析并校验成确定 IP，Nginx 只能转发到配置中的 upstream。HTTP 模式是固定 httpHost 的反向代理，不接受 CONNECT；TCP/UDP 也是一条规则一个固定目标，不支持 SOCKS 或由客户端运行时选择任意公网地址。所以它是显式授权的出口控制面，不是通用 Forward Proxy。",
            "为什么代理放VM不放Pod": "我们这个交付环境已有独立 Proxy VM 到 NATGW 的可用出口，而 Underlay 出网组件会修改主机网卡和默认路由，直接装在 TCE 工作节点上会与底座网络管理冲突。所以让代理 VM 承担出网数据面，Pod 只访问受管入口，我负责把代理规则和发布过程自动化。这是当前环境的部署约束，不能推广成 Kubernetes Pod 天生无法出公网。",
            "unatgw已经能出公网_为什么还需要这个组件": "代理 VM 已经能出公网，但业务接入一个新目标仍要人工改配置，多台 VM 也难保证一致。我解决的是这段人工维护：把目标声明成规则，自动生成访问入口、逐台发布配置并管理健康后端。NATGW 继续负责底层出公网。",
            "为什么用CRD不用ConfigMap": "因为这里管理的不是一份静态 Nginx 文本，而是一类有完整生命周期的业务资源。每条规则都有协议、目标、端口、状态、权限和删除流程，还要支持失败重试与 Controller 重启恢复。使用 ConfigMap 最终仍要自行实现校验、状态管理、多规则合并、清理和幂等；CRD 则能直接使用 generation、finalizer、RBAC 和 status 表达这些语义。",
            "EgressForwardRule为什么需要归属Namespace": "出口规则属于具体租户和业务，Namespace 承载归属、RBAC 和生命周期。Service 以 ownerReference 归属同 Namespace 的 Rule，Endpoints 与 EndpointSlice 都归属 Service；集群级 DNSRecord 通过 managed-by 和 Rule UID/Namespace/name 元数据关联。代理 VM 和 listener 端口池是共享资源，仍要跨 Namespace 检查冲突，但每条规则独立生成与发布配置。",
            "Namespace级CR能否保证租户流量隔离": "Namespace 和 RBAC 限制谁能管理规则，不能自动保证报文隔离。当前 listener 使用全局来源 ACL：DNS 限制为实际 RegionDNS 转发来源，HTTP/TCP/UDP 限制为 Service/LB 转发来源；没有每条规则独立的客户端来源字段。开放一个 zone 后，targetClusters 中其他客户端也可能通过 RegionDNS 查询它。需要严格租户流量隔离时，还要结合实际入口、身份和网络策略另行落实。",
            "多个Namespace的规则怎么汇总和判定冲突": "Controller 在集群范围检查共享名称和端口冲突，但每条规则独立调和，不把全部规则合并成一个配置包。ExplicitService 的 HTTP/TCP/UDP 入口位于各自 Namespace；DNS zone 和 OriginalHost FQDN 涉及共享服务发现，需要跨 Namespace 检查。VM listener 端口来自共享池，分配结果以受管 Service 的 targetPort 持久化；全局策略或节点变化时，把各规则分别重新入队。",
            "LB_VIP还没分配或发生变化怎么处理": "新建 LoadBalancer Service 尚无唯一可用 IPv4 VIP 时，规则保持 Available=False/WaitingForServiceAddress，不发布空入口。OriginalHost 的 A 记录绑定 Service UID 与 VIP；VIP 或 Service 身份变化时先关闭本规则后端、撤销旧记录，保留 Service/VIP 至少 120 秒加该 VIP 曾观测的最大 TTL，再按新的入口重新发布。等待不是 RegionDNS 已撤销的确认，仍要满足 LB 保留 VIP 的前提。",
            "observedGeneration_appliedGeneration_revision分别表达什么": "observedGeneration 是 Controller 已处理的 Spec 代数，appliedGeneration 只在当前规则的数据面与平台资源完全收敛、Configured=True 后推进。desiredRevision 是这条规则实际下发物料的摘要，不是全局版本，也不直接对整份 Spec 求哈希。backendStatuses 按 VM IP 记录该规则的 appliedRevision、serving、lastStep 和原因，只是最近观测；恢复时仍要查询 VM 的 current-revision、实际物料和 Doctor。",
            "项目整体链路怎么讲_控制面和数据面怎么划分": "一条请求从业务 Pod 出发，访问 Service 后，被转发到代理 VM 上的 Nginx，再经 NATGW 到达固定公网目标。Controller 负责让这条路径可用：用户提交 EgressForwardRule 后，它创建 Service、把配置发布到代理 VM，并维护合格后端。发布新版本时先检查容量、逐台摘流，Apply 后通过检查才重新接流；Controller 本身不转发业务报文。",
            "Controller会不会并发跑两轮reconcile": "同一 Rule 以自己的 workqueue key 顺序调和，不同 Rule 可以独立推进。全局策略或 Product 节点变化时，把全部 Rule 入队分别收敛，不再依赖 global key 加 single worker 串行发布所有规则。多个 Rule 落到同一 VM 时，Builder 只在配置和 ACL 切换、nginx -t、reload 及失败恢复期间持有本地锁，保护共享 nginx；这不等于多 VM 原子事务。",
            "为什么selectorlessService": "代理 VM 没有 Pod selector，所以 ClusterIP 和 LoadBalancer 两类 Service 都不设 selector。Controller 显式把通过配置版本和 Doctor 检查的外部 VM 写入 Endpoints/EndpointSlice；Service 只提供入口和后端选择，不执行公网代理。",
            "Service_Endpoints_EndpointSlice分别负责什么": "Service 提供 ClusterIP 或平台 LB VIP 和端口，本身不发现 VM，也不做 Doctor 健康检查。Controller 自管的 Endpoints 保存可以接收这条规则流量的 VM IP:targetPort；EndpointSlice 把这份后端列表提供给集群转发组件。Controller 标记 Endpoints 跳过自动镜像并自己维护 EndpointSlice，避免两个 Controller 同时改写。",
            "为什么Endpoints是权威而EndpointSlice只是投影": "如果 Controller 分别把 Endpoints 和 EndpointSlice 当作两个可独立修改的真相，就可能出现两边后端集合不一致，不同转发组件观察到不同结果。当前 Controller 先从受管 Endpoints 恢复某条规则正在服务的 VM，所有摘流、恢复和重新加入都先修改 Endpoints，再同步 EndpointSlice；EndpointSlice 同步失败时不会把本轮当成完整成功。因此 Endpoints 是流量准入的权威状态，EndpointSlice 只是向 Kubernetes 转发面发布同一后端集合的投影，不能独立增加 VM。",
            "为什么不用ExternalName_LB_VIP解决什么": "ExternalName 只发布 CNAME，无法用受控 Endpoints 按 VM 健康状态快速摘流。ClusterIP 适合集群内 Pod；节点和集群外 Underlay VM 需要可路由入口时，规则选择 LoadBalancer Service，由平台内网 LB 分配 VIP，两种类型仍共用 Controller 维护的健康代理 VM 后端。",
            "三个端口分别是什么": "servicePort 是客户端访问 Service 的端口，targetPort 是 Controller 分配给代理 VM listener 的内部端口，upstream.port 是公网目标端口。ExplicitService 可做 servicePort→targetPort→upstream.port 映射；OriginalHost 要求 servicePort 等于 upstream.port。",
            "ExplicitService和OriginalHost区别": "ExplicitService 是默认模式，客户端直接访问生成的 ClusterIP 或 LB VIP；OriginalHost 只用于 TCP，客户端继续访问原公网 FQDN。OriginalHost+ClusterIP 用 service-id 导向 Service，OriginalHost+LoadBalancer 则用受管 DNS A 记录导向 VIP。",
            "YUM用户URL改了吗": "没有。用户仍使用原 rsync URL，ClusterIP 路径通过 service-id 把 FQDN 导向 Service ClusterIP，LoadBalancer 路径通过受管 A 记录把 FQDN 导向 VIP。Controller 不修改 URL，Service 也不解析公网域名。",
            "用户是否真的只填域名": "是产品层体验：用户只配置原公网域名，生产平台把它转换为 TCP+OriginalHost 的 EgressForwardRule，并补齐相等的 servicePort/upstream.port；Controller 消费规范化后的规则，不在调和器内部耦合页面或 MirrorSyncJob。",
            "service-id和DNSRecord什么关系": "OriginalHost+ClusterIP 在 Service 上发布 service-id，让平台服务发现把原域名导向 ClusterIP；OriginalHost+LoadBalancer 不发布 service-id，而是在已授权 zone 下发布指向 VIP 的叶子 A 记录。protocol=DNS 的 DNSRecord 用于 zone 转发，是另一种资源语义。",
            "为什么OriginalHost只做TCP": "生产需求是 rsync 的 TCP 原域名兼容，当前 OriginalHost 契约也按 TCP 定义。HTTP 还有 Host、SNI 和证书语义，UDP 缺少连接态，它们继续走显式 Service 模式。",
            "为什么端口必须相等": "服务发现能把域名对应的目标地址替换成 Service ClusterIP，但客户端已根据 URL 选择了目标端口；如果 servicePort 不同，没有额外端口重定向层就到不了 Service 正确入口。",
            "为什么最后才发布OriginalHost域名入口": "先确认 VM listener、ACL、配置版本和健康检查正常，并且至少一个 Endpoint 可用，再发布 ClusterIP service-id 或 LB VIP A 记录。这样可以避免原域名已经指向新入口，但后端尚未就绪。",
            "后端全挂为何不立即撤销域名映射": "立即撤销 service-id 或 A 记录可能让原 FQDN 回退到真实公网解析，客户端反而绕过代理。保留已验证的域名映射并清空 Endpoints，会让请求在受管入口失败；VIP 或 Service UID 确定变化时，再走受控撤销和 TTL 等待。",
            "怎么证明域名入口已生效": "ClusterIP 路径要同时看 Service 上的 service-id、外部消费者的观测状态和真实客户端解析/rsync；LoadBalancer 路径要看 Service VIP、与 Service UID 绑定的受管 A 记录和真实客户端流量。只看 annotation、Status 或 DNSRecord 都只能证明控制面已发布。",
            "怎么防DNS重绑": "由 Controller 组织解析并校验每个返回 IP，确保满足公网地址和 allowedEgressCIDRs 策略，再把同一组已校验 IP 写入当前 Rule 的 manifest、Probe 和 nginx 配置。Builder 的 Probe 不重新解析 FQDN，nginx 连接固定 IP，避免校验对象与实际连接对象不同。发现 FQDN IP 变化时重新校验、探测并发布；新结果不安全时关闭本规则入口并清理配置。",
            "Probe和Doctor能证明什么": "Validate 检查当前 Rule 的候选 conf、ACL、manifest、revision 和允许变更范围。Probe 在新增规则、upstream 更新、FQDN IP 变化或新增 VM 时执行：HTTP/TCP 只对已校验 IP 做 TCP 建连，DNS 发真实查询并接受 NXDOMAIN 等合法响应，首版不提供通用 UDP Probe；同一 FQDN 多个 IP 逐个记录，至少一个可达才通过。Doctor 检查共享 nginx/unatgw 基础状态和该规则的 revision、listener、ACL，默认 60 秒巡检不重复 Probe。它们都不能证明 TLS、业务协议成功或流量一定经过指定 unatgw 路径。",
            "代理VM健康为什么不能简化成一个布尔值": "同一台 VM 可以让一条规则正常、另一条规则失败。SSH host key、Builder/runtime 是接入预检；Doctor 的共享项检查 nginx、基础 ACL 和 unatgw，规则项分别检查当前 Rule 的 revision、listener 和 ACL。共享状态明确不可信时整机隔离，只有单 Rule 失败时只摘该规则后端；Probe 另外验证发布前的 upstream 可达性，真实客户端请求才验证端到端效果。",
            "为什么一条规则只允许一个upstream": "这样白名单、解析、Probe、状态和审计都对应一个明确逻辑目标，revision 也可确定。一个 FQDN 可以解析出多个经校验的 IP，但多个业务目标用多条规则分别表达，避免在一条规则里引入模糊的部分健康和选路语义。",
            "为什么采用每条Rule独立revision": "一条规则变化只发布自己的 conf、ACL 和 revision，可以缩小预检、摘流及回滚影响，其他规则独立收敛。revision 仍覆盖该 Rule 实际下发的规范化完整物料，保证重试与恢复有明确版本；Builder 在本地短锁内保护共享 nginx，并保留当前和上一成功目录。仅影响 Service/DNSRecord 的变化不必 reload，周期全量盘点负责发现增量发布期间的缺失、漂移和孤儿物料。",
            "跨VM能原子吗": "不能。一条 Rule 在多个 VM 上逐台更新，可以短暂存在新旧 revision；Builder 的本地锁和失败恢复只保护单 VM 上当前 Rule 的变更。Controller 预检、检查该规则剩余容量、摘流、Apply、Doctor 后再加回；中途失败时停止该 Rule 后续滚动，已成功 VM 保留新版，失败 VM 经可信回滚和检查后可用旧版服务，之后继续收敛或完成回滚。",
            "HTTP为什么不支持CONNECT_HTTPS怎么处理": "HTTP 模式面向固定目标反向代理，客户端显式使用 status.entrypoints 返回的 Service 地址，servicePort 省略时默认 80；本组件不为 HTTP 目标创建 CoreDNS 或 RegionDNS 记录。Nginx 连接已校验的公网 IP，以规范化后与 upstream host 一致的 httpHost 设置 Host、SNI 和证书校验目标，并启用可信 CA 校验。目标在规则中固定，不由客户端 CONNECT 请求指定；需要端到端 TLS 时可使用显式授权的固定目标 TCP 字节流转发。",
            "Available和Configured有什么区别": "Available=True 表示本规则至少有一台通过 Doctor 的 VM 在受管后端中，LoadBalancer 还要求 VIP 就绪；滚动时可以继续由旧 revision 服务。Configured=True 要求 Product 节点列表非空、全部目标 VM 对本规则达到 desiredRevision 且 Doctor 通过，相关 Service、后端和所需 DNSRecord 也已收敛。所以可以 Available=True、Configured=False，单台失败不一定导致业务完全不可用；这些状态仍不能代替真实请求验证。",
            "当前支持IPv6或双栈吗": "当前生产 V1 按 IPv4 链路设计，公网 upstream 校验会拒绝 IPv6。生产口径只承诺 IPv4 客户端入口、IPv4 代理 VM、IPv4 NATGW 和 IPv4 upstream 的闭环；不能因为 Service 或 EndpointSlice 某一层能够表示 IPv6，就外推出端到端双栈可用。",
            "只剩最后一台怎么办": "更新已有规则前，检查移除当前 VM 后本规则是否仍至少有一台合格后端；如果会归零，就暂停这条规则的滚动，保留现有流量。新增规则没有旧流量，可以从第一台通过检查的 VM 开始开放，LB 入口另需等待 VIP。容量保护只约束计划内摘流，明确不安全的后端仍必须隔离。",
            "Apply超时会不会盲目回滚": "不会。先按 Rule UID 查询 current-revision 和 Doctor，核对该规则的真实 conf、manifest、ACL 与 listener；目标版本已生效且健康就继续，旧版本可信且健康就恢复该规则后端。需要回滚时，必须有更新前已核验的上一成功 revision 目录，包含完整 conf、ACL plan 和 manifest，不能只凭旧 revision 字符串恢复。远端结果或回滚物料不可信时保持该规则摘流，后续再确认。",
            "为什么恢复时CurrentRevision和Doctor两个都要看": "current-revision 用来定位当前 Rule 的版本，Doctor 再检查该版本的 listener、ACL 及共享运行状态。相同 revision 也可能发生实际物料漂移，因此还要校验本地 conf/manifest 内容、实际 nftables 规则和 current 指向；目标版本健康就按成功处理，旧版本可信且健康就恢复原后端。上一成功版本的完整目录才是回滚依据，无法确认时保持该规则摘流。",
            "摘流等待能证明连接已经清零吗": "不能。先从当前 Rule 的 Endpoints/RS 摘除 VM，再等待入口变更传播，目的是停止该规则的新流量；已有 TCP 长连接或 UDP 会话可能继续存在。传播等待和 nginx reload 都不等于所有连接已经结束，因此普通滚动不能仅凭固定等待承诺零中断，实际影响还要结合协议、连接时长和现场转发行为验证。",
            "发布门禁和VM本地锁是不是事务": "不是。发布门禁限制同一 Rule 连续修改，避免当前远端操作未结束时叠加新版本；VM 本地锁防止多个 Rule 同时切换文件和 reload。它们分别保护版本推进与共享进程操作，不能撤销其他 VM 已完成的发布，也不能让 Endpoints、DNSRecord 和多台 VM 原子生效；部分失败仍靠逐 Rule 状态核对、回滚和后续调和恢复。",
            "Controller重启如何恢复": "重读 Rule、Service、后端、Product 节点列表和 host key 记录，从全部受管 Service 的 targetPort 重建端口占用，再逐 Rule 查询各 VM 的 current-revision 和 Doctor，核对真实物料后继续收敛。backendStatuses 是最近观测，不能单独决定 VM 是否接流；发布门禁也要结合远端事实判断。周期盘点再用 VM 上的受管 manifest 或 list-rules 与 Kubernetes Rule 集合对账，修复缺失和漂移，按安全删除流程清理孤儿物料。",
            "DNS_HTTP_TCP_UDP统一了什么又有哪些不能统一": "统一的是每条规则的校验、独立 conf/ACL/revision、发布、健康后端和删除流程，各规则使用相同管理方式但有自己的配置与版本。HTTP 需要主动设置 Host、SNI 和证书校验，且不注册目标域名解析；TCP/UDP 按固定目标转发字节流或报文，首版没有通用 UDP Probe；DNS 通过 RegionDNS 白名单分流，同时提供 TCP/UDP 53 relay，每条规则只有一个公网 resolver，不提供多 resolver 自动故障转移。",
            "一次SSH失败就摘VM吗": "不是。暂时联系不上 VM 并不等于它已经不健康，周期检查会先保留此前健康的后端，避免控制链瞬断造成全量抖动；只有 host-key mismatch、身份漂移或有效 Doctor 明确失败时才摘流。",
            "规则故障会拖垮整台VM吗": "当前 Rule 的 revision、listener 或 ACL 不一致，只从这条规则的 Endpoints/RS 摘除该 VM，修复和回滚也只改这条规则的物料。共享 nginx、基础 ACL 或 unatgw 明确不可信时才整机隔离。不同规则虽然独立发布，仍共用 nginx 进程，因此必须由 Builder 本地锁保护共享配置切换和 reload，不能宣称它们具有完全独立的进程故障域。",
            "删除为何要finalizer": "必须先关入口，再清 VM listener/ACL，最后删资源；否则 CR 消失后，Controller 无法证明远端配置和 service-id/DNSRecord 已经撤销。只有 VM 清单明确为空时，才允许跳过远端清理。",
            "SSH下发安全吗": "私钥放 Secret，校验固定 host key；VM 上专用低权限用户只能 sudo 固定 wrapper。wrapper 是命令 allow-list，严格校验 request ID/revision/tar 路径和大小，清理环境并只调用固定 Builder，不把 SSH 变成任意 root shell。",
            "Product_Controller_TAD怎么分工": "Product 在已准备且已有 unatgw 的 Underlay VM 安装 nginx、Builder、wrapper、CA 和 ACL，并通过 Compose 启动常驻 nginx；Controller 只发布规则和 reload，不负责启停进程。TAD 从 Product Component nodes 生成 product-tcs-egressproxy-config 中的 proxy-nodes.conf，这是目标节点成员的唯一来源，Controller 只读；egress-proxy-endpoints 另存 IP 到 SSH host key 的固定结果，不决定成员资格。新 IP 出现在列表里不等于已安装就绪，还要逐 Rule 验证后才接流。",
            "你个人到底做了什么": "我主要实现 EgressForwardRule CRD、Controller、Service/Endpoints/EndpointSlice 管理，以及代理 VM 配置生成和滚动发布。Kubernetes Service/kube-proxy、代理 VM、Nginx 和底层 NATGW 都是复用现网已有能力。",
            "现在算上线了吗": "项目原有 Controller、Product/Builder、平台服务发现和真实协议链路已投入生产；逐 Rule 发布改造是当前架构方向，这轮改造的实现与上线状态待确认。",
            "公网代理讲得有点散_能重新解释解决什么吗": "比如一个业务要访问短信接口，原来需要运维逐台修改代理 VM 的 Nginx 配置，机器多了容易漏改。我把它变成提交一条规则，系统自动生成访问入口、发布配置，并把不符合接流条件的后端摘掉。业务通过这个入口访问原有公网出口，后续变更也走同一套自动流程。",
            "这是入站还是出站_Service后面究竟是Pod还是VM": "这里是业务 Pod 主动访问固定公网目标的出站请求，Service 后端是独立代理 VM。这个 Service 不设 selector，Controller 显式维护合格 VM 的地址和端口；Service 转发面把请求送到 VM 上的 Nginx，再由它经 NATGW 访问目标。",
            "场景是Overlay还是Underlay_为什么VM能出公网而Pod不能": "这里明确的是 Proxy VM 处在 Underlay 网络，并已经配置了到 NATGW 的公网出口；业务 Pod 使用集群内可达的 Service 入口访问它。Pod 能否直接出公网，还取决于 CNI、路由、SNAT 和访问策略，不能仅凭 Overlay 或 Underlay 判断。我们的部署约束是不能直接把修改主机网卡和默认路由的出网组件装到 TCE 工作节点，所以复用独立代理 VM 的出口。",
            "同一条Rule连续修改怎么避免三个版本叠加": "尚未开始远端同步时允许修改，Controller 按最新 generation 重算；第一次 apply/doctor 结果未知时暂时拒绝。首次发布已确认失败、且没有任何 VM 以新版本接流时允许修正；已有 VM 完成新版本检查并准备接流后，锁定当前 Spec，直到整条规则收敛或回滚完成。Controller 在远端动作前先持久化 desiredRevision 和 Configured=False/RollingOut 门禁，Webhook 只限制这一条 Rule，不阻止其他规则更新。",
            "revision相同还需要做什么": "先查当前 Rule 的 current-revision 和 Doctor，并核对真实 conf、manifest、ACL 与 listener。版本相同且健康时只修复 Service、后端或 DNSRecord，不重复 reload；版本相同但运行状态异常时，先从本规则摘除异常 VM，再幂等重放同一 revision。独立增量发布仍要周期对照全部 Rule 与 VM 受管物料，发现缺失、漂移和孤儿配置。",
            "代理VM扩缩容怎么避免影响已有流量": "Product 节点列表变化后，各 Rule 分别重新入队。新 VM 先完成安装和身份、runtime、端口池、unatgw 检查，再逐 Rule validate/probe/apply/doctor，通过哪条规则才加入哪条后端；新 VM 失败不让已有 VM 重新摘流或发布。缩容或升级时先从 Product 列表移除 IP，Controller 从所有受管后端摘除，集群侧交付任务确认并等待传播后才卸载或升级。代理 VM 不持有 Kubernetes 凭证；升级 runtime 时还要同步 Controller 期望版本，恢复原先全部规则的后端后再处理下一台。",
            "安全策略收紧和普通更新失败有什么区别": "普通候选校验或 Probe 失败时，还没触碰的旧配置和后端继续服务；滚动中失败则停止当前 Rule 后续更新，恢复可信旧版本后可重新接流。如果 FQDN 新解析结果、收紧后的目的地址白名单或 TLS 安全基线已经证明旧配置不安全，就立即清空该规则后端，置 Available/Configured=False、UnsafeConfiguration，并清理该 Rule 的配置和 ACL，保留 CR 等运维修正。若全局 ConfigMap 本身格式非法，则停止新发布并报 InvalidProxyConfig，不把它误判成所有旧规则均不安全。",
            "NTP拨测和Underlay补DNS有什么区别": "要看程序部署在哪里，以及已有哪段网络能力。已有 unatgw 的 Underlay 业务缺少公网解析，补齐 DNS 后，业务仍走自身出口；另一项需求是拨测程序运行在 Pod 中，需要访问公网一级 NTP 源，但 Pod 没有直接出网链路，因此还需要对指定 NTP 目标转发业务报文，只补 DNS 不能解决。对应 UDP 代理也不能仅凭 listener 和 Doctor 正常就宣称拨测成功，最终要由拨测程序通过入口完成真实 NTP 请求。",
            "这个项目最难的地方是什么_能举个具体例子吗": "最难的是多条规则共享 Proxy VM 时，修改正在服务的规则。比如 A、B 两条规则共用 VM-1、VM-2、VM-3，更新 A 时，先校验候选配置并按变更类型完成目标 Probe，确认 A 移除 VM-1 后仍有可用后端，再只摘除 A→VM-1 并等待入口传播。A 继续由 VM-2、VM-3 服务，B 仍可使用 VM-1；随后只切换 VM-1 上 A 的 conf 和 ACL，在 Builder 本地锁内校验并 reload，通过共享项和 A 的 Doctor 检查后恢复 A→VM-1，再处理下一台。若 A 更新失败就停止后续滚动，确认可信旧版恢复且健康后才接回，状态不明则保持隔离。难点在于同时保护 A 的存量服务、限制其他规则受影响，并处理多 VM 短暂新旧版本共存；共享 nginx 本身故障时仍需整机隔离。",
            "共享代理的安全设计是怎么考虑的": "我在开发中识别出两个风险，并与产品、交付对齐：其他来源可能绕过 Service 直接使用 VM listener，规则也可能因域名解析变化或错误配置访问未授权目标。来源侧由平台按实际网络链路维护 RegionDNS 和 Service/LB 的可信来源网段，Controller 将网段、规则协议和独立监听端口组合为 nftables ACL；目标侧只允许一个逻辑目标，解析并校验后将允许的 IP 和端口固定到 Nginx 配置，HTTP 上游同时配置 Host、SNI 和证书校验。这样把来源准入与目标约束落实在真正收包和转发的节点上。来源网段只能表达网络边界，不能证明原始客户端身份，也不能直接当成租户隔离。",
            "以短信网关为例_业务怎么使用这个组件": "交付人员在 sms Namespace 创建一条 HTTP 规则，选择 ClusterIP，入口 servicePort 为 80，httpHost 和 upstream host 都填 yun.tim.qq.com，上游端口为 443。Controller 创建 Service、发布配置并加入合格后端；Available=True 后，从 status.entrypoints 取得实际 Service 地址和端口，填入消息网关的代理入口配置，公网短信目标保持 yun.tim.qq.com:443。请求由 Pod 经 Service 到 Proxy VM，Nginx 再以 HTTPS 连接固定目标。业务无需维护 VM 地址、内部监听端口、ACL 或 NAT 配置；最后用真实短信接口请求验证接入，Available 只表示入口具备接流条件。",
            "公网代理下一步准备做什么_为什么要做": "下一步先补 Nginx 流量、连接数和异常请求等指标，按代理实例和访问规则观察，用于判断容量与定位问题；在此基础上补按规则限流。YUM 同步软件源可能持续占用大量带宽，大型客户也可能要求资源隔离，因此计划让不同规则选择不同 Proxy VM 实例组，分散流量并减少共享资源争用。当前已经有规则状态、Doctor 和日志等运行观测，规则独立配置与摘流也能控制变更影响，但还不能据此宣称具备按规则流量治理或独立资源。这部分流量监控、限流和实例组隔离属于后续规划。",
            "这个项目在跨团队协作上有什么体会": "私有云需求涉及研发、产品、测试、售前架构师和交付，大家关注的颗粒度不同，需求对齐本身也是工程工作。以公网代理为例，不能只约定支持哪些协议，还需要提前对齐谁在什么网络位置访问、使用哪种入口、能力边界、异常时如何处理，以及怎样算交付成功。我在开发中把共享代理的来源准入和目标约束风险拿出来与产品、交付对齐，再落实到规则校验与 ACL。体会是把使用场景、异常行为和验收标准提前说清楚，能够减少后期因理解偏差反复修改。"
          },
          "绝对红线": [
            "不要说 Controller 改写用户 URL 或直接从裸域名生成规则；用户保留原 FQDN，由产品平台生成规范化 EgressForwardRule。",
            "OriginalHost+ClusterIP 使用 service-id，OriginalHost+LoadBalancer 使用指向 VIP 的受管 A 记录，protocol=DNS 的 DNSRecord 则表达 zone 转发；不要混淆三者，也不要把控制面对象已创建等同于真实客户端流量成功。",
            "不要说 Service、Endpoints、EndpointSlice 或 Controller 执行公网代理；Endpoints 记录允许接收流量的代理 VM，EndpointSlice 把这份列表提供给集群转发组件，Nginx 负责转发，NATGW 提供公网 SNAT。",
            "不要把每 Rule revision、发布门禁、Builder 本地锁或单机 Apply 说成跨 VM、Kubernetes 和服务发现的原子事务；共享 nginx 进程仍是共同故障边界。",
            "不要把 SSH、Doctor 或 DNS 解析的一次观测失败直接等价成真实故障；明确不安全与暂时无法确认采用不同策略。",
            "不要把 Validate、Probe、Doctor 说成 TLS、应用协议或真实业务 E2E，也不要说通用 UDP 已被 Probe 验证。",
            "不要把现有代理 VM 的出网能力讲成本组件新实现的能力；先说明统一接入和配置维护的问题，再解释当前部署的网络约束。",
            "不要用 Overlay、Underlay 或容器/虚机身份直接推断能否出公网；必须说明实际路由、NAT 和访问策略。DNS 规则可以只补解析，已具备 unatgw 的客户端业务流量仍走自身出口。"
          ]
        },
        {
          "bullet_title": "排查 KubeVM 磁盘 I/O 异常并补充 QEMU Native AIO、Nydus 运行时适配",
          "项目背景": {
            "异常现象": "局点一台 KubeVM 的数据盘 vdb 出现 I/O error，Guest 内的 XFS 报告检测到内存数据损坏并主动关闭文件系统。这个现象说明文件系统进入保护状态，但不能仅凭日志把根因归结为缺少 io=native。",
            "排查思路": "先沿 Guest 文件系统与块层 → virtio → QEMU disk driver → LocalPV disk.img 或块设备 → Host 文件系统与物理盘分层核对，再对比 KubeVM 与 CVM 的实际 Domain XML。受测 CVM 在 cache=none 下显式使用 io=native，而 KubeVM 的部分 file-backed 稀疏盘没有命中既有 Native AIO 自动选择。",
            "镜像加速背景": "大镜像冷启动需要等待镜像下载和展开，弹性扩容时大量节点集中回源还会放大 Registry 压力。Nydus 用 RAFS 元数据与数据 chunk 分离实现按需加载，Dragonfly 可在其数据获取链路上提供 piece 级 P2P 复用；两者分别解决启动所需数据量和集中回源问题。",
            "Nydus适配问题": "RuntimeClass 表达 Pod 的运行时选择，但目标 containerd 1.7 在 PullImage 阶段拿不到标准的独立 runtime handler，只能从 PodSandboxConfig 的 io.containerd.cri.runtime-handler annotation 推导 runtime-specific snapshotter，因此需要在准入阶段把两个信号对齐。",
            "改造目标": "不改变用户显式 I/O 配置，也不把 cache、iothread 和 blocksize 混成一个开关；扩展 KubeVirt 既有 I/O 后端选择，使满足 O_DIRECT 前提的更多 file/device 磁盘能够自动选择 Native AIO，并提供集群级回滚。同时补充 Nydus 的准入桥接，让已经选择 nydus RuntimeClass 的 Pod 自动携带 containerd 1.7 PullImage 选择 runtime-specific snapshotter 所需的 annotation。"
          },
          "一句话": "这项工作包括两部分：针对 KubeVM 数据盘 I/O 异常，我分层核对存储路径和实际 Domain XML，扩展 KubeVirt 既有 SetOptimalIOMode 逻辑，对符合条件的磁盘自动生成 io=native，并补齐参数链路和回滚开关；同时在 tcs-admission-webhook 中补充 Nydus RuntimeClass 适配，为已选择 nydus 的 Pod 注入 containerd runtime-handler annotation。",
          "简历推荐写法": "排查 KubeVM 磁盘 I/O 异常，扩展 KubeVirt 的 QEMU I/O 后端自动选择：对最终 cache=none、源为 file/device 且未显式指定 I/O 模式的磁盘自动配置 io=native，保留显式配置优先和集群级回滚开关，补充参数链路、单元测试及多磁盘/不同扇区几何兼容性验证；补充 Nydus 镜像加速的 RuntimeClass 适配，基于 Admission Webhook 为已指定 runtimeClassName=nydus 的 Pod 幂等注入 containerd runtime-handler annotation。",
          "20秒首答": "这项工作分两块。KubeVM 侧，我从数据盘 I/O error 和 XFS 主动关闭现象出发，对比 Domain XML 并拆开 cache、I/O backend、iothread 和 blocksize，随后扩展 Native AIO 自动选择和回滚链路；这部分目前仍在功能分支。镜像加速侧，我在 tcs-admission-webhook 中补充 Nydus RuntimeClass 适配，让已选择 nydus 的 Pod 自动获得 containerd PullImage 所需的 runtime-handler annotation，同时保持幂等并覆盖异常值纠正。",
          "项目状态": {
            "代码归属": "远程分支 dev/kubevm-qemu-io-native 上有两个提交：b410ea206 为设计文档，c0abe90fa 为实现；Author 和 Committer 均为 leandrozhao。",
            "分支状态": "截至 2026-09-04，功能分支相对 origin/tcs-develop 为 0 behind、2 ahead；tcs-develop、已核对的 release 分支和 tag 均不包含 c0abe90fa，也未发现 DisableAutoNativeAIO 或 --auto-native-aio。",
            "交付口径": "KubeVirt 部分可以说完成排查、功能分支代码改造和测试用例补充，但不能说已经合入主干、形成正式镜像、进入发布版本或生产上线。Nydus Webhook 部分已进入 tcs-extensions master 和 release/tcs2.3.5.1，但当前只确认源码分支状态，不把它扩大成具体集群已部署或完整 Nydus 栈已交付。"
          },
          "个人实现与复用边界": {
            "既有能力": "KubeVirt 原有 SetOptimalIOMode 已支持用户显式 io 配置，并会在 cache=none 下为块设备或预分配文件选择 io=native；Native AIO、cache=none、Domain XML 转换和 VMI 生命周期都不是从零实现。",
            "本人增量": "KubeVirt 侧，把 autoNativeAIO 布尔决策接入 Converter，默认开启时把普通/稀疏 file-backed 磁盘也纳入选择；增加 DisableAutoNativeAIO 集群级负向 Gate；从 virt-controller 生成的 launcher 参数一路传到 virt-launcher、Domain Manager 和 Converter；补充配置、参数透传、Manager 和 Converter 边界测试，并调整已有 VMI E2E 的期望。Nydus 侧，在 tcs-admission-webhook 中增加受开关控制的 Pod Mutating Handler，对 runtimeClassName=nydus 的 Create/Update 请求幂等注入或纠正 runtime-handler annotation，并补充七类边界测试。",
            "未改内容": "KubeVirt 部分没有同时修改 blocksize 探测和注入逻辑，没有把 iothread 合入本轮变量，也没有强制设置 io=threads；用户已经显式配置 driver.io 时始终保持原值。Nydus Webhook 不创建 RuntimeClass，不给普通 Pod 自动选择 nydus，也不部署 snapshotter、nydusd 或节点 containerd 配置。"
          },
          "implementation_details": {
            "自动选择条件": "Converter 首先检查 driver.io；非空立即返回。随后只接受 Source.File 或 Source.Dev，其他 source 跳过。只有最终 Driver.Cache 为 cache=none 时才考虑 Native AIO，因为 io=native 需要 O_DIRECT 条件。",
            "默认开启行为": "autoNativeAIO=true 时，符合上述条件的 file/device 磁盘都可以设置 io=native，因此覆盖 LocalPV 常见的稀疏 disk.img。",
            "回滚行为": "FeatureGate API 以名称是否存在表达开关，而新能力要求默认开启，因此采用 DisableAutoNativeAIO 负向 Gate。关闭后恢复旧逻辑：块设备或预分配文件仍可使用 Native AIO，普通稀疏文件不再自动设置。",
            "参数传播": "virt-controller 读取 ClusterConfig.AutoNativeAIOEnabled，生成 --auto-native-aio 参数；virt-launcher 解析后传入 NewLibvirtDomainManagerWithAutoNativeAIO，Manager 再将该值带入 VMI 到 Domain XML 的转换上下文。",
            "生效时机": "该值在 VMI 转换并生成 Domain XML 时生效；只修改控制面配置不会原地改写已经运行的 QEMU 磁盘参数，需要新建或重建 VMI，并检查实际 virt-launcher Domain XML 确认。",
            "Nydus准入触发": "源码支持 Pod CREATE 和 UPDATE；正式 .tad 模板也注册二者，但 legacy charts 模板只注册 CREATE，必须按实际交付形态回答。EnablePatchNydusRuntimeHandler 开启且 spec.runtimeClassName 明确等于 nydus 时才进入注入逻辑，runtimeClassName 为空或其他值均跳过。",
            "Nydus幂等与纠正": "annotations 为空时先初始化；目标 annotation 已经为 nydus 时不重复修改，存在但值错误时纠正为 nydus。Admission Handler 对修改前后的 Pod 生成 JSON Patch 返回给 API Server。",
            "Nydus运行时边界": "Webhook 只把 Pod 的 RuntimeClass 选择映射为 containerd 1.7 PullImage 所需 annotation；RuntimeClass 对象、containerd handler/snapshotter、nydusd、镜像转换和可选 Dragonfly P2P 链路必须由其他组件提供。准入成功只证明控制面意图被补齐，不证明容器实际走了 Nydus。",
            "Nydus版本与同名契约": "实现只匹配 runtimeClassName=nydus 并写入值 nydus，不会查询 RuntimeClass.handler，因此部署必须保证 RuntimeClass 名、handler、containerd runtime 键同名。本地总体设计稿同时出现 containerd 1.6.9 基线与 1.7 annotation 示例；本模块只能按源码和上游文档确认为 1.7 适配，1.6.9 是否有内部 backport 需另行验证。",
            "Nydus共享Webhook语义": "Nydus Handler 位于共享 Pod mutating 链末尾；前序 Handler 报错时它不会执行。部署 failurePolicy=Fail，Webhook 不可达、TLS/超时或通用框架失败会阻断所有未被排除的 Pod CREATE/UPDATE，而不仅是 Nydus Pod；带 infra.tce.io/tcs-admission-webhook-exclude 标签的 Pod 则跳过整个 Webhook。"
          },
          "验证与结果": {
            "代码测试覆盖": "新增或调整的测试覆盖默认开启、DisableAutoNativeAIO 回滚、Controller 参数透传、Manager 默认与显式关闭，以及 Converter 的稀疏文件、旧行为回滚、显式 io=threads 保留、非 cache=none 跳过和块设备边界。当前确认的是测试代码存在，没有可复核的 JUnit、CI 链接或本轮完整测试输出，因此不说全部 CI 已通过。",
            "实验与最终提交关系": "2026-08-13 至 08-20 的候选镜像和实验早于 08-21 的最终实现提交，验证的是 cache=none 稀疏文件选择 Native AIO 这一核心语义，但没有覆盖最终 DisableAutoNativeAIO Gate 和完整参数链，因此不能当成 c0abe90fa 的正式发布验收。",
            "兼容性观察": "在受测 3.10.11 LocalPV raw 路径中，Guest 512/512 几何下 512B 和 4KiB Direct I/O 均未出现 fio 错误；4Kn Guest 对 512B Direct I/O 返回 EINVAL、4KiB 正常，符合对齐约束，不能把它解释成 io=native 单独导致的缺陷。",
            "性能观察": "实验没有得到统一性能提升：部分 512/512 或 512e 小块写场景出现约 15% 至 37% 回退；4Kn 路径有方向性提升，但仅单 VM、固定顺序且存在基线漂移，不足以外推为产品收益。面试回答应说完成兼容性和性能取舍验证，而不是性能优化成功。",
            "XFS边界": "普通 fio 与几何兼容性测试只能观察吞吐、延迟和 I/O 错误，不能证明 flush/FUA、异常掉电持久化或 XFS 根因。要验证因果关系，还需要独立的 Guest XFS workload、fsync/flush 记录、异常停止或掉电注入、重启后 dmesg 与 xfs_repair -n 检查。"
          },
          "高压追问首答": {
            "为什么不是直接修XFS": "XFS 日志表明文件系统检测到一致性问题并主动关闭，但没有指出问题一定来自 QEMU I/O backend。我的处理是先保存现场并沿存储链分层排查，再用单变量实验验证 io=native 的兼容性和性能；文件系统持久化根因需要单独的 flush、异常退出和重启后检查，不能由普通 fio 推断。",
            "为什么要自动选择io_native": "LocalPV 常见后端是 file-backed disk.img，原逻辑只为块设备或预分配文件自动选择 Native AIO，稀疏文件即使最终 cache=none 也不会命中。新逻辑在 O_DIRECT 前提满足、用户没有显式选择时补齐这个默认，使生成的 Domain XML 与预期后端策略一致。",
            "为什么保护显式配置": "用户显式指定 io 表示已经做出 workload 或兼容性选择，自动策略只能补默认值，不能覆盖明确意图。因此函数第一步检查 driver.io，io=native、io=threads 或其他非空值都原样保留。",
            "为什么使用负向FeatureGate": "能力要求默认开启，而当前 FeatureGate API 只表达某个名称是否出现在列表中。使用 DisableAutoNativeAIO 才能在默认没有配置时启用新行为，同时允许管理员显式加入 Gate 快速恢复旧逻辑。",
            "io_native和cache_none_iothread有什么关系": "cache=none 决定 QEMU 通常以 O_DIRECT 绕过宿主页缓存，是 Native AIO 自动选择的必要条件；io=native 决定 QEMU 使用哪种宿主 I/O 提交后端；iothread 决定 I/O 处理线程绑定。三者属于不同维度，本轮只改 I/O backend 选择。",
            "怎么确认真正生效": "不能只看 FeatureGate 或 virt-controller 参数。需要新建或重建 VMI，进入对应 virt-launcher 查询实际 libvirt Domain XML，确认目标磁盘 driver 同时出现预期 cache=none 和 io=native，再核对其他磁盘属性没有被意外改变。",
            "性能是否变好": "没有统一变好。实验用于判断兼容性和取舍，主线小块写场景存在明显回退，4Kn 方向性收益也受样本和顺序限制，所以我不会承诺性能提升；这个改造的价值是补齐可控的自动选择和回滚能力。",
            "现在是否已经上线": "要分开回答。KubeVirt Native AIO 改造目前只在个人远程功能分支，不能说已合入或上线；Nydus Webhook 代码已经合入 tcs-extensions master 和 release/tcs2.3.5.1，但是否部署到具体生产集群仍需部署和运行态证据。",
            "Nydus为什么需要这个annotation": "RuntimeClass 查到的 handler 标准上通过 RunPodSandboxRequest 交给 CRI，但目标 containerd 1.7 的 PullImageRequest 没有独立 runtime 信息。containerd 会从 PodSandboxConfig 读取 io.containerd.cri.runtime-handler，再用对应 runtime 配置覆盖默认 snapshotter。Webhook 在准入时把 RuntimeClass 意图映射为这个 PullImage 信号，避免镜像准备和 Sandbox 运行时走两套选择。",
            "Webhook是不是自动注入RuntimeClass": "不是。它只处理已经设置 runtimeClassName=nydus 的 Pod，再补 runtime-handler annotation；没有 RuntimeClass 选择的普通 Pod不会被自动切换到 Nydus。",
            "Nydus这部分你做到了哪一层": "我补充的是 tcs-admission-webhook 的准入适配、默认值到 CLI/Helm 的开关接线和七类分支测试，把 runtimeClassName=nydus 映射为 containerd 1.7 PullImage 使用的 runtime-handler annotation。RuntimeClass、snapshotter、nydusd、containerd 节点配置、镜像转换和 Dragonfly P2P 属于 Webhook 之外的运行时与分发链路。",
            "Nydus怎么确认真正生效": "先确认最终 Pod 的 RuntimeClass 和 annotation，再核对 RuntimeClass.handler、节点 containerd runtimes.nydus 与 snapshotter socket；随后从实际容器反查 Snapshotter=nydus，并确认 fuse.nydusd 或对应真实挂载及 nydusd 读取。最后才能在受控冷热缓存条件下比较启动时间和回源字节。只看 Pod YAML 不能证明 Lazyload。",
            "Nydus_1_6和1_7是否都适用": "不能这么说。源码注释和上游 Nydus runtime-level snapshotter 文档都绑定 containerd 1.7；总体设计稿写的 1.6.9 是另一条需要现场核验的版本基线，除非能证明内部 backport 或实际 CRI 行为，否则不宣称这段 annotation 桥接在 1.6.9 生效。",
            "Nydus_Webhook故障影响": "核心 Nydus 函数没有外部 I/O、当前分支都返回 nil，但它复用 failurePolicy=Fail 的共享 Pod Webhook。服务不可用或共享框架失败会阻断所有未排除 Pod 的创建更新，所以生产上要把它按集群级关键依赖做高可用和监控。"
          },
          "绝对红线": [
            "不要说缺少 io=native 是 XFS 异常的已确认根因，也不要说该改造修复了 XFS Bug。",
            "不要说从零实现 Native AIO；准确说法是扩展 KubeVirt 既有 I/O 模式自动选择和参数传播。",
            "不要把功能分支、测试代码或候选镜像说成已合入主干、CI 全部通过、发布或生产上线。",
            "不要把 cache=none、io=native、iothread、Guest 512B/4KiB 请求和设备 512/512、512e、4Kn 几何混成一个配置。",
            "不要承诺统一性能提升；主线部分小块写场景存在回退，4Kn 观察也不足以外推。",
            "不要把 Nydus Webhook 说成完整 Nydus 镜像加速栈；本人补充的是 RuntimeClass 到 containerd runtime-handler annotation 的准入桥接、开关接线和测试。"
          ]
        },
        {
          "bullet_title": "LocalPV 容量感知调度优化与容量记账修复",
          "项目背景": "平台通过 CSI 动态供应本地卷。原有插件能检查容量和选择节点内磁盘，但缺少候选节点之间的容量优选，容量分布不合理会影响副本扩容；另一项问题发生在容量对象删除重建后，PVC 占用明细仍在，磁盘聚合占用却丢失，导致可用容量被高估。",
          "20秒首答": "我主要做了两件事：在已有 LocalPV 插件上增加跨节点评分，按多卷需求和节点容量支持分散、装箱放置；修复容量对象删除重建后已有 PVC 占用漏记的问题。前者改善节点选择，后者避免因少算占用而把容量不足的节点误判为可调度。",
          "个人实现与复用边界": {
            "本人增量": "跨节点 Score/NormalizeScore、多卷需求按 StorageClass 聚合、Spread/Binpack/None 策略配置；容量条目创建时汇总已记录的 PVC 占用，并完善缺条目时的明细去重更新。",
            "复用部分": "评分改动复用原有逐卷选盘、容量过滤、预留和 VolumeBinding 协作链路；重建补偿复用已有 PVC Informer、UID 占用明细和 AssumeMap 账本。存储侧负责实际创建和回收卷。"
          },
          "复习备注_版本与口径": {
            "使用方式": "以下是准备备注，不逐句口述。两项工作按各自源码版本解释；L05–L11、L49–L51中的评分路径指评分版本，L12–L15显式比较两版，L40–L48讲重建补偿。",
            "评分版本": "TX-kubernetes 工作树 f2c1f934c6c40182abe4edbf6fb75cec1d6ce701，LocalPV 目录与 7f442406153 一致；采用上报可用量、PodInUse 同步保护和按盘 Reserve，具有 Score/NormalizeScore。",
            "补账版本": "固定提交 eb06b79fc28f95de0f1b6a94fe5f0765751fdedb，并对照其父提交；本地引用位于 origin/feat/optimize-localpv-interface 和 origin/release/tcs2.3.4.1。该快照的多盘路径采用 TotalSize、DiskUsage、AssumeMap，未包含评分版本的 Score/NormalizeScore；按两次改动讲，不拼成一份已集成源码。",
            "补偿范围": "补回已有 pvcUsageMap 明细对应的请求量，匹配键为 NodeName+DiskName；不恢复仅存在于被删除条目中的在途预留。这里采用 TotalSizeMultiDisk 的多盘例子，旧单盘 Available 兼容路径另行解释。",
            "卷绑定交接": "节点注解是 volume.kubernetes.io/selected-node，磁盘注解是 loopdevice.infra.tce.io/disk-selected。Reserve 按配置顺序执行，源码推导的安全顺序为 VolumeBinding 先复制 PVC，再由 LocalCapacityScheduling 修改副本；部署配置未在本轮核实。",
            "手算与调用边界": "L41 的 100/60/50 GiB 和 L50 的跨 SC 共盘属于源码手算；L51 讨论同一 CycleState 被重复调用时的代码路径，没有将其记为已复现的框架故障。"
          },
          "证据边界": {
            "现有验证": "评分版本已有单元与集成测试代码，以及调用实际插件、人工构造容量和请求、模拟最终绑定成功的组件对照记录；扩展补测记录包含三轮执行、请求顺序和权重敏感性及退化反例。",
            "可讲收益": "在指定 CPU/内存评分与容量输入下，混合大小卷场景后续接纳副本由 1 个增至 2 个；装箱场景让原先放不下的 60 GiB 大卷得到接纳。这些数字衡量对应请求序列的调度接纳能力。",
            "验证局限": "评分对照覆盖同 StorageClass 双 PVC，不覆盖真实 API/CSI、跨 StorageClass、多磁盘、并发和生产业务效果；跨 SC 共盘例子单独作为源码推演。",
            "容量补记验证": "已对照 eb06b79fc28f 与父提交核实补偿入口和调用链；该提交只修改容量文件，未新增测试。已有 PVC 去重测试不等于重建补偿回归，专属回归矩阵当前标为未执行；评分版本的 31,320 个对照单元不计入这项修复的验证。"
          },
          "高压追问首答": {
            "L01_总空闲容量够为什么副本仍然部署不了": "因为本地卷必须在某个具体节点上放得下，多个副本还可能要求分布在不同节点。例如三个节点分别剩 10、10、40 GiB，总共还有 60 GiB，但三个各需 20 GiB、要求跨节点的副本只能部署一个。我的优化是在前面的放置过程中尽量保留符合目标容量要求的节点，不能把不同节点的空间拼起来。",
            "L02_原来完全不看存储吗_CPU内存调度为什么不够": "原来已经检查本地卷能不能放下，但多个节点都能放下时，没有用本地存储容量进一步排序。CPU、内存需求相同的两个 Pod，可能分别申请 10 GiB 和 40 GiB 本地卷，因此 CPU 分散不一定带来存储分散。我补的是候选节点之间的容量偏好，原来的容量硬约束继续保留。",
            "L03_你具体改了什么_是不是重写了一套调度器": "我做了两项增量：在已有插件上增加按 StorageClass 聚合多卷需求的跨节点评分，以及在容量条目重建时补回已记录的 PVC 占用。评分解决多个节点都可行时怎样选，补记解决容量余额是否算对。节点内选盘、卷绑定、PVC 事件跟踪和基础账本结构沿用各版本已有实现。",
            "L04_有状态和无状态对这个调度有什么区别": "插件不根据 Pod 是 StatefulSet 还是 Deployment 来决定是否评分，而是识别它是否引用受管 StorageClass 的 Pending 本地 PVC。有状态业务常常依赖固定本地数据，所以更容易受节点和卷位置约束；无状态服务如果申请这样的本地 PVC，也可以走同一条链路。没有相关 PVC 的 Pod 不参与这项评分。",
            "L05_讲一下从Pod提交到卷可用的完整链路": "按评分版本讲，PreFilter 收集待分配本地 PVC，Filter 在每个候选节点逐卷试算选盘，并保存节点状态。Score 读取其中按 StorageClass 聚合的请求量，结合该节点的容量汇总打分，框架再综合其他插件选节点。Reserve 阶段建立卷绑定的内存状态、补充磁盘选择并预留本地容量，VolumeBinding 的 PreBind 才提交 API 更新并等待卷创建和绑定；成功 PostBind 解除容量同步保护，失败走 Unreserve。Pod 绑定后由 kubelet 完成卷挂载。",
            "L06_一个Pod申请多个卷怎么避免重复使用容量": "先把待分配本地 PVC 按容量从大到小处理，在每个候选节点内用同一份 diskAssumeMap 累计已经分给前面卷的容量。后一个卷只能使用扣除这些假设占用后的余额，所有卷检查通过后，这个节点才算可行。评分再按 StorageClass 汇总需求；既有选盘是逐卷贪心，没有穷举所有磁盘组合，所以不能承诺只要存在可行组合就一定找到。",
            "L07_评分公式是什么_现场算一个例子": "对每个 StorageClass，按节点上支持该类存储的容量汇总，计算 r=(可用量−当前 Pod 该类请求量)/总量，再按各类请求字节数加权得到 R。Spread 取 round(100R)，Binpack 取 round(100(1−R))，随后做候选节点间归一化。单 SC 例子：两节点总量都为 100 GiB，可用量为 80、40 GiB，当前请求 20 GiB，R 分别为 0.6、0.2，Spread 原始分为 60、20，Binpack 为 40、80。这里是按 SC 汇总的容量偏好，Score 不直接读取选中磁盘的剩余量。",
            "L08_为什么用剩余比例_会不会选了绝对空闲更少的节点": "剩余比例表达这类存储相对总容量的余量，能够让不同容量节点按相对占用程度比较；按这套规则，确实可能优先选择绝对空闲更少、但剩余比例更高的节点。Filter 先保证当前卷能放下，Score 再表达放置偏好。这是一个启发式目标，对未来大卷是否更好要通过请求分布验证，不能把它解释成始终选绝对空闲最多的节点。",
            "L09_多个StorageClass怎么配置_策略冲突怎么办": "在本地 PVC 对应的 StorageClass 上配置 storage.infra.tce.io/scheduler-policy，值可为 Spread、Binpack 或 None；未配置或留空时使用插件默认策略，默认未设置时为 Spread。同一 Pod 涉及的待分配本地卷必须解析成同一种策略，否则这个 Pod 的本插件评分统一为 0，容量过滤仍然执行。非法注解或读取失败也会停止该 Pod 的容量评分，不擅自按某一类存储的偏好替用户决定。",
            "L10_节点间Spread和节点内Balanced有什么区别": "Spread/Binpack 决定多个候选节点之间怎么排序；原来的 DiskPolicy 决定某个节点里面选哪块盘。Balanced 优先选可用容量最大的盘，Concentrated 选剩余最少但够用的盘，两层策略可以独立组合。两块盘各剩 30 GiB，不能只凭总和就判定能放一个 60 GiB 卷，是否有单个受管分配单元能容纳它仍由 Filter 检查。",
            "L11_已经是0到100分为什么还要NormalizeScore_权重怎么影响结果": "归一化是在本轮候选节点中按最大最小分拉开差异，例如 60、20 会变成 100、0，全部同分则保持原值。最终节点由各插件分数乘权重后相加决定，存储分最高不保证一定中选；甚至 80、81 的小差异也可能被拉成 0、100，需要一起评估权重。现有补测里 CPU:LocalPV 为 1:1 时两个原始收益例子仍成立，改成 10:1 后消失，这只是该对照配置下的结果。",
            "L12_容量数据哪里来_你统计的是申请量还是实际磁盘写入量": "两项工作使用的容量实现不同：评分快照读取 LocalStorageCapacity 上报的总量和可用量，并在缓存余额上记录调度预留。七月容量修复版的多盘路径读取 TotalSizeMultiDisk，总量减去 PVC 请求聚合 DiskUsage 和在途 AssumeMap 才得到余额。PVC 的需求都取 requests.storage，记账衡量的是申领容量，不是应用实际写入量。",
            "L13_Filter通过以后容量被别人占了怎么办": "评分版本的 Filter 只试算；选定节点后，Reserve 在共享缓存的锁内重新检查并扣减余额，不够就失败重试。补账版本则按 PVC UID 写入 AssumeMap，AddAssume 不再次检查余额，磁盘条目不存在时也只是记录告警返回。两版分别对应“余额复检后扣减”和“登记在途预留”，加锁保护的都是本进程账本。",
            "L14_多个磁盘只预留成功一部分_或者后续绑定失败怎么回滚": "评分快照按盘扣减并记录 reservedMap，Unreserve 只归还已成功扣减的部分并解除容量同步保护；PostBind 只解除保护。七月容量修复版则按 PVC UID 写入预留，Unreserve 幂等移除对应 AssumeMap 条目，PostBind 只输出容量快照。已经由 PVC 事件写入的 DiskUsage 随 PVC 删除或选择注解清除而释放，不由一次 Pod 调度失败直接清空。真实卷的创建和回收继续由卷绑定与存储侧流程处理。",
            "L15_容量上报延迟或Scheduler重启会不会把预留覆盖掉": "评分快照用 PodInUse 标记正在绑定的 Pod，周期 SetCapacity 看到仍有保护时跳过余额覆盖。七月容量修复版的多盘周期同步只更新 TotalSize，保留 DiskUsage 和 AssumeMap，PodInUse 已经弃用。容量条目被删除则会丢失条目里的聚合和预留，这次补偿从仍在内存中的 PVC 明细恢复聚合；进程重启后明细本身也要通过 Informer 对象处理重新建立。",
            "L16_为什么需要WaitForFirstConsumer_卷是谁创建的": "延迟绑定让调度器在选节点时一起考虑 Pod 的资源需求和本地存储位置，避免先把卷固定到一个不满足 Pod 条件的节点。这里的本地存储是平台提供的容量上报和供给链路，本插件给出节点、磁盘选择并协作绑定，实际创建卷由存储侧完成。上游原生 local 卷以预创建 PV 为基础，不能因为平台支持这条供给流程，就说所有原生 LocalPV 都自带动态创建能力。",
            "L17_已有PVC或已运行Pod能不能换策略后重新均衡": "这次优化主要作用于尚未分配的本地卷；已绑定卷的位置仍由 PV 的节点约束限制，不能为了拿高分把已有数据调到别处。如果一个 Pod 同时有已绑定和待分配卷，也必须先满足已有卷的位置，再考虑新卷。修改策略影响后续调度，不会主动迁移已运行 Pod 或整理已有磁盘数据。",
            "L18_Spread是不是已经保证副本跨节点_和反亲和有什么区别": "没有，存储 Spread 只是按剩余容量比例给节点打分，不保证同一业务的副本一定分开。副本必须跨节点时，要另外配置硬反亲和或相应的拓扑硬约束；它们先决定哪些节点允许使用，存储评分再在其中排序。收益例子里的后续副本明确有跨节点约束，不能把这个保证算到 Spread 本身。",
            "L19_副本从1到2怎么来的_现场画一下": "三个节点总容量各 100 GiB、初始可用各 60 GiB，先依次提交 40、10、10、40、10、10 GiB 六个单卷请求，Pod 的 CPU、内存需求相同。这组对照使用 CPU:LocalPV=1:1000；保留 CPU/内存评分但关闭本地存储评分时，前置请求后的剩余容量排序为 10、10、40 GiB；启用 Spread 后为 10、20、30 GiB。随后三个各需 20 GiB、要求跨节点的副本，可接纳数就从 1 个变成 2 个。两边都接纳了全部前置请求，总剩余仍是 60 GiB，第三个副本依然不能部署。",
            "L20_装箱怎么让60GiB大卷放下来_代价是什么": "在同样的 CPU:LocalPV=1:1000 对照配置下，三个节点各可用 60 GiB，先提交六个 20 GiB 卷；均衡放置后剩 20、20、20 GiB，总计还有 60 GiB，但没有一个节点能独立放下后续 60 GiB 大卷。装箱后可以剩 0、0、60 GiB，因此大卷可以接纳。代价是可供跨节点小副本使用的节点变少：如果后续换成三个必须分开的 20 GiB 副本，均衡能放三个，装箱只能放一个。",
            "L21_分散一定更好吗_有没有负收益例子": "不一定，现有补测保留了明确反例：初始每节点可用 60 GiB，前置请求为 40、15、15、40、15、15 GiB。基线最终剩 5、5、30 GiB，还能放一个 30 GiB 卷；Spread 剩 5、15、20 GiB，反而一个也放不下。评分只根据当前请求和余量做选择，没有预知下一批卷的大小，所以容量更均衡不代表未来接纳数一定更高。",
            "L22_收益是不是挑了有利顺序_基线和验证怎么做": "对照使用同一套实现和相同输入，基线只关闭本地存储评分，同时保留容量过滤、CPU/内存评分和后续副本的硬反亲和。后续又改变了请求顺序、卷大小、初始容量和评分权重，并补了同 StorageClass 双 PVC、预留回滚和成功后释放保护的检查。节点与请求是人工构造，组件调用是真实代码，最终 API/CSI 绑定由测试模拟成功，因此它验证的是组件接纳和状态变化，不能代替真实存储及生产验收。",
            "L23_怎么量化整体价值_能说提升利用率或扩容成功率吗": "现有证据适合说特定请求序列下能接纳更多副本或更大的卷，没有生产前后数据支持统一提升率。若做线上验证，我会在相近容量和请求分布下比较因本地容量不足而 Pending 的数量及等待时间、目标副本的部署完成情况、按卷大小统计的可用节点数，并同时检查调度耗时。这些才反映流程是否改善，单看总剩余容量或磁盘写入量不足以说明调度收益。",
            "L24_上线如何选策略和回退_装箱会不会造成IO热点": "策略要根据卷大小、扩容方式和节点故障域选择，并用业务请求分布验证；需要保留多个可部署节点时可以评估 Spread，需要保留大卷空间时可以评估 Binpack。装箱可能增加单节点磁盘或网络争用，这次评分没有纳入 IOPS、带宽和延迟指标，所以不能据此保证性能更好。配置 None 或移除本插件的 Score 注册可以停止容量优选，同时保留容量检查和预留；它不会撤销已经发生的卷放置。",
            "L25_容量数据缺失是忽略打分还是拒绝调度": "需要区分阶段：Filter 找不到能满足请求的受管容量，会把该节点判为不可行；若 Filter 已通过但 Score 缺少总容量等评分数据，当前实现对该节点返回 0 分。这个 0 分会参与后续比较，不等于从候选列表删除，也不保证是中性处理。策略冲突导致整个 Pod 的本插件分数都为 0，则只是该插件不能再区分这些节点。",
            "L26_这项能力能用于训练吗_有没有和训练平台联动": "可以服务申请受管本地 PVC 的训练 Pod，帮助它在部署阶段找到同时满足卷容量和其他硬约束的节点。但这次改动没有对接训练 Job 的生命周期，也没有实现 Gang、checkpoint 协调、任务重试或训练吞吐优化。训练平台可以使用这项基础能力，任务什么时候能整体启动和失败后怎么恢复仍需要另外的机制。",
            "L27_GPU够但本地盘不够_或者GPU拓扑与存储偏好冲突怎么办": "同一个 Pod 必须在同一个候选节点上同时满足 GPU、CPU、内存、卷位置和容量等硬约束；不能把 A 节点的 GPU 和 B 节点的本地盘拼成一次可行分配。若本地盘最合适的节点不满足 GPU 硬约束，它会先被过滤掉；若冲突的只是评分偏好，就由插件权重参与取舍。交集为空时 Pod 继续等待，需要增加兼容资源或调整约束，提高存储分数不能解决。",
            "L28_分布式训练多个Worker一起启动_这个插件能保证吗": "不能，它按单个 Pod 评估本地卷，不掌握整个训练任务的 Worker 数和启动条件。成组调度需要上层 Job/PodGroup 与支持 Gang 的调度机制一起判断各 Worker 的 GPU、存储和拓扑是否满足，并协调预留、等待及失败退出。把单 Pod 的容量 Filter 和 Score 接进去只是其中一环，不等于已经实现整组原子启动。",
            "L29_对推理有什么作用_能提高模型加载速度或缓存命中吗": "如果推理实例用受管本地 PVC 存放模型文件或缓存，这项评分能改善申请新卷时的容量放置，减少容量分布导致的部署受阻。它没有模型版本、文件位置或缓存命中信息，也没有测量模型加载、TTFT 和服务吞吐，所以不能据此承诺推理加速。若要优先调到已有模型缓存的节点，还需要平台暴露缓存位置与版本，并另行评估放置和数据准备机制。",
            "L30_本地模型缓存和训练Checkpoint能按同一种方式处理吗": "不能只因为都落在本地盘上就按同一种恢复方式处理：有远端可靠源的模型或数据缓存通常可以重新拉取，代价是冷启动；checkpoint 若是唯一有效进度副本，节点故障后可能无法读取，甚至丢失。容量过滤和绑定链路约束卷的放置，评分只在可行节点间表达偏好，都不负责数据复制。要支持跨节点恢复，需要训练系统保存完整一致、且目标节点可访问的 checkpoint，并实现对应恢复流程。",
            "L31_和GPU节点自愈一起用_Drain后能把训练Pod迁到别处吗": "不能默认可以，Pod 使用已绑定本地卷时，原卷的节点约束仍然存在；驱逐 Pod 不会把卷里的数据搬走。节点只是重启且盘和数据完好时，任务可以在节点恢复后按上层机制重新拉起；如果要换节点，就要有可访问的数据副本、缓存重建或数据恢复方案。这项存储评分和节点自愈都没有补齐训练状态迁移，PDB 也不能代替它。",
            "L32_训练推理用COS_共享文件系统_emptyDir也能用这套策略吗": "当前实现只处理受管 StorageClass 下的 Pending 本地 PVC，不直接识别应用通过 SDK 访问 COS 的需求，也不把 emptyDir 或 hostPath 当作这条 PVC 路径。共享文件系统和云盘的可达性、拓扑及供给方式不同，需要各自的容量和绑定机制，不能把本地节点容量公式原样套过去。判断是否适用先看卷的实际类型和申领路径，不看业务是不是训练、推理。",
            "L33_有状态业务扩容是不是指把原来的PVC变大": "这里主要指增加业务副本，每个新副本再申请本地卷，因此需要足够多符合容量和部署约束的节点。把一个已有 PVC 从 20 GiB 扩成 40 GiB 属于卷扩容，要看存储后端和原节点能否完成扩展。这次节点评分不会搬走旧卷，也不能保证已有卷原地扩容成功。",
            "L34_高优先级训练任务来了_抢占能整理本地存储碎片吗": "仅抢占或删除低优先级 Pod，不能假定它的 PVC 和实际存储会立即释放；持久卷可能仍然保留数据和绑定关系。就算某些容量被回收，也必须在目标节点形成足够的可用分配空间，并满足其他硬约束。这次没有实现面向存储的数据搬迁、碎片整理或任务级抢占协议，不能用 CPU/GPU 的释放逻辑直接套到持久卷。",
            "L35_这不就是一个简单打分吗_难点和算法局限在哪里": "公式不复杂，工程重点是让逐卷选盘、按 SC 评分、容量预留和卷绑定使用能衔接的状态，并处理策略冲突和权重影响。另一个具体问题是容量条目重建时，PVC 明细与磁盘聚合脱节，需要在条目创建入口补回占用。前者要用请求序列和退化反例检验放置取舍，后者要验证同一故障时序下余额及容量过滤结果是否正确。",
            "L36_节点和卷很多时开销如何_下一步还会做什么": "评分路径会逐卷查找可用盘，按 StorageClass 汇总时还存在容量集合扫描；补账版本在新建磁盘条目时扫描已有 PVC 明细，成本随明细数量增加。频繁重建还会延长容量写锁的持有时间，因此要同时看调度耗时和锁等待。优化时可以按节点、StorageClass 或节点磁盘键建立索引，减少重复扫描，并验证事件更新、补偿和删除时索引是否一致。",
            "L37_这个插件运行在哪里_代码怎么生效和回退": "插件运行在 kube-scheduler 进程内，评分能力随 Scheduler 构建，并通过 Score 扩展点配置启用和设置权重。把策略设为 None 或移除 Score 注册，可以停止容量优选，保留原有容量过滤与预留。这个开关只控制评分，容量账本修复随对应代码版本生效；切换评分策略也不会迁移已经绑定的卷。",
            "L38_说用索引太泛了_容量连续变化具体用什么数据结构": "可以先按节点和 StorageClass 用哈希表缓存容量汇总；需要按容量阈值查候选时，再维护按可用容量排序的树或容量桶。容量上报、预留和释放时更新索引，单维查询的开销包括定位和返回候选的数量。多卷、单盘容量和 GPU 等约束仍要在同一节点上复核，最终预留时再核对可用状态。",
            "L39_除了冲突回滚_如何让训练大任务与普通任务提前避让": "可以让大训练任务先公布关键候选节点及预留需求，普通小任务优先使用其他可行节点，减少同时争抢；最终仍通过共同的容量账本确认预留。小任务持续到达时，单靠打分保证不了大任务启动，还需要队列层的等待优先级、受保护预留或准入限制。预留设置超时释放，避免长期阻塞其他业务。",
            "L40_容量对象删除重建_具体删了什么_磁盘和PVC还在吗": "删除的是平台的 LocalStorageCapacity 对象，它描述节点、StorageClass 和磁盘容量；对应的 PVC、PV 和磁盘数据可以一直保留。删除处理会移除相应 StorageClass 支持，只有某个节点和磁盘条目不再支持任何 StorageClass 时，才把它从容量 map 中删除。故障发生在后续重建这个内存条目时：PVC 占用明细仍在，新条目的聚合占用却从零开始。",
            "L41_容量漏记会怎样影响调度_用数字讲一下故障时序": "举一个多盘容量条目的手算例子：节点 A 的 d1 总量 100 GiB，已有 PVC 请求 60 GiB，没有在途预留，正确余额是 40 GiB。LSC 删除使磁盘条目消失，但按 PVC UID 保存的 60 GiB 明细还在。旧逻辑重建条目时把 DiskUsage 初始化为 0，余额就被误算成 100 GiB。此时新请求 50 GiB，本地容量 Filter 可能错误放行 A/d1，而正确账本应当拒绝。",
            "L42_PVC明细还在_为什么等Informer再发一次事件不能自动恢复": "因为已有逻辑按 PVC UID 去重，只把新请求量与明细中旧请求量的差值加到 DiskUsage。同一个 60 GiB PVC 再来一次，差值是 60 减 60，仍然为零，补不回磁盘条目重建时丢失的 60 GiB 聚合占用。根因是明细还在、聚合丢了，而增量更新依赖两者原本一致。",
            "L43_具体在哪补回占用_为什么同时覆盖PVC先到和LSC先到": "补偿放在容量 map 的 Add 创建分支：目标节点和磁盘条目不存在时，先创建新条目，再汇总 pvcUsageMap 中属于这个 key 的请求量，填好 DiskUsage 后放入 map。PVC 先到时先保存明细，LSC 到来建条目时补回；LSC 先到时，后续 PVC 事件按原有逻辑增加占用。这样，同一入口同时处理容量条目删除重建和首次建账时的事件先后差异。",
            "L44_重复Add或PVC重放会不会多扣_不同磁盘怎么避免混账": "只有条目不存在时才从零汇总，已有条目再次 Add 只更新 TotalSize，保留 DiskUsage 和 AssumeMap。PVC 明细以 UID 去重，同 UID 同请求量不增加占用，请求量改变时按最新值更新或增加差值。补偿按 NodeName 加 DiskName 完整匹配，因此 A/d1、A/d2 和 B/d1 分别记账。PVC 在补偿前被删除时，删除处理先移除对应 UID 的明细，重建时就不会再计入。",
            "L45_补偿扫描时又来了PVC事件_会不会读到半成品或重复计算": "Add 补偿与 PVC 明细更新使用同一把容量 map 写锁，两类操作依次完成。Add 先在锁内汇总完整的 DiskUsage，再把新条目放入 map；Filter 持同一把锁的读锁，一起读取总量、占用和预留。因此本进程内不会读到“新条目已出现，但占用还没补完”的中间状态。",
            "L46_PVC什么时候计入DiskUsage_和Reserve预留怎样交接": "这个容量版本按 PVC 的选择信息记账：受管 StorageClass 的 PVC 同时带有 selected-node 和 disk-selected，并且有 storage 请求量，就会被跟踪，Pending PVC 也可能计入。Reserve 先按 PVC UID 写 AssumeMap；后续 PVC 事件在同一把写锁内移除该 UID 的预留，再更新明细与 DiskUsage，完成占用交接。若调度失败，Unreserve 只移除在途预留，已经由 PVC 事件确认的占用继续随 PVC 删除或选择注解清除而释放。",
            "L47_容量条目重建和Scheduler重启有什么区别_预留能全部恢复吗": "容量条目重建时进程还在，pvcUsageMap 里的明细可以用于补回 DiskUsage。旧条目中的 AssumeMap 随删除丢失，新条目从空预留开始；仅有在途预留、尚未进入 PVC 明细的请求不会被这次补偿找回。Scheduler 进程重启则连明细也丢了，要通过 Informer 的初始对象处理重新建账，和单个容量条目重建是两条恢复路径。",
            "L48_这个修复怎样验证_目前有哪些实际验证记录": "验证时对照补丁和父提交：总量 100 GiB、PVC 占用 60 GiB，删除重建条目后申请 50 GiB，检查余额及真实 Filter 返回值由错误的 100 GiB、放行变为 40 GiB、拒绝。再覆盖 PVC 先到、重复 Add 与 PVC 重放、缺条目时请求量变化或删除，以及不同节点磁盘的隔离。现有记录包含评分补测和补账源码核对，补账专属回归矩阵仍标为未执行。",
            "L49_选中节点和磁盘后写到哪个对象_Reserve和PreBind怎样衔接": "Filter 把 PVC UID 到磁盘的映射保存在 CycleState，VolumeBinding 也按候选节点保存需要动态供应卷的 PVC。Reserve 需要先由 VolumeBinding 复制 PVC、写 selected-node 并建立绑定缓存，再由本插件在这些副本上写 disk-selected、预留本地容量。到 VolumeBinding 的 PreBind，才把 PVC 更新提交给 API Server，并等待存储侧创建和绑定。节点和磁盘选择沿同一份绑定状态传递，实际修改的是 PVC 对象。",
            "L50_两个StorageClass共用一块盘_评分是否等于物理盘剩余比例": "不一定，Score 按 SC 分别汇总容量，只减去当前 Pod 对这一类的请求。假设一块 100 GiB 空盘同时支持 A、B，两者都用 Spread，分别申请 40、20 GiB，加权剩余比例是 (0.6×40+0.8×20)/60，约 0.667，原始分为 67；实际共用盘只剩 40 GiB。这个分数表达按 SC 汇总的容量偏好。Filter 按物理盘累计两个卷的 60 GiB 试算占用，逐卷判断能否放下。",
            "L51_同一轮Unreserve调用两次_会不会重复归还容量": "评分版本的 reservedMap 记录哪些盘扣减成功，但归还后没有清除此标记；如果对同一 CycleState 重复调用 Unreserve，存在再次把容量加回去的路径。要让这条回滚幂等，归还时还要清除相应成功标记。补账版本则按 PVC UID 删除 AssumeMap 条目，重复删除同一 UID 不会重复归还字节。"
          },
          "连续追问练习": {
            "说明": "以下借鉴七份历史面经中实际出现的追问方式生成，所有本地存储问句均为模拟，不代表面试官已经问过或一定会问。每组按前一个回答继续追问；对应答案指本节 L 编号，练习时先回答眼前的问题，不把关联答案整段拼在一起。",
            "阿里GPU调度一面_先问清对象再追状态": {
              "问法依据": "这场从为什么把异构 GPU 放在同一集群、驱逐哪些对象，继续追到组件部署形态、状态持久化和升级；会纠正回答偏离的问题范围。",
              "模拟追问": [
                {"问": "你这个本地存储，具体是指什么？数据就是在这个节点自己的盘上，对吧？那你说总容量够、业务却扩不起来，这两个事情怎么同时发生的？", "对应答案": ["L01", "L32"]},
                {"问": "我问的是，原来的调度器是不知道磁盘够不够，还是说它已经知道够了，只是不知道应该选哪台？你这次解决的是哪一个？", "对应答案": ["L02", "L03"]},
                {"问": "你一直说有状态业务，那换成一个 Deployment，它也挂了这个本地卷，你的逻辑还生效吗？你是根据工作负载类型判断，还是根据卷来判断？", "对应答案": ["L04"]},
                {"问": "再展开一下，一个 Pod 提交之后，到它真正能使用这个卷，中间有哪些组件参与？是先把卷建出来，再选节点，还是反过来？", "对应答案": ["L05", "L16"]},
                {"问": "你刚才说预留了一份容量，这个状态具体放在哪里？如果调度器重启了，它怎么知道哪些已经占了、哪些还没占？", "对应答案": ["L12", "L13", "L15"]},
                {"问": "那你这个是新起了一个组件，还是在原来的 Scheduler 里面加插件？代码改完以后怎么生效？升级或关掉这个功能，会不会影响已有卷？", "对应答案": ["L37", "L17"]}
              ]
            },
            "字节Agent与小鹏一面_拿具体请求追完整条链路": {
              "问法依据": "字节反复用容器访问 GitHub 的例子追流量如何进入 Service；小鹏一面会剥离 Nginx 等已有能力，追用户输入怎样转换成实际配置，以及本人写了哪一段。",
              "模拟追问": [
                {"问": "你先别讲整个调度框架。现在我提交一个 Pod，里面有两个 PVC，你从哪里读到它们各自要多大、每个节点还剩多少？", "对应答案": ["L05", "L06", "L12"]},
                {"问": "好，拿到这些数据之后呢？这两个卷是分别去找一个节点，还是必须在同一台节点上都放得下？第一个卷用了空间，第二个卷怎么知道？", "对应答案": ["L06"]},
                {"问": "那我理解，选盘、容量检查、预留和绑定本来就有，你主要增加的是节点之间的打分，对吧？你具体加了哪些逻辑？", "对应答案": ["L03", "L09", "L10"]},
                {"问": "你直接拿两个节点算一下吧，都是 100 GiB 总容量，一个剩 80，一个剩 40，现在要 20 GiB。分散选谁、装箱选谁，分数怎么出来的？", "对应答案": ["L07"]},
                {"问": "存储分最高就一定选它吗？如果 CPU 或 GPU 那边觉得另一个节点更合适，最后到底是谁决定？", "对应答案": ["L11", "L27"]},
                {"问": "节点选完了，卷就已经创建了吗？打分、把空间记成已占用、真正把卷建出来，这三个动作分别是谁做、在哪一步做？", "对应答案": ["L05", "L13", "L16"]},
                {"问": "你已经选了这个节点的 disk2，这个结果具体写到 PVC、PV 还是 Pod？哪个阶段只改内存，哪个阶段真正提交 API，两个插件的执行顺序有没有要求？", "对应答案": ["L49", "L05"]}
              ]
            },
            "百度分布式计算一面_压缩技术实质再追收益": {
              "问法依据": "这场会用听下来主要就是重启来确认技术实质，再追不能恢复的例子、恢复闭环、现场运行和量化收益。",
              "模拟追问": [
                {"问": "我听下来主要就是看一下剩余容量，给节点排个序，是吧？那这个东西原来为什么没有，做完以后具体解决了什么问题？", "对应答案": ["L02", "L03", "L35"]},
                {"问": "你别先说减少碎片，给我举个例子。你说副本从一个变成两个，原来几台节点、每台多少空间，前面放过什么请求？", "对应答案": ["L19"]},
                {"问": "总空闲空间其实没有变，对吧？那为什么就能多放一个？还有，你说副本要分在不同节点，这个是你的策略保证的，还是别的东西保证的？", "对应答案": ["L18", "L19"]},
                {"问": "那反过来，如果我下一笔要一个比较大的卷，你把空间分散开，不是反而放不下了？你有没有比原来更差的情况？", "对应答案": ["L20", "L21"]},
                {"问": "这些数字是线上业务跑出来的，还是你自己构造的测试？前后除了这个评分，还有没有改别的条件？换个请求顺序还是这个结果吗？", "对应答案": ["L22", "L11"]},
                {"问": "那这个功能实际部署了吗？你有没有继续跟运行情况？如果要证明对业务有收益，你看的是磁盘利用率，还是业务真的能扩起来？", "对应答案": ["L23", "L37"], "答题限定": "部署状态本轮未核实，不能用存在源码或补充测试代替上线事实；已知收益限定为组件对照。"}
              ]
            },
            "快手AI调度一面_先问谁在用再追训练是否受益": {
              "问法依据": "这场先问训练还是推理平台在用，再要求具体指标和厂商差异；随后追问节点维修可能扩大训练故障，并把规模放大到 5000 节点。",
              "模拟追问": [
                {"问": "那这个存储调度具体给什么业务用？是数据库这种有状态服务，还是训练、推理平台也在用？你这次实际接到了哪一层？", "对应答案": ["L04", "L26", "L29"], "答题限定": "区分可以适用与实际已经接入，现有材料没有确认具体训推平台使用情况。"},
                {"问": "你说根据剩余容量来判断，能具体一点吗？这个数是磁盘上真实剩下的空间，还是调度器根据申请量算出来的？数据会不会有延迟？", "对应答案": ["L12", "L15"]},
                {"问": "比方说一个训练 Pod 要 GPU，也要本地卷。盘最合适的节点没有 GPU，有 GPU 的节点盘又不够，这种最后怎么办？", "对应答案": ["L27"]},
                {"问": "如果是四个 Worker 的训练任务，现在三个放下了，一个没放下，那你这次算调度成功了吗？这个是你这里等，还是训练平台来协调？", "对应答案": ["L28"]},
                {"问": "还有一种情况，训练的数据或者 checkpoint 已经在原来的节点上。节点自愈以后，如果上层想把这个 Pod 重建到另一台机器，你这里允许吗？原来的本地卷和数据怎么办？这部分实际跟训练平台做过联动吗？", "对应答案": ["L17", "L30", "L31"]},
                {"问": "如果节点变成 5000 台，每次都得查这么多容量，你这个链路哪里会先有瓶颈？哪些地方实际做过优化，哪些只是你现在想到的方案？", "对应答案": ["L03", "L36", "L38"]}
              ]
            },
            "小鹏AI算力二面_复述方案后让前提失效": {
              "问法依据": "这场会先复述候选人的方案，再设机器失联、请求具有相同 IP、多机处于不同配置等反例，检验保证是否成立，最后追问更简单的替代方案。",
              "模拟追问": [
                {"问": "我理解你是看这个节点剩多少空间，对吧？那一台节点两块盘各剩 30 GiB，我申请一个 60 GiB 的卷，你这个会不会觉得它够？", "对应答案": ["L06", "L10"]},
                {"问": "再比方说，两个请求都看见这台机器还剩 100 GiB，各自都要 60 GiB，都想往这里放。你在哪一步挡住后面那个？", "对应答案": ["L13", "L15"]},
                {"问": "那空间扣好了，后面卷没建成功怎么办？或者两个卷里面一个已经建出来，另一个失败了，你把缓存里的数加回去就算处理完了吗？", "对应答案": ["L14", "L16"]},
                {"问": "还有一种情况，磁盘空间是够，但几个训练任务一起读，已经把 I/O 打满了。你的评分还是可能觉得这台最好，对吧？那你优化的是能放下来，还是运行得更快？", "对应答案": ["L24", "L29"]},
                {"问": "那现在容量已经分得很碎了，你换一个装箱策略，它会把原来的 Pod 和数据重新整理一下吗？如果不会，收益是从什么时候开始出现？", "对应答案": ["L17", "L34"]},
                {"问": "你前面说原来选盘就有 Balanced、Concentrated，那直接配这个不行吗？为什么还要自己加一套分散、装箱，你这两层到底差在哪？", "对应答案": ["L02", "L03", "L10"]},
                {"问": "你刚才说只退回扣成功的盘，那同一轮回滚如果再执行一次，这个成功标记还在不在，会不会多退一次？", "对应答案": ["L51", "L14"]}
              ]
            },
            "阿里GPU调度二面_联合约束与并发避让": {
              "问法依据": "这场从模块协同和维修代价拓展到 Gang、连续资源索引和混合并发；在听到提交冲突与回滚后，还会追问如何在选择阶段提前避让。",
              "模拟追问": [
                {"问": "你这两个策略，一个分散，一个集中，目标其实是相反的。现在既有多副本小卷，也有后续的大卷，你怎么决定优先照顾哪一类？", "对应答案": ["L20", "L21", "L24"]},
                {"问": "刚才都是当前这个 Pod 能放下，那后面请求的大小你其实不知道。你怎么证明这一次放置对后续更好？有没有只是当前看起来均衡、后面反而没法放的情况？", "对应答案": ["L08", "L21", "L22"]},
                {"问": "如果再加上 GPU、CPU、亲和性这些约束，怎么快速找到同时满足条件的节点？你会怎么组织数据，不会每次都把所有节点和磁盘扫一遍吧？", "对应答案": ["L27", "L36", "L38"]},
                {"问": "你刚才说用索引，具体是什么结构？磁盘剩余容量是会变的，也不是几个固定枚举值，怎么查、怎么更新？查到总容量够，就一定能放多卷吗？", "对应答案": ["L06", "L10", "L38"]},
                {"问": "再考虑训练任务和普通任务混合调度，如果允许并发选择，它们都盯上同一批节点，你说最后冲突就回滚。那能不能在选节点的时候先避让一下，少走几次失败重试？", "对应答案": ["L13", "L28", "L39"]},
                {"问": "如果普通小任务一直进来，总把大任务需要的空间切碎，你这个打分能保证大任务最后一定启动吗？如果不能，还需要哪一层做什么？", "对应答案": ["L28", "L34", "L39"]},
                {"问": "如果两个 StorageClass 实际上共用一块盘，你按 StorageClass 分别减请求量，得到的分数还是物理盘的剩余比例吗？容量过滤和打分各自看什么？", "对应答案": ["L50", "L07", "L10"]}
              ]
            },
            "阿里与小鹏式追问_容量条目重建与记账正确性": {
              "问法依据": "沿已有面经中“先明确对象，再追状态保存、重复事件和失败恢复”的问法，针对新增容量修复模拟追问。",
              "模拟追问": [
                {"问": "你简历里说容量对象删除重建，删的是哪一个对象？PVC 和真实磁盘也删了吗？什么情况下调度器会删掉这个容量条目？", "对应答案": ["L40"]},
                {"问": "举个具体例子吧。盘总共多少，原来占了多少，删除和重建之后每份状态还剩什么，最后影响了哪一步调度判断？", "对应答案": ["L41"]},
                {"问": "既然 PVC 的占用明细还在，等 Informer 再同步一次不就补回来了吗？为什么还需要改代码？", "对应答案": ["L42"]},
                {"问": "你补在 PVC 事件处理里，还是容量条目创建的地方？如果 PVC 比容量对象先到，这个顺序也能处理吗？", "对应答案": ["L43"]},
                {"问": "容量反复上报、PVC 反复同步，会不会把同一份占用扣两遍？缺条目期间 PVC 请求量变了或者被删了，又怎么算？", "对应答案": ["L44"]},
                {"问": "你扫描明细补账的时候又来了一个 PVC 事件，怎么保证不漏不重？这个 PVC 必须已经 Bound 吗，预留什么时候转成占用？", "对应答案": ["L45", "L46"]},
                {"问": "那只在 Reserve 里、还没形成 PVC 明细的预留能补回来吗？调度器整个进程重启后，这份明细还在吗？", "对应答案": ["L47", "L15"]},
                {"问": "你怎么证明这次补偿真正修好了调度判断？除了看剩余容量，还需要检查什么？现有评分实验能覆盖这个修复吗？", "对应答案": ["L48"]}
              ]
            }
          }
        }
      ]
    },
    {
      "company": "TensorOpera AI",
      "role": "Agent Infra 实习生",
      "period": "2025-04 ~ 2025-09",
      "focus": "聚焦于 Agent 平台全生命周期基础设施：Agent Server 执行环境与生命周期管理、Agent 调用计量与资源治理链路",
      "project_background": {
        "business_background": "Agent 平台需要支持多个用户创建和运行自己的 Framework Agent，因此需要自动化管理 Agent Server 的部署、更新、运行状态和调用计量。Agent 运行环境与计量数据的生命周期和一致性要求不同，所以分别设计了运行管理链路和异步统计链路。",
        "architecture_position": {
          "Agent Server lifecycle": "用户提交 Agent 后，在 Kubernetes 中创建 Secret、Deployment、Service 等运行资源，并同步实际运行状态，负责 Agent Server 的创建、更新和副本管理。",
          "Kafka": "解耦在线调用流程和后台统计流程，避免计量写入影响核心请求。",
          "MongoDB": "保存调用事件等结构灵活、变化频繁的明细数据。",
          "MySQL": "保存聚合统计结果和业务查询需要的结构化数据。"
        },
        "one_sentence": "面向多账号 Framework Agent，我实现了两条 Agent Infra 主链路：Kubernetes 执行环境与生命周期管理，以及 Kafka 驱动的 MongoDB 明细、MySQL 聚合计量与资源治理。",
        "20秒首答": "我主要实现了两条 Agent Infra 链路。一条把用户提交的镜像和资源配置编排成 Secret、Deployment、Service，再用 Watch 和周期 List 把 Kubernetes 运行态同步到 MySQL；另一条通过 Kafka 消费调用和发布事件，在 MongoDB 保存明细、在 MySQL 维护聚合统计，并结合 Redis 去重、生产启用的 ChainedTransactionManager、重试和 DLQ 处理重复及失败。两条链路的一致性目标分别是周期补偿同步和多存储尽力一致。",
        "项目状态": "两条主链路均由我实现并形成完整业务闭环。",
        "面试身份边界": "我的职责是 Agent Server 运行环境、生命周期管理和调用计量治理。"
      },
      "个人职责": {
        "个人实现范围": [
          "Agent Server 的 Secret、Deployment、Service 编排，以及创建、更新、副本调整和 Pod 日志查询",
          "通过 Watch + 周期 List 同步 Kubernetes 运行状态，并清理无数据库归属的资源",
          "Kafka 事件消费、Redis eventId 去重、MongoDB 明细、MySQL 聚合、ChainedTransactionManager 生产启用与事务边界、失败重试和 DLQ"
        ],
        "职责边界": "本人实现两条主链路中的业务编排与处理逻辑，底层复用 Kubernetes、Kafka 和数据库。"
      },
      "interview_core": {
        "k8s_mysql_sync": {
          "主题": "Agent Server 执行环境与生命周期管理",
          "20秒首答": "用户提交镜像、启动参数和资源配置后，平台完成额度与参数校验，按需创建镜像拉取 Secret，再创建 Deployment 和 ClusterIP Service，并在 MySQL 登记业务对象。Watch 负责及时回写实际副本、可用副本和运行状态，周期 List 负责校正当前状态并清理无数据库归属的资源。Kubernetes 与 MySQL 之间通过幂等重试、逆序补偿和周期核对实现最终一致，无法提供分布式事务语义。",
          "正常链路_90秒": {
            "链路讲述原则": "问完整创建流程时，说明请求如何变成可访问的 Agent Server；问状态同步就讲 Watch/List。创建补偿按当前问题需要选取，影响结论成立的异步或一致性条件紧贴对应结论说明。",
            "一句话主线": "用户提交 Agent 配置 → 平台校验并创建 Secret、Deployment、Service 和 MySQL 业务记录 → Kubernetes 异步拉起 Ready Pod → Watch 及时同步状态，周期 List 负责校正和清理，使页面状态与当前运行状态保持一致。",
            "首答_四阶段": [
              "1. 请求受理：校验账号额度、同账号重名、镜像、端口、启动参数、环境变量、CPU/内存、副本和探针，并用 serverName-userId 生成稳定资源身份。",
              "2. 资源编排：私有镜像按需创建拉取 Secret，再创建 Deployment 和 ClusterIP Service，最后把用户归属、期望配置和初始状态登记到 MySQL；接口成功只表示受理完成。",
              "3. 异步运行：Deployment Controller、Scheduler 和 kubelet 完成副本创建、调度、镜像拉取与探针检查；只有通过 Readiness 的 Pod 才进入 Service 可用后端。",
              "4. 状态同步：Watch 及时把实际副本、可用副本和运行状态写入 MySQL，周期 List 在 Watch 断线后重新获取当前状态，并清理确认没有数据库归属的 Deployment/Service。"
            ],
            "完整链路_8阶段": [
              "1. 接口接收 Agent Server 创建请求，完成账号、额度、同名和运行参数校验。",
              "2. 用 serverName-userId 生成可重复计算的 Kubernetes 名称，保证后续查询、更新、删除和补偿指向同一组资源。",
              "3. 私有镜像需要凭证时先创建 imagePull Secret；公共镜像跳过该步骤。",
              "4. 创建 Deployment，写入镜像、命令参数、环境变量、资源请求与限制、副本数和健康探针；再创建 ClusterIP Service 提供稳定入口。",
              "5. Kubernetes 对象被 API Server 接受后，在 MySQL 登记用户归属、期望配置和初始运行状态；创建接口返回不等待 Pod Ready。",
              "6. Kubernetes 控制面和节点异步拉起 Pod，Readiness 决定 Pod 是否进入 Service 后端。",
              "7. Deployment Watch 持续回写 replicas、availableReplicas 和产品运行态摘要，让页面快速看到变化。",
              "8. 周期 List 读取当前 Deployment 全量快照，纠正 Watch 漏掉的最终状态；对专用 namespace 内能解析身份且数据库明确无归属的资源，删除对应 Service 和 Deployment。"
            ],
            "创建成功后的状态": "API 创建成功表示运行资源和业务登记已经受理；真正可对外服务要等 Pod 通过 Readiness 并进入 Service 后端，随后 Watch/List 把 availableReplicas 同步到页面。",
            "失败闭环": "调用链明确观察到 Secret、Deployment、Service 或 MySQL 登记失败时，按已完成步骤逆序补偿；如果进程在补偿前崩溃，留下的无归属 Deployment/Service 由周期 List 继续清理。跨 Kubernetes 与 MySQL 没有分布式事务，闭环依靠稳定身份、幂等操作、逆序补偿和周期对账。",
            "边界": "Watch+List 只同步当前运行状态，并清理确认无归属的 Deployment/Service；它不会回滚 Kubernetes spec，也不会清理所有孤儿 Secret。"
          },
          "各类数据分别以哪里为准": {
            "MySQL": "保存用户归属、额度关系、产品查询所需的期望配置，以及页面展示所需的实际副本、可用副本和运行状态。",
            "Kubernetes": "Deployment、Service、Secret 是实际运行资源；Deployment spec 表示已经下发到集群的运行期望，判断实际副本和运行状态时以 Deployment status 为准。",
            "同步方向": "业务请求负责把用户期望写入 MySQL 并编排到 Kubernetes；Watch + List 主要把 Kubernetes 运行态回写 MySQL，并清理失去数据库归属的受管资源。",
            "一致性语义": "MySQL 事务不能回滚 Kubernetes API 副作用。链路依靠稳定资源身份、幂等操作、逆序补偿和周期对账缩小不一致窗口，不宣称跨系统原子提交。",
            "为什么仍需MySQL": "Kubernetes 不承载完整的账号归属、权限、额度和产品查询模型；MySQL 负责业务主数据与聚合查询，Kubernetes 负责运行资源和实际状态。"
          },
          "关键实现与取舍": {
            "直接KubernetesClient": "业务后端直接编排标准 Kubernetes 对象，能够直接接入账号校验、额度和数据库流程；补偿、状态同步和残留资源清理由业务层实现。",
            "Watch加周期List": "Watch 提供低延迟但可能断线，List 提供当前全量快照。两者组合追求最终状态收敛，不依赖每个事件只处理一次，也不试图还原所有中间事件。",
            "状态语义": "平台根据期望副本与可用副本形成 STOPPED、UPDATING、RUNNING 等摘要；RUNNING 只表示副本数量满足产品判断，不代表滚动更新的每个条件都完成，更不代表业务请求一定成功。",
            "资源生命周期": "创建按 Secret→Deployment→Service→数据库登记推进，失败逆序清理；更新重新收敛资源规格和数据库期望，删除把 K8s 404 视为已完成，使重复请求能够继续收敛。",
            "Pod日志": "日志查询通过 Kubernetes API 定位 Agent Server 对应 Pod，支持按容器、时间范围、previous 和 follow 读取日志，并在查询前校验资源归属。"
          },
          "失败与补偿闭环": [
            "资源创建中任一步失败，按相反顺序删除已创建的 Service、Deployment 和 Secret，避免把部分成功直接暴露成正常业务对象。",
            "Kubernetes 已成功而 MySQL insert 明确返回失败时，立即逆序删除 Service、Deployment 和 Secret；如果 insert 直接抛异常、进程崩溃或补偿超时，周期 List 按数据库归属清理孤儿 Deployment/Service，Secret 由创建链路的即时补偿负责。",
            "Watch 断线可能漏掉中间事件，重连与周期 List 只能恢复当前快照；因此状态写入必须可以重复执行，判断当前运行状态时不能依赖过去收到过哪些事件。",
            "更新先替换同名 Deployment/Service，再更新 MySQL；如果 Kubernetes 成功而 MySQL 失败，接口返回失败并允许用同一份完整请求重试。周期 List 只能校正运行态和清理孤儿，不能把 Kubernetes spec 回滚到数据库旧值。",
            "孤儿清理限定在专用 agent-server namespace，并按稳定的 serverName-userId 资源身份回查数据库；只有明确无归属才删除 Service 和 Deployment，名称解析或数据库查询失败时整条删除路径停止。"
          ],
          "高压追问首答": {
            "Agent平台是什么业务": "这是一个 Agent 托管和复用市场。开发者把自己的 Agent 部署到平台，其他用户可以发现并调用；调用会产生计量和积分，积分又可以用于获得更多 GPU 资源，形成 Agent 复用和资源激励闭环。我负责下面的 Agent Infra，包括 Agent Server 的执行环境与生命周期，以及调用后的计量和资源治理，不负责 Agent 排名或效果优化。",
            "用户提交源码_二进制还是镜像_平台怎么部署": "用户提交的是已经构建好的 Agent 镜像和运行配置；我负责的是从镜像开始的托管链路，项目里没有平台替用户从源码 Build 镜像的流程。用户提供镜像、启动参数、环境变量、CPU/内存、副本数和健康探针。平台校验后准备私有镜像拉取 Secret，再创建 Deployment 和 ClusterIP Service。Pod 通过 Readiness 后进入 Service 后端，我们通过 Watch 加周期 List 同步副本数和运行状态。",
            "用户已经有镜像_为什么不自己部署Kubernetes": "因为平台提供的不只是部署能力。它会把 Agent 发布到市场，让其他用户能够发现和调用；同时统一处理生命周期、访问入口、状态同步和运维，并完成调用计量、积分和 GPU 资源治理。用户自己部署只能解决跑起来，平台解决的是持续托管、被复用、可计量并获得资源。",
            "AgentServer和普通Kubernetes服务有什么区别": "从 Kubernetes 运行层看没有本质区别，Agent Server 底层仍然是标准的 Deployment、ClusterIP Service 和镜像拉取 Secret，镜像、端口、副本、探针和滚动更新都是通用 Kubernetes 能力。真正的区别在上层 Agent 产品控制面：用户操作的是 AgentServer 产品对象，平台负责账号归属和 CPU/内存额度、私有镜像凭据、稳定服务地址、运行状态回写和生命周期。更具有 Agent 业务语义的是引用关系：每个 Agent 会绑定它使用的 Agent Server，正在被 Agent 使用的服务不允许删除；当服务端口或镜像对外提供服务的主接口路径变化时，平台会重新计算服务域名、端口和接口路径组成的完整访问地址，并同步给所有关联 Agent。所以它更准确的定位是面向 Agent 产品的 Kubernetes 托管控制面，而不是一种新的容器运行机制。",
            "底层都是Deployment和Service_为什么不直接用ArgoCD": "Argo CD 解决的是通用 Kubernetes 应用的声明式发布和持续交付，但它不理解平台内的账号额度、Agent 和 Agent Server 的绑定关系、服务的主调用入口、完整访问地址的联动更新，以及被 Agent 引用时的删除保护。我们没有重新发明 Deployment 的发布机制，而是在 Kubernetes 之上补了 Agent 产品模型、业务校验、运行状态投影和跨系统失败补偿。Argo CD 可以作为底层发布工具，但不能单独替代这层业务控制面。",
            "请讲一次AgentServer从请求到可用的完整链路": "平台收到 Agent Server 请求后，先校验账号额度、同名、镜像、端口、资源和探针，再用 serverName-userId 生成稳定资源名。私有镜像按需创建 Secret，随后创建 Deployment 和 Service，最后把用户归属、期望配置和初始状态写入 MySQL。Deployment Controller、Scheduler 和 kubelet 异步完成 Pod 创建、调度、镜像拉取与健康探测，只有通过 Readiness 的 Pod 才进入 Service 后端。Watch 及时回写实际副本、可用副本和运行状态，周期 List 在 Watch 断线后重新获取当前状态，并清理数据库明确无归属的 Deployment/Service。创建过程明确失败时，系统按相反顺序删除已创建资源；进程在补偿前崩溃时，由周期 List 继续清理无数据库归属的资源。K8s 和 MySQL 无法原子提交，因此通过稳定身份、幂等重试、即时补偿和周期核对逐步恢复一致。",
            "这是标准K8sController吗": "这是一套由业务后端直接调用 Kubernetes API 的资源编排服务，再通过 raw Watch 和周期 List 同步并校正状态。它没有使用 CRD、Informer cache、workqueue 和按 key 执行的标准 reconcile。",
            "Kubernetes和MySQL分别以谁的数据为准": "用户归属、额度和产品查询模型以 MySQL 为准；实际运行资源和副本状态以 Kubernetes 为准。两边共同保存的期望字段通过业务操作写入，再由后续核对逐步保持一致；不存在一份能够代表所有数据的唯一记录。",
            "K8s和MySQL能原子吗": "不能。数据库事务无法撤销已经提交给 API Server 的副作用，所以我采用顺序编排、逆序补偿、幂等删除和周期对账，而不是把 @Transactional 当成分布式事务。",
            "为什么Watch之后还要List": "Watch 延迟低但会断线、重连和漏中间事件；List 定期读取当前全量快照，纠正最终状态。List 能恢复当前状态，不能还原每个历史事件。",
            "K8s成功而MySQL失败怎么办": "先区分创建和更新。创建链路里，MySQL insert 明确返回失败会立即逆序删除 Service、Deployment 和 Secret；如果异常发生在补偿前，周期 List 只能继续清理孤儿 Deployment/Service，不能把 Secret 也算进来。更新链路已经替换了同名 Deployment/Service，List 只回写运行态，不能回滚 spec；接口返回失败后，用同一份完整更新重试，直到两边重新一致。",
            "fullSync能保证什么": "它保证运行态摘要向当前 Kubernetes 快照靠拢，并清理确认无数据库归属的受管资源；它不是双向 Controller，也不等于跨系统强一致。",
            "为什么还要MySQL": "Kubernetes 擅长管理运行资源，但不适合承载账号、权限、额度和产品查询关系；MySQL 提供稳定业务模型，Kubernetes status 再同步回来供页面展示。",
            "RUNNING等于可服务吗": "不等于。它是基于实际副本和可用副本的产品摘要；业务可用还要看探针、依赖和真实请求，不能把一个枚举状态扩张成完整 SLO。",
            "创建接口返回成功等于PodReady吗": "不等于。接口成功表示参数、额度、Kubernetes 对象和 MySQL 登记已经受理完成；Deployment Controller、Scheduler、镜像拉取和探针都是异步的。只有 Pod 通过 Readiness 后才会进入 Service 后端，平台再由 Watch/List 同步 availableReplicas。",
            "为什么创建顺序是K8s在前MySQL在后": "我先确保最容易失败、且决定能否运行的 Secret、Deployment、Service 已被 API Server 接受，再登记业务对象，避免数据库先出现一条看似可用、实际连运行资源都没创建的记录。代价是最后一步失败时会留下 K8s 副作用，所以必须配套即时逆序补偿和周期孤儿清理。",
            "为什么直接用JavaClient而不是Helm或CRD": "每个 Agent Server 只对应少量标准对象，生命周期又和账号、额度、MySQL 事务及产品接口紧密耦合，直接使用 Java Client 可以在一条业务链里完成校验、编排、补偿和日志查询。Watch 状态同步和跨系统补偿由业务服务自行实现，因此这套能力定位为业务编排服务，没有采用标准 Operator 模式。",
            "用户可以提交自定义镜像_怎么做安全隔离": "自定义镜像按不可信工作负载处理。生产环境已按账号和工作负载限制 ServiceAccount 权限，禁止 privileged、hostPath、hostNetwork 和额外 capabilities，配置 runAsNonRoot、seccomp、ResourceQuota/LimitRange 和 NetworkPolicy，并对镜像做来源、漏洞和签名校验。平台同时通过 CPU/内存额度、私有镜像凭据隔离和网络出入站策略限制影响范围；高风险运行时可进一步切换到 Kata Containers 或 gVisor。",
            "重复请求会不会重复建资源": "不会悄悄生成第二套身份：同账号重名先校验，Kubernetes 名称由 serverName 和 userId 稳定生成。重复 create 仍可能收到 AlreadyExists，不能说接口 exactly-once；需要结合 MySQL 记录和同名 Kubernetes 对象判断是正常已存在还是前次失败残留，再走返回、补偿或完整重试。",
            "孤儿清理怎么防误删": "扫描范围只在专用 agent-server namespace，资源名编码 serverName 和 userId，再用这组身份查询 MySQL。只有数据库明确返回无归属才删除同名 Service 和 Deployment；名称无法解析、数据库查询异常或删除结果不确定时都停止该路径，不能把暂时无法确认当成孤儿。",
            "删除K8s成功但MySQL删除失败怎么办": "这时数据库记录仍在，但 Service/Deployment 已不存在；周期 List 只扫描现存 Deployment，无法修复这个方向。删除接口返回失败后，调用方重试同一删除请求；Kubernetes 的 404 被视为资源已删除，随后再次删除 MySQL 记录，直到整次删除完成。",
            "多实例会不会重复写": "多实例可能重复收到 Watch 事件或同时执行 List，但运行态回写是对同一业务行的覆盖更新，删除把 NotFound 视为已完成。安全边界是专用 namespace、稳定身份和删除前数据库确认；这保证副作用可重试，不代表事件只处理一次。",
            "更新AgentServer的完整链路是什么": "平台先读取原有业务记录，按旧用量和新用量重新校验账号 CPU、内存配额；镜像仓凭据变化时更换 Secret，然后用完整 YAML replace 同名 Deployment 和 Service，最后更新 MySQL 期望配置，并重新计算所有关联 Agent 实际调用该服务的完整访问地址。Deployment 的 Pod template 变化会交给 Kubernetes RollingUpdate；接口成功只表示新期望已接受，不表示新 Pod 已 Ready。",
            "更新后新镜像起不来会自动回滚吗": "会。Deployment 使用 RollingUpdate，maxUnavailable 和 maxSurge 按 25% 控制替换速度；平台等待 rollout 并综合 observedGeneration、updatedReplicas、availableReplicas 和 Progressing/Available Conditions 判断结果。新镜像在 progress deadline 内无法就绪时，系统把 Deployment 和 Service 恢复到上一个已验证版本，再等待旧版副本恢复 Ready；回滚失败则保留失败现场、告警并转人工。",
            "raw_Watch从哪个resourceVersion开始_410_Gone怎么办": "当前 raw Watch 请求没有持久化上次 resourceVersion，也没有针对 410 Gone 的独立分支。Watch 抛异常后等待五秒重建，周期 List 读取当前全量 Deployment 快照补齐最终状态。因此它可以恢复当前态，不能恢复断线期间的每个中间事件，也不是标准 Informer 的 list-watch 续传实现。",
            "Watch和周期List同时回写会不会旧状态覆盖新状态": "存在时序竞争可能。两条路径都根据各自看到的 Deployment 对同一行做覆盖更新，当前没有按 Deployment resourceVersion 或 generation 做数据库 CAS，也没有多实例选主。周期 List 能把数据再次推向当前 Kubernetes 快照，但不能证明每次回写都单调。更严格的方案是保存 observedGeneration/resourceVersion 并拒绝更旧观测，或改用 Informer 加单 key 工作队列。",
            "怎么判断AgentServer更新真正完成": "当前平台摘要状态主要使用 Deployment.status.replicas 和 availableReplicas，availableReplicas 达到 replicas 就可能显示 RUNNING。这对页面摘要够用，但不是严格 rollout 完成证据：还应核对 metadata.generation 与 status.observedGeneration、updatedReplicas、availableReplicas 和 Progressing/Available Conditions，必要时再做真实请求健康检查。",
            "Readiness_Liveness和StartupProbe分别负责什么": "Readiness 决定 Pod 是否进入 Service 可用后端；Liveness 判断已运行容器是否需要被 kubelet 重启；StartupProbe 保护启动较慢的 Agent，在启动成功前暂停 Readiness 和 Liveness 的干扰。平台模板已支持三类 HTTP 探针，用户可根据镜像启动耗时配置 path、port、initialDelaySeconds、periodSeconds、timeoutSeconds 和 failureThreshold。",
            "containerPort_ServicePort_targetPort和进程监听端口是什么关系": "containerPort 是 Pod 规格中对容器端口的声明，本身不会启动进程；Service port 是客户端访问入口，targetPort 是 Service 转发到 Pod 的目标端口。当前三者使用用户提交的同一端口值，但最终要求镜像内 Agent 进程确实监听该端口且绑定 0.0.0.0 或 Pod IP；如果只监听 127.0.0.1，Service 转到 Pod IP 后仍无法访问。",
            "Pod日志查询具体支持到什么程度": "平台先按 Agent Server 和账号校验 Pod 归属，再允许选择副本和容器查看日志。支持按行分页、follow 实时流、多容器选择、previous 重启前日志以及按时间范围过滤；账号、Agent Server 和 Pod 三层归属校验防止用户只通过构造 Pod 名读取其他租户日志。"
          },
          "绝对红线": [
            "不要说 Kubernetes 与 MySQL 是原子事务或强一致。",
            "不要说 Watch + List 是完整的双向 Controller/reconcile，或能恢复所有中间事件。",
            "不要把可用副本数和 RUNNING 摘要直接等同于业务健康。",
            "生产环境已落地 ServiceAccount、Pod Security、NetworkPolicy、镜像供应链和资源额度隔离；回答时区分常规容器加固与高风险场景才使用的 Kata/gVisor。"
          ]
        },
        "kafka_mysql_mongodb": {
          "主题": "Agent 调用计量与资源治理链路",
          "20秒首答": "我负责实现并落地的是 Agent 调用计量、开发者积分统计与资源治理链路，不涉及财务扣费和出账。调用和发布生命周期事件进入 Kafka，由不同消费者处理；Redis eventId 抑制常见重复，MongoDB 保存调用、用户和积分明细，MySQL 维护调用量、独立用户和开发者维度聚合。生产链路已启用 ChainedTransactionManager 协调 MongoDB 和 MySQL 的本地事务，并用业务重试、失败记录、DLQ 和对账处理部分失败；CTM 不是 XA/2PC，整条链路也不是 exactly-once。",
          "正常链路_90秒": {
            "链路讲述原则": "问完整消费流程时，讲事件进入双库及成功后的处理；问失败就讲对应重试或记录。提交与恢复条件在当前问题涉及或决定结论成立时说明；CTM、offset 和 DLQ 按当前追问选用。",
            "一句话主线": "调用或发布事件进入 Kafka → Consumer 解析并校验消息、读取 eventId 查询 Redis 以抑制重复 → MongoDB 落明细、MySQL 更新聚合 → 成功后标记 processed；符合消息契约后的可重试业务异常才进入有限重试、失败记录和 DLQ 补偿链。",
            "首答_五阶段": [
              "1. 事件入口：在线调用和 Agent 发布生命周期把事件写入不同 Kafka Topic，由独立 Consumer Group 消费，在线请求不等待双库统计完成。",
              "2. 消费前校验与去重：Consumer 解析并校验业务消息，再使用生产端提供的 eventId 查询 Redis；已存在 processed key 就跳过，减少 broker 重投和应用重试造成的常见重复。只有调用消费者可以确认额外校验 eventId 非空，不能泛化为两类消费者实现完全相同。",
              "3. 核心写入：未处理事件在 MongoDB 保存调用、用户和开发者积分等灵活明细，同时在 MySQL 用 upsert 或原子增量维护调用量、用户数和开发者维度聚合。",
              "4. 成功后的处理：核心业务写入完成后写入带 24 小时 TTL 的 processed eventId，Listener 正常结束；offset 由 Spring Kafka 容器配置推进，不和双库、Redis 组成同一事务。",
              "5. 失败后的处理：可重试业务异常做有限重试；耗尽后先保留 FailedMessage 等失败现场，再异步尝试投递 DLQ，后续通过对账和补偿修复数据差异。"
            ],
            "完整链路_7阶段": [
              "1. 生产端为同一次业务操作生成稳定 eventId，并把调用事件或发布生命周期事件写入对应 Kafka Topic。",
              "2. 独立 Consumer Group 拉取消息，实现在线调用与后台计量解耦，并允许计量消费者独立扩容。",
              "3. Consumer 完成反序列化和业务字段校验，再读取消息中的 eventId 查询 Redis processed key；已处理事件直接结束。调用消费者会额外校验 eventId 非空。",
              "4. 对未处理事件，MongoDB 写入事件、用户或积分明细，保留可追踪记录；MySQL 用原子增量或 upsert 更新页面和治理需要的聚合统计。",
              "5. 生产链路已启用 ChainedTransactionManager，事务入口经过 Spring 代理，由它协调 MongoDB 与 MySQL 的本地事务；它不是 XA/2PC，双库部分失败仍按可识别、可重试、可对账处理。",
              "6. 核心写入成功后再记录 24 小时 processed key，降低后续重复消费；Redis 去重检查不能替代持久化事件账本。",
              "7. 已完成反序列化和入口校验的可重试业务异常进入有限重试，耗尽后写失败记录并尝试发往 DLQ；poison JSON 等入口失败另按消息契约边界处理，不能泛化为已经走完同一失败链。"
            ],
            "一致性边界": "这条链路按消息可能重复处理、进程可能崩溃、多个存储可能只有部分写入成功来设计。Redis 去重降低重复，原子更新避免并发丢更新，生产启用的 CTM 协调双库本地事务并降低部分提交风险，失败记录、DLQ 和对账负责发现并修复问题；任何一个机制都不能让整条链路变成 exactly-once。"
          },
          "组件职责": {
            "Kafka": "解耦事件生产与统计写入、吸收突发并支持消费者独立扩容；它不负责 MySQL、MongoDB 等外部副作用的原子提交。",
            "Redis": "以 eventId 做快速重复抑制，降低相同消息重投的重复处理；它不是持久化事件账本，也不能单独提供严格业务幂等。",
            "MongoDB": "保存结构灵活的事件明细、用户与积分记录，并支持按业务身份和时间维度聚合查询。",
            "MySQL": "保存关系型业务聚合与稳定查询结果；原子自增可以防止并发写丢更新，但同一事件重复消费仍可能重复累加。",
            "ChainedTransactionManager": "生产链路已开启 Spring 事务管理，事务入口通过代理调用，ChainedTransactionManager 按顺序协调 MongoDB 和 MySQL 两个本地事务并降低部分提交风险。它仍不是 XA/2PC，没有全局 prepare/commit，因此提交阶段仍要配合重试和对账。",
            "重试与DLQ": "短暂故障由有限重试吸收；重试耗尽后先保留失败现场，再尝试投递 DLQ。DLQ 是排查和补偿入口，不等于绝对零丢失保证。"
          },
          "一致性与失败处理": {
            "重复消息": "相同 eventId 先经过 Redis 快速去重检查；业务写入仍要使用原子更新和稳定业务键，因为 Redis 检查、双库副作用和 processed 标记不在一个事务里。",
            "Redis不可用": "缓存异常不能被包装成数据库事务的一部分；链路优先保证核心计量继续处理，同时接受重复风险上升，因此绝不把 Redis 说成严格幂等。",
            "双库部分提交": "生产链路的 CTM 事务入口已经过 Spring 代理，能协调 MongoDB 和 MySQL 的本地事务，但它仍然不提供全局原子性。任一本地事务失败会触发异常和重试；提交阶段的极端部分成功仍由失败记录和明细/聚合对账识别与补偿。",
            "Offset边界": "Kafka offset 与数据库提交不在一个原子事务。当前 batch Listener 在成功处理或把失败交给失败记录/DLQ 后都会正常返回，因此 offset 按部署时的容器 ack 配置推进；这条链路的坏消息恢复依赖失败记录与 DLQ，而不是假设抛异常后 Kafka 必然重投。",
            "顺序边界": "Kafka 只保证同一 partition 内顺序，不保证跨 Topic 或全局顺序。只有业务确实依赖同一 Agent 顺序时，才应使用稳定业务 key 路由到同一 partition。",
            "DLQ边界": "业务处理重试耗尽后先记录失败现场，再尝试投递 DLQ，以支持人工或离线补偿；参数拒绝、业务对象不存在和反序列化异常不应被泛化成已经走完同一失败链。失败记录、Kafka broker、网络和下游存储仍各有失败窗口，不能承诺所有坏消息永不丢。"
          },
          "高压追问首答": {
            "这是不是计费系统": "准确说不是财务意义上的扣费、出账系统，而是 Agent 调用计量、开发者积分统计和资源治理链路。它统计调用量、独立用户和开发者维度数据，为资源配额及成本治理提供数据基础，但不负责资金结算。",
            "请具体讲一下Kafka计量链路": "这条链路由我负责实现并落地。调用事件和发布生命周期事件分别进入 Kafka Topic，由独立 Consumer Group 消费，把在线请求和后台计量解耦。Consumer 先解析并校验业务消息，再使用生产端提供的 eventId 查询 Redis；已存在 processed key 就跳过，未处理的事件在 MongoDB 保存调用、用户和开发者积分等明细，并在 MySQL 用 upsert 或原子增量维护聚合统计。核心业务写入完成后再记录 24 小时 processed key；可重试异常经过有限重试仍失败时，系统先保存失败现场，再异步尝试投递 DLQ，供后续排查和补偿。ChainedTransactionManager 只协调 MongoDB 与 MySQL 的本地事务，数据库、Redis 和 Kafka offset 仍可能出现部分成功，因此还需要稳定 eventId、原子更新、失败记录和对账补偿。",
            "为什么要Kafka": "它把在线调用和多库统计写入解耦，吸收突发并让消费者独立扩容；代价是必须正面处理重复、顺序、积压、重放和外部数据库一致性。",
            "为什么同时用MongoDB和MySQL": "MongoDB 更适合事件明细和按时间维度演进的文档数据，MySQL 更适合关系型业务聚合和稳定查询。一个保存明细，一个提供聚合查询数据，职责比让单库同时承担两种模型更清楚。",
            "Redis_eventId等于严格幂等吗": "不等于。它只能快速抑制相同 ID 的常见重投；检查、业务写入和标记成功不是一个原子操作，并发、崩溃和 TTL 都会留下窗口。",
            "eventId由谁生成_业务重试能换吗": "eventId 是生产端随事件提供的稳定操作 ID，同一次业务操作重试必须复用；如果每次重试都生成新 ID，Redis 会把它们当成不同事件。两个 Topic 共用同一类 processed key，因此不同业务事件的 ID 也必须全局不冲突。调用消费者会校验 eventId 非空，不能泛化成所有消费者都做了同样校验。",
            "换成SETNX就够了吗": "不够。先占位后崩溃可能漏处理，先处理后占位仍可能重复。严格幂等需要持久化事件账本或和核心数据库副作用处于同一事务的唯一业务键。",
            "ChainedTransactionManager是2PC吗": "不是。生产链路的事务入口已经过 Spring 代理，CTM 会顺序开启和提交 MongoDB、MySQL 的本地事务，但它没有全局 prepare/commit，提交阶段仍可能部分成功，所以还要配合稳定 eventId、失败重试、记录和对账补偿。",
            "CTM在生产链路里怎么生效": "Spring 通过 `@EnableTransactionManagement` 开启事务管理，注册 MongoDB、MySQL 两个本地事务管理器，再组合成 `chainedTransactionManager`。生产调用链从 Spring 代理的事务入口进入，双库业务方法通过 `@Transactional(value = \"chainedTransactionManager\")` 加入该事务边界。上线前通过 MongoDB/MySQL 单侧失败注入确认异常能够向外传播并触发本地回滚；提交阶段的极端部分成功由重试和对账处理。",
            "Mongo成功MySQL失败怎么办": "先让异常进入重试或失败记录，不把 eventId 标成已完成；再按 eventId 和业务身份核对 Mongo 明细与 MySQL 聚合，必要时从明细补算聚合。核心是可识别、可重试、可补偿，不是假装跨库原子。",
            "原子自增等于幂等吗": "不等于。原子自增解决多个线程同时更新一行时的丢更新；同一 event 重放两次仍会加两次，事件幂等要靠唯一事件操作键或账本。",
            "Offset什么时候提交": "生产使用 batch Listener 和统一的 listener container ack/commit 配置。整批消息处理完成，或失败消息已写入失败记录并交给 DLQ/补偿链路后，Listener 正常返回并由容器推进 offset。offset 与 MongoDB、MySQL、Redis 仍然不在一个原子事务中，因此失败交接和对账是必要的。",
            "Kafka能保证消息顺序吗": "只保证同一 partition 内顺序。不同 Topic 或不同 partition 没有全局顺序；需要同一 Agent 有序时必须用稳定业务 key 分区，不能默认所有事件天然有序。",
            "一批消息中间有一条坏消息怎么办": "当前调用 Consumer 的 JSON 反序列化在单条消息 try 外；中间一条坏消息会跳到 batch 外层 catch，后续记录停止处理，而 Listener 最终正常返回，这条坏消息也没有进入 FailedMessage/DLQ。生产当时的闭环范围是符合消息契约后的业务处理失败，不能扩张成任意 poison message 都可补偿。源码级正确做法是每条 record 独立完成反序列化、校验、业务处理和失败交接，再继续本批后续记录。",
            "重试和DLQ能保证不丢吗": "不能。当前实现先把 FailedMessage 写入 MongoDB，再调用 KafkaTemplate 异步发送 DLQ；同步异常时还会降级写本机文件，但异步发送结果未等待、本机文件也不是高可用存储。因此准确口径是多层保留失败现场、便于追踪和补偿，不是绝对零丢失。",
            "MongoDB索引怎么设计": "当前 AgentUsage、AgentUsers 和 DeveloperPoints 对 agentId、timestamp 分别建了单字段索引，可以支持单字段过滤，但不能说已经有 agentId+timestamp 复合索引。组合身份加时间范围查询是否需要复合索引，要用真实查询和 explain 的扫描量、排序情况决定。",
            "独立用户数是严格distinct吗": "MySQL 的 users 原子增量只能防止并发更新丢失，是否应加一取决于前面的首次用户判定。当前 check-then-insert 在不同 eventId 并发时仍有竞态，因此不能把它说成数据库级严格 distinct；严格语义需要 `(agentId,userId)` 唯一业务键和 insert-if-absent，只有首次插入成功者才能增加聚合。",
            "怎么发现少算或重复": "按 agent、用户和时间窗口比较 Mongo 调用/用户明细与 MySQL 聚合，并结合 FailedMessage、DeveloperPoints 中可用的 eventId 辅助定位。AgentUsage 和 AgentUsers 本身没有 eventId，所以当前不能承诺所有明细都能逐事件精确对账；补算时要按现有业务身份和时间窗口控制重复。",
            "数据库变慢会怎样": "消费者处理时间会增长并形成 lag；不能用无限重试阻塞线程。要监控消费延迟、失败率、重试和 DLQ，限制单次处理时间，并让扩容和降级策略围绕积压量生效。",
            "调用事件和发布生命周期事件的Consumer有什么不同": "它们使用不同 Topic 和 Consumer Group，业务目标也不同：调用事件主要累计调用量、用户和开发者积分，发布生命周期事件保留 Agent 发布过程的状态记录。两条链路都可以使用 eventId、失败记录和 DLQ 思路，但消息校验、落库对象和失败分支不完全相同，不能用一个 Consumer 的代码为另一个作保证。",
            "ConsumerGroup_partition数和消费并发度是什么关系": "同一 Consumer Group 内，一个 partition 同一时刻只能由一个消费实例处理，所以有效并发度不会超过 partition 数。增加消费者只能在尚有未分配 partition 时提升吞吐；再往上扩容会出现空闲实例。分区 key 还要避免单个热门 Agent 将大量事件压在一个热 partition 上。",
            "消费中发生rebalance会不会重复计量": "会存在重复窗口。如果数据库副作用已经成功，但 offset 还没有推进就发生崩溃或 rebalance，新消费者可能再处理同一条消息。Kafka 不会和 MongoDB、MySQL、Redis 组成同一事务；因此仍需稳定 eventId、持久化唯一业务键、原子更新和对账，不能依赖消费者一次性。",
            "DLQ消息怎么重新消费": "平台已提供受控回放链路。DLQ 保留原 eventId、失败原因和原始负载，运维修复消息或下游故障后，由回放服务将消息投入专用 replay Topic。消费端继续使用原 eventId 去重和业务唯一约束，成功后记录补偿结果；再次失败时限制回放次数并保留在失败队列，避免在主 Topic 和 DLQ 之间无限循环。",
            "Redis去重key过期后旧消息重放怎么办": "24 小时 TTL 只能覆盖常见的短期重投窗口。超过 TTL 的旧消息会再次被当成未处理事件，如果 MySQL 只做原子自增就可能重复累加。所以 Redis 只是快速重复抑制；需要更长幂等周期时，应在持久存储中保留 eventId 唯一账本，或让关键写入使用唯一业务键。",
            "MongoDB明细和MySQL聚合不一致时怎么补算": "先按 Agent、用户和时间窗口从 MongoDB 重算期望值，再与 MySQL 当前聚合、FailedMessage 和可用 eventId 交叉核对。补算不能直接在旧聚合上再做增量，否则可能二次累加；应以一个固定时间窗口重建后覆盖，或写入临时表校验通过再切换。当前部分明细对象没有 eventId，因此不能承诺所有统计都能逐事件精确重建。"
          },
          "绝对红线": [
            "不要说 Redis eventId 已经实现 exactly-once、永久幂等或永不重复。",
            "不要说 ChainedTransactionManager 是 XA/2PC 或保证 MySQL 与 MongoDB 强一致；生产链路已启用 CTM，但它仍只协调两个本地事务。",
            "不要把 MySQL 原子自增说成事件幂等。",
            "不要说所有解析、校验和业务异常都会进入 DLQ；当前重试/DLQ 首先覆盖业务处理失败。",
            "不要说 Listener 只有业务成功才推进 offset；失败交给失败链路后也会正常返回。",
            "不要说重试和 DLQ 能绝对保证消息不丢。"
          ]
        },
        "rca_investigation_agent": {
          "主题": "面向异常事件的只读取证、预算受控 RCA 调查循环",
          "20秒首答": "RCA Agent 接收一次异常事件后，Planner 先生成候选根因和可验证 Probe，再按信息价值选择下一项只读工具查询。工具结果统一写入 Evidence Ledger，并更新各候选的支持证据、反证和置信度；证据满足完成条件时输出 RCA，预算耗尽、工具不可用或证据仍不足时转 ESCALATED，由人工继续调查。",
          "调查循环": [
            "Planner 根据当前异常上下文生成候选假设，以及能够区分这些假设的只读 Probe。",
            "调度器综合信息价值、查询成本、依赖关系和剩余预算选择下一项 Probe。",
            "Tool Gateway 按允许的 DAG 和权限边界执行查询，限制工具、参数、超时和最大调用次数。",
            "每份结果先进入 Evidence Ledger，记录来源并关联到对应假设，再更新支持、反证和置信度。",
            "证据达到完成条件时进入 COMPLETED；预算、时间或工具能力不足时进入 ESCALATED。"
          ],
          "冲突证据": "支持与反对同一假设的证据都保留在 Evidence Ledger 中，降低对应置信度，并优先选择能够区分冲突解释的新 Probe；弱证据或单一日志不能直接升级成确定根因。",
          "安全边界": "调查工具默认只读，并受允许列表、参数校验、超时、调用预算和 DAG 依赖约束；RCA 输出提供根因判断与证据关联，不直接执行破坏性修复动作。",
          "与LangGraph_Jenkins链路边界": "LangGraph/Jenkins 属于另一条部署审批和状态恢复链路；JenkinsClient 读取 Queue/Build JSON 状态，不构成‘读取 Jenkins console log 后由 LangGraph 完成 RCA’的证据闭环。"
        }
      },
      "resume_bullets": [
        {
          "bullet_title": "Agent Server 执行环境与生命周期管理",
          "ownership": "个人实现",
          "业务问题": "面向多账号 Framework Agent 的容器化运行需求，建设标准化的 Agent Server 执行环境与生命周期管理能力。",
          "简历原文口径": "基于 Kubernetes Deployment / Service / Secret 编排镜像、启动参数、环境变量、CPU/内存、副本及健康探针；通过 Watch + 周期 List 将实际副本、可用副本和运行状态同步至平台，并清理无数据库归属的孤儿资源，支持 Agent Server 创建、更新、副本调整及 Pod 日志查询。",
          "面试展开重点": "以下主题按当前问题选取：对象编排、哪些数据以 Kubernetes 或 MySQL 为准、Watch+List、跨系统补偿和状态语义。"
        },
        {
          "bullet_title": "Agent 调用计量与资源治理链路",
          "ownership": "个人实现",
          "业务问题": "面向 Agent 调用及发布生命周期事件的异步计量需求，建设多账号 Agent 的调用计量与资源治理链路。",
          "简历推荐写法": "基于 Kafka 异步消费事件，分别维护 MongoDB 事件明细与 MySQL 调用量、独立用户及开发者维度聚合统计；通过 Redis eventId 重复抑制、失败重试、失败记录与死信队列处理重复及异常消息，并在生产链路启用 ChainedTransactionManager 协调 MongoDB/MySQL 本地事务，结合对账与补偿处理多存储部分失败，为资源配额和成本治理提供数据基础。",
          "面试展开重点": "以下主题按当前问题选取：投递语义、Redis 幂等边界、MongoDB/MySQL 分工、ChainedTransactionManager 事务边界、重试与 DLQ。"
        }
      ]
    },
    {
      "company": "硕士毕业论文",
      "role": "计算机科学硕士研究课题（进行中）",
      "period": "2026",
      "focus": "MoE Serving 的请求调度与KV容量管理：在原生vLLM上检验并发、prefill预算、等待排序和抢占恢复如何改变完整请求；近期主线是请求等待与资源约束，RCBA保留为历史未验证候选。",
      "resume_positioning": {
        "简历当前标题": "MoE Serving 请求调度与KV容量管理研究 | 硕士毕业论文（进行中） | 2026",
        "标题解释": "当前材料以近期原生请求级调度实验为主讲。早期混合精度、执行一致性和RCBA分别保留为研究背景；不再用RCBA的未完成Oracle作为当前项目介绍。",
        "一句话": "围绕请求等待与KV容量约束，在原生vLLM上实现并验证准入、计算预算和等待顺序的实验动作，比较完整请求的收益与代价。",
        "20秒首答": "我研究MoE推理服务中，请求在进入GPU前和生成过程中为什么会等待，以及怎样通过调度改善完整请求。我在原生vLLM上比较了并发上限、prefill预算、等待顺序和KV容量，发现局部指标改善经常伴随另一段等待增加。现在已有真实请求级实验，还没有证明专家感知方法的稳定增量。",
        "项目状态口径": "MEASUREMENT_ONLY：单模型OLMoE BF16、单张RTX 5090、原生vLLM同步in-process请求级对照。已观察到吞吐、首token等待和生成暂停之间的权衡；没有稳定的专家感知策略收益、质量评估、多卡EP或生产部署结论。",
        "个人工作": "搭建实验侧动作接入、受控运行与请求级分析：非抢占准入上限、引擎排空后的token预算切换、never-started等待请求排序，以及请求身份、调度step、KV块、抢占重算和token返回时刻的关联采集。原生KV分配、执行kernel和默认抢占恢复机制复用vLLM；早期自定义runtime的同状态分叉与数值定位作为另一条已测工作保留。"
      },
      "resume_update_recommendation": {
        "标题": "MoE Serving 请求调度与KV容量管理研究 | 硕士毕业论文（进行中） | 2026",
        "bullet_1": "基于原生vLLM搭建MoE请求级调度实验链路，实现并发准入、prefill token预算和首次入场排序对照，关联采集请求到达、调度、KV状态、抢占重算与逐token时间，验证动作是否真实约束运行。",
        "bullet_2": "通过同引擎静态/反馈对照、长短输入排序和长上下文KV压力实验，量化吞吐、TTFT与生成暂停的权衡：原生cap32较保守cap29吞吐约高17%，同时少数请求承担最长约4.5秒暂停；结论限定为当前运行域的测量结果。",
        "现有简历风险": "约17%是允许原生抢占的cap32相对保守cap29的实测配置差异，不能写成自研调度算法相对vLLM提升17%。短请求TTFT改善不代表整体加速；不同实验的阈值、长度和显存预算不能混用。"
      },
      "project_background": {
        "业务背景": "在线生成请求持续到达，prefill处理输入，decode逐步生成；两类工作共享计算与KV容量。用户既关心多久看到首token，也关心生成时是否停顿，所以吞吐高、GPU忙或平均TPOT低都不能单独代表体验好。",
        "为什么落在调度": "同一模型与执行后端下，可以调整何时接纳请求、每步安排多少token、谁先开始，以及如何应对KV继续增长。这些动作会改变实际batch、排队和恢复时机，适合先用原生请求级对照判断有没有完整收益。",
        "为什么使用MoE": "MoE是本论文的研究对象，专家执行与请求、KV共享GPU资源；但近期已测权衡可能同样存在于Dense模型。当前先建立普通调度与资源基线，只有专家状态在这些基线之外还解释并改变请求损害，才讨论MoE特有贡献。",
        "从旧工作到现在": "早期混合精度和局部机会评估提醒我，局部节省未必进入完整请求；动态batch实验又说明，动作可能改变后续数值和route，不能拿固定route伪造动作反事实。RCBA试图直接研究关键路径上界，但正式依赖、容量契约和Oracle没有闭合。近期因此把主讲落到已经能真实执行和测量的原生请求调度，先验证普通动作能解释多少现象。",
        "当前研究问题": "在固定模型与明确资源约束下，准入、计算预算和等待顺序能否改善完整请求，还是主要把等待从一个请求或阶段转移到另一个？只有强简单策略后仍有可重复损害，才继续找专家相关动作。",
        "已测运行域": "单模型、单GPU、原生同步in-process引擎；短输入、长短混合和长上下文分别组成不同实验，SLO与配置按各自报告解释。当前数据不足以推出生产负载分布或跨模型规律。"
      },
      "research_story_for_interview": {
        "2分钟完整讲述": [
          "我的论文现在围绕MoE在线推理的请求调度和KV容量管理。问题是，同一块GPU要处理新请求的prefill和已有请求的decode；新请求接纳太慢会增加TTFT，接纳太多又可能随着生成触到KV边界。我的目标是看完整请求是否改善，而不是只把GPU利用率或某个算子做快。",
          "早期我做过混合精度和动态batch执行一致性研究，逐渐确认局部机会不等于请求收益，而且调度会改变后续batch和route。RCBA是当时的关键路径候选，但没有完成正式Oracle，所以当前介绍转到已跑通的原生vLLM调度实验。",
          "最近我实际接入了并发上限、prefill预算和首次入场排序，也补了KV压力下的原生抢占基线。结果不是所有优化都赢：新反馈没有稳定超过最好静态cap；小预算减少部分token尾间隔，却增加TTFT；短prompt优先改善短请求，但增加长请求等待。",
          "最具体的一组结果是原生cap32比保守cap29吞吐约高17%，同时两条请求出现约2秒和4.5秒的生成暂停；保守cap29没有抢占，但后到请求的TTFT接近18秒。下一步固定cap32，只比较90%和95%显存预算，判断普通KV配置能否先解决暂停。已有的是请求级测量和实验实现，专家感知方法的独立收益还没有证明。"
        ],
        "被打断时怎么接": "先回答对方正在问的动作、结果或代价，再按scheduling_followup_sequences接下一问。不要每次从所有历史方向讲起；问旧工作时再进入historical_research_lines及原有数值定位问答。"
      },
      "current_primary": {
        "name": "MoE Serving中的请求等待与KV容量管理",
        "status": "MEASUREMENT_ONLY / NATIVE_SERVING_IN_PROCESS / NO_VERIFIED_EXPERT_AWARE_METHOD_GAIN",
        "question": "普通准入、计算预算和KV配置能否解决已观察到的请求等待；哪些改善只是把成本转移给另一个阶段或另一组请求？",
        "目前可直接回答": "已经测到真实动作和完整请求的权衡。更小的prefill预算会增加首token等待；短prompt优先会把等待分给长prompt；保守KV接纳消除了测得的抢占，却让后到请求更晚开始。原生抢占恢复基线可以更快完成整批，同时让少数请求承担秒级暂停。",
        "最弱未闭合环节": "原生cap32的长暂停，能否仅通过足量KV预算消除，并进一步改善完整请求？如果普通配置已解决，就没有理由从这个现象直接引出复杂专家调度器。",
        "下一实验": "已准备的固定cap32、gpu_memory_utilization 0.95→0.90→0.90→0.95四项对照；r01在模型加载前因资源冲突退出，r02尚未执行。",
        "主张上限": "当前配置和自然文本来源上的原生请求级测量；不是同预算算法加速、动态Oracle、跨模型定律或生产SLO验证。"
      },
      "recent_experiment_progression": [
        {
          "问题": "调并发和反馈档位就能改善吗？",
          "动作与基线": "旧反馈cap档位8/12/16/32，对照对齐后的8/16/24/32；共同引擎内测静态8/12/16/24/32。短输入输出均128，steady和bursty分开。",
          "已测": "两轮共64个正式episode、2048次请求执行，复用同一32条源文本。bursty新反馈比旧反馈高约44%至61%，但4组都低于同期最好静态点；steady仅一组胜出，重复没有保住。",
          "改变的判断": "比旧反馈好不足以构成策略收益。首个反馈前轨迹已经可能因host时间跨到达边界而分叉，首动作waiting为0时也没有即时准入作用，不能把变化全部归因于反馈动作。",
          "来源": "refine-logs/expert_saturation/outputs/admission_capacity/20260908_capture_ladder_paired_repeat_r01/REPORT.md"
        },
        {
          "问题": "限制prefill计算量能保护生成吗？",
          "动作与基线": "cap8与共同编译容量1024固定，引擎排空后切换调度预算256/1024，再做预选512/1024；16条请求，混合输入128/2048，固定128输出。",
          "已测": "256对照8个正式episode与8个warmup完整回传；mixed TTFT增加32.06%/34.56%，整批时间增加9.05%/9.14%。512对照4个正式与4个warmup完整回传，ITL p99下降约29.9%/29.5%，但TTFT增加6.36%/7.80%、整批时间增加2.12%/2.38%。",
          "改变的判断": "更小chunk的局部平滑不能代表完整收益。全短输入对照的预算没有约束动作；两个较小预算都出现代价后停止继续扫描静态阈值。",
          "来源": "refine-logs/independent_ideas_20260908/host_timing_boundary_r01/prefill_budget_midpoint_r01/REPORT.md",
          "补充来源": "refine-logs/independent_ideas_20260908/host_timing_boundary_r01/next_prefill_budget/REPORT.md"
        },
        {
          "问题": "保持预算不变，只让短prompt先入场呢？",
          "动作与基线": "固定cap8、预算1024；只对原生WAITING里从未执行的请求按prompt长度稳定排序。FCFS对照承担相同队列操作和采集，已开始的prefill及decode保持原生语义。",
          "已测": "首次两块和原样重复forward共三个block有完整本地raw；重复reverse远端完成但raw仍缺。可核对的三块中，短TTFT下降约14.06%/10.38%/9.45%，长TTFT增加3.09%/6.44%/5.19%；整体平均请求延迟只下降2.52%/0.22%/0.03%，长请求完成尾部增加。",
          "改变的判断": "确认了顺序动作与受益、受损请求，但没有稳定的整体净收益，也没有有限实验之外的防饥饿保证。未回传的第四块不参与结论。",
          "来源": "refine-logs/independent_ideas_20260908/host_timing_boundary_r01/waiting_order_repeat_r01/REPORT.md"
        },
        {
          "问题": "prefill能放下，为什么后续生成仍缺KV？",
          "动作与基线": "32条自然文章，3072输入、1024输出、50ms到达；从live池读到7677可用块，每块16tokens，每条最大需要256块，得到保守cap29，并与cap16比较。",
          "已测": "早期cap32在全部prefill后随decode增长触边，被守卫中止，不能当完整原生基线。有效cap29对照cap16吞吐提高3.75%/4.53%，同时TTFT p99增加约4.45/4.28秒、请求平均TPOT的中位数增加约5.10/4.96ms。",
          "改变的判断": "准入需要考虑KV后续增长；最大长度预留可以给容量边界，但cap29不是性能最优的证明。需要补齐引擎原生恢复能力后再判断代价。",
          "来源": "refine-logs/expert_saturation/outputs/admission_capacity/20260908_kv_safe_static_r02/REPORT.md"
        },
        {
          "问题": "真正放行原生抢占恢复，谁的完整请求更好？",
          "动作与基线": "相同长上下文负载和0.90显存预算，四个新引擎按native32→safe29→safe29→native32执行；全部正式请求与warmup完整回传。",
          "已测": "native32吞吐比safe29高16.99%/17.08%，TTFT p99约0.76/0.80秒，对照约17.90/18.11秒；但每轮两条victim暂停约2秒和4.5秒，safe29最大ITL约0.108秒。native32的请求平均TPOT中位数也更差。",
          "改变的判断": "零抢占不是性能目标。最长4.47秒间隔中约4.36秒在首个重算调用前，约0.116秒是重算调用首尾跨度，不能统称4.5秒GPU重算。pooled-token ITL p99约28ms仍可能掩盖这些受损请求。",
          "来源": "refine-logs/expert_saturation/outputs/admission_capacity/20260908_native_preemption_r01/REPORT.md"
        },
        {
          "问题": "普通KV预算配置是否足以消除长暂停？",
          "动作与基线": "固定cap32，预定0.95→0.90→0.90→0.95；其他模型、负载、预算和原生恢复语义相同，先读取足量live KV。",
          "已测": "r01首项在import torch与模型加载前发现其他GPU进程，退出1，0warmup、0请求测量；失败记录已经回传。r02执行包已准备但未上传、未启动。",
          "改变的判断": "这个问题尚未得到性能回答。资源冲突属于UNRUN，不能把它写成0.95容量配置失败；即使将来有效，也先解释为增加KV资源预算的作用。",
          "来源": "refine-logs/expert_saturation/outputs/admission_capacity/20260908_kv_budget_r01/REPORT.md",
          "补充来源": "refine-logs/expert_saturation/outputs/admission_capacity/20260908_kv_budget_r02/ADDENDUM.md"
        }
      ],
      "concrete_future_plan": {
        "当前唯一下一实验": "完成已准备的kv_budget_r02四项KV预算对照。固定cap32、32篇原文、3072输入/1024输出、50ms到达、token预算1024；四个新引擎顺序0.95→0.90→0.90→0.95，两臂都允许原生抢占恢复。",
        "执行前资格": "排除资源冲突，读取每次真实KV布局。0.95臂在block_size=16时要求至少8192可用块，即32×ceil((3072+1024)/16)；这是完整最大长度同时驻留的充分条件，不是原生请求完成或无抢占的必要条件。0.90臂也读live池，不能硬编码7677。",
        "主指标": "全部请求完成、完整吞吐、TTFT、请求平均TPOT、completion、每请求最大ITL，以及实际抢占、重复计算与恢复前等待；pooled-token ITL作为补充。保持5s/200ms参考SLO并同时报告连续指标，不能用平均TPOT掩盖秒级暂停。",
        "如果结果为正": "若足量KV同时消除长暂停并改善完整请求，先接受普通资源配置已解释当前损害。明确记录增加的物理KV资源，不包装成同显存预算下的新算法收益。",
        "如果结果为负或无效": "若只消除抢占却不改善完整指标，说明零抢占仍不是有效代理；若0.90不复现抢占，说明本轮缺少相同压力对照。0.95初始化或容量资格失败时保留资料、测量记UNRUN并停止，不继续扫0.96/0.99或改负载。",
        "只有满足条件才扩展": "只有普通KV、长度和队列策略后仍有可重复请求损害，才选一个实际可执行动作，比较普通负载、当前每专家负载和额外路由历史的增量。先做单动作三臂对照，再在正信号后增加独立文档、steady/bursty和第二模型；尚无这些实验结果。",
        "设计题与执行计划的区别": "多租户、公平性、过载、阶段隔离和多副本路由用于回答设计追问，均不是已经实现的系统，也不与这四项实验同时开工。等待排序缺失raw属于已有结果回收，不能靠新跑替代。",
        "来源": "refine-logs/expert_saturation/outputs/admission_capacity/20260908_kv_budget_r02/DECISIONS.md"
      },
      "research_workflow": {
        "一句话主线": "先确认参数实际改变了谁能运行，再比较强简单基线，最后用请求全程记录解释收益和受损者。",
        "已经执行的研究步骤": [
          "比较准入上限与普通延迟反馈，记录actual active/waiting和首个动作；发现改写cap时waiting可能已为0，且更好的反馈档位未稳定超过同引擎最好静态点。",
          "用固定cap的prefill预算和never-started等待顺序对照，分开检查单步计算量与首次入场顺序；记录整体、短请求和长请求的变化，保留无动作负控。",
          "把输入和输出长度放到真实KV压力域，读取live池布局并计算保守接纳上限；区分预检未运行、守卫中止与完整请求结果。",
          "补齐允许vLLM原生抢占恢复的完整cap32基线，再与保守cap29比较；从每请求最大ITL和实际重算记录解释吞吐收益与暂停代价。"
        ],
        "下一步原则": "先执行已经冻结的同cap KV预算对照。只有普通配置后仍有可重复的完整请求损害，且新动作可执行，才进入专家信号或机制；不同时展开多个控制器。",
        "为什么仍保留Oracle思路": "最优静态点用于暴露简单基线是否已经覆盖收益，并不是动态Oracle。只有提出具体的新动作后，才构造尊重容量、依赖和动作后状态的上界；旧RCBA Oracle不再阻塞当前小规模存在性实验。"
      },
      "evaluation_method": {
        "请求时钟": "到达、首token返回、后续token返回与完成都使用一致host时间原点。TTFT=首token时刻-到达；请求延迟=完成-到达；请求平均TPOT=(末token-首token)/(输出token数-1)。当前queue_s=提交-到达，仅是host提交滞后，不能冒充原生waiting时间。",
        "等待与成本": "请求全程按到达至首token、首token至结束区分，再用调度和抢占记录定位内部等待、prefill、decode及恢复。吞吐与completion分母已包含这些成本；不能把多个请求重叠的等待或局部stage重新相加。",
        "尾部指标": "同时报告TTFT分布、每请求平均TPOT分布、pooled-token ITL及每请求最大ITL，单列受抢占请求。数万token中的p99可能掩盖少数请求的秒级暂停，16或32条请求的p99也接近最慢样本。",
        "SLO与goodput": "预先声明阈值、观察区间和全部到达请求集合；goodput按该区间内完成且满足声明SLO的请求数/时间计算，另报完成、失败和未完成。短cap实验是200ms TTFT/9ms平均TPOT，后续长输入探针采用5s/200ms参考口径，不跨实验比较达标率或倒推阈值。",
        "强基线与因果": "默认策略、同动作空间的强简单策略优先；事后最好静态点是诊断参照，部署比较需校准后冻结。各策略独立推进队列、KV、batch和生成，不能共享未来route；同时保留负控和动作实效记录。",
        "重复与样本": "同引擎对照尽量保持编译容量和KV池；必须重建引擎的预算实验使用反向顺序与受控重复。所有运行保留，warmup与正式测量分开；复用文档的累计执行数不充当独立样本，未回传raw的block不进入已验证结果。"
      },
      "scheduling_interview_notes": {
        "核对日期": "2026-09-09",
        "口述使用": "先回答当前问题，通常2至4句；下面的来源与边界供复习核对，不作为固定开场或每题免责声明。近期调度问答与旧数值一致性实验按各自证据回答。",
        "最近调度工作20秒首答": "我研究MoE推理服务中，请求在进入GPU前和生成过程中为什么会等待，以及怎样通过调度改善完整请求。我在原生vLLM上比较了并发上限、prefill预算、等待顺序和KV容量，发现局部指标改善经常伴随另一段等待增加。现在已有真实请求级实验，还没有证明专家感知方法的稳定增量。",
        "复用与实现": "复用vLLM原生引擎、kernel、KV分配及默认抢占机制；新增实验侧动作、采集与分析。多个Codex进程的执行记录分别保留，不把同一32条源请求或别的执行重复计为新增样本。",
        "事实来源": [
          {
            "主题": "普通反馈与强静态基线",
            "来源": "refine-logs/expert_saturation/outputs/admission_capacity/20260908_capture_ladder_paired_repeat_r01/REPORT.md",
            "可引用": "64个正式episode复用32条文本；bursty四组均未超同期最好static；steady一次胜出未复现。200ms/9ms为本实验SLO。"
          },
          {
            "主题": "prefill预算256与1024",
            "来源": "refine-logs/independent_ideas_20260908/host_timing_boundary_r01/next_prefill_budget/REPORT.md",
            "可引用": "8正式加8预热均回传；cap8，mixed输入128/2048、输出128；TTFT增加32.06%/34.56%，wall增加9.05%/9.14%。"
          },
          {
            "主题": "预选512中间预算",
            "来源": "refine-logs/independent_ideas_20260908/host_timing_boundary_r01/prefill_budget_midpoint_r01/REPORT.md",
            "可引用": "4正式加4预热均回传；TTFT增加6.36%/7.80%，wall增加2.12%/2.38%，ITL p99下降29.90%/29.48%；静态预算扫描停止。"
          },
          {
            "主题": "首次入场排序与原样重复",
            "来源": "refine-logs/independent_ideas_20260908/host_timing_boundary_r01/waiting_order_repeat_r01/REPORT.md",
            "补充来源": "refine-logs/independent_ideas_20260908/host_timing_boundary_r01/waiting_order_r01/REPORT.md",
            "可引用": "只有三个block的完整raw已回传；原样重复reverse性能不可引用。短TTFT下降、长TTFT和尾部增加，整体净收益未成立。"
          },
          {
            "主题": "实际waiting动作及请求计时",
            "来源": "refine-logs/independent_ideas_20260908/host_timing_boundary_r01/waiting_order_r01/runtime/waiting_order.py",
            "补充来源": [
              "refine-logs/independent_ideas_20260908/host_timing_boundary_r01/waiting_order_r01/runtime/native_capture.py",
              "refine-logs/independent_ideas_20260908/host_timing_boundary_r01/waiting_order_r01/runtime/metrics.py"
            ],
            "可引用": "只重排never-started WAITING，记录已有decode推进；queue_s仅为提交滞后，所有host时刻共享起点。"
          },
          {
            "主题": "KV容量公式",
            "来源": "refine-logs/expert_saturation/outputs/admission_capacity/20260908_kv_safe_static_r02/REPORT.md",
            "可引用": "7677可用块，每块16tokens；ceil((3072+1024)/16)=256，floor(7677/256)=29；只适用该live布局和最大长度预留条件。"
          },
          {
            "主题": "原生抢占完整基线与等待分解",
            "来源": "refine-logs/expert_saturation/outputs/admission_capacity/20260908_native_preemption_r01/REPORT.md",
            "可引用": "cap32相对cap29吞吐增加16.99%/17.08%，每次两条请求出现约2s/4.5s暂停；pooled-token ITL p99约28ms。报告的TPOT p50是每请求平均TPOT的中位数，不能改称总体均值。"
          },
          {
            "主题": "原生MoE局部数值负结果",
            "来源": "refine-logs/expert_saturation/outputs/native_companion/20260906_layer3_r01/REPORT.md",
            "可引用": "零索引layer3/M16/target row7，两个自然target、两个进程，共36次调用；内部布局改变，目标输出max-abs为0。不是旧decode事件重放，也不是性能实验。"
          },
          {
            "主题": "已尝试但没有GPU测量的下一轮KV预算对照",
            "来源": "refine-logs/expert_saturation/outputs/admission_capacity/20260908_kv_budget_r01/REPORT.md",
            "补充来源": [
              "refine-logs/expert_saturation/outputs/admission_capacity/20260908_kv_budget_r02/ADDENDUM.md",
              "refine-logs/expert_saturation/outputs/admission_capacity/20260908_kv_budget_r02/DECISIONS.md"
            ],
            "可引用": "r01模型加载前资源冲突退出1、0warmup/0请求，r02已准备未执行。计划固定cap32按95→90→90→95对照；8192块为最大长度同时驻留的充分条件，不是原生完成必要条件。"
          }
        ],
        "指标不可混用": [
          "不同实验的SLO、输入长度、memutil和引擎容量不同；只在各自对照中引用相对效果。",
          "TTFT、请求平均TPOT、pooled-token ITL和每请求最大ITL分别报告；不能把它们统称P99。",
          "已到达的失败和unfinished保留；未回传block不进入已验证结果；正式、预热、复用文档和累计请求执行数分开。",
          "这些新原生实验没有完成RCBA的双模型Oracle，不改变docs/current的正式候选状态。"
        ]
      },
      "adjacent_scheduling_research": {
        "核对日期": "2026-09-09",
        "用途": "相邻调度知识与面试追问；条目是文献事实，与当前实验的关系是本次分析，未来动作不计为个人已实现。",
        "范围与阅读层级": "15篇定向论文及5份官方文档，覆盖基础工作与2024–2026近邻；非穷尽综述、非新颖性认证，未复现论文代码或性能。部分条目只核验摘要，详见reading_level。",
        "检索来源": "会议正式论文、作者固定版本arXiv与官方文档；本地已有Aurora/Lina先读前3页再定位相关段。无本地arXiv API helper，采用web核验；不下载新PDF。",
        "阅读版": "docs/interview/moe-adjacent-scheduling-20260909.md",
        "层次对照": [
          {
            "层次": "集群作业准入与资源份额",
            "状态": "租户配额、资源需求、作业优先级",
            "动作": "允许作业启动、借用资源或回收配额",
            "代表": [
              "drf",
              "kueue"
            ],
            "当前关系": "设计与面试知识，未由当前论文实现"
          },
          {
            "层次": "Pod放置与组调度",
            "状态": "节点容量、拓扑、组成员与预留",
            "动作": "选节点、Reserve/Permit或回队",
            "代表": [
              "k8s_framework",
              "kueue_gang"
            ],
            "当前关系": "不能替代LLM引擎的KV/每步调度"
          },
          {
            "层次": "副本与阶段分配",
            "状态": "阶段负载、KV、带宽、专家签名",
            "动作": "选择PD资源、decode worker或迁移请求",
            "代表": [
              "distserve",
              "tetriinfer",
              "llumnix",
              "eldr",
              "hpa"
            ],
            "当前关系": "跨GPU动作未被单卡实验覆盖"
          },
          {
            "层次": "请求选择与迭代合批",
            "状态": "长度、服务账本、运行阶段与token预算",
            "动作": "选下一请求、切prefill、迭代边界抢占",
            "代表": [
              "orca",
              "sarathi",
              "fastserve",
              "vtc"
            ],
            "当前关系": "与cap、预算、等待排序最直接相邻"
          },
          {
            "层次": "KV和权重驻留",
            "状态": "物理块、前缀引用、专家工作集与搬运成本",
            "动作": "分页、复用、回收、offload或预算重分配",
            "代表": [
              "pagedattention",
              "sglang",
              "wisp",
              "fluxmoe"
            ],
            "当前关系": "当前只测普通KV预算和原生恢复，未实现权重回收"
          },
          {
            "层次": "专家部署与通信",
            "状态": "专家热度、traffic matrix、设备与链路",
            "动作": "复制/放置专家、重排通信",
            "代表": [
              "lina",
              "aurora"
            ],
            "当前关系": "需真实多卡EP或对应仿真语义，不能沿用单卡请求结论"
          }
        ],
        "文献卡片": [
          {
            "id": "orca",
            "name": "Orca",
            "title": "Orca: A Distributed Serving System for Transformer-Based Generative Models",
            "authors": "Gyeong-In Yu et al.",
            "publication": "OSDI 2022",
            "dedup_key": "usenix:osdi22:yu",
            "url": "https://www.usenix.org/conference/osdi22/presentation/yu",
            "primary_text": "https://www.usenix.org/system/files/osdi22-yu.pdf",
            "signal_state": "请求阶段、迭代完成、batch上限和KV容量",
            "action": "逐迭代调度与selective batching；新请求按max_tokens预留KV",
            "objective": "生成服务的吞吐与延迟",
            "evidence_regime": "Transformer serving及模型并行；不是MoE专家信号实验",
            "guarantee_and_cost": "最大长度预留保证其模型下后续KV可分配，不保证性能最优；有逐迭代控制成本。",
            "relation_to_current": "continuous batching及保守预留已有先例；当前cap29的价值是实测现代原生恢复基线下的等待代价。",
            "reading_level": "官方书目、摘要、selective batching与Algorithm 1/KV reservation相关正文"
          },
          {
            "id": "pagedattention",
            "name": "PagedAttention / vLLM",
            "title": "Efficient Memory Management for Large Language Model Serving with PagedAttention",
            "authors": "Woosuk Kwon et al.",
            "publication": "SOSP 2023",
            "dedup_key": "arxiv:2309.06180",
            "url": "https://arxiv.org/abs/2309.06180",
            "primary_text": "https://arxiv.org/abs/2309.06180",
            "signal_state": "动态增长的KV与可共享前缀",
            "action": "分页管理KV并支持共享，减少碎片和重复存储",
            "objective": "相同延迟水平下的服务吞吐",
            "evidence_regime": "论文在其模型、基线和解码设置中报告收益；不等同当前vLLM版本",
            "guarantee_and_cost": "减少存储浪费不等于物理KV容量无限；当前抢占行为另外查v0.26官方文档。",
            "relation_to_current": "本研究复用该类KV管理能力；不能把分页或cap29公式归为自研。",
            "reading_level": "作者arXiv书目与摘要；当前行为另核v0.26调优文档，未重审2023全文实现"
          },
          {
            "id": "sarathi",
            "name": "Sarathi-Serve",
            "title": "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve",
            "authors": "Amey Agrawal et al.",
            "publication": "OSDI 2024",
            "dedup_key": "arxiv:2403.02310",
            "url": "https://www.usenix.org/conference/osdi24/presentation/agrawal",
            "primary_text": "https://www.usenix.org/system/files/osdi24-agrawal.pdf",
            "signal_state": "decode集合、prefill余量、token预算及TBT目标",
            "action": "切分prefill并按预算与decode合批，控制迭代工作量",
            "objective": "尾部token延迟约束下的服务容量",
            "evidence_regime": "单GPU、TP与PP；模型和硬件需校准",
            "guarantee_and_cost": "小chunk存在利用率、重复KV读取及固定开销；不能解释成预算越小越好。",
            "relation_to_current": "256/512/1024实验是特定原生运行域的预算干预，不是完整复现，也不判死chunked prefill。",
            "reading_level": "官方书目与正文4.1–4.3、5.4.1"
          },
          {
            "id": "distserve",
            "name": "DistServe",
            "title": "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving",
            "authors": "Yinmin Zhong et al.",
            "publication": "OSDI 2024",
            "dedup_key": "arxiv:2401.09670",
            "url": "https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin",
            "primary_text": "https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf",
            "signal_state": "TTFT/TPOT目标、长度/到达率、profiling与互联带宽",
            "action": "PD放到不同GPU，分别配置资源与并行方案，并按带宽约束放置",
            "objective": "满足TTFT与TPOT的每GPU goodput",
            "evidence_regime": "多GPU；部分高跨节点带宽方案为模拟评估",
            "guarantee_and_cost": "KV传输、阶段排队和两侧容量均需计费；分离不自带高级抢占与容错。",
            "relation_to_current": "当前单引擎预算与排序未测PD传输，不能直接外推分离净收益。",
            "reading_level": "官方书目、正文4.1–4.3、通信代价及评估边界"
          },
          {
            "id": "tetriinfer",
            "name": "TetriInfer",
            "title": "Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads",
            "authors": "Cunchen Hu et al.",
            "publication": "2024 arXiv preprint；本轮未核验到正式venue",
            "dedup_key": "arxiv:2401.11181",
            "url": "https://arxiv.org/abs/2401.11181",
            "primary_text": "https://arxiv.org/html/2401.11181v1",
            "signal_state": "prompt长度、预测输出区间及decode实例负载",
            "action": "PD分离；有限排序窗口内FCFS/SJF/LJF；固定prefill chunk与预测式decode放置",
            "objective": "TTFT、JCT与资源效率",
            "evidence_regime": "多实例分离式服务；已有request-level KV传输",
            "guarantee_and_cost": "长度预测有误差和成本；排序窗口限制新到短请求持续插队，非任意过载下SLO保证。",
            "relation_to_current": "与短prompt优先直接相邻；当前没有其排序窗口、预测器和跨实例放置。",
            "reading_level": "arXiv v1书目与正文3.2、3.3.1–3.3.4"
          },
          {
            "id": "fastserve",
            "name": "FastServe",
            "title": "FastServe: Iteration-Level Preemptive Scheduling for Large Language Model Inference",
            "authors": "Bingyang Wu et al.",
            "publication": "NSDI 2026；2023预印本旧题Fast Distributed Inference Serving for Large Language Models",
            "dedup_key": "arxiv:2305.05920",
            "url": "https://www.usenix.org/conference/nsdi26/presentation/wu-bingyang",
            "primary_text": "https://www.usenix.org/system/files/nsdi26-wu-bingyang.pdf",
            "signal_state": "输入长度、已执行时间、队列层级与KV压力",
            "action": "Skip-join MLFQ、迭代边界抢占、超时提级及KV换入换出",
            "objective": "降低长任务阻塞并提高满足延迟目标的吞吐",
            "evidence_regime": "主要A100/FP16与Dense模型；正式论文vLLM基线为0.6.1",
            "guarantee_and_cost": "抢占有KV搬运和恢复等待；MLFQ不需要真实总输出长度，也不保证任意请求必达SLO。",
            "relation_to_current": "当前never-started排序没有MLFQ执行权切换；原生缺KV抢占也不是实现了FastServe。",
            "reading_level": "NSDI正式PDF 2.3、3、4.1–4.2、6.1及baseline说明；旧预印本去重"
          },
          {
            "id": "vtc",
            "name": "VTC",
            "title": "Fairness in Serving Large Language Models",
            "authors": "Ying Sheng et al.",
            "publication": "OSDI 2024",
            "dedup_key": "arxiv:2401.00588",
            "url": "https://www.usenix.org/conference/osdi24/presentation/sheng",
            "primary_text": "https://www.usenix.org/system/files/osdi24-sheng.pdf",
            "signal_state": "各客户端已获得的输入/输出token服务成本",
            "action": "逐token记账；有接纳容量时优先低counter客户端，并校正重新入队的计数基线",
            "objective": "服务量公平与闲置份额利用",
            "evidence_regime": "共享LLM的多个客户端；基本方案无运行中抢占",
            "guarantee_and_cost": "论文模型下持续backlogged客户端的服务差有界，不是逐请求TTFT或SLO保证。",
            "relation_to_current": "比请求数轮询或短prompt排序多了跨客户端服务账本；当前未实现。",
            "reading_level": "OSDI正文2.1、3、4.1、Theorem 4.4与4.3加权变体"
          },
          {
            "id": "llumnix",
            "name": "Llumnix",
            "title": "Llumnix: Dynamic Scheduling for Large Language Model Serving",
            "authors": "Biao Sun et al.",
            "publication": "OSDI 2024",
            "dedup_key": "arxiv:2406.03243",
            "url": "https://www.usenix.org/conference/osdi24/presentation/sun-biao",
            "primary_text": "https://arxiv.org/html/2406.03243v1",
            "signal_state": "实例KV占用、等待需求、优先级与virtual usage/freeness",
            "action": "分发新请求，并分阶段复制KV以迁移运行中请求",
            "objective": "跨实例负载、容量、优先级与弹性",
            "evidence_regime": "论文16张A10、LLaMA-7B/30B多实例",
            "guarantee_and_cost": "迁移需要目的端余量、复制带宽及最终切换停顿；低停顿是实测结果，非零成本保证。",
            "relation_to_current": "跨实例空闲容量碎片不同于PagedAttention解决的KV地址管理；单卡实验未覆盖迁移。",
            "reading_level": "官方书目；作者正文4.2、4.4、5、6.1"
          },
          {
            "id": "sglang",
            "name": "SGLang / RadixAttention",
            "title": "SGLang: Efficient Execution of Structured Language Model Programs",
            "authors": "Lianmin Zheng et al.",
            "publication": "NeurIPS 2024",
            "dedup_key": "arxiv:2312.07104",
            "url": "https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html",
            "primary_text": "https://papers.nips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf",
            "signal_state": "radix树的匹配前缀、KV引用及可回收缓存",
            "action": "复用前缀KV，按匹配前缀组织请求并回收缓存",
            "objective": "减少重复prefill和提高执行吞吐",
            "evidence_regime": "共享前缀、多轮与分支程序等负载",
            "guarantee_and_cost": "离线缓存命中定理有给定请求集及容量条件；在线贪心可能饥饿，命中最优不等于TTFT最优。",
            "relation_to_current": "按已缓存前缀与按总prompt长度排序会选中不同请求；当前无prefix cache，未验证该动作。",
            "reading_level": "NeurIPS正式PDF第3节、Theorem 3.1与附录相关算法"
          },
          {
            "id": "wisp",
            "name": "WiSP",
            "title": "WiSP: A Working-Set View of Mixture-of-Experts Serving on Extremely Low-Resource Hardware",
            "authors": "Jiamu Zhang et al.",
            "publication": "2026 arXiv preprint，v2（2026-08-30）",
            "dedup_key": "arxiv:2606.21868",
            "url": "https://arxiv.org/abs/2606.21868v2",
            "primary_text": "https://arxiv.org/html/2606.21868v2",
            "signal_state": "专家工作集、复用与KV需求",
            "action": "专家分页、expert/KV预算分配；动态resize在排空边界执行",
            "objective": "低显存、低并发下的serving延迟与容量",
            "evidence_regime": "v2主结果为真实RTX3090 24GiB；OLMoE分配实验另为H100受限预算",
            "guarantee_and_cost": "PCIe带宽决定预取是否有利；输出一致是作者测试结论。v1模拟小卡结果不能混入v2。",
            "relation_to_current": "专家与KV联合分配已被直接研究；当前提高memutil不等于实现可回收专家池。",
            "reading_level": "v2摘要、设计与第5节设置；核对v1/v2硬件变化"
          },
          {
            "id": "fluxmoe",
            "name": "FluxMoE",
            "title": "FluxMoE: Decoupling Expert Residency for High-Performance MoE Serving",
            "authors": "Qingxiu Liu et al.",
            "publication": "2026 arXiv preprint，v2（2026-04-30）",
            "dedup_key": "arxiv:2604.02715",
            "url": "https://arxiv.org/abs/2604.02715v2",
            "primary_text": "https://arxiv.org/html/2604.02715v2",
            "signal_state": "KV压力与计算/加载成本",
            "action": "PagedTensor、无损压缩GPU存储与host DRAM分层、按需流水加载专家",
            "objective": "受显存约束时的总token吞吐",
            "evidence_regime": "四张L40 48GB、vLLM0.10.2、TP2/4、batch32–256",
            "guarantee_and_cost": "主评估不以TTFT/TPOT为目标；未核实统一dtype，不擅填BF16；搬运必须进入完整成本。",
            "relation_to_current": "权重驻留回收不同于普通KV参数调优；大batch吞吐不能转写为当前单卡SLO收益。",
            "reading_level": "v2摘要、正文4–6及评估指标"
          },
          {
            "id": "eldr",
            "name": "ELDR",
            "title": "ELDR: Expert-Locality-Aware Decode Routing for PD-Disaggregated MoE Serving",
            "authors": "Sangjin Choi et al.",
            "publication": "2026 arXiv preprint，v2（2026-07-02）",
            "dedup_key": "arxiv:2607.00466",
            "url": "https://arxiv.org/abs/2607.00466v2",
            "primary_text": "https://arxiv.org/html/2607.00466v2",
            "signal_state": "prefill expert signature、decode worker负载与KV-block signature cache",
            "action": "PD交接时，在局部性候选worker内按负载选择decode目的地",
            "objective": "平衡负载并提高decode batch的专家复用",
            "evidence_regime": "MI300X 192GB/400Gbps IB的PD分离；主结果24GPU，EP扩展最多40GPU；模型有BF16与MXFP4",
            "guarantee_and_cost": "维持gate/kernel；作者报告输出不变，不等于任意batch位级一致保证。",
            "relation_to_current": "按专家签名分流有直接先例；当前单引擎cap/排序是另一动作域，但不同位置本身不构成新颖性。",
            "reading_level": "v1机制与设置初读，增量核对v2摘要、机制、实验设置及输出声明；这些段落未见实质变化"
          },
          {
            "id": "aurora",
            "name": "Aurora",
            "title": "Optimizing Mixture-of-Experts Inference Time Combining Model Deployment and Communication Scheduling",
            "authors": "Jialong Li et al.",
            "publication": "2024 arXiv preprint，v1；未核验正式venue",
            "dedup_key": "arxiv:2410.17043",
            "url": "https://arxiv.org/abs/2410.17043",
            "primary_text": "https://arxiv.org/abs/2410.17043",
            "signal_state": "历史traffic matrix、阶段时长与设备能力",
            "action": "跨模型专家共置、GPU分配与all-to-all发送顺序",
            "objective": "模型推理时间",
            "evidence_regime": "视觉MoE统计驱动的网络/部署仿真",
            "guarantee_and_cost": "理论最优性受建模假设限制；仿真平均1.07倍最优值不是一般近似保证，也无原生请求级证据。",
            "relation_to_current": "属于部署与通信层；不能用其仿真结果支持单卡连续decode改进。",
            "reading_level": "本地PDF前3页初筛，再定位前提与实验设置；arXiv核验书目"
          },
          {
            "id": "lina",
            "name": "Lina",
            "title": "Accelerating Distributed MoE Training and Inference with Lina",
            "authors": "Jiamin Li et al.",
            "publication": "USENIX ATC 2023",
            "dedup_key": "usenix:atc23:li-jiamin",
            "url": "https://www.usenix.org/conference/atc23/presentation/li-jiamin",
            "primary_text": "https://www.usenix.org/system/files/atc23-li-jiamin.pdf",
            "signal_state": "前层路由估计的专家热度及偏差",
            "action": "推理专家复制/打包与设备映射；训练另调度all-to-all与allreduce",
            "objective": "分布式MoE训练及一批推理的时间",
            "evidence_regime": "最多16张A100 40GB、100Gbps IB；推理top1、训练top2",
            "guarantee_and_cost": "批次推理时间不是在线请求TTFT；训练通信优先级不能直接迁移为decode策略。",
            "relation_to_current": "需要真实多卡专家与通信路径；当前没有这些动作或共同测量分母。",
            "reading_level": "本地PDF前3页及相关推理/环境段，官方ATC书目"
          },
          {
            "id": "drf",
            "name": "DRF",
            "title": "Dominant Resource Fairness: Fair Allocation of Multiple Resource Types",
            "authors": "Ali Ghodsi et al.",
            "publication": "NSDI 2011（基础工作）",
            "dedup_key": "usenix:nsdi11:ghodsi",
            "url": "https://www.usenix.org/conference/nsdi11/dominant-resource-fairness-fair-allocation-multiple-resource-types",
            "primary_text": "https://www.usenix.org/events/nsdi11/tech/full_papers/Ghodsi.pdf",
            "signal_state": "租户在各资源上的份额与需求向量",
            "action": "按最大资源份额做多资源max-min公平分配",
            "objective": "多资源共享的公平与效率",
            "evidence_regime": "固定资源需求模型及Mesos评估",
            "guarantee_and_cost": "理论性质依赖需求/分配模型；异构GPU效率、动态KV增长和不可切分gang不能直接套用。",
            "relation_to_current": "适合解释资源份额公平；VTC计的是随时间获得的token服务，二者对象不同。",
            "reading_level": "官方书目与正文摘要、动机、分配定义及性质段"
          }
        ],
        "官方文档": [
          {
            "id": "vllm026",
            "title": "vLLM v0.26.0 Optimization and Tuning",
            "url": "https://docs.vllm.ai/en/v0.26.0/configuration/optimization/",
            "checked": "Preemption、Chunked Prefill及调优部分",
            "fact": "V1重算恢复与chunked prefill预算权衡；文档建议必须在具体配置中验证，默认行为不能覆盖实验实际patch。"
          },
          {
            "id": "k8s_framework",
            "title": "Kubernetes Scheduling Framework",
            "url": "https://kubernetes.io/docs/concepts/scheduling-eviction/scheduling-framework/",
            "checked": "Filter/Score、Reserve/Unreserve、Permit",
            "fact": "选择可行节点、排序、临时记账及绑定前等待；不是每个token的调度循环。"
          },
          {
            "id": "kueue",
            "title": "Kueue Overview",
            "url": "https://kueue.sigs.k8s.io/docs/overview/",
            "checked": "职责与队列/配额机制",
            "fact": "决定workload何时准入或被抢占，不替代kube-scheduler的Pod到Node放置。"
          },
          {
            "id": "kueue_gang",
            "title": "Kueue All-or-nothing Scheduling",
            "url": "https://kueue.sigs.k8s.io/docs/concepts/all_or_nothing/",
            "checked": "quota、TAS、waitForPodsReady与ProvisioningRequest边界",
            "fact": "总配额可预留不等于节点布局放得下；等待Ready超时回队属于补救，不能称所有Pod原子启动。"
          },
          {
            "id": "hpa",
            "title": "Kubernetes Horizontal Pod Autoscaling",
            "url": "https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/",
            "checked": "周期控制、指标与稳定窗口",
            "fact": "HPA改变副本数；稳定窗口可限制抖动。扩副本不是引擎内立即放行请求。"
          }
        ],
        "综合判断": [
          "迭代合批、prefill切分、最大长度预留和按prompt排序都已有明确先例。当前实验更适合解释这些普通动作在特定原生运行域中的完整代价；某个小预算或排序负结果，不构成对应方法家族失败。",
          "公平性的对象需要先明确：VTC计量已获得的token服务，DRF讨论多资源份额；缓存局部性、平均完成时间和最坏等待也不是同一目标。调度器必须说明让谁受益、谁承担等待，以及保证在哪些条件下成立。",
          "多副本系统增加了阶段资源配置、decode worker选择和运行中迁移等动作，但同时引入KV复制、网络与目的端容量。单卡cap或队列结果不能直接支持这些系统的收益；核对时按共同动作和完整成本分母比较。",
          "MoE邻近工作已经覆盖专家/KV预算、权重驻留回收、专家签名路由和通信调度。下一步问题应来自强简单策略之后仍存在的请求损害；不同硬件、不同精度或将特征放在另一个层次，本身都不是独立新颖性。"
        ],
        "口述答案与来源": {
          "相邻01_逐迭代加请求不就是Orca_你的增量在哪": [
            "orca"
          ],
          "相邻02_cap29按最大长度预留_Orca不是早做过了吗": [
            "orca"
          ],
          "相邻03_PagedAttention已经减少碎片_为什么还会缺KV": [
            "pagedattention",
            "vllm026"
          ],
          "相邻04_Sarathi说chunked_prefill有效_你的小预算结果是不是反例": [
            "sarathi",
            "vllm026"
          ],
          "相邻05_短prompt优先与TetriInfer的SJF有什么区别": [
            "tetriinfer"
          ],
          "相邻06_FCFS_SJF_SRPT_MLFQ到了LLM里怎样区分": [
            "fastserve",
            "tetriinfer"
          ],
          "相邻07_既然FastServe能抢占_为什么不频繁抢占照顾短请求": [
            "fastserve",
            "vllm026"
          ],
          "相邻08_每个租户轮流取一条请求为什么不够公平": [
            "vtc"
          ],
          "相邻09_VTC每个token记账_是不是每个token都抢占": [
            "vtc"
          ],
          "相邻10_DRF与VTC都讲公平_可以直接互换吗": [
            "drf",
            "vtc"
          ],
          "相邻11_DistServe已经做PD分离_为什么还调单卡预算": [
            "distserve"
          ],
          "相邻12_TetriInfer预测输出长度_为什么你还要保守预留": [
            "tetriinfer"
          ],
          "相邻13_Llumnix为什么还迁移运行中请求_到达时路由不够吗": [
            "llumnix"
          ],
          "相邻14_缓存命中最多优先一定比短prompt优先好吗": [
            "sglang"
          ],
          "相邻15_你说的碎片到底是KV碎片_副本碎片还是节点碎片": [
            "pagedattention",
            "llumnix",
            "kueue_gang"
          ],
          "相邻16_专家和KV抢显存_WiSP不是已经做了吗": [
            "wisp"
          ],
          "相邻17_ELDR已经加expert_signature_你再加特征有什么不同": [
            "eldr"
          ],
          "相邻18_WiSP说预取没用_FluxMoE却流水加载_谁对": [
            "wisp",
            "fluxmoe"
          ],
          "相邻19_Lina和Aurora优化调度_能直接当你的性能基线吗": [
            "lina",
            "aurora"
          ],
          "相邻20_Kueue准入_kube_scheduler和你的cap各管哪一步": [
            "kueue",
            "k8s_framework",
            "vllm026"
          ],
          "相邻21_集群一共空8张卡_为什么8卡任务还可能起不来": [
            "kueue_gang"
          ],
          "相邻22_Filter_Score_Reserve_Permit怎样接住调度并发与失败": [
            "k8s_framework"
          ],
          "相邻23_GPU利用率高就HPA扩容_还需要引擎内调度吗": [
            "hpa"
          ],
          "相邻24_这些论文都要复现一遍才算强基线吗": [
            "sarathi",
            "tetriinfer",
            "fastserve",
            "distserve",
            "llumnix"
          ],
          "相邻25_这些方向都有人做_你的论文还能贡献什么": [
            "orca",
            "tetriinfer",
            "wisp",
            "eldr"
          ]
        },
        "已有追问的补充来源": {
          "调度17_prefill预算变小不是能保护decode吗_结果如何": [
            "sarathi",
            "vllm026"
          ],
          "调度20_这就是最短作业优先吧_为什么还要做实验": [
            "tetriinfer",
            "fastserve"
          ],
          "调度21_短请求持续进来_长请求会不会饿死": [
            "tetriinfer",
            "vtc",
            "sglang"
          ],
          "调度24_为什么安全cap恰好是29_是试出来的吗": [
            "orca",
            "pagedattention"
          ],
          "调度32_下一步真加专家负载信号_怎么证明不是普通负载的替身": [
            "eldr"
          ],
          "调度36_多租户按请求数还是token数公平_高优先级能无限插队吗": [
            "vtc",
            "drf"
          ],
          "调度37_prefill干扰decode_为什么不直接分离两个阶段": [
            "distserve",
            "tetriinfer"
          ],
          "调度40_多副本选最短队列还是前缀缓存_能迁移正在生成的请求吗": [
            "sglang",
            "llumnix"
          ]
        },
        "最先复习": "先用相邻01–05回答与当前实验直接相撞的问题，再按面试方向选公平性、PD/迁移或MoE工作；发表状态、版本和实验运行域需要与方法名一起记。",
        "下一步保持聚焦": "当前仍先完成已有kv_budget_r02；文献调研不自动开启新控制器或改变原实验状态。"
      },
      "scheduling_followup_sequences": {
        "项目来龙去脉与下一步": [
          "现在到底研究什么",
          "早期混合精度工作怎样影响现在的研究",
          "为什么从旧RCBA候选转向请求级调度实验",
          "你已经做到了什么",
          "最重要的正向发现是什么",
          "最重要的负向发现是什么",
          "调度34_下一步到底改什么_不是再泛泛探索吧"
        ],
        "范围与实际代码": [
          "现在到底研究什么",
          "调度01_你做的是Kubernetes选GPU还是推理引擎调度",
          "调度02_这些问题Dense模型也有_为什么说是MoE研究",
          "调度03_一个请求从到达到完成_你到底在哪改了代码",
          "调度31_除了调用vLLM_你自己实现了什么"
        ],
        "准入反馈为什么没赢": [
          "调度04_cap和token预算有什么区别_举个具体例子",
          "调度05_把cap从32降到16_是不是立刻停掉一半请求",
          "调度06_你怎么证明写入的调度参数真正起作用了",
          "调度16_看到延迟变坏再降低并发_为什么可能已经来不及",
          "调度11_新反馈比旧反馈提高61%_为什么不作为主要成果",
          "调度12_最好静态cap是事后选的_拿它比是不是不公平"
        ],
        "预算与等待顺序的代价": [
          "调度17_prefill预算变小不是能保护decode吗_结果如何",
          "调度18_256不好又试512_不就是看结果调参吗",
          "调度19_短prompt优先改了哪个队列_已开始的prefill怎么办",
          "调度20_这就是最短作业优先吧_为什么还要做实验",
          "调度21_短请求持续进来_长请求会不会饿死",
          "调度22_短请求TTFT降低14%_为什么仍不说有效"
        ],
        "KV容量与抢占拷问": [
          "调度23_prefill已经成功_为什么decode过程中还会KV不足",
          "调度24_为什么安全cap恰好是29_是试出来的吗",
          "调度25_线上不知道实际输出长度_这个容量公式怎么用",
          "调度26_零抢占不是更安全吗_为什么原生cap32反而更好",
          "调度27_那4_5秒是不是全部花在重算_重算税怎么算",
          "调度08_平均TPOT和token_p99都很好_用户还可能卡几秒吗"
        ],
        "指标与因果证据": [
          "调度07_TTFT_TPOT_请求延迟分别怎么计算_排队在哪里",
          "调度09_goodput怎么定义_失败和没完成的请求怎么算",
          "调度10_SLO阈值是不是换一下就能让你的策略赢",
          "调度13_为什么不能固定一条route_trace离线换cap评估策略",
          "调度14_相同输入和seed_为什么还不能说同状态因果对照",
          "调度15_不同策略生成的token都不同_性能比较还有效吗",
          "调度30_才16或32条请求_你的p99和重复有说服力吗"
        ],
        "研究贡献与边界": [
          "调度28_并发越小单步不该越快吗_为什么要保持同一个引擎比较",
          "调度29_原生MoE换了Companion输出没变_是不是推翻旧发现",
          "调度32_下一步真加专家负载信号_怎么证明不是普通负载的替身",
          "创新点是什么",
          "如果最终没有收益怎么办",
          "调度33_重复数据还没回传_退出0能代替结果吗"
        ],
        "过载公平性与SLO设计": [
          "调度35_持续过载时排队还是拒绝_怎么避免拒绝越多指标越好",
          "调度36_多租户按请求数还是token数公平_高优先级能无限插队吗",
          "调度38_怎么定义SLO剩余时间_快超时就一定先跑吗",
          "调度39_输出长度未知还允许KV超卖_预测错了怎么办"
        ],
        "阶段分离与多副本设计": [
          "调度37_prefill干扰decode_为什么不直接分离两个阶段",
          "调度40_多副本选最短队列还是前缀缓存_能迁移正在生成的请求吗",
          "调度41_专家信号延迟或丢失_调度器怎么回退",
          "调度42_设计都听起来合理_怎样证明真有收益"
        ],
        "相邻工作_批处理与长度策略": [
          "相邻01_逐迭代加请求不就是Orca_你的增量在哪",
          "相邻02_cap29按最大长度预留_Orca不是早做过了吗",
          "相邻03_PagedAttention已经减少碎片_为什么还会缺KV",
          "相邻04_Sarathi说chunked_prefill有效_你的小预算结果是不是反例",
          "相邻05_短prompt优先与TetriInfer的SJF有什么区别",
          "相邻06_FCFS_SJF_SRPT_MLFQ到了LLM里怎样区分",
          "相邻07_既然FastServe能抢占_为什么不频繁抢占照顾短请求"
        ],
        "相邻工作_服务公平与资源公平": [
          "相邻08_每个租户轮流取一条请求为什么不够公平",
          "相邻09_VTC每个token记账_是不是每个token都抢占",
          "相邻10_DRF与VTC都讲公平_可以直接互换吗"
        ],
        "相邻工作_阶段分离缓存与迁移": [
          "相邻11_DistServe已经做PD分离_为什么还调单卡预算",
          "相邻12_TetriInfer预测输出长度_为什么你还要保守预留",
          "相邻13_Llumnix为什么还迁移运行中请求_到达时路由不够吗",
          "相邻14_缓存命中最多优先一定比短prompt优先好吗",
          "相邻15_你说的碎片到底是KV碎片_副本碎片还是节点碎片"
        ],
        "相邻工作_MoE专家内存与通信": [
          "相邻16_专家和KV抢显存_WiSP不是已经做了吗",
          "相邻17_ELDR已经加expert_signature_你再加特征有什么不同",
          "相邻18_WiSP说预取没用_FluxMoE却流水加载_谁对",
          "相邻19_Lina和Aurora优化调度_能直接当你的性能基线吗"
        ],
        "相邻工作_从引擎到集群与贡献": [
          "相邻20_Kueue准入_kube_scheduler和你的cap各管哪一步",
          "相邻21_集群一共空8张卡_为什么8卡任务还可能起不来",
          "相邻22_Filter_Score_Reserve_Permit怎样接住调度并发与失败",
          "相邻23_GPU利用率高就HPA扩容_还需要引擎内调度吗",
          "相邻24_这些论文都要复现一遍才算强基线吗",
          "相邻25_这些方向都有人做_你的论文还能贡献什么"
        ]
      },
      "high_pressure_answers": {
        "现在到底研究什么": "我研究MoE在线推理中，怎样在请求延迟约束下安排新请求和每步计算量。最近我在原生vLLM上比较了并发上限、prefill token预算、等待队列顺序和KV容量，重点看完整请求是否更快，以及代价落在首token前还是生成过程中。普通策略已经能解释不少变化，专家负载是否还提供独立增量，目前没有证明。",
        "为什么从旧RCBA候选转向请求级调度实验": "RCBA当时想回答局部专家等待能否通过关键路径形成完整收益，但依赖和资源契约没有闭合，正式Oracle也没运行。现在我把工作收敛到已经能真实执行的原生动作：并发、prefill预算、等待排序和KV容量，先用完整请求数据判断普通策略解释了多少变化。它是研究重心调整，不能把未运行的RCBA写成已经被实验全面判死。",
        "你已经做到了什么": "我已经完成动态batch执行差异与KV传播测量，也在原生vLLM上实现并跑了接纳上限、prefill预算、等待顺序和KV容量的请求级对照。保留的数据能回答实际并发、排队、重算和完成时延如何变化，目前主要得到普通策略的收益与代价边界。专家感知策略的稳定增量、任务质量和多卡性能还没有结果。",
        "最重要的正向发现是什么": "最近最明确的结果是，把原生恢复基线补全后，允许抢占的cap32比保守cap29吞吐约高17%，说明少量KV超量需求不等于整批运行失败。但它同时让两条请求承担约2秒和4.5秒的暂停，而cap29把更多等待放到末尾请求的首token前。这个结果纠正了把零抢占当性能目标的判断，也给出了下一步KV预算对照的具体问题。",
        "最重要的负向发现是什么": "近期几种直觉优化都没有直接形成稳定的整体收益：新反馈没稳定超过最好静态cap，小prefill预算改善部分token尾间隔却增加TTFT，短prompt优先又把等待转给长请求。这说明需要同时看动作是否生效、代价落给谁和完整请求目标。当前失败只限这些动作与运行域，不能推出所有调度方法都没有价值。",
        "创新点是什么": "目前还没有验证出独立的专家感知调度创新。已经测到的是局部改善与完整请求目标之间的具体冲突，例如缩短部分token间隔会增加首token等待，避免抢占也可能降低吞吐并把等待推到请求入场前。要形成论文贡献，还需要证明这些边界在代表性场景中可复现，并解释普通长度、KV和负载指标之外是否存在MoE特有原因；已有规则的组合本身不算创新。",
        "为什么不先做Controller": "我已经做过简单反馈和真实调度动作对照，但没有继续扩展复杂控制器，因为稳定地超过强简单基线这一点还没有成立。先比较固定并发、普通延迟反馈、token预算和队列顺序，可以判断究竟缺动作空间、反馈时机还是可用信号。只有这些策略留下可重复的完整请求收益空间，才值得增加专家特征或更复杂模型。",
        "调度34_下一步到底改什么_不是再泛泛探索吧": "下一步固定并发32，只把vLLM显存预算在90%和95%之间做四项反向顺序对照，先确认95%臂真实KV足以容纳所有请求的声明最大长度，再比较完整请求和长暂停。第一次尝试在模型加载前被资源冲突拦住，当前没有这轮性能结果。即使有效，它首先证明普通KV配置能解决问题，不能当成同资源预算下的新调度算法收益。",
        "为什么Oracle优先": "在相同动作集合、资源约束和目标下，理想Oracle可以帮助判断还剩多少可改进空间；如果这个上界本身很小，就不值得继续优化在线选择器。这个判断要求Oracle确实覆盖待比较策略的动作，并正确重算动作后的状态与成本，不能删掉必需依赖或复用不成立的未来route。RCBA的正式Oracle尚未运行，现有最优静态点也不是动态Oracle。",
        "单张RTX5090能证明什么": "单卡也能测请求级系统行为，关键是有没有真实到达、排队、逐token时间和完整结束记录。我已经在单卡原生vLLM上测过TTFT、平均TPOT、ITL、吞吐以及准入和抢占的影响；旧算子微基准本身仍不能支持这些请求级结论。单卡数据不能证明多卡EP通信、NCCL/RDMA、跨节点拥塞或生产服务效果。",
        "如何避免调参救结果": "每轮先写清唯一问题、对照、指标和停止条件，所有正式、预热、失败和重复都保留，结果变号时先原样重复。探索可以依据新证据选择下一实验，例如在256预算出现权衡后预先选定512中间点，但不能覆盖旧结果或把看过的输入称为全新holdout。当前静态预算已经停止继续扫描，不能靠换阈值或只挑最好一次得到加速结论。",
        "如果最终没有收益怎么办": "就停止这个具体策略在当前运行域里的扩展，说明失败的是动作时机、完整成本还是简单基线已经覆盖收益。已有结果可以积累为调度代价如何转移的测量证据，但单个小样本负结果还不足以自动成为一篇论文。要写成主线，需要一条统一的因果解释和代表性验证；达不到就继续收敛问题，不承诺已有在线加速方法。",
        "这项研究生产落地了吗": "没有生产部署。已落地的是实验侧实现和实际GPU运行：既有自定义执行一致性测量，也有原生vLLM的请求级调度对照。科研主机制和RCBA正式Oracle仍未成立，不能说已经上线一个专家感知推理调度器。",
        "调度01_你做的是Kubernetes选GPU还是推理引擎调度": "这部分论文工作做的是推理引擎内部的请求和token调度，模型已经放在GPU上。Kubernetes调度决定Pod放到哪个节点；我这里决定等待请求什么时候进入运行集合、每步安排多少prefill和decode，以及KV不足后如何等待或恢复。两层会耦合，但本轮没有实现跨节点GPU放置或迁移。",
        "调度02_这些问题Dense模型也有_为什么说是MoE研究": "队列、KV和prefill竞争在Dense模型里也存在，所以在MoE模型上跑出差异还不能证明MoE专属贡献。MoE还会把token分到不同专家，专家收到的token数、分组和实际执行形状可能改变每步成本；当前首先要检验普通长度、并发和KV指标能解释多少。只有相同简单基线之后，专家状态仍有稳定且能被动作利用的增量，才有理由主张专家感知方法。",
        "调度03_一个请求从到达到完成_你到底在哪改了代码": "驱动按冻结到达时间把请求提交给vLLM，进入native waiting队列，由scheduler安排prefill分块和后续decode，模型执行后返回token，直到请求完成。我在实验侧接入scheduler调用前后的观测，并分别修改接纳上限、实际token预算或首次入场前的等待顺序；每轮只变其中一个问题。所有到期请求都提交给引擎，没有用客户端并发上限把等待藏在测量之外。",
        "调度04_cap和token预算有什么区别_举个具体例子": "cap限制同时运行的请求数，token预算限制一个调度步骤安排的prefill与decode token总量。比如cap8且当前有6条decode，每条本步推进1个token就先占6个预算；若KV等条件允许，剩余预算可以给至多2个新请求做prefill，而cap已满时不能仅因还有token预算就再接新请求。我做预算对照时固定cap8和编译容量1024，只在引擎排空后切换实际调度预算为256、512或1024。",
        "调度05_把cap从32降到16_是不是立刻停掉一半请求": "不会，降cap只限制新的接纳，已经运行的请求继续推进并自然结束。假设还有24条在运行，目标降到16后先不补新请求，等实际运行数降到16以下才有新接纳空间；不能截取running前16条继续算。我还逐步核对已有decode是否都被实际调度，不能只靠参数名称宣称非抢占。",
        "调度06_你怎么证明写入的调度参数真正起作用了": "我看实际running和waiting数量、每步安排的token以及首次限制新接纳的时刻，区分写了目标和动作真正生效。反馈轨迹中有过第一次降cap时waiting为零的情况，那个动作当下并没有改变排队请求的入场；短prompt实验则记录到真实重排和首次调度顺序改变。预算对照也检查基线是否确实安排过超过小预算的token，否则只能报告这次没有暴露动作差异。",
        "调度07_TTFT_TPOT_请求延迟分别怎么计算_排队在哪里": "TTFT是首token接收时刻减计划到达时刻，请求延迟是完成时刻减到达时刻；平均TPOT是首末token时间差除以输出token数减一。它们使用同一host时钟，包含实际等待和执行开销。我保存了提交、首次调度和逐token接收时间；已有queue_s字段只是提交滞后，不能把它当成native队列的全部等待时间。",
        "调度08_平均TPOT和token_p99都很好_用户还可能卡几秒吗": "可能，平均值会摊薄停顿，把所有token间隔混在一起的p99也可能漏掉极少数受损请求。原生cap32实验中，pooled-token ITL p99约28毫秒，但两条请求仍分别停顿约2秒和4.5秒；每请求最大ITL的p99则约3.7秒。因此我会同时看单请求平均TPOT、逐token间隔和每请求最坏间隔，并明确分位数的统计单位。",
        "调度09_goodput怎么定义_失败和没完成的请求怎么算": "这里的请求goodput是同时满足预定TTFT和平均TPOT要求的完整请求数，除以整个episode的实际持续时间。已经到达的失败和未完成请求保留在结果中，不能算成功或从达标比例的分母中删掉；中止前尚未到达的计划请求另外列出。它与输出tokens每秒不同，固定小episode的结果也不能直接称为长期稳定承载容量。",
        "调度10_SLO阈值是不是换一下就能让你的策略赢": "阈值确实可能改变达标数量，所以同一对照固定阈值，并同时报告连续TTFT、TPOT和完成时间。短文本档位实验用200毫秒TTFT和9毫秒平均TPOT，预算与长上下文实验另有5秒和200毫秒的参考口径，两组不能混着比较。预算和排序对照全部通过宽松参考SLO时，只能讨论连续延迟，不能据此宣称安全容量提升；若要换成有区分度的SLO，应先说明依据并在独立运行中验证。",
        "调度11_新反馈比旧反馈提高61%_为什么不作为主要成果": "因为旧反馈不是最强基线。突发到达的四组对照里，新档位相对旧反馈改善约44%到61%，但四组都没有超过同引擎已测的最好静态cap；平稳到达只出现一次胜出，原样重复没有支持。这个结果说明旧规则还有改进空间，不能证明动态反馈优于简单静态策略。",
        "调度12_最好静态cap是事后选的_拿它比是不是不公平": "它是已测静态点的事后上包络，我用它回答当前策略是否留下了超过简单配置的研究空间，没有把它称为可在线部署的选择器。要做实际部署比较，应该在校准数据上选择固定cap，再到独立到达episode上与动态策略比较。事后最好静态点也不是动态Oracle，更不能把不同cap的未来trace拼成一个可执行策略。",
        "调度13_为什么不能固定一条route_trace离线换cap评估策略": "因为cap会改变接纳请求、batch形状、KV和后续路由，另一档位的未来route不是原trace的外生输入。我让不同策略从相同请求与到达计划独立执行，重新生成各自的状态和完成时间。历史route仍可以用于预测下一窗口，但要遵守动作前的信息截止；预测相关性和动作收益需要分别验证。",
        "调度14_相同输入和seed_为什么还不能说同状态因果对照": "相同输入和seed只固定了实验的一部分，host调度时刻跨过某个到达点后，实际batch就可能不同。已有反馈重复中，第一次prefill或batch差异出现在降cap之前，所以不能把最终差异全部归因于反馈动作。A/B/C/D数值实验从同一中间KV状态分叉，而整段serving对照是相同输入到达计划下独立推进，这两种控制强度不同。",
        "调度15_不同策略生成的token都不同_性能比较还有效吗": "它可以回答各策略在相同输入、固定输出预算下实际承担了多少系统成本，因为KV、route和生成轨迹本来就会随动作变化。但不能把这个结果说成相同token序列上的纯执行加速，也不能据此说质量保持。当前固定生成128或1024个token是为了控制输出预算，真实EOS、输出长度分布和任务质量需要另外验证。",
        "调度16_看到延迟变坏再降低并发_为什么可能已经来不及": "非抢占降cap只能影响后续接纳，已经进入running的请求和它们占用的KV不会因为目标变小立刻消失。如果拥塞主要由这批已有请求造成，或者触发时已没有waiting请求，控制动作短时间内就没有足够作用对象。实际轨迹里我分别记录触发时刻和首次限制接纳的时刻，不能把响应慢直接归咎于预测模型不够复杂。",
        "调度17_prefill预算变小不是能保护decode吗_结果如何": "它可以限制一次prefill与decode混合执行的工作量，但长prompt要分更多次推进，也可能增加其他请求经历混合步骤的次数。在cap8的长短混合输入里，256相对1024预算使平均TTFT增加约32%到35%、整批完成时间增加约9%，虽然部分ITL尾部缩短。预先选择的512中间点仍增加TTFT约6%到8%和完成时间约2%，因此当前结果是权衡，不能只拿ITL p99下降约29%到30%宣称整体加速。",
        "调度18_256不好又试512_不就是看结果调参吗": "这是依据前一轮权衡选择的一次探索性中间点检查，512在新一轮执行前已经固定，256的原始结果也完整保留。我没有把这称为独立新数据上的确认，更没有继续扫预算直到找到赢家。512仍没消除完整请求代价后就停止这条静态预算扫描。",
        "调度19_短prompt优先改了哪个队列_已开始的prefill怎么办": "只重排native FCFS scheduler中已经到达、从未开始执行的waiting请求，按prompt长度稳定排序，同长度保持原序。代码核对它从未出现在实际调度结果中，computed和output token都为零；running里的partial prefill与decode保持原对象、顺序和状态。随后仍调用原生scheduler，实验没有把已开始的长请求抽回等待队列。",
        "调度20_这就是最短作业优先吧_为什么还要做实验": "它只是短prompt优先，prompt长度不是请求的完整服务时间，生成长度、KV和共享batch执行成本也会影响结果。我把它作为普通简单基线，检查等待队列里是否存在真实可用动作，以及完整成本由谁承担，没有把这个排序规则当创新。固定输出长度的实验有利于减少一个变量，但还没有覆盖真实生成长度未知的场景。",
        "调度21_短请求持续进来_长请求会不会饿死": "当前纯短prompt优先没有无饥饿保证，16条固定请求最终全部完成不能证明持续到达下的公平性。若扩展成在线策略，我会先加明确的最大等待或aging约束，并记录短请求要为这个保障付出多少代价。那属于后续设计，当前实现没有这项保障，也不能只展示短请求收益。",
        "调度22_短请求TTFT降低14%_为什么仍不说有效": "因为当前目标还包括长请求和完整请求成本。三个已回传block中，短请求TTFT下降约14%、10%、9%，长请求TTFT却增加约3%、6%、5%，长请求完成延迟p95也上升；整体平均完成延迟只下降约2.52%、0.22%、0.03%。所以能说排序改变了入场和代价分配，还不能说形成了稳定的整体净收益。",
        "调度23_prefill已经成功_为什么decode过程中还会KV不足": "prefill只建立输入前缀的KV，后续生成每推进token还会增加缓存需求。当前长输入实验里，32条请求全部完成prefill后，随着decode长度增长才触到KV池边界；只检查入场瞬间的free blocks就可能低估后续需求。Paged KV按块分配能改变管理方式，但不会消除总容量约束。",
        "调度24_为什么安全cap恰好是29_是试出来的吗": "这个值由该次引擎的live布局和声明最大长度计算，先于测量写入记录。可用KV池是7677块，每块16个token；每条请求最多3072输入加1024输出，需要ceil(4096/16)=256块，所以floor(7677/256)=29。它只是在该单组布局、无prefix共享及无额外lookahead等条件下的保守容量上限，不是性能最优cap，也不是所有显存占用的通用安全证明。",
        "调度25_线上不知道实际输出长度_这个容量公式怎么用": "当前公式用请求声明的最大输出长度做保守预留，并没有预测实际EOS。线上如果按max_tokens预留，短输出请求会导致部分容量闲置；如果按预计剩余长度超卖，就需要承担预测误差和抢占恢复风险。我的固定长度实验没有验证这种预测接纳策略，也不能把cap29直接推广成所有模型或请求组合的最优值。",
        "调度26_零抢占不是更安全吗_为什么原生cap32反而更好": "容量不触边不等于请求性能更好。允许原生抢占并完整恢复的cap32，相对保守cap29的吞吐在两次对照中提高约17%，但两条请求承担了约2秒和4.5秒的生成暂停；cap29没有抢占，却让末尾三条请求的首token等待接近18秒。要按业务约束选择在哪一段允许等待，不能把零抢占当成最终目标。",
        "调度27_那4_5秒是不是全部花在重算_重算税怎么算": "不是，大部分时间发生在被抢占请求重新开始执行之前。一次约4.47秒的最长间隔中，上次token返回到首个重算调用约4.36秒，重算调用首尾跨度约0.116秒；后者也包含其他请求工作与采集，并非纯GPU kernel时间。每轮记录到7691个重复计算位置，可以说明额外工作量，但不能直接换算成4.5秒计算税，也不能把重叠的请求等待跨请求相加成总耗时。",
        "调度28_并发越小单步不该越快吗_为什么要保持同一个引擎比较": "减少逻辑请求数不保证实际kernel时间按比例下降，图捕获档位、padding和执行形状都可能使成本呈阶梯变化。我因此保留共同的编译容量和KV池，在同一引擎内比较实际接纳档位或调度预算，避免把不同初始化配置的成本混进策略收益。究竟是哪一层kernel造成某个差值仍需直接测量，不能从吞吐变化反推专家瓶颈。",
        "调度29_原生MoE换了Companion输出没变_是不是推翻旧发现": "没有，两次实验控制的对象不同。原生局部探针固定零索引layer3、16行和目标输入位置，共36次调用，专家内部位置和负载确实改变，但目标router、top-k与输出逐位相同；旧实验还涉及不同批宽以及上游Attention、KV和padding。当前只能说这组原生局部干预没有复现目标输出差异，既不能宣布全局数值稳定，也不能继续把旧现象说成已在原生serving复现。",
        "调度30_才16或32条请求_你的p99和重复有说服力吗": "这足以做小规模存在性和动作对照，但不能支撑生产级尾延迟分布或跨负载泛化。每格只有16或32条请求，p99基本接近最慢样本；多次运行复用这些文档，也不能把累计请求执行数当成新的独立文档数。当前逐block报告方向和完整结果，不把相邻token当独立样本；进一步确认应使用独立文档或到达episode，并覆盖不同长度和负载域。",
        "调度31_除了调用vLLM_你自己实现了什么": "我实现的是实验侧的真实动作接入和证据采集：接纳限制、排空后的预算切换、首次入场排序，以及请求、step、KV、抢占和完成时间的关联记录与复算。Paged KV、原生running调度、Attention和MoE执行核，以及默认抢占重算流程都复用vLLM；允许原生抢占的实验是观测并放行已有机制。我的贡献可以具体落到动作边界和完整成本的验证，不能说重写了推理引擎。",
        "调度32_下一步真加专家负载信号_怎么证明不是普通负载的替身": "先固定可执行动作和成本口径，把输入长度、输出预算、active与waiting、KV占用、当前每专家负载和近期延迟作为基线，再检查额外路由历史是否提供稳定增量。特征必须在决策前可见，按独立请求或到达episode切分，不能随机打散相邻decode窗口。预测更准还不够，还要在同资源约束下真实执行动作，证明完整请求收益超过最强简单策略；这部分目前未验证。",
        "调度33_重复数据还没回传_退出0能代替结果吗": "不能，退出0只能证明进程结束，不能代替请求时间、输出、环境和动作记录。短prompt对照目前只有首次forward、reverse及原样重复forward这三个block的完整数据在本地，重复reverse只确认过远端完成，不能引用它的性能数字。回传缺口和实验失败是两件事，应该先补齐已有raw，而不是重跑一份更好看的结果替换。",
        "调度35_持续过载时排队还是拒绝_怎么避免拒绝越多指标越好": "如果扩展到在线服务，我会同时限制等待请求数和待处理token量，超过预定边界就拒绝新请求，避免队列无限增长；已接纳请求按原契约处理。准入还要结合等待时间和资源余量，不能只看GPU利用率。验证保留全部到达请求，分别报告拒绝、超时、达标比例及goodput，不能只展示被接受请求的低延迟；当前实验没有实现这套过载控制。",
        "调度36_多租户按请求数还是token数公平_高优先级能无限插队吗": "未来先明确公平性的资源口径，把可借用的并发和KV额度与长期服务份额分开；请求数和token数都只是成本近似。队列先按租户权重分配服务机会，再在租户内处理优先级和等待时间，高优先级也受额度约束。验证要看低优租户的最长等待与实际服务份额；当前单租户有限请求实验没有实现或证明多租户公平。",
        "调度37_prefill干扰decode_为什么不直接分离两个阶段": "阶段分离是可讨论的设计，但要先区分单卡执行时段隔离和不同副本物理隔离。若进入这一分支，我会先比较带最大等待限制的prefill准入窗口，检查保护decode后是否损害TTFT和整批完成，再决定是否需要跨副本分离。物理分离还必须计入KV传输、额外排队和两侧负载不均；当前没有实现或验证该方案。",
        "调度38_怎么定义SLO剩余时间_快超时就一定先跑吗": "若设计SLO调度，我会按契约分别定义余量：首token前用TTFT截止时间减当前时刻和预计剩余prefill成本，首token后只有业务明确要求最大token间隔才定义下一token截止时间。平均TPOT不能直接改写成逐token硬截止，成本估计也要保留误差余量。余量小不等于无限优先，还要区分是否仍能挽救，避免持续失约请求挤掉其他请求；这套策略尚未实现。",
        "调度39_输出长度未知还允许KV超卖_预测错了怎么办": "如果探索超卖，我会保留按声明最大输出长度预留的保守基线，再测试按已用KV加预计增长量接纳的风险策略。策略要明确空闲块余量、停止新接纳的时机，以及已有请求继续增长时的恢复顺序；停止接纳本身不能保证已有请求不触边。只有恢复路径和暂停能被业务接受才允许超卖，否则继续保守预留；比较必须计入恢复成本和受损请求，目前没有实现这种预测接纳。",
        "调度40_多副本选最短队列还是前缀缓存_能迁移正在生成的请求吗": "未来先只对未执行请求选择副本，综合预计等待、可实际命中的前缀长度和剩余prefill成本。缓存亲和是偏好，副本过载或信息过期时应允许转向其他副本，否则可能为省计算付出更长等待。对照包含轮询、普通负载选择和受负载约束的缓存亲和，并保持总GPU资源一致；当前没有跨副本结果，也不假定生成中请求能无成本迁移。",
        "调度41_专家信号延迟或丢失_调度器怎么回退": "如果以后接入专家信号，每条记录都要带采集时刻、执行窗口和配置版本，超时或无法对齐就判不可用，不能把缺失当零负载。决策不等遥测返回，而退回已验证的普通长度、队列和KV规则，并限制信号恢复时的策略频繁切换。验证会注入延迟、丢失和旧版本记录，看完整请求代价及回退行为；这些是未来设计要求。",
        "调度42_设计都听起来合理_怎样证明真有收益": "先冻结一个动作、目标、总资源和失败口径，让默认策略、校准后选定的强简单策略与待测策略在独立到达episode中真实执行。加一个承担相同采集与决策开销、但取消关键选择逻辑的对照，区分选择收益和额外成本。每个策略独立推进队列、KV和完成轨迹，保留拒绝、超时、恢复及受损请求；只有简单基线之外有完整请求余量，才扩大确认实验。",
        "Serial_Width-only_原始Companion_等长替换Companion分别控制什么": "A 是 target 的同 pre-step 串行参照；B 保留原 batch width，其他行用 target 独立拷贝，用来识别单纯批宽影响；C 恢复原始 Companion、行顺序、KV 长度和 padding；D 在保持 width、各行 KV 长度和 padding 向量不变的前提下，把 Companion 换成等长的不同文档请求。A/B 识别批宽执行影响，A/C 包含异构物理形状和原始 Companion 影响，C/D 进一步识别 Companion 身份影响。",
        "怎么保证四组对照从同一pre-step状态开始": "我先按原 batch 历史重建到目标步，核对 batch membership、decode step、token、各请求逻辑 KV 长度、Route 和预测 token，然后在目标调用前深拷贝各层 K/V 到 A/B/C/D，并检查存储地址不别名。每个 arm 重复三次、每次重置随机种子，运行顺序正反交替；相同 arm 的 target tensor、Route、Logits、KV 和 token 指纹都稳定后，才比较跨 arm 差异。",
        "首次执行差异定位到哪里": "在六个已选事件中，A/C 的第一个 non-allclose 差异都出现在第 0 层 self-attention output，说明异构 KV/padding 物理形状可以在进入 router 前改变 target 的数值路径。C/D 保持 width 和完整 KV-length/padding 向量相同，第一个观测差异出现在 MoE output，支持 Companion 身份与 MoE 输出差异相关。实验没有采集每个 expert 的物理 M、token 顺序、kernel 选择和 combine 中间量，因此具体算子原因仍未定位。",
        "Route和Logits差异是否证明输出质量或Serving性能已受影响": "当前证据只能说差异在 Companion 移除后仍能通过 arm 自己的 KV 继续传播。四个事件的后续两个 teacher-forced 步骤中，A/C 在 8/8 步保留 Logits 差异，6/8 步保留 Route membership 差异，预测 token 变化是 0/8。这些结果没有得到自由生成质量、TTFT、TPOT、P99、有效吞吐或多卡 EP 结论。",
        "怎么排除它只是自定义runtime现象": "旧的A/C/D事件还没有完成原生decode同状态复现，因此那条数值结论仍限定在自定义runtime。后来做过另一组原生Triton MoE局部对照：固定零索引layer3、16行和目标位置，改变同批请求的顺序与身份，专家内部布局确实变化，但两个自然target的输出都逐位相同。它说明不能把custom现象直接搬到原生后端，也不能用这个固定层宽的局部负结果推翻包含Attention、KV和批宽变化的旧实验。",
        "早期混合精度工作怎样影响现在的研究": "早期混合精度工作让我先问局部节省能不能进入完整请求。fixed RankLane在p_return不超过20%、codec等税按零计的冻结乐观域里，相对uniform FP8的最优端到端改善上界也只有4.1667%，所以停止的是那个具体actuator。现在做调度时，我同样先看完整请求分母和强简单基线，不能因为局部更快就宣布方法成立。",
        "相邻01_逐迭代加请求不就是Orca_你的增量在哪": "是，逐迭代调整请求集合已有明确先例，我复用的是原生引擎能力。我的近期实验检查已有引擎里cap、KV容量和恢复如何改变完整请求等待，因此不把continuous batching本身当创新。能主张的增量要落到实际测到的新边界或超过强基线的动作收益。",
        "相邻02_cap29按最大长度预留_Orca不是早做过了吗": "是，最大长度预留不是新的思想。我的实验是把当前引擎live池换算成保守边界，再与允许原生恢复的cap32比较，说明少抢占是否值得付出入场等待和吞吐代价。公式成立与性能最优是两个问题。",
        "相邻03_PagedAttention已经减少碎片_为什么还会缺KV": "分页减少地址连续性要求、碎片与重复存储，但物理容量仍然有限。生成继续增长时，活跃请求需要的总KV仍可能超过池容量，原生vLLM会通过抢占和重算恢复。我的长输入实验测到的正是这个容量与等待问题，不能说分页管理已经解决所有内存问题。",
        "相邻04_Sarathi说chunked_prefill有效_你的小预算结果是不是反例": "它反映了同一个吞吐与延迟权衡，不是对整个方法的反例。Sarathi-Serve按token间隔目标和硬件成本选择预算；我的cap8对照里小预算改善部分ITL却增加TTFT与整批时间。要比较同目标下的完整代价，不能把预算越小理解成越好。",
        "相邻05_短prompt优先与TetriInfer的SJF有什么区别": "按prompt长度排序已有直接近邻，当前没有新排序算法主张。TetriInfer还在有限请求窗口内排序，并结合PD分离与输出长度预测；我目前只重排native WAITING里从未执行的请求。有限窗口可以限制后来短请求持续插队，但我的小实验尚未实现并验证这类持续流量公平机制。",
        "相邻06_FCFS_SJF_SRPT_MLFQ到了LLM里怎样区分": "FCFS看到达顺序，SJF按预计总服务成本，SRPT需要剩余成本，MLFQ则根据已经获得的执行服务调整优先级。LLM通常知道输入长度，却不知道真实输出长度，而且合批会改变单位成本，所以按prompt排序不能冒充知道最短剩余作业。FastServe用迭代边界的MLFQ处理这一问题；我当前动作只改变首次入场顺序。",
        "相邻07_既然FastServe能抢占_为什么不频繁抢占照顾短请求": "暂停执行不等于释放成本为零，KV仍要驻留、搬运或重算。FastServe专门配套KV换入换出，而我原生缺KV抢占实验已经观察到长恢复等待。若扩展主动抢占，要把换入、带宽竞争和受损请求最大ITL与整体收益一起比较。",
        "相邻08_每个租户轮流取一条请求为什么不够公平": "请求数相同不代表获得的服务量相同，输入、输出长度和成本都可能不同。VTC按已服务的输入与输出token成本记账，在有容量时优先服务不足的客户端。我目前没有实现跨租户账本，短请求先完成也不能证明多租户公平。",
        "相邻09_VTC每个token记账_是不是每个token都抢占": "不是，基本VTC逐token更新服务量，但主要在加入新请求时选择客户端，不抢占已经运行的请求。它在论文模型下限制持续有积压客户端之间的服务差，保证对象是服务量，而不是每条请求的TTFT相同或必达SLO。需要先说清记账频率、动作时机和保证对象。",
        "相邻10_DRF与VTC都讲公平_可以直接互换吗": "不能直接互换，DRF比较租户在多种资源上的最大占用份额，VTC比较随时间获得的token服务成本。前者适合讨论集群资源分配，后者针对共享推理服务的在线服务量。GPU异构效率、KV动态增长和不可切分作业都会影响模型假设，不能只写一个权重就宣称两种公平都保证。",
        "相邻11_DistServe已经做PD分离_为什么还调单卡预算": "两者动作和资源条件不同：单卡预算改变每步工作量，DistServe把两阶段放到不同GPU并分别配置资源。分离能减少共置干扰，但增加KV传输、阶段排队和两侧负载不均。我的当前数据可以解释单卡权衡，尚不能证明分离后净收益。",
        "相邻12_TetriInfer预测输出长度_为什么你还要保守预留": "长度预测可以帮助放置和提高利用率，但预测值不是硬容量保证。TetriInfer把预测输出区间用于调度，我当前用声明最大长度计算边界，回答的是保守预留的成本。若以后采用预测接纳，还必须明确低估后如何停止新接纳、恢复已有请求，并把代价计入结果。",
        "相邻13_Llumnix为什么还迁移运行中请求_到达时路由不够吗": "到达时不知道最终输出长度，后续KV增长会让实例负载改变。Llumnix既分发新请求，也分阶段搬运KV来迁移运行中请求；迁移需要目的端余量、复制带宽和切换协调。当前单实例实验没有覆盖借用其他副本余量，不能拿它的低停顿结果当作本机恢复成本。",
        "相邻14_缓存命中最多优先一定比短prompt优先好吗": "不一定，长prompt若大部分前缀已缓存，未缓存prefill可能更少，但排队时间仍会进入TTFT。RadixAttention利用匹配前缀组织请求，其在线贪心排序也存在饥饿风险。未来应比较未缓存token、等待和尾延迟；我的当前无prefix-cache实验还不能回答这项收益。",
        "相邻15_你说的碎片到底是KV碎片_副本碎片还是节点碎片": "这三个问题分属不同层：PagedAttention处理单实例KV的存储管理，Llumnix处理空闲容量散在多个实例而单实例仍放不下请求，集群调度还要检查GPU是否落在满足Pod和拓扑要求的节点上。一层资源总量够，并不能推出另一层也可执行。定位时应明确资源单位和动作对象，避免把所有等待都叫显存碎片。",
        "相邻16_专家和KV抢显存_WiSP不是已经做了吗": "是，WiSP已经研究专家分页和expert/KV预算分配，所以联合预算本身不能作为我的创新。我目前完成的是普通cap、KV容量与抢占测量，没有实现可回收专家权重的池。若进入这一方向，需要找到同资源条件下强简单策略与已有方法之外的完整请求增量。",
        "相邻17_ELDR已经加expert_signature_你再加特征有什么不同": "仅加专家特征不构成独立贡献。ELDR在prefill后根据专家签名与负载选择decode worker，我当前关注的是单引擎内的准入、预算和等待顺序。即使动作位置不同，也必须证明额外信号在线可得，并在普通负载基线之后实际改善完整请求。",
        "相邻18_WiSP说预取没用_FluxMoE却流水加载_谁对": "要看运行域和真正可覆盖的传输时间。WiSP的低并发PCIe场景里，投机传输可能与必需加载争带宽；FluxMoE在更大batch和分层存储下组织加载流水。专家预测准只是条件之一，计算是否足以覆盖搬运、KV多出来的容量是否改善请求，都要分别测量。",
        "相邻19_Lina和Aurora优化调度_能直接当你的性能基线吗": "它们主要改变专家部署和分布式通信，而我当前动作在单GPU请求队列与KV层。Lina有真实多卡实验，Aurora这次核验的证据主要是统计驱动仿真，二者也不能混成同一种证据。进入相应EP运行域后，才能用相同资源和完整请求分母比较，当前不能直接比较加速数字。",
        "相邻20_Kueue准入_kube_scheduler和你的cap各管哪一步": "Kueue决定workload何时获得配额并启动，kube-scheduler决定Pod落在哪个节点，推理引擎决定哪些请求进入本次运行及每步安排多少token。我的cap作用在最后这一层，前两层容量就绪不代表引擎KV有余量。产品化可以联动这些控制，但不是把每个token都提交给Kubernetes调度。",
        "相邻21_集群一共空8张卡_为什么8卡任务还可能起不来": "总量满足不代表节点和拓扑布局满足，例如单个8卡Pod不能拆到两个各剩4卡的节点。Kueue的配额准入、拓扑容量检查和等待Pod就绪机制处理不同问题；Ready超时回收重试也不是所有Pod原子启动。讨论gang时要说明保证的是配额、放置还是启动后的进度。",
        "相邻22_Filter_Score_Reserve_Permit怎样接住调度并发与失败": "Filter判断节点可行性，Score给可行节点排序，Reserve维护绑定前的临时资源状态，Permit决定批准、拒绝或等待。Reserve或后续阶段失败会触发Unreserve清理，Unreserve必须幂等且不能失败。若以后把实验做成资源控制组件，需要复用这些失败边界，但当前没有实现新的集群调度插件。",
        "相邻23_GPU利用率高就HPA扩容_还需要引擎内调度吗": "需要，HPA改变副本数，引擎内调度决定现有副本怎样服务请求，两个动作的生效时间和成本不同。扩容还要等模型加载和实例可用，当前KV等待也不会因副本目标改写立刻消失。未来设计会分别处理快速准入和较慢容量扩展，避免二者同时调节造成抖动；这套联动尚未实现。",
        "相邻24_这些论文都要复现一遍才算强基线吗": "不需要把不同动作层的系统机械地全跑一遍。先在相同资源、后端和SLO下比较默认及强简单策略，再选与当前动作最接近的机制：预算对应chunked prefill，等待排序对应长度/公平策略，跨副本才考虑迁移和PD分离。论文旧版本的加速比只能作背景，不能代替当前同引擎基线。",
        "相邻25_这些方向都有人做_你的论文还能贡献什么": "需要找到已有方法没有回答、且能重复验证的具体问题，而不是换一个特征或把几个组件拼起来。当前请求等待与暂停的实验能帮助定位边界，但还不足以证明新的系统规律或方法。下一步先完成已冻结的普通KV预算对照；只有剩下完整请求损害，再围绕可执行动作和最接近的工作界定增量。"
      },
      "historical_research_lines": {
        "mixed_precision_fixed_ranklane": {
          "status": "NO_GO_RANKLANE_ACTUATOR_UNDER_P_RETURN_MAX_0_20",
          "已完成": [
            "跨模型 rank-tail 质量结构分析",
            "RTX 5090 单卡 LUT/codec 局部微基准",
            "fixed RankLane 冻结域内端到端上界评估"
          ],
          "裁决": "在 p_return≤20%、codec 等税按零计算的最乐观冻结域内，相对 uniform FP8 的最优端到端改善上界仅 4.1667%，因此停止 fixed RankLane actuator。",
          "边界": "该结果不等于真实 8×A100 return path 不存在；在新的多卡证据达到 reopen condition 前，不实现 RankLane codec/controller，也不把简历表述成方法有效。"
        },
        "dynamic_batch_execution_conformance": {
          "status": "MEASUREMENT_ONLY / CUSTOM_CONTINUOUS_RUNTIME / SUPPORTED_WITH_QUALIFIERS",
          "研究问题": "从同一 target 的重建 pre-step 状态分叉，判断动态 batch 中首次稳定差异来自批宽、异构 KV/padding 物理形状还是 Companion 身份，并检查差异在移除 Companion 后是否继续传播。",
          "共同前置": "A/B/C/D 使用相同 target token、position 和 logical KV，各层 KV 深拷贝且校验存储不别名；每个 arm 运行三次并交替顺序，每次重置随机种子，相同 arm 的 target tensor、Route、Logits、KV 和 token 指纹必须稳定。",
          "四组对照": {
            "A_Serial": "target 单独执行，width=1，使用自然物理 KV 长度，作为同 pre-step 串行参照。",
            "B_Width_only": "保留原 batch width，其他行都是 target 的独立拷贝，token、逻辑长度、position 和语义相同，用于识别单纯批宽执行影响。",
            "C_Original_companions": "恢复原始行顺序、请求身份、forced token、逻辑长度、物理最大长度和 padding，作为原始 batch 执行参照。",
            "D_Matched_alternative_companions": "target 保持原行位置，Companion 尽可能换成不同文档且逻辑 KV 精确等长的请求，使 width、各行长度向量、物理最大长度和 padding 向量与 C 相同，用于识别 Companion 身份影响。"
          },
          "观察": "六个已选事件中，A/C 的第一个 non-allclose 差异均位于第 0 层 self-attention output；在 width 和完整 KV-length/padding 向量相同的 C/D 中，第一个观测差异位于 MoE output。后续传播实验中，A/C 在 8/8 个 teacher-forced 步骤保留 Logits 差异，6/8 保留 Route membership 差异，预测 token 变化为 0/8。",
          "证据边界": "证据只来自一个 OLMoE revision、BF16、单张 RTX 5090 和自定义 cached-decode runtime。它没有定位到具体 MoE grouping/kernel 机制，也没有证明原生 serving 可转移性、最终 token/质量影响、TTFT/TPOT/P99、容量、多卡 EP/NCCL 或可部署方法。源事件的 /tmp 原始 capture 未保留，因此原始事件选择来源不能从当前 bundle 独立重放。"
        },
        "rcba_unvalidated_candidate": {
          "name": "Route-Conditioned Barrier Amplification Boundary",
          "status": "PRIMARY_NEXT_CANDIDATE / UNVALIDATED / BLOCKED_PROTOCOL_AMBIGUITY / FORMAL_GATE_NOT_RUN",
          "question": "自然 continuous-decode workload 中，MoE route identity 与 expert-tail 是否经真实 runtime barrier 放大为两个模型共同存在的 charged full-request critical-path headroom。",
          "已冻结输入": [
            "两个模型及精确 revision 与 BF16 dtype",
            "WikiText-103 test 文本来源",
            "BurstGPT 到达 trace 来源",
            "greedy decode、最多 16 steps、max batch 8、固定 workload seed",
            "候选主指标 J、H 与 route attribution A"
          ],
          "未完成": [
            "唯一 common natural regime",
            "removable barrier whitelist 与逐边依赖语义",
            "resource pool/capacity/concurrency model",
            "route-decorrelation scope 与 seed",
            "双模型 identity-complete canonical runtime trace",
            "覆盖 full DAG 的 measured service surface",
            "evaluator、正式 Oracle Gate 与 headroom 结果"
          ],
          "正式门槛": "同一 common natural regime 中两个模型都满足 H≥10%，且 route attribution A≥3 percentage points；当前没有计算 H 或 A。",
          "在本材料的位置": "历史候选，不再作为当前主讲或唯一下一步；原始未验证、协议未闭合和正式Gate未运行状态保留。没有已完成Oracle能够把整个问题判成实证NO-GO。",
          "当时的工作流": {
            "一句话主线": "冻结唯一研究问题和证据契约 → 获取完整 runtime trace 与 measured service surface → 用受容量约束的 Oracle replay 判断 headroom → 过 Gate 才设计机制，不过门就冻结方向。",
            "当前顺序": [
              "先唯一冻结 natural regime、barrier whitelist、逐边依赖语义、resource capacity 与 route-decorrelation contract。",
              "再获得两个模型的 identity-complete canonical runtime trace 和覆盖 full DAG 的 measured service surface。",
              "实现并验证 REAL_BARRIER、CAPACITY_CONSTRAINED_NO_BARRIER_ORACLE 和 ROUTE_DECORRELATED 三种 replay。",
              "先判断两个模型共同自然场景中的 H 与 A 是否过门；不过门就冻结研究问题。",
              "只有 Oracle headroom 成立后，才选择最小 action space；单卡通过最多授予多卡候选资格，多卡 serving 才能回答系统收益。"
            ],
            "为什么现在没有evaluator": "当前协议歧义中的任一选择都可能翻转 10% Gate；自行补写字段会把实现者偏好带入结论，因此按 fail-closed 原则先不实现。"
          },
          "当时的评价定义": {
            "主要变量": {
              "J": "所有请求 flow time 的均值，flow=request completion-request arrival。",
              "H": "(J_real_barrier-J_no_barrier)/J_real_barrier，衡量删除合法 barrier 后的 charged full-request headroom。",
              "A": "H_original_route-H_route_decorrelated，衡量 route identity 对 barrier headroom 的归因。"
            },
            "证据纪律": [
              "冻结模型、revision、数据来源、split、指标、baseline、阈值与停止条件",
              "保留 raw ledger、manifest、失败尝试和独立重算",
              "区分 CPU exact、single-GPU microbenchmark、fresh proxy、Oracle upper bound 与多卡 serving 证据",
              "失败后不修改 C、预算、workload、分母、split、特征或指标救结果"
            ],
            "停止原则": "若两个模型共同自然 cells 的 headroom 均低于正式门槛、actionable mass 不足，或 route attribution 被简单负载统计完全解释，就冻结 RCBA；不得先写 controller 再为实现寻找问题。"
          }
        }
      },
      "evidence_ledger": {
        "StableBatch": {
          "evidence_tier": "fresh document-disjoint single-GPU proxy / exact Outcome Oracle；不是 serving 或在线机制证据",
          "positive": "16 个全新 requests、240 cells、1,920 actions 上，Outcome Oracle reward=57，恢复 57/84=67.86% unprotected route distance，证明冻结代理栈中存在干预机会。",
          "negative": "Static Compatibility Map 与 Online Observable Ridge 均为 -7，低于 matched shuffle 的 -4；已停止 pre-action selector、partition planner 和 controller。"
        },
        "SemanticFence": {
          "evidence_tier": "pretrained OLMoE / BF16 / single RTX 5090 / fresh route-top-k-stability proxy；不是 task semantics、M2 runtime、serving 或 controller 证据",
          "positive": "SFV2-O1 在 128 条自然 edges 上得到 60.3922% matched-row coverage，证明 fresh proxy action space 与 observability 值得保留。",
          "negative": "frozen witness-v1 只执行 5 pairs，其中 4 unsafe，覆盖 3.9216%；计入实测原型开销后 net projection=-183.3504%，裁决 PIVOT_TO_SHADOW_VERIFY，不授权在线执行。"
        },
        "JoinStream": {
          "evidence_tier": "CPU exact Oracle + synthetic/realistic single-GPU grouped-expert microbenchmark；不是多卡 serving 证据",
          "result": "realistic MoE-tail 最终四个 cells 有 3/4 natural windows、最大 161.664 us，但安全收益为 0/4；保留 action validity、memory legality、single-GPU schedulability 与 producer safety，critical-path utility 被削弱。",
          "verdict": "FREEZE / NO_MORE_EXPERIMENTS_FOR_CURRENT_FORMULATION"
        },
        "统一结论": "局部 overlap opportunity、Oracle action value、在线可观测性和 full-request critical-path leverage 是不同层级的命题；低层证据不能直接升级为系统 GO。"
      },
      "terminology_translation": {
        "Dispatch": "路由后把token按目标expert分组；在Expert Parallel中还要把相应hidden states发送到承载expert的设备。",
        "Combine": "将各expert输出按gate weight加权汇聚成token结果；跨rank专家并行还涉及输出返回与通信，单卡本地汇聚不能当作通信测量。",
        "local tail": "单层或单 expert 的 service-time 长尾；它不自动等于请求级瓶颈。",
        "barrier amplification": "局部长尾经 layer、decode step、batch 或 iteration 同步传播出的额外等待。",
        "full-request leverage": "移除合法 runtime barrier 后，request completion 真正提前的幅度。",
        "headroom": "相对当前 baseline 的理论改进上界；Oracle headroom 不等于在线机制收益。",
        "Oracle": "使用 future-known 信息、但仍受真实 work、依赖和资源容量约束的理想上界，用于判断问题是否值得做，不代表可部署策略。",
        "identity-complete full-request DAG": "能够追踪 request、step、layer、route/expert、算子、资源、输入输出依赖及真实 barrier 身份的完整请求图。",
        "proxy action space": "在局部或代理语义下存在可行动作；不能直接外推为任务质量、在线安全或 serving 加速。",
        "formal Gate": "在模型、数据、split、指标、门槛和停止条件预先冻结后运行的可证伪验证。"
      },
      "absolute_redlines": [
        "论文不套用生产项目的已上线口径；可以说明已经实现和运行的实验动作与请求级结果，正式主机制和生产部署尚未成立。",
        "RCBA放在历史候选，保留UNVALIDATED与FORMAL_GATE_NOT_RUN；不冒充已实现方法，也不把未运行改写成全方向实证NO-GO。当前调度主线仍是测量，不能写成已验证专家感知算法。",
        "不要把 exact Oracle、single-GPU microbenchmark 或 proxy action space 说成在线 selector、controller、vLLM serving 或端到端收益。",
        "旧算子微基准和自定义数值传播实验不能推出请求延迟；新的原生vLLM单卡请求实验可以报告其自身TTFT、TPOT、ITL和吞吐，但不能外推为多卡EP、NCCL/RDMA、跨节点或生产部署结果。",
        "动态 batch 实验只支持单模型、单卡自定义 cached-decode runtime 中的执行一致性和 teacher-forced 传播结论；不要声称已证明最终 token/质量、原生 runtime 可转移性或请求延迟影响。",
        "C/D 的首次观测差异位于 MoE Output，具体 expert grouping、GEMM、sorting、combine 或其他算子原因未测量；不要把位置定位升级成机制定因。",
        "不要说简历中的混合精度方法已被证明有效；fixed RankLane 在冻结乐观域内已经 NO-GO。",
        "不得覆盖旧结果、按有利结果重选canonical，或把看过的数据称为untouched holdout。新实验可在事先说明新问题和变化依据后调整运行域；探索性调参不等于独立确认。"
      ]
    }
  ],

  "safe_answer_patterns": {
    "usage": "按当前问题检索内容，遵守 system_short_rules 中的三项核心约束。长答案和完整链路仅提供可选细节；效果题提取对应数据与范围，取舍题提取约束、替代方案和成本，交互题提取发起方、接口与结果处理方。复习提醒和事实红线用于内部核对，口述只保留当前结论必需的前提。",
    "generic_concepts": {
      "k8s_controller_informer": [
        "通用原理（不是项目实现）。",
        "标准 controller 主线：list/watch → informer cache → workqueue → reconcile。",
        "Informer 通过 List 和 Watch 维护本地缓存并通知控制器；Workqueue 保存待处理对象的 key，提供去重、限速、失败重试和退避——事件处理函数应快速入队，耗时调和由 worker 完成。",
        "Workqueue 传递的是'哪个对象需要重新调和'，不是完整对象，也不保证保留每次中间版本，相同 key 可能被合并，不能把队列当成状态来源。",
        "Reconcile 是根据对象当前状态计算下一步，不是按事件内容执行固定命令；真正的事实来自 API Server 里的最新对象和可观察的外部状态，不是事件本身。",
        "为什么 Controller 必须幂等：同一对象可能因重试、Watch 重连、进程重启或周期同步被重复处理，实现思路是先观察当前状态再执行最小必要动作，关键步骤持久化，外部请求结果不确定时先查询结果不能直接重发。",
        "resourceVersion/乐观并发控制：防止多个写者静默覆盖同一对象，旧版本写入会冲突，调用方需重新读取计算；它不能自动解决业务层冲突。",
        "Leader Election 只保证同一时刻只有一个活跃控制器，不保证外部动作只执行一次——旧 Leader 的请求可能已发出，重启/重试等外部副作用仍要靠任务状态、幂等标识和外部状态观测防重。",
        "At-least-once 与 exactly-once：Controller 调和通常是 at-least-once，跨 Kubernetes、云 API 和 SSH 很难实现 exactly-once；工程上应让普通请求可以安全重试，同时避免重启等破坏性结果重复发生。"
      ],
      "alibaba_interview_gap_qa": {
        "Controller有状态和无状态怎么区分": "判断标准不是进程内有没有缓存和队列，而是关键正确性是否依赖某个实例的本地状态。无状态 Controller 把期望、进度和可恢复事实放在 Kubernetes API 对象或外部系统，进程重启后可以重新 List/Watch 并继续 Reconcile；有状态 Controller 若依赖本地数据库、WAL、固定分片或不可迁移会话，就需要处理状态复制、恢复和实例身份。",
        "Controller有内存队列和InformerCache还算无状态吗": "算。Informer cache、workqueue、limiter 和正在执行的 goroutine 都是可重建的运行时状态，只要它们不是唯一事实源，进程重启后能够从 API 对象和外部实际状态恢复，就仍然是逻辑上的无状态 Controller。",
        "Controller应该用Deployment还是StatefulSet": "多数 Controller 使用 Deployment，因为副本可替换，关键状态应外置；需要多副本时再结合 Leader Election 或安全的并行调和。只有确实需要稳定网络身份、固定分片或本地持久化状态时才考虑 StatefulSet，而且使用 StatefulSet 也不自动说明业务逻辑有状态。",
        "单副本等于有状态吗_多副本等于无状态吗": "都不等于。单副本进程如果重启后能从 API 对象恢复，仍然可以是无状态；多副本进程如果共同依赖本地不可恢复状态，也可能是有状态。副本数描述部署可用性，有无状态描述恢复和替换边界。",
        "LeaderElection解决什么_没解决什么": "Leader Election 解决多个 Controller 副本中谁执行有副作用的调和，降低同时派单或同时修改外部系统的冲突。它不保证 exactly-once，不恢复旧 Leader 的内存，也不自动解决新旧版本兼容；外部动作仍要依靠持久化任务、幂等键和实际状态查询防重。",
        "多副本Controller怎么滚动升级": "先保证 CRD、Status 和外部协议向前向后兼容，再用 Deployment RollingUpdate 逐步替换实例；新旧版本短暂共存时，只有拿到 Leader 的实例执行有副作用的调和。新 Leader 接管后必须从 API 对象和外部实际状态恢复，不能依赖旧进程内存；远端操作结果不确定时先查询，不能直接重做。",
        "把状态持久化到APIServer准确吗": "工程口语可以这样说，但更准确的是把状态建模为 Kubernetes API 对象的 Spec/Status，通过 API Server 读写，最终由 etcd 持久化。Controller 不直接访问 etcd；API Server 负责认证鉴权、校验、准入、版本和并发控制，etcd 是控制面的持久化存储。",
        "Deployment从提交到Pod运行经过哪些组件": "客户端提交 Deployment 后，API Server 完成认证、鉴权和准入处理，将对象持久化到 etcd；Deployment Controller 创建 ReplicaSet，ReplicaSet Controller 再创建 Pod。Scheduler 为未绑定的 Pod 选择节点，通过 API Server 完成绑定，目标 kubelet 观察到分配给本节点的 Pod 后协调卷和镜像准备；所需卷挂载由 kubelet 经卷插件或 CSI 完成，再通过 CRI 请求运行时创建 Pod sandbox 和容器，网络由运行时调用 CNI 配置。Pod 启动后 kubelet 更新状态并按配置执行探针；匹配 Service 的 Pod 通常在 Ready 后成为可用后端，未就绪后端也可能已记录在 EndpointSlice 中，发布未就绪地址的 Service 另有配置。直接创建 Pod 时不经过 Deployment 和 ReplicaSet 这两层。",
        "Kubernetes集群为什么不能无限扩大": "规模上升后，Node、Pod、Lease 和 Event 数量增大，API Server 的请求和 List/Watch 扇出、etcd 的对象与写入压力、Scheduler 的调度吞吐以及各 Controller 的 Reconcile 量都会上升。5000 节点是经过验证的组合边界，不是某一个固定 if 判断；继续扩大通常需要控制面扩容、限流和参数调优，或者拆成多个集群隔离压力和故障域。",
        "有状态应用一个Pod故障会怎样": "取决于它的角色、数据是否持久化、复制协议和剩余副本是否还能形成 Quorum。从副本故障可能只降低冗余，Leader 故障通常需要重新选主；Kubernetes 能重建 Pod 和重新挂载 PVC，但数据一致性、主从关系和业务可用性由应用自身保证。",
        "有状态应用怎么跨集群迁移": "重点是先迁状态，再切流，最后下线旧集群。先在目标集群准备运行和存储环境，通过复制、备份恢复或存储迁移追平数据；切换前控制最后写入窗口并确认不会双写分叉，再切 Service/DNS/入口，验证目标集群后才停止源集群，同时保留回滚条件。",
        "节点自愈后的Pod重调度是集群内还是集群间": "当前节点自愈链路里的 Drain 和 Pod 重建发生在同一 Kubernetes 集群内，上层工作负载 Controller 重新创建 Pod，本地 Scheduler 再选择健康节点。跨集群迁移需要额外的多集群控制面先选择目标集群，不是单集群 Scheduler 自动完成。",
        "多集群调度和单集群调度怎么分层": "多集群控制面根据集群容量、地域、GPU 类型、队列、数据位置和故障域完成 Cluster Selection，再把 workload 下发到目标集群；目标集群内部 Scheduler 完成 Node Selection，kubelet/Device Plugin 再完成设备分配。Karmada 更偏多集群资源传播和调度，Volcano 更偏单集群批任务 Queue、Gang 和抢占，不能混为同一层。",
        "SLI_SLO_SLA和错误预算是什么_节点自愈如何关联": "SLI 是实际测量的服务指标，SLO 是给指标设定的目标和统计窗口，SLA 是对用户的服务约定及未达标责任。比如用成功的有效请求数除以全部有效请求数衡量可用性，或用阈值内完成的请求比例衡量延迟，都要提前明确哪些请求、超时和错误计入统计。错误预算就是 SLO 允许的失败范围：假设一个固定窗口有 100 万次有效请求，成功率目标为 99.9%，允许失败 1000 次；已失败 800 次，预算就消耗了 80%，还剩 200 次。滚动窗口的请求量会变化，需要随窗口重算，也不能把请求失败次数直接换成停机时长。还要看消耗速度，最近错误率为 1%、目标允许错误率为 0.1% 时，burn rate 就是 10 倍，可以结合长短窗口判断是否持续恶化。预算紧张时，按团队事先约定的政策控制发布风险、优先修复可靠性问题。结合 GPU 自愈，节点侧看发现延迟、修复时长和资源回池，业务侧看推理可用性与延迟，或训练恢复时间和进度损失；节点修复成功不自动代表业务恢复，Drain 和 Reboot 的代价也要算进去。上述数字是计算示例，项目效果仍要以实际观测为准。",
        "StatefulSet如何滚动升级_怎样灰度和处理失败": "StatefulSet 提供稳定的副本序号、网络身份和每副本存储关联，Pod 重建后 UID 会变化，但可以继续使用对应的 PVC。以默认 OrderedReady 和滚动更新并发为例，修改模板后从最大序号开始逐个替换，等待当前 Pod Running、Ready，配置了 minReadySeconds 时还要满足稳定时间，再继续更新前一个。三个副本 app-0、app-1、app-2 可以先设 partition=2，只让 app-2 使用新模板，验证后再依次降到 1 和 0；低于 partition 的副本即使重建也保持旧版本。OnDelete 策略则不会主动替换已有 Pod，需要删除后才按相应模板重建。有状态升级还要检查副本是否追平、剩余成员能否维持 Quorum、是否需要切换 Leader，以及新旧版本和数据格式是否兼容；Ready 只证明配置的就绪条件通过，PVC 保留也不会自动完成数据复制或迁移。StatefulSet 自身滚动更新不通过 Eviction 受 PDB 约束，因此必须自行控制更新并发和应用恢复门槛。新版本一直不 Ready 时先停止推进并排查启动、挂载和应用恢复；默认有序更新可能卡在坏 Pod 上，回退模板后还需要视情况重建已使用坏配置的 Pod。重建前确认副本和数据安全，镜像回退无法撤销已经发生的不兼容数据迁移。"
      },
      "gpu_scheduler_deep_dive": {
        "90秒首答": "我会把 GPU 调度分成调度周期和绑定周期。Pod 出队后先经过 PreFilter、Filter；有候选节点才执行 Score 选出 Node，没有可行节点则由 PostFilter 决定是否抢占或重新排队。选中 Node 后 Scheduler 先在本地 Cache AssumePod，防止后续 Pod 重复看到这份资源，再进入 Reserve 和 Permit；绑定周期继续执行 WaitOnPermit、PreBind、Bind、PostBind。绑定链路失败时要把插件资源 Unreserve，并从 Scheduler Cache ForgetPod。GPU 的特殊点是 Filter/Score 不只看卡数，还要看型号、显存、共享份额和拓扑；如果提前选择具体设备，还必须维护一份可回滚、可对账的设备分配状态。",
        "失败追问": "我会按失败位置回答：Filter 无解就记录 Unschedulable 原因并等待相关事件激活；Reserve 失败后按逆序调用全部已启用 Reserve 插件的 Unreserve；Permit 拒绝或超时、PreBind 失败、Bind 失败也都执行 Unreserve 和 ForgetPod。越靠近 Bind，越可能已经写了 Pod 标注、CR 或外部系统，所以不能只相信内存回调，还要用 Pod UID 等幂等键、状态查询和 Controller 对账处理部分成功与 Scheduler 崩溃。",
        "调度队列首答": "Scheduler 内部通常把待调度 Pod 分成三种状态：activeQ 负责立即尝试，backoffQ 负责失败后的限速重试，unschedulableQ 负责等待相关集群事件。新 Pod 进入 activeQ，失败后根据退避和事件竞态进入另外两个队列；Node、Pod、存储或资源变化只激活可能受益的 Pod。这个设计兼顾吞吐、防热循环和事件驱动重试。",
        "公平性首答": "优先级决定当前谁先，公平性决定多个租户长期各拿多少。常见做法是层级 Queue 加最低保障、最大上限、权重、借用和回收；多资源场景可以用 DRF 比较 dominant share。GPU 还要处理型号不可替代、Gang 大小和 Checkpoint 成本，所以不能只按卡数平均。",
        "抢占首答": "高优 Pod 没有可行节点时，PostFilter 可以在候选 Node 上模拟移除低优 Pod并重新执行 Filter，找到最小必要受害者集合，再综合 PDB 违反和受害者代价选择 Node。驱逐后只会写 nominatedNodeName 等待资源真正释放，抢占者仍需重新调度和绑定；硬约束不匹配不能靠抢占解决。",
        "现场编码首答": "我会先澄清 Pod 需求、Node 资源模型和硬约束，再把方案映射到 Framework：PreFilter 解析并缓存，Filter 做无副作用的可行性判断，Score 做归一化排序，需要临时占用才实现 Reserve/Unreserve，需要 Bind 前写状态才用 PreBind。编码时区分不可调度与系统错误，最后覆盖空值、边界、并发、重复回滚和阶段失败。",
        "二面连续追问路由": "跟随当前追问回答，分别选用成组准入、提交补偿、索引或设备分配等相关材料；需要解释原因时补一个具体对象或例子，不预先展开下一层问题。",
        "回答边界": "本节作为通用原理和设计题材料。个人实现从对应项目事实读取；本地存储调度讲 LocalPV 插件增量，GPU 项目讲检测、自愈与资源恢复。项目归属只在回答个人经历或举项目例子时说明。"
      },
      "k8s_service_networking": [
        "通用原理：客户端访问 Service ClusterIP，由 kube-proxy 规则或替代数据面选择 Endpoint IP:port，后续可达性依赖集群网络与路由。Endpoint 可以是 Pod，也可以是自定义 Controller 管理的外部 VM，Service 不区分业务意图上的入站或出站。",
        "普通 Service 靠 selector 匹配 Pod，由内置 EndpointSlice/Endpoints Controller 自动生成后端；selectorless Service 没有这个自动生成过程，后端地址必须由自定义 Controller 显式维护。",
        "ExternalName 类型 Service 只做 DNS CNAME，不参与转发，没有 ClusterIP、没有 Endpoints，kube-proxy 完全不介入；因此它无法根据后端健康状态及时移除流量，也无法对多个后端做负载均衡。",
        "kube-proxy 和 Cilium eBPF kube-proxy replacement 是两种可能的 Service 数据面实现，装了 Cilium 不代表一定用 eBPF 转发——要看是否开启了 kube-proxy replacement，需要检查实际集群配置才能确认用的是哪一种。"
      ],
      "tls_sni_vs_host_header": [
        "通用原理。SNI 属于 TLS 层，在 Client Hello 里携带，发生在加密通道建立之前，服务器靠它在握手阶段选择要返回哪张证书。",
        "Host Header 属于 HTTP 层，发生在加密通道建立之后的请求报文里，服务器靠它在同一 IP/端口上区分请求路由到哪个站点配置。",
        "两者信息可能指向同一个域名，但解决的是不同阶段的问题：SNI 解决'握手前该给哪张证书'，Host Header 解决'握手后这个请求该给哪个站点'。",
        "反向代理场景里，如果连接地址用的是已校验的 IP、身份信息（SNI/Host/证书校验）用域名，两者要刻意拆开配置，避免代理在请求过程中重新解析域名从而绕过前置的地址白名单校验。"
      ]
    },
    "通用话术模板": {
      "被challenge时的回答结构": "先直接回答质疑的具体点：问风险就说当前是否存在及处理条件，问效果就给数据或具体缺失项，问复杂度是否值得就说明约束、替代方案和成本。只补支持这个答案的原因。",
      "了解但没落地时": "原理题直接讲原理；问个人是否做过时直接回答实际参与情况。提出方案用“我会”表达，不额外追加自我辩护。",
      "从通用原理过渡到项目": "需要项目例子时，选一个已确认的具体动作或结果解释当前原理，讲清两者联系后停止。",
      "通用原理题默认顺序": "按问题选择内容：问是什么就解释概念，问怎么工作就解释机制，问为什么用就讲收益与取舍，问限制才展开限制；必要前提紧贴结论说明。",
      "被要求重新整理项目时": "明确要求重新介绍项目时，用一个业务例子说明原来哪里麻烦、本人改了什么和已知效果；只是某个环节没听懂，就换一种说法解释该环节。",
      "只回答核心部分时怎么避免漏掉关键点": "逐句检查是否直接回答当前问题，或支撑结论成立。保留必要前提，例如训练中断所需的状态保存和恢复条件；删除不影响当前答案的并列机制、背景和提醒。"
    }
  }
}
