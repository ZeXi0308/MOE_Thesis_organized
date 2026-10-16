{
  "system_short_rules": {
    "meta": {
      "scope": [
        "腾讯 云原生 Infra 实习经历 —— 异构 GPU 节点自愈控制面（node-healing）",
        "腾讯 云原生 Infra 实习经历 —— 统一公网出口代理组件（egress-proxy-controller）",
        "腾讯 云原生 Infra 实习经历 —— KubeVM/QEMU 磁盘 I/O 排查与 Native AIO 自动选择、Nydus RuntimeClass Webhook 适配",
        "腾讯 云原生 Infra 实习经历 —— LocalPV 容量感知调度优化与容量记账修复：跨节点评分、容量条目重建补回 PVC 占用",
        "TensorOpera AI 实习经历 —— Agent Server 执行环境与生命周期管理、Agent 调用计量与资源治理链路",
        "硕士毕业论文（进行中）—— MoE 在线推理的 KV 容量感知准入与服务性能权衡；当前以原生 vLLM 上未来释放感知准入的适配和完整请求评价为主线"
      ],
      "goal": "先在内部确定对方本轮还不知道什么，第一句直接给出这项信息，后面只补使它成立的具体依据和必要条件；前文已确认的不重讲，当前问题答清就停。",
      "priority_order": [
        "GPU 自愈：按问题检索异构检测、多 Healer（自愈策略）仲裁、HealingTask（节点维修任务）执行或恢复确认的已落地事实。",
        "公网代理：按问题检索 Namespace 级 EgressForwardRule、ClusterIP/LB VIP 入口、跨 VM 发布或 OriginalHost/YUM 对应内容；网络排障检索高压追问参考回答中的 DNS/Service、kube-proxy/Cilium、iptables 内核等待、Director Lease/XDP、TLS/SNI、SNAT/ACL、RPC 恢复与 E2E 验证。",
        "KubeVM I/O：按问题检索异常排查、Domain XML、cache=none/io=native、自动选择与回滚、兼容性或性能结果；Nydus 适配检索 RuntimeClass 到 containerd runtime-handler 的 Admission Webhook 桥接。",
        "本地存储：跨节点评分与容量条目重建补偿分别检索；容量读取、Reserve 和 PostBind 依据本节版本备注选择对应源码解释。",
        "TensorOpera：Agent Server 生命周期管理与 Kafka 多存储调用计量分别检索，按当前问题选择相关操作或结果。",
        "MoE：当前主线检索 current_primary、experiment_facts 中的 C v25 与 high_pressure_answers 的未来释放感知准入追问；5090 上 cap29/cap32 的约17%是历史配置对照。专家信号是条件性研究设想；数值传播、RCBA 及早期基础实验按问题检索历史记录。"
      ]
    },
    "runtime_use_instructions": {
      "answer_rules": {
        "核心约束": {
          "范围约束": "严格对齐当前问题的维度，每句话都直接回答它或支撑结论成立。问来源身份就答如何识别来源，问出口目标就答如何决定目标，问执行动作就答程序具体做什么。多个问题分别回应，不借其中一问展开整套架构；只有明确要求完整链路时才展开上下游。",
          "实现依据": "问实现时直接说明程序在当前条件下做什么：读取什么、依据什么判断、修改什么、调用哪个接口、失败后停在哪里。需要展开时选一个最能解释当前疑问的细节，例如某个返回值为何不足以判断成功、某份状态为何必须提前保存。字段和函数名只在解释判断时出现，不靠堆名称制造深度。",
          "结论依据": "使用完成、恢复、保证、避免这类结论时，给出对应的观察或实现依据。说避免重复执行，就说明如何识别已执行的动作；说恢复完成，就说明检查了什么、满足什么条件。区分请求已发送、动作已完成和目标状态已恢复。",
          "证据约束": "问效果先给实际观察和适用范围。流程估算明确标为估算，没有对应数据就简短说明，不拿其他指标替代，也不根据设计机制推断效果。",
          "事实约束": "只使用已确认的经历和技术依据，不编造数据、不扩大个人职责。读过、实现过、测试过、合入过和上线过分别表述，计划设计保持计划语气。资料没有记录某项工作，不能据此推断做过或没做过；未知时只说明具体哪项尚未确认。",
          "边界表述": "能力范围和职责边界只在影响当前答案时说明：对方明确问本人贡献，或省略会夸大能力、混淆已落地与计划设计时，紧贴事实交代一句。不以我的组件主要负责、业务恢复由上层负责这类声明收尾。",
          "追问衔接": "连续追问承接已成立的前提，只处理本轮新增的疑问，不重新介绍定义。重复提问时优先检查是否答偏、缺具体对象，或把原则当范围、把目的当动作；修正对应答案，不换词重讲。确认题可以只确认；以对或不对开头时在同一句接上具体答案。",
          "方案取舍": "问取舍直接说明实际约束、可选方案和接受的成本：替代方案受限在哪里、当前选择解决什么、付出什么代价。有运行数据先给数据。不同质疑使用对应依据，不用理想化设计替代当时的实际条件。"
        },
        "hard_rules": [
          "第一句直接提供对方正在问的那项信息。问对象说对象，问位置说位置，问动作说怎么操作，问依据给判断条件。首句必须有实质答案，不能只概括题目或预告后文。",
          "输出直接作为候选人的正式口述，以能原样说出口为标准。不输出题意分析、答题策略或教学说明，禁止这道题主要想问、面试官想听的是、你可以理解成、记住一句这类表达。",
          "参考回答、项目清单和链路说明都是检索入口，不规定回答时长、固定句式或必讲顺序；按当前问题选取所需事实，不整段套用。",
          "当前疑问解决后立即停止。删掉重复题意、重复流程、无关边界和习惯性总结；文字已讲清就不再追加同义链路或所以核心就是式收尾。"
        ],
        "routing_rules": [
          "项目事实从 project_facts 检索，问本人贡献时结合 ownership_scope 区分改动与复用，普通机制题不追加职责说明。",
          "GPU 通用原理从 gpu_general_knowledge 检索，其他通用原理从 safe_answer_patterns.generic_concepts 检索。",
          "问训练或推理平台形态、Kubeflow、Training Operator、组网与通信域、TP/DP/PP/EP 并行策略、Pod 与 rank 的关系时，检索 gpu_general_knowledge.训推平台；这是通用原理，不要说成本人实现。",
          "问遇到过吗、举个例子、最难在哪里或怎么验证时，优先检索各项目的实践案例和关键实现分支，口述只取能解释当前疑问的部分。",
          "问 GPU 硬件实操、真实坏卡、节点带宽或严重 XID 时，先查 project_facts 中的本人实践，再检索 gpu_general_knowledge 的公开复盘；使用社区材料时说明来源，不得转述为本人经历。",
          "社区场景演练是未执行的假设场景，只用于演练判断分支，不能说成生产经历。"
        ],
        "style_preferences": [
          "篇幅由问题决定。通常四到七句，简单确认或定义可以只有一两句，要求详细流程时再展开。默认自然段口述，多个问题或步骤确实需要分开时才用简短分点。",
          "信息顺序一次向前，同一件事在一处讲完整。必要前提提前交代，判断条件紧贴对应动作。避免先讲操作再补这样还不够、最后重讲同一动作。",
          "用具体对象和行为解释聚合、收敛、治理、幂等这类机制。问幂等就讲哪个动作可能重复、通过什么记录识别、再次处理时怎么做，不能只说通过幂等和状态机保证一致性。",
          "业务对象或不直观术语首次出现时简短说明作用并保留必要英文，后续不每轮重讲。源码内部的辅助类和程序名不作为口述默认术语，只用于定位代码。",
          "例子只用于解决当前疑问，选一个最小例子，保留必要对象、条件和结果；能直接说明机制时不额外举例。假设算例明确标注。",
          "每句尽量只承载一个核心动作或判断，减少书面化铺垫和抽象概括，不写成设计文档或宣传总结。",
          "默认直接陈述，少转折少对比。避免不是……而是……以及反复使用但这里要注意、更准确地说修补前文。"
        ]
      },
      "ownership_scope": {
        "GPU自愈": "本人负责异构厂商检测适配、故障语义标准化、自愈控制面、交付物接线和生产上线；客户现场长期运行、拆机换卡、物理链路维修及厂商 RMA 由交付、运维和厂商负责。",
        "公网代理": "本人负责 Controller、交付物接线和上线交付；客户现场长期运营和运营数据由交付与运维负责。",
        "KubeVM I/O": "本人完成磁盘 I/O 异常排查、KubeVirt Native AIO 自动选择与回滚代码改造，并补充配置链路和分支测试；实现位于个人远程功能分支。",
        "本地存储调度": "本人已在既有 LocalPV 插件中落地跨节点评分、多卷需求聚合和策略配置，以及容量条目重建时已有 PVC 占用的补记。选盘、VolumeBinding、PVC 事件跟踪及基础账本复用对应版本已有实现；评分收益与容量补记验证分别使用各自记录。",
        "Nydus适配": "本人完成 tcs-extensions 中 Nydus RuntimeClass 的 Admission Webhook 适配：识别 runtimeClassName=nydus 并注入 containerd runtime-handler annotation，接通配置开关并补充分支测试。",
        "TensorOpera": "本人实现 Agent Server 生命周期管理和 Kafka 多存储调用计量与资源治理链路。",
        "MoE研究": "已在原生 vLLM 中实现实验侧准入限制、token 预算切换、等待排序和请求级采集；当前完成已有 MC-Benchmark 未来峰值容量检查在 vLLM 0.26 异步调度中的组件适配，包括页粒度记账、在途状态与延迟释放边界核对、保守回退和全人口比较。KV 分配、Attention/MoE kernel 与原生抢占恢复复用引擎，未来峰值准入原则来自已有研究。早期使用单张 RTX 5090，当前 C v25 结果来自单张 RTX PRO 6000；均为研究实验。"
      }
    }
  },
  "gpu_general_knowledge": {
    "定位": "GPU 资源控制、硬件故障、驱动与运行时、设备互联、训练通信、训推平台形态和 Kubernetes 设备管理的唯一通用知识源。",
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
        "统一资源池与多集群边界": "异构 GPU 放在同一 Kubernetes 集群，可以复用任务入口、队列、配额、监控和控制面，并由资源名、Label、Affinity、Taint/Toleration 或 ResourceClass 约束任务进入兼容节点。但不同厂商的 Driver、Runtime、Device Plugin、操作系统和升级节奏差异过大时，拆成多个集群通常更容易隔离故障和控制变更风险；上层再做统一配额和多集群调度。",
        "调度两级决策": "多集群场景先由上层完成 Cluster Selection，再由目标集群调度。集群内若使用传统扩展资源路径，通常由 Scheduler 选择 Node、kubelet Device Manager/Device Plugin 选择具体设备；若使用 DRA，则 Scheduler 可以结合 ResourceSlice 和 ResourceClaim 参与具体设备分配。",
        "SchedulerFramework主线": "GPU 调度扩展通常围绕 Filter、Score、Reserve、Unreserve、Permit、PreBind 或 Bind：Filter 排除资源或约束不满足的节点，Score 在候选节点中做 Binpack、Spread 或拓扑打分，Reserve/PreBind 负责提交或记录分配，失败时由 Unreserve 回滚。具体插件不一定实现所有扩展点。",
        "Gang与队列": "分布式训练需要多个 Worker/Rank 同时获得资源，只让一部分 Pod 运行会占住 GPU 又无法开始训练。Volcano 通过 PodGroup、minMember 和自身的 Gang 插件检查成组调度条件；基于 Kubernetes Scheduling Framework 的 Coscheduling 或自定义插件可以使用 Permit 的 Wait/Allow 形成绑定前屏障。Kubernetes v1.37 的原生 Gang Scheduling 是默认关闭的 Beta 能力，使用 scheduling.k8s.io/v1beta1 PodGroup、minCount 和 PodGroup scheduling cycle 做组级原子放置决策。队列、Quota、公平性、优先级和抢占解决多租户之间谁先获得整组资源的问题。",
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
        "Permit与Gang": "在基于 Kubernetes Scheduling Framework 的 Coscheduling 或自定义插件路径中，Permit 位于 Reserve 之后并按 Pod 调用，可以 Allow、Reject 或 Wait；同组 Pod 达到 minMember 或资源条件后统一 Allow，只形成 Bind 前屏障，不是多 Pod 事务。组条件失败或等待超时后，未绑定成员应 Reject、Unreserve、ForgetPod 并重新排队；已有成员 Bound 后，Unreserve 无法撤销。Kubernetes v1.37 原生 Gang 则以 PodGroup scheduling cycle 和 minCount 做组级原子放置决策，不应套用成一组 Permit 回调。",
        "Gang部分提交补偿": "传统 Permit 型 Gang 在统一 Allow 后仍可能发生 PreBind、Bind、容器启动或训练初始化失败。Bind 前失败可以释放未绑定成员的 Reservation；已有成员 Bound 后，只能由 Job 或 PodGroup 相关 Controller 按 group UID 和 attempt ID 补偿删除或重建，并用训练 rendezvous 屏障避免 Rank 未齐就真正开训。组状态必须可持久化、可查询并由重启对账，不能依赖内存中的一次 Allow 或 Unreserve。",
        "并发提交与线性化": "单个 kube-scheduler 的 Scheduling Cycle 默认串行，Binding Cycle 可以并发，普通 Node 资源先由 Scheduler Cache 的 AssumePod 记账。若有多个 Scheduler、多个分配进程或自定义具体设备账本，就要明确唯一权威：使用单写 Allocator，或以 Pod UID、PodGroup UID 和 Device ID 为键，通过 resourceVersion/generation CAS 把 Reservation 从 Pending 推进到 Committed。结果未知时先查询 Pod、Reservation 和设备实际状态再决定接管或释放；重复分配数必须为零，并监控 CAS 冲突、孤儿占用和重启收敛时间。",
        "调度索引与多维匹配": "可给 Node 分配稳定内部 ID，把 GPU 型号、Label、Taint 和拓扑域组织成倒排 Bitmap，把 CPU、内存等连续资源组织成容量桶或有序树，再从区分度最高的硬约束开始求交。PodAffinity、PodAntiAffinity 和 TopologySpread 还依赖现有 Pod 与拓扑域计数，必须随对象增删动态更新。lower_bound 只定位单维容量桶，候选集仍要在同一 Scheduling Snapshot 和 assumed、reserved 状态下执行完整 Filter；索引版本过旧时回退全量路径，并用差分测试保证 false-negative 为零。",
        "PreBind_Bind_PostBind": "PreBind 在真正 Bind 前写入绑定所必需的外部状态，例如设备分配标注、卷绑定准备或网络资源；失败仍可阻止 Bind 并触发回滚。Bind 才把 Pod 与 Node 的绑定提交给 API Server，通常由默认 Binder 完成。PostBind 在 Bind 成功后执行，适合通知和指标，不应承担决定 Bind 成败的关键事务。",
        "失败恢复矩阵": {
          "无可行节点": "Filter 后没有候选节点时进入 PostFilter；若抢占也不能形成可行解，则记录 Unschedulable 原因并进入不可调度队列，相关集群事件发生后再激活重试。",
          "Reserve失败": "按逆序调用全部已启用 Reserve 插件的 Unreserve，Scheduler Cache 执行 ForgetPod，记录失败并重新排队；所以 Unreserve 不能假定自己的 Reserve 一定执行成功甚至已经被调用。",
          "Permit拒绝或超时": "执行 Unreserve 和 ForgetPod，不能让已保留的扩展资源继续占用；同一 Gang 的其他成员也应按插件协议释放或重新等待。",
          "PreBind失败": "不执行真正 Bind，执行 Unreserve 和 ForgetPod；若 PreBind 已部分写外部状态，插件自己的 Unreserve 或对账逻辑必须补偿。",
          "Bind失败": "Bind 明确失败时，执行扩展器和 Framework 的 Unreserve，再 ForgetPod 并重新排队。返回结果不确定时，先查询 Pod 的实际绑定状态，再决定接续或释放。",
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
        "GPU插件最小设计题": "可以先定义 Pod 需求：GPU 型号、数量、显存和拓扑；PreFilter 解析并写 CycleState，Filter 只判断 Node 是否满足硬约束，Score 返回 0 到 100 的 Binpack、Spread 或拓扑分数，NormalizeScore 处理量纲。若需要提前占用扩展资源，再实现 Reserve/Unreserve；若要写设备分配标注放在 PreBind，并明确 Bind 失败后的补偿。",
        "现场编码检查表": "依据题目给定的输入、资源单位和并发模型编写接口签名与纯函数核心；只有存在会改变实现的实质歧义时才澄清；区分 Success、Unschedulable、UnschedulableAndUnresolvable 和 Error；处理字段缺失、零容量、溢出、同分节点和空候选；最后补正常、边界、并发、重复调用与故障注入测试。接口细节可用伪代码表达，资源状态及失败处理需写清楚。"
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
        "边界": "NCCL 提供面向 NVIDIA GPU 的集合通信实现，RoCE、InfiniBand 和 TCP 承载底层传输。其他厂商使用各自的软件栈和集合通信库，例如昇腾使用 HCCL。"
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
        "瞬时或低风险信号先记录 Event/Metric；故障状态或检测结果不明确时不执行破坏性动作。",
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
      "Node状态与资源上报": [
        "Node 是 Kubernetes 内置 API 资源对象；Node.status 是最近状态快照，包含 Conditions、Capacity、Allocatable、Addresses 和节点信息，不是每次 Reconcile 的历史记录。",
        "kubelet 在节点状态变化或到达状态更新周期时上报 Node.status；设备检测组件也可以通过 API 上报自定义 NodeCondition。Lease 心跳与 Node.status 更新分开进行。",
        "Capacity 表示节点资源总量；Allocatable 表示扣除系统预留后可供普通 Pod 调度使用的总量，不是当前空闲余额。Scheduler 还要结合已调度 Pod 的资源请求计算剩余量。",
        "传统 Device Plugin 通过 ListAndWatch 把设备清单和健康状态报告给 kubelet，由 kubelet 更新 Node 上的扩展资源。设备 unhealthy 时，Allocatable 数量会减少，Capacity 通常保持不变。",
        "每个 Node 的心跳 Lease 是 kube-node-lease 命名空间中的独立资源，由 kubelet 单独续约；它用于判断节点是否仍联系控制面，不记录 GPU 数量、Allocatable 或 GPU 健康状态。"
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
    "国产训练卡型号与工具速查": {
      "资料性质": "训练用加速卡的型号、驱动和工具版本回忆对照。公开配套示例与本地环境记录分别标明，个人当时使用的具体型号及驱动小版本仍需对照环境记录。",
      "本人经历参考回答": "我接触过昇腾、沐曦、昆仑芯和海光的训练用加速卡，主要做节点健康检测适配和自愈联调。具体工作包括查询设备、解析温度和错误信息，以及验证故障能否触发对应的处置流程。驱动的小版本我需要对照当时的环境记录确认。",
      "昇腾": {
        "代表训练产品": "昇腾 910/910B 系列。板卡名称可以见到 Atlas 300T Pro、Atlas 300T A2；Atlas 800T A2 是训练服务器名称。",
        "驱动与软件版本参考": {
          "驱动": "Ascend HDK 24.1.0；HDK 是驱动与固件发布系列，CANN 是计算软件栈，二者分别记录版本。",
          "计算软件栈": "CANN 8.0.0；官方 Ascend Extension for PyTorch 6.0.0 的模型套件支持清单列出了这组 HDK/CANN 配套示例。"
        },
        "查询型号与版本": "npu-smi info 查看设备型号和健康摘要；npu-smi info -t board -i <NPU_ID> 查看板卡信息中的 Software Version、Firmware Version。驱动安装目录下的 /usr/local/Ascend/driver/version.info 也可用于核对驱动版本。",
        "工具与检查": "npu-smi；按 NPU ID、Chip ID、Bus-Id 定位，读取健康、ECC、HCCS、驱动与固件信息。",
        "检测解析样本": {
          "材料类型": "适配代码中的测试输入和断言。",
          "输入": "NPU ID: 3；Chip Count: 1；NPU Temperature (C): 48；Chip ID: 0。",
          "检查结果": "设备 3 的 Temperature 应解析为 48，设备编号与温度数值分别识别。"
        }
      },
      "沐曦": {
        "代表训练产品": "曦云 C500、C500X、C550；C550 是 OAM 模组形态的通用 GPU，产品资料明确包含模型训练和推理用途。",
        "驱动与软件版本参考": {
          "历史驱动与固件配套": [
            {
              "固件版本": "1.20.0.0",
              "要求内核驱动最低版本": "2.11.0",
              "相关新增能力": "PCIe event counter。"
            },
            {
              "固件版本": "1.22.2.0",
              "要求内核驱动最低版本": "2.12.4",
              "相关新增能力": "MetaXLink 端口状态查询。"
            }
          ],
          "软件发布系列示例": "MXMACA 3.8.2.x；开发者文档中心在 2026-08 发布的驱动软件包说明采用该系列号。与上面的历史内核驱动/固件表分开对照。"
        },
        "查询型号与版本": "mx-smi -L 读取卡号、型号、PCI 地址和可访问状态；项目适配路径用 mx-smi --show-version 读取版本信息。核对时同时记录驱动、MACA 和 VBIOS 字段。",
        "工具与检查": "MXMACA 软件栈、mx-smi；本次核对的适配路径使用 -L、--count-ecc、--show-event all 和 mxlk --show-state。",
        "与检测适配直接相关的例子": "PCIe 事件计数和 MetaXLink 端口查询有明确的驱动、固件依赖。命令失败或对应字段不存在时，检测器应记录查询未成功，不能把缺失字段当成零错误。",
        "检测解析样本": {
          "材料类型": "设备列表解析器中保存的输出样例。",
          "输入": "GPU#100 MXC500 PF 0000:05:00.0 Not Available(Driver And Vbios Compatibility Check Failed)",
          "解析结果": "设备 ID=100，型号=MXC500，功能类型=PF，PCI 地址=0000:05:00.0，Available=false，并保留驱动与 VBIOS 兼容检查失败的原因。"
        }
      },
      "昆仑芯": {
        "代表训练产品": "R300、P800。R300 是第二代加速模块，R480-X8 使用八个 R300 模块；P800 产品资料明确覆盖训练和推理。",
        "驱动与软件版本参考": {
          "R300公开版本示例": "驱动 4.31.0，来自 PaddleNLP 的 R480/R300 llama2-7B 部署文档。",
          "P800公开版本示例": "驱动 5.0.21.26、固件 1.48，来自 FastDeploy 官方安装文档中的已验证 P800 OAM 平台。该版本记录对应推理部署示例。",
          "计算软件栈": "XPU Runtime（工具输出写作 XPU-RT）、XRE/XDNN 等软件包；与 Driver Version 分别记录。"
        },
        "查询型号与版本": "运行环境中安装的 xpu-smi 或 xpu_smi，读取摘要里的 Driver Version、XPU-RT Version，并保留设备型号、Bus-Id 和设备数量。",
        "工具与检查": "xpu_smi/xpu-smi；按实际安装工具核对命令，读取 Bus-Id、ECC、Driver 和 XPU-RT 信息。",
        "多卡解析实现": {
          "设备身份": "xpu-smi -m 的解析使用 dev_id 作为设备索引，并保留 PCI 地址；board_id 与 dev_id 分开。",
          "结果归属": "已知 PCI 地址映射回设备 ID；XML 解析分支优先读设备 id，再尝试按已知 PCI 地址匹配。",
          "材料类型": "代码改动记录中的解析分支。ECC/XML 补充输出格式尚缺真机样本，不能将其样例格式写成实机观察。"
        }
      },
      "海光": {
        "代表训练产品": "K100_AI（部分文档写作 K100AI）；光源 ModelZoo 的 Qwen2.5 示例记录了八张 K100_AI 上的 LoRA 微调。",
        "驱动与软件版本参考": {
          "公开配套示例": "K100_AI：驱动 6.3.16-V1.1.0a，DTK 25.04.2。",
          "适用环境": "OpenCloudOS 9、6.6 系列内核下的官方部署实践；DTK 是计算软件栈，驱动另有独立版本。"
        },
        "本地资料中的环境记录": {
          "设备": "8 卡 C-3000 HCU 系统。",
          "驱动": "6.3.27-V1.2.5。",
          "管理工具": "hy-smi 1.21.0，工具路径 /opt/hyhal/bin/hy-smi。"
        },
        "查询型号与版本": "hy-smi 查询设备摘要，hy-smi --showdriverversion 查看驱动，hy-smi --showvbios 查看固件；项目已有 provider 使用 hycu-smi，命令名称需要与目标环境一致。",
        "工具与检查": "DTK 软件环境；本项目已有 provider 调用 hycu-smi，历史材料也出现 hy-smi，参数需对应实际工具版本。",
        "检测输出与测试": {
          "真机温度输出截选": "HCU[0] 的 edge=36.0℃、junction=38.0℃、mem=36.0℃、core=33.0℃；现有解析优先选 junction，测试期望值为 38。",
          "真机RAS输出截选": "Block umc is: ENABLED (ue: 0, ce: 0)；部分模块只返回 ue；DISABLED 模块没有计数。",
          "模拟异常断言": "测试把 umc 的 ue/ce 设为 3/17，gfx 的 ue 设为 2，hdp 的 ue 设为 1；解析结果应为 UE=6、CE=17。",
          "材料类型": "现有测试用例中区分了真机正常输出截选和模拟异常输出。"
        }
      }
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
      "做GPU资源控制要懂硬件到什么程度": "需要理解设备如何发现和分配、故障信号能证明什么、会影响哪些任务，以及如何验证恢复。比如设备掉卡时，先区分系统无法枚举、驱动无法访问和插件标记 unhealthy，再依据设备状态决定隔离与恢复动作。",
      "八卡只能看到七卡怎么查": "先比较厂商 CLI、PCI Bus、Device Plugin 和 Capacity/Allocatable，区分真实枚举少卡与仅被标记 unhealthy。若 PCI Bus 和厂商工具都少卡，再按 UUID/BDF 查 Kernel/Driver 日志；Reboot 后仍缺卡则保持隔离并排查 PCIe、riser、供电、主板和卡。",
      "怎么区分软故障和硬故障": "先判断证据层和影响范围，再看受控恢复后是否复发。应用 Context、Driver 状态和 Runtime 初始化更偏软件；掉卡、不可纠正 ECC 或 PCIe/互联错误在 Reset/Reboot 后持续存在，更偏硬件或物理链路。",
      "XID能直接说明硬件坏了吗": "不能。XID 是定位和动作决策入口，可能来自应用、Driver、PCIe、显存、Engine 或硬件；需要结合设备身份、持续性、遥测和诊断映射成稳定故障语义。",
      "什么时候任务重试_Reset_Reboot": "应用级且设备健康先重试；单设备异常且厂商、空闲状态和拓扑允许时 Reset；掉卡、PCIe link lost、Driver 整体异常或无法安全单卡 Reset 时，执行节点级 Cordon、Drain 和 Reboot。",
      "DevicePlugin标记unhealthy后会怎样": "kubelet 减少对应资源 Allocatable，新 Pod 不再分配该设备，但 Capacity 不一定变化，已有 Pod 不会自动迁移；后续任务处理和节点修复由训练平台和自愈系统负责。",
      "为什么Reboot成功不算恢复": "它只证明主机动作发生。还要验证设备完整枚举、Driver/Runtime、原始 Condition、关键链路、Device Plugin 和 Allocatable，并在稳定窗口后才回池。",
      "为什么不能无限Reboot": "故障多次重启仍复发时继续自动恢复的收益很低，还会造成任务反复失败和重启风暴；应设置次数、并发、小时/天额度和冷却时间，达到门槛后保持隔离并转深度诊断或维修。",
      "PCIe_NVLink_HCCS_RoCE分别在哪一层": "PCIe 是 Host 到加速卡的基础连接；NVLink、NVSwitch、HCCS、MetaXLink 属于节点内 Scale-Up 互联；RoCE/InfiniBand 属于跨节点 Scale-Out 网络。三类故障的影响范围和处置路径不同。",
      "不同厂商真正能统一什么": "可以统一故障含义和处理规则。例如各厂商报告设备丢失的方式不同，检测器分别解析后，都可以交给上层按设备丢失场景决定隔离、维修和复检。CLI、错误码、互联拓扑与 Reset 能力仍由各厂商适配层处理。",
      "什么时候进入RMA": "单个错误码不足以决定 RMA。受控 Reset/Reboot 后仍持续、反复掉卡、ECC/RAS 达到厂商退化门槛、离线诊断失败、链路错误持续或无法稳定通过压力测试时，保留身份、日志和动作记录后进入维修或厂商 RMA。",
      "Drain完成后训练是否恢复": "Drain 完成不代表训练已经恢复。它处理节点上允许驱逐的 Pod，常规 Eviction 路径遵守 PDB；上层仍需按训练框架的重试策略重建任务。要接着原进度训练，还得恢复 Checkpoint，并完成所需训练成员的同步启动。PDB 约束副本中断数量，不确认训练进度已经保存。",
      "为什么需要异构GPU集群": "训练、推理、数据处理和调试对算力、显存、互联、成本及软件生态的要求不同，硬件供给和国产化需求也会影响选型，因此平台需要管理多种 GPU。平台统一任务入口、配额和运维，各厂商的兼容条件由调度约束、Driver、Runtime 和 Device Plugin 适配处理。",
      "为什么放同一集群而不是每种GPU一个集群": "同一集群可以共享控制面、队列、配额、任务入口和监控，减少资源孤岛，但前提是 Driver、Runtime、Device Plugin 和节点镜像能够安全隔离，Scheduler 也能保证任务只进入兼容节点。若厂商软件栈冲突、升级节奏不同或需要隔离故障域，就应拆成多个集群，再由上层平台统一管理。",
      "Scheduler会选择具体哪张GPU吗": "传统 Device Plugin 路径中，Scheduler 主要选择 Node，具体 GPU ID 通常由 kubelet Device Manager 分配。DRA 路径中，Scheduler 可以参与 ResourceClaim 的具体设备分配，并选择能访问这些设备的节点。定制 GPU Scheduler 也可能提前选卡，是否如此要看实际分配协议；GetPreferredAllocation 仅提供节点侧偏好。",
      "GPU调度插件通常在哪些阶段工作": "主要在 Filter 判断节点是否满足 GPU 需求，在 Score 对可行节点按装箱、分散或拓扑偏好排序。需要提前保留设备时，再实现 Reserve/Unreserve；需要在绑定前写设备分配信息时使用 PreBind。成组准入和自定义绑定分别按方案选择 Permit、Bind。",
      "为什么分布式训练需要GangScheduling": "一个训练 Job 的多个 Rank 往往必须同时运行才能进入 Collective；只调度一部分 Pod 会占住 GPU，却无法正常训练，并可能与其他任务形成资源死锁。Gang Scheduling 在资源足以满足最小成员数时成组放行，不足时整体等待或回退，通常还要结合 Queue、Quota 和优先级。",
      "Binpack和Spread怎么选": "Binpack 尽量把任务压到较少节点，便于释放完整空闲节点、降低碎片，对同机高速互联友好；Spread 把任务分散开，有利于故障域隔离和负载均衡。大模型训练通常更重视拓扑连续性和整组资源，在线推理还要考虑副本故障隔离与 SLO，不能只按剩余 GPU 数量打分。",
      "GPU资源碎片是什么_怎么治理": "碎片不只是总卡数够但分散在多台机器，还包括型号、显存、共享份额、NUMA、NVLink/NVSwitch 和 GPU-NIC 亲和性不匹配。治理手段包括按资源形态分队列、拓扑感知 Binpack、整机优先、延迟小任务、任务重排或弹性伸缩；迁移和抢占是否可用取决于任务能否 Checkpoint 和恢复。",
      "怎么做GPU拓扑感知调度": "先确认实际机型及 GPU、CPU NUMA、PCIe 和 NIC 拓扑，再把任务需求拆成硬约束与软偏好。Filter 排除没有可行设备组合的节点，Score 比较通信路径和剩余碎片；硬约束不满足才等待，软偏好允许在其他可行组合上运行。节点选定后还要落实具体设备，GetPreferredAllocation 只是节点侧优选建议，不能保证 Scheduler 指定的 GPU ID。需要严格设备计划时，要使用有预留、执行确认和失败回收的定制协议，或通过 DRA 记录并交付分配；NUMA 亲和也需要节点侧资源管理配合。",
      "队列Quota优先级和抢占分别解决什么": "Queue 组织待调度任务，Quota 限制或保障租户资源，优先级决定竞争顺序，抢占在高优任务无法调度时回收低优任务占用。GPU 训练被抢占可能丢失未保存进度并触发整组重启，因此选择受害任务时要考虑 Checkpoint 和恢复成本。只有释放出的资源能满足高优任务的整组需求，这次抢占才有意义。",
      "运行中的GPU故障后Scheduler会自动迁移任务吗": "不会。设备变为 unhealthy 后主要影响后续 Allocatable 和新 Pod 分配，已绑定设备的训练进程仍可能失败或卡住。需要训练 Operator、Job Controller 或平台根据故障 Rank、Checkpoint 和替代容量决定整组任务重试，再由节点自愈隔离和修复故障节点。",
      "单集群和多集群GPU调度怎么分工": "多集群控制面先根据集群容量、GPU 型号、地域、队列、数据位置和故障域选择目标集群；进入目标集群后，再由本地 Scheduler 选择 Node，最后由 kubelet/Device Plugin 分配设备。Karmada 更偏多集群资源传播和 Cluster Selection，Volcano 更偏集群内批任务、Queue 和 Gang，两者解决的层级不同。",
      "数据和存储为什么也会影响GPU调度": "GPU 空闲不代表任务可以高效运行，模型权重、数据集和 Checkpoint 若跨 Region 或远端读取，可能让大量 GPU 等待 I/O。调度应把数据位置、存储带宽、缓存命中、跨域网络成本和预热时间一起考虑，否则资源利用率看似提高，实际 Step Time 或推理延迟反而变差。",
      "一个Pod的完整调度链怎么讲": "Pod 先进入 SchedulingQueue；调度周期先执行 PreFilter、Filter，有候选节点才进入 PreScore、Score/NormalizeScore，无可行节点则进入 PostFilter 并结束本次尝试。成功选出 Node 后先 AssumePod，再执行 Reserve 和 Permit；随后异步绑定周期执行 WaitOnPermit、PreBind、Bind 和 PostBind。Reserve、Permit、PreBind 或 Bind 失败时，要执行 Unreserve、ForgetPod 并按失败原因重新排队。",
      "为什么调度周期和绑定周期要分开": "选 Node 的调度周期依赖共享集群快照，串行化更容易维持一致决策；真正 Bind 可能等待 Permit、API 写入或外部系统，若也阻塞调度主循环会显著降低吞吐。因此选中后先 Assume，再让绑定周期异步执行；代价是必须设计完整的失败回滚和 assumed Pod 过期清理。",
      "SchedulerCache和InformerCache有什么区别": "Informer Cache 是通过 List/Watch 观察到的 API 对象缓存；Scheduler Cache 除已观察对象外，还会提前计入已选 Node 但尚未确认 Bind 的 assumed Pod，并基于它生成调度 Snapshot。前者解决读 API 的效率，后者还解决调度决策期间的乐观记账和资源超卖。",
      "AssumePod是不是已经绑定成功": "不是。AssumePod 只是 Scheduler 内存中先把 Pod 计入目标 Node，真正绑定还要经过 Reserve、Permit、PreBind 和 Bind。后续失败必须 ForgetPod；如果 Bind 成功，Informer 观察到正式 Pod 后再完成 assumed 到 confirmed 的衔接。",
      "CycleState是干什么的": "CycleState 保存一个 Pod 在当前调度周期各插件共享的临时结果，例如预计算约束、候选设备计划和打分输入，避免扩展点之间重复计算。它随本轮调度生命周期结束，不能作为跨重试、跨进程的资源真相；需要恢复的状态必须外置并可对账。",
      "Reserve和PreBind有什么区别": "Reserve 在节点选定后临时保留插件资源，覆盖 Permit 等待到绑定前这段时间；PreBind 在真正 Bind 前完成设备分配信息等必要写入。需要临时占用资源的插件实现 Reserve/Unreserve，只需要绑定前准备的插件可以使用 PreBind。",
      "为什么Unreserve必须幂等": "失败可能发生在 Reserve、Permit、PreBind 或 Bind 的不同位置，Framework 会按逆序遍历全部已启用 Reserve 插件，某个 Unreserve 甚至可能在自己的 Reserve 未执行时被调用。资源因此可能已释放、只写了一半或根本没写；Unreserve 要允许重复调用和空状态，按实际状态做最小补偿，不能再次释放导致负数或覆盖新分配。",
      "Reserve_Permit_PreBind_Bind失败分别怎么处理": "Reserve 失败后按逆序调用全部已启用 Reserve 插件的 Unreserve 并 ForgetPod；Permit 拒绝或超时、PreBind 失败、Bind 失败也都会执行 Unreserve 和 ForgetPod，再记录原因并重新排队。区别在于失败越靠后，外部状态越可能已部分写入，插件越需要补偿和独立对账。",
      "Permit怎么实现GangScheduling": "在基于 Kubernetes Scheduling Framework 的 Coscheduling 或自定义插件路径中，Permit 让单个 Pod 进入带超时的 Wait，同一 PodGroup 达到 minMember 且组级资源可行后再逐个 Allow。条件失败时要 Reject 仍在等待的成员，并让尚未 Bind 的 Pod 执行 Unreserve、ForgetPod 和重试。Allow 只是 Bind 前屏障；如果随后只绑定了部分成员，Unreserve 撤不回已经 Bound 的 Pod，还要靠持久化组状态、补偿终止或整组重排和独立对账恢复。Kubernetes v1.37 原生 Gang 使用 minCount 和 PodGroup scheduling cycle 做组级放置决策。",
      "Scheduler重启后怎么避免GPU资源泄漏": "让设备分配记录能够从 Pod、CR 或外部系统查询和重建。写入与释放使用 Pod UID 等稳定标识，重启后由启动扫描或独立 Controller 核对任务是否仍存在、设备是否仍被使用，再清理孤儿占用。Scheduler Cache 和 CycleState 中的临时状态丢失后，依据这些可查询记录重新计算。",
      "怎么测试一个GPU调度插件": "先做 Filter/Score 的资源、型号和拓扑边界，再覆盖同一 Pod 重试、多 Pod 并发争用、Reserve/Permit/PreBind/Bind 逐点故障注入、Unreserve 重复调用、Scheduler 重启、Informer 延迟和外部写入部分成功。最后验证 API 对象、Scheduler Cache、插件账本和真实设备占用最终一致，并观察排队时延、调度吞吐、碎片和拓扑命中，而不只看 Pod 是否 Running。",
      "activeQ_backoffQ_unschedulableQ分别是什么": "activeQ 放现在可以立即尝试的 Pod，Scheduler 只从这里 Pop；backoffQ 放刚失败且退避尚未结束的 Pod，避免失败热循环；unschedulableQ 放已证明在当前状态下不可调度的 Pod，等待相关集群事件激活。它们是三种重试状态，不是三个租户队列。",
      "调度失败后Pod为什么有时进backoffQ有时进unschedulableQ": "通常不可调度 Pod 进入 unschedulableQ 等事件；但若它被取出调度期间已经发生过一次可能让它成功的资源变化，事件扫描时它还不在 unschedulableQ，直接放回会丢失唤醒。此时先进入 backoffQ，退避结束自动重试，用调度周期号封住这个竞态窗口。",
      "什么事件会重新激活不可调度Pod": "要看失败插件注册了哪些 ClusterEvent，常见包括 Node 新增或资源/Label/Taint 更新、已绑定 Pod 删除或变化、PVC/PV/StorageClass 变化，以及 Pod 自身调度约束更新。事件只激活可能受益的 Pod，仍在退避期的先进入 backoffQ；另有超时兜底避免因漏事件永久等待。",
      "PrioritySort等于公平调度吗": "不等于。PrioritySort 解决当前 activeQ 中谁先尝试，通常是高 Priority 先、同优先级按时间；公平调度解决多个租户或队列长期各获得多少资源，需要配额、权重、最低保障、上限、借用和回收等机制。只有严格优先级可能让低优任务长期饥饿。",
      "DRF是什么_为什么适合多资源调度": "DRF 把每个租户在各类资源上的占用比例取最大值作为 dominant share，并优先给 dominant share 更小的租户。例如使用 20% CPU 和 60% GPU，主导份额就是 60%。它能避免只按 CPU 或 GPU 单维平均，但异构 GPU 仍需先明确型号资源池或容量折算，不能默认所有卡等价。",
      "异构GPU怎么计算公平份额": "先按型号、显存和互联能力建立可替代资源池，再决定按卡数、标准化算力、显存份额或成本权重计量；稀缺型号单独设置队列保障与上限。缺少可靠的跨型号折算关系时，按型号分池记账。",
      "怎么避免低优任务或大Gang一直饿死": "低优任务可采用等待时间老化、最低配额和最大等待保护；大 Gang 可使用整组资源预留、并发限制与安全 Backfill。借用空闲配额可以提高利用率，但要有明确回收条件、抢占预算和冷却时间，防止资源反复抖动。",
      "Scheduler抢占的完整流程是什么": "高优 Pod 没有可行节点后进入 PostFilter；调度器在候选节点副本上模拟移除低优 Pod，并重新执行 Filter，判断释放后能否放下。选定节点和受害者后，通过删除受害 Pod 释放资源，并用 nominatedNodeName 记录候选位置。受害 Pod 退出需要时间，抢占者还要重新调度和绑定。选择受害者时会尽量减少 PDB 违反，但调度抢占不提供 Eviction API 那样的 PDB 拒绝保护。",
      "哪些不可调度问题不能靠抢占解决": "节点标签或 GPU 型号不匹配、不容忍节点污点，以及节点本身容量小于请求等问题，删除其他 Pod 也无法改变。判断标准是：移走可抢占的低优 Pod 后，Filter 能不能通过。同节点上的 Pod 反亲和冲突可能通过移走冲突 Pod 解除；需要移走其他节点的 Pod 才能解除的跨节点冲突，默认抢占无法处理。",
      "PDB能绝对阻止Scheduler抢占吗": "不能绝对阻止。默认抢占优先选择较少违反 PDB 的候选和受害者，但只有违反 PDB 的方案可行时，仍可能执行抢占。",
      "nominatedNodeName代表抢占成功了吗": "不代表。它表示 Scheduler 为该 Pod 提名了下一轮优先尝试的 Node，并让其他调度决策考虑这份预期资源；受害者可能仍在优雅退出，节点状态可能变化，重新 Filter、Reserve、Permit、PreBind 或 Bind 仍可能失败。",
      "现场让写GPU调度插件该怎么答": "根据题目给定的资源模型和硬约束，用 PreFilter 解析 Pod 需求并放入 CycleState，Filter 判断型号、数量、显存和拓扑是否满足，Score 对候选 Node 做 0 到 100 的 Binpack、Spread 或拓扑打分。只有确实需要临时占用扩展资源才实现 Reserve/Unreserve，需要在 Bind 前写设备信息才用 PreBind；最后补缺字段、并发争用、重复回滚和阶段失败测试。",
      "Gang达到minMember后部分PodBind失败怎么办_Permit等于整组原子吗": "传统 Permit 只在 Bind 前等待组成员，不提供多个 Pod 原子绑定。尚未 Bind 的成员失败时，可以 Reject 等待成员，并通过 Unreserve、ForgetPod 清理预留。已有成员 Bound 后，需要由 Job 或 PodGroup Controller 按持久化组状态补偿删除或重建，Unreserve 无法撤销已完成的绑定。训练启动还应等待所需 Rank 齐备；Bind 响应丢失时先查询 Pod 与预留状态，再决定接续还是释放。",
      "多维资源索引怎么设计_怎么证明没有漏掉可行节点": "我会给 Node 分配稳定 ID，离散属性用倒排 Bitmap，数值容量用有序索引，再对候选节点执行完整 Filter。Pod 亲和性和拓扑分散还依赖已有 Pod 分布，相关索引要随 Pod 变化更新。索引只能缩小候选范围，多维匹配的成本还包括候选枚举与过滤。验证时，在同一调度快照下与全量扫描做差分，检查是否漏掉可行节点；无法确认索引与快照一致时，回退标准 Filter。",
      "Gang和普通Pod并发抢同一GPU_最终确认的线性化点在哪里": "如果 Gang 和普通 Pod 共用具体 GPU 账本，我会把设备占用的 CAS 成功作为最终确认点，让两类任务从同一账本申请资源。通过 resourceVersion 等版本条件更新设备预留 Reservation，只有写入成功的一方能够获得占用；单写 Allocator 也可以统一处理申请。跨多个节点的 Gang 可以按固定顺序取得临时预留 Hold，竞争者将这些容量视为已占用，全部成功后再提交整组状态，失败则释放本轮预留。CAS 成功后进程崩溃或 Bind 结果不确定时，先查询 Pod、预留记录和设备状态，再决定接续、释放或重新规划。",
      "Gang并发1_普通任务并发5_怎么主动避让又不浪费资源": "先在队列准入层判断大 Gang 是否进入资源预留窗口；尚未形成可行整组时，只按稀缺 GPU 型号、整机和拓扑形状对关键节点做 Score 降权，避免过早硬锁资源。整组方案可行并创建有效 Reservation 后，普通任务才不得破坏这批关键容量，并优先 Binpack 到其他碎片节点。尚未兑现的保留量可以借给非 Gang 做 Backfill，但只接纳能在大任务启动期限前结束，或能够安全 Checkpoint、抢占和恢复的任务。Reservation 要有 TTL 和取消条件，并结合等待老化、Queue Quota、借用回收、抢占预算与冷却时间。",
      "GPU碎片率怎么定义_如何证明Binpack确实有效": "碎片率需要先限定任务规格，不能只看集群还剩多少张卡。假设只接纳单节点四卡任务，所有卡同型号、其他资源都充足，可以把每个节点空闲卡数除以四的余数相加，再除以总空闲卡数，表示这类任务暂时用不上的空闲容量比例。例如两台各空三张卡，这六张卡都无法接纳四卡任务；换成三卡任务就没有这个问题。比较 Binpack 时固定同一批请求，观察任务接纳数、排队时间和完整空闲节点数，并检查集中放置是否增加了其他资源竞争。",
      "Node选好后具体GPU_CPU_NUMA_NIC由谁落位_怎么保证计划没有被改掉": "传统 Device Plugin 路径由 kubelet Device Manager 分配具体 GPU，GetPreferredAllocation 仅提供偏好，拓扑 Score 只影响节点排序。严格 CPU 绑核还需要 Guaranteed Pod 的整数 CPU 请求、CPU Manager static，以及合适的 Topology Manager restricted 或 single-numa-node 策略。如果控制面必须指定具体 GPU 或 NIC，可以通过 DRA 的 ResourceClaim 记录分配，由节点 Driver 准备并通过 CDI 等方式交付，也可以使用厂商自定义的预留记录与 Device Plugin 协议。执行时核对分配记录、容器实际设备和 cpuset；设备失效、拓扑不再满足要求或实际分配偏离计划时，严格任务应释放资源并重新排队，避免静默换成不符合要求的设备。",
      "八卡GPU全互联是否等于同一个CPU_NUMA": "不等于。NVSwitch 可以让八张 GPU 高速互通，同时它们到 CPU 的 PCIe 上行和 NUMA 亲和仍分布在不同域。先确认机型，再分别查看 nvidia-smi topo -m 的 GPU 间连接与 CPU/NUMA Affinity；卡间通信、CPU 供数和 GPU-NIC 路径要分别判断。",
      "虚拟机GPU直通场景怎么做拓扑感知调度": "先根据宿主机的物理 GPU 互联、PCIe、NUMA 和 NIC 位置选择宿主及具体设备，再由虚拟化层落实 GPU/NIC 直通、vCPU 绑核和内存 NUMA 绑定。Guest 看到的虚拟 NUMA 不能直接证明宿主物理亲和正确，需要核对两层映射。多 GPU 的 P2P 或 NVSwitch 能力还取决于直通方式、Fabric 分区和虚拟化配置，最后在 Guest 内验证实际设备、通信可达性和带宽。",
      "多卡任务凑够卡但拓扑不好_一定要Pending吗": "只有任务必需的拓扑条件无法满足时才需要继续等待。例如要求同一个高速互联域，凑够卡数但卡分布在不同域，就无法启动；如果互联只是性能偏好，则在满足硬约束的组合中选择得分更高的方案，必要时接受性能退让。",
      "GPU_Pod从创建到运行相比普通Pod多了什么": "主要多了 GPU 资源发布与约束、具体设备分配和运行时设备注入。传统 Device Plugin 路径中，插件通过 ListAndWatch 向 kubelet 报告设备，kubelet 更新节点扩展资源，Pod 用 nvidia.com/gpu 等资源名声明需求；型号和拓扑还需要标签、亲和性或调度扩展支持。Scheduler 选 Node 后，kubelet Device Manager 选择健康设备 ID 并调用 Allocate，再由运行时根据返回的设备、挂载、环境变量或 CDI 信息交付给容器。只请求一张 GPU 不会让默认调度器自动理解卡间拓扑。",
      "Gang整组可行方案怎么搜_举一个贪心失败后回溯的例子": "我会先基于同一份资源快照生成候选节点，把已有占用、assumed Pod 和有效 Reservation 都扣掉，再优先放候选最少的 Pod，候选数相同时先放需求大的。搜索过程中只扣减临时容量，后续成员放不下就撤销试放并换位置。假设一个 minMember=3、三个成员都必须启动的任务，N1 剩 6 核、24 GiB、3 卡，N2 剩 8 核、32 GiB、4 卡；P1 要 6 核、24 GiB、3 卡，P2/P3 各要 4 核、16 GiB、2 卡，其他硬约束均满足。如果因为软偏好先放 P1→N2，再放 P2→N1，两个节点都只剩 2 核、8 GiB、1 卡，P3 就放不下；撤销 P2 和 P1 后改成 P1→N1、P2/P3→N2，整组恰好可行。这说明逐个 Pod 贪心失败不代表整组无解。实际搜索要设时间或次数预算，耗尽只表示本轮没找到方案；找到完整方案后，再通过权威资源账本校验和预约，遇到并发冲突就释放本轮占用并重新规划。",
      "八卡申请两卡怎么剪枝和打分_举一个CPU_NIC与碎片联合选择的例子": "先按健康、占用、型号等条件缩小设备集合，再过滤任务要求的互联、CPU NUMA 和 NIC 条件，最后给可行组合打分，兼顾通信代价与后续任务能否利用剩余资源。举一个假设拓扑：八卡中 3 号已占用，可用的高速互联组合来自 0/1/2、4/5、6/7 三组，组间不满足本任务的互联要求；三组分别靠近不同 CPU NUMA 域，可用 CPU 和 RDMA VF 分别为 8 核加 1 个 VF、4 核加 1 个 VF、2 核且无 VF。当前任务要求两卡、4 核和 1 个 VF 位于同一 CPU NUMA 域。八选二原本有 28 种，排除占用卡后剩 21 种，按互联过滤后剩 0/1、0/2、1/2、4/5、6/7 五种，再排除 CPU 和 VF 不足的 6/7，得到四种可行组合。在通信条件相当时，我会选 4/5，保留 0/1/2 及其 CPU、VF 给后续需要三卡、6 核和 1 个 VF 的任务；如果拆掉三卡组，总空闲卡数即使够，也可能凑不出后续任务需要的形状。",
      "故障卡上的Pod迁移是运行中的热迁移吗": "这里通常是终止旧 Pod 后，在健康 GPU 上重建。新进程需要重新加载模型与状态；继续原训练进度时要从最近可用的 Checkpoint 恢复，分布式训练还可能需要整组重启。仅重新调度 Pod 不会搬走原 GPU 显存和进程中的训练进度。",
      "节点_Pod_训练任务和通信组的恢复分别处理什么": "节点恢复处理机器能否重新提供资源，Pod 恢复处理计算进程的重建。分布式训练还要恢复各 rank 的通信关系和一致的训练进度，因此故障可能需要按整个任务或通信组处理。新的 Pod 已经 Running，只说明进程起来了；还要确认各 rank 重新集合并持续推进训练步数。",
      "训练节点坏了_健康进程能原地等新成员吗": "以 torchrun 的常规故障恢复为例，健康机器可以保留，存活的训练进程也会随 worker 组停止和重启。新进程重新 rendezvous、建立通信组，再由训练脚本加载 Checkpoint。若要保留其他训练进程原地等待、只替换故障成员，需要框架支持这种成员变更和状态恢复方式。",
      "示例设计_千卡资源池上如何用Kubernetes驱动PyTorch": "在这套示例中，我会用 Kubeflow Training Operator v1 管理 PyTorchJob，配合 Volcano 分配资源和成组调度，Pod 内用 torchrun 启动 PyTorch 进程。自研平台接收镜像、启动命令、GPU 数量和数据路径，再创建 PyTorchJob。训练数据和 Checkpoint 放在独立 Ceph 集群的 CephFS，通过 PVC 挂载；完成后的模型由平台上传到 Ceph RGW，以 S3 接口管理归档。",
      "示例设计_一个64卡训练具体怎么启动": "在上述示例中，平台创建 1 个 Master Pod 和 7 个 Worker Pod，每个 Pod 申请 8 张 GPU，Master 也参与训练。配置 Training Operator 与 Volcano 集成，将 PodGroup 的最小成员数设为 8，满足整组资源与放置约束后准入。每个 Pod 内的 torchrun 启动 8 个本地训练进程，使用同一任务的 rendezvous 配置，共 64 个进程，再由训练代码初始化 NCCL 通信组。其余 GPU 可以运行其他独立任务。",
      "换机器后训练进度从哪里恢复_torchrun会自动保存吗": "训练进度从持久化 Checkpoint 恢复，保存和加载由训练代码实现。以共享 CephFS 为例，代码保存模型、优化器、训练步数和恢复所需的其他状态，确认全部必要文件或分片完成后才标记这次 Checkpoint 可用。新任务挂载原来的 PVC，加载最近一次完整记录继续训练；torchrun 负责进程启动和重启。",
      "示例设计_一台机器坏了会重启整个千卡资源池吗": "在每台八卡机器由该 64 卡任务独占、共享网络和存储正常的示例里，只需恢复受影响的这个任务，其他独立任务继续运行。我会让平台停止旧任务，确认失联节点的旧实例已停止，或通过基础设施隔离阻止它继续通信和写 Checkpoint，再申请替代资源重建任务。新进程重新建立通信组并加载有效 Checkpoint；健康的物理服务器可以继续使用。"
    },
    "社区场景演练": {
      "使用说明": "以下8案依据官方和社区公开资料构造，用于练习故障定位与GPU使用，均未实际执行。引用时保留场景或假设语气，按当前问题选择对应症状、判断或操作；本人实践从project_facts的实践案例检索。",
      "案例": {
        "XID79掉卡与可分配资源减少": {
          "场景类型": "社区场景演练；依据 NVIDIA 官方排障材料构造，未实际执行，不是本人线上事故。",
          "场景设定": "一台预期八卡的 Kubernetes GPU 节点只公布七张可分配 GPU，业务报设备不可访问；这是演练输入，尚未判断为 PCIe 枚举丢失、驱动访问失败或设备被标为 unhealthy。",
          "排查步骤": [
            {
              "命令": "kubectl describe node <node>",
              "观测": "核对 Capacity、Allocatable、事件及节点资源基线；仅凭七张可分配卡不能断言物理少卡。"
            },
            {
              "命令": "nvidia-smi --query-gpu=index,uuid,pci.bus_id --format=csv",
              "观测": "建立 GPU index、UUID、BDF 对应，记录命令返回码；不可访问时使用故障前清单与日志定位。"
            },
            {
              "命令": "sudo lspci -Dnn",
              "观测": "核对目标 BDF 是否仍可被 PCI 总线枚举，与驱动可见设备区分。"
            },
            {
              "命令": "sudo journalctl -k -b --no-pager",
              "观测": "按故障时间和 BDF 查 XID79、PCIe/AER 及关联错误，避免只摘一行日志归因。"
            },
            {
              "命令": "kubectl logs -n <operator-namespace> <device-plugin-pod> -c nvidia-device-plugin",
              "观测": "检查同一 UUID 的 unhealthy 记录，与内核日志和可分配资源减少的时间对齐。"
            }
          ],
          "处理动作": {
            "前提": "先确定受影响设备、节点上的任务和允许中断的范围；主机重启前停止新任务进入，确认需要中断的任务已停止，并留存故障日志。",
            "动作": "参照对应架构的 XID79 指引安排主机重启和支持排查；不能只重建 Pod 就宣布完成。重复出现时保留隔离，连同完整日志、系统与驱动版本、nvidia-bug-report 交给平台或厂商进一步检查 PCIe、供电及驱动因素。"
          },
          "验收要点": [
            "对照基线检查 PCIe 枚举、GPU UUID 与驱动访问均正常。",
            "核对 Device Plugin 健康状态及预期可分配 GPU 数量；资源数恢复后，在隔离环境执行已知可工作的 GPU 小任务。",
            "观察新一轮日志及实际工作负载；原始错误持续或检测无结果时不回池。"
          ],
          "口述示例": "按社区场景演练，八卡节点只剩七张可分配卡时，我会先把资源上报和物理掉卡分开。先对齐故障 GPU 的 UUID、BDF，再查 PCIe 枚举、内核 XID79 和设备插件的 unhealthy 记录。79 说明驱动已经无法经 PCIe 访问这张卡，但还不能定性为板卡损坏。确认任务已停止并留存日志后，再按平台要求重启。恢复时重新核对设备清单、可分配数量和 GPU 小任务；错误复发就继续隔离，转查链路、供电或驱动。"
        },
        "ECC相关XID94与95的不同恢复范围": {
          "场景类型": "社区场景演练；基于支持错误隔离的 NVIDIA GPU 官方语义，未实际执行，不是本人线上事故。",
          "场景设定": "同一套演练分别注入两种日志材料：应用退出并伴随 ECC 相关 XID94；另一分支报 XID95。仅用来比较故障影响域和恢复范围，不伪造真实故障注入或检测结果。",
          "排查步骤": [
            {
              "命令": "sudo journalctl -k -b --no-pager",
              "观测": "按时间、GPU UUID/BDF 关联 48、63、64、94、95，若有 154 则读取其恢复要求；不只看最后一个错误码。"
            },
            {
              "命令": "nvidia-smi -q -i <gpu-uuid>",
              "观测": "读取设备型号、驱动、ECC、页退休/行重映射和 MIG 状态；字段及含义以该型号支持范围为准。"
            },
            {
              "命令": "nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv",
              "观测": "识别占用故障设备的进程，并关联 Pod、训练 Job 或推理实例。"
            },
            {
              "命令": "kubectl get pods -A --field-selector spec.nodeName=<node> -o wide",
              "观测": "关联节点上任务与控制器，确认影响的是单个应用、同一 GPU 的其他实例还是分布式任务整组。"
            },
            {
              "命令": "nvidia-smi topo -m",
              "观测": "与平台拓扑清单一起判断 NVLink/NVSwitch 连接及实际 Reset 范围；这一命令不能独立证明允许单卡 Reset。"
            }
          ],
          "处理动作": {
            "前提": "核对 GPU 架构及 XID154/厂商恢复要求；Reset 前停止使用目标设备及实际 Reset 组的全部任务、监控和服务，MIG 也需覆盖受影响物理 GPU 的其他实例。",
            "动作": "94 分支先恢复受影响应用，并安排合适维护时机；95 分支在恢复应用前完成所需 GPU Reset。无法单卡操作或具体平台要求主机/VM 重启时，扩大到平台规定的范围。训练 Job 是否整组重启由任务控制器与通信依赖决定。"
          },
          "验收要点": [
            "确认恢复操作后设备可用，检查原始错误及行重映射相关状态；历史累计 ECC 计数不要求清零。",
            "用已知可工作的任务检验运行，核对是否产生新的不可纠正错误及设备健康状态。",
            "训练验证任务重建与训练步推进；推理验证实际推理成功和接流状态，不能只看进程存在。"
          ],
          "口述示例": "按社区场景演练，同样是内存错误，我会把 XID94 和95分开处理。94 的错误被限制在受影响应用内，先定位该应用；如果是分布式训练，还要看是否需要整组重建。95 会影响多个应用，必须先完成要求的 GPU Reset，再恢复任务。动手前查清 MIG、互联拓扑和占卡进程，不能为了修一个实例中断未确认的其他任务。验收看原错误是否复发、设备状态和任务是否真正推进；历史 ECC 累计值没有归零不等于修复失败。"
        },
        "NVSwitch_FabricManager未就绪导致CUDA初始化失败": {
          "场景类型": "社区场景演练；依据 NVIDIA 官方 GPU Operator 与 Fabric Manager 文档，未实际执行，不是本人线上事故。",
          "场景设定": "NVSwitch 节点上 nvidia-smi 能看到 GPU，但 GPU Operator validator 或 CUDA 程序初始化失败并报 system not yet initialized。FM 未安装、未运行或初始化失败是排查分支，不预先认定根因。",
          "排查步骤": [
            {
              "命令": "kubectl logs -n <operator-namespace> <validator-pod> -c <failing-container> --previous",
              "观测": "确认是 CUDA 初始化错误；若没有上次容器日志，读取本次日志，并与同节点其他失败任务对齐。"
            },
            {
              "命令": "nvidia-smi -q",
              "观测": "查看 GPU 和 Fabric 信息；长期停在 In Progress 时继续查 fabric 初始化，而非仅凭枚举成功判断 GPU 可用。容器驱动模式在 driver container 内执行。"
            },
            {
              "命令": "kubectl logs -n <operator-namespace> <driver-pod> -c nvidia-driver-ctr",
              "观测": "确认是否发现 NVSwitch、是否安装并启动 FM，以及是否有软件包或启动错误。"
            },
            {
              "命令": "cat /var/log/fabricmanager.log",
              "观测": "在实际运行 FM 的宿主机或 driver container 内读取日志，区分启动、配置、驱动兼容与设备初始化错误。"
            },
            {
              "命令": "systemctl status nvidia-fabricmanager --no-pager",
              "观测": "仅用于宿主机 systemd 管理的 FM；结合驱动和 FM 软件包版本确认部署是否符合该平台要求，容器模式看容器进程和 Operator 日志。"
            }
          ],
          "处理动作": {
            "前提": "先确认属于依赖相应 fabric 管理组件的 NVSwitch 平台；按实际 GPU/NVSwitch 代际和驱动部署方式选修复步骤。需要重启 FM、Reset 或更换驱动时，先停止受影响 CUDA 任务及占卡服务。",
            "动作": "若证据表明 FM 缺失或启动配置错误，修复对应组件及驱动兼容关系；若是 fabric 注册或链路失败，按该平台官方恢复流程处理。H100 及更新平台与 A100 初始化机制不同，新一代还可能涉及 NVLSM，不能照抄一种服务重启流程。"
          },
          "验收要点": [
            "在提供该字段的平台确认各目标 GPU 的 Fabric State=Completed、Status=Success；不将某一张卡的成功当成全节点成功。",
            "重新执行原来失败的 validator 或已知 CUDA 小任务，确认初始化与计算都成功。",
            "对实际需要多卡通信的业务验证 P2P/集合通信功能，与基线比较；再恢复调度和流量，并观察新的 XID/SXID 或对应架构错误遥测。"
          ],
          "口述示例": "按社区场景演练，nvidia-smi 能看到卡，但 CUDA 报系统尚未初始化时，我会继续查 NVSwitch 的 fabric 状态。若停在 In Progress，就查驱动容器是否安装了 Fabric Manager，再看它的启动日志和兼容关系，不会直接把卡判坏。确认是软件初始化问题后，先停受影响任务，再按该平台流程修复。验收既看 fabric 注册完成，也重跑原失败任务；需要多卡通信的业务还要检验通信，单卡可见不能代表整机能训练。"
        },
        "跨节点NCCL初始化卡住_误选不互通网卡": {
          "场景类型": "社区场景演练；依据 NVIDIA 官方排障文档编写，未执行，不是本人处理过的事故。",
          "场景设定": "同一镜像和程序单机多卡能完成；扩成两个节点后，所有进程已启动但 NCCL 初始化卡住。节点同时有业务网和管理网，待验证的假设是 NCCL 选择了处于 UP 状态却不能跨节点通信的接口。",
          "排查步骤": [
            "用 NCCL_DEBUG=INFO、NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,NET、NCCL_DEBUG_FILE=/tmp/nccl_%h_%p.log 收集每个进程日志；确认 rank 数量完整，区分初始化连接失败与已经进入集合通信后的阻塞。",
            "在实际任务容器内执行 ip -br addr 与 ip route get <对端地址>，将接口、地址和路由与 NCCL 日志中选中的接口对照；宿主机能访问不代替容器网络可达。",
            "在两端实际任务网络内，用获准的临时 TCP listener/client 双向验证所选地址可达，再检查 NCCL 使用的端口是否被防火墙阻断；ping 成功不足以证明 TCP 建连。",
            "只有日志与连通性共同指向误选接口时，进行一次只改变 NCCL_SOCKET_IFNAME 的对照；检查新日志是否确实选中了目标网卡。"
          ],
          "处理前提": "各 rank 已正确启动，单机测试及设备初始化正常，已有获准且双向可达的业务接口。只针对当前测试任务设置；若卡在集合通信顺序或某个 rank 已退出，应转查对应问题。",
          "处理动作": "在任务启动环境中设置 NCCL_SOCKET_IFNAME 为精确接口名，例如 NCCL_SOCKET_IFNAME=\"=eth1\"；eth1 仅为示例，每台主机按实测接口配置。该参数选择 IP/socket 接口，不替代 RDMA HCA 选择或底层网络修复。",
          "验收要点": [
            "先用原来相同节点、镜像和设备分配执行最小双节点 AllReduce；所有 rank 正常完成，集合结果校验无错误，日志无连接超时。",
            "再恢复原任务规模和输入，确认原阻塞不复现；保留修改前后日志及测试命令，不把小规模通过等同于整体训练性能达标。"
          ],
          "口述示例": "这是社区场景演练：单机多卡能跑，两节点却卡在NCCL初始化。我会收齐各rank日志，先区分连接没建立还是集合通信卡住，再核对选中的网卡。接口显示UP不等于节点间可达；若误选了隔离的管理网，就在任务容器中确认业务网双向连通，再为这次任务指定该接口。复测先跑小规模AllReduce，要求所有rank结束且结果正确，再回到原任务；日志不支持误选接口，就不能把改参数当作修复。"
        },
        "容器能运行nvidia_smi但CUDA不可用_漏挂compute能力": {
          "场景类型": "社区场景演练；由 NVIDIA 官方驱动能力说明推导的可复现场景，未执行，不是本人事故。",
          "场景设定": "已验证驱动正常的节点上，某个使用 NVIDIA Container Runtime 的容器可以运行 nvidia-smi，但 CUDA 程序加载驱动或初始化失败；容器启动配置显式把 NVIDIA_DRIVER_CAPABILITIES 覆盖成 utility。具体错误文本取决于应用，不预设固定报错。",
          "排查步骤": [
            "保留失败容器的镜像摘要、启动参数和原始错误；在同节点、同一 GPU 上用已知兼容镜像执行最小 CUDA 程序作对照，不能只比较 nvidia-smi。",
            "检查容器启动配置和进程环境中的 NVIDIA_VISIBLE_DEVICES、NVIDIA_DRIVER_CAPABILITIES，以及实际 NVIDIA runtime 模式；确认 utility 是显式覆盖值。",
            "在容器内用动态库检查确认 libnvidia-ml 可用而 CUDA 驱动库 libcuda.so.1 是否缺失或不能加载；例如使用 ctypes.CDLL(\"libcuda.so.1\") 保存原始加载异常。",
            "核对声明值与实际挂载；若 compute 已启用且 libcuda 加载正常，就不把故障归因于本案例，继续根据真实错误检查驱动兼容或设备访问。"
          ],
          "处理前提": "已通过健康容器排除该节点/GPU 基础故障；使用支持这些变量的 NVIDIA Container Runtime 配置路径，且实际证据确认缺少 compute 能力挂载。",
          "处理动作": "修改工作负载启动配置为 NVIDIA_DRIVER_CAPABILITIES=compute,utility，并重建一个测试容器。该变量决定创建时挂入的驱动库与工具，只在已启动容器里 export 不能补齐挂载；不通过安装容器内宿主驱动或启用 privileged 处理这个问题。",
          "验收要点": [
            "重建后既检查 GPU UUID 与预期分配一致，也确认 libcuda.so.1 可加载。",
            "运行最小 CUDA 程序完成设备初始化、显存分配、kernel 执行、同步和结果校验，再验证原应用首个实际计算；nvidia-smi 成功不能单独结案。"
          ],
          "口述示例": "这是社区场景演练：容器能看到GPU，CUDA却初始化失败。我会先在同节点用正常镜像验证计算，再查启动配置和实际驱动库。若只声明utility，显卡管理工具能工作，却没启用CUDA需要的compute。确认缺失后，修改启动配置并重建测试容器；在旧容器里补环境变量不会补挂驱动库。验收必须完成显存分配、一次计算和同步，再恢复原应用，不能只看显卡管理工具正常就结案。"
        },
        "训练越跑显存越高_统计代码保留GPU张量": {
          "场景类型": "场景演练；按PyTorch官方内存语义构造，未执行。",
          "场景设定": "固定模型与batch起初能训练，运行一段时间后OOM；假设统计列表每轮追加GPU输出且一直不释放，待用内存快照检验这个假设。",
          "排查步骤": [
            "保留OOM堆栈和每轮输入shape；用nvidia-smi核对实际占用进程，先排除另一进程占卡。",
            "在原进程内按固定步数记录torch.cuda.memory_allocated()、memory_reserved()和max_memory_allocated()；分别观察活跃张量与缓存池。",
            "检查统计、评估、日志和缓存容器是否持有GPU张量；调用torch.cuda.memory_summary()，必要时按部署版本使用内存快照定位仍活跃的分配。",
            "在同样输入与步数下，只改统计对象的生命周期，对比allocated曲线、峰值和是否仍OOM。"
          ],
          "处理动作": "如果确认统计列表持有不再需要的GPU输出，对不参与反传的统计先detach再转CPU，单值使用item()，并及时释放原GPU引用。empty_cache只释放空闲缓存，不能释放仍被引用的张量；若峰值来自合法计算需求，再单独调整batch、序列长度或训练内存策略。",
          "验收要点": "固定输入与重复步数下，allocated不再逐轮无界增长，训练结果和统计保持正确；保留修改前后内存曲线。这里只给预期判据，没有声称跑出了稳定曲线。",
          "口述示例": "如果固定batch开始能跑、后来显存持续增长，我会先记录每轮allocated和reserved，区分活跃张量增加与缓存池保留。这个演练设定的是统计列表一直存GPU输出：定位到持有引用的列表后，统计先detach再转CPU，单值用item()，并及时释放原GPU引用。然后固定输入和运行步数复测内存曲线，而不是只调用empty_cache看nvidia-smi数字降没降。若allocated依旧上涨，就继续查持有者；只有曲线和结果都验证过才算修复。"
        },
        "vLLM长请求增多后频繁重算_吞吐和尾延迟取舍": {
          "场景类型": "场景演练；依据vLLM官方调优文档构造，未执行。",
          "场景设定": "模型已经加载成功，短请求可正常服务；长请求和并发增加后生成出现停顿，日志出现KV空间不足和preemption/recompute信息。",
          "排查步骤": [
            "固定模型、vLLM版本、启动参数与输入/输出长度分布，记录是加载阶段失败还是请求运行阶段出现问题。",
            "检查运行/等待请求、KV使用、抢占重算日志与指标，并保留每请求到达、首token、结束和token间隔记录。",
            "确认实际同时运行的请求是否达到配置上限；不要把客户端并发或max_num_seqs配置值直接当引擎实际并发。",
            "用同一批请求只降低一项准入参数复测，例如max_num_seqs；同时比较完成率、吞吐、排队、TTFT与生成停顿。"
          ],
          "处理动作": "如果证据指向运行期KV压力，先通过减少同时运行请求或调整预算做对照；有显存余量时可按版本评估缓存预算。加载或CUDA graph捕获阶段OOM则另查权重、上下文配置与graph占用，不能用同一调参结论解释所有阶段。",
          "验收要点": "同一输入下核对抢占次数、实际并发、排队暴露和完整请求延迟；抢占减少但TTFT或吞吐变差时保留这个代价，不把零抢占直接判为最优。",
          "口述示例": "模型能启动、长请求增多后才卡顿，我会先看KV使用和抢占重算日志。若确实是KV压力，就固定模型和请求集合，只降低同时运行请求数做对照，同时记录实际并发、排队、首token和最长生成停顿。减少抢占可能改善生成连续性，也可能把等待转移到入口，所以我不会只报抢占降了多少。加载阶段或CUDA graph捕获时的OOM要另查，不能和运行期KV不足混成一个问题。"
        },
        "GPU利用率低_先定位输入等待还是设备瓶颈": {
          "场景类型": "场景演练；依据NVIDIA指标定义与PyTorch Profiler能力构造，未执行。",
          "场景设定": "训练step耗时长，GPU-Util时高时低；假设CPU解码或数据读取赶不上GPU，需要时间线验证，不能先认定GPU性能不足。",
          "排查步骤": [
            "在同一观测窗口记录nvidia-smi dmon -s pucm -d 1 -c 30，并记录step耗时和应用吞吐；指标不支持时保留N/A，不当成零。",
            "在固定少量step上用torch.profiler同时采CPU/CUDA，在取batch、预处理、H2D和forward/backward外加record_function标记。",
            "如果时间线显示GPU空档主要在取数据阶段，检查CPU、存储与数据加载；用预加载的同一小批数据作诊断对照，注明它只用于定位瓶颈。",
            "如果GPU持续执行但吞吐低，再检查算子、访存、通信、时钟与功耗限制；按证据一次只调整一个变量。"
          ],
          "处理动作": "确认输入侧阻塞后再调整数据预处理、加载并行度或预取；确认传输问题后再评估固定页内存与异步拷贝。不能仅凭GPU-Util低就提高并发，也不把显存占用高当成算力饱和。",
          "验收要点": "回到原始端到端数据路径，固定模型与输入重测吞吐和step分布，并确认CPU/内存代价；预加载数据的诊断结果不充当业务提速结果。在线推理还需检查TTFT及尾延迟。",
          "口述示例": "GPU利用率低时，我会把GPU采样与训练step时间线对齐。假设Profiler显示大段空档落在取batch和CPU解码上，我会用预加载的同一批数据作定位对照，确认GPU是否一直在等输入，再调整实际数据加载路径。最后必须回到原始数据集和读盘方式看端到端吞吐，不能拿预加载的数字当业务收益。GPU-Util表示采样窗口里有kernel执行的时间比例，单看它也不能判断算力或显存带宽是否充分利用。"
        }
      }
    },
    "公开实操复盘与口述": {
      "使用方式": "以下为他人公开实操、厂商排障实例和官方故障解释，保留原环境、观察、实际处理与结果。引用时简短说明公开来源，按当前问题选取所需事实；作者判断与已确认根因分别记录。",
      "案例": {
        "H100_XID94后设备可见但可分配资源仍为七卡": {
          "类型": "公开 issue，原作者附操作前后输出",
          "环境": "2024-10，GPU Operator v23.9.2，8×H100 80GB，驱动535.129.03。",
          "现象": "XID94之后，Node Capacity为8而Allocatable为7；后续nvidia-smi仍能枚举8卡，第0卡Volatile Uncorrectable ECC仍为2。",
          "实际操作与结果": "作者重启device-plugin Pod后，贴出Allocatable恢复为8的结果。",
          "实操要点": "把GPU UUID、设备健康日志和Node资源状态对齐。物理可见、CUDA任务可运行、插件愿意分配，是三个验收点。",
          "结果边界": "帖子证明插件重启后资源重新上报，不证明底层内存故障已消除；不能直接把重启插件当成XID94修复方案。",
          "维护者解释": "2025-01，k8s-device-plugin #1014的维护者解释：当时版本的设备一旦被置为unhealthy，不会自行变回healthy，需要重启插件。这是历史版本行为，不能据此声称所有后续版本相同。"
        },
        "Dell_R730_P100可枚举但nvidia_smi无设备": {
          "类型": "NVIDIA论坛，原作者回帖并附恢复识别输出",
          "环境": "2023-08，Dell R730，Riser 3，Tesla P100 PCIe 16GB，Ubuntu22.04.3。",
          "现象": "lspci能看到03:00.0的P100，驱动模块已加载，但nvidia-smi显示No devices were found；尝试BIOS和多种驱动安装方式未解决。",
          "实际操作与结果": "作者更换适配Dell riser针脚定义的供电线，之后又处理驱动通信失败，清理重装驱动，最终贴出P100识别成功的输出。",
          "实操要点": "PCIe枚举存在不能证明设备供电和驱动初始化都正常。服务器GPU供电线要按对应机型和riser规格核对，不能只看插头能否插入。",
          "结果边界": "这是供电接线与驱动安装共同处理后的识别结果，没有单变量归因或长期计算验证；原帖没有XID79，不能补成XID79事故。"
        },
        "XID79交叉换卡后决定送RMA": {
          "类型": "NVIDIA论坛，原作者持续跟进",
          "环境": "2019—2020，EVGA GTX980Ti Hybrid；消费卡案例，用于说明故障隔离方法。",
          "现象": "特定游戏中冻结，SSH日志有XID79，作者提交nvidia-bug-report；更换驱动、系统、BIOS、插槽、线缆和PSU未解决。",
          "实际操作与结果": "原卡放到朋友电脑仍复现；朋友RTX2070SUPER放到原机短时正常，另一张RX580在原Linux主机连续使用正常。作者据此决定送RMA。",
          "实操要点": "故障跟着原卡跨机器复现，加上替换卡在原机运行正常，增强卡侧故障判断；保留换PSU无效的排除证据。",
          "结果边界": "只有决定送RMA的后续，没有厂商检测报告或换卡后验收，不能说已定位芯片损坏。数据中心PCIe/SXM设备的拆换需按对应服务手册由现场人员操作。"
        },
        "Crusoe_GPU内存重映射待执行与失败分流": {
          "类型": "云厂商公开排障实例及处置规则，不是单个用户事故",
          "现象": "文档给出两种Remapped Rows输出：Pending=Yes、Failure=No；以及Pending=No、Remapping Failure Occurred=Yes。",
          "实际处置": "待执行分支按该云平台流程安排GPU重置和实例重启，复查Pending与Failure均为No；失败分支提交nvidia-bug-report、nvidia-smi -q和Xid日志，由云厂商处理更换。",
          "实操要点": "看待执行与失败标志，区分DRAM/SRAM及Volatile/Aggregate；历史累计计数不归零不能独立作为修复失败依据。",
          "适用边界": "Crusoe的换卡服务规则不能推广为所有厂商统一RMA阈值。NVIDIA的row-remapping RMA判断还要求Field Diagnostic验证；DCGM检查通过也不能代替所有厂商诊断。"
        },
        "八卡诊断只有GPU0触发散热违规": {
          "类型": "工程师公开实操笔记中的诊断输出",
          "现象": "Stas Bekman公布的dcgmi diag -r 2输出的GPU Memory子项中，GPU1—7通过，GPU0失败；测试开始9.7秒后出现累计13.3秒的thermal violations。",
          "定位动作": "同节点逐卡比较诊断结果，把GPU0散热异常单独定位出来；诊断提示进一步检查机器冷却、散热接触和风扇。",
          "结果边界": "公开记录给出了异常定位，没有给出更换某个散热器后的复测结果。13.3秒是该次违规累计时间，不能当统一阈值。",
          "额外观察": "同一笔记还给出PCIe启动日志：设备支持32GT/s x16，但路径被上游16GT/s x16限制。它是路径能力瓶颈的示例，不能直接说GPU损坏或声称测得端到端带宽减半。"
        },
        "A100多机NCCL初始化挂住_更换bootstrap接口": {
          "类型": "NVIDIA/nccl issue，作者确认调整后持续运行",
          "环境": "2022，每节点4×A100-SXM4-40GB、2×Mellanox HDR100；16卡运行更容易挂住。",
          "现象": "作者从TensorFlow/Horovod缩小到OSU/NCCL测试；gdb显示阻塞在socket接收，升级NCCL2.11.4仍复现。",
          "实际操作与结果": "按维护者建议，把socket/bootstrap使用的接口从ib-bond0换成eno1np0，作者报告稳定运行；数据传输继续使用IB Verbs。",
          "实操要点": "建连用的IP接口与数据传输HCA分别检查。任务卡在初始化时，先看Bootstrap、NET日志和进程栈，再决定查哪个网络路径。",
          "结果边界": "作者报告换接口有效；具体MTU根因没有得到证明，没有前后带宽跑数。原接口名只适用于该现场。"
        },
        "双机H100_IB测试通过但NCCL失败_核对ACS与GDR": {
          "类型": "NVIDIA/nccl-tests issue，作者与维护者有完整跟进",
          "环境": "2024-09，两个节点、每节点4张H100 80GB，四个200Gb/s训练网口。",
          "现象": "默认host-memory ib_write_bw通过，NCCL报remote process exited or network error；暂时设置NCCL_NET_GDR_LEVEL=0后NCCL能通过。",
          "实际操作与结果": "维护者指出host-memory测试没有覆盖GPUDirect RDMA路径；作者修正裸金属ACS配置后报告NCCL和HPL无须禁用参数即可通过。随后为核对网络规格，按建议用Ring重新测量，贴出的平均AllReduce busbw为95.3009GB/s，正确性检查为0错误。",
          "实操要点": "先区分CPU内存到网卡和GPU内存到网卡的路径；临时禁用GDR用于定位，确认根因后恢复正常路径。不能凭驱动模块存在认定GDR实际被使用。",
          "带宽口径": "每个200Gb/s端口名义单向25GB/s，四口合计100GB/s；该次Ring结果约95.3GB/s属于nccl-tests折算busbw。算法和rank数影响指标含义，不能普遍把busbw等同网口吞吐。",
          "适用边界": "ACS处理是该裸金属环境的处置记录，虚拟化涉及ACS/ATS和设备隔离，不能照搬为关闭ACS的通用建议。Ring用于该次核对，不是所有业务的最优算法。"
        },
        "双机3090指定万兆接口后发送仍走千兆": {
          "类型": "NVIDIA/nccl issue，作者报告临时解决",
          "环境": "2025，双机Debian、RTX3090，千兆eno1与万兆enp37s0f0并存。",
          "现象": "作者观察千兆接口发送、万兆接口接收；多种方式设置NCCL_SOCKET_IFNAME仍未解决。路由表中两个接口覆盖相同192.168.0.0/20。",
          "实际操作与结果": "增加指向万兆接口的更具体/24路由后，作者报告暂时解决。",
          "实操要点": "环境变量选中了接口之后，仍需核对实际路由和网口发送/接收计数；配置值不能替代实际流量路径。",
          "结果边界": "没有修复后吞吐和长期验收，也没有已确认的NCCL缺陷结论。不能直接复制原网段到其他环境。"
        },
        "Meta_Llama3训练平台的GPU网络存储画像": {
          "类型": "平台建设方公开工程文章，2024-03历史实例",
          "实际平台": "Meta公开两套各24576张H100的训练集群；一套RoCE，一套Quantum2 InfiniBand，均使用400Gbps端点；文中明确Llama3在RoCE集群训练。",
          "系统组成": "计算基于Grand Teton；软件采用PyTorch；训练数据与Checkpoint存储经FUSE API接入为Flash优化的Tectonic。",
          "实操要点": "描述GPU平台时一起说明工作负载、卡型与规模、节点间网络、数据读取和Checkpoint路径；这些约束共同决定训练能否持续推进。",
          "结果边界": "24576是每套集群容量，不能写成该篇文章证明每个训练任务都占用全部卡；这是Meta的公开环境，不是本人客户集群。"
        }
      },
      "面经口述": {
        "公开复盘有哪些典型硬件故障现象": {
          "口述": "公开实操里值得准备的三类现象是掉卡或设备不可访问、内存不可纠正错误，以及高温导致的异常。GPU资源不可分配可以讲GPU Operator的一次H100案例：nvidia-smi还能看到8张卡，但Kubernetes只报7张可分配卡，作者重启设备插件后才恢复资源上报。这说明排障要把物理枚举、驱动状态和插件健康状态分开。散热也有具体诊断例子：同一八卡节点的GPU Memory检查里只有GPU0失败，报告明确记录thermal violations，其他七张通过，此时应继续查这张卡的散热路径。",
          "追问": "问题若限定本人遇到最多，频次必须来自本人记录；这些公开案例只说明典型现象，不提供可靠的统一故障排名。",
          "案例": [
            "H100_XID94后设备可见但可分配资源仍为七卡",
            "八卡诊断只有GPU0触发散热违规",
            "Crusoe_GPU内存重映射待执行与失败分流"
          ]
        },
        "PCIe具体有什么故障_举一个实操例子": {
          "口述": "我研究过一个Dell R730安装P100的公开案例：lspci已经能列出卡，nvidia-smi却报没有设备。作者先试BIOS和驱动，后来发现供电线需要匹配Dell riser的针脚定义，换正确线缆并清理重装驱动后，才贴出成功识别的结果。这个例子里，枚举得到设备只是第一步，接下来还要核对供电、驱动初始化和实际CUDA任务。对于真正的PCIe链路报错，我会把GPU的BDF和上游端口对上，结合AER严重性、Link状态和同路径其他设备判断影响范围。",
          "追问": "lspci -Dnn看完整地址，lspci -t看上游关系，lspci -vv -s <BDF>比较LnkCap/LnkSta及错误状态。错误由Root Port报告不等于Root Port损坏；记录中的P100案例也没有XID79，不能补成同一种故障。",
          "案例": [
            "Dell_R730_P100可枚举但nvidia_smi无设备",
            "八卡诊断只有GPU0触发散热违规"
          ]
        },
        "硬件真的坏掉_如何确认而不只重启": {
          "口述": "判断卡侧硬件问题，我会找重复性和交叉对照证据。一个公开XID79案例中，作者换驱动、系统、线缆和电源都没解决；原卡放朋友机器仍复现，替换卡放原机却能运行，最后决定送RMA。真正有价值的是故障跟着原卡走，前面换电源无效也被保留下来。数据中心内存故障还能看更直接的证据：Crusoe文档把row-remapping待执行和失败分开，失败标志出现就带日志交厂商；这里的DRAM行重映射失败，按NVIDIA对应内存RMA规则还要由Field Diagnostic验证。",
          "追问": "交叉换卡要按UUID或序列号追踪，换槽后BDF可能变化；拆换由现场按机型手册执行。社区帖只记录决定送RMA，不能扩写成厂商已经确认芯片损坏、换卡后长期恢复。",
          "案例": [
            "XID79交叉换卡后决定送RMA",
            "Crusoe_GPU内存重映射待执行与失败分流"
          ]
        },
        "公开案例_Meta的Llama3训练平台怎样使用GPU": {
          "口述": "我研究过Meta公开的Llama3训练平台，它会把GPU、网络和存储放在一起讲：两套集群各24576张H100，一套RoCE、一套InfiniBand，端点都是400Gbps；Llama3实际在RoCE集群训练。数据和Checkpoint通过FUSE接到Tectonic存储。",
          "追问": "回答本人平台时，使用本人确认的模型、卡型、拓扑和任务记录；不套用Meta规模。本人单卡推理实验从project_facts中硕士毕业论文的resume_positioning、experiment_facts检索，不能冒充客户集群训练。",
          "案例": [
            "Meta_Llama3训练平台的GPU网络存储画像",
            "A100多机NCCL初始化挂住_更换bootstrap接口"
          ]
        },
        "GPU节点之间带宽怎么样_实际怎样排查": {
          "口述": "拿NVIDIA社区一个实际双机测试说：每台4张H100、四个200G网口，单端口名义单向25GB/s，四口合计100GB/s。现场先出现ib_write_bw能过、NCCL却失败；临时关闭GDR后能跑，最后调整裸金属ACS配置后恢复正常GDR路径。为核对网络能力，作者再用Ring测AllReduce，平均busbw约95.3GB/s且正确性检查通过。我的排查顺序会是单机通信、跨机单端口、GPU内存通信，最后回到原训练任务。",
          "追问": "algbw是逻辑数据量/耗时；AllReduce busbw按2(n-1)/n折算，该例8个rank的因子是1.75。它不是网卡计数器，混合NVLink与RDMA时不能机械换算NIC利用率。默认host-memory ib_write_bw通过不证明GDR通过；不要把临时关闭GDR当性能修复。",
          "案例": [
            "双机H100_IB测试通过但NCCL失败_核对ACS与GDR"
          ],
          "补充追问": "初始化挂住可讲A100换bootstrap接口案例；配置了快网卡却没有走对路径可讲双机3090路由案例。两个案例都没有可引用的提速百分比。"
        },
        "实践中哪些XID严重_怎么判断": {
          "口述": "我会先看设备还能不能访问、错误影响哪个应用，以及驱动要求什么恢复动作。79代表驱动已经无法通过PCIe访问GPU，应先保留日志并按对应平台恢复流程处理，但原因可能在卡、链路或其他环节。内存错误里，支持错误隔离的GPU上，94先恢复受影响应用，95要求在应用重启前完成GPU重置；48是不可以忽略的不可纠正ECC，还要结合同时出现的错误和恢复提示。63是页退休或重映射的记录，64是记录失败。像公开H100案例，插件资源恢复8张后仍应继续核对ECC和任务执行，不能把资源数字回来等同故障修好了。",
          "追问": "复核GPU型号、驱动与Xid154或GPU Recovery Action；对齐GPU UUID/BDF和最早错误时间，查看nvidia-smi -q中的ECC、重映射与恢复标志。需要reset时先协调受影响工作负载，具体重置范围受互联、MIG和虚拟化条件限制。",
          "案例": [
            "H100_XID94后设备可见但可分配资源仍为七卡",
            "Crusoe_GPU内存重映射待执行与失败分流",
            "XID79交叉换卡后决定送RMA"
          ]
        }
      }
    },
    "训推平台": {
      "定位": "训练与推理平台的分层结构、任务编排、通信域与并行策略、推理服务化差异；回答平台形态、Operator、组网和部署链路类问题时检索本节。",
      "平台分层": {
        "三层结构": "底层是 Kubernetes、GPU 调度、存储与网络、Operator 和 CRD；训练控制层是 PyTorchJob/MPIJob 或自研 TrainingJob、分布式拓扑、checkpoint 与重试、队列配额和 gang 调度；平台层是任务提交、参数配置、镜像与数据集管理、日志监控、实验管理、权限租户和计费。",
        "Kubeflow的定位": "Kubeflow 把这三层做成一套相对完整的开源 ML Platform，是训练平台的参考实现之一，不等于所有平台都直接部署它。大规模 GPU 集群通常演化成 Kubernetes 加自研 Operator、自研调度器和自研控制面，因为调度、利用率、故障恢复、训练队列和成本治理要深度定制。",
        "核心主链": "用户提交 TrainingJob，Operator 创建训练 Pod，Scheduler 做 GPU placement，Runtime 与 NCCL 完成组网开始训练，Controller 持续维护 Job 生命周期。"
      },
      "TrainingOperator": {
        "为什么训练任务要写Operator": "训练任务不是一个 Pod，而是一组强关联资源：一个 master 加 N 个 worker、配套 Service 和 ConfigMap、设置 rank 和 world size、等所有 worker Ready、失败后判断重试或重建、结束后更新状态并清理。Operator 把训练领域逻辑和 Kubernetes 控制循环封装在一起，典型实现是 Kubeflow 的 PyTorchJob 和 MPIJob。",
        "和Deployment的区别": "Deployment 少一个 Pod 就补一个 Pod。训练 Operator 面对一个 worker 退出时要按训练语义判断：这个 Job 还能不能继续、重启单个 worker 还是整组重建、要不要从 checkpoint 恢复、rendezvous 是否需要重新组网。根本差别在于训练 Pod 不是对等可替换的副本，它们绑定了 rank 编号并处在同一个通信域里。",
        "期望状态是什么": "期望值不是简单的 Pod 数，而是整个训练 Job 的拓扑和生命周期状态。用 CRD 描述训练拓扑和期望状态，由 Controller 持续 reconcile Pod、角色、失败恢复和任务生命周期，单纯用 Deployment 或 Job 很难表达这些语义。"
      },
      "组网与通信域": {
        "组网指什么": "把参与分布式训练的多台 GPU 服务器连接起来并让训练进程能建立集合通信。包含两层：底层网络是网卡、交换机和 InfiniBand 或 RoCE 提供的高带宽低延迟通道；任务通信配置是保证进程间可达，并分配各进程的地址、rank 编号和通信网卡，让 NCCL 能建立连接。",
        "通信域是什么": "一组参与集合通信的进程或 GPU 的集合，NCCL 和 MPI 里叫 communicator，PyTorch 里叫 process group。集合通信的语义是所有成员一起完成一次归约或收集，所以必须先定义所有成员是谁。它在创建时固定成员数量、rank 编号、彼此的连接通道、算法拓扑和按成员数预分配的缓冲区。",
        "rank与进程的关系": "rank 是逻辑编号，每个 rank 对应一个具体的进程实例。通信域内部维护 rank 到物理位置的映射：在哪台机器、监听哪个端口、用哪张卡、从本端过去走什么链路。bootstrap 阶段所有进程通过 MASTER_ADDR 汇合，上报各自的 IP、端口、设备信息和可用传输方式，汇总广播后按表建立点对点连接；之后点对点直连，不再经过 master。",
        "为什么不能动态增删成员": "ring 算法依赖固定的环形顺序，少一个节点环就断了；连接是建立通信域时预先打通的，加人要重新握手，减人要清理通道；缓冲区和数据切分方式按成员数预分配。所以 NCCL 没有增删 rank 的接口，唯一出路是销毁重建。torchrun elastic 和 ncclCommAbort 让这个过程更自动、让阻塞调用能中止，但本质仍是销毁重建而不是热插拔。",
        "重新组网的触发与代价": "触发场景包括故障节点摘除、重调度或任务迁移、任务扩缩容、通信网络方案变更。重建的原因是成员变化而不是拓扑变化——即使新 Pod 落在规格完全相同的节点上也必须重建，因为通信域绑定的是具体进程实例的映射表和已建立的连接。而且所有通信域一起重建，不只是包含故障卡的那个，因为恢复做法是整组进程退出重启。实践流程是摘除变更节点、重算通信配置、任务重启拉起、从 checkpoint 续训。"
      },
      "并行策略与放置": {
        "分布式推理不只是MoE": "判据是单卡放不下或单卡吞吐不够。TP 把一层拆到多张卡，dense 大模型也常用，主要通信是 AllReduce 和 AllGather；PP 把不同层放不同卡，主要是 Send/Recv；Expert Parallel 把不同 expert 放不同卡，典型通信是 All-to-All；DP 每组卡跑完整副本，为的是吞吐。Dense 模型分布式推理主要是 TP 和 PP，MoE 在此基础上引入 Expert Parallel 和大量 All-to-All。",
        "MoE的通信特点": "一次 forward 有两次数据交换：dispatch 把 token 按 router 结果发到 expert 所在 GPU，combine 把结果发回原 GPU，中间的 expert FFN 是本地计算。这两次本质是 All-to-All，而且通常是 All-to-All-v，因为每张卡发给其他卡的 token 数量取决于 router 负载分布。底层走 NCCL，但很多实现不是调用独立的 AllToAll collective，而是用 Send/Recv 组合或框架封装的 primitive。MoE 的瓶颈常在 token dispatch 和 expert 间通信而不是算力，所以 serving 特别关注 expert placement、负载均衡和 All-to-All。",
        "Expert与GPU的映射": "不是固定一张卡一个 expert。八卡六十四 expert 时可以每卡放八个；expert 本身太大时，单个 expert 也可能再做张量并行跨多张卡。层次是 MoE Layer 包含多个 experts，experts 分布到不同 GPU，一个 GPU 可放多个 expert，单个 expert 也可能跨多个 GPU；具体取决于 expert 大小、显存和并行策略。",
        "一个任务有多个通信域": "按并行维度划分。以四卡 TP 等于二、DP 等于二为例，模型切成分片 A 和 B，副本零放 GPU0 和 GPU1，副本一放 GPU2 和 GPU3。TP 通信在同一副本内，即 GPU0 与 GPU1、GPU2 与 GPU3，用于拼接层内计算结果。DP 通信在持有相同分片的卡之间，即 GPU0 与 GPU2 同步分片 A 的梯度、GPU1 与 GPU3 同步分片 B。GPU0 和 GPU1 不能同步梯度，因为它们持有完全不同的参数，相加没有意义。所以四张卡上有四个通信域，每张卡同时属于两个，在不同域里 rank 编号也不同；不能合并成一个四卡大域，否则会把不同分片的梯度混着加。",
        "为什么TP必须同机": "TP 每层都要通信，一个 step 上百次，每次数据量不大但延迟敏感，必须走 NVLink；DP 一个 step 通信一次，数据量大但能容忍稍高延迟，可以跨机走 RDMA。所以调度硬约束是同一 TP 组的卡必须在同一节点且有 NVLink 直连，DP 组可以跨节点但机器之间要有足够 RDMA 带宽。TP 组被拆到两台机器时通信域照样能建、任务照样能跑，但每层都要过网络，训练速度掉一个量级，而且现象只是慢、没有报错，最难排查。"
      },
      "物理层次与进程模型": {
        "层次与互联": "集群下面是节点，节点内多张 GPU 之间是 NVLink 或 NVSwitch，节点之间只能走 InfiniBand 或 RoCE，NVLink 不跨机器。带宽量级上 NVLink 约每秒九百 GB，PCIe 5.0 x16 约一百二十八 GB，IB NDR 四百 Gb 约五十 GB，二十五 G 以太网约三 GB，机内比跨机快一个数量级以上。机内 NCCL 按 NVLink、PCIe P2P、共享内存、socket 的顺序自动选最快通路。bootstrap 阶段即使同机也走 TCP 交换元信息，但真正的数据传输建好后直接走 NVLink。",
        "Pod与rank的数量关系": "典型形态是一个 Pod 占一台机器，六十四卡任务就是八台机器、八个 worker Pod、每个 Pod 内起八个进程，共六十四个进程、六十四张卡、六十四个 rank。一进程一卡一 rank 三者一一对应，而 Pod 是 Kubernetes 的调度单位、rank 是训练框架的并行单位，所以 Pod 数和 rank 数对不上是正常的。也有一个 Pod 一张卡的形态，多见于推理和共享集群；大规模训练更常用一 Pod 一机，因为整机分配天然保证 TP 组落在机内 NVLink 上。",
        "为什么一进程一卡": "技术上一个进程可以用 cudaSetDevice 管多卡，PyTorch 早期的 DataParallel 就是这么做的，已被官方废弃。主要原因是 CPU 侧下发不过来：每个算子都要 CPU 先发一次 kernel launch，一张高端卡满载每秒要接收几万到几十万个 kernel，单个 Python 解释器受 GIL 限制串行下发八张卡，CPU 成为瓶颈，GPU 等着挨饿。改成八个进程后各有独立解释器并行下发。其次是 NCCL 的标准模型就是一 GPU 一 rank 一进程，一进程一卡省去 context 切换，代码保持单卡逻辑，DDP 的梯度同步也更对称。",
        "启动命令": "torchrun --nproc_per_node=8 --nnodes=8 --master_addr=<地址> --master_port=29500 train.py，其中 nproc_per_node 是每台机器的进程数也就是卡数，nnodes 是机器数，两者相乘是 world size。"
      },
      "推理平台的差异": {
        "和训练的核心差别": "训练是离线批处理，目标是吞吐和训练时长，任务有明确起止，失败可以从 checkpoint 重来；推理是在线服务，目标是延迟 SLO 和 QPS，服务长期运行，单个请求失败直接影响用户。所以推理平台关注弹性扩缩容、灰度发布、请求级路由和排队策略，训练平台关注队列配额、gang 调度和断点续训。",
        "推理的资源特点": "推理请求的显存占用随上下文长度变化，KV cache 是主要变量，所以准入控制和 token 预算比静态资源配额更重要。单卡放不下时同样要做 TP 或 PP，但通信域一旦建立就长期不变，不像训练会因节点故障频繁重建。多副本之间没有梯度同步，副本间基本无通信，扩缩容只需增减副本。",
        "故障影响的差别": "训练里一个 rank 挂掉会让同通信域的其他 rank 全部阻塞在集合通信里，整组空转，必须整组重建；推理里一个副本挂掉只影响落到它上面的请求，其他副本继续服务，网关摘除后自动恢复。所以推理的故障隔离粒度是副本，训练是整个通信域。"
      }
    }
  },
  "project_facts": [
    {
      "company": "腾讯",
      "role": "云原生 Infra 实习生",
      "period": "2026-04 ~ 至今",
      "focus": "TCS/TKE 云原生基础设施：异构 GPU 节点自愈控制面、统一公网出口代理组件、KubeVM/QEMU 磁盘 I/O 排查与 Native AIO 自动选择、Nydus RuntimeClass Admission Webhook 适配、本地存储感知调度优化",
      "terminology_translation": {
        "TCS节点": "TCS 私有云平台管理的 Kubernetes 工作负载节点。",
        "unatgw": "平台已有的公网 NAT 出口能力。",
        "COS-CGI": "对象存储相关的公网访问服务。",
        "OriginalHost": "为了兼容已有业务的域名访问方式，让客户端继续使用原公网域名；ClusterIP 入口通过 service-id 导向 Service，LoadBalancer 入口通过受管 DNS A 记录导向 VIP。",
        "NPDPlus": "部署在节点侧的故障检测组件，负责把厂商工具和驱动日志中的异常统一上报为节点状态、事件和监控指标。",
        "NodeCondition": "NodeCondition 是 Kubernetes Node 对象 status.conditions 中的状态条目。本项目用它表达某类 GPU 故障，True 表示故障仍在，明确为 False 并满足恢复条件后才能解除隔离。记录维修过程的是独立的 HealingTask CR。",
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
        "高压追问参考回答": {
          "TCS是什么_为什么说是PaaS平台": "TCS 全称 Tencent Cloud-native Suite，是腾讯面向企业私有化交付的云原生 PaaS 平台。它可以部署在客户的服务器、虚拟化环境或第三方 IaaS 上，把容器管理、应用部署以及按需选用的微服务、中间件等能力提供给业务团队。称为 PaaS，是因为它提供运行和管理应用的平台能力，客户在上面部署自己的业务系统。",
          "你们支撑的是什么业务_什么平台": "TCS 服务企业客户，上层承载企业业务系统、微服务，以及按方案接入的 AI 训练和推理平台；平台可以组合容器、微服务、消息队列和数据库等产品。我接触的基础设施还包括通过 KubeVM 管理的虚拟机，以及异构 GPU 节点。我做的是这些业务共用的 Kubernetes 基础设施，主要包括 GPU 节点自愈、公网出口代理和本地存储调度。",
          "TCS对Kubernetes做了哪些产品化封装": "TCS 把集群、节点、应用、网络、存储、GPU 和监控运维做成可以通过页面、API 和命令行使用的平台能力，同时接入权限管理和组件交付升级。例如我的 GPU 自愈项目，用户在页面选择启用节点、故障处理规则和维修限额，后端把配置写入自愈策略，由控制器执行相应处置。",
          "对业务开发者提供什么能力_能接入自己的开发流程吗": "业务团队可以通过平台部署应用、配置资源和依赖、调整副本并查看运行状态，也可以通过 API 或命令行接入已有的 CI/CD。以我接触的 TAD 应用交付机制为例，开发者提供镜像和部署描述，声明组件依赖及配置，交付系统按依赖顺序部署，并把中间件参数传给应用。代码构建可以接入既有流水线，应用运行和发布由平台提供相应能力。",
          "主要卖软件还是卖服务器和GPU": "主要交付平台软件，并按项目约定提供部署实施和技术支持。TCS 可以使用客户已有的服务器、GPU、网络和存储，也可以部署在既有虚拟化或第三方 IaaS 上，不要求客户购买腾讯的服务器。",
          "把PaaS打包交付给客户_具体交付什么": "交付的是选定版本的平台和组件，包括程序镜像、部署模板、配置与依赖描述，以及安装升级所需的工具和文档。交付人员按客户的硬件、网络和存储环境完成部署与验证，客户随后通过控制台或 API 使用这些能力。容器、微服务、数据库和消息队列按项目需求组合，不必每套环境都安装全部产品。",
          "十家客户是不是共用一个平台管理所有机器": "按私有化交付模式，每家客户有自己的部署环境、集群和权限边界，各自在自己的资源上运行。统一管理可以是同一客户内部管理多个集群、机房或资源池，不意味着把十家客户的机器合成一个共享 Kubernetes 集群。",
          "Kubernetes开源免费_客户为什么还要买TCS": "客户购买的是完成集成、适配并能够持续维护的平台。自行建设时，除了安装 Kubernetes，还要解决网络、存储、权限、监控、中间件和升级兼容，并投入人员长期维护。TCS 把这些能力产品化交付，客户可以利用已有基础设施，把研发精力更多放在业务上；采用哪些组件取决于自身需求和建设成本。",
          "TCS_TKE_TCE分别是什么关系": "TKE 是容器管理产品，也是 TCS 套件的容器平台底座，既有公有云服务，也有私有化形态。TCS 在容器底座上组合应用管理、微服务和中间件等能力，提供云原生 PaaS。TCE 则是覆盖计算、网络、存储及上层云服务的全栈专有云平台；TCS 可以适配已有 IaaS，不要求客户先购买整套 TCE。",
          "你写的组件怎样成为平台交付的一部分": "GPU 自愈以 Addon 组件包交付，包含检测配置、控制器、CRD、权限和厂商模板，并接入平台的配置、查询与通知入口。公网代理的控制器接入 tcs-controller-manager，代理 VM 上的 Nginx 和配置生成程序通过独立 Product 提供安装物料。这样交付时能按版本安装组件，用户通过规则配置使用能力，后续由控制器持续处理配置变化。",
          "用户能直接拿到Kubernetes_API或kubeconfig吗": "Kubernetes API 是否向用户开放，由平台的 RBAC 和租户权限配置决定。账号凭据如何提供、是否直接交付 kubeconfig，需要按具体平台的交付约定确认。",
          "介绍一下你做的两个组件": "第一个是异构 GPU 节点自愈，把不同厂商的故障检测接入统一维修流程。节点侧将 NVIDIA、昇腾、昆仑芯等厂商的故障上报为节点故障状态（NodeCondition），控制器依据自愈策略（NodeHealer）创建维修任务（HealingTask），执行 Cordon、Drain 和必要的重启。最后重新检查本次任务触发的原始故障状态，确认恢复后再把节点放回集群。第二个是统一公网出口代理，让业务通过出口规则（EgressForwardRule）声明允许访问的固定公网目标。Controller 自动创建 Service、管理代理 VM 后端并发布 Nginx 配置，替代原来人工 SSH 维护多台代理 VM 的操作。",
          "你还做过KubeVM存储相关工作吗": "做过一次 KubeVM 数据盘 I/O 异常排查。我先对比 KubeVM 与 CVM 的 launcher Domain XML，拆开 cache、I/O backend、iothread 和 blocksize 等变量，再在 KubeVirt 中扩展 Native AIO 自动选择：最终为 cache=none、磁盘源为 file/device、且用户未显式指定 I/O 模式时选择 io=native，并通过集群级负向开关保留旧行为。代码和测试在我的远程功能分支，但目前没有合入主干或发布分支。",
          "Nydus的RuntimeClass适配是你做的吗": "是，我补充了 Nydus 的 Kubernetes 准入适配。在目标 containerd 1.7 配置中，Pod 已选择 runtimeClassName=nydus 后，Webhook 会自动补齐或纠正 runtime-handler annotation，让 containerd 在拉镜像时选择对应的 Nydus snapshotter，也就是负责准备容器镜像文件系统的组件。我还接通了功能开关，并补充分支测试。",
          "更偏控制层还是调度层": "主要经历在 Kubernetes 控制面，也做过调度插件。GPU 节点自愈改变节点的可用状态，公网代理维护业务网络入口和代理配置。在调度方面，我在已有本地存储插件中实现了跨节点容量评分，支持分散与装箱、多卷需求，以及按 StorageClass 配置，直接参与 Scheduler 的节点选择。",
          "你接触的TCS实际使用哪些CNI": "我在 TCS 环境接触过 Calico 和 Cilium，具体使用哪种方案取决于交付集群的配置。Calico 我主要接触 Pod 网段、节点路由、BGP 和隧道相关路径；Cilium 则主要了解 eBPF 转发、网络策略和 Service 接管配置。CNI 的基本职责是为 Pod 配置网络接口、地址和连接能力，Service 是否由它接管还要单独看数据面配置。",
          "Calico和Cilium都接触过吗_怎么比较优缺点": "两套我都接触过，对比主要集中在配置方式和网络路径。比较 Calico 传统 Linux 数据面与 Cilium eBPF 数据面时，前者可以沿路由、隧道、iptables 和连接跟踪排查，后者还需要查看 eBPF 程序、映射表与 endpoint 状态，并可用 Hubble 观察流量。传统模式需要关注路由和规则规模，eBPF 模式需要关注内核兼容、BPF 状态和升级排障能力。Calico 自身也支持 eBPF 数据面和 kube-proxy 替代，因此具体选型要固定版本与启用模式，再比较业务需要的能力、运维条件和同场景测试结果。",
          "你在多大规模集群写过Operator_遇到过什么问题": "我接触的 TCS Kubernetes 集群约 300～400 个节点，开发的控制器包括公网代理和 GPU 自愈。一个具体问题是，配置已下发到代理 VM，Controller 却在记录结果前重启了。再次处理时要先读取 VM 的实际配置版本，已经更新成功的后端继续复用，尚未完成的后端再推进，避免从头重做整轮发布。",
          "平台组件都在Kubernetes里运行吗": "两种形态都有。公网代理的 Controller 在 Kubernetes 中运行，实际转发流量的 Nginx 部署在独立 Proxy VM 上。GPU 检测组件部署在需要检测的 Kubernetes 节点上。",
          "Underlay的BM_VM和KubeVM分别由谁管理": "普通 Underlay BM、VM 的供应和生命周期由底层基础设施平台管理，加入 Kubernetes 后才以 Node 承载工作负载。公网代理的 Proxy VM 由底层资源平台提供，Controller 使用 VM 地址，通过 SSH 调用远端程序发布配置。KubeVM 通过 KubeVirt 将虚拟机生命周期纳入 Kubernetes，以 VM、VMI 和 virt-launcher Pod 管理。",
          "你在腾讯哪个项目投入时间最长": "统一公网出口代理，前后投入接近三个月，主要完成规则控制、VM 配置发布、失败恢复和交付联调。",
          "你属于哪个小组_小组有多少人": "我所在的 PaaS 平台产品化小组有十几个人，主要做 Kubernetes 及相关基础设施能力。"
        }
      },
      "supplementary_personal_implementations": {
        "Nydus_RuntimeClass_Admission_Webhook": {
          "模块定性": "这是 Kubernetes RuntimeClass 意图到 containerd 1.7 PullImage snapshotter 选择之间的准入桥接器：业务已经显式选择 runtimeClassName=nydus 时，Webhook 再补齐 io.containerd.cri.runtime-handler=nydus。真正的数据面仍由 Nydus 格式镜像、nydus-snapshotter、nydusd、containerd 节点配置以及可选的 Dragonfly P2P 链路完成。",
          "镜像加速背景": {
            "默认OCI路径的问题": "普通 OCI 镜像冷启动通常要先解析 manifest、下载所需 layer 并完成解压展开，镜像越大、节点弹性扩容越集中，Pod 越容易把时间耗在镜像准备阶段；大量节点同时从 Registry 回源，还会放大源站带宽和并发压力。",
            "Nydus解决什么": "Nydus 把 RAFS 文件系统元数据与数据 blob 分离，先取得启动所需的 bootstrap/元数据，再按文件访问以 chunk 粒度取数据，并可配合预取和本地缓存。因此可以在完整镜像下载与展开结束前准备 rootfs，把部分 I/O 延后到运行期读取；冷启动时间与首读成本需要一起比较。",
            "Dragonfly解决什么": "Dragonfly 解决的是大规模分发与回源压力：dfdaemon 作为节点侧 Peer/代理，Scheduler 为下载任务选择 Peer 或 Seed Peer，数据按 piece 在节点间复用，未命中时再回源。Nydus 与 Dragonfly 可以组合成按需取 chunk 加 P2P 分发，但二者是不同层次，Nydus Webhook 本身不包含 Dragonfly 逻辑。",
            "chunk与piece粒度": "Nydus chunk 是 RAFS 中文件数据的寻址和校验单位，Dragonfly piece 是 P2P 传输与缓存调度单位。一次 Nydus Range 读取可映射到一个或多个 Dragonfly piece，二者不是固定一一对应。",
            "Lazyload代价": "Lazyload 是把一部分网络与解压 I/O 从启动前移到容器运行期首读：它能缩短启动关键路径，但未缓存文件第一次访问可能出现延迟抖动，后端不可用时也可能在运行期暴露读取失败；如果业务最终读取几乎整个镜像，总下载量收益会收窄。prefetch 是启动速度、首读稳定性和提前流量之间的权衡。",
            "为什么选择性启用": "Nydus 更适合镜像大、冷启动下载耗时明显，且启动阶段只读取部分镜像内容的业务。我们的适配采用显式选择：Pod 已指定 runtimeClassName=nydus 才处理，普通 Pod 继续使用既有默认运行时和镜像准备路径。启用前还要准备 Nydus 镜像产物，并通过节点全覆盖或 RuntimeClass 的调度约束保证 Pod 落到支持 Nydus 的节点。小镜像或启动时接近全量读取的业务，需要结合启动时间和首读开销决定是否启用。"
          },
          "运行时与分发组件": {
            "RuntimeClass": "Kubernetes 集群级资源。Pod 用 spec.runtimeClassName 引用它，kubelet 查到其中的 handler，并在 RunPodSandboxRequest 中把 runtime handler 传给 CRI；它主要表达运行时选择，不会自动向 Pod 注入任意 containerd 私有 annotation。",
            "runtime_handler_annotation": "Pod annotation io.containerd.cri.runtime-handler=nydus 会随 Pod annotations 进入 CRI PodSandboxConfig。containerd 1.7 的 PullImage 请求本身没有独立 runtime 信息时，会读取这个实验性 annotation，再按 runtime 配置覆盖默认 snapshotter。",
            "runtime_handler不等于更换OCI执行器": "默认 runtime 与 nydus runtime 都可以使用 io.containerd.runc.v2。runc 负责创建和运行容器进程，Nydus 负责准备镜像文件系统与按需读取；runtime handler 在这里选择的是绑定了不同 snapshotter 的 containerd 配置。",
            "Nydus数据面": "containerd 的 runtimes.nydus 需要绑定 nydus snapshotter，proxy_plugins.nydus 需要指向可用 socket，同时启用 snapshot annotations，并避免过早丢弃 unpack 所需内容；nydus-snapshotter 再启动或管理 nydusd，为转换后的 Nydus 镜像提供 RAFS/FUSE 或相应后端的按需读取。",
            "Dragonfly分发面": "可选的 dfdaemon/Seed Peer/Scheduler/Manager 链路负责 piece 级 P2P 分发和回源治理，不负责 Kubernetes RuntimeClass 选择，也不能替代 Nydus snapshotter。"
          },
          "为什么仅有RuntimeClass还不够": {
            "时序差异": "RuntimeClass handler 的标准传递点是 RunPodSandbox；镜像拉取通常发生在创建容器之前。目标 containerd 1.7 实现明确说明 PullImageRequest 没有携带运行时信息，所以 runtime-level snapshotter 选择需要从 PodSandboxConfig annotation 取 runtime handler。",
            "缺少桥接的后果": "Pod 的 Sandbox 后续可能按 nydus handler 创建，但镜像拉取和 unpack 仍可能使用默认 snapshotter，结果可能是退回普通完整拉取、选错镜像准备路径，或因下游配置组合而失败，具体表现取决于节点配置与镜像产物。",
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
          "幂等与Patch行为": {
            "幂等含义": "正确值不重写，缺失或错误值收敛到唯一目标值；Nydus 子模块在目标值正确时不产生 diff，共享 Webhook 的其他 Handler 仍可能生成 Patch。",
            "理论Patch": "annotations 为空时通常增加 annotations 对象；map 已有而 key 缺失时增加目标 key；值错误时替换 value；值正确时 Nydus 部分无 diff。具体 JSON Pointer/op 由通用 jsonpatch 库根据序列化结果生成。",
            "测试覆盖": "现有单测只检查函数执行后内存 Pod 中 annotation 的最终值，没有断言 AdmissionReview 到 AdmissionResponse 的最终 JSONPatch opcode、path 或空 Patch。"
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
            "containerd版本": "上游 Nydus 文档把 runtime-level snapshotter 模式限定为 containerd >=1.7，并要求 sandbox spec 带 runtime-handler annotation；本模块源码也明确写 containerd 1.7。其他版本按其 CRI 接口和 snapshotter 选择方式适配。",
            "与1_6设计稿的冲突": "CRI Runtime Specific Snapshotter 在 containerd v1.7 首次发布，上游 Nydus 文档也要求 >=1.7，stock containerd 1.6.9 缺少对应能力。本地总体设计稿同时保留 1.6.9 基线和 1.7 annotation 示例，两个版本需要分开处理。内部 1.6.9 fork 若做了 backport，核对项包括 fork 源码、节点 config dump 和实际 PullImage 行为。",
            "交付模板差异": "正式交付模板为 Pod 注册 CREATE+UPDATE，旧版 Chart 的 Pod 规则只注册 CREATE。因此正式模板会把 UPDATE 送入处理器，旧版 Chart 只会触发 CREATE。",
            "snapshotter与镜像variant选择": "runtime-handler annotation 解决的是一次 PullImage 该用哪个 runtime-specific snapshotter；Nydus variant 选择解决的是同一 image index 中选 OCI 还是 Nydus manifest。二者相关但不是同一个机制，镜像转换/manifest matcher 问题不能由 Webhook 单独解决。",
            "演进边界": "更新的 CRI RuntimeClass-aware image pull 正在把 runtime handler 变成显式字段；在采用该能力的新版本栈里，这类实验性 annotation 桥接可能不再需要，适配方式随目标 containerd/CRI 接口调整。"
          },
          "怎么确认真正生效": {
            "第一层_准入意图": "检查最终 Pod 同时具有 spec.runtimeClassName=nydus 和 io.containerd.cri.runtime-handler=nydus，确认准入桥接已经完成。",
            "第二层_配置前置": "检查 RuntimeClass nydus 的 handler、调度约束，节点 containerd config dump 中 runtimes.nydus 到 snapshotter 的绑定、proxy_plugins.nydus socket，以及 nydus-snapshotter/nydusd 健康状态。",
            "第三层_实际数据面": "从目标容器反查 containerd 实际 Snapshotter，应看到 nydus，并核对 nydus snapshot 列表；再检查 fuse.nydusd 或目标实现对应的真实挂载、nydusd/snapshotter 日志与读取指标。接入 Dragonfly 时，另外检查 dfdaemon、Scheduler/Peer/Seed Peer 的命中与回源路径。",
            "第四层_加速效果": "在相同节点、相同镜像、冷缓存/热缓存明确的条件下比较镜像准备和 Pod 启动关键时间，并同时观察实际回源字节、P2P/本地缓存命中、首读延迟和运行期 I/O。"
          },
          "个人职责_本人确认": "本人完成 RuntimeClass 到 containerd runtime-handler annotation 的 Admission Webhook 适配。",
          "参考回答": "Nydus 通过按需读取，减少容器启动前必须下载和展开的镜像内容。它使用 RAFS 镜像格式，把文件系统元数据与数据块分开；启动时先准备元数据并挂载文件系统，程序访问文件时再获取对应 chunk 并缓存。这样大镜像可以在只获取启动所需内容后开始运行。代价是未缓存文件的首次读取仍依赖后端，若应用很快读遍整个镜像，收益就会缩小。",
          "实现细节参考": "我在已有准入服务中增加 Pod Handler，开关开启后检查 runtimeClassName 是否为 nydus，命中才处理。目标 io.containerd.cri.runtime-handler 缺失就补齐、错误就纠正、正确就保持，并保留其他 Annotation。共享框架比较修改前后的 Pod，生成 JSON Patch 返回 API Server。这项 Annotation 供目标 containerd 1.7 在 PullImage 时选择对应的 snapshotter；节点 runtime 配置提前部署。我还接通配置和启动开关，并补充边界分支测试。",
          "高压追问": {
            "为什么RuntimeClass不够": "RuntimeClass handler 的标准传递点是 RunPodSandbox，而目标 containerd 1.7 的 PullImageRequest 没有独立运行时信息。containerd 因此从 PodSandboxConfig 的实验性 annotation 取 runtime handler，再选择该 runtime 绑定的 snapshotter；Webhook 补的是这段时序和接口信息差。",
            "为什么不直接让业务写annotation": "业务已经通过 RuntimeClass 表达一次运行时意图，再要求手填 containerd 私有键会形成两个可漂移的配置源。Webhook 以 RuntimeClass 为权威统一补齐和纠正，降低接入成本；代价是硬编码同名契约和共享 Webhook 的影响面。",
            "错值为什么覆盖而不是拒绝": "当前策略把 runtimeClassName=nydus 视为权威意图，错误 annotation 会导致拉镜像 snapshotter 与 Sandbox runtime 不一致，所以直接收敛到 nydus。它没有设计单 Pod opt-out；如需允许自定义 handler，应把映射配置化而不是继续硬编码。",
            "怎么证明真的加速": "在同镜像、同节点且冷热缓存条件一致的对照中，比较镜像准备、Pod 启动时间、首读延迟和回源字节。测试前先从容器反查实际 Snapshotter，并核对 Nydus 挂载和读取，确认请求确实走了目标路径；RuntimeClass、annotation 和 socket 是前置配置检查。",
            "Webhook挂了会怎样": "正式 .tad 模板使用 failurePolicy=Fail，共享 Webhook 不可达或超时会阻断所有未被排除的 Pod CREATE/UPDATE，影响范围超过 Nydus Pod；已运行的 Pod 不会因此自动停止。Nydus Handler 只修改内存对象，共享链上的其他处理器报错也可能拒绝请求。为控制影响，我会配置多副本、健康检查和调用超时，并通过匹配规则排除 Webhook 自身恢复所需的资源。",
            "为什么UPDATE也处理": "UPDATE 与 CREATE 共用最终状态收敛逻辑；只要最终 Pod 仍是 nydus，annotation 被删除或改错就会补回。它不使用 oldPod，也不是事件 diff 逻辑。实际 runtimeClassName 通常在调度后不可改，UPDATE 只保持对象元数据一致性，不会把已经运行的容器重新拉镜像或重新挂载到 Nydus；而且 legacy Chart 没有为 Pod 注册 UPDATE。",
            "1_6和1_7怎么回答": "这段桥接针对 containerd 1.7；stock 1.6.9 缺少 runtime-specific snapshotter 能力。总体设计稿中的 1.6.9 基线与 1.7 annotation 示例需要分开核对。内部 1.6.9 fork 若做了 backport，再通过源码、节点配置和实际 PullImage 行为确认。",
            "模块还能怎么增强": "把 RuntimeClass 名到 handler 的映射配置化或动态解析，并根据 Webhook 顺序选择 reinvocation；如果节点覆盖不完整，再通过 RuntimeClass.scheduling 绑定 Nydus 能力标签，避免 Pod 落到不支持该运行时的节点。",
            "按需加载会不会拖慢模型服务的首个请求": "会有这种可能，尚未缓存的数据在首次读取时仍要等待下载。模型服务如果启动时就要完整读取镜像里的权重，这部分 I/O 仍在启动路径上；可以按实际访问模式预取关键文件，并分别观察容器启动、模型 Ready 和首个请求耗时。",
            "关闭Webhook开关能回滚已有Pod吗": "不能，Handler 列表在 Webhook 启动时构造，改开关需要重启或滚动 Webhook，随后只影响新准入请求。已有 Pod 的 annotation 和节点侧 containerd/Nydus 配置会保留，已经运行的容器也不会因此切换 snapshotter。回退已有工作负载时，要按目标运行时配置重建 Pod，并检查新容器实际使用的 snapshotter。",
            "给指定Namespace的Pod统一注入Annotation怎么做": "我会用 Mutating Admission Webhook，通过 namespaceSelector 匹配目标 Namespace 的标签，只接收这些命名空间的 Pod 请求。处理器补齐目标 Annotation，已有正确值时保持不变，再返回 JSON Patch；如要求固定值，就纠正冲突值。需要防止后续修改移除它时，同时处理 UPDATE。已经存在且没有新请求的 Pod，要额外批量 Patch，注册 Webhook 不会自动遍历补齐它们。",
            "每个组件都部署Webhook会怎样_怎么复用": "多个匹配的 Mutating Webhook 会增加准入调用次数、延迟，以及证书和升级维护成本。相近的修改能力可以共用一个服务，在内部注册多个处理器，我的 Nydus 适配就是这样接入的。不同可用性要求的能力应拆开匹配范围和 failurePolicy，避免辅助注入故障也阻断关键业务创建。",
            "Annotation一般怎么使用_写上就会执行吗": "Annotation 用来给组件传递额外配置或记录信息，具体行为由读取它的组件决定。例如目标 containerd 1.7 会读取 Pod 的 io.containerd.cri.runtime-handler，选择拉镜像使用的 runtime 配置和 snapshotter。没有组件识别的 Annotation 只会作为元数据保存；Namespace 上的 Annotation 也不会自动继承到 Pod。",
            "镜像预热不能解决吗": "能，镜像版本和目标节点可以提前确定时，预热很直接。临时扩容、版本多或调度位置不确定时，完整预热会提前消耗带宽和磁盘，部分内容还可能用不上。Nydus 可以先取实际需要的数据，也能配合关键文件预取。比较时看业务 Ready 时间、首读延迟和实际传输量。"
          }
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
            "直接用户与最终受益者": "直接使用者是 TCS 私有云的平台运维和 GPU 集群管理员，他们按厂商配置节点范围、故障场景、限流和通知。最终受益的是使用 GPU 资源的业务团队：故障处理按统一流程推进，维修确认后可用容量自动回池。",
            "典型业务场景": "典型场景是 GPU 节点持续出现设备丢失、PCIe 链路或驱动异常，并且该故障已有审核过的维修策略。管理员先确认对应业务能够承受中断及恢复，再将节点和故障场景纳入自动维修范围；运行时通过持续时间、维修额度及 Eviction/PDB 控制动作，执行隔离、驱逐、必要的重启和原故障复检。硬件故障未消失或 Drain 受阻时停止并转人工。训练任务还要与上层的任务重试、Checkpoint 和整组恢复机制配合，当前节点控制器不自行判断训练状态是否已安全保存。",
            "为什么需要平台化": "平台化把检测结果、处理策略、限流、执行步骤和恢复标准统一起来。运维配置故障场景与维修规则，Controller 按统一策略执行；节点中断和重启受预置动作模板及容量限制约束。",
            "一次业务使用流程": "管理员在页面选择节点范围、故障类型、持续时间、限流和通知策略。故障满足策略后，系统生成维修任务，隔离节点并通过 Eviction 驱逐允许中断的 Pod，再执行必要的重启和健康复检；替代 Pod 是否重建由上层工作负载控制器负责。处理过程写入历史并发送通知，检测结果不完整或动作失败时由运维介入。",
            "业务价值": "按流程估算，整体处置时间缩短约 60%，主要减少逐节点登录、故障确认和人工操作衔接的时间。厂商适配让昇腾、昆仑芯、沐曦与已有 NVIDIA、海光场景进入统一控制面；平台配置、任务记录和通知减少跨厂商维护与故障跟踪成本。限流、重启结果确认和原始故障复检控制维修节奏，修复后的 GPU 容量自动回到可用池。",
            "业务成功标准": "一次自愈成功不能只看重启命令返回，而要看到原始 GPU 故障明确恢复、节点解除隔离并重新具备承载工作负载的条件，同时维修任务和历史记录完整。训练或推理任务是否恢复到业务 SLO 仍由其工作负载控制器和业务监控确认，不能由 HealingTask 状态替代。",
            "量化口径": "按处置流程估算，整体处置时间缩短约 60%。比较相同厂商、故障类型和任务条件下，从首次明确故障到恢复调度资格的流程，包含持续时间等待、排队、Drain、重启与复检；主要节省人工登录、操作交接和等待人工推进的时间。厂商覆盖由已有 NVIDIA、海光扩展到昇腾、昆仑芯、沐曦，共五类厂商。容量损失按不可用 GPU 数量对时间累计，人工投入按介入次数和操作时长衡量。",
            "实际业务范围": "项目面向 TCS 私有云的异构 GPU 节点，完成故障隔离、受控维修和健康确认后回池。启用节点自动维修前，需要确认所承载业务允许中断及其恢复方式。训练任务的 Checkpoint、整组重试和进度恢复由上层训练系统处理，推理服务还需结合副本容量、预热和请求状态安排维修；训练平台联动属于后续演进。",
            "规模口径": "建设银行 GPU 资源池为 128 台八卡服务器，共 1,024 张 GPU。另有 TCS 集群约 300～400 个 Kubernetes 节点的既有记录；GPU 资源池服务器数与集群 Node 数分别使用，5000 节点用于扩容题设。"
          },
          "一句话": "我把异构 GPU 节点中已确认可处理的故障接入受控自动维修，减少逐节点人工操作，并在原故障明确恢复后让 GPU 容量重新回池。",
          "简历推荐写法": "负责异构 GPU 节点自愈平台化闭环并完成生产交付：复用 NVIDIA/海光检测与 node-healing 已有动作执行能力，补齐昇腾、昆仑芯、沐曦检测及统一 NodeCondition 契约，完善 NodeHealer 限流、HealingTask 可恢复执行与恢复复检；打通 tcs-platform 四个 YunAPI、页面配置与历史查询、tcs-extensions 阶段通知并纳入 TCS Addon/TAD 交付。",
          "参考回答": "异构 GPU 节点自愈已经落地。我在已有框架上补齐厂商检测适配和平台管理能力，支持选择启用自愈的节点、配置故障规则与维修限额，并提供维修记录和状态通知。符合策略的故障按节点维修流程自动处理，恢复检查通过后重新开放调度，减少人工操作衔接。",
          "项目状态": "已完成从五厂商异构检测、页面配置、任务执行、恢复判定、历史通知到 Addon/TAD 的端到端实现，并由我完成组件实现、交付物接线和最终上线交付。",
          "个人贡献边界": {
            "复用部分": "NPDPlus 检测框架、NVIDIA/海光既有检测能力，以及 node-healing 已有的 NodeHealer、MachineHealingTemplate、HealingTask、limiter 和 Cordon/Drain/Reboot 动作执行能力。",
            "本人增量": "异构厂商检测与 Condition 契约、页面配置模型和四个 YunAPI、多 Healer selector/仲裁与单节点单活、恢复判定和失败隔离处理、TCS SSH+boot_id 重启防重、HealingTask 阶段通知、Addon/TAD 跨组件接线。",
            "前端边界": "项目闭环包含配置页面，前端同学负责页面实现；我负责后端 YunAPI、配置模型、NodeHealer 期望状态、节点启用关系和历史/通知契约。"
          },
          "角色与状态所有权": {
            "检测层": "NPDPlus 部署在节点侧，消费不同厂商的 SDK、管理工具和驱动日志，统一输出 NodeCondition（Kubernetes 节点状态条件）、Event 和 Metric，让上层控制面不直接依赖具体厂商工具；它只报告故障事实，不决定是否重启。",
            "策略与派单层": "NodeHealer（节点自愈策略）表达节点范围、故障场景、优先级、限流和恢复模板引用，把运维处理规则从代码中抽离成声明式配置；控制器消费 Condition，完成多 Healer 仲裁并创建 HealingTask。",
            "动作模板层": "MachineHealingTemplate（机器自愈动作模板）定义经过审核的 Cordon（节点隔离）、Drain（Pod 驱逐）、Reboot（节点重启）、UnCordon（解除隔离）等恢复动作，页面或普通用户不能随意组合破坏性操作。",
            "执行层": "HealingTask（节点维修任务）记录一次具体故障处理的节点、故障、动作和执行阶段。节点恢复可能持续较长时间，因此任务状态会持久化，Controller 重启后仍能从原来的进度继续执行隔离、Drain、可选重启和安全解封。",
            "平台配置层": "配置页面与 YunAPI 管理模板入口、候选节点、场景开关、限流、通知和历史，只决定期望配置，不直接判断硬件健康。",
            "历史与通知": "历史页面和通知组件读取 HealingTask 状态，但不参与健康判定；即使消息服务出现故障，也不会阻塞主自愈流程。"
          },
          "链路说明": {
            "链路讲述原则": "问完整自愈流程时，按检测、决策、执行和恢复说明；问其中一步就只讲该步。页面配置、交付和历史通知按对应问题选用。",
            "一句话主线": "NPDPlus 把异构 GPU 故障统一成 NodeCondition → Node Controller 完成多 Healer 匹配、候选仲裁和限流检查 → HealingTask Controller 隔离、驱逐并按模板修复节点 → 原始 Condition 明确为 False 且稳定后解除自愈隔离。",
            "链路概览": [
              "1. 检测事实：NPDPlus 只负责把当前硬件故障持续维护为 NodeCondition=True，Event 和 Metric 用于解释与观测。",
              "2. 决策派单：Node Controller 检查 selector、场景开关和 duration，从多个 `(NodeHealer, Condition)` 候选中稳定选一个，再通过 limiter 与 NDB 创建 HealingTask。",
              "3. 执行修复：HealingTask Controller 先 Cordon、再遵守 PDB 做 Drain，随后按模板执行可选 Reboot；TCS 路径用持久化旧 boot_id 防止结果未知时重复重启。",
              "4. 恢复闭环：重启完成不等于 GPU 恢复，只有触发任务的原始 Condition 存在、明确为 False 且稳定至少一分钟，才 UnCordon 并完成任务；复检结果缺失或不明确时保持隔离。"
            ],
            "完整链路_8阶段": [
              "1. 故障检测：节点侧 NPDPlus 调用厂商管理工具并读取驱动日志，把需要持续表达的故障写成 NodeCondition=True；某个可恢复 Condition 只有在对应检查真实执行、设备覆盖完整且未再观察到故障时，才允许改为 False。",
              "2. 策略匹配：Node Controller 遍历 selector 命中的全部 NodeHealer；只有 Condition 类型匹配、场景已启用且基于 LastTransitionTime 计算的持续时间达到 duration，才形成候选。",
              "3. 仲裁与互斥：先检查该节点是否已经有还没有结束的 HealingTask；如果有，就继续原来的任务，不再创建新任务。只有该节点不存在尚未结束的任务时，才按 priority 选择最高优先级，再按 HealerName、ConditionType 稳定排序，每轮只选一个候选。",
              "4. 执行前检查：只有最终选中的策略进入所属 Healer 的 limiter，检查并发、小时、天、节点冷却和 NDB；其他策略不占额度。最高优先级策略未通过检查时等待重试，不改为执行第二名。",
              "5. 记录本次维修内容：Controller 从 MachineHealingTemplate 展开受控步骤并创建 HealingTask，把 nodeName、来源 Healer、触发 Condition、动作和参数写进任务；真正执行前，再检查同一节点是否还有其他没有结束的任务。",
              "6. 隔离与驱逐：先添加自愈专用 NoSchedule taint，再按 Kubernetes Eviction 语义 Drain 并遵守 PDB；Pod 的重建和调度由工作负载控制器与 Scheduler 完成。",
              "7. 修复与动作确认：按模板执行可选 Reboot。CVM 路径观察实例操作；TCS 路径先持久化旧 boot_id，再通过受控 SSH 下发 reboot，SSH 断开后观察 boot_id 变化而不盲目重发。",
              "8. 恢复与解封：Node Controller 检查 HealingTask 中记录的原始 Condition，要求它仍然存在、明确为 False 且稳定至少一分钟，随后写 PhaseRecovered；执行器完成 UnCordon 后进入 PhaseCompleted，其他仍为 True 的故障再参加下一轮仲裁。"
            ],
            "失败后怎么处理": "任何已经进入隔离阶段却没有获得明确恢复证据的任务，默认保持节点隔离并转人工；Controller 重启后会读取还没有结束的 HealingTask、节点和云实例的当前状态以及已经占用的 limiter 额度，从原来的进度继续处理，不会从第一步重新执行所有动作。"
          },
          "失败与补偿闭环": {
            "配置保存部分失败": "ApplyNodeHealer 先保存 NodeHealer，再核对多台节点的启用关系，跨对象没有事务。任何一步失败都向调用方返回错误；调用方用同一份完整节点选择集重试时，接口重新比较需要增加或移除的关系，直到实际配置与提交配置一致。部分成功通过幂等重试逐步收敛。",
            "检测证据不足": "管理工具缺失、字段解析失败、只检查到部分 GPU 或检测器未运行都不能当作健康；只有本轮检查真实完成并覆盖目标设备，才允许把可恢复 Condition 改为 False。",
            "未达持续时间": "继续等待剩余 duration，避免一次采样抖动直接触发 Drain 或重启。",
            "限流或NDB拒绝": "最高优先级候选延迟一分钟后重试，不绕过它执行低优先级候选；NDB 无法确认剩余容量时停止继续维修，避免检测系统异常反而触发批量操作。",
            "重复任务": "Node Controller 创建任务前会查询同一节点是否已有还没有结束的任务，HealingTask Controller 执行动作前还会再检查一次；已有任务就继续原来的流程或等待它结束，不启动第二条维修流程。前一次检查避免正常派单产生重复任务，后一次检查挡住绕过 Node Controller 创建的任务并发操作节点；List→Create 提供前后两层保护，严格原子唯一还需要唯一锁对象或 resourceVersion CAS。",
            "创建HealingTask失败": "释放刚占用的并发额度，小时/天窗口保留已发生的维修尝试，避免靠重复失败绕过累计限额。",
            "TCS重启结果未知": "SSH 在 reboot 后断开时，命令可能已经执行，先保留并核对节点注解中持久化的旧 boot_id。boot_id 变化才确认主机发生重启；旧值未变也可能是重启尚未完成，因此结果未知时继续观察，不立即重发。",
            "Controller重启": "Controller 重启后，以还没有结束的 HealingTask 记录为恢复依据，找回原始 Condition、这个任务已经占用的并发额度，并根据任务时间恢复仍处于小时、天和节点冷却窗口内的限流记录；先从原来的进度继续已有任务，再处理新的候选。",
            "动作或恢复超时": "节点仍无健康证据时默认保持隔离；如果节点随后恢复且模板明确允许 UnCordon，系统才自动解除隔离，否则转人工处理。",
            "通知失败": "通知独立于主状态机做有限重试和去重；可能重复也可能最终漏发，但不会回滚或阻塞自愈主链。"
          },
          "高压追问参考回答": {
            "这个能力直接给谁用_最终服务谁": "直接给 TCS 私有云的平台运维和 GPU 集群管理员使用，他们配置节点范围、允许自动处理的故障、维修额度和通知。最终帮助 GPU 业务团队减少人工处置环节，并在维修确认后重新获得可用容量。",
            "为什么有告警还不够_一定要做自愈": "告警之后，故障确认、节点隔离、工作负载驱逐、修复和恢复复查仍需要有人逐步执行，标准故障也会卡在人工接手和操作衔接上。自愈把已有明确策略、业务能够承受中断的场景自动串起来，减少这部分人工依赖，并用统一恢复标准决定何时回池。硬件需要更换、故障性质不明或业务无法迁移时，告警和人工处置仍然是正确路径，不应强行自动重启。",
            "这个项目的业务价值怎么衡量_有没有数据": "按流程估算，整体处置时间缩短约 60%，主要减少人工登录节点、故障确认和操作交接的时间。接入范围从已有 NVIDIA、海光扩展到昇腾、昆仑芯、沐曦，五类厂商进入统一自愈流程。运维可以统一配置策略、查看维修任务和历史通知，符合策略的故障自动推进，维修后按原始故障复检结果恢复调度资格。容量侧可以按不可用 GPU 数量和持续时间计算损失，人工投入按介入次数与操作时长衡量。",
            "自动自愈会不会反而扩大训练任务损失": "会，驱逐一个仍在运行的训练 worker，同通信域的其他 rank 会全部阻塞在集合通信里，整组任务失败，丢掉最近一次 Checkpoint 之后的进度。当前支持业务排空确认后再维修：自愈侧先隔离节点，等待业务停止相关进程并在节点上设置约定标签，再继续重启。\n这里不依赖 PDB，因为 PDB 有两层不够用。一是语义对不上，它表达的是同时不可用的副本数，而训练任务的可用性是全有全无的，八个 worker 少一个不是八分之七可用，而是整组卡住，minAvailable 这种表达没有意义。二是训练 Pod 通常由 Job 或 PyTorchJob 创建，往往没有关联 PDB，Eviction 请求会直接成功。所以真正的闸门只有业务确认这一道。\n驱逐之后损失多大，取决于业务侧的 Checkpoint 频率和训练 Operator 的重试策略，这部分自愈组件不参与也拿不到反馈；训练平台自动协调 Checkpoint 和任务恢复的接入还在规划中。",
            "节点上有训练作业时怎么处理_等待是默认行为吗": "不是默认行为，取决于策略配置。落地的机制是业务主动排空确认：自愈侧先给节点打上隔离 taint 阻止新 Pod 进来，然后停在 Cordon 这一步，等业务在节点上设置约定的排空标签，收到确认才继续 Drain 和 Reboot；拿不到确认就一直等，超时后任务失败收口、节点保持隔离并通知人工。\n这道等待由 cordonByUserSelf 开关控制，只有开启的策略才会停下来等，未开启的会直接进入 Drain；Drain 那一步也有同样的 drainByUserSelf 握手开关。所以被问到业务不配合怎么办，答案是看该 NodeHealer 怎么配的。\n还没有自动接入的是 Checkpoint 保存和整组停止，业务要在确认前自己处理：能继续跑的先保存状态，已失败的按已有 Checkpoint 安排恢复。自愈侧和训练 Operator 之间没有通知机制，这是后续演进方向。",
            "你个人到底做了什么": "我补齐了昇腾、昆仑芯、沐曦检测适配与统一 Condition，完善多 Healer 仲裁、恢复判定和失败处理，并接入 TCS boot_id 重启结果确认。平台侧完成四个 YunAPI、阶段通知和 Addon/TAD 接线。底层复用 NPDPlus、NVIDIA/海光检测和 node-healing 已有 CRD 与动作执行能力；配置页面由前端同学实现，我负责后端和配置契约。",
            "你负责整个healing_engine还是平台编排控制面": "我主要负责异构 GPU 场景的平台编排控制面，以及已有自愈执行链路的增强。已有 node-healing 提供策略、动作模板、维修任务和限流，以及 Cordon、Drain、Reboot、UnCordon 的基本执行框架。我的工作包括厂商检测接入和统一节点故障状态（NodeCondition），以及多个自愈策略（NodeHealer）的协调和恢复分支增强。执行侧补了 TCS SSH 重启和 boot_id 结果确认，平台侧接入页面需要的 YunAPI、历史通知，以及 Addon/TAD 交付。",
            "请讲一次完整的GPU自愈过程": "用海光的一条 PCIe 链路丢失说明：内核日志含 XID160 和 PCIe link lost，解析设备 BDF 后映射为 HygonDCUPCIeLinkLost=True。故障持续达到阈值并通过策略仲裁和限流后创建 HealingTask，再按模板隔离、排空和重启；恢复必须检查原始 Condition 明确 False 且稳定，才能回池。",
            "为什么选择Kubernetes_CRD和Controller_而不是自己做任务系统": "因为故障事实和主要动作都位于 Kubernetes：输入是 NodeCondition，隔离和 Drain 要操作 Node、Pod 与 PDB，恢复后还要重新开放调度。CRD 提供持久化的 Spec/Status，Controller 提供事件驱动、乐观并发和持续调和，使任务在进程重启后仍能从原来的进度继续。它不会自动提供跨对象事务或业务唯一性，所以单节点互斥、外部动作防重和限流仍由项目自己设计。",
            "为什么不用旧NodeRemediation链路": "旧模型更偏一个场景一组 Remediation/Operation/Job，多厂商多故障会让对象数量、页面拼装和限流归属变复杂。新模型让一个 NodeHealer 聚合某厂商的节点范围、多个 Condition 和独立限流，MachineHealingTemplate 管动作，HealingTask 记录每次维修的执行内容和恢复状态。",
            "为什么策略模板任务要拆成三个CR": "因为它们分别解决策略、动作定义和执行实例三个问题。NodeHealer 描述哪些节点和故障适用、优先级、限流及引用哪个模板；MachineHealingTemplate 保存经过审核的 Cordon、Drain、Reboot、UnCordon 动作；HealingTask 则记录某台节点这一次维修的目标、来源 Condition、动作步骤和执行状态。拆开后模板可以复用，策略变更不会改写正在执行的任务，每次维修也能留下独立历史。",
            "为什么一定要有HealingTask_Controller直接收到Condition后处理不行吗": "可以直接由 Controller 执行动作，但每次维修的目标和进度仍需要持久化。我们用记录一次维修的 HealingTask 保存这些信息，让 Controller 重启后能够接着处理。比如重启请求已经发出、结果还没记录时，恢复后的 Controller 先读取任务里的步骤和保存的旧 boot_id，再核对节点实际状态，决定等待还是继续执行。单独保存任务也方便查询维修历史和失败位置。",
            "HealingTask的reconcile主流程是什么": "HealingTask Controller 的 reconcile 会读取所有尚未结束的任务，并确保对应的执行 handler 已经启动；handler 按任务中记录的步骤顺序执行 Cordon、Drain、可选 Reboot，随后等待健康复检，同时持续写入阶段和步骤结果。这个实现会由 handler 连续推进多个动作。进程重启后会重新启动这些任务的执行流程，因此普通步骤要能重复检查当前状态并继续完成目标；Reboot 这类破坏性动作则要先查询实例操作或比较 boot_id，避免没有确认上次结果就再次执行。",
            "HealingTask状态机为什么不能只有Pending_Running_Success_Failed": "因为节点自愈存在“动作已经执行但健康尚未确认”和“健康已确认但节点尚未解封”两类关键中间状态。任务需要记录原始 Condition、步骤结果以及 Processing、Recovering、Recovered、Completed/Failed 等阶段，才能区分是在执行动作、等待复检，还是等待 UnCordon。进程重启后还要结合任务 Spec 和外部实际状态，从原来的进度继续处理，不能只凭一个 Success 字段跳过动作。",
            "自愈系统怎么保证幂等": "普通动作先查当前状态：隔离污点已经存在就不重复添加，目标 Pod 已退出就不再驱逐。重启需要单独处理，因为请求超时也可能已经执行；TCS 路径先持久化旧 boot_id，再下发命令，结果不确定时先观察重启结果。boot_id 变化可以确认已经重启，旧值没变还可能是重启尚未完成，不能据此立即重发。Controller 重启后沿用已记录的维修进度，重新核对实际结果。",
            "HealingTask状态和实际节点状态冲突时信谁": "以维修任务（HealingTask）的 Spec 确认目标，以外部实际状态确认进度，再修正 Status 或继续执行剩余动作。Spec 记录本次维修需要达到的目标和执行的动作，Status 记录 Controller 已观察到的流程进度。Node、Pod、云实例、boot_id 和故障 Condition 用来判断动作是否真正完成。例如 Status 已写成成功，但 GPU Condition 仍为 True，就仍然不满足解除隔离的条件。",
            "怎么发现节点故障_Controller自己轮询硬件吗": "节点上的 NPDPlus 负责采集厂商工具、SDK 或驱动日志，并通过 Kubernetes API 把持续故障写到 Node 的 status.conditions。Node Controller 通过 Informer 观察 Node 状态，再结合策略和持续时间决定是否创建 HealingTask。原始硬件指标不会直接推给维修 Controller，厂商检测与节点维修在这层状态契约上解耦。",
            "XID故障是不是看到就重启": "不是。日志中的 XID 先由检测层按故障语义映射为 Event 或持续 Condition；即使形成 Condition，也不等于立刻重启。NodeHealer 还会依据场景开关、持续时间、优先级、动作模板、限流和 NDB 决定是否维修。XID 是故障输入，Condition 是持续事实，NodeHealer 才是维修策略。",
            "所以你们主要就是重启对吧": "对，当前自动修复主要靠节点 Reboot，控制面负责限制它什么时候能执行、一次能执行多少。已有维修任务就接着处理；新任务要通过故障持续时间和维修额度检查，排空失败就停止，不继续重启。重启后还要检查本次任务记录的原始故障 Condition，明确为 False 并满足至少一分钟的状态稳定时间，才进入解除隔离流程。",
            "为什么没有把单卡Reset和驱动重载纳入自动自愈链路": "当前优先落地的是多厂商节点自愈，单卡 Reset 和驱动重载还缺少设备占用、互联拓扑及适用型号的完整检查。比如部分互联配置下 Reset 会影响关联 GPU，驱动重载也可能影响同节点所有使用驱动的进程，不能只凭一张卡报错就执行。现阶段复用节点排空和 Reboot，减少需要分别适配的恢复动作。代价是维修影响整台节点，所以必须先满足业务排空条件，并限制同时维修的数量。",
            "五个厂商都是你从零写的吗": "NVIDIA 和海光复用了既有 NPDPlus 能力；我补齐昇腾、昆仑芯、沐曦适配，并把五家输出统一到 Condition/Event/Metric 契约，再接入同一套派单、恢复、页面配置和交付链。",
            "国产卡实际做过哪些工作_能举一个操作例子吗": "我实际做的是国产卡故障检测适配和自愈联调，新增的重点是昇腾、昆仑芯和沐曦，NVIDIA、海光复用已有检测能力。具体工作是读取设备清单和健康输出、解析错误字段，再验证它们能否形成正确的故障状态。比如沐曦先通过 mx-smi -L 对齐卡号和 PCI 地址，再核对温度、ECC、PCIe 和互联查询是否覆盖这些设备；少返回一张卡时，检测结果不能把整台节点判成健康。联调还会检查故障未恢复时是否继续隔离，以及恢复检查通过后任务能否继续推进。",
            "这个项目让我对GPU异常深入到了哪一层": "我主要深入到厂商状态采集、故障判断和恢复条件这一层。比如沐曦节点有多张卡，某个查询只返回其中一张卡的正常结果时，不能清除整台节点的故障。我先取得本轮设备列表，再核对各检测结果有没有覆盖这些设备，把缺结果、查询失败和明确故障分别处理。形成 NodeCondition 时保留设备定位信息，让后续能查到是哪张卡、哪类异常触发维修。",
            "故障语义映射是不是简单的错误码翻译": "错误码解析只是其中一步，还要判断这次结果能不能代表设备当前健康。比如两张卡只返回一张卡的正常 ECC 结果，直接翻译成“ECC 正常”就会误清故障。我补的是设备覆盖校验和异常分类：查询完整且正常时才能按规则清除故障，缺结果或解析失败时保留检测不确定性，明确异常则映射到对应的 NodeCondition，并保留设备 ID。",
            "具体看什么指标_沐曦_NVIDIA和海光有哪些不同": "我主要看设备是否完整且可访问、温度、ECC，以及 PCIe 和卡间互联状态。不同厂商的差别落在采集入口和字段含义上：NVIDIA 通过 NVML 取数；沐曦、海光这两条适配路径解析管理工具输出。比如温度，沐曦解析多个测温点后取最高值，海光优先取 junction，没有该字段再取 edge。ECC 要区分可纠正与不可纠正、累计值与增量，不能只找一个名字相似的字段就套同样判断。每项结果还要关联到具体设备，确认本轮查询成功且覆盖完整。",
            "海光ECC输出是什么样_为什么不能直接复用NVIDIA解析": "海光这条路径按硬件模块读取 ue 和 ce，分别对应不可纠正与可纠正错误。已有真机样本中，一行是 HCU[0] : Block umc is: ENABLED (ue: 0, ce: 0)，没有 uncorrectable 这个完整单词，按该单词匹配就会漏掉。现有解析器逐个读取 ENABLED 模块的计数并累加，跳过 DISABLED 模块。对应模拟异常测试把 umc、gfx、hdp 的 ue 分别设成 3、2、1，断言合计为 6；ce 合计为 17。",
            "阈值怎么定_不同厂商能用同一套高温判断吗": "阈值由检测配置提供，调整时要对应具体型号、测温点和厂商要求，不能把一个温度值套给所有卡。现有 NVIDIA 和海光检测代码的默认温度阈值都是 90℃，这是组件默认值，并不代表所有型号的硬件极限。两条路径的持续性判断也不同：NVIDIA 在统计窗口内至少取得两次有效采样，达到阈值的样本严格过半才判高温；海光在成功查询的温度样本中连续两次达到阈值才报错，正常读数会清零计数。检测形成 Condition 后，NodeHealer 还会检查故障持续时间，再决定是否创建维修任务。",
            "温度只超标一次会写Condition吗_窗口过半与连续超温有什么区别": "以当前 NVIDIA 温度检测代码为例：统计窗口内至少取得 2 次有效采样，温度达到或超过阈值的样本数严格超过总样本数的一半，才判定高温故障。这里判断的是窗口内比例，不要求每次采样都连续超温。\n示例假设：阈值配置为 85℃，它只是演示值；此前没有高温故障和历史样本，同一张卡的以下采样均成功且仍在同一统计窗口内。\n① 第一次得到 [90]：只有 1 次有效采样，未达到至少 2 次的门槛，暂不因这个样本把高温 Condition 置为 True。\n② 第二次后得到 [90, 80]：2 次采样中只有 1 次达到阈值，占比 1/2，没有严格超过一半，仍不判高温故障。\n③ 第三次后得到 [90, 80, 90]：3 次采样中有 2 次达到阈值，占比 2/3，超过一半，判定高温故障并交由上报链路更新对应 Condition。中间出现过正常读数，仍可满足规则。\n这不是固定要求采样 3 次，也不要求等待完整窗口时长；例如窗口内已有 2 次有效采样且都达到阈值，就满足条件。过期样本会移出窗口。首次采样不触发的例子也不能用于推断已有故障已经恢复。\n检测形成 Condition=True 后，NodeHealer 还会检查故障持续时间、开关和维修额度。该窗口规则属于当前 NVIDIA 检测实现，不能推广为所有厂商或原生 NPD 的固定规则；海光的连续超温判断见相邻的阈值问答。",
            "PCIe这边具体监控哪些": "PCIe 这边，我主要检查设备能不能正常查询，以及管理工具报告的 PCIe 错误。比如沐曦通过 mx-smi 查询事件，解析 aer_ce 和 aer_ue，分别是可纠正和不可纠正的 PCIe 错误计数。海光会通过 hycu-smi --showbus 查询设备的 PCIe 信息，检查输出里有没有 error、n/a 等异常。检测结果会关联到具体设备，保留设备 ID 或设备地址，方便继续定位是哪张卡出了问题。",
            "同样是PCIe掉卡_不同厂商的故障信息和处理有什么区别": "差别主要在故障证据和恢复检查，节点维修动作可以复用。NVIDIA 的 XID 79 表示驱动已经无法通过 PCIe 访问 GPU；海光的 XID 160 日志带有 PCIe link lost 和设备地址；沐曦则结合设备列表中的可访问状态与 PCIe 事件判断。检测器分别解析这些信号，形成对应厂商的故障 Condition，由策略选择是否隔离、排空和尝试重启。重启后要用对应厂商的查询确认设备重新可访问，并复检原故障；如果仍然失联，就保持隔离继续排查链路或硬件。",
            "NVLink_HCCS_MetaXLink具体看什么_只有卡还在够吗": "卡能被枚举出来，还要单独检查卡间链路。NVIDIA 通过 NVML 按 GPU 和 Link 编号读取是否激活，再读取 CRC Flit、CRC Data 错误计数。实现里每轮检查部分链路，保留上次计数，按统计窗口判断新增错误是否超过阈值，异常结果记录 GPU 和链路编号。昇腾通过 npu-smi 的 HCCS 查询读取 hccs health status 的 OK/NOK，以及 hccs error count 中各 lane 的计数；当前解析会汇总这些计数，没有把 first error lane 单独做成状态字段。沐曦通过 mx-smi mxlk --show-state 解析端口的 up/down、enabled/disabled，并把结果关联到对应设备。部署时还要确认哪些链路按实际拓扑应当启用，避免把未连接端口和故障链路混在一起。",
            "用一个例子讲清楚_NVML信号怎么变成NodeCondition": "以我们复用的 NVIDIA 高温检测为例，检测器先调用 NVML 读取温度，确认接口成功后，再按阈值和时间窗口判断。假设配置阈值为 85℃、窗口为一分钟，GPU 0 原先没有高温故障，窗口内依次成功采样为 [80, 89, 90]℃；代码要求至少两次采样，且达到或超过阈值的样本严格过半，所以第三次达到 2/3 时判为持续高温，不要求先等满一分钟。随后生成检测器内部的故障原因 GPUTempError，匹配 reason=GPUTempError、condition=GPUTempError 的 permanent 规则，并通过 Kubernetes API 写入 Node.status.conditions：Type 为 GPUTempError，Status 为 True，Reason 为 GPUTempError，Message 保留 GPU 编号、温度和阈值。这里 True 表示高温故障成立，permanent 表示维护持续状态，不代表硬件永久损坏；读取失败不能当作高温或恢复正常的证据。Kubernetes Event 与 Condition 是两类上报输出，不需要先生成 Event 再转成 Condition；上层 NodeHealer 再按策略决定是否维修。",
            "各厂商GPU故障具体有什么差异_为什么需要分别适配": "各家 GPU 的故障大类相通，常见的有掉卡、ECC、过热和互联异常；适配差异主要在三个层面。①故障编号和报告格式：NVIDIA XID 79 表示驱动无法通过 PCIe 访问 GPU；海光 XID 160 日志会带 PCIe link lost 和设备地址；沐曦则要结合设备可访问状态和 PCIe 事件判断。相似的错误编号不能跨厂商直接套用。②采集入口：当前适配中，NVIDIA 直接调用 NVML API；昇腾、昆仑芯、沐曦和海光通过各自管理命令获取输出，再由适配代码解析。API 返回结构化字段和调用状态；CLI 适配还要处理命令执行失败、空输出、格式变化和设备身份对应。③字段语义和计数规则：NVIDIA ECC 的 DBE 是双比特错误，沐曦 --show-event 中的 dbe 却是 PCI device base error；沐曦温度取多个测温点中的最高值，海光优先取 junction、缺失时回退到 edge。NVIDIA 有驱动重载后清零的 ECC 计数，也有跨重启保留的累计计数；沐曦 ECC 计数重启会清零，坏页计数则保留。检测器需先确认采集成功且覆盖预期设备，再按厂商语义解析、聚合和判断阈值，最终映射成统一的 NodeCondition/Event，供自愈策略使用。",
            "详细讲讲你怎么做异构GPU故障语义标准化_举几个例子": "我主要在厂商检测器里补 Go 解析和异常判断代码，再配置故障与 NodeCondition 的对应关系。NPDPlus 按周期查询设备状态，NVIDIA 这条路径复用 NVML 接口，国产卡按各自管理工具取数；比如沐曦由 Go 程序执行 mx-smi，读取表格或按行列出的字段。我的解析代码先从 mx-smi -L 建立设备清单，再把 ECC 计数、PCIe 事件等结果关联到对应设备，按各字段的含义和检测规则判断异常；少返回一张卡的数据时，不把缺失结果当作正常。驱动日志走日志解析路径，例如项目里的海光日志，解析器先提取 XID 编号、设备地址和错误信息，整理成 DCU-Xid 160, Device 0000:3b:00.0, Info PCIe link lost 这样的文本，供故障规则匹配。映射配置明确指定：匹配到 DCU-Xid 160，就对应 HygonDCUPCIeLinkLost，监控器将该 Condition 置为 True，并填入故障原因和描述。最后复用 NPDPlus 的上报模块，通过 Kubernetes API 写入 Node.status.conditions。各厂商统一使用这套状态结构，设备丢失、ECC 和 PCIe 异常仍分别维护不同的 Condition，类型名也可以保留厂商标识。",
            "你能判断哪些故障可自动恢复吗": "当前可以自动尝试的是策略允许的节点 Reboot，例如 PCIe 链路丢失后，完成隔离和业务排空，再尝试恢复设备访问。高温示例模板先排空和检查散热，没有 Reboot；不可纠正 ECC 按现有策略先隔离并上报，由交付继续排查。需要断电重上电、修复物理链路或换卡的故障转人工。重启是否有效还要复检原始故障，boot_id 变化只能证明机器重启过；设备仍然异常时继续隔离。",
            "可纠正和不可纠正ECC分别怎么处理_什么时候重启_什么时候转工单": "单次可纠正 ECC 通常不会让训练失败，硬件已经纠正了这次数据错误。检测时看错误增量和持续时间；例如我适配的昇腾样本中，计数从 10 增到 12、低于高频阈值时只生成 Event，不置持续故障。达到持续故障规则并满足维修条件后，再由策略创建任务、隔离排空并尝试重启。不可纠正 ECC 需要及时隔离、保留设备和错误信息，按处置规则通知运维进一步诊断；自动维修后仍异常时继续保持隔离。",
            "故障语义标准化和硬件RCA有什么区别": "故障语义标准化回答“哪个节点或设备出现了哪类可操作异常，以及控制面下一步怎么安全处置”；硬件 RCA 要进一步回答“为什么发生，是芯片、显存、PCIe root port、riser、主板、供电、固件还是环境导致”。前者是我这个组件的核心工作，能够做到设备/链路/错误类型级定位；后者通常需要更多现场日志、拓扑、AER、厂商黑匣子信息和物理检查，不应仅凭 NodeCondition 下结论。",
            "讲一个具体厂商适配_不要只列XID_ECC_PCIe": "做昇腾适配时，我把几种容易混淆的输出拆开处理。枚举从预期 8 张变成 7 张，测试要求连续三次才报 DeviceLost；但 npu-smi 根本没找到，不能也当成少卡，它进入检测源/驱动错误分支，不报设备丢失或 ECC 故障。另一个样本是可纠正 ECC 从 10 增到 12，低于高频阈值就只生成 Event，不置持续故障。对应代码给这些输入分别写断言，让异常类型和后续动作强度有依据。",
            "没有真实坏卡_怎么验证检测规则是否有效": "检测判断可以用保存的工具输出和构造的异常样本验证。比如海光测试保留了正常温度与 RAS 输出，再用模拟计数检查解析结果；昇腾测试把设备数设成 8 到 7，检查前两次不报掉卡、第三次才报。另一个用例让 npu-smi 不存在，断言应报检测源或驱动异常，而不报设备丢失和 ECC。恢复测试则检查可恢复的高温、健康异常能清除，不可纠正 ECC 保持故障状态；这些用例验证的是检测逻辑。",
            "昇腾适配具体看什么": "昇腾先通过 npu-smi info 和 info -m 对齐设备编号、Chip ID 与 PCI 地址，再分别查询健康、温度、ECC、HCCS 和驱动状态。温度解析只读取温度字段，例如测试样本同时带有 NPU ID 3、Chip ID 0 和温度 48℃，断言要求把 48℃归到设备 3，不能抓到前面的编号。HCCS 检查健康状态和链路错误计数，驱动查询保留 Compatibility 等结果。每项检查还要确认对应设备确实返回了有效字段，才能参与故障或恢复判断。",
            "昆仑芯适配具体看什么": "昆仑芯适配先建立设备清单，把设备 ID 和 PCI 地址对应起来，再读取温度、ECC、驱动及 XPU-RT 状态。xpu-smi -m 的机器可读输出同时有 board_id 和 dev_id，这条解析路径使用 dev_id 作为设备索引。补充结果只带 PCI 地址时，已知地址先映射回清单里的设备 ID，避免卡号不连续或输出乱序时把计数记到另一张卡。设备缺失、不可纠正 ECC 和驱动异常分别形成对应故障状态；某项查询失败时保留本轮未取得有效结果的信息。",
            "多卡查询顺序变化或卡号不连续_怎么避免把故障记错卡": "设备身份用设备 ID 和 PCI 地址对应，输出位置只表示这一条记录排在哪里。昆仑芯适配建立了 PCI 地址到设备 ID 的映射，补充查询中的已知地址先回到这份映射里找设备。假设两张卡的 ID 是 3 和 7，即使设备 7 的结果先返回，也应写到设备 7，不能按第一条结果写到设备 0。昇腾的单设备查询还会按请求的设备编号归并结果，只保留本次请求的设备记录。",
            "沐曦适配具体看什么": "我这条沐曦适配路径先通过 mx-smi -L 建立设备清单，读取卡号、PCI 地址和可访问状态，再解析温度、ECC、PCIe、MMIO 与 MetaXLink 查询。设备缺失、PCIe 错误、MMIO 错误和 MetaXLink 异常分别形成对应 Condition。这里特别处理查询覆盖不完整的问题：设备列表有八张卡，某项检查只返回七张时，第八张记为未取得有效结果，不默认填零或清除原故障。",
            "工具能列出卡_为什么还会判设备不可用": "设备清单里有这张卡，还要看它的 Available 状态。沐曦解析样例中，GPU#100 带有完整 PCI 地址，同时返回 Not Available(Driver And Vbios Compatibility Check Failed)。解析器保留这张卡的编号、地址和不可用原因，设备数量没有减少也不能把它判为健康。这类输出先提示核对驱动与 VBIOS 配套，不能直接断定卡本体损坏。",
            "三个新增厂商适配真正共用的框架是什么": "共用一套检测契约：先完成设备枚举，再执行各检查项；每个检查项记录是否真正执行和覆盖了哪些设备；原始字段映射成稳定的 Condition Type、Reason 和 Message；只有检查完成且无异常才允许恢复为 False。厂商差异留在命令、解析器和错误映射层，NodeHealer、HealingTask、限流与恢复状态机不感知 npu-smi、xpu-smi 或 mx-smi。",
            "Condition的Type_Reason_Message和时间怎么设计": "Type 要稳定表达上层策略真正关心的故障语义，例如 `HygonDCUPCIeLinkLost`；Reason 表达本次转换原因，例如 `DCUPCIeLinkLostDetected`；Message 保存可排障但不适合参与机器判断的说明。LastTransitionTime 由状态实际变化驱动，NodeHealer 用它计算故障持续时间，恢复侧也用它计算连续健康时间。不要把设备地址等高基数字段塞进 Type；这些信息更适合 Message、Event 或 Metric 标签。",
            "怎么区分厂商工具失败_驱动失败_设备真实故障": "先分证据层。命令不存在、超时或返回码异常，说明检测链路失败；驱动模块、设备节点或管理服务不可用，说明驱动/运行时异常；命令成功且枚举到设备后，温度、RAS、PCIe、设备数量等字段异常，才是设备健康证据。三类问题应映射成不同 Condition 或检测状态，不能把 CLI 没输出直接解释成设备健康，也不能把检测器自身故障误报成某张卡硬件损坏。",
            "检测命令超时或卡死怎么办": "厂商检测命令已统一通过 `CommandContext` 设置超时；超时后终止子进程，限制输出大小，记录命令、设备索引、耗时和 stderr，并把本轮检查标记为未完成。检测链路失败和设备故障分开上报，旧故障 Condition 不会因为一次命令超时被误清为 False。",
            "故障消失后Condition怎么恢复": "对应检查真实运行、覆盖预期设备且没有检测错误时，检测器才把可恢复 Condition 写成 False。Condition 缺失或 Unknown 时继续等待有效检查。Node Controller 要求 False 从 LastTransitionTime 起稳定至少一分钟，避免短暂恢复后又出现故障时提前解除隔离。",
            "故障恢复了_为什么监控指标还可能保持1": "如果恢复分支只更新 Condition，没有同步更新故障指标，监控里的值就可能继续保持 1。这个指标按故障类型和 Reason 区分记录，我在沐曦、昆仑芯的恢复分支补了指标清理：复检通过后，先用原来的 Type 和 Reason 把故障值置为 0，再把 Condition 改为 False 并更新恢复原因。这样恢复状态和原故障指标会在同一条处理分支中更新。",
            "如何避免瞬时抖动触发自愈": "入口使用 Condition 的 LastTransitionTime 和 NodeHealer 配置的 duration，只有 True 持续达到阈值才创建 HealingTask；恢复端再要求 False 稳定一分钟。这样触发和恢复两侧都有时间门槛。对计数型 ECC 还应比较增量和趋势，而不是看到历史累计值非零就重启。",
            "为什么XID160选择Reboot而不是任务重试或GPU_Reset": "XID 160 表示 PCIe link lost，故障域已经超出单个训练进程，任务重试无法让丢失的 PCIe 设备重新枚举。当前 node-healing 动作集合包含 Cordon、Drain、Reboot 和 UnCordon，未接入 GPU Reset，因此由模板选择节点隔离和可选 Reboot。重启后仍然掉卡时保持隔离，交给运维检查插槽、供电、主板或板卡。",
            "Reboot失败多少次后转人工": "次数由 MachineRebootConfig 的 `MaxRebootTimes` 和重试间隔控制。每次重试前先确认上一次动作是否已经发生；达到配置上限或健康检查超时后，HealingTask 进入失败收口，节点继续保持隔离，系统通过阶段通知报告失败，并保留节点、厂商、故障 Condition、已执行步骤和失败原因，供交付、运维或厂商继续处理。",
            "Drain以后训练任务会自动恢复吗": "不保证，中间有好几个断点。Drain 本身只是对节点上的 Pod 发 Eviction 请求，也就是删掉它们，Kubernetes 里没有迁移 Pod 这回事。删完之后会不会有新 Pod 顶上，取决于这些 Pod 有没有 owner 控制器：裸 Pod 永久消失；Job 要看 backoffLimit；PyTorchJob 这类看 Operator 的 restartPolicy，而训练任务常见配置是 Never，一个 worker 没了整个 Job 直接判失败。\n即使控制器决定重建，后面还有三关。新 Pod 要调度到有空闲 GPU 的节点，原节点还在维修，集群满载时会一直 Pending，分布式训练还要凑齐整组才能起；起来之后所有 rank 要重新建立 NCCL 通信域，因为 communicator 不能动态增删成员，存活的 worker 也得跟着重启；最后还要从 Checkpoint 加载，业务没存或者存得太稀疏，进度就丢了。\n我的组件负责节点隔离、受控维修和健康后回池，成功判据是原始 Condition 变 False 并稳定一分钟；Drain 受阻时不能直接进入 Reboot。节点恢复不等于训练恢复，被驱逐的任务后来怎么样了，组件既不知道也没有接口查。所以收益指标要分开算，我们能证明的是节点从故障到回池的时间，不能声称训练恢复时间。",
            "GPU自愈怎么和训练平台联动_后续怎么演进": "当前已经支持业务确认排空后继续维修，训练平台自动接入还在规划中。接入时，自愈侧先禁止故障节点接收新任务；训练平台定位受影响的 Job，任务还能推进且故障允许延后时，确认完整 Checkpoint 保存成功，再按框架要求停止相关训练进程。任务已经失败时，使用最近一次有效 Checkpoint 恢复，并确认旧实例停止或已被隔离到无法继续通信和写数据。训练平台通过约定的节点标签给出排空确认后，自愈组件才继续维修；健康资源充足时，任务重建和原节点维修可以并行。维修顺序也要考虑任务影响：假设两台待维修节点分别属于两个大训练 Job，同时下线两台就可能中断两组训练。我会结合受影响的 Job、Checkpoint 是否完整和替补资源是否满足整组恢复，决定哪些节点先维修。",
            "驱逐时有没有考虑Pod是Master还是Worker": "当前没有按 Master、Worker 分别决定维修动作，我们已经支持的是业务确认排空后再维修。同步训练里，一个普通 Worker 退出，也可能让其他节点卡在梯度同步，所以只看它是不是 Master，不足以决定能不能驱逐。\n\n后续接入训练平台时，我会先通过节点上的 Pod 找到受影响的训练 Job，再由训练平台根据框架决定需要停止哪些进程。训练还能推进且故障允许延后处理时，先完成 Checkpoint 保存；已经失败时，就使用上一次有效的 Checkpoint 恢复。相关旧进程退出、业务确认可以维修后，自愈组件再继续操作节点。",
            "GPU_UUID怎么定位到Pod_训练Job和Rank_容器编号不同怎么办": "我会先查询故障节点 kubelet 的 PodResources API，按设备插件的 ID 格式，将设备分配记录与 GPU UUID 对齐，定位使用它的 Namespace、Pod 和容器。再查询 Pod 的 UID、ownerReferences 或平台任务标识，关联到训练 Job。Rank 需要训练侧补充：每个进程绑定设备后，读取实际使用的 GPU UUID，将它与 Pod UID、任务本次运行标识、global rank 和 local rank 一起登记。\n\n容器里的 cuda:0 可能对应宿主机的第 5 张卡，因此关联时使用实际 UUID，不按数字编号推断。插件设备 ID 如果采用索引或实例标识，也要先通过对应设备清单转换，不能直接当成整卡 UUID。这份映射在训练启动时建立并保存，进程重启后重新登记，因为弹性重启可能改变 Rank。",
            "维修请求怎样传入PyTorch_各Rank如何一致保存Checkpoint": "我会在训练代码里接入维修控制 Hook，让 Rank 0 的后台线程通过 HTTP 长轮询获取本 Job、本次运行的维修请求；线程收到后只设置待处理标志，保存动作由训练主循环执行。每个 Rank 都在梯度累积结束、一次完整的 optimizer.step() 完成后进入固定 Hook，每轮按相同顺序参加控制广播，由 Rank 0 发布继续训练或保存的决定。\n\n收到保存决定后，各 Rank 停止进入下一轮，交换并核对 global step，等待尚未完成的异步通信和 CUDA 操作，再进入保存流程。使用 Distributed Checkpoint 时，参与保存的 Rank 使用同一个 Checkpoint ID，完成模型、优化器和其他恢复所需状态的写入；异步保存也要等所有必要分片持久化完成，才能报告成功。训练平台收到保存完成和相关进程退出的确认后，再允许维修。如果某个 Rank 已失联或卡在通信中，无法在期限内到达这个边界，就按任务失败处理，使用上一次有效 Checkpoint 恢复。",
            "Condition_Event_Metric分别解决什么问题": "NodeCondition 记录节点当前的持续状态，供策略派单、防抖和恢复确认使用；Event 记录发生过的事件，用于排障；Metric 是监控时间序列，用来观察趋势。Condition 和 Event 通过 Kubernetes API 写入相应对象，Metric 通常由监控系统抓取，不是每次采样都写入 etcd。自愈 Controller 主要根据 Condition 与策略决定动作。",
            "为什么用NodeCondition而不是Event或单独的故障CRD": "NodeCondition 表达的是节点当前是否仍处于某种故障状态，Controller 可以围绕它持续调和，并使用 Status 和 LastTransitionTime 做触发、防抖与恢复确认。Event 更适合记录某一时刻发生过什么，可能过期、聚合或重复，不能作为解除隔离的可靠依据。当前修复单位是节点，如果再创建一套故障 CRD，就要额外维护它与 NodeCondition 的创建、更新、删除和一致性；因此当前用 NodeCondition 保存故障事实，用 Event/Metric 保存解释与观测，用 HealingTask 保存长流程执行状态。若未来要表达每张 GPU 的独立故障实例、同类型多次故障和完整诊断证据，再引入独立 Fault CRD 会更合适。",
            "permanent是不是永远不能恢复": "permanent 表示故障需要持续维护为 Condition；能否自动改为 False 由具体恢复策略决定。瞬时或可复检故障在主动检查通过后可写 False，永久硬件故障则保持 True 并转人工。",
            "厂商CLI没输出会不会被当健康": "不能把没输出当作健康。比如沐曦适配要先确定设备集合；假设枚举出了八张卡，温度查询只返回一张卡的正常结果，就不能认为整台机器的温度检查通过。每个检查项还要核对目标设备和必需字段是否都成功解析，命令返回成功也不代表每张卡都检查成功。命令缺失、解析失败、设备覆盖不完整或检测器没有运行，都只能说明本轮没有足够证据；只有与该 Condition 对应的检查实际完成、覆盖满足要求且结果正常时，才允许将可恢复的故障状态明确写回 False。",
            "为什么Reboot成功还不算恢复": "重启请求成功、主机完成重启和 GPU 恢复分别对应不同状态。云 API 返回成功说明请求被接受；CVM 路径还要查询并核对实例操作状态，TCS 路径以 boot_id 变化确认主机重启。主机起来后，GPU 仍可能掉卡、驱动未恢复或检测器未运行，因此还要等触发任务的原始 Condition 明确变成 False 并稳定至少一分钟，才确认恢复。",
            "多Healer会不会同时修一台节点": "当前有在途任务检查和策略优先级仲裁，但检查与创建之间仍有并发窗口。正常调和先查节点是否已有未结束的 HealingTask，有就继续原任务；没有才从命中的策略中选一个创建任务。优先级解决先修哪个故障，严格限制同节点并发还需要原子取得执行权，单靠先查再创建不能保证。",
            "同一节点同时出现多个故障怎么处理": "假设同一节点同时出现健康异常和 PCIe 错误，健康异常的策略优先级更高，本轮就先为它创建维修任务，另一项等待。如果节点已经有未结束的任务，就先处理现有任务。任务结束后重新读取节点状态，仍存在的故障再参加下一轮选择。这是当前的策略仲裁顺序；跨并发创建的唯一性还取决于节点执行权是否有原子约束。",
            "怎么避免同一节点创建重复HealingTask": "当前在创建任务前和执行节点动作前，都会按 nodeName 检查未结束的 HealingTask，发现已有任务就不继续派发或执行。两次检查能减少重复处理，但检查与创建没有形成原子操作。要封住这个窗口，可以为每个节点使用固定名称的锁对象，通过创建冲突或 resourceVersion 校验争取执行权，成功取得执行权的控制器才创建并推进维修任务。",
            "为什么多个故障不合并到同一张HealingTask": "多个故障可以合并，但任务必须分别保存每个故障的恢复条件，只有这些条件都满足才能判整张任务成功。当前采用一张 HealingTask 对应本轮选中的故障，便于记录触发原因、动作和恢复结果。其他故障在任务结束后重新仲裁，代价是可能需要继续处理下一张任务。",
            "Detector重启或原Condition消失怎么办": "继续保持节点隔离，等待检测器给出有效复检结果。Condition 消失或变成 Unknown，可能是 Detector 尚未启动、检查失败或状态上报中断；恢复仍要求原始 Condition 明确为 False 并满足稳定时间。超过恢复期限仍未取得有效健康结果时，任务转失败并通知人工处理。",
            "四个YunAPI分别做什么": "四个接口分别提供模板查询、策略查询、策略保存和维修历史查询。模板查询接口 ListMachineHealingTemplates 返回厂商入口、预置模板和默认自愈策略；策略查询接口 ListNodeHealers 返回当前配置、候选节点和已选节点。策略保存接口 ApplyNodeHealer 保存完整策略，并按用户提交的完整节点列表核对启用关系。历史查询接口 ListHealingTasks 返回维修任务的执行历史。",
            "同优先级为什么还要按名称排序": "固定排序可以保证 Controller 重启或重复 Reconcile 时仍选出同一个候选，避免任务和 limiter 的归属发生变化。",
            "最高优先级被限流为什么不跑第二名": "那会绕过优先级和容量保护。正确行为是让最高优先级候选一分钟后重试；当前任务结束后，其他仍为 True 的故障再重新参加仲裁。",
            "多Healer怎么兼容旧链路": "通过开关兼容原有单归属模式；异构生产链路启用 selector 后，一个节点可以被多个 Healer 匹配，但最终仍是多候选、单胜者、单任务。",
            "TCS重启如何防止重复执行": "我在接 TCS 的 SSH 重启时，专门处理了一个结果不确定的问题：reboot 命令发出去后连接断了，可能是命令已经执行、机器开始重启，不能直接当失败再重试。实现上先读取旧 boot_id，持久化到节点注解，写成功后才发重启；再次处理发现这份证据还在，就先查状态。SSH 暂时连不上就继续等，重新连上但 boot_id 没变也不能算完成，只有变化才确认这次主机重启发生。代码里相应区分了断连、鉴权错误、旧 boot_id 缺失和 boot_id 变化；主机重启确认之后，GPU 能否回池还要单独复检。",
            "Reboot超时后怎么办": "超时后不能直接重新下发命令。CVM 路径继续查询实例操作和主机状态；TCS 路径在 SSH 断开后先把结果视为未知，持续比较 boot_id。只要没有拿到动作完成和健康恢复的明确证据，节点就保持隔离；超过任务的动作或恢复期限后转失败或人工处理，不能盲目再次 reboot。",
            "Controller重启后如何继续任务和恢复限流记录": "Controller 启动时先读取所有还没有结束的 HealingTask，并根据任务记录恢复原始 Condition、已经占用的并发额度，以及仍处于小时、天和冷却窗口内的维修记录，然后从原来的步骤继续处理，再接收新的候选。恢复执行时先检查节点、实例操作或 boot_id 等实际状态，已完成的步骤继续推进，未完成的普通步骤重试，结果未知的重启继续确认。",
            "多个Controller实例如何避免重复执行": "node-healing 部署清单采用单副本并关闭 Leader Election，代码提供可选的选主开关。多副本时可以通过 Leader Election，让同一时刻只有一个 active Controller。但旧 Leader 可能已经发出重启请求、还没记录结果，新 Leader 就开始接管，因此选主本身不能保证外部重启只发生一次。接管后仍需要读取维修任务（HealingTask）、CVM 操作状态或 TCS boot_id，判断动作是否已经发生，再决定继续等待还是执行。",
            "为什么要先Cordon再Drain": "先隔离是为了阻止不容忍隔离污点的新 Pod 调度到故障节点，再 Drain 驱逐允许中断的已有 Pod。这个项目通过添加自愈专用 NoSchedule taint 隔离，Drain 遵守 PDB；如果反过来执行，旧 Pod 尚未退出就可能有新 Pod 进入。替代 Pod 的重建和调度由上层控制器与 Scheduler 负责。",
            "Drain成功Reboot失败是否解封": "默认保持隔离。只有节点随后恢复、本次任务触发的 Condition 明确为 False，且模板允许 UnCordon 时，系统才会解除 taint。",
            "Drain被PDB阻塞怎么办": "Drain 走 Eviction API，API Server 会依据 PDB 的 minAvailable 或 maxUnavailable 和当前 disruptionsAllowed 判断是否允许；预算不足时通常返回 429，自愈任务保持节点隔离，等待副本恢复或预算释放后重试。达到任务超时后进入失败或人工处理，不能为了完成维修自动退化成直接 Delete，因为直接 Delete 会绕过 PDB。确需强制处理时只能走单独授权、限范围和全审计的 break-glass 路径，不能隐藏在普通自愈重试里。",
            "直接Delete会绕过PDB_怎么补充类似保护": "如果扩展受管 Delete，我会增加独立的中断预算预约。PDB 只在 Eviction API 路径执行原生保护，直接 Delete 不会自动获得同等预算检查；因此普通场景仍统一走 Eviction，并通过 RBAC 限制业务方直接 Delete。确实需要受管 Delete 时，我会让入口先向独立预算控制器申请以 Pod UID 和工作负载 UID 为键的 DisruptionReservation，由单一账本通过 resourceVersion CAS 原子计入正在中断的副本；Validating Webhook 只放行持有有效 Reservation 的请求，不能只读取一次 disruptionsAllowed 后放行，否则仍有并发超额竞态。Delete 失败时释放预约，结果未知时先观察 Pod UID、deletionTimestamp 和替代副本状态，替代副本 Ready 后归还额度，超时对账清理孤儿预约。节点彻底失联或硬件风险继续扩大时保留独立权限、明确原因和全审计的管理员 break-glass 路径；正常预算服务异常时默认拒绝。",
            "跨应用多故障等级如何选择最小维修动作": "设计这类联动时，我会先确认故障和恢复动作会影响哪些卡、哪些任务，再选择处理单卡还是整机。例如只有一张卡异常，且型号和互联配置支持独立 Reset，就先隔离这张卡，由业务停止受影响的进程，再尝试 Reset；需要连带重置的其他卡也要计入影响范围。局部恢复不可用或复检仍失败，再评估节点重启。整机维修前，分别确认受影响训练任务的进度和退出已处理、推理服务已完成容量准备和摘流，不能用一个业务的空闲副本抵消另一个业务的中断风险。仍有业务不允许中断时，暂缓可以推迟的维修，保留故障隔离并协调处理时机。",
            "怎么判断一次自愈真正成功": "项目层的成功结论是故障节点恢复了可用资格。Cordon、Drain、Reboot 完成只是动作结束，还要确认本次 HealingTask 记录的原始 Condition 仍存在、明确为 False 且稳定至少一分钟，再完成 UnCordon，任务才能结束。Node Ready=True 或重启命令成功都不能单独证明 GPU 恢复；业务侧是否恢复请求或训练进度，还要由工作负载状态和业务指标确认。",
            "节点恢复过程中再次故障怎么办": "原始故障 Condition 再次变成 True 时，当前任务继续保持隔离；再次恢复为 False 后，状态稳定时间重新计算。如果原故障恢复、另一条故障仍为 True，当前记录的流程是完成这次任务并解除隔离，再让其他故障重新仲裁。这会留下故障仍在却短暂允许调度的窗口。要补齐这个判断，应在解除隔离前重新检查节点上所有阻断调度的故障，仍有阻断就保留隔离并继续处理。",
            "NodeReady为True为什么还不能认为GPU恢复": "Node Ready 说明 kubelet 和节点基础链路可用，GPU 数量、驱动和设备故障需要独立检测。例如节点 Ready=True 时，仍可能有一张卡无法访问。因此解除自愈隔离前，还要检查任务记录的原始故障 Condition 明确为 False，并稳定至少一分钟。",
            "PhaseRecovered为什么还不算任务结束": "它只是 Node Controller 确认健康证据已经满足的信号，此时节点可能仍带 taint；HealingTask Controller 完成 UnCordon 后才能写 PhaseCompleted，任务才真正结束。否则会出现任务显示完成但节点仍未重新开放调度的问题。",
            "NPDPlus_NDB和PDB分别是什么": "NPDPlus 是节点侧的故障检测组件，负责调用厂商工具、读取驱动日志并输出 NodeCondition；NDB 是 NodeDisruptionBudget，在节点维修入口按 selector、MaxUnavailable/MinAvailable、CPU/内存比例和可调度节点数判断节点池还能否下线一台机器；PDB 是 Kubernetes PodDisruptionBudget，在 Drain 阶段保护应用 Pod 副本可用性。三者分别属于检测层、节点容量准入层和 Pod 驱逐保护层。",
            "为什么TEP里写了NPDPlus却没单独写NDB": "这份 TEP 的主线是异构故障检测、NodeCondition 标准化和自愈产品接入，所以组件章节重点写了 NPDPlus；NDB 属于 node-healing 内部的维修准入能力，最终交付代码已经通过 `HealingPolicy.nodeDisruptionBudget` 接入 limiter。它不是另一个检测组件：Controller 在创建 HealingTask 前检查节点池的不可用数、可调度节点数以及 CPU、内存容量比例；预算不足或检查异常时都不启动新维修。",
            "NDB和PDB有什么区别": "NodeDisruptionBudget 在节点维修入口判断还能不能再下线一台节点；PodDisruptionBudget 在 Drain 阶段保护 Pod 副本可用性，作用层级不同。",
            "怎么避免故障风暴触发大面积维修": "当前通过 NodeHealer 的维修并发限额，以及 NodeDisruptionBudget 的节点容量预算，控制同时维修和主动下线的范围。假设同一策略下十台节点出现允许延后处理的异常、训练仍能运行，并发限额为两台，在其他准入检查通过时最多同时推进两台维修，其余暂缓，避免一次中断过多任务并耗尽替补资源。如果十台节点已经全部失效，只修两台无法保住另外八台的算力，限额过紧还会拖慢恢复。后续我会把主动下线预算与已失效节点的维修并发分别管理：前者保护仍在运行的业务，后者根据维修资源和共享依赖的承受能力设置。",
            "这个项目最难的部分是什么": "最难的是外部动作返回错误时，判断到底要重试还是等待。我接TCS的SSH重启时，reboot发出后断连可能意味着机器已经开始重启；如果立即重发，会把一次维修变成重复重启。所以我先把旧boot_id写入节点注解，持久化成功再派发命令；派发后断连保留记录，重新处理先看boot_id有没有变化。这几个分支分别写了断连、鉴权错误、旧证据缺失和boot_id变化的测试代码，最后再由原始GPU Condition确认能否回池。",
            "页面保存策略和节点选择能原子吗": "不能。一个策略对象加多台节点的启用关系没有跨对象事务；接口每次都拿用户提交的完整节点列表重新核对，部分失败后继续重试，直到实际选择和用户提交的一致。",
            "为什么页面不能编辑CordonDrainReboot步骤": "这是权限和安全边界。页面只管节点、场景开关、priority、限流和通知；破坏性动作由 Addon 预置并评审过的 MachineHealingTemplate 管理，避免用户任意拼步骤。",
            "为什么MachineHealingTemplate单独做成CR_不能直接内嵌到NodeHealer": "可以内嵌，Kubernetes 并不要求模板单独做 CR。当前拆成 MachineHealingTemplate，是为了把“哪些节点和故障允许维修”的可变策略，与“具体允许执行哪些破坏性动作”的审核配置分开；Addon 可交付厂商预置模板，客户也可定义自有模板，多个 NodeHealer 可以按名称和 namespace 复用。独立资源也便于单独交付、审计以及按资源类型配置权限，实际权限由平台 RBAC/API 决定。代价是多一层引用、namespace 解析和查找校验。当前每个 NodeHealer 只引用一个全局模板，Node Controller 创建 HealingTask 时读取它并把本次步骤配置写入任务快照；模板不被直接 watch，模板更新不会改写在途任务，也不会单独触发 Node reconcile。若不需要复用、独立审核或权限边界，把配置内嵌在 NodeHealer 会更简单。",
            "通知能保证不重不漏吗": "不能保证不重不漏。阶段和渠道标记用于去重，但发送成功后若记录结果失败，仍可能再次发送；有限重试全部失败后也可能漏发。通知独立于主自愈链路，消息服务异常时节点维修继续推进。",
            "NoSchedule是不是绝对封锁": "不是，带匹配 toleration 的 Pod 仍可能调度。它是自愈专用隔离标记，Drain 和集群 toleration 策略共同决定实际隔离强度。",
            "为什么不用DevicePlugin健康状态直接自愈": "Device Plugin 报告设备不健康后，主要影响后续设备分配，不能完成已有 Pod 迁移、节点级动作、限流和恢复复检，因此不能替代 NodeCondition→HealingTask 控制链。",
            "8卡节点只坏1张卡_为什么还可能Cordon和Drain整台节点": "当前自愈链路的执行单位是节点，是否进入 Cordon、Drain 和 Reboot 由故障类型和动作模板决定。Device Plugin 把单卡标记 unhealthy 可以阻止它继续分配给新 Pod，却不会迁移已在使用它的训练进程。驱动、PCIe、GPU 互联或节点级故障共享更大故障域时，整机隔离和重启更符合安全边界。卡级调度隔离和训练恢复联动属于后续演进。",
            "怎么定义项目闭环": "配置部分包括模板、YunAPI、NodeHealer 和节点启用标签；运行部分从检测、仲裁、限流、创建任务、执行动作一直到 Condition 复检和重新开放调度；产品部分包括历史、通知和 Addon 交付。运行状态由 HealingTask 记录，供执行恢复、历史查询和通知使用；节点健康由原始故障复检确认。",
            "两个用户同时修改同一个NodeHealer怎么办": "ApplyNodeHealer 会先读取当前对象，再带着当前 resourceVersion 执行 Update。并发修改时，先成功的写入会推进 resourceVersion，后一个旧版本写入应收到 Conflict，调用方需重新读取后再提交，不能静默覆盖。但 NodeHealer 更新和多节点标签对账仍不是一个原子事务，后半段失败仍要用完整期望状态重试。",
            "NodeHealer或模板更新会不会改变正在执行的HealingTask": "不会改写已经生成的任务快照。创建 HealingTask 时，Controller 已经把本次触发的 Condition、动作步骤、Cordon、Drain、Reboot、UnCordon 配置和超时参数拷贝到 HealingTask Spec。后续 NodeHealer 或 MachineHealingTemplate 的修改只影响新创建的任务，在途任务继续按自己的 Spec 执行，否则会出现任务执行一半语义突变。",
            "HealingTask按模板执行时_是不是每次Reconcile只做一步": "不是。Node Controller 创建任务时，会按本次触发的 Condition 从模板展开适用步骤和参数，写入 HealingTask Spec 快照。HealingTask Controller 收到新任务后启动 handler，由它按 Spec 顺序执行这些步骤；每个动作完成后更新对应的步骤状态，任务整体阶段推进到 Recovering。普通步骤状态更新不会各自触发一次 Reconcile。Node Controller 另行确认原始 Condition 已明确恢复且稳定后写入 Recovered，HealingTask Controller 收到后执行 UnCordon 并完成任务；模板中未匹配本次故障的其他场景步骤不会执行。",
            "取消某些节点自愈时标签如何收敛": "ApplyNodeHealer 接收的是完整 SelectedNodeNames，而不是单次增删事件。平台会把当前已打标节点与这份完整集合比较，对新选节点加标签，对已取消节点移除标签。中途失败会返回错误；重试同一份完整列表时再次计算差集，直到实际标签收敛。多节点启用关系通过幂等重试逐步收敛。",
            "HealingTask历史查询和长期保留怎么做": "ListHealingTasks 支持按厂商、节点、状态和时间范围过滤分页，页面查询依据是持久化 HealingTask，而不是 Controller 内存。任务已配置 TTL 和 GC：执行中任务保留用于重启恢复，进入完成或失败等终态后，按保留期供历史查询，到期后由 GC 清理，避免 HealingTask 长期累积增加 API Server 和 etcd 负担。",
            "是在训练平台还是推理平台使用": "项目部署在 TCS 私有云的 GPU Kubernetes 集群，提供节点故障检测和维修能力，训练和推理节点都可以接入。",
            "具体看什么信号才能判定GPU故障": "我们主要看设备和驱动是否正常、ECC/RAS 是否出现不可纠正错误，以及 PCIe、NVLink、HCCS 等互联是否异常。温度、功耗也会监测。\n这些检查已经封装在节点侧的 NPDPlus 插件里：通过厂商 CLI、SDK 和驱动日志获取信息，按检测规则判定后，上报对应的 NodeCondition。自愈控制器直接消费这些故障状态，再按配置的持续时间和策略决定是否启动维修。\n比如海光驱动日志中的 `PCIe link lost`，会被识别为 PCIe 链路故障；普通的 CLI 超时或输出不完整，则需要与明确的设备故障区分。",
            "驱动日志和NVML接口查询有什么区别": "两者是两条并行通道，都由驱动提供但彼此不相交，NVML 不会去读日志。驱动日志记录哪张卡在什么时候报了什么错；NVML 周期查询返回设备状态、ECC 计数、链路状态这类结构化数据，反映的是查询那一刻的状态。同一次故障两边都能看到属于信息重叠，比如一次不可纠正 ECC，既留下 XID 48 日志，也反映在 NVML 计数里。\n开发上，日志侧是两级正则：先提取时间戳、XID 编号、设备 PCI 地址和描述并重组成规范文本，再用规则 pattern 匹配决定属于 fatal、warning 还是可忽略，加一个错误码只改配置不动解析代码。接口侧先处理调用结果，失败不能当健康，返回设备丢失或未知错误时还要累加计数交给上层判定掉卡；取到数据后按三种方式判断，温度功率比阈值，ECC 计数和重映射行数比增量，InfoROM 是否损坏、ECC Mode 的 current 与 pending 是否一致、链路是否激活这类看状态位，多数项再过一道滑动窗口确认持续性。\n还有两点差别：时间信息只有日志侧有，它能给出故障发生时刻，NVML 只有采样时刻；漏检特性也不同，日志采集断掉或轮转超过回看窗口，事件会整条丢失，NVML 不会漏当前值但会漏中间过程，比如温度短暂冲高又回落，落在两次轮询之间就看不到。接口调用成功只表示查询成功，不代表 GPU 健康；错误计数也不等于每次故障的准确发生时间。",
            "NVML是不是去读驱动日志拿XID": "不是。NVML 和内核日志是两条并行通道，都由 NVIDIA 驱动提供，但彼此不相交。调用 nvmlDeviceGetTemperature、nvmlDeviceGetMemoryErrorCounter 时，是程序经 NVML 进入驱动直接查询设备状态，不会去读 dmesg 或 /var/log/messages。同一次故障两边都能看到，是因为驱动同时做了两件事：一边把 Xid 79 写进内核日志，一边通过事件接口对外暴露；这是信息重叠，不是同一条数据链。日志侧适合看某一时刻发生了什么并保留上下文，接口侧适合看当前设备处于什么状态。",
            "你们的XID走哪条通道_有没有用NVML事件订阅": "我们的 XID 走日志通道，由独立的 xid 插件解析文本：日志路径配的是 /var/log/messages，用正则匹配 NVRM: Xid (PCI:...): 编号, 描述 这类行，提取 XID 编号和设备 PCI 地址，再按配置分成 fatal、warning、application 几类，74 和 79 配为 fatal，映射到 GPUXIDFatalError。NVML 通道由另一个 gpu 插件负责，只做周期性的状态查询，没有使用 NVML 的事件订阅接口。所以两条通道在我们这里是职责分离，不是互为备份：事件类信号只有日志侧有，日志采集断了这部分就会丢。掉卡这类最严重的情况例外，NVML 侧轮询也能独立发现——先取 UUID，再用一次取温度做二次探测，返回设备丢失或未知错误时判为设备不可访问，上报设备丢失的 Condition。",
            "NVML轮询的指标分几类_判断方式一样吗": "分三类，判断方式不同。第一类是瞬时量，比如温度和功率状态，直接和阈值比大小。第二类是累积计数器，比如 ECC 的单比特和双比特计数、重映射行数、链路 CRC 错误数，这些只增不减，绝对值没有判别力——一张服役很久的卡单比特累计上千也可能是正常的，所以要保留上一次读到的值做差，看这个检查周期内新增了多少。做差还得处理计数器归零：驱动重载或设备复位后新值会小于旧值，代码里识别为计数器重置，本轮增量按零处理并记日志，不能算出一个回绕后的巨大值；这也是为什么状态里有一组 LastDBECount、LastCorrRows 这样的字段专门存上次读数。第三类是状态位，比如 InfoROM 是否损坏、ECC Mode 的当前值与 pending 值是否一致、NVLink 链路是否激活、重映射行的 pending 和 failure 标志，这类既不比阈值也不比增量，直接看布尔或枚举结果。三类判出异常之后，温度、功率、单比特 ECC、链路 CRC 这些还要再过一道滑动窗口，确认是持续状态而不是一次抖动。",
            "同样是ECC_双比特和单比特为什么处置不一样": "因为两者的物理含义不同。双比特错误不可纠正，说明显存已经出现物理损坏，所以只要增量大于零就判为异常，并在进程内置上永久标记，后续每轮检查都继续报告这个故障，不会因为下一次读数没有新增就当作恢复。单比特错误可以纠正，偶发几次不代表坏卡，所以比的是增量是否超过阈值，再用时间窗口判断是不是持续高错误率。温度和功率同理走窗口判断，而且是可恢复的：只有本轮真的执行了对应检查、并且这一轮没有再命中错误，才允许把 Condition 置回 False；如果这一轮压根没到检查周期，就不能拿这一轮没有报错当成恢复证据。上报规则里的 permanent 指的是维护一个持续的 Condition，区别于只发 Event 的 temporary，它和硬件永久损坏是两件事。",
            "NVIDIA和沐曦的采集适配具体差在哪": "NVIDIA 这条路径直接调用 NVML API，取得结构化数据和调用状态，不需要解析 nvidia-smi 的文本；沐曦这条路径执行 mx-smi，再解析命令输出。比如先用 mx-smi -L 建立设备清单，并通过 PCI 地址等信息把后续结果对应到具体设备；ECC 数据从 --count-ecc 读取。要特别注意 mx-smi --show-event 中 dbe 表示 PCI device base error，不是 NVIDIA ECC 的双比特错误。命令报错、空输出、解析失败或八张卡只返回七张卡的数据，都应记为采集不完整，不能当成零错误或健康。之后两条路径再把判断结果映射到统一的 NodeCondition。",
            "你实际接触的集群有多大_大概多少个节点": "以建设银行的 GPU 资源池为例，有 128 台八卡服务器，共 1,024 张 GPU。",
            "扩到5000节点这条自愈链路哪里需要加固": "扩到 5000 节点，我会从控制面突发压力和执行并发两方面加固。\n① 大量节点同时异常会集中写 Node Condition、创建 HealingTask。我会错开检测周期、减少没有实际变化的状态写入，并对任务创建限速，避免把 API Server 和队列瞬间打满。\n② 如果慢操作在 Reconcile 里同步等待，Drain、SSH 或云 API 调用会各占一个 worker；worker 数量是有限并发上限，全部占满后其他 key 就会排队。不过当前项目的 HealingTask Controller 会启动 handler goroutine 后返回，慢维修步骤不会持续占住 Reconcile worker。扩容时仍要给 handler 和远程操作设并发上限、超时；查询节点关联 Pod 使用索引，避免全量扫描。\n另外，控制器能同时处理多少任务，不等于业务能承受同时下线多少节点。维修并发还要受节点池容量和业务中断预算约束。验证时观察 API 延迟、队列等待时间、handler 并发数和同时下线节点数。",
            "慢维修动作堵住控制器是什么意思_多个Worker也会堵吗": "多个 worker 只能并行处理有限数量的 Reconcile。如果慢的 Drain、SSH 或云 API 调用在 Reconcile 里同步等待，每个等待中的调用都会占住一个 worker；并发槽位全部占满后，其他对象的 Reconcile 请求只能排队。当前项目的 HealingTask Controller 启动 handler goroutine 后就从 Reconcile 返回，所以维修步骤的慢等待不会持续占住这个 worker；不过 handler goroutine、网络连接和远程服务仍有资源成本，仍要限制并发并设置超时。",
            "NodeController具体在哪些地方访问APIServer": "读取侧通过 List/Watch 建立并更新 Node、NodeHealer 和 HealingTask 的 Informer 缓存，调和时很多查询从缓存读取，不是每次都直连 API Server。写入侧会创建 HealingTask；任务控制器执行隔离时更新 Node，Drain 时调用 Pod Eviction API，并把任务阶段写回 status。NodeCondition 是 Node.status.conditions 中的字段，不是一个单独的 Kubernetes 资源。",
            "这个项目从零到一还是优化_具体改善了什么": "我是在已有自愈框架上完成异构适配、执行增强和平台化交付。检测侧补齐昇腾、昆仑芯和沐曦，把厂商覆盖从两类扩到五类；控制侧完善多策略仲裁、恢复判定，并接入 TCS 的 SSH 重启和 boot_id 结果确认。平台侧实现配置 API、节点启用管理、维修历史和阶段通知，让运维通过统一入口管理这些能力。按同类故障的处置流程估算，整体时间缩短约 60%，主要减少人工登录、执行操作和等待下一步处理的耗时。",
            "举一个适合自动自愈的场景_收益具体落在哪里": "海光 PCIe 链路丢失是一个例子：检测器把对应日志映射成持续故障，策略允许且业务完成排空后，组件自动执行维修和复检。收益在于运维不必逐步登录节点、下命令、等重启再检查，把每个步骤之间等待人工推进的时间省下来。检测映射有日志样本验证，这个案例没有单独的真机处置耗时记录。",
            "如果追问收益数字_恢复耗时和GPU容量怎么计算": "约 60% 是按处置流程估算的整体时间降幅，计算为“人工流程耗时减去自动流程耗时，再除以人工流程耗时”。比较时保持厂商、故障类型和任务条件一致，时间从首次明确故障算到恢复调度资格，包含持续时间等待、排队、Drain、重启和复检。节省主要来自人工登录、操作交接和等待人工推进，机器重启与健康稳定窗口仍要保留。容量损失按不可用 GPU 数量对时间积分，例如整台八卡节点退出资源池半小时，对应 4 GPU·小时。人工投入另外统计接手次数和操作时长，未恢复事件及误触发造成的额外隔离单独记录。",
            "训练有Checkpoint就算无状态吗": "不算。训练任务需要维护模型参数、优化器状态、训练进度，以及按需恢复的数据迭代位置和随机数状态。Checkpoint 把这些状态保存到持久化存储，使新的训练进程能够从某个进度恢复；训练 Pod 可以替换，只表示计算进程可重建，不代表训练没有状态。训练平台的 API 或 Controller 可以把状态放在 Kubernetes 对象或数据库中，使实例逻辑上无状态，但它们管理的训练任务仍有状态。",
            "项目里的PDB实际由谁创建和维护": "自愈组件使用集群中已有的 PDB。预算数值由业务方或模型平台按副本数和可用性要求确定，自愈侧在 Drain 时遵守驱逐结果。",
            "哪种国产卡最接近NVIDIA_按什么维度比较": "如果按 CUDA 接口和源码迁移方式比较，我会选沐曦。\n\n沐曦提供 cu-bridge，对 CUDA Runtime、Driver API 和部分加速库做兼容，可以保留已支持的接口和代码结构，再通过它的工具链编译到 MACA 环境。这里的接近是指已有 CUDA 程序的迁移路径，不代表所有 CUDA 程序都能原样运行，也不代表性能等同于 NVIDIA。\n\n真要判断一个业务迁移难不难，我会看它有没有自定义 CUDA 算子、依赖哪些库、目标版本支持到哪里，以及多卡通信和性能是否达到要求。只跑通一个普通 PyTorch 模型，还不足以说明兼容性完整。\n\n昇腾有自己的 CANN 和 HCCL，PyTorch 模型通常通过 torch_npu 接入，没有对应实现的自定义 CUDA 算子需要替换或适配，自研底层算子会接触 Ascend C。越深入算子、内存搬运和性能调优，与 CUDA 开发方式的差异越明显。生态成熟度和开发方式接近程度需要分开评价。海光 DTK 和昆仑芯 XPU 也有 CUDA 兼容路径，具体迁移改动仍取决于代码和版本，不能据此给所有国产卡排一个绝对名次。",
            "同一Condition持续上报为什么不能每轮刷新LastTransitionTime": "LastTransitionTime 记录的是状态发生变化的时间。相同的 True 持续上报时如果每轮刷新，策略看到的故障持续时间就会不断归零，可能一直达不到 duration；相同的 False 每轮刷新，也可能一直满足不了一分钟稳定恢复窗口。因此状态不变时保留转换时间，只有 True、False 或 Unknown 发生变化时更新，Message 和观测信息按需单独更新。",
            "你在评审里实际遇到过什么取舍": "我们扩展同一节点匹配多个自愈策略时，旧逻辑靠 belong-to 直接找到唯一策略，改成遍历 selector 会影响共用组件的其他环境。评审时确定用默认关闭的 FeatureGate 保留旧路径，由 Addon 配置开启新能力。还有一个被指出的问题：如果出队时阻塞等待某个节点的锁，会挡住同队列其他节点，所以不能靠一直等锁推进维修。这个取舍后来体现在配置更新也分两条路径：新模式只更新策略缓存，避免走旧 belong-to 的队列清理。",
            "节点上已有训练作业时实际怎么处理": "启用业务主动排空方式时，自愈侧先隔离节点，等待业务设置约定的排空标签，收到确认后才继续维修。当前已经接通的是这个确认入口，训练 Checkpoint 保存和整组停止还没有自动接入。业务需要在确认前处理好训练中断：能继续运行的任务先保存状态，已经失败的任务按已有 Checkpoint 安排恢复。拿不到确认就不继续重启。",
            "NPD除了GPU卡本身实际还检测什么": "NPD 是带有现成检测能力、又允许扩展的节点故障检测框架。社区 NPD 已有日志读取与解析、规则匹配、自定义检查执行和状态上报能力，不需要把所有解析都写成脚本。本项目主要使用日志规则、外部检查程序和编译进 NPDPlus 的 Go 检测模块三种方式，这不是社区 NPD 全部 monitor 的分类。日志型启用了 kernel-monitor 读取 /dev/kmsg 和 docker 日志监控，XID 日志检测也属于这一类；脚本型周期执行外部脚本或二进制，覆盖 kubelet、containerd、dockerd 存活，以及 fd、pid、thread 压力、overlay 和网络等检查；Go 扩展模块可以直接调用 SDK 或厂商工具并维护设备状态。已有 NVIDIA gpu-monitor 和海光 dcu-monitor 是复用能力，我新增的是昇腾、昆仑芯、沐曦的检测适配与故障语义接入。\nkernel-monitor 的规则包括 OOM 和 Pod 级 OOM、任务长时间阻塞、内核 Oops、Ext4/XFS 错误、Buffer I/O 错误和 unregister_netdevice；Pod 级 OOM 能从 oom_memcg 路径定位到具体 Pod。这里要区分上报形态：大部分规则是 temporary，只发 Event 不维护 Condition，不会触发自愈；只有 KernelDeadlock、ReadonlyFilesystem 和 XFSError 三个是 Condition。\n这些检查的作用在故障归因。训练任务跑不了，可能是卡坏、容器被 OOM、kubelet 或 containerd 挂了、磁盘转只读、overlay 异常、fd 或 pid 耗尽，后几种 GPU 完全健康。有这些插件分担，排查时先确认异常来自设备、驱动还是节点基础服务，不会把训练报错都归成坏卡。非 GPU 这部分复用 NPDPlus 已有能力，我新增的是异构 GPU 适配和故障语义接入。",
            "NPD的三类monitor分别怎么工作": "这里指本项目主要使用的三种检测方式，不是社区 NPD 只有三类 monitor。① 日志型对应 config.system-log-monitor：配置日志路径、格式、回看窗口和正则，NPD 负责读取、解析及匹配，再按规则生成 Event 或 Condition；新增简单日志故障通常只需补配置。② 脚本型对应 config.custom-plugin-monitor：配置可执行文件、参数、间隔、超时和并发度，由脚本或二进制调用厂商工具、解析结果并返回检查状态，NPD 负责运行和上报。③ NPDPlus 的 Go 扩展检测模块：直接调用 NVML 等 SDK 或厂商工具，并管理设备身份、跨轮次计数、滑动窗口与恢复状态，已有 gpu-monitor 和 dcu-monitor 属于这一类。选择依据是取数方式、状态复杂度和维护成本；脚本也能保存跨轮次状态，不应说只有 Go 模块才能计算增量。社区 NPD 还提供 SystemStatsMonitor、HealthChecker 等能力。",
            "NPD是不是只抓日志_解析都要自己写脚本": "NPD 自带日志读取、解析与规则匹配能力，也支持主动运行检查。日志已经包含明确故障码时，通常配置匹配规则即可，不一定要写脚本；需要执行厂商命令时，可以通过 CustomPluginMonitor 调用脚本或二进制，由检查程序解析并判断；需要直接调用 SDK 或管理复杂状态时，可以使用代码扩展模块。新增厂商需要补充的是专用的取数、故障判断和映射逻辑，具体采用配置、脚本还是 Go 模块取决于需求。",
            "NPDPlus的Plus增强在哪里_哪些是复用哪些是本人增量": "这里的 NPDPlus 是项目沿用的公有云扩展版本。社区 NPD 已有日志监听与解析、规则匹配、自定义检查执行、NodeCondition/Event 上报和指标输出；已有 NPDPlus 增加了 NVIDIA GPU、海光 DCU 的专用检测模块和厂商日志规则，这两家是复用能力。本项目补齐昇腾、昆仑芯、沐曦的厂商接口和输出解析、设备身份对应、故障映射及恢复判断，并对齐下游 Condition Type。增强重点包括识别有效采样和设备覆盖完整性：八张卡只取得七张卡的结果时，第八张记为采集缺失；命令失败或检查未执行不能清除旧故障。NPDPlus 输出检测事实，是否维修由 NodeHealer 策略决定，动作由 HealingTask 执行。",
            "DCGM与NPD在检测上有什么区别_能否组合使用": "DCGM 自带 NVIDIA GPU 的指标采集、健康检查和诊断能力，可面向单卡或 GPU 组；NPD 提供通用节点问题的检测和上报机制，GPU 能查出什么取决于接入的规则、脚本或模块。DCGM 的具体健康检查与 NPDPlus 中对应的 GPU 检测逻辑存在重叠，例如 ECC 判断。可以直接复用 DCGM 的检查结果，再经 NPD 适配模块写 NodeCondition；也可以由 NPDPlus 直接调用厂商 SDK 或工具实现检测。DCGM 接入是可选方案，当前本地 NVIDIA 检测路径是 NVML 查询和 XID 日志，不能说已集成 DCGM。",
            "NVIDIA哪些XID比较严重_实际按什么证据决定动作": "我重点关注设备访问、显存 ECC 和卡间互联异常。XID 79 表示驱动通过 PCIe 无法访问 GPU，定位时把 GPU UUID、PCI Bus ID 和内核日志对应起来，再检查设备枚举与查询结果。XID 74 指向 NVLink 异常，设备可能仍能枚举，需要结合链路状态、拓扑和报错时间定位；对端一起报错也可能是同一条链路受影响。不可纠正 ECC 再结合设备状态和恢复建议处理。具体故障规则决定隔离、重启或转人工，错误编号本身不足以决定动作。",
            "推理机器故障怎么隔离_谁让故障Pod停止接流量": "当前自愈侧负责节点隔离，停止故障 Pod 接流量需要模型平台或服务健康检查配合，自动摘流还没有接进这条链路。实例已经无法推理时，应立即从服务后端摘掉，同时补建副本；Cordon 只限制新调度，不会停止现有请求。计划维修且实例仍能服务时，应先准备好替代容量，再停止接新请求并等待在途请求结束。业务确认排空后，自愈侧才继续操作节点。",
            "场景演练_XID79掉卡与可分配资源减少": "按社区场景演练，八卡节点只剩七张可分配卡时，我会先把资源上报和物理掉卡分开。先对齐故障 GPU 的 UUID、BDF，再查 PCIe 枚举、内核 XID79 和设备插件的 unhealthy 记录。79 说明驱动已经无法经 PCIe 访问这张卡，但还不能定性为板卡损坏。确认任务已停止并留存日志后，再按平台要求重启。恢复时重新核对设备清单、可分配数量和 GPU 小任务；错误复发就继续隔离，转查链路、供电或驱动。",
            "场景演练_ECC相关XID94与95的不同恢复范围": "按社区场景演练，同样是内存错误，我会把 XID94 和95分开处理。94 的错误被限制在受影响应用内，先定位该应用；如果是分布式训练，还要看是否需要整组重建。95 会影响多个应用，必须先完成要求的 GPU Reset，再恢复任务。动手前查清 MIG、互联拓扑和占卡进程，不能为了修一个实例中断未确认的其他任务。验收看原错误是否复发、设备状态和任务是否真正推进；历史 ECC 累计值没有归零不等于修复失败。",
            "场景演练_NVSwitch_FabricManager未就绪导致CUDA初始化失败": "按社区场景演练，nvidia-smi 能看到卡，但 CUDA 报系统尚未初始化时，我会继续查 NVSwitch 的 fabric 状态。若停在 In Progress，就查驱动容器是否安装了 Fabric Manager，再看它的启动日志和兼容关系，不会直接把卡判坏。确认是软件初始化问题后，先停受影响任务，再按该平台流程修复。验收既看 fabric 注册完成，也重跑原失败任务；需要多卡通信的业务还要检验通信，单卡可见不能代表整机能训练。",
            "场景演练_跨节点NCCL初始化卡住_误选不互通网卡": "这是社区场景演练：单机多卡能跑，两节点却卡在NCCL初始化。我会收齐各rank日志，先区分连接没建立还是集合通信卡住，再核对选中的网卡。接口显示UP不等于节点间可达；若误选了隔离的管理网，就在任务容器中确认业务网双向连通，再为这次任务指定该接口。复测先跑小规模AllReduce，要求所有rank结束且结果正确，再回到原任务；日志不支持误选接口，就不能把改参数当作修复。",
            "场景演练_容器能运行nvidia_smi但CUDA不可用_漏挂compute能力": "这是社区场景演练：容器能看到GPU，CUDA却初始化失败。我会先在同节点用正常镜像验证计算，再查启动配置和实际驱动库。若只声明utility，显卡管理工具能工作，却没启用CUDA需要的compute。确认缺失后，修改启动配置并重建测试容器；在旧容器里补环境变量不会补挂驱动库。验收必须完成显存分配、一次计算和同步，再恢复原应用，不能只看显卡管理工具正常就结案。",
            "场景演练_训练越跑显存越高_统计代码保留GPU张量": "如果固定batch开始能跑、后来显存持续增长，我会先记录每轮allocated和reserved，区分活跃张量增加与缓存池保留。这个演练设定的是统计列表一直存GPU输出：定位到持有引用的列表后，统计先detach再转CPU，单值用item()，并及时释放原GPU引用。然后固定输入和运行步数复测内存曲线，而不是只调用empty_cache看nvidia-smi数字降没降。若allocated依旧上涨，就继续查持有者；只有曲线和结果都验证过才算修复。",
            "场景演练_vLLM长请求增多后频繁重算_吞吐和尾延迟取舍": "模型能启动、长请求增多后才卡顿，我会先看KV使用和抢占重算日志。若确实是KV压力，就固定模型和请求集合，只降低同时运行请求数做对照，同时记录实际并发、排队、首token和最长生成停顿。减少抢占可能改善生成连续性，也可能把等待转移到入口，所以我不会只报抢占降了多少。加载阶段或CUDA graph捕获时的OOM要另查，不能和运行期KV不足混成一个问题。",
            "场景演练_GPU利用率低_先定位输入等待还是设备瓶颈": "GPU利用率低时，我会把GPU采样与训练step时间线对齐。假设Profiler显示大段空档落在取batch和CPU解码上，我会用预加载的同一批数据作定位对照，确认GPU是否一直在等输入，再调整实际数据加载路径。最后必须回到原始数据集和读盘方式看端到端吞吐，不能拿预加载的数字当业务收益。GPU-Util表示采样窗口里有kernel执行的时间比例，单看它也不能判断算力或显存带宽是否充分利用。",
            "公开实操_有哪些典型硬件故障现象": "公开实操里值得准备的三类现象是掉卡或设备不可访问、内存不可纠正错误，以及高温导致的异常。GPU资源不可分配可以讲GPU Operator的一次H100案例：nvidia-smi还能看到8张卡，但Kubernetes只报7张可分配卡，作者重启设备插件后才恢复资源上报。这说明排障要把物理枚举、驱动状态和插件健康状态分开。散热也有具体诊断例子：同一八卡节点的GPU Memory检查里只有GPU0失败，报告明确记录thermal violations，其他七张通过，此时应继续查这张卡的散热路径。",
            "公开实操_PCIe具体有什么故障_举一个实操例子": "我研究过一个Dell R730安装P100的公开案例：lspci已经能列出卡，nvidia-smi却报没有设备。作者先试BIOS和驱动，后来发现供电线需要匹配Dell riser的针脚定义，换正确线缆并清理重装驱动后，才贴出成功识别的结果。这个例子里，枚举得到设备只是第一步，接下来还要核对供电、驱动初始化和实际CUDA任务。对于真正的PCIe链路报错，我会把GPU的BDF和上游端口对上，结合AER严重性、Link状态和同路径其他设备判断影响范围。",
            "公开实操_硬件真的坏掉_如何确认而不只重启": "判断卡侧硬件问题，我会找重复性和交叉对照证据。一个公开XID79案例中，作者换驱动、系统、线缆和电源都没解决；原卡放朋友机器仍复现，替换卡放原机却能运行，最后决定送RMA。真正有价值的是故障跟着原卡走，前面换电源无效也被保留下来。数据中心内存故障还能看更直接的证据：Crusoe文档把row-remapping待执行和失败分开，失败标志出现就带日志交厂商；这里的DRAM行重映射失败，按NVIDIA对应内存RMA规则还要由Field Diagnostic验证。",
            "公开实操_Meta的Llama3训练平台怎样使用GPU": "我研究过Meta公开的Llama3训练平台，它会把GPU、网络和存储放在一起讲：两套集群各24576张H100，一套RoCE、一套InfiniBand，端点都是400Gbps；Llama3实际在RoCE集群训练。数据和Checkpoint通过FUSE接到Tectonic存储。",
            "公开实操_GPU节点之间带宽怎么样_实际怎样排查": "拿NVIDIA社区一个实际双机测试说：每台4张H100、四个200G网口，单端口名义单向25GB/s，四口合计100GB/s。现场先出现ib_write_bw能过、NCCL却失败；临时关闭GDR后能跑，最后调整裸金属ACS配置后恢复正常GDR路径。为核对网络能力，作者再用Ring测AllReduce，平均busbw约95.3GB/s且正确性检查通过。我的排查顺序会是单机通信、跨机单端口、GPU内存通信，最后回到原训练任务。",
            "公开实操_实践中哪些XID严重_怎么判断": "我会先看设备还能不能访问、错误影响哪个应用，以及驱动要求什么恢复动作。79代表驱动已经无法通过PCIe访问GPU，应先保留日志并按对应平台恢复流程处理，但原因可能在卡、链路或其他环节。内存错误里，支持错误隔离的GPU上，94先恢复受影响应用，95要求在应用重启前完成GPU重置；48是不可以忽略的不可纠正ECC，还要结合同时出现的错误和恢复提示。63是页退休或重映射的记录，64是记录失败。像公开H100案例，插件资源恢复8张后仍应继续核对ECC和任务执行，不能把资源数字回来等同故障修好了。",
            "客户在Kubernetes上用什么训练框架": "自愈组件工作在节点和设备层，对上层训练框架无感知，接入不区分框架类型。",
            "NVML怎么查询和订阅事件_超时能说明恢复吗": "NVML 提供设备查询和事件订阅两类接口，Linux 下对应 libnvidia-ml.so。事件订阅先创建 EventSet，为设备注册支持的事件，再通过 nvmlEventSetWait 等待；返回结果包含设备、事件类型，XID 事件还携带错误编号。温度、ECC 计数和当前设备状态按周期查询。等待超时只说明这段时间没有收到事件，恢复判定仍需要成功的设备复检。我们没有使用事件订阅这条路径，XID 由日志解析获得，NVML 侧只做周期性状态查询。",
            "DBE计数不再增长_检测器为什么还保留故障": "计数不再增长只说明没有新增错误，原有不可纠正 ECC 仍需要恢复确认。假设第一次读到 DBE 为 2，后面仍为 2，仅按本轮增量判断就会误清故障。现有检测逻辑处理首次读到非零 DBE 的情况，并通过 HasDBEError 保留已发现的故障状态；解除故障要依据对应恢复检查。",
            "ECC看累计值还是增量_重启后计数清零怎么处理": "两者都要保留，判断用途不同。累计值说明已经记录了多少错误，增量用来判断最近是否继续发生；可纠正错误还要结合统计窗口和阈值。当前 NVIDIA 的不可纠正 ECC 检测会处理首次读到非零 DBE 的情况，并保留已发现的故障，后续计数不增长不会直接清除。读取值小于上次值时，增量计算先按计数重置处理，避免无符号相减得到一个巨大错误数；计数清零本身不能作为恢复证据，还要复检设备访问和对应故障状态。",
            "有没有检测机器之间的RDMA或NCCL通信连通性": "这套 GPU 自愈组件没有接入跨机器的 RDMA 或 NCCL 通信检测，当前覆盖设备健康和节点内卡间互联。NPDPlus 的通用网络检查也不能替代这类训练通信检测。跨节点异常需要继续检查网卡、交换网络和集合通信结果，目前不会由这套 GPU 检测直接形成跨机通信故障 Condition 并启动维修。",
            "训练已经失败但NPDPlus正常_是不是漏检_要不要重启": "会，训练或推理任务失败时，节点和 GPU 仍可能正常。比如训练进程因 CUDA OOM 退出，GPU 设备本身仍可正常工作；推理服务也可能因模型加载失败或程序异常退出。NPDPlus 按已配置的检测项检查节点和 GPU，没报异常不能证明业务运行正常。排查时先对齐任务退出日志和同一时刻的设备、驱动记录，确认最先出现的错误。如果设备已经出现检测规则应识别、且达到触发条件的掉卡或 ECC 异常，却没有对应上报，再检查采集、规则匹配和 Condition 写入是否遗漏。仅凭任务失败，不触发节点重启。",
            "Kubernetes会补Pod_DevicePlugin会禁用故障卡_为什么还需要节点自愈": "Kubernetes 可以补建 Pod，Device Plugin 可以减少可分配设备，但故障 GPU 可能一直无法重新使用。节点自愈补的是故障检测后的维修过程：根据故障和维修额度决定是否下线节点，业务排空后执行恢复动作，再检查原故障是否消失。它的直接价值是把可恢复的故障节点重新变成可用算力，减少人工逐步操作。训练进度能否接着跑，还需要训练程序的 Checkpoint 和任务恢复配合。"
          },
          "implementation_details": {
            "对象模型": [
              "NodeHealer 表达一个厂商节点范围、故障场景、优先级、限流和模板引用，是可变的策略对象。",
              "MachineHealingTemplate 可由 Addon 提供经过审核的厂商预置模板，也可由平台管理员或客户创建自定义模板；它把经过审核的 Cordon、Drain、Reboot、UnCordon 等破坏性动作配置独立管理，NodeHealer 通过名称和 namespace 引用。",
              "HealingTask 是一次维修的持久化记录，里面写明目标节点、来源 Healer、触发 Condition 和动作步骤；策略后续变化不会改写正在执行的任务。"
            ],
            "模板独立CR的设计取舍": "MachineHealingTemplate 不必单独做成 CR，也可以内嵌在 NodeHealer 中。独立 CR 的价值是把可变的节点/故障策略与经过审核的破坏性动作配置分开管理，允许多个 NodeHealer 复用同一模板，并可独立交付、审计或按资源类型配置权限（实际权限取决于 RBAC 和平台 API）。代价是增加引用、namespace 解析、缺失对象和配置校验等复杂度。当前一个 NodeHealer 使用一个全局模板引用；Node Controller 在创建 HealingTask 时读取模板并将本次配置复制进任务快照，模板对象本身没有直接 watch，修改模板不会改写在途任务，也不会单独触发 Node reconcile。若没有复用、独立审核或权限边界需求，把模板字段内嵌到 NodeHealer 会更简单。",
            "NPD复用与Plus增强边界": {
              "社区NPD已有能力": "SystemLogMonitor 的日志读取、解析与规则匹配，CustomPluginMonitor 的外部检查执行，以及 NodeCondition、Event 和监控指标输出都来自已有框架。NPD 自身已有通用节点检查；不能写成不含检测逻辑，也不能把这些通用机制全部算作 Plus 新增。",
              "已有NPDPlus扩展": "本项目沿用公有云 NPD 扩展版本，复用 NVIDIA gpu-monitor、海光 dcu-monitor 和相关厂商日志规则。现有 NVIDIA 模块直接调用 NVML，并维护设备状态、ECC 计数和检查窗口。",
              "本人检测增量": "补齐昇腾、昆仑芯、沐曦适配，处理厂商 CLI/SDK 的输出差异、设备身份对应、故障分类和恢复判定；配置、脚本、Go 模块按实际取数方式选用，不把新增适配一概说成 Go 检测器。",
              "统一接口与有效结果": "沿用 Condition/Event/Metric 上报机制，补充厂商 Condition Type、Reason、Message 及恢复规则，并与 NodeHealer 和模板中的故障类型对齐。可恢复 Condition 清除前要求检查真实执行、相关设备覆盖完整且明确无异常；采集失败或数据缺失不能当成健康。",
              "与DCGM的关系": "DCGM 的 GPU 专用健康检查可以替代或补充一部分 NVIDIA 检测逻辑，再由 NPD 适配后上报。当前本地路径为 NVML 查询和 XID 日志；DCGM 接入仅作为可选架构说明，不作为已落地事实。",
              "职责边界": "NPDPlus 负责检测事实；维修持续时间门槛、开关和额度属于 NodeHealer 策略，动作执行属于 HealingTask。"
            },
            "异构检测与故障语义": {
              "厂商适配": "复用 NVIDIA/海光检测能力，补齐昇腾、昆仑芯、沐曦适配；底层分别消费厂商 SDK、管理工具和驱动日志，下游统一只认 NodeCondition、Event 和 Metric。",
              "统一维度": "覆盖设备丢失、健康状态、温度、ECC、PCIe/互联、驱动与固件等故障维度；需要持续维护的故障写 Condition，瞬时或仅用于观察的异常写 Event。",
              "健康门槛": "只有对应检查在本轮真实执行、相关设备覆盖满足要求、检测过程无错误且没有再观察到故障时，才允许把可恢复 Condition 写为 False。命令失败、字段缺失、检测器未运行或 Condition 消失时，继续等待有效检测结果。",
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
              "同节点在途任务检查": "按 nodeName 在创建任务前和真正操作节点前检查尚未结束的任务，已有任务时继续原流程或等待，减少重复派单和并发操作。List 检查与 Create 之间没有原子唯一约束；若要支持多执行者下的严格互斥，需要额外的唯一锁对象或 resourceVersion CAS。"
            },
            "HealingTask执行与恢复": {
              "Cordon": "给节点加自愈专用 NoSchedule taint，阻止普通工作负载继续调度；它不是绝对封锁，带匹配 toleration 的 Pod 仍可能进入。",
              "Drain": "通过 Kubernetes 驱逐语义移除可迁移 Pod，并遵守 PDB；工作负载是否重建、重建到哪里由 Deployment、StatefulSet 等控制器和调度器负责。DaemonSet、本地盘、静态 Pod 或 PDB 阻塞都可能让 Drain 等待或失败；替代 Pod 就绪和业务恢复由上层另行确认。",
              "Reboot": "腾讯云路径调用 CVM 重启并查询实例操作状态；TCS 路径通过受控 SSH 下发重启，以 boot_id 变化确认主机确实发生过重启。",
              "Recover": "重启完成只代表修复动作已经执行。Node Controller 继续检查 HealingTask 中记录的原始 Condition，要求它仍然存在、明确为 False 且稳定一分钟，才标记 Recovered；随后执行 UnCordon 并完成任务。",
              "状态推进": "HealingTask 创建事件触发 Controller reconcile 后，Controller 启动 handler；handler 按任务 Spec 中本次适用的步骤顺序执行，并在各动作完成后写入对应步骤状态。任务整体从 Processing 推进到 Recovering，再等待 Node Controller 根据原始 Condition 的稳定恢复结果写入 Recovered；收到后执行 UnCordon 并标记 Completed。普通步骤状态更新不会各自触发新一轮 Reconcile。",
              "慢动作与Reconcile worker": "在标准 controller-runtime worker 模型中，一个正在同步执行的 Reconcile 会占用一个 worker；若它等待 Drain、SSH 或云 API，等待期间就不能处理其他 key。worker 数量是有限并发上限，全部占满时其他请求排队。当前 HealingTask Controller 在 Reconcile 中启动 handler goroutine 后返回，因此慢维修步骤不会持续占用该 Reconcile worker；但 handler goroutine 和远程连接仍消耗进程资源，仍需限制并发、设置超时并持久化步骤状态。",
              "Failure": "如果流程已经执行过 Cordon，而后续动作失败、恢复超时或未取得有效的 GPU 健康复检结果，默认保留自愈 taint，避免故障节点重新承载业务；如果失败发生在 Cordon 之前，节点保持原有调度状态。节点随后恢复且模板包含 UnCordon 时才自动解除隔离，否则转人工。"
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
              "投递边界": "通知采用有限重试和阶段、渠道去重。发送成功后若记录结果失败，可能重复；超过重试上限也可能漏发。通知结果独立于主自愈流程。"
            }
          },
          "验证方式": "通过厂商输出样本、配置检查、自动化测试、日志回放和联调验收，检查原始信号映射、任务推进和恢复判定。",
          "实践案例": {
            "SSH断连后的重启结果确认": {
              "实践类型": "本人改动记录中的实现与测试设计；没有生产事故或本次测试运行记录。",
              "口述回答": "我在接 TCS 的 SSH 重启时，专门处理了一个结果不确定的问题：reboot 命令发出去后连接断了，可能是命令已经执行、机器开始重启，不能直接当失败再重试。实现上先读取旧 boot_id，持久化到节点注解，写成功后才发重启；再次处理发现这份证据还在，就先查状态。SSH 暂时连不上就继续等，重新连上但 boot_id 没变也不能算完成，只有变化才确认这次主机重启发生。代码里相应区分了断连、鉴权错误、旧 boot_id 缺失和 boot_id 变化；主机重启确认之后，GPU 能否回池还要单独复检。",
              "具体检查": "发命令前持久化旧boot_id；派发后断连保留证据；连接恢复后旧值不变继续等，值变化才确认重启发生。",
              "版本说明": "研发记录包含多轮变更，后面的保留证据改动替代早期错误即清记录的分支；不把拼接片段当可直接编译的最终文件。"
            },
            "昇腾少卡与检测源失败分开判定": {
              "实践类型": "厂商检测适配和样本测试代码；8→7和CE计数均为明确测试输入。",
              "口述回答": "做昇腾适配时，我把几种容易混淆的输出拆开处理。枚举从预期 8 张变成 7 张，测试要求连续三次才报 DeviceLost；但 npu-smi 根本没找到，不能也当成少卡，它进入检测源/驱动错误分支，不报设备丢失或 ECC 故障。另一个样本是可纠正 ECC 从 10 增到 12，低于高频阈值就只生成 Event，不置持续故障。对应代码给这些输入分别写断言，让异常类型和后续动作强度有依据。",
              "具体检查": "8→7 连续三次才产生 DeviceLost；npu-smi缺失进入DriverError而不报掉卡/ECC；CE 10→12低于高频阈值只生成临时Event。",
              "验证范围": "已核对测试输入和断言；未找到这组测试执行输出或对应真机事故。"
            },
            "多策略改造的兼容性评审": {
              "实践类型": "真实设计评审及后续代码分支。",
              "口述回答": "我们扩展同一节点匹配多个自愈策略时，旧逻辑靠 belong-to 直接找到唯一策略，改成遍历 selector 会影响共用组件的其他环境。评审时确定用默认关闭的 FeatureGate 保留旧路径，由 Addon 配置开启新能力。还有一个被指出的问题：如果出队时阻塞等待某个节点的锁，会挡住同队列其他节点，所以不能靠一直等锁推进维修。这个取舍后来体现在配置更新也分两条路径：新模式只更新策略缓存，避免走旧 belong-to 的队列清理。",
              "验证范围": "队列被阻塞是评审发现的设计风险；不写成实际生产死锁或压测收益。"
            }
          },
          "公开实践补充": {
            "使用方式": "以下是公开故障示例和处置参考，引用时说明来源，不当作本人经历。",
            "NVIDIA八卡只报七卡": {
              "公开记录": "GPU Operator官方排障示例中，Device Plugin记录某GPU的XID79并将其标为unhealthy，节点只报七张可分配GPU。",
              "排查次序": "先核对Node资源和Device Plugin日志中的GPU UUID，再对齐节点Xid/BDF；结合nvidia-smi与PCIe枚举区分设备被撤出分配和设备实际不可访问。后两项是结合项目故障分层补充的检查。",
              "只读命令示例": [
                "kubectl describe node <node>",
                "kubectl logs -n <namespace> <device-plugin-pod> -c nvidia-device-plugin --since=30m",
                "journalctl -k --since \"30 min ago\" | rg -i \"NVRM|Xid|AER|PCIe\"",
                "nvidia-smi -L",
                "lspci -D"
              ]
            },
            "严重程度与动作范围": {
              "要点": "79关注设备访问丢失；支持错误隔离的GPU上94先恢复受影响应用、95先重置受影响GPU。不能将海光XID160映射套用到NVIDIA。"
            },
            "排空留证再决定Reset": {
              "要点": "48需结合63/64及型号判断页退休或行重映射路径；执行Reset前确认应用已退出，并核对NVSwitch、Fabric Manager和虚拟化限制。保留系统日志、驱动版本及nvidia-bug-report，必要时按厂商指导诊断。"
            },
            "推理摘流的补充方案": {
              "方案": "将模型执行器/设备异常映射到服务未就绪状态，通过Readiness影响新流量；存量请求、重试和容量检查另由服务/入口处理。",
              "验证设计": "隔离测试里注入执行器不可用，观察健康接口失败、Pod Ready变化及新请求目的地，并持续记录在途请求与正常副本延迟。此验证尚未执行。"
            },
            "训练排空与推理摘流的官方实践依据": {
              "适用范围": "官方机制核对与联动方案补充。业务自行排空确认已支持；以下 Checkpoint、推理引擎健康与流量入口的自动对接仍在规划中，不将公开机制当作本项目已交付的业务联动。",
              "训练侧": "torchrun 在 worker 失败时按重启策略恢复 worker 组，应用需自行加载已有 Checkpoint。异步 Checkpoint 的 staging 完成与持久化上传完成是不同阶段，维修前必须确认选定检查点实际写入完成；具体 API 以框架版本为准。",
              "推理侧": "所核对的 vLLM 实现中，/health 调用引擎 check_health，EngineDeadError 返回 503。支持 shutdown_timeout 的版本可等待在途请求，到期终止剩余请求；是否配置到业务 Readiness 和排空流程需要部署侧接入，不能据此认定已自动摘流。",
              "PDB边界": "PDB 约束主动驱逐，不负责训练 Checkpoint，也不验证剩余推理容量；硬件故障等非自愿中断仍可能发生。"
            },
            "软硬重启_PDB与厂商差异的官方依据": {
              "适用范围": "用于面试回答的公开机制核对，补充软重启恢复判断、PDB 并发预算和厂商接口差异；不作为本人生产事故、厂商对比跑分或业务自动联动已交付的证明。",
              "软硬重启": "故障类型与厂商建议限定可选动作，主机 boot_id 变化只确认操作系统重启，设备恢复需独立复检；断电重上电和板卡维修不能仅凭一次 Reboot 失败直接等同。",
              "PDB": "Eviction 在删除前扣减预算并记录在途驱逐；任意工作负载可用整数 minAvailable，自定义资源按期望副本数计算预算需要 scale 等适配条件。",
              "沐曦": "cu-bridge 提供 CUDA 接口兼容与源码迁移路径；所核对 C500 mx-smi 手册中 dbe 是 PCI device base error，ECC 计数重启清零而坏页计数不清零。"
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
              "unatgw": "负责最终公网出口 SNAT；组件将受控流量导向平台已有的 NAT 出口。"
            }
          },
          "业务视角": {
            "直接用户与最终受益者": "直接使用者主要是平台运维、云产品交付人员和需要访问固定公网依赖的业务组件负责人。他们负责申请或创建规则；YUM、短信、邮件等业务 Pod 是规则入口的实际消费者。最终客户不需要理解代理 VM 和 Nginx 的维护细节，只需要按产品约定使用受管入口。",
            "典型业务场景": "具体有四类需求：COS 回源的 Underlay 机器需要公网域名解析，以前由交付侧临时搭 DNS Server；Pod 内的 NTP 拨测程序要访问公网一级源，受限于 Pod 的出网能力；YUM 容器化后需要通过专门的 rsync 代理同步公网软件源；短信、邮件组件需要访问固定公网消息网关。它们的目标和端口相对明确，适合逐条授权并共享代理资源。其中 DNS 需求只补解析，Pod 内 NTP 拨测还需要向指定 NTP 目标转发业务报文。",
            "一次业务使用流程": "授权方为固定公网目标创建 EgressForwardRule，选择协议和 ClusterIP 或 LoadBalancer 入口。HTTP/TCP/UDP 业务使用 status.entrypoints 返回的 Service 地址，TCP OriginalHost 也可以保留原域名；DNS 场景则由客户端先询问 RegionDNS，命中白名单 zone 才转入对应 DNS relay。Controller 分别维护每条规则的配置、健康后端、更新和删除流程。",
            "为什么只允许固定目标": "企业私有云的目标是满足必要的公网依赖，同时控制出口范围。把一个规则绑定到一个逻辑目标后，审批、白名单、配置发布、健康状态和审计记录都能对应到明确对象；如果允许客户端运行时选择任意地址，就会退化成通用代理，难以满足最小授权和问题定位要求。",
            "业务价值": "项目把按文档手工配置代理、解析和转发规则的流程改成规则驱动的自动交付，单次配置耗时从约 4 小时降到 10 分钟以内；由每个需求单独新增 KubeVM，转为多条规则共享 Proxy VM，相比原分散方案减少约 70% 的代理机资源使用。统一控制面与滚动发布减少多台 VM 配置不一致造成的访问异常，来源准入和固定目标约束降低误访问与代理滥用风险。扩容时新增 VM 自动同步现有规则，通过配置与健康检查后加入对应后端，交付人员不再逐条手工配置。",
            "业务成功标准": "控制面创建成功只表示规则已受理。真正成功要看到至少一个合格代理后端、入口发布完成，并由真实客户端通过受管入口完成目标协议事务；发生后端故障时，还要能够把问题 VM 摘流并让健康后端继续服务，或在全部不可用时留在受管入口内失败。",
            "量化口径": "单次配置耗时由约 4 小时降至 10 分钟以内，衡量代理、解析和转发规则的配置交付效率。代理机资源使用减少约 70%，比较多个需求分别新增 KubeVM 与共享代理 VM 的方案，统计范围是代理机资源。单次滚动更新另按 VM 数量、摘流等待、配置应用、检查和失败重试计算；配置一致性、安全约束和扩容自动纳管作为定性收益。",
            "能力边界": "组件为固定公网目标提供受控访问入口、规则管理和代理配置发布。集群已有的直连公网路径由原有路由和网络策略管理；任意互联网访问、全网流量接管或强制防绕行需要相应的网络与安全方案。"
          },
          "一句话": "我把固定公网目标的代理接入和多台 VM 配置维护做成统一控制面，让业务提交规则后自动获得可用入口。",
          "简历推荐写法": "设计并实现 Namespace 级 EgressForwardRule，支持 ClusterIP 和 LoadBalancer VIP 两类入口，以 selectorless Service 和受控 Endpoints/EndpointSlice 管理外部代理 VM；架构采用每 Rule 独立配置、ACL 和 revision，结合规则级容量检查、逐 VM 滚动发布及 VM 本地锁控制变更影响。项目将单次配置耗时从约 4 小时缩短至 10 分钟以内，并通过共享 Proxy VM 减少约 70% 的代理机资源使用。",
          "参考回答": "我把固定公网目标的接入和多台代理 VM 的配置维护做成了统一控制面。业务访问短信、邮件或软件源时，原来需要运维逐台修改配置，容易漏改或不一致。现在提交一条出口规则，就能自动创建访问入口、发布配置，并只让检查通过的 VM 接流量。",
          "项目状态": "Controller、代理 VM 安装包与配置工具、TAD、平台服务发现和真实协议链路已完成生产交付，简历中的逐 Rule、逐台摘流发布配置机制也已落地。流量监控、按规则限流和代理实例组隔离属于后续演进。",
          "配置与发布粒度": {
            "规则生命周期单位": "EgressForwardRule，也就是 Namespace 内的一条 CR；新增、修改和删除都从这条规则的生命周期进入。",
            "配置编译单位": "单条 EgressForwardRule。Controller 为当前 Rule 独立生成 nginx conf、ACL 和 manifest，并根据实际下发物料计算自己的 desiredRevision；不再把全部规则合并为一份配置包。",
            "发布执行单位": "当前 Rule × Proxy VM。同一 Rule 的目标 revision 按 VM 逐台执行预检、Drain、Apply、Doctor 和 Rejoin；摘流、失败恢复和后端状态都限定到该 Rule。",
            "流量准入单位": "规则 × Proxy VM；每条规则自己的 Endpoints 表达哪些 VM 已通过该规则的 listener、ACL 和健康检查，可以承载该规则流量。",
            "批量变更": "多条 Rule 各自入队、各自收敛。全局策略或 Product 节点列表变化时把所有 Rule 重新入队，每条规则只更新受影响的物料和后端；同一 VM 的共享 Nginx 切换与 reload 由 VM 本地短锁保护。",
            "粒度说明": "Rule 是生命周期和配置编译单位，Rule 与 VM 的组合是发布及流量准入单位；共享 VM 上的 Nginx 临界操作由本地锁串行保护。"
          },
          "个人实现与复用边界": {
            "我实现和交付的": "EgressForwardRule CRD、Controller 主链路、规则校验、Service/Endpoints/EndpointSlice 管理，以及代理 VM 配置生成和滚动发布；同时负责 Controller、代理 VM 安装包与配置工具、TAD 等交付物的接线与上线交付。",
            "我复用的": "Kubernetes Service/kube-proxy、现网代理 VM、Nginx 转发引擎和底层 NATGW；我的实现位于规则控制、配置发布和健康后端管理层。"
          },
          "链路说明": {
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
              "5. 配置发布：远端动作前持久化当前规则的发布门禁，逐 VM 执行 Validate、按需 Probe、该规则容量检查、Drain、Apply 和 Doctor。只摘除并恢复当前规则的后端；不同规则可独立推进，同一 VM 的共享 nginx 变更由 VM 本地锁保护。",
              "6. 入口开放与状态：ClusterIP 取 Service 的 ClusterIP，LoadBalancer 则等待 status 中的单个 IPv4 VIP；VIP 尚未分配时继续调和，不发布空地址。OriginalHost+ClusterIP 在后端可用后发布 service-id，OriginalHost+LoadBalancer 则发布指向 VIP 的受管 DNS A 记录，最后回写 Configured/Available。"
            ],
            "第二条_VM配置发布链": [
              "1. 为当前 Rule 准备独立 conf、ACL、manifest 和 desiredRevision；任何远端发布动作前，先持久化对应 generation 的 desiredRevision 与 Configured=False/RollingOut，建立连续修改门禁。",
              "2. 对目标 VM 核验 SSH 身份、VM 上的配置工具、运行环境和共享基础状态，Validate 当前 Rule 候选物料；新增规则、目标更新、FQDN IP 变化或新增 VM 时执行适用协议的 Probe，普通 UDP 首版没有通用 Probe。",
              "3. 更新已有 Rule 前，确认摘掉当前 VM 后本 Rule 仍至少有一个可服务后端；容量不足就暂停当前 Rule，其他 Rule 独立推进。新规则没有存量流量，可以逐 VM 收敛后逐步开放。",
              "4. 只从当前 Rule 的 Endpoints/RS 摘掉这台 VM，并等待入口传播，避免新流量进入待更新 listener；这不等于所有存量连接已经结束，也不摘除其他健康 Rule 的后端。",
              "5. VM 上的 apply 子命令校验当前 Rule 的可信回滚物料，在 VM 本地短锁内切换该 Rule conf/ACL，执行 nginx -t、reload 和实际物料检查；失败只恢复该 Rule 的上一确认版本。",
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
            "状态边界": "Available 表示当前规则至少有一个合格后端；LoadBalancer 还要求 VIP 已就绪。Configured 表示 Product 目标 VM 对该规则全部达到 desiredRevision、Doctor 通过，且相关平台资源已收敛；两者分别用于判断入口可用和配置收敛，业务验收通过真实客户端请求完成。"
          },
          "容器化YUM_OriginalHost_ClusterIP闭环": {
            "一句话主线": "原公网 FQDN 不变 → service-id 让集群侧把该域名导向受管 Service → Service 只选择健康代理 VM → Nginx 按固定目标转发 → NATGW SNAT 出公网。",
            "推荐口述版": "容器化 YUM 用户继续填写原公网 rsync 域名，URL 不需要改。上层平台据此生成保留原域名的 TCP 规则，也就是 TCP+OriginalHost。Controller 校验 FQDN 和端口，创建不带 selector 的 Service，再把监听端口、上游目标和 ACL 配置滚动发布到代理 VM。通过运行检查 Doctor 的 VM 才会进入 Endpoints，至少一个后端可用后才在 Service 上发布域名标识 service-id。平台服务发现读取这个标识，让集群客户端将原域名解析到 Service ClusterIP。请求随后经 Service 到健康代理 VM，由 Nginx 转发，再经 NATGW 访问公网仓库。",
            "记住这四步": [
              "用户入口不变",
              "Controller 先准备 Service 和 VM",
              "service-id 最后开放域名入口",
              "真实流量走 Service-VM-Nginx-NATGW"
            ],
            "用户体验": "用户继续配置和使用原始公网 rsync FQDN/URL。集群侧服务发现将该域名解析到受管 Service ClusterIP，客户端继续使用原域名和端口访问。",
            "规则契约": "上层平台生成 TCP + OriginalHost 的 EgressForwardRule，单一 upstream 必须是 FQDN，servicePort 必须等于 upstream.port；YUM/rsync 常见链路是原域名:873。端口必须相等，因为 service-id 能把目标地址映射到 Service，却不能改变客户端已经选择的目标端口。",
            "控制面闭环": "Controller 完成当前 TCP 规则的安全校验，独立生成并逐 VM 发布该 Rule 的 listener、upstream、ACL 和 revision；至少一个 VM 通过 Doctor 并进入受管后端后，再在 selectorless Service 上发布 infra.tce.io/service-id，host 仍是原公网域名。",
            "服务发现闭环": "Controller 在 Service 上发布 service-id 注解；平台的 service-id 消费器及 RegionDNS 同步路径读取该注解，将原公网域名在集群客户端路径解析到该 Service ClusterIP。",
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
            "为什么用无selector Service+受控Endpoints而不是ExternalName": "代理 VM 没有可供 Service selector 选择的 Pod 对象。无 selector 的 ClusterIP 或 LoadBalancer Service 提供稳定入口。ClusterIP 入口由公网代理 Controller 自管 Endpoints 和 EndpointSlice；LoadBalancer 入口由公网代理声明静态后端，再由平台 LB 组件投递和维护关联资源。后端准入依据每台 VM 对当前 Rule 的配置版本和检查结果。ExternalName 只做 DNS CNAME，不参与转发，也无法按 Rule 的 Doctor 结果管理多个代理后端。",
            "EndpointSlice维护者": "在 ClusterIP 自管后端路径中，selectorless Service 没有 Pod selector，内置 Controller 不会凭 VM label 自动发现集群外后端。公网代理 Controller 显式维护受管 Endpoints 和 EndpointSlice，给受管 Endpoints 标记 skip-mirror，避免自动镜像与主动维护冲突。Endpoints 和 EndpointSlice 的 controller ownerReference 都指向同 Namespace 的 Service，并保留 Rule UID 等受管元数据；共享环境和当前 Rule 的检查均通过，才发布该 VM 为本 Rule 后端。\n\nLoadBalancer 路径由公网代理声明静态后端，再交给平台组件维护后端资源；不由公网代理直接套用上述写入流程。",
            "CRD设计": {
              "为什么需要Namespace级资源": "出口规则属于具体租户和业务，Namespace 承载资源归属、RBAC 和生命周期管理。Service 的 controller ownerReference 指向同 Namespace 的 Rule，Endpoints 与 EndpointSlice 都归属该 Service。代理 VM 池、监听端口池和跨 Namespace 的域名冲突仍需统一协调，但每条 Rule 独立生成并发布自己的配置。",
              "Service/Endpoints为何跟随Rule所属Namespace": "Service、Endpoints 和 EndpointSlice 都是 namespaced 资源，跟随 Rule 创建可以保留租户资源边界。Controller 校验 Namespace、Rule UID、Service UID、标签、注解和 ownerReference，不接管归属不匹配的同名对象。集群级 DNSRecord 不能通过跨 scope ownerReference 归属 Rule，因此用 managed-by 和 owner rule UID、Namespace、name 建立归属并显式清理。",
              "跨Namespace冲突范围": "ExplicitService 的 HTTP/TCP/UDP 入口身份包含 Namespace，不同租户可以使用相同目标；DNS zone 和 OriginalHost FQDN 会进入共享服务发现平面，仍按集群范围判断冲突。listener targetPort 则由 Controller 从集群共享端口池按 Rule UID 分配，直到所属 Service 确认删除后才释放。",
              "Upstreams强制min=max=1的原因": "spec.upstreams 只允许一个逻辑公网目标，使白名单、状态和失败归因都对应明确对象。一个 FQDN 可以解析出多个通过安全校验的 IP，并作为同一目标写入本 Rule 的配置；多个业务目标则分别创建多条规则。DNS upstream 必须是固定公网 resolver IP，HTTP/TCP/UDP 按各协议校验 FQDN 或 IP，OriginalHost 要求 FQDN。",
              "accessMode为何单独设计": "省略 accessMode 等价 ExplicitService，客户端直接访问生成的 Service；OriginalHost 仅给需要保留原公网域名的 TCP 客户端使用，ClusterIP 通过 service-id 发布原 FQDN，LoadBalancer 通过受管 A 记录发布 VIP，并要求单一 FQDN 和 servicePort 等于 upstream.port。protocol、accessMode、servicePort、serviceType 决定入口形态，禁止原地修改；dnsZone、httpHost 和 upstreams 可以在通过协议校验、冲突检查及同 Rule 发布门禁后更新。",
              "字段校验如何分层": "API Server 的 OpenAPI Schema 校验基础类型、枚举和端口范围；支持 CEL 的版本可进一步校验组合和不可变字段。TCE Kubernetes 1.22 的交付物不携带 CEL，由 Validating Webhook 补充跨字段、不可变字段及连续更新门禁，Controller 再兜底目标规范化、地址安全和跨规则冲突。FQDN 统一转小写并去掉末尾点，只接受 ASCII 域名；非法解析结果不能进入新配置。"
            },
            "四类协议的共同点与差异": {
              "共同发布方式": "DNS/HTTP/TCP/UDP 共用规则校验、版本计算、逐 VM 发布、Doctor 和后端准入流程，但每条 Rule 都有自己的 conf、ACL、revision 和状态；更新一个协议的一条规则，不会合并重发其他规则。",
              "DNS": "DNS 规则的 Service 端口固定为 53，唯一 upstream 为公网 DNS Server IP:53，nginx stream 承担 TCP/UDP DNS 转发。固定 resolver IP 避免解析上游时依赖同一转发链，并把地址校验、ACL 和探测绑定到明确目标。DNS Probe 发送真实查询，收到包括 NXDOMAIN 在内的合法响应即可判为可达；不使用 httpHost，也不提供多 resolver 自动故障切换。",
              "HTTP": "HTTP 规则是固定目标反向代理，不是 HTTP CONNECT。客户端使用 status.entrypoints 返回的 Service 地址发送明文 HTTP，servicePort 省略时默认 80；本组件不为 HTTP 目标注册 CoreDNS 或 RegionDNS 解析。Nginx 连接已校验并写入当前 Rule 配置的公网 IP，以与 upstream host 规范化后一致的 httpHost 设置 Host、TLS SNI 和证书校验目标，并启用可信 CA 校验。需要客户端保持端到端 TLS 时使用显式授权的固定目标 TCP 规则。",
              "TCP/UDP": "都要求显式 servicePort。ExplicitService 可由 Service 端口映射到独立 upstream.port；OriginalHost 只支持 TCP，要求 FQDN 且 servicePort 等于 upstream.port，因为域名注册只替换目标地址，不能改变客户端选择的端口。普通 UDP 协议异构，首版不提供通用 UDP Probe，协议可达性通过对应客户端的实际请求验证。"
            },
            "Controller并发模型": "每条 Rule 使用独立 workqueue key、配置和状态调谐；全局策略或 Product 节点列表变化时将全部 Rule 重新入队，各自收敛，不使用 Controller 级全局串行锁。同一 VM 上只有一个共享 Nginx，VM 上的配置工具在文件和 ACL 切换、nginx -t、reload 及失败恢复的短临界区内加本地锁。不同 Rule 可以独立推进，但这些外部动作不构成跨 VM 或 Kubernetes 资源的原子事务。",
            "一次Rule调谐的执行顺序": "先处理删除或安全停用，关闭本 Rule 入口；正常调谐时校验目标、冲突和资源归属，准备 Service 并读取已分配 targetPort，再生成本 Rule conf、ACL 和 desiredRevision。远端发布动作前持久化当前 generation 的 desiredRevision 及 Configured=False/RollingOut 门禁；随后逐 VM 预检、检查容量、摘流、Apply、Doctor 并维护本 Rule 后端。新规则在首台后端与入口就绪后发布所需域名入口，已有规则滚动完成后按需更新 DNSRecord，最后回写后端观测和 Conditions。",
            "分层抽象": {
              "RemoteBuilder": "代理 VM 上部署一个配置管理工具，提供 validate（校验配置）、apply（应用配置）、doctor（检查运行状态）等子命令，Controller 通过受限 SSH 调用并读取执行结果。工具还提供删除规则、查询当前版本、解析域名、探测目标和列举规则的子命令。一次 apply 在 VM 本地锁内切换当前规则的配置和 ACL，失败时恢复该规则的原配置；跨 VM 的发布顺序和后端摘流由 Controller 协调。",
              "forced-command安全边界": "生产环境通过 SSH forced-command wrapper 访问 VM：公钥认证成功后，authorized_keys 中的 command= 强制启动该入口程序，原始请求从 SSH_ORIGINAL_COMMAND 读取，同时禁用端口、Agent、X11 转发和 TTY。入口程序检查子命令是否在白名单内，再校验参数、request ID、revision 和上传内容，只调用固定路径的 VM 配置工具。远程请求只能执行预先允许的操作，不能提交任意 shell 命令。",
              "DataPlane": "当前 Rule 的发布计划：生成独立候选物料和 revision，安排目标 VM 预检、容量检查、逐台更新和恢复，不再维护全部规则共用的全局 revision。",
              "BackendWriter": "统一维护当前 Rule 的 Endpoints/RS 和 EndpointSlice 投影，执行摘流、恢复与重新加入；只有共享环境及当前 Rule 检查均通过的 VM 才能接流，不能因其他 Rule 成功而替它开放。"
            },
            "为什么改为每条Rule独立revision": "全局配置包使一条规则变化也重发全部规则，扩大摘流、校验和失败恢复的影响面。现在每条 Rule 独立生成完整 conf、ACL 和 manifest，desiredRevision 覆盖实际下发的规范化配置、targetPort、已校验 upstream IP 及物料格式版本；逐 VM 只更新当前 Rule。单 Rule 仍保留确定性版本、完整回滚物料和周期盘点；多台 VM 逐台更新，期间可以短暂存在新旧版本，后续通过调和完成收敛或回滚。",
            "确定性渲染": "先恢复当前 Rule 的 targetPort，再规范化域名、IP、CIDR 和协议参数，对去重后的地址及 ACL 条目采用稳定排序，避免直接用 Go map 遍历顺序输出物料。SHA-256 计算的是本 Rule 实际下发的规范化 conf、ACL、监听端口、upstream IP 和格式版本，相同物料必得相同 revision；仅影响 Service/DNSRecord 而未改变 VM 物料的字段更新，不触发无意义 reload。",
            "同一Rule连续修改怎么避免版本叠加": "Controller 在远端摘流或 Apply 前，先持久化目标 revision 和 Configured=False/RollingOut 状态；Validating Webhook 在 API Server 的更新准入阶段据此限制当前 Rule 的 Spec 修改。尚未进入远端发布时，已成功提交的新 generation 可以替代未下发候选；执行结果未知或目标 revision 已在部分后端生效时，先完成本轮发布或恢复。首次失败已确认且没有后端应用目标 revision 时允许修正。被拒绝的更新不会写入 CR，也没有自动排队，调用方需重新提交；仅修改 metadata/status 不参与这项 Spec 门禁。",
            "targetPort不可变": "targetPort 是本 Rule 在代理 VM 上的稳定 listener 端口，写入受管 Service 并作为端口占用的唯一事实源；Controller 重启后 list 全部受管 Service 重建分配表，同 Rule 生命周期内不换端口。Service、Endpoints 和 VM 物料必须使用同一分配结果。全局 listenerPortRange 与 Product 安装参数需使用同一现场值，当前不自动跨 Application 同步；VM 上的配置工具对比候选端口池与 VM 已安装值，不一致拒绝 apply。端口池在组件生命周期内不可变，换池需要维护窗口关闭规则并重装数据面。",
            "单台VM滚动步骤": "先确认 SSH 身份、VM 上的配置工具及共享运行环境，再对当前 Rule 执行 Validate 及按场景需要的 Probe；预检不通过，不触碰该 VM 的旧规则配置。更新已有 Rule 时，确保移除当前 VM 后本 Rule 仍至少有一个可服务后端，才从本 Rule 的 Endpoints/RS 摘流并等待入口传播；随后 VM 上的配置工具在本地短锁内切换本 Rule conf/ACL、nginx -t 和 reload。共享项和本 Rule revision/listener/ACL 的 Doctor 均通过后才加回当前 Rule，再滚动下一台；固定等待结束后仍可能存在未关闭的存量连接。",
            "容量下限与幂等恢复": "容量下限保护当前 Rule 的计划内滚动，摘掉当前 VM 会让本 Rule 无可服务后端时暂停；已明确不安全的后端仍需隔离。某 VM 发布失败就停止该 Rule 后续滚动，已成功 VM 保留新版本；失败 VM 恢复旧物料并通过 Doctor 后，可以旧 revision 加回本 Rule。Apply 失败或响应丢失时先查本 Rule current-revision 和 Doctor：目标版本可信且健康则继续，旧版本可信且健康则恢复本 Rule 后端，无法确认时保持摘流；回滚必须有经验证的旧物料锚点，不能仅凭旧 revision 字符串盲目执行。",
            "故障分级隔离": "接入预检确认 SSH 身份、VM 上的配置工具及运行环境；Doctor 的共享项检查 nginx 进程、基础 ACL 和 unatgw，明确不可信时从该 VM 的所有 Rule 后端摘除。规则项按 Rule UID 检查 revision、listener 和 ACL，仅本 Rule 不符合时只摘它自己的后端。暂时 SSH 观测失败与有效检查明确失败要分开处理，身份指纹不符不能静默接受。Doctor 默认每 60 秒巡检本地运行状态；公网 upstream 的 Probe 在发布触发条件满足时执行，业务成功通过真实客户端请求验收。",
            "两类源地址范围分别管什么": "regionDNSSourceCIDRs 限制 DNS listener 实际看到的来源，serviceForwardSourceCIDRs 限制 HTTP/TCP/UDP 经 Service 或 LB 转发后在 VM 上呈现的来源；两者是 listener 入向 ACL，应按现场真实转发源配置，不是公网目标白名单，也不是 Kubernetes loadBalancerSourceRanges。allowedEgressCIDRs 才限制 upstream 目的地址；来源与目的范围目前都按全局策略配置。Namespace/RBAC 控制 API 对象权限，报文准入由 VM 上的来源网段 ACL 实施。",
            "规则删除的finalizer流程": "先清空本 Rule 的 Endpoints/RS 和 EndpointSlice 后端并等待入口传播，再删除其拥有的 DNSRecord，逐 VM 按 Rule UID 删除 conf、ACL 和 manifest，并在本地锁内校验、reload 和检查；全部必要清理完成后再删 Service/Endpoints 并移除 finalizer。OriginalHost+ClusterIP 的 service-id 随 Service 清理；OriginalHost+LoadBalancer 撤销 A 记录后须保留 Service/VIP 至少 120 秒加该 VIP 曾观测的最大 TTL，等待结束才释放。远端删除失败保留 finalizer 重试。安全校验失败则立即关闭本 Rule 入口并清理物料，CR 保留供修正，不当作用户删除。",
            "落地场景举例": [
              "消息网关短信/邮件：以前需要人工建代理机手动维护 nginx 配置，现在运维创建 HTTP/TCP 规则声明目标域名和端口，Controller 建独立 Service 和独立 nginx listener，短信/邮件 Pod 直接访问该规则专属 Service。",
              "容器化 YUM rsync：上层把用户的原公网 FQDN/873 转成 TCP+OriginalHost 规则；ClusterIP 用 service-id 导向 Service，LoadBalancer 用受管 A 记录导向 VIP，用户 URL 均不变，后续共用健康代理 VM、Nginx 和 unatgw 出公网链路。"
            ],
            "失败恢复需要保留哪些物料": "每条 Rule 至少保留当前和上一已确认成功的不可变 revision 目录，目录内包含 conf、ACL plan 和 manifest，不额外维护 .bak。更新前读回旧物料并校验摘要及 current 指向，一致时才能作为可信回滚锚点；候选 conf/manifest 写入后逐项读回比较，ACL 激活后与实际 nftables 规则结构化比对。只有运行检查确认后提交 appliedRevision，失败恢复当前 Rule，不能改写其他 Rule 的物料。",
            "同revision为什么还要检查和修复": "相同 revision 下，文件、ACL 或进程状态仍可能发生漂移，因此恢复时要同时核对版本和实际运行状态。Controller 仍查询当前 Rule 的 current-revision 和 Doctor；物料及运行检查均通过时只修复 Service、Endpoints/RS、DNSRecord 等平台资源，不重复 reload；同 revision 但运行状态异常时先关闭本 Rule 在该 VM 的流量，再幂等重放并检查后恢复。",
            "逐Rule更新后如何发现历史漂移和孤儿配置": "Controller 周期读取 Kubernetes Rule 集合，并调用 VM 上的 list-rules 子命令盘点 VM 上所有受管 manifest 和实际配置。缺失物料补发，revision 或实际摘要不符时修复，已无对应 Rule 的物料按安全删除流程清理；Rule 文件名使用 r_<ruleUID>.conf，manifest 记录 Namespace、name 和 UID，避免同名 Rule 删除重建后误认旧配置。增量写入缩小变更范围，周期全量核对保证最终一致性。",
            "状态字段怎样表达可用性和收敛": "每条 Rule 用 Available 表达当前是否有合格后端且入口就绪，LoadBalancer 还要求 VIP 可用；Configured 要求 Product 节点列表非空、全部目标 VM 的本 Rule appliedRevision 达到 desiredRevision、Doctor 通过且相关 Kubernetes 资源收敛。observedGeneration 表示处理到的 Spec，appliedGeneration 在该代完全收敛后推进。backendStatuses 按 IP 记录本 Rule appliedRevision、serving、lastStep、reason 和 message，不另建 phase；恢复时查询 VM 真实物料及检查结果决定是否接流，业务验收通过客户端完成目标协议请求。",
            "Product节点列表和SSH身份分别从哪里恢复": "代理节点成员的唯一来源是 TAD 根据 Product Application Component nodes 渲染的 tce/product-tcs-egressproxy-config 中 proxy-nodes.conf，Controller 只读且不经 local.json 中转。节点列表表示交付声明；新 VM 完成安装，并通过 SSH 身份、VM 配置工具与 Nginx、端口池、unatgw 及本 Rule apply/doctor 检查后才接流。tce/egress-proxy-endpoints 只保存 IP 到 SSH hostKey 的 TOFU 固定结果，不决定成员。节点配置缺失或为空时 Controller 继续运行、后端为空并报告 WaitingForProxyEndpoints；配置恢复后各 Rule 重新入队收敛。",
            "代理VM扩缩容和Product升级怎么配合": "新增 IP 先进入 Product 节点列表，Controller 将全部 Rule 入队，待 Product 安装完成后逐 Rule 补齐新 VM 配置并检查加回；新节点失败不触发已有健康 VM 重发或摘流。缩容或升级先从 Product 列表移除 IP，Controller 从所有 Rule 后端摘除，集群侧交付任务只读确认摘流和传播后才远端卸载、升级或销毁。Product 负责 Compose 常驻 Nginx 的启动与安装，Controller 只调用 VM 上的子命令增删规则并执行 reload；runtime 升级还需同步宿主 Controller 期望版本，升级 VM 恢复原有全部规则后再滚动下一台。"
          },
          "高压追问参考回答": {
            "所有公网流量都会经过这个组件吗": "不会。HTTP/TCP/UDP 的受管代理请求要进入对应规则的 Service，再由代理 VM 转发；DNS 规则只补齐解析链路，客户端先问 RegionDNS，命中白名单 zone 后才经 DNS relay 到公网 resolver，已具备 unatgw 的 Underlay 业务拿到公网 IP 后从自身出口访问目标。组件统一的是这些场景的规则管理，不透明接管整个集群的所有公网流量。",
            "这个组件的直接用户是谁": "平台最终服务企业客户，但这个组件的直接使用方主要是平台运维人员，以及运行在 TCS 集群里的上层云产品和业务组件。例如 YUM、短信、邮件等业务需要访问固定公网依赖时，由授权方创建规则，业务 Pod 再通过对应的受管入口访问。",
            "为什么企业私有云需要受控公网出口": "私有云不能因为少量外部依赖就给所有工作负载开放任意公网，但 YUM、短信、邮件等业务确实需要访问固定目标。受控公网出口是在保留网络隔离的前提下，为这些必要依赖提供明确入口和固定白名单，同时把规则变更、代理配置和故障处理统一管理起来。",
            "为什么只支持固定目标_不做任意公网代理": "固定目标符合这类业务的最小授权需求：一个规则对应一个公网依赖，审批、配置、健康和审计都有明确对象。通用 CONNECT 或 SOCKS 允许客户端临时选择任意目标，会扩大出口权限，也让安全校验和故障归因更困难，因此不属于这个组件的定位。",
            "这个项目的业务价值怎么衡量_有没有数据": "单次配置耗时从大约 4 小时降到 10 分钟以内，代理机资源使用相比原方案减少约 70%。耗时对应的是以前手工配置代理、解析和转发规则，到现在自动交付的变化。资源节省来自多条规则共享代理 VM，替代以前每个需求往往单独新增 KubeVM 的方式。统一管理和滚动发布还减少了配置不一致造成的访问异常，来源准入与目标约束控制出口范围，新 VM 也能自动同步现有规则。",
            "客户需要访问任意公网地址怎么办": "这超出了固定目标出口组件的范围，需要走单独的网络与安全评审，选择通用代理、独立网关或其他受控上网方案。EgressForwardRule 继续按一个规则对应一个固定目标管理，便于逐条授权、校验和审计。",
            "Pod访问GitHub会自动经过代理吗": "不会，需要先配置规则，而且应用必须进入受管入口。例如要允许访问 github.com:443，先创建对应的 EgressForwardRule，声明目标、协议和端口。Controller 完成目标校验、Service 创建和代理 VM 配置发布后，业务通过这个入口访问，流量才会经过代理。",
            "DNS请求也必须经过Service吗": "本组件纳管的 DNS 查询先到 RegionDNS，只有命中显式 dnsZone 的请求才转给该规则的 DNS relay Service，再经代理 VM nginx stream 到一个公网 resolver。普通客户端不能直接访问 relay，否则会绕过 zone 白名单；限制依据是代理 VM 实际看到的 RegionDNS 转发来源。它不转发所有 DNS 查询，也不替换原有内网解析。",
            "没有代理VM_Pod就不能出公网吗": "对我们纳管的交付场景来说，这些 Pod 没有获准的默认公网路径，需要借助已经准备好 unatgw 的代理 VM 访问公网。不过这是部署环境的网络前提，不是这个 Controller 单独实现的阻断能力；如果要求强制所有相关流量必须经过代理，还需要路由、防火墙或 NetworkPolicy 等外部网络策略配合。",
            "代理VM在集群内还是集群外_可以跨集群共享吗": "代理 VM 位于集群外的 Underlay 网络，由当前集群的公网代理 Controller 纳管。Controller 为本集群的规则创建 Service，并把对应 VM 写入后端。当前实现没有供多个 Controller 安全共享同一 VM 池的协议，跨集群共享尚不受支持。",
            "正向代理和反向代理有什么区别": "正向代理代表客户端访问外部服务，服务端主要看到代理；反向代理代表后端服务接收客户端请求，客户端不需要知道真实后端。判断重点是代理代表哪一侧，而不只是客户端是否知道代理存在。",
            "先简单介绍一下项目背景和原方案问题": "比如业务要接入一个短信公网接口，原来要找运维准备代理 VM，再逐台修改 Nginx 配置，后续修改和删除规则也需要人工处理。我的组件把它变成提交一条出口规则，由 Controller 自动准备入口、发布配置并维护健康后端。底层继续使用已有的代理 VM 和公网 NAT 出口。",
            "原来三类出公网链路分别是什么_为什么要统一": "一类是已有 unatgw 出口、但缺少公网解析的 COS-CGI、部分 NTP 和虚拟机 YUM 等 Underlay 业务；一类是要人工创建代理机、维护 nginx 的短信邮件业务；还有一类是容器化 YUM 自己维护的 rsync 代理和专属 VM。统一规则声明、交付和维护，可以共用代理 VM 并减少人工配置；DNS 场景仍保留客户端自己的出网路径。另有 Pod 内访问公网一级 NTP 源的拨测需求，它还缺业务出网链路，不能归入只补 DNS 的场景。",
            "这是不是通用ForwardProxy": "不是。EgressForwardRule 创建时已经固定目标、协议和端口，Controller 会把目标解析并校验成确定 IP，Nginx 只能转发到配置中的 upstream。HTTP 模式是固定 httpHost 的反向代理，不接受 CONNECT；TCP/UDP 也是一条规则一个固定目标，不支持 SOCKS 或由客户端运行时选择任意公网地址。",
            "为什么代理放VM不放Pod": "主要是当前交付环境的网络约束：已有独立代理 VM 到 NATGW 的可用出口，Underlay 出网组件又会修改主机网卡和默认路由，直接装在 TCE 工作节点上会与底座网络管理冲突。因此由独立代理 VM 承担出网转发，Pod 通过受管 Service 入口访问。Controller 统一管理规则与配置发布，代理 VM 保留已有公网出口。",
            "unatgw已经能出公网_为什么还需要这个组件": "VM 能出公网，只解决了出口连通性。业务要使用这台 VM 访问指定目标，还需要配置代理监听端口、上游目标和访问入口，这部分原来主要靠人工完成。\n\n比如新增一个短信或邮件服务，运维需要逐台登录代理 VM 修改 Nginx，再把入口交给业务。多台 VM 可能出现部分修改成功、部分失败，删除和变更规则也要重复操作。\n\n我做的组件把这些步骤自动化：用户提交出口规则，Controller 创建入口、为该规则生成配置并逐台发布，确认代理状态后维护 Service 后端。多个业务也能复用同一组代理 VM，底层出公网继续使用已有 NATGW。",
            "为什么用CRD不用ConfigMap": "出口规则需要独立的结构和处理状态，CRD 可以把协议、目标、端口等字段定义成 API，并通过 status 单独回写发布结果。这样规则校验、查询和控制器处理都围绕同一种资源展开，也方便只授权操作出口规则。ConfigMap 可以保存配置文本，但规则格式、校验和执行状态的存放方式还得自行约定。选择 CRD 的代价是多维护一套 API 定义、版本兼容和控制器。",
            "EgressForwardRule为什么需要归属Namespace": "可以按业务分别管理出口配置，并复用 Kubernetes 的权限控制。比如业务 A 和业务 B 各有自己的 Namespace，都可以创建名叫 yum 的显式 Service 模式规则；平台可以分别授权，让各自的用户只能修改获授权的规则。规则和生成的 Service 放在同一个 Namespace，也方便一起查询和清理。DNS zone 和 OriginalHost 域名仍进入共享服务发现范围，需要跨 Namespace 检查冲突。",
            "租户隔离和规则权限具体做了什么": "我把出口规则和生成的 Service 按 Namespace 管理，平台可以通过 RBAC 限制用户操作哪些 Namespace 下的规则。共享代理上，每条规则有独立的监听端口、配置和版本，分别更新和清理。流量进入代理时，通过 nftables 检查来源网段、协议和监听端口。目前来源白名单按全局配置，没有做到每个租户只能访问自己的出口规则。",
            "RBAC怎样限制用户操作Rule_用户属于某个Namespace吗": "API Server 收到用户查看或修改 Rule 的请求时，会检查这个身份在目标 Namespace 下是否有相应权限。比如给用户 A 配置 RoleBinding，绑定 team-a 下允许操作出口规则的 Role；如果 A 没有获得 team-b 的相应权限，修改那里的规则就会返回 403 Forbidden。用户身份由认证系统提供，RBAC 负责授权；同一个用户可以获得多个 Namespace 的权限，查看、列举、监听、创建、修改和删除也能分别授权。用户配置 CR 的权限，与 Pod 访问代理入口时的来源 CIDR 检查分别生效。",
            "Namespace级CR能否保证租户流量隔离": "不能。其他 Namespace 的 Pod，只要能访问对应的代理入口，并通过来源白名单，也可能使用这条规则。Namespace 配合 RBAC 控制的是谁能通过 API Server 管理规则；业务请求进入代理时，按网络可达性和代理实际看到的来源判断，不检查调用方是否拥有这条 CR 的管理权限。\n\n追问：把 Rule 放在不同 Namespace，就实现网络隔离了吗？\n实现的是规则和关联资源的隔离。项目里验证了两侧同名规则的入口和端口独立，删除一侧不影响另一侧。限制谁能访问哪个网络目标，还需要由 ACL 或具备支持能力的网络插件执行相应策略。",
            "业务方伪造来源IP能否访问其他租户的规则": "如果底层网络允许伪造来源 IP，确实可能绕过来源白名单。当前更直接的问题是，同一个允许网段里的业务，可能不用改 IP 就能访问另一条规则的入口。要实现严格租户隔离，需要底层网络校验来源 IP 与虚拟网卡的绑定，再根据可信来源限制每个租户能使用哪些规则；如果转发途中做了 SNAT，还要确认代理能否识别原始来源。直接修改规则里的 CIDR 则属于配置权限问题，应由相应的 API 授权限制。",
            "多个Namespace的规则怎么汇总和判定冲突": "Controller 在集群范围检查共享名称和端口冲突，但每条规则独立调和，不把全部规则合并成一个配置包。ExplicitService 的 HTTP/TCP/UDP 入口位于各自 Namespace；DNS zone 和 OriginalHost FQDN 涉及共享服务发现，需要跨 Namespace 检查。VM listener 端口来自共享池，分配结果以受管 Service 的 targetPort 持久化；全局策略或节点变化时，把各规则分别重新入队。",
            "LB_VIP还没分配或发生变化怎么处理": "新建 LoadBalancer Service 尚无唯一可用 IPv4 VIP 时，规则保持 Available=False/WaitingForServiceAddress，不发布空入口。OriginalHost 的 A 记录绑定 Service UID 与 VIP；VIP 或 Service 身份变化时先关闭本规则后端、撤销旧记录，保留 Service/VIP 至少 120 秒加该 VIP 曾观测的最大 TTL，再按新的入口重新发布。RegionDNS 没有撤销 ACK，这段传播等待要求 LB 持续保留旧 VIP。",
            "observedGeneration_appliedGeneration_revision分别表达什么": "observedGeneration 是 Controller 已处理的 Spec 代数，appliedGeneration 只在当前规则的数据面与平台资源完全收敛、Configured=True 后推进。desiredRevision 是这条规则实际下发物料的摘要，不是全局版本，也不直接对整份 Spec 求哈希。backendStatuses 按 VM IP 记录该规则的 appliedRevision、serving、lastStep 和原因，只是最近观测；恢复时仍要查询 VM 的 current-revision、实际物料和 Doctor。\n\n追问：怎么避免拿旧状态判断新配置？\n核对 Rule 的 `status.observedGeneration`、`status.appliedGeneration` 与当前 `metadata.generation` 一致，再检查每台 VM 的 `appliedRevision` 是否等于 `desiredRevision`。",
            "项目整体链路怎么讲_控制面和数据面怎么划分": "一条请求从业务 Pod 出发，访问 Service 后，被转发到代理 VM 上的 Nginx，再经 NATGW 到达固定公网目标。Controller 负责让这条路径可用：用户提交 EgressForwardRule 后，它创建 Service、把配置发布到代理 VM，并维护合格后端。发布新版本时先检查容量、逐台摘流，Apply 后通过检查才重新接流；Controller 本身不转发业务报文。\n\n追问：请求会经过公网代理 Controller 吗？\n不会。Controller 负责配置和后端管理；业务报文经过节点转发规则到代理 VM，再由 Nginx 访问公网。",
            "Controller会不会并发跑两轮reconcile": "同一 Rule 以自己的 workqueue key 顺序处理，不同 Rule 可以并发推进。全局策略或代理 VM 列表变化时，把各条 Rule 重新入队。多个 Rule 更新同一台 VM 时，VM 上的配置工具在切换文件、更新 ACL、执行 nginx -t 和 reload 期间持有本地锁，失败恢复也在这个锁内完成。",
            "为什么selectorlessService": "后端是代理 VM 的 IP，Pod selector 无法选出这些地址，所以 Service 不设置 selector。Controller 把健康 VM 的地址和监听端口写入后端对象，节点上的 Service 数据面据此转发。",
            "Service_Endpoints_EndpointSlice分别负责什么": "Service 提供 ClusterIP 或平台 LB VIP 和端口，本身不发现 VM，也不做 Doctor 健康检查。ClusterIP 入口由公网代理 Controller 自管 Endpoints 和 EndpointSlice；LoadBalancer 入口由公网代理声明静态后端，再由平台 LB 组件投递和维护关联资源。ClusterIP 模式下，受管 Endpoints 保存本规则可以接流的 VM IP:targetPort，EndpointSlice 同步相同后端列表供集群转发组件消费；受管 Endpoints 标记 skip-mirror，避免自动镜像与主动维护冲突。",
            "为什么Endpoints是权威而EndpointSlice只是投影": "在 ClusterIP 自管后端路径中，如果 Controller 分别把 Endpoints 和 EndpointSlice 当作两个可独立修改的真相，就可能出现两边后端集合不一致，不同转发组件观察到不同结果。当前 Controller 先从受管 Endpoints 恢复某条规则正在服务的 VM，所有摘流、恢复和重新加入都先修改 Endpoints，再同步 EndpointSlice；EndpointSlice 同步失败时不会把本轮当成完整成功。因此 Endpoints 是流量准入的权威状态，EndpointSlice 只是向 Kubernetes 转发面发布同一后端集合的投影，不能独立增加 VM。\n\nLoadBalancer 路径由公网代理声明静态后端，交给平台组件投递，不套用这套直接维护 Endpoints 的流程。",
            "为什么不用ExternalName_LB_VIP解决什么": "ExternalName 只发布 CNAME，无法用受控 Endpoints 按 VM 健康状态快速摘流。ClusterIP 适合集群内 Pod；节点和集群外 Underlay VM 需要可路由入口时，规则选择 LoadBalancer Service，由平台内网 LB 分配 VIP，两种类型仍共用 Controller 维护的健康代理 VM 后端。",
            "三个端口分别是什么": "servicePort 是客户端访问 Service 的端口，targetPort 是代理 VM 的 listener 端口，upstream.port 是公网目标端口。例如两条规则可以共享一台代理 VM，分别使用 20001 和 20002 作为 targetPort，各自转到固定公网目标。ExplicitService 可以映射入口与目标端口；OriginalHost 要求 servicePort 等于 upstream.port。",
            "ExplicitService和OriginalHost区别": "ExplicitService 是默认模式，客户端直接访问生成的 ClusterIP 或 LB VIP；OriginalHost 只用于 TCP，客户端继续访问原公网 FQDN。OriginalHost+ClusterIP 用 service-id 导向 Service，OriginalHost+LoadBalancer 则用受管 DNS A 记录导向 VIP。",
            "YUM用户URL改了吗": "没有。用户仍使用原 rsync URL，ClusterIP 路径通过 service-id 把 FQDN 导向 Service ClusterIP，LoadBalancer 路径通过受管 A 记录把 FQDN 导向 VIP。Controller 不修改 URL，Service 也不解析公网域名。",
            "用户是否真的只填域名": "是，产品层面用户只需要配置原公网域名。生产平台将它转换为保留原域名的 TCP 出口规则 EgressForwardRule，也就是 TCP+OriginalHost，并补齐相等的 servicePort 和 upstream.port。Controller 接收这条完整规则，继续处理入口和代理配置。",
            "service-id和DNSRecord什么关系": "OriginalHost+ClusterIP 在 Service 上发布 service-id，让平台服务发现把原域名导向 ClusterIP；OriginalHost+LoadBalancer 不发布 service-id，而是在已授权 zone 下发布指向 VIP 的叶子 A 记录。protocol=DNS 的 DNSRecord 用于 zone 转发，是另一种资源语义。",
            "为什么OriginalHost只做TCP": "生产需求是 rsync 的 TCP 原域名兼容，当前 OriginalHost 契约也按 TCP 定义。HTTP 还有 Host、SNI 和证书语义，UDP 缺少连接态，它们继续走显式 Service 模式。",
            "为什么端口必须相等": "服务发现能把域名对应的目标地址替换成 Service ClusterIP，但客户端已根据 URL 选择了目标端口；如果 servicePort 不同，没有额外端口重定向层就到不了 Service 正确入口。",
            "为什么最后才发布OriginalHost域名入口": "先确认 VM listener、ACL、配置版本和健康检查正常，并且至少一个 Endpoint 可用，再发布 ClusterIP service-id 或 LB VIP A 记录。这样可以避免原域名已经指向新入口，但后端尚未就绪。",
            "后端全挂为何不立即撤销域名映射": "立即撤销 service-id 或 A 记录可能让原 FQDN 回退到真实公网解析，客户端反而绕过代理。保留已验证的域名映射并清空 Endpoints，会让请求在受管入口失败；VIP 或 Service UID 确定变化时，再走受控撤销和 TTL 等待。",
            "怎么证明域名入口已生效": "ClusterIP 路径要同时看 Service 上的 service-id、外部消费者的观测状态和真实客户端解析/rsync；LoadBalancer 路径要看 Service VIP、与 Service UID 绑定的受管 A 记录和真实客户端流量。验收同时核对入口配置、客户端实际解析结果和目标协议请求。",
            "怎么防DNS重绑": "由 Controller 组织解析并校验每个返回 IP，确保满足公网地址和 allowedEgressCIDRs 策略，再把同一组已校验 IP 写入当前 Rule 的 manifest、Probe 和 nginx 配置。VM 上的 probe 子命令 不重新解析 FQDN，nginx 连接固定 IP，避免校验对象与实际连接对象不同。发现 FQDN IP 变化时重新校验、探测并发布；新结果不安全时关闭本规则入口并清理配置。",
            "Probe和Doctor能证明什么": "Probe 检查发布前的目标可达性，Doctor 检查代理 VM 的运行状态。候选校验 Validate 先检查当前 Rule 的 conf、ACL、manifest、revision 和允许变更范围。新增规则、upstream 更新、FQDN IP 变化或新增 VM 时执行 Probe：HTTP/TCP 对已校验 IP 做 TCP 建连，DNS 发送查询并接受 NXDOMAIN 等合法响应，首版没有通用 UDP Probe。同一 FQDN 的多个 IP 逐个记录，至少一个可达才通过。Doctor 检查共享 Nginx/unatgw 基础状态，以及当前规则的 revision、listener 和 ACL，默认每 60 秒巡检，不重复 Probe。TLS、应用协议事务和实际 NAT 出口路径由真实客户端请求结合链路检查确认。",
            "怎么避免公网代理配置错误": "配置提交时先检查协议、端口、目标地址和参数组合。下发到代理机后，先校验候选物料，再把它和其他规则组合起来执行 nginx -t，通过才加载，提前发现语法和组合配置问题。加载后继续检查实际版本、监听端口和 ACL，失败就结合代理机实际生效的版本执行恢复。实际公网服务是否能正常使用，还要通过对应协议的客户端请求验证。",
            "代理VM健康为什么不能简化成一个布尔值": "同一台 VM 可以让一条规则正常、另一条规则失败。SSH 主机身份、VM 上的配置工具及运行环境 是接入预检；Doctor 的共享项检查 nginx、基础 ACL 和 unatgw，规则项分别检查当前 Rule 的 revision、listener 和 ACL。共享状态明确不可信时整机隔离，只有单 Rule 失败时只摘该规则后端；Probe 另外验证发布前的 upstream 可达性，真实客户端请求才验证端到端效果。",
            "为什么一条规则只允许一个upstream": "这样白名单、解析、Probe、状态和审计都对应一个明确逻辑目标，revision 也可确定。一个 FQDN 可以解析出多个经校验的 IP，但多个业务目标用多条规则分别表达，避免在一条规则里引入模糊的部分健康和选路语义。",
            "为什么采用每条Rule独立revision": "一条规则变化只发布自己的 conf、ACL 和 revision，可以缩小预检、摘流及回滚影响，其他规则独立收敛。revision 仍覆盖该 Rule 实际下发的规范化完整物料，保证重试与恢复有明确版本；VM 上的配置工具在本地短锁内保护共享 nginx，并保留当前和上一成功目录。仅影响 Service/DNSRecord 的变化不必 reload，周期全量盘点负责发现增量发布期间的缺失、漂移和孤儿物料。\n\n追问：多个规则共用 VM，最需要防止什么？\n防止端口冲突、配置互相覆盖和误删公共访问控制。每条规则需要独立身份与配置范围，共享配置切换和清理要保证互不破坏。",
            "跨VM能原子吗": "不能。一条 Rule 在多个 VM 上逐台更新，可以短暂存在新旧 revision；VM 本地锁和失败恢复只保护单 VM 上当前 Rule 的变更。Controller 预检、检查该规则剩余容量、摘流、Apply、Doctor 后再加回；中途失败时停止该 Rule 后续滚动，已成功 VM 保留新版，失败 VM 经可信回滚和检查后可用旧版服务，之后继续收敛或完成回滚。",
            "HTTP为什么不支持CONNECT_HTTPS怎么处理": "当前 HTTP 模式提供固定目标的反向代理，目标由出口规则预先确定，没有提供客户端通过 CONNECT 临时指定目标的隧道入口。访问 HTTPS 上游时，客户端向代理发送 HTTP 请求，Nginx 再与已校验的目标 IP 建立 TLS，并用目标域名设置 Host、SNI 和证书校验。需要客户端与目标直接建立端到端 TLS 时，可以使用显式授权的固定目标 TCP 透传，由客户端完成握手和证书校验。",
            "Available和Configured有什么区别": "Available=True 表示本规则至少有一台通过 Doctor 的 VM 在受管后端中，LoadBalancer 还要求 VIP 就绪；滚动时可以继续由旧 revision 服务。Configured=True 要求 Product 节点列表非空、全部目标 VM 对本规则达到 desiredRevision 且 Doctor 通过，相关 Service、后端和所需 DNSRecord 也已收敛。所以可以 Available=True、Configured=False，单台失败不一定导致业务完全不可用。\n\n追问：Rule 的 Available、Configured 都是 True，就算网络通了吗？\n还要验证真实请求。`Available` 表示有可用代理后端，`Configured` 表示代理配置与受管资源已收敛；客户端所在节点的转发规则、DNS 缓存或路由仍可能影响访问。我会把状态、实际入口、双 VM 版本和业务请求一起核对。",
            "当前支持IPv6或双栈吗": "当前生产 V1 支持 IPv4 单栈，客户端入口、代理 VM、NATGW 和公网 upstream 都按 IPv4 配置。upstream 地址校验会拒绝 IPv6；支持双栈需要各段网络、入口和转发配置一起适配。",
            "只剩最后一台怎么办": "更新已有规则前，检查移除当前 VM 后本规则是否仍至少有一台合格后端；如果会归零，就暂停这条规则的滚动，保留现有流量。新增规则没有旧流量，可以从第一台通过检查的 VM 开始开放，LB 入口另需等待 VIP。容量保护只约束计划内摘流，明确不安全的后端仍必须隔离。",
            "Apply超时会不会盲目回滚": "不会。先按 Rule UID 查询 current-revision 和 Doctor，核对该规则的真实 conf、manifest、ACL 与 listener；目标版本已生效且健康就继续，旧版本可信且健康就恢复该规则后端。需要回滚时，必须有更新前已核验的上一成功 revision 目录，包含完整 conf、ACL plan 和 manifest，不能只凭旧 revision 字符串恢复。远端结果或回滚物料不可信时保持该规则摘流，后续再确认。",
            "为什么恢复时CurrentRevision和Doctor两个都要看": "current-revision 用来定位当前 Rule 的版本，Doctor 再检查该版本的 listener、ACL 及共享运行状态。相同 revision 也可能发生实际物料漂移，因此还要校验本地 conf/manifest 内容、实际 nftables 规则和 current 指向；目标版本健康就按成功处理，旧版本可信且健康就恢复原后端。上一成功版本的完整目录才是回滚依据，无法确认时保持该规则摘流。\n\n追问：故障 VM 恢复后，什么时候重新加入？\n确认身份、目标配置版本、监听和访问控制恢复，再按健康判据重新加入。复测要检查后端集合确实恢复，并验证新连接能正常访问。",
            "摘流等待能证明连接已经清零吗": "不能。当前默认等五秒，给入口变更传播留出缓冲，没有查询连接数并等待它降到零。Nginx reload 后，已有长连接仍可能由旧 worker 继续处理，当前没有设置统一的强制结束期限。如果后续需要限定旧进程的退出时间，就要接受部分连接被关闭，并配合客户端重连。",
            "更新配置时已有TCP长连接怎么处理": "更新前先确认还有其他健康后端，再摘掉当前规则在这台代理上的后端，等待入口变更传播，让新连接转向其他健康代理。随后通过 Nginx reload 加载配置，旧 worker 继续处理已经建立的连接，处理完再退出。已有连接保留在原来的代理上，不会迁到另一台机器。这是正常配置 reload 的处理方式；直接重启代理进程或 VM 会中断其上的连接。\n\n追问：摘除一个后端后，已有连接会自动迁移到另一台 VM 吗？\n不会自动迁移。后端摘除主要影响后续新连接的选择，已有连接还绑定原来的连接跟踪或代理 socket。原后端仍能服务时可以排空连接；原后端已经故障时，应用需要重连或按自身语义重试。",
            "发布门禁和VM本地锁是不是事务": "不是。发布门禁限制同一 Rule 连续修改，避免当前远端操作未结束时叠加新版本；VM 本地锁防止多个 Rule 同时切换文件和 reload。它们分别保护版本推进与共享进程操作，不能撤销其他 VM 已完成的发布，也不能让 Endpoints、DNSRecord 和多台 VM 原子生效；部分失败仍靠逐 Rule 状态核对、回滚和后续调和恢复。",
            "Controller重启如何恢复": "重读 Rule、Service、后端、Product 节点列表和 host key 记录，从全部受管 Service 的 targetPort 重建端口占用，再逐 Rule 查询各 VM 的 current-revision 和 Doctor，核对真实物料后继续收敛。backendStatuses 是最近观测，不能单独决定 VM 是否接流；发布门禁也要结合远端事实判断。周期盘点再用 VM 上的受管 manifest 或 list-rules 与 Kubernetes Rule 集合对账，修复缺失和漂移，按安全删除流程清理孤儿物料。",
            "DNS_HTTP_TCP_UDP统一了什么又有哪些不能统一": "统一的是每条规则的校验、独立 conf/ACL/revision、发布、健康后端和删除流程，各规则使用相同管理方式但有自己的配置与版本。HTTP 需要主动设置 Host、SNI 和证书校验，且不注册目标域名解析；TCP/UDP 按固定目标转发字节流或报文，首版没有通用 UDP Probe；DNS 通过 RegionDNS 白名单分流，同时提供 TCP/UDP 53 relay，每条规则只有一个公网 resolver，不提供多 resolver 自动故障转移。",
            "一次SSH失败就摘VM吗": "SSH 失败首先说明管理通道不可达，业务监听仍可能正常。系统需要区分读不到状态与确认后端故障，对未知状态重试观察；满足持续不可达判据后再撤流，避免一次管理请求失败就影响业务。\n\nSSH 主机身份不符、身份漂移或有效 Doctor 明确失败时，按对应故障范围隔离；暂时读不到状态与可信异常分开处理。",
            "规则故障会拖垮整台VM吗": "当前 Rule 的 revision、listener 或 ACL 不一致，只从这条规则的 Endpoints/RS 摘除该 VM，修复和回滚也只改这条规则的物料。共享 nginx、基础 ACL 或 unatgw 明确不可信时才整机隔离。不同规则虽然独立发布，仍共用 nginx 进程，因此必须由 VM 本地锁保护共享配置切换和 reload，共享进程故障会影响该 VM 上的多条规则。",
            "删除为何要finalizer": "finalizer 让 Controller 在删除规则前完成入口撤销和远端清理。删除时先关闭流量入口，再清理 VM 上的 listener、ACL 和相应域名入口，最后删除受管资源并移除 finalizer。远端清理失败时保留规则重试；只有 VM 清单明确为空时才跳过远端清理。\n\n追问：为什么还要检查自动清理？\n网络组件不仅要能建入口，还要能撤入口。残留 Listener、后端或 ACL 会占用资源，也可能影响后续规则和访问控制。",
            "SSH下发安全吗": "Controller 的 SSH 私钥放在 Secret 中，连接前核验 VM 的固定 host key。VM 端通过专用账号和固定入口程序，只放行配置校验、应用、状态查询等预置子命令，并检查参数、上传文件路径和大小。即使通过了 SSH 认证，也不能直接执行任意 shell 命令。",
            "Product_Controller_TAD怎么分工": "Product 负责在代理 VM 上安装并启动 Nginx、配置管理工具和必要的访问控制。TAD 负责交付配置，把声明的代理 VM 列表提供给 Controller。Controller 读取出口规则，生成配置和版本，再通过 SSH 调用 VM 上的 validate、apply、doctor 等子命令，完成校验、配置应用和运行状态检查。逐台更新的顺序、规则后端的摘除和恢复由 Controller 控制。",
            "你个人到底做了什么": "我主要实现 EgressForwardRule CRD、Controller、Service/Endpoints/EndpointSlice 管理，以及代理 VM 配置生成和滚动发布。Kubernetes Service/kube-proxy、代理 VM、Nginx 和底层 NATGW 都是复用现网已有能力。",
            "现在算上线了吗": "是，公网代理组件已经完成上线交付，简历中的逐 Rule、逐台配置发布机制也已经落地。每次仅摘除当前规则的一台代理后端，配置下发和健康检查通过后恢复流量。",
            "公网代理讲得有点散_能重新解释解决什么吗": "我把原来人工搭建代理、逐台维护配置的工作，做成了提交规则后自动交付入口和持续维护后端的组件。多条规则可以共享代理 VM；配置变更时逐台发布，检查通过后再恢复该后端接流。交付耗时从约 4 小时降到 10 分钟以内，代理 VM 资源占用减少约 70%。",
            "这是入站还是出站_Service后面究竟是Pod还是VM": "这里是业务 Pod 主动访问固定公网目标的出站请求，Service 后端是独立代理 VM。这个 Service 不设 selector，Controller 显式维护合格 VM 的地址和端口；Service 转发面把请求送到 VM 上的 Nginx，再由它经 NATGW 访问目标。",
            "场景是Overlay还是Underlay_为什么VM能出公网而Pod不能": "这里明确的是 Proxy VM 处在 Underlay 网络，并已经配置了到 NATGW 的公网出口；业务 Pod 使用集群内可达的 Service 入口访问它。Pod 能否直接出公网，还取决于 CNI、路由、SNAT 和访问策略，不能仅凭 Overlay 或 Underlay 判断。我们的部署约束是不能直接把修改主机网卡和默认路由的出网组件装到 TCE 工作节点，所以复用独立代理 VM 的出口。",
            "同一条Rule连续修改怎么避免三个版本叠加": "A 尚未进入远端发布时，B、C 如果已经提交成功，Controller 按最新的 C 重算，跳过未下发的中间版本。A 已进入发布且结果未确认，或已有后端应用新版本但整条规则尚未完成时，Webhook 拒绝后续 Spec 修改。被拒绝的 B、C 不会保存在 CR 里排队，用户或调用方要在本轮结束后重新提交。首次发布明确失败、没有结果未知的后端且没有后端应用目标版本时，允许修正配置；其他 Rule 可以独立更新。",
            "revision相同还需要做什么": "先查当前 Rule 的 current-revision 和 Doctor，并核对真实 conf、manifest、ACL 与 listener。版本相同且健康时只修复 Service、后端或 DNSRecord，不重复 reload；版本相同但运行状态异常时，先从本规则摘除异常 VM，再幂等重放同一 revision。独立增量发布仍要周期对照全部 Rule 与 VM 受管物料，发现缺失、漂移和孤儿配置。",
            "代理VM扩缩容怎么避免影响已有流量": "Product 节点列表变化后，各 Rule 分别重新入队。新 VM 先完成安装和身份、runtime、端口池、unatgw 检查，再逐 Rule validate/probe/apply/doctor，通过哪条规则才加入哪条后端；新 VM 失败不让已有 VM 重新摘流或发布。缩容或升级时先从 Product 列表移除 IP，Controller 从所有受管后端摘除，集群侧交付任务确认并等待传播后才卸载或升级。代理 VM 不持有 Kubernetes 凭证；升级 runtime 时还要同步 Controller 期望版本，恢复原先全部规则的后端后再处理下一台。",
            "安全策略收紧和普通更新失败有什么区别": "普通候选校验或 Probe 失败时，还没触碰的旧配置和后端继续服务；滚动中失败则停止当前 Rule 后续更新，恢复可信旧版本后可重新接流。如果 FQDN 新解析结果、收紧后的目的地址白名单或 TLS 安全基线已经证明旧配置不安全，就立即清空该规则后端，置 Available/Configured=False、UnsafeConfiguration，并清理该 Rule 的配置和 ACL，保留 CR 等运维修正。若全局 ConfigMap 本身格式非法，则停止新发布并报 InvalidProxyConfig，不把它误判成所有旧规则均不安全。",
            "NTP拨测和Underlay补DNS有什么区别": "要看程序部署在哪里，以及已有哪段网络能力。已有 unatgw 的 Underlay 业务缺少公网解析，补齐 DNS 后，业务仍走自身出口；另一项需求是拨测程序运行在 Pod 中，需要访问公网一级 NTP 源，但 Pod 没有直接出网链路，因此还需要对指定 NTP 目标转发业务报文，只补 DNS 不能解决。NTP 接入验收由拨测程序通过受管入口完成真实请求，listener 和 Doctor 用于确认代理配置及运行状态。",
            "这个项目最难的地方是什么_能举个具体例子吗": "我遇到过一次部署检查通过，但两台代理 VM 始终不能接流，连测试规则的删除也卡住。VM 上的配置工具报的是 nftables 缺 input chain，我进 VM 看，实际只有一个空表。继续按程序真实调用复现，才发现老版本 nft 把 `-f -` 当成文件名，打印失败却返回 0，规则根本没写进去；修好写入后，20000 又被它显示成服务名，导致回读校验误判。我先用可撤回的兼容层验证根因，再把正式程序改成通过 `/dev/stdin` 写入、统一用数字格式回读；部署冒烟也改成相同的 stdin 路径和真实端口池。隔离环境里，两台 VM 随后 apply 成功、Doctor 恢复健康，DNS/UDP 规则重新出现健康后端，原来卡住的 11 条删除任务也清理完成。这次让我补齐的是部署检查与运行时调用不一致的测试盲区。",
            "共享代理的安全设计是怎么考虑的": "安全设计分为来源准入和目标约束，分别处理绕过 Service 直连 VM listener，以及访问未授权目标的风险。来源侧由平台按实际网络链路维护 RegionDNS 和 Service/LB 的可信来源网段，Controller 将网段、规则协议和独立监听端口组合为 nftables ACL。目标侧每条规则只允许一个逻辑目标，解析并校验后将允许的 IP 和端口固定到 Nginx 配置，HTTP 上游同时配置 Host、SNI 和证书校验。这些检查在实际收包和转发的代理 VM 上生效。来源网段表达网络准入范围；原始客户端身份识别和严格租户隔离还需要相应的身份与网络策略。",
            "以短信网关为例_业务怎么使用这个组件": "交付人员在 sms Namespace 创建一条 HTTP 规则，选择 ClusterIP，入口 servicePort 为 80，httpHost 和 upstream host 都填 yun.tim.qq.com，上游端口为 443。Controller 创建 Service、发布配置并加入合格后端；Available=True 后，从 status.entrypoints 取得实际 Service 地址和端口，填入消息网关的代理入口配置，公网短信目标保持 yun.tim.qq.com:443。请求由 Pod 经 Service 到 Proxy VM，Nginx 再以 HTTPS 连接固定目标。业务无需维护 VM 地址、内部监听端口、ACL 或 NAT 配置，接入后用真实短信接口请求验证。",
            "公网代理下一步准备做什么_为什么要做": "下一步计划补齐流量监控、按规则限流和代理实例组隔离。先增加 Nginx 流量、连接数和异常请求等指标，按代理实例和访问规则观察，帮助判断容量和定位问题，再据此补充按规则限流。YUM 同步软件源可能持续占用大量带宽，大型客户也可能要求资源隔离。因此还计划让不同规则选择不同的代理 VM 实例组，分散流量并减少共享资源争用。",
            "这个项目在跨团队协作上有什么体会": "我的体会是提前对齐使用场景、异常行为和验收标准，能减少后期因理解偏差反复修改。公网代理涉及研发、产品、测试、售前架构师和交付，需要共同确认谁从什么网络位置访问、使用哪种入口，以及故障时如何处理。开发中我发现共享代理存在来源绕行和访问未授权目标的风险，便与产品、交付对齐，再落实到规则校验和 ACL。这样讨论的结果能直接对应实现和验收条件。",
            "滚动更新成功率高吗_耗时多久_SSH失败怎么办": "整体滚动成功率和单次滚动耗时没有单独的统计值。滚动耗时会累加每台 VM 的摘流传播等待、配置应用、健康检查和失败重试时间，逐台执行是为了保留其他后端继续服务。SSH 或 Apply 结果不确定时，先停止该规则后续滚动，并查询实际 revision 和健康状态。目标版本健康就继续，旧版本可信且健康就恢复后端，无法确认时保持该规则在这台 VM 上摘流。",
            "配置交付十分钟能当成滚动更新时间吗": "不能直接这样算。已有约 4 小时降到 10 分钟以内的数据，衡量的是代理、解析和转发规则的配置交付流程。单次滚动更新时间还需要明确 VM 数量、起止点和失败重试是否计入，并单独统计。",
            "有没有测试暴露的Controller代码缺陷_怎么修的": "有一次隔离 E2E，两台 VM 都还不能服务，Controller 本来应该保留 Service、把后端置空，但 apiserver 一直报 `must specify addresses or notReadyAddresses`。我沿着 Endpoints 构造和写入点查，发现程序把只带端口的内存模板直接写到了 API，这种 subset 没有地址，是非法对象。我保留端口模板供后续发布使用，在所有 API 写入入口统一去掉没有地址的 subset。换修复后的镜像复测，Service 和 `subsets: null` 的 Endpoints 能正常创建，错误消失；当时 VM 的 ACL 故障还没修好，所以可用状态仍然是 False。这一轮修的是无健康后端时也必须能表达合法关闭态，不能把流量不通和控制器写资源失败混在一起。",
            "你实际怎么确认流量经过代理_遇到过什么坑": "遇到过 DNS 能返回公网 IP，但查询没有经过代理的问题。联调时从同一个 RegionDNS 容器对比代理 Service 和两台 VM：直连 VM 正常，Service 超时。核对 DNSRecord 和 forward 后，发现子集群被配置成访问 global 集群的 ClusterIP。调整发布集群后，默认 DNS 和代理 Service 的 UDP、TCP 四项查询全部通过。",
            "为什么部署检查通过_实际运行却失败": "那次nft兼容故障里，部署冒烟用真实文件输入，运行时却用stdin；测试又没有覆盖20000被显示成服务名的行为。于是部署检查通过，实际ACL一直没有正确写入或回读。我把冒烟改成程序同样的 /dev/stdin 路径，用真实20000–25000端口池并检查数字输出，补的是测试与运行路径不一致的问题。",
            "有没有对比社区网关_为什么没有直接采用": "比较过 Istio Egress Gateway、Cilium Egress Gateway 和 APISIX，最终选择复用现有 Proxy VM、Nginx 和 NATGW，减少对私有云网络底座的替换。Istio Egress Gateway 覆盖不了我们的 UDP 场景；Cilium 的方案需要配合它的网络数据面，也不直接管理现有独立代理 VM。APISIX 更接近需求，但规则分配监听端口、返回 Service 入口和多 VM 配置交付仍要适配。我们把开发集中在这些平台控制逻辑上，接受了自行维护配置发布、故障恢复和 VM 环境兼容的成本。",
            "多个VM共享代理_怎么区分来源身份与出口规则": "当前没有逐台 VM 身份与出口规则的绑定，来源准入按代理实际看到的源 IP/CIDR 判断。listener 端口选择出口规则和固定目标，多个 VM 访问同一 listener 时会使用同一目标。允许的来源网段由平台全局配置；如果 Service/LB 转发改变了源地址，代理看到的可能是转发节点地址，ACL 就需要按这条实际链路配置。\n\n追问：白名单已经放行 Pod 网段，为什么代理仍可能拒绝请求？\n代理实际看到的源地址可能经过了 SNAT。TLS 复测中，客户端本地地址是 Pod IP，但两台 VM 抓到的来源是客户端所在节点的 IP。因此 ACL 要按实际入站路径核对来源，同时确认这个来源确实属于允许访问的业务。",
            "TCS实际用哪种Service数据面_怎么确认": "我接触的 TCS 环境里，kube-proxy 使用 iptables 模式。它在各节点监听 Service 和后端地址变化，生成 ClusterIP、端口到后端的转发规则。请求命中规则后，由内核选择后端并执行 DNAT；conntrack 保存连接映射，让同一连接的后续报文沿用这个后端。\n\n以公网代理的 ClusterIP 入口为例，公网代理 Controller 维护两台 VM 的后端地址和监听端口，kube-proxy 将这些信息同步为节点规则，流量到达 VM 后再由 Nginx 访问公网。\n\n现场同时运行 Cilium 1.12.4。\n\n追问：既然用了 Cilium，为什么还保留 kube-proxy？\nCilium 是否接管 Service，要看启用的功能。当时 Cilium 为 Partial 接管，Socket LB 未启用，kube-proxy 仍在运行；现场核对的宿主机 Service 请求走的是 iptables 路径。\n\n追问：你怎么确认确实走了 iptables？\n我核对了 kube-proxy 模式、Cilium 运行状态和目标 Service 的 NAT 规则。现场宿主机请求对应的路径是 `OUTPUT → KUBE-SERVICES → KUBE-SVC-* → KUBE-SEP-* → DNAT 到代理 VM`，后端地址和端口与 VM 监听一致。",
            "DNS返回正确IP能否证明查询经过代理": "解析结果只能证明拿到了答案。缓存或其他 DNS 路径也可能返回同一个 IP，要确认代理生效，还需要看到查询确实到达代理监听端口。",
            "代理VM抓不到包怎么确认抓包有效": "先确认抓包有效。我会在同一窗口发起直连代理的对照请求，检查抓包接口、协议、端口及丢包计数，再判断目标查询是否到达。",
            "公网访问失败如何分段排查": "先固定出问题的客户端和目标，把链路拆成三段：客户端到入口、入口到代理 VM、代理 VM 到公网。分别测试域名、Service 或 VIP、VM 监听端口和公网上游，找出从哪一段开始失败，再查这一段的路由、转发规则、ACL 和进程状态。Service 与后端直连对照是关键。",
            "网络排查为什么固定客户端和时间窗口": "换客户端可能同时换了节点、路由和来源白名单。固定来源、协议、目标和时间窗口，才容易判断差异来自哪一跳。",
            "公网代理排障常用哪些工具": "`dig` 看解析，`curl` 看连接阶段和响应，`ss` 看监听与连接，`ip route get` 看路由，`tcpdump` 看报文；再结合 Kubernetes 对象、组件日志和对应的 iptables、nftables 或 BPF 状态。",
            "Service不通但VM直连正常怎么定位": "先查 Service 的协议和端口、后端地址及端口，再查请求所在节点的转发规则、路由和实际源地址。这个对照把范围缩到了入口路径。",
            "KUBE_SERVICES_KUBE_SVC_KUBE_SEP分别做什么": "`KUBE-SERVICES` 根据 Service IP、端口和协议找到对应入口，`KUBE-SVC-*` 按规则选择后端，`KUBE-SEP-*` 执行到该后端的 DNAT。同一连接后续报文通过 conntrack 沿用映射。",
            "DNS完成reload但查询仍超时怎么定位": "DNS 配置生效后，到转发目标的网络路径仍可能没就绪。现场五个 RegionDNS 都在失败前完成 reload，但访问代理 Service 仍超时。两个节点从收到 Service 更新到日志记录添加端口，分别滞后约 63 秒和 94 秒，同窗口还有数十秒的 iptables 操作，线索指向节点 Service 同步慢。",
            "业务Pod直连Service成功但默认DNS失败为什么": "业务 Pod 和 RegionDNS 在不同节点，Service 规则在各节点分别同步。业务所在节点已经可用，DNS 所在节点仍可能没有完成同步。",
            "DNS慢同步定位到了哪一层_是否确认规则生效时刻": "查到了 Service 处理滞后和 iptables 慢操作；具体内核等待对象及持有者还没确定。63 秒、94 秒是日志阶段间隔，不能当成精确的规则生效时间。",
            "iptables操作慢但CPU未满怎么查": "先看进程在计算还是在等待。现场连续采样发现，kube-proxy 和 cilium-router 的 iptables 子进程多次处于 D 状态，等待点落在 nftables、netlink 路径，同窗口也有慢 save、restore 日志。我据此把范围缩到规则更新的内核等待，再查调用栈和等待关系。",
            "iptables进程D状态是否意味着死锁": "D 状态表示不可中断等待。判断死锁还要找到等待对象、持有者及无法推进的依赖关系，单看进程状态不够。",
            "iptables慢能否认定为xtables文件锁冲突": "需要持锁证据。用户态文件锁和 nftables 内核路径中的等待是两件事；这次采样确认了后者，未确认具体持锁方。",
            "LB已分配VIP但访问被拒绝怎么查": "VIP 分配和后端发布是不同阶段。这套 LB 根据 Director Lease 检查成员活跃状态，再核对 Listener 版本。六条过期 Lease 仍在成员集合里，后端发布被成员检查阻塞。确认旧 Director 已退役并完成清理后，只保留活跃成员，双后端发布和 VIP 访问恢复。",
            "平台LB的Director_Listener_Lease分别是什么": "在这个平台里，Director 执行 LB 转发，Listener 描述监听端口及后端，Director Lease 记录成员活跃状态。发布流程还要核对 Director 是否确认了对应的 Listener 版本。",
            "LB发布失败如何区分公共LB与公网代理职责": "核对交接两侧：公网代理声明的静态后端是否正确，平台 Listener 和 Director 实际转发表是否收敛。这次阻塞点在公共 LB 的成员确认环节。",
            "过期DirectorLease能否直接删除": "先确认对应节点已经退出转发。Lease 过期只能说明没有续租，旧进程或内核程序仍可能存在。这次核对了部署成员、Director 和 BGP 进程、连接、VIP 表、XDP 挂载及流量计数，保留恢复点并卸载旧 Director XDP 入口后，才备份删除六条旧 Lease。",
            "Director进程退出后为何还要检查XDP": "挂载到网卡的 BPF 程序可以继续被内核引用，加载它的进程退出后仍可能存在。要查网卡实际挂载的程序和关联 map。",
            "旧XDP残留是否就是VIP拒绝的根因": "这次明确的阻塞点是过期 Lease 检查。XDP 核验解决的是旧节点能否安全退役；程序仍然挂载，并不能证明它导致了连接拒绝。",
            "connection_refused和timeout的排查区别": "`refused` 通常说明收到主动拒绝，例如 TCP RST；`timeout` 表示规定时间内没有完成期待的阶段。我先抓包确认拒绝来自哪一跳，或者最后成功的报文是哪一个，再查监听、拒绝规则、路由和回程。",
            "SYN重传没有SYN_ACK能否认定目标未监听": "不能。正向丢包、回程不通、过滤和服务端未响应都可能表现为这样。要在客户端、代理入口和目标侧对齐抓包，确认 SYN 到了哪里、响应从哪里消失。",
            "抓到RST是否一定说明业务请求失败": "还要看发生时机。建连阶段的 RST 与成功收到响应后的连接关闭含义不同，必须结合请求结果和报文时序判断。",
            "TCP已连接但TLS超时如何分段排查": "TCP 建连只确认了客户端到代理的连接，TLS 还需要代理向上游转发握手数据。我会把客户端到 Service、客户端到各 VM、各 VM 到公网分开测试，对照 TCP、TLS、首字节耗时和抓包，判断停在代理入口、上游连接还是握手响应。",
            "三次TLS超时与1500次复测分别说明什么": "后续把五条路径各测了 300 次，合计 1500 次全部成功，并同步核对抓包和配置状态。旧的三次超时没有复现，具体原因仍未确定。",
            "curl如何区分TCP连接与TLS握手耗时": "看 `time_connect` 和 `time_appconnect`。单次新建直连请求中，前者是到 TCP 建连完成的累计时间，后者是到 TLS 握手完成的累计时间，两者之差可用于观察握手阶段耗时。",
            "TCP_VIP访问HTTPS怎样保留SNI和证书校验": "URL 保留原公网域名，用 `--connect-to` 把连接定向到 VIP 和代理端口。这样改变的是连接目标，TLS 仍使用原域名进行 SNI 和证书校验，HTTP Host 也保持原域名。\n\n```bash\ncurl --noproxy '*' \\\n  --connect-timeout 3 --max-time 8 \\\n  --connect-to \"dns.alidns.com:443:${VIP:?请设置 VIP}:${PORT:?请设置入口端口}\" \\\n  \"${TARGET_URL:?请设置原域名 dns.alidns.com 端口443的HTTPS请求地址}\"\n```",
            "HTTPS直连IP加Host头是否足够": "只改 Host 不够。Host 属于 HTTP，TLS 握手发生在它之前；URL 改成 IP 后，SNI 和证书校验目标也会受影响。",
            "curl关闭证书校验后成功能否算通过": "只能说明关闭证书校验后请求可达。正常验收要保留校验，并检查证书链、主机名和业务响应。",
            "TLS透传是否只有一条端到端TCP连接": "这个 Nginx stream 代理会建立两条 TCP 连接：客户端到 Nginx、Nginx 到公网上游。两边的握手和序列号各自独立；在这个 TLS 透传场景中，Nginx 转发加密字节，TLS 会话仍在客户端和公网服务之间。",
            "如何关联Nginx客户端连接与独立上游连接": "结合 Nginx 的连接记录、时间窗口、目标和数据交互顺序建立关联。两条 TCP 连接的序列号不同，不能直接按相同序列号配对。",
            "代理链路中如何区分DNAT和SNAT": "DNAT 改目的地址，例如把 Service IP 和端口改成 VM 地址和监听端口；SNAT 改源地址，例如把 Pod IP 改成节点 IP。分别对照转换前后的报文和连接跟踪记录。",
            "如何确认具体哪条ACL拦截请求": "先确认包已到 VM，再检查目标端口、实际源地址、规则顺序和命中计数。只有看到对应拒绝规则命中，才能把失败归到这条 ACL。",
            "RPC读取失败恢复后为何会重复发布和摘后端": "旧实现把状态读取失败的 VM 留在待发布队列，RPC 恢复后对同一 revision 重复 Apply，触发摘后端和 Nginx reload。问题在于把“读不到状态”当成了“需要重新部署”。",
            "RPC恢复重复发布缺陷如何修复和验证": "读状态失败的 VM 不进入本轮发布；下一轮读到相同 revision 且健康时直接收敛。修复后的 RPC 故障恢复专项完成 600 次请求，全部成功，双后端保持且没有重复下发或 reload。",
            "双后端是否能保证所有请求无损": "不能保证所有请求无损。双后端让新连接有其他可选目标，但已有连接仍可能留在被更新或故障的后端，检测和转发表更新也有延迟。长连接是否中断，还取决于原连接能否继续服务及客户端重连能力；切换效果需要用持续请求记录错误和延迟来验证。",
            "同一个Service为什么要从多个节点测试": "节点分别维护本地转发状态，路由和来源地址也可能不同。一个节点成功只能证明这条来源路径可用，不能替代其他实际业务来源。",
            "清理规则为什么要核对UID而非只看名称": "同名对象可以删除后重建，名称相同不代表还是原资源。清理时要核对 UID 和归属，避免旧规则误删新对象或其他规则的资源。",
            "LB网络故障修复后如何验证完整恢复": "用最初失败的入口和客户端复测，同时检查后端与清理结果。这次 LB 恢复后，HTTP、TCP 分别从 Pod 和集群外 VM 各请求 5 次，合计 20 次全部成功，TLS 校验通过；双 VM 正常，删除规则后关联资源也自动清理完成。",
            "网络E2E通过次数能否证明性能提升": "这些结果证明了对应场景的功能和恢复行为。性能提升需要固定负载和对照条件，比较吞吐、连接速率、延迟分位数及资源消耗。",
            "只允许某个Namespace的某个业务访问出口_怎么实现Pod级授权": "我会先把可能访问代理的业务 Pod 纳入出站限制，再按 Namespace 和业务标签给获准 Pod 开放指定规则的目标地址和端口。其他业务保留必需权限，但不开放这个出口，授权标签和策略由平台管理。Service 入口和直连代理 VM 的路径都要验证，避免换一个访问地址就绕过限制。当前组件已有规则和来源 CIDR ACL，按 Pod 标签自动管理这些网络策略还需要接入。",
            "为什么不用NetworkPolicy替代来源CIDR白名单": "NetworkPolicy 可以限制 Pod 到集群外代理 VM 的出站访问，适合补上业务粒度授权。原项目同时承接 Pod、节点和集群外业务 VM，代理 CIDR ACL 用于限制这些来源，节点和外部 VM 的连接还需要各自的网络控制。增加 Pod 级权限时，我会保留代理 ACL，再接入源 Pod 策略，代价是同步维护业务授权与网络规则。",
            "Namespace是否天然对应独占Pod网段": "Namespace 默认没有独占 Pod 网段。要按 Namespace 使用专用 CIDR，需要额外的 IPAM 配置，例如 Calico 为 Namespace 指定地址池，并控制地址池的使用范围。按业务授权时可以用 Namespace 和 Pod 标签选择源工作负载，不依赖每个 Pod 的 IP 固定不变。",
            "到了Nginx已看不到Pod身份_NetworkPolicy在哪里生效": "身份判断发生在源 Pod 的出站路径，CNI 可以把发包端对应到 Pod 和标签，决定是否允许连接；流量通过后才转发到 Nginx。策略建在源 Pod 的 Namespace，用 podSelector 选中业务 Pod，配置 Egress 的目的地址和端口。后续即使做了 SNAT，源端授权仍可执行；代理 ACL 则只能检查转换后的地址。Service 转换前后的匹配位置要按实际 CNI 验证，标准 NetworkPolicy 不通过 Service 名称选择策略对象。",
            "只给业务A添加允许规则_其他Pod就被禁止了吗": "其他 Pod 也必须受到出站限制，否则仍可能默认放行。先覆盖可能访问代理的业务范围，再只为业务 A 开放指定出口，并保留 DNS 等必需访问。多个 NetworkPolicy 的允许范围会合并，还要检查有没有其他策略重新放行这条路径。",
            "业务方修改Pod标签能否绕过授权": "如果业务方能把自己 Pod 的标签改成 app=A，就可能冒用业务 A 的访问权限。授权标签应由平台管理，在工作负载创建和更新时检查，同时限制业务方修改网络策略；只限制已有 Pod 的标签，还要防止通过 Deployment 模板创建同标签的新 Pod。业务方可随意控制的标签不能直接作为可信身份。",
            "多业务共享代理_怎样分别只访问域名A或B": "可以给 A、B 分别配置固定目标规则和独立监听端口，再让两组源 Pod 的出站策略只开放各自获准使用的出口。代理按规则连接固定上游，Pod 访问另一条规则的 Service 或直连其代理端口都应被限制。现有组件已经提供独立规则和端口，按 Pod 标签自动下发授权策略属于需要接入的扩展。",
            "TCP固定目标IP是否等于只允许一个域名": "多个域名共享同一个 IP 时，只固定 TCP 目标 IP 和端口不能实现严格的域名授权。现有 HTTP 规则会固定上游 Host、TLS SNI 和证书校验名称；端到端 TLS 的 TCP 透传还需要额外的应用层控制，才能限制具体域名。源 Pod 的网络策略负责能否到达这个出口，代理的目标约束负责出口连接哪个服务。",
            "Spec更新被拒绝后_B_C会自动排队执行吗": "不会，Webhook 拒绝后，B、C 没有写入这个 CR，Controller 也没有保存它们等待执行。本轮结束后，需要用户或调用方重新提交最新配置。若产品希望自动保留编辑并重试，需要由页面或上层请求服务保存待提交内容。",
            "为什么不一直接受Spec修改_只发布最新版本": "一轮发布可能已经改变部分 VM，这时立即换目标，会同时出现旧版、A 和 C，恢复时还要判断每台机器该保留或回退到哪个版本。我选择限制同一规则在发布期间再次修改，减少中间版本和恢复分支。代价是用户要等本轮结束后再修改，其他规则仍可独立推进。",
            "Service和Nginx已经能转发_你是不是只生成了一份配置": "Service 和 Nginx 的转发能力确实直接复用，我实现的是让一条出口规则能持续、可靠地落到多台代理 VM 上。比如更新规则时，Controller 先摘除这条规则的一台后端，检查 VM 上的实际配置版本，下发配置并完成健康检查后再恢复接流。中途失败或 Controller 重启后，要按 VM 的实际版本继续处理，不能只凭数据库里写了新版本就认为发布成功。这些规则管理、发布和恢复逻辑是我的开发增量。",
            "为什么不用脚本或Ansible批量下发配置": "脚本或 Ansible 可以完成一次配置下发，当前需求还包括规则删除、VM 增减和控制器重启后的持续处理。比如某台 VM 更新失败，系统要记住这条规则在哪些后端可用，并在恢复后继续核对配置和健康状态。我们用 CR 保存期望规则，由 Controller 反复检查实际状态并推进操作。代价是需要维护控制器和状态逻辑；只有少量机器、低频人工变更时，脚本会更简单。",
            "为什么逐台更新_一起更新不是更快吗": "一起更新会让这条规则的所有代理后端同时处于变更中，配置错误或重启就可能中断整个入口。逐台更新保留其他健康后端，当前这一台检查通过后再继续下一台。代价是发布耗时随后端数量增加，而且旧连接仍需单独处理，逐台更新本身不保证长连接无损。"
          },
          "实践案例": {
            "旧nftables让部署冒烟与真实运行不一致": {
              "实践类型": "2026-07-24隔离E2E排障；临时验证后正式代码修复。",
              "口述回答": "我遇到过一次部署检查通过，但两台代理 VM 始终不能接流，连测试规则的删除也卡住。VM 上的配置工具报的是 nftables 缺 input chain，我进 VM 看，实际只有一个空表。继续按程序真实调用复现，才发现老版本 nft 把 `-f -` 当成文件名，打印失败却返回 0，规则根本没写进去；修好写入后，20000 又被它显示成服务名，导致回读校验误判。我先用可撤回的兼容层验证根因，再把正式程序改成通过 `/dev/stdin` 写入、统一用数字格式回读；部署冒烟也改成相同的 stdin 路径和真实端口池。隔离环境里，两台 VM 随后 apply 成功、Doctor 恢复健康，DNS/UDP 规则重新出现健康后端，原来卡住的 11 条删除任务也清理完成。这次让我补齐的是部署检查与运行时调用不一致的测试盲区。",
              "结果范围": "两VM的apply、Doctor与后端恢复，11条Terminating规则清理；未测整个故障期间的请求连续性。"
            },
            "没有健康后端也要写出合法Endpoints": {
              "实践类型": "2026-07-24隔离Controller E2E发现与复测的代码问题。",
              "口述回答": "有一次隔离 E2E，两台 VM 都还不能服务，Controller 本来应该保留 Service、把后端置空，但 apiserver 一直报 `must specify addresses or notReadyAddresses`。我沿着 Endpoints 构造和写入点查，发现程序把只带端口的内存模板直接写到了 API，这种 subset 没有地址，是非法对象。我保留端口模板供后续发布使用，在所有 API 写入入口统一去掉没有地址的 subset。换修复后的镜像复测，Service 和 `subsets: null` 的 Endpoints 能正常创建，错误消失；当时 VM 的 ACL 故障还没修好，所以可用状态仍然是 False。这一轮修的是无健康后端时也必须能表达合法关闭态，不能把流量不通和控制器写资源失败混在一起。",
              "结果范围": "空后端API写入合法、错误消失；当时VM ACL另有故障，未因此宣称业务流量恢复。"
            },
            "DNS解析成功但流量绕过代理": {
              "实践类型": "2026-08-06真实请求、对照域名和VM抓包排障；后续更换TOS3.3 VM组验证同集群路径。",
              "口述回答": "有次 DNS 联测，直接访问代理 Service 能得到公网解析结果，但走集群 DNS 时结果不一致；其中一个 DNS 入口返回了正确 IP，乍看像成功。我先用没有出口规则的域名作对照，发现它们也能解析，说明这个 DNS 自己就能递归出网。然后在两台代理 VM 的监听端口抓包，六次查询一个包都没收到，而同窗口直连代理的查询能抓到。继续核对 DNSRecord 的消费集群和实际 Corefile，发现规则发给了另一个集群；同时查到来源白名单没覆盖实际 DNS VM 地址。后续把验收链路切到入口 Service 所在的 global 集群，核对 Corefile 确实 forward 到动态 Service，再从业务 Pod 分别测默认 DNS、Service 和两台 VM 的 UDP/TCP，才完成这条链路的复测。我现在会同时核对路径和响应，返回正确内容不等于经过了我们管理的出口。",
              "结果范围": "前一环境保留DNS阻塞结果，后一TOS3.3环境完成默认DNS/Service/VM三层UDP与TCP请求；不能替代当前共享Controller版本验收。"
            }
          },
          "公开方案对比": {
            "证据类型": "当前官方能力核对与项目约束分析；历史是否做过PoC待本人补充，不从资料缺失推断当时做过或没有做过。",
            "Istio_Egress_Gateway": {
              "已确认能力": "ServiceEntry配合Gateway/VirtualService集中管理外部HTTP/HTTPS/TCP出口；官方说明不代理UDP。",
              "项目适配判断": "单独采用这条路径无法覆盖本项目DNS UDP relay；Product VM成员管理、TCS DNSRecord/OriginalHost及离线交付还要集成。"
            },
            "APISIX": {
              "已确认能力": "HTTP 与 stream 可同时启用，stream 支持 TCP/UDP 和外部 IP:port 上游。Stream Route 可以通过 Admin API 创建，监听地址和端口需在 stream_proxy 配置中准备；另外提供主动/被动健康检查，具体检查类型需按协议核对。",
              "项目适配判断": "可作为替代数据面。仍需将 Namespace Rule 转换为路由与监听配置，对接 TCS Service、DNSRecord、OriginalHost 及 VM 交付；需验证端口池与新增监听端口的配置生效方式、全部后端不健康时的转发语义，以及发布失败后的恢复。"
            },
            "现有实现的实际成本": "复用Nginx、VM和NATGW，但Controller 和 VM 配置工具共同维护规则发布、ACL、回读、恢复与平台接口；旧nft兼容与空Endpoints问题就是已经付出的维护成本。",
            "若重新选型的最小验证": "同一出口规则分别验证HTTP、TCP、UDP，动态加改删端口、上游DNS变化、配置部分失败和全后端不健康；比较TCS适配代码与升级/回滚步骤。该PoC为建议，尚未执行。",
            "Cilium_Egress_Gateway": {
              "已确认能力": "通过 Pod 选择器和目标 CIDR 选择出口流量，经指定网关节点执行 SNAT；官方稳定版文档要求接入 Cilium，并启用 BPF masquerading 和 kube-proxy replacement，网关接口与出口 IP 由环境提供。",
              "项目适配判断": "侧重集群网络数据面的出口路由与 SNAT。用于本项目需要核对现有 CNI、Service 转发方式、网关节点和出口资源，不能直接等同于对已有集群外 Proxy VM 的配置管理，也不能仅凭文档断言目标环境不兼容。"
            }
          }
        },
        {
          "bullet_title": "排查 KubeVM 磁盘 I/O 异常并补充 QEMU Native AIO、Nydus 运行时适配",
          "项目背景": {
            "异常现象": "现场反馈 KubeVM 数据盘 vdb 的 XFS 检测到一致性异常并进入 shutdown，Guest 文件系统停止正常 I/O。导师给出的调查方向是核对缺省 io 与 io=native 的差异；历史 XFS 损坏的最终根因尚未确定。",
            "排查思路": "先沿 Guest 文件系统与块层 → virtio → QEMU disk driver → LocalPV disk.img 或块设备 → Host 文件系统与物理盘分层核对，再对比 KubeVM 与 CVM 的实际 Domain XML。受测 CVM 在 cache=none 下显式使用 io=native，而 KubeVM 的部分 file-backed 稀疏盘没有命中既有 Native AIO 自动选择。",
            "镜像加速背景": "大镜像冷启动需要等待镜像下载和展开，弹性扩容时大量节点集中回源还会放大 Registry 压力。Nydus 用 RAFS 元数据与数据 chunk 分离实现按需加载，Dragonfly 可在其数据获取链路上提供 piece 级 P2P 复用；两者分别解决启动所需数据量和集中回源问题。",
            "Nydus适配问题": "RuntimeClass 表达 Pod 的运行时选择，但目标 containerd 1.7 在 PullImage 阶段拿不到标准的独立 runtime handler，只能从 PodSandboxConfig 的 io.containerd.cri.runtime-handler annotation 推导 runtime-specific snapshotter，因此需要在准入阶段把两个信号对齐。",
            "改造目标": "保留用户显式 I/O 配置，将 cache、I/O backend、iothread 和 blocksize 分别处理；扩展 KubeVirt 既有 I/O 后端选择，使满足 O_DIRECT 前提的更多 file/device 磁盘能够自动选择 Native AIO，并提供集群级回滚。同时补充 Nydus 的准入桥接，让已经选择 nydus RuntimeClass 的 Pod 自动携带 containerd 1.7 PullImage 选择 runtime-specific snapshotter 所需的 annotation。"
          },
          "一句话": "这项工作包括两部分：针对 KubeVM 数据盘 I/O 异常，我分层核对存储路径和实际 Domain XML，扩展 KubeVirt 既有 SetOptimalIOMode 逻辑，对符合条件的磁盘自动生成 io=native，并补齐参数链路和回滚开关；同时在 tcs-admission-webhook 中补充 Nydus RuntimeClass 适配，为已选择 nydus 的 Pod 注入 containerd runtime-handler annotation。",
          "简历推荐写法": "排查 KubeVM 磁盘 I/O 异常，扩展 KubeVirt 的 QEMU I/O 后端自动选择：对最终 cache=none、源为 file/device 且未显式指定 I/O 模式的磁盘自动配置 io=native，保留显式配置优先和集群级回滚开关，补充参数链路、单元测试及多磁盘/不同扇区几何兼容性验证；补充 Nydus 镜像加速的 RuntimeClass 适配，基于 Admission Webhook 为已指定 runtimeClassName=nydus 的 Pod 幂等注入 containerd runtime-handler annotation。",
          "参考回答": "KubeVM 这部分，我从数据盘 I/O error 和 XFS 主动关闭现象出发，对比实际 Domain XML，扩展了 Native AIO 自动选择和回滚链路。Nydus 这部分，我补充了准入适配：已选择 nydus 的 Pod 会自动获得 containerd PullImage 所需的 runtime-handler annotation，缺失或错误值会被纠正，正确值保持不变。",
          "项目状态": {
            "代码归属": "远程分支 dev/kubevm-qemu-io-native 上有两个提交：b410ea206 为设计文档，c0abe90fa 为实现；Author 和 Committer 均为 leandrozhao。",
            "分支状态": "截至 2026-09-04，功能分支相对 origin/tcs-develop 为 0 behind、2 ahead；tcs-develop、已核对的 release 分支和 tag 均不包含 c0abe90fa，也未发现 DisableAutoNativeAIO 或 --auto-native-aio。",
            "交付记录": "KubeVirt 已完成排查、功能分支改造和测试用例补充，保存的记录停留在功能分支与候选镜像实验。Nydus Webhook 已进入 tcs-extensions master 和 release/tcs2.3.5.1；现有材料记录到源码合入，具体集群部署和节点数据面运行记录未随材料保存。"
          },
          "个人实现与复用边界": {
            "既有能力": "KubeVirt 原有 SetOptimalIOMode 已支持用户显式 io 配置，并会在 cache=none 下为块设备或预分配文件选择 io=native；改造复用这些既有能力及 Domain XML 转换、VMI 生命周期。",
            "本人增量": "KubeVirt 侧，把 autoNativeAIO 布尔决策接入 Converter，默认开启时把普通/稀疏 file-backed 磁盘也纳入选择；增加 DisableAutoNativeAIO 集群级负向 Gate；从 virt-controller 生成的 launcher 参数一路传到 virt-launcher、Domain Manager 和 Converter；补充配置、参数透传、Manager 和 Converter 边界测试，并调整已有 VMI E2E 的期望。Nydus 侧，在 tcs-admission-webhook 中增加受开关控制的 Pod Mutating Handler，对 runtimeClassName=nydus 的 Create/Update 请求幂等注入或纠正 runtime-handler annotation，并补充七类边界测试。",
            "未改内容": "KubeVirt 部分没有同时修改 blocksize 探测和注入逻辑，没有把 iothread 合入本轮变量，也没有强制设置 io=threads；用户已经显式配置 driver.io 时始终保持原值。Nydus Webhook 不创建 RuntimeClass，不给普通 Pod 自动选择 nydus，也不部署 snapshotter、nydusd 或节点 containerd 配置。"
          },
          "implementation_details": {
            "自动选择条件": "Converter 首先检查 driver.io；非空立即返回。随后只接受 Source.File 或 Source.Dev，其他 source 跳过。只有最终 Driver.Cache 为 cache=none 时才考虑 Native AIO，因为 io=native 需要 O_DIRECT 条件。",
            "默认开启行为": "autoNativeAIO=true 时，符合上述条件的 file/device 磁盘都可以设置 io=native，因此覆盖 LocalPV 常见的稀疏 disk.img。",
            "回滚行为": "FeatureGate API 以名称是否存在表达开关，而新能力要求默认开启，因此采用 DisableAutoNativeAIO 负向 Gate。关闭后恢复旧逻辑：块设备或预分配文件仍可使用 Native AIO，普通稀疏文件不再自动设置。",
            "参数传播": "virt-controller 读取 ClusterConfig.AutoNativeAIOEnabled，生成 --auto-native-aio 参数；virt-launcher 解析后传入 NewLibvirtDomainManagerWithAutoNativeAIO，Manager 再将该值带入 VMI 到 Domain XML 的转换上下文。",
            "生效时机": "该值在 VMI 转换并生成 Domain XML 时生效；只修改控制面配置不会原地改写已经运行的 QEMU 磁盘参数，需要新建或重建 VMI，并检查实际 virt-launcher Domain XML 确认。",
            "Nydus准入触发": "源码支持 Pod CREATE 和 UPDATE；正式 .tad 模板也注册二者，但 legacy charts 模板只注册 CREATE。EnablePatchNydusRuntimeHandler 开启且 spec.runtimeClassName 明确等于 nydus 时才进入注入逻辑，runtimeClassName 为空或其他值均跳过。",
            "Nydus幂等与纠正": "annotations 为空时先初始化；目标 annotation 已经为 nydus 时不重复修改，存在但值错误时纠正为 nydus。Admission Handler 对修改前后的 Pod 生成 JSON Patch 返回给 API Server。",
            "Nydus运行时边界": "Webhook 只把 Pod 的 RuntimeClass 选择映射为 containerd 1.7 PullImage 所需 annotation；RuntimeClass 对象、containerd handler/snapshotter、nydusd、镜像转换和可选 Dragonfly P2P 链路必须由其他组件提供。准入阶段补齐控制面意图；容器实际使用的 snapshotter、挂载与按需读取由节点运行态确认。",
            "Nydus版本与同名契约": "实现只匹配 runtimeClassName=nydus 并写入值 nydus，不查询 RuntimeClass.handler，因此部署要求 RuntimeClass 名、handler、containerd runtime 键同名。该桥接针对 containerd 1.7；总体设计稿中的 1.6.9 基线需要内部 backport 才能具备对应能力，核对项为 fork 源码、节点配置和实际 PullImage 行为。",
            "Nydus共享Webhook语义": "Nydus Handler 位于共享 Pod mutating 链末尾；前序 Handler 报错时它不会执行。部署 failurePolicy=Fail，Webhook 不可达、TLS/超时或通用框架失败会阻断所有未被排除的 Pod CREATE/UPDATE，而不仅是 Nydus Pod；带 infra.tce.io/tcs-admission-webhook-exclude 标签的 Pod 则跳过整个 Webhook。"
          },
          "验证与结果": {
            "代码测试覆盖": "新增或调整的测试代码覆盖默认开启、DisableAutoNativeAIO 回滚、Controller 参数透传、Manager 默认与显式关闭，以及 Converter 的稀疏文件、旧行为回滚、显式 io=threads 保留、非 cache=none 跳过和块设备边界。",
            "实验与最终提交关系": "2026-08-13 至 08-20 的候选镜像和实验早于 08-21 的最终实现提交，验证的是 cache=none 稀疏文件选择 Native AIO 这一核心语义，但没有覆盖最终 DisableAutoNativeAIO Gate 和完整参数链。",
            "兼容性观察": "在受测 3.10.11 LocalPV raw 路径中，Guest 512/512 几何下 512B 和 4KiB Direct I/O 均未出现 fio 错误；4Kn Guest 对 512B Direct I/O 返回 EINVAL、4KiB 正常，符合对齐约束。",
            "性能观察": "2026-08-20同盘原配置→候选→恢复原配置的三阶段对照中，512/512链路双盘512B随机写中位数为40,784→26,782→42,549 IOPS，候选较原始下降34.33%；单盘4KiB随机写下降27.02%，恢复原配置后回升。4Kn受测4KiB场景方向为提升，但双盘写原基线存在漂移。结果限单VM、固定顺序及候选产品态，不概括为普遍性能优化。",
            "XFS验证方法": "2026-08-13 在受控 file-backed 512e 路径完成四轮 QEMU SIGKILL 配对实验，缺省 io 与 native 每轮交换盘位。两组均为 4/4 恢复通过，检查已确认写入、文件哈希、挂载与 xfs_repair -n，没有复现原始 XFS 损坏。普通 fio 性能结果与这组异常退出实验分别记录。",
            "TOS4小请求兼容性": "2026-08-14，TOS4/TL4 6.6.98、4Kn loop 与 XFS 4096B 扇区组合，无 Guest。受测 QEMU 2.6 的 raw/qcow2 在 cache=none 下，native 和 threads 的 512B 请求均失败 3/3，4KiB 通过；受测 QEMU 9 构建同条件通过。AIO 完成结果为 -22/EINVAL，报错后 QEMU monitor 仍可响应。"
          },
          "高压追问参考回答": {
            "为什么不是直接修XFS": "XFS shutdown 是检测到一致性异常后的保护动作，先要确认异常来自哪一层存储。排查时我核对实际 XML 和磁盘映射，再分别验证 I/O 后端选择、读写性能与异常退出后的文件系统恢复。四轮 QEMU 异常退出配对测试里，缺省 io 和 native 都恢复成功，原始 XFS 损坏的根因还没有确定。",
            "为什么要自动选择io_native": "LocalPV 常见后端是 file-backed disk.img，原逻辑只为块设备或预分配文件自动选择 Native AIO，稀疏文件即使最终 cache=none 也不会命中。新逻辑在 O_DIRECT 前提满足、用户没有显式选择时补齐这个默认，使生成的 Domain XML 与预期后端策略一致。",
            "为什么保护显式配置": "用户显式指定 io 表示已经做出 workload 或兼容性选择，因此显式值优先，自动策略只补默认值。函数第一步检查 driver.io，io=native、io=threads 或其他非空值都原样保留。",
            "为什么使用负向FeatureGate": "能力要求默认开启，而当前 FeatureGate API 只表达某个名称是否出现在列表中。使用 DisableAutoNativeAIO 才能在默认没有配置时启用新行为，同时允许管理员显式加入 Gate 快速恢复旧逻辑。",
            "io_native和cache_none_iothread有什么关系": "cache=none 决定 QEMU 通常以 O_DIRECT 绕过宿主页缓存，是 Native AIO 自动选择的必要条件；io=native 决定 QEMU 使用哪种宿主 I/O 提交后端；iothread 决定 I/O 处理线程绑定。三者属于不同维度，本轮只改 I/O backend 选择。",
            "怎么确认真正生效": "新建或重建 VMI 后，进入对应 virt-launcher 查看实际 libvirt Domain XML，确认目标磁盘同时使用 cache=none 和 io=native，再核对其他磁盘属性。FeatureGate 和 virt-controller 参数只能说明配置意图，最终生成的 XML 才能确认这次选择是否生效。",
            "性能是否变好": "没有得到普遍提升，在一个同盘小块写对照中反而明显下降。2026-08-20 的 512/512 链路双盘 512B 随机写，中位数依次是原配置 40,784、候选配置 26,782、恢复原配置 42,549 IOPS，每种配置重复三次。受测 4Kn 的 4KiB 场景方向不同，说明结果与存储路径和 I/O 大小有关。这组结果支持按实际后端验证并保留回滚，不能直接把 Native AIO 当作通用提速开关。",
            "现在是否已经上线": "KubeVirt Native AIO 改造仍在功能分支。Nydus Webhook 已合入 tcs-extensions master 和 release/tcs2.3.5.1，具体集群部署状态还需确认。",
            "Nydus为什么需要这个annotation": "RuntimeClass 查到的 handler 标准上通过 RunPodSandboxRequest 交给 CRI，但目标 containerd 1.7 的 PullImageRequest 没有独立 runtime 信息。containerd 会从 PodSandboxConfig 读取 io.containerd.cri.runtime-handler，再用对应 runtime 配置覆盖默认 snapshotter。Webhook 在准入时把 RuntimeClass 意图映射为这个 PullImage 信号，避免镜像准备和 Sandbox 运行时走两套选择。",
            "Webhook是不是自动注入RuntimeClass": "它只处理已经设置 runtimeClassName=nydus 的 Pod，再补齐 runtime-handler Annotation。普通 Pod 没有选择 nydus 时，处理器直接跳过。",
            "Nydus这部分你做到了哪一层": "我做的是 tcs-admission-webhook 的准入适配，把 runtimeClassName=nydus 转成 containerd 1.7 PullImage 使用的 runtime-handler annotation，并接通配置开关、补充七类分支测试。实际按需读取仍由节点上的 snapshotter 和 nydusd 完成。",
            "Nydus怎么确认真正生效": "先确认最终 Pod 的 RuntimeClass 和 annotation，再核对 RuntimeClass.handler、节点 containerd runtimes.nydus 与 snapshotter socket；随后从实际容器反查 Snapshotter=nydus，并确认 fuse.nydusd 或对应真实挂载及 nydusd 读取。随后在受控冷热缓存条件下比较启动时间、首读延迟和回源字节。",
            "Nydus_1_6和1_7是否都适用": "这段桥接针对 containerd 1.7 的 runtime-specific snapshotter 选择。stock 1.6.9 缺少这项能力；内部 1.6.9 fork 若做了 backport，需要核对源码、节点配置和实际 PullImage 行为。总体设计稿同时写了 1.6.9 基线和 1.7 annotation 示例，使用时要先对齐实际版本。",
            "Nydus_Webhook故障影响": "正式 .tad 模板使用 failurePolicy=Fail，共享 Webhook 不可达或超时会阻断所有未被排除的 Pod CREATE/UPDATE，影响范围超过 Nydus Pod；已运行的 Pod 不会因此自动停止。Nydus Handler 只修改内存对象，共享链上的其他处理器报错也可能拒绝请求。为控制影响，我会配置多副本、健康检查和调用超时，并通过匹配规则排除 Webhook 自身恢复所需的资源。",
            "关闭NativeAIO开关会把所有磁盘切回threads吗": "不会，DisableAutoNativeAIO 恢复的是旧自动选择逻辑：块设备或预分配文件在满足条件时仍可使用 Native AIO，普通稀疏文件不再自动设置。用户显式指定的 io 值继续保留。开关在生成 Domain XML 时生效，已运行 VMI 需要新建或重建后再检查 XML。",
            "讲一次实际排查_怎么避免一上来认定根因": "现场是数据盘 vdb 的 XFS 检测到一致性异常并停止正常 I/O，当时怀疑与未配置 io=native 有关。我对比实际 Domain XML，确认 KubeVM 数据盘已经是 cache=none，CVM 还显式设置了 io=native；再沿 PVC、disk.img、loop 和宿主挂载点确认物理盘与扇区大小。查 SetOptimalIOMode 时发现，旧逻辑只给块设备和预分配文件自动选择 native，LocalPV 稀疏文件的实际分配量小于逻辑容量，因此没有命中。我补充普通及稀疏文件的选择条件和回滚开关，然后保持 blocksize、iothread 不变，在同一组数据盘上比较原配置、候选配置和回滚结果。历史 XFS 故障另做异常退出实验，四轮配对测试两种配置都恢复成功。",
            "排查链路是自己识别的还是有AI辅助": "用了 AI，它参与了环境采集、源码定位、实验脚本编写执行和结果复核。原始故障、io=native 的调查方向和第一轮测试要求来自导师，我按要求推进原配置与候选配置对照，保持 blocksize、iothread 不变。后来导师指出 TOS4 下旧 QEMU 处理 512B I/O 属于另一类问题，我再让 AI 将它从业务 fio 性能实验中拆出来单独验证。",
            "具体怎么使用AI_如何核对它给的结果": "我给 AI 现场日志、两套 Domain XML 和测试环境，让它沿 VMI、PVC/PV、backing file 查询实际磁盘，再用 findmnt、losetup、lsblk 核对。例如原本要测 /dev/sdb 的 512/4096 路径，查下来两块盘经过 loop0、loop2，实际都在 /data1 对应的 /dev/sdc，是 512/512，因此另补目标路径测试。我让 AI 同时核对脚本退出码、原始输出和 strace；有一轮 qemu-io 退出码为 0，输出却有 write failed 和 Pattern verification failed。继续核对 AIO 完成结果为 res=-22，即 EINVAL，这批用例按读写失败记录。",
            "磁盘问题能稳定复现吗_具体复现了什么": "测试中稳定复现的是旧 QEMU 在模拟 4Kn 后端上的 512B Direct I/O 报错。TOS4、6.6 内核下，用 loop 模拟 4096B 扇区并创建 XFS，直接测试 QEMU 读写镜像，没有启动 Guest；受测 QEMU 2.6 的 raw、qcow2 组合各重复三次，512B 均返回 EINVAL，4KiB 通过。原始 XFS 损坏另做四轮 QEMU 异常退出配对测试，缺省 io 和 native 均恢复成功，没有复现那次损坏。",
            "TOS4的512B错误关掉native就能避开吗": "在受测 4Kn loop/XFS 路径上，旧 QEMU 改为 cache=none、aio=threads 后，512B 请求仍然失败，单纯关闭 native 无法避开。4KiB 请求通过，受测 QEMU 9 构建在同样条件下也完成了 512B 写入和读回校验。这个对照把问题范围缩到旧 QEMU 在该 Direct I/O 路径上的小请求处理。",
            "IO报错_QEMU退出和XFS_shutdown是同一个现象吗": "这次记录了三种不同观察：性能实验看 IOPS 和延迟，TOS4 实验复现的是 512B 请求返回 EINVAL，现场则是 XFS shutdown 后文件系统停止正常 I/O。TOS4 报错后 QEMU 仍能响应 monitor 的 info status，这组实验没有复现 QEMU 退出。历史 XFS shutdown 的最终根因还没有确定。",
            "只改一个io配置_为什么还需要修改KubeVirt代码": "原来的自动选择逻辑没有覆盖普通稀疏文件，这类磁盘不会自动得到预期的 Native AIO 配置。直接修改某台 VM 的 Domain XML 只能改变当次结果，后续重新生成配置时还会回到原逻辑。我补的是配置生成阶段的选择分支、控制开关和测试，让符合条件的磁盘按规则生成配置，并保留回退入口。",
            "原始XFS异常没有完整复现_你最后修复了什么": "我修复的是 Native AIO 自动选择没有覆盖普通稀疏文件的问题，补齐了选择分支、开关和配置链路测试。原始现象是 Guest I/O error 后 XFS shutdown；对照实验验证了配置差异和特定存储路径下的 I/O 行为，没有建立这个选择缺口与历史 XFS 故障之间的完整因果关系。"
          },
          "实践案例": {
            "开启NativeAIO后小块写反而下降": {
              "实践类型": "本人项目的真实 VM 对照验证；2026-08-20 记录，候选镜像阶段结果。",
              "现象与判断": "数据盘 XFS 异常引出 I/O 后端排查。读写能否完成、性能变化和历史文件系统损坏分别通过兼容性测试、fio 对照及异常退出实验验证。",
              "实际操作": "保存实际 Domain XML 和 PVC/物理盘映射；同一数据盘先测原 launcher，再切候选并重启确认 io=native，最后恢复原 launcher 重启复测。保持 cache、blocksize、IOThread 和 fio 参数一致，每场景三次，保存原始 JSON/TSV。",
              "观察与结果": "512/512 链路双盘 512B 随机写中位数为 40,784→26,782→42,549 IOPS，候选比原始下降约 34%，回滚后回升；单盘 4KiB 写也下降，4Kn 受测 4KiB 方向不同。当前设计决策为默认开启，性能差异纳入按存储路径的观测与回滚判断。",
              "口述回答": "没有得到普遍提升，在一个同盘小块写对照中反而明显下降。2026-08-20 的 512/512 链路双盘 512B 随机写，中位数依次是原配置 40,784、候选配置 26,782、恢复原配置 42,549 IOPS，每种配置重复三次。受测 4Kn 的 4KiB 场景方向不同，说明结果与存储路径和 I/O 大小有关。这组结果支持按实际后端验证并保留回滚，不能直接把 Native AIO 当作通用提速开关。",
              "适用范围": "单 VM、固定顺序、候选产品态对照；不能归因为仅数据盘一个 XML 字段，也不能说已经修复历史 XFS corruption。最终功能分支提交晚于该实验，沿用原有实验与最终提交关系。"
            },
            "物理盘参数不能代替实际PVC链路": {
              "实践类型": "本人项目排查与实验链路核对。",
              "现象与判断": "最初预期覆盖 /dev/sdb 的512/4096物理盘，但原VM的数据盘实际落到 /data1 对应的 /dev/sdc，不能拿目标节点的某块盘参数给所有PVC贴标签。",
              "实际操作": "从 VMI 的 volume 与 PVC 开始，追 PV、LocalPV backing file、loop、宿主挂载点，再对照 launcher 的 disk.img 和 guest vdb/vdc；另行使用hostDisk固定物理盘时单独记录为受控实验。",
              "观察与结果": "确认原PVC实际链路是 /dev/sdc 512/512→/data1→LocalPV loop→disk.img→guest，而不是预期的 /dev/sdb 512/4096；后续报告分开记录物理与guest几何以及产品/hostDisk路径。",
              "口述回答": "排查中我发现，VM落到目标节点不代表它的数据盘就落在我们想测的那块物理盘。最初想覆盖的是 /dev/sdb 的512e路径，沿着 PVC、PV、backing file和loop追下去，原数据盘实际在 /data1 对应的 /dev/sdc，物理几何是512/512。确认后我把物理盘、loop和guest看到的扇区大小分别记录，后续再按真实链路分组比较，避免把另一块盘的结果当成目标路径的结论。"
            }
          }
        },
        {
          "bullet_title": "LocalPV 容量感知调度优化与容量记账修复",
          "项目背景": "Kubernetes 原生的 VolumeBinding 插件已经负责卷的匹配与绑定协调，并在调度时检查卷的节点约束。平台通过 CSI 动态供应本地卷，在 Kubernetes Scheduler 中已有适配自身存储的 LocalPV 插件，负责具体磁盘的容量检查、节点内选盘和容量预留。原有插件能判断节点是否放得下，但多个节点都满足要求时，缺少按本地存储容量进一步排序的能力，容量分布不合理会影响后续副本扩容。因此，我在已有插件上增加跨节点 Score/NormalizeScore，支持 Spread、Binpack 和 None 策略。另一项问题发生在容量对象删除重建后：PVC 占用明细仍在，磁盘聚合占用却丢失，导致可用容量被高估，我补充了条目重建时的已有占用恢复。",
          "参考回答": "LocalPV 容量感知调度优化和容量记账修复都已经落地。我在已有插件上增加跨节点评分，按多卷需求和节点容量支持分散、装箱放置；同时修复容量对象删除重建后已有 PVC 占用漏记的问题。前者改善节点选择，后者避免因少算占用而把容量不足的节点误判为可调度。",
          "项目状态": "简历中的跨节点 Score/NormalizeScore、多卷需求聚合、Spread/Binpack/None 策略配置，以及容量对象重建后的 PVC 占用补记均已落地。源码解释和测试记录仍按两项改动对应的版本分别使用。",
          "个人实现与复用边界": {
            "本人增量": "跨节点 Score/NormalizeScore、多卷需求按 StorageClass 聚合、Spread/Binpack/None 策略配置；容量条目创建时汇总已记录的 PVC 占用，并完善缺条目时的明细去重更新。",
            "复用部分": "评分改动复用原有逐卷选盘、容量过滤、预留和 VolumeBinding 协作链路；重建补偿复用已有 PVC Informer、UID 占用明细和 AssumeMap 账本。存储侧负责实际创建和回收卷。"
          },
          "复习备注_版本与口径": {
            "使用方式": "以下是准备备注，不逐句口述。两项工作按各自源码版本解释；L05–L11、L49–L51中的评分路径指评分版本，L12、L14–L15显式比较两版，L13只讲评分版本的Reserve复检，L40–L48讲重建补偿。",
            "评分版本": "采用上报可用量、PodInUse 同步保护和按盘 Reserve，具有 Score/NormalizeScore。",
            "补账版本": "多盘路径采用 TotalSize、DiskUsage、AssumeMap，不含 Score/NormalizeScore；两项改动按各自版本解释字段和机制。",
            "补偿范围": "补回已有 pvcUsageMap 明细对应的请求量，匹配键为 NodeName+DiskName；不恢复仅存在于被删除条目中的在途预留。这里采用 TotalSizeMultiDisk 的多盘例子，旧单盘 Available 兼容路径另行解释。",
            "卷绑定交接": "节点注解是 volume.kubernetes.io/selected-node，磁盘注解是 loopdevice.infra.tce.io/disk-selected。Reserve 按配置顺序执行，安全顺序是 VolumeBinding 先复制 PVC，再由 LocalCapacityScheduling 修改副本。",
            "Reserve版本差异": "L13口述使用评分版本：在共享容量账本锁内复检并扣减。补账版本的AddAssume按PVC UID登记预留，不再次检查余额，目标磁盘条目不存在时告警返回；不要把两版行为混入同一条执行链路。"
          },
          "验证记录": {
            "现有验证": "评分版本已有单元与集成测试代码，以及调用实际插件、人工构造容量和请求、模拟最终绑定成功的组件对照记录；扩展补测记录包含三轮执行、请求顺序和权重敏感性及退化反例。",
            "对照结果": "在指定 CPU/内存评分与容量输入下，混合大小卷场景后续接纳副本由 1 个增至 2 个；装箱场景让原先放不下的 60 GiB 大卷得到接纳。这些数字衡量对应请求序列的调度接纳能力。",
            "对照覆盖": "评分对照覆盖同 StorageClass 双 PVC，不覆盖真实 API/CSI、跨 StorageClass、多磁盘、并发和生产业务效果；跨 SC 共盘例子单独作为源码推演。",
            "容量补记验证": "现有 PVC 去重测试覆盖重复事件。31,320 个对照单元属于评分版本的组件测试。"
          },
          "高压追问参考回答": {
            "L01_总空闲容量够为什么副本仍然部署不了": "因为本地卷必须在某个具体节点上放得下，多个副本还可能要求分布在不同节点。例如三个节点分别剩 10、10、40 GiB，总共还有 60 GiB，但三个各需 20 GiB、要求跨节点的副本只能部署一个。不同节点的空闲空间不能拼成一个本地卷。",
            "L02_原来完全不看存储吗_CPU内存调度为什么不够": "原来已经检查本地卷能不能放下，但多个节点都能放下时，没有用本地存储容量进一步排序。CPU、内存需求相同的两个 Pod，可能分别申请 10 GiB 和 40 GiB 本地卷，因此 CPU 分散不一定带来存储分散。我补的是候选节点之间的容量偏好，原来的容量硬约束继续保留。",
            "L03_你具体改了什么_是不是重写了一套调度器": "我在平台已有的 LocalPV 调度插件上做了两项增量：增加按 StorageClass 聚合多卷需求的跨节点评分，以及在容量条目重建时补回已记录的 PVC 占用。评分解决多个节点都可行时优先选谁，补记解决容量余额是否算对。这个插件是平台在 Kubernetes Scheduler 中扩展的存储调度能力，节点内选盘、容量检查和预留已有实现，卷绑定沿用 VolumeBinding 的流程。",
            "L04_有状态和无状态对这个调度有什么区别": "插件不根据 Pod 是 StatefulSet 还是 Deployment 来决定是否评分，而是识别它是否引用受管 StorageClass 的 Pending 本地 PVC。有状态业务常常依赖固定本地数据，所以更容易受节点和卷位置约束；无状态服务如果申请这样的本地 PVC，也可以走同一条链路。没有相关 PVC 的 Pod 不参与这项评分。",
            "L05_讲一下从Pod提交到卷可用的完整链路": "按评分版本讲，PreFilter 收集待分配本地 PVC，Filter 在每个候选节点逐卷试算选盘，并保存节点状态。Score 读取其中按 StorageClass 聚合的请求量，结合该节点的容量汇总打分，框架再综合其他插件选节点。Reserve 阶段建立卷绑定的内存状态、补充磁盘选择并预留本地容量，VolumeBinding 的 PreBind 才提交 API 更新并等待卷创建和绑定；成功 PostBind 解除容量同步保护，失败走 Unreserve。Pod 绑定后由 kubelet 完成卷挂载。",
            "L06_一个Pod申请多个卷怎么避免重复使用容量": "先把待分配本地 PVC 按容量从大到小处理，在每个候选节点内用同一份 diskAssumeMap 累计已经分给前面卷的容量。后一个卷只能使用扣除这些假设占用后的余额，所有卷检查通过后，这个节点才算可行。评分再按 StorageClass 汇总需求；既有选盘采用逐卷贪心，没有穷举所有磁盘组合，可能漏掉需要改变前序分配才能得到的可行组合。",
            "L07_评分公式是什么_现场算一个例子": "对每个 StorageClass，按节点上支持该类存储的容量汇总，计算 r=(可用量−当前 Pod 该类请求量)/总量，再按各类请求字节数加权得到 R。Spread 取 round(100R)，Binpack 取 round(100(1−R))，随后做候选节点间归一化。单 SC 例子：两节点总量都为 100 GiB，可用量为 80、40 GiB，当前请求 20 GiB，R 分别为 0.6、0.2，Spread 原始分为 60、20，Binpack 为 40、80。这里是按 SC 汇总的容量偏好，Score 不直接读取选中磁盘的剩余量。",
            "L08_为什么用剩余比例_会不会选了绝对空闲更少的节点": "剩余比例表达这类存储相对总容量的余量，能够让不同容量节点按相对占用程度比较；按这套规则，确实可能优先选择绝对空闲更少、但剩余比例更高的节点。Filter 先保证当前卷能放下，Score 再表达放置偏好。这是按相对余量选择的启发式策略；未来大卷的接纳情况取决于请求分布，现有对照同时包含收益例子和退化反例。",
            "L09_多个StorageClass怎么配置_策略冲突怎么办": "在本地 PVC 对应的 StorageClass 上配置 storage.infra.tce.io/scheduler-policy，值可为 Spread、Binpack 或 None；未配置或留空时使用插件默认策略，默认未设置时为 Spread。同一 Pod 涉及的待分配本地卷必须解析成同一种策略，否则这个 Pod 的本插件评分统一为 0，容量过滤仍然执行。非法注解或读取失败也会停止该 Pod 的容量评分，不擅自按某一类存储的偏好替用户决定。",
            "L10_节点间Spread和节点内Balanced有什么区别": "Spread/Binpack 决定多个候选节点之间怎么排序；原来的 DiskPolicy 决定某个节点里面选哪块盘。Balanced 优先选可用容量最大的盘，Concentrated 选剩余最少但够用的盘，两层策略可以独立组合。两块盘各剩 30 GiB，不能只凭总和就判定能放一个 60 GiB 卷，是否有单个受管分配单元能容纳它仍由 Filter 检查。",
            "L11_已经是0到100分为什么还要NormalizeScore_权重怎么影响结果": "归一化是在本轮候选节点中按最大最小分拉开差异，例如 60、20 会变成 100、0，全部同分则保持原值。最终节点由各插件分数乘权重后相加决定，存储分最高不保证一定中选；甚至 80、81 的小差异也可能被拉成 0、100，需要一起评估权重。现有补测里 CPU:LocalPV 为 1:1 时两个原始收益例子仍成立，改成 10:1 后消失，对应的请求序列和容量输入保持相同。",
            "L12_容量数据哪里来_你统计的是申请量还是实际磁盘写入量": "PVC 记账取 requests.storage，表示申请的容量，应用实际写了多少数据不在这份账里。评分版本读取 LocalStorageCapacity 的总量和可用量，再维护调度预留。容量修复所在的多盘版本则读取 TotalSizeMultiDisk，用总量减去 PVC 占用汇总和在途预留得到余额；两版容量来源不同，计算时要按对应实现解释。",
            "L13_Filter通过以后容量被别人占了怎么办": "在我增加评分的这个版本里，Reserve 会在共享容量账本的锁内重新检查余额，够用才扣减，不够就返回失败，让调度重新尝试。Filter 只在当轮快照上试算，没有实际占用容量，所以中间其他 Pod 的预留可能改变结果。同一 Scheduler 进程内，对这份账本的检查和扣减放在同一临界区，避免两个请求各自检查通过后重复使用容量。",
            "L14_多个磁盘只预留成功一部分_或者后续绑定失败怎么回滚": "评分快照按盘扣减并记录 reservedMap，Unreserve 只归还已成功扣减的部分并解除容量同步保护；PostBind 只解除保护。七月容量修复版则按 PVC UID 写入预留，Unreserve 幂等移除对应 AssumeMap 条目，PostBind 只输出容量快照。已经由 PVC 事件写入的 DiskUsage 随 PVC 删除或选择注解清除而释放，不由一次 Pod 调度失败直接清空。真实卷的创建和回收继续由卷绑定与存储侧流程处理。",
            "L15_容量上报延迟或Scheduler重启会不会把预留覆盖掉": "评分快照用 PodInUse 标记正在绑定的 Pod，周期 SetCapacity 看到仍有保护时跳过余额覆盖。七月容量修复版的多盘周期同步只更新 TotalSize，保留 DiskUsage 和 AssumeMap，PodInUse 已经弃用。容量条目被删除则会丢失条目里的聚合和预留，这次补偿从仍在内存中的 PVC 明细恢复聚合；进程重启后明细本身也要通过 Informer 对象处理重新建立。",
            "L16_为什么需要WaitForFirstConsumer_卷是谁创建的": "延迟绑定让调度器在选节点时一起考虑 Pod 的资源需求和本地存储位置，避免先把卷固定到一个不满足 Pod 条件的节点。这里的本地存储是平台提供的容量上报和供给链路，本插件给出节点、磁盘选择并协作绑定，实际创建卷由存储侧完成。上游原生 local 卷以预创建 PV 为基础，动态供应来自这里的平台存储链路。",
            "L17_已有PVC或已运行Pod能不能换策略后重新均衡": "这次优化主要作用于尚未分配的本地卷；已绑定卷的位置仍由 PV 的节点约束限制，不能为了拿高分把已有数据调到别处。如果一个 Pod 同时有已绑定和待分配卷，也必须先满足已有卷的位置，再考虑新卷。修改策略影响后续调度，不会主动迁移已运行 Pod 或整理已有磁盘数据。",
            "L18_Spread是不是已经保证副本跨节点_和反亲和有什么区别": "没有，存储 Spread 只是按剩余容量比例给节点打分，不保证同一业务的副本一定分开。副本必须跨节点时，要另外配置硬反亲和或相应的拓扑硬约束；它们先决定哪些节点允许使用，存储评分再在其中排序。收益例子里的后续副本由硬反亲和提供跨节点约束。",
            "L19_副本从1到2怎么来的_现场画一下": "三个节点总容量各 100 GiB、初始可用各 60 GiB，先依次提交 40、10、10、40、10、10 GiB 六个单卷请求，Pod 的 CPU、内存需求相同。这组对照使用 CPU:LocalPV=1:1000；保留 CPU/内存评分但关闭本地存储评分时，前置请求后的剩余容量排序为 10、10、40 GiB；启用 Spread 后为 10、20、30 GiB。随后三个各需 20 GiB、要求跨节点的副本，可接纳数就从 1 个变成 2 个。两边都接纳了全部前置请求，总剩余仍是 60 GiB，第三个副本依然不能部署。",
            "L20_装箱怎么让60GiB大卷放下来_代价是什么": "在同样的 CPU:LocalPV=1:1000 对照配置下，三个节点各可用 60 GiB，先提交六个 20 GiB 卷；均衡放置后剩 20、20、20 GiB，总计还有 60 GiB，但没有一个节点能独立放下后续 60 GiB 大卷。装箱后可以剩 0、0、60 GiB，因此大卷可以接纳。代价是可供跨节点小副本使用的节点变少：如果后续换成三个必须分开的 20 GiB 副本，均衡能放三个，装箱只能放一个。",
            "L21_分散一定更好吗_有没有负收益例子": "不一定，现有补测保留了明确反例：初始每节点可用 60 GiB，前置请求为 40、15、15、40、15、15 GiB。基线最终剩 5、5、30 GiB，还能放一个 30 GiB 卷；Spread 剩 5、15、20 GiB，反而一个也放不下。评分只根据当前请求和余量做选择，没有预知下一批卷的大小，所以容量更均衡不代表未来接纳数一定更高。",
            "L22_收益是不是挑了有利顺序_基线和验证怎么做": "对照使用同一套实现和相同输入，基线只关闭本地存储评分，同时保留容量过滤、CPU/内存评分和后续副本的硬反亲和。后续又改变了请求顺序、卷大小、初始容量和评分权重，并补了同 StorageClass 双 PVC、预留回滚和成功后释放保护的检查。节点与请求是人工构造，组件调用是真实代码，最终 API/CSI 绑定由测试模拟成功，记录的是组件接纳和状态变化；真实 API/CSI、存储创建与挂载不在这组模拟范围内。",
            "L23_怎么量化整体价值_能说提升利用率或扩容成功率吗": "现有对照能证明特定请求序列的接纳能力改善，线上整体利用率和扩容成功率没有对应统计。模拟绑定对照中，混合卷序列后续可接纳的跨节点副本由 1 个增至 2 个，装箱场景接纳了原先放不下的 60 GiB 卷。这些结果的前提是相同初始容量和请求顺序；换一组卷大小或顺序，结果可能不同。",
            "L24_上线如何选策略和回退_装箱会不会造成IO热点": "策略要根据卷大小、扩容方式和节点故障域选择，并用业务请求分布验证；需要保留多个可部署节点时可以评估 Spread，需要保留大卷空间时可以评估 Binpack。装箱可能增加单节点磁盘或网络争用，这次评分使用容量指标，IOPS、带宽和延迟需要另外评估。配置 None 或移除本插件的 Score 注册可以停止容量优选，同时保留容量检查和预留；它不会撤销已经发生的卷放置。",
            "L25_容量数据缺失是忽略打分还是拒绝调度": "需要区分阶段：Filter 找不到能满足请求的受管容量，会把该节点判为不可行；若 Filter 已通过但 Score 缺少总容量等评分数据，当前实现对该节点返回 0 分。这个 0 分会参与后续比较，不等于从候选列表删除，也不保证是中性处理。策略冲突导致整个 Pod 的本插件分数都为 0，则只是该插件不能再区分这些节点。",
            "L26_这项能力能用于训练吗_有没有和训练平台联动": "申请受管本地 PVC 的训练 Pod 可以使用这项能力，在部署阶段选择同时满足卷容量和其他硬约束的节点。这次交付到单 Pod 的本地卷调度，训练 Job 生命周期、Gang、checkpoint 协调和任务重试属于后续平台联动。任务整体启动和失败恢复由相应的训练调度与恢复机制处理。",
            "L27_GPU够但本地盘不够_或者GPU拓扑与存储偏好冲突怎么办": "同一个 Pod 必须在同一个候选节点上同时满足 GPU、CPU、内存、卷位置和容量等硬约束；不能把 A 节点的 GPU 和 B 节点的本地盘拼成一次可行分配。若本地盘最合适的节点不满足 GPU 硬约束，它会先被过滤掉；若冲突的只是评分偏好，就由插件权重参与取舍。交集为空时 Pod 继续等待，需要增加兼容资源或调整约束，提高存储分数不能解决。",
            "L28_分布式训练多个Worker一起启动_这个插件能保证吗": "不能，它按单个 Pod 评估本地卷，不掌握整个训练任务的 Worker 数和启动条件。成组调度需要上层 Job/PodGroup 与支持 Gang 的调度机制一起判断各 Worker 的 GPU、存储和拓扑是否满足，并协调预留、等待及失败退出。单 Pod 的容量 Filter 和 Score 为这层协调提供每个 Worker 的可行性判断。",
            "L29_对推理有什么作用_能提高模型加载速度或缓存命中吗": "推理实例用受管本地 PVC 存放模型或缓存时，评分改善新卷的容量放置，减少容量分布造成的部署受阻。当前输入是卷请求和节点容量，模型版本、缓存位置与命中情况不参与评分。若扩展为缓存感知放置，需要平台暴露缓存位置和版本，再比较数据准备时间、模型加载时间与首个请求耗时。",
            "L30_本地模型缓存和训练Checkpoint能按同一种方式处理吗": "不能只因为都落在本地盘上就按同一种恢复方式处理：有远端可靠源的模型或数据缓存通常可以重新拉取，代价是冷启动；checkpoint 若是唯一有效进度副本，节点故障后可能无法读取，甚至丢失。容量过滤和绑定链路约束卷的放置，评分只在可行节点间表达偏好，都不负责数据复制。要支持跨节点恢复，需要训练系统保存完整一致、且目标节点可访问的 checkpoint，并实现对应恢复流程。",
            "L31_和GPU节点自愈一起用_Drain后能把训练Pod迁到别处吗": "不能默认可以，Pod 使用已绑定本地卷时，原卷的节点约束仍然存在；驱逐 Pod 不会把卷里的数据搬走。节点只是重启且盘和数据完好时，任务可以在节点恢复后按上层机制重新拉起；如果要换节点，就要有可访问的数据副本、缓存重建或数据恢复方案。训练状态迁移需要训练平台另外协调，PDB 负责的是驱逐期间的可用性约束。",
            "L32_训练推理用COS_共享文件系统_emptyDir也能用这套策略吗": "当前实现只处理受管 StorageClass 下的 Pending 本地 PVC，不直接识别应用通过 SDK 访问 COS 的需求，也不把 emptyDir 或 hostPath 当作这条 PVC 路径。共享文件系统和云盘的可达性、拓扑及供给方式不同，需要各自的容量和绑定机制，不能把本地节点容量公式原样套过去。判断是否适用先看卷的实际类型和申领路径，不看业务是不是训练、推理。",
            "L33_有状态业务扩容是不是指把原来的PVC变大": "这里主要指增加业务副本，每个新副本再申请本地卷，因此需要足够多符合容量和部署约束的节点。把一个已有 PVC 从 20 GiB 扩成 40 GiB 属于卷扩容，要看存储后端和原节点能否完成扩展。这次节点评分不会搬走旧卷，也不能保证已有卷原地扩容成功。",
            "L34_高优先级训练任务来了_抢占能整理本地存储碎片吗": "仅抢占或删除低优先级 Pod，不能假定它的 PVC 和实际存储会立即释放；持久卷可能仍然保留数据和绑定关系。就算某些容量被回收，也必须在目标节点形成足够的可用分配空间，并满足其他硬约束。持久卷的回收、数据搬迁与碎片整理需要存储和任务管理流程单独协调。",
            "L35_这不就是一个简单打分吗_难点和算法局限在哪里": "打分公式本身不复杂，难点是评分必须沿用 Filter 对多卷选盘的试算结果，否则评分看中的容量和最终实际占用可能对不上。我复用这份结果，再按 StorageClass 策略和放置后的剩余比例比较节点。算法只知道当前请求和当前余额，不知道后面会来什么卷，因此 Spread 和 Binpack 都不能保证后续所有请求最优。对照要同时看受益序列和退化反例，判断应该在哪类负载使用。",
            "L36_节点和卷很多时开销如何_下一步还会做什么": "评分路径会逐卷查找可用盘，按 StorageClass 汇总时还存在容量集合扫描；补账版本在新建磁盘条目时扫描已有 PVC 明细，成本随明细数量增加。频繁重建还会延长容量写锁的持有时间，因此要同时看调度耗时和锁等待。优化时可以按节点、StorageClass 或节点磁盘键建立索引，减少重复扫描，并验证事件更新、补偿和删除时索引是否一致。",
            "L37_这个插件运行在哪里_代码怎么生效和回退": "这两项改动均已落地。插件运行在 kube-scheduler 进程内，评分能力随 Scheduler 构建，并通过 Score 扩展点配置启用和设置权重。把策略设为 None 或移除 Score 注册，可以停止容量优选，保留原有容量过滤与预留。这个开关只控制评分，容量账本修复随对应代码版本生效；切换评分策略也不会迁移已经绑定的卷。",
            "L38_说用索引太泛了_容量连续变化具体用什么数据结构": "如果增加容量索引，我会按 StorageClass 分别维护平衡有序树，用“可用容量、节点名”作为键，再用哈希表记录每个节点当前的键。假设节点 A 的可用量从 40 GiB 变成 25 GiB，就删除旧键并插入新键，和容量账本在同一临界区更新。查找至少 20 GiB 的候选节点时，从 lower_bound(20 GiB) 开始枚举；定位是 O(log N)，返回 k 个候选还需要 O(k)。这个索引只缩小容量候选范围，多卷能否分别放到盘上仍由 Filter 试算，Reserve 再按最新余额确认预留。",
            "L39_除了冲突回滚_如何让训练大任务与普通任务提前避让": "可以让大训练任务先公布关键候选节点及预留需求，普通小任务优先使用其他可行节点，减少同时争抢；最终仍通过共同的容量账本确认预留。小任务持续到达时，单靠打分保证不了大任务启动，还需要队列层的等待优先级、受保护预留或准入限制。预留设置超时释放，避免长期阻塞其他业务。",
            "L40_容量记录为什么会重建_具体删了什么_磁盘和PVC还在吗": "调度器按存储侧上报的 LocalStorageCapacity 对象维护节点和磁盘的容量记录。容量对象被删除时，会移除对应的 StorageClass 关联；某个节点和磁盘条目不再支持任何 StorageClass，才清理这条内存记录。之后同一节点、同一块磁盘的容量重新上报，就会重新创建条目；普通容量更新只修改已有记录。PVC、PV 和磁盘数据可以一直保留，按 PVC UID 保存的申请量明细也还在。旧逻辑却把新条目的聚合占用初始化为零，因此我在创建条目时按节点和磁盘汇总已有 PVC 明细，补回占用后再用于容量判断。",
            "L41_容量漏记会怎样影响调度_用数字讲一下故障时序": "举一个多盘容量条目的手算例子：节点 A 的 d1 总量 100 GiB，已有 PVC 请求 60 GiB，没有在途预留，正确余额是 40 GiB。LSC 删除使磁盘条目消失，但按 PVC UID 保存的 60 GiB 明细还在。旧逻辑重建条目时把 DiskUsage 初始化为 0，余额就被误算成 100 GiB。此时新请求 50 GiB，本地容量 Filter 可能错误放行 A/d1，而正确账本应当拒绝。",
            "L42_PVC明细还在_为什么等Informer再发一次事件不能自动恢复": "因为已有逻辑按 PVC UID 去重，只把新请求量与明细中旧请求量的差值加到 DiskUsage。同一个 60 GiB PVC 再来一次，差值是 60 减 60，仍然为零，补不回磁盘条目重建时丢失的 60 GiB 聚合占用。根因是明细还在、聚合丢了，而增量更新依赖两者原本一致。",
            "L43_具体在哪补回占用_为什么同时覆盖PVC先到和LSC先到": "补偿放在容量 map 的 Add 创建分支：目标节点和磁盘条目不存在时，先创建新条目，再汇总 pvcUsageMap 中属于这个 key 的请求量，填好 DiskUsage 后放入 map。PVC 先到时先保存明细，LSC 到来建条目时补回；LSC 先到时，后续 PVC 事件按原有逻辑增加占用。这样，同一入口同时处理容量条目删除重建和首次建账时的事件先后差异。",
            "L44_重复Add或PVC重放会不会多扣_不同磁盘怎么避免混账": "只有条目不存在时才从零汇总，已有条目再次 Add 只更新 TotalSize，保留 DiskUsage 和 AssumeMap。PVC 明细以 UID 去重，同 UID 同请求量不增加占用，请求量改变时按最新值更新或增加差值。补偿按 NodeName 加 DiskName 完整匹配，因此 A/d1、A/d2 和 B/d1 分别记账。PVC 在补偿前被删除时，删除处理先移除对应 UID 的明细，重建时就不会再计入。",
            "L45_补偿扫描时又来了PVC事件_会不会读到半成品或重复计算": "Add 补偿与 PVC 明细更新使用同一把容量 map 写锁，两类操作依次完成。Add 先在锁内汇总完整的 DiskUsage，再把新条目放入 map；Filter 持同一把锁的读锁，一起读取总量、占用和预留。因此本进程内不会读到“新条目已出现，但占用还没补完”的中间状态。",
            "L46_PVC什么时候计入DiskUsage_和Reserve预留怎样交接": "这个容量版本按 PVC 的选择信息记账：受管 StorageClass 的 PVC 同时带有 selected-node 和 disk-selected，并且有 storage 请求量，就会被跟踪，Pending PVC 也可能计入。Reserve 先按 PVC UID 写 AssumeMap；后续 PVC 事件在同一把写锁内移除该 UID 的预留，再更新明细与 DiskUsage，完成占用交接。若调度失败，Unreserve 只移除在途预留，已经由 PVC 事件确认的占用继续随 PVC 删除或选择注解清除而释放。",
            "L47_容量条目重建和Scheduler重启有什么区别_预留能全部恢复吗": "容量条目重建时进程还在，pvcUsageMap 里的明细可以用于补回 DiskUsage。旧条目中的 AssumeMap 随删除丢失，新条目从空预留开始；仅有在途预留、尚未进入 PVC 明细的请求不会被这次补偿找回。Scheduler 进程重启则连明细也丢了，要通过 Informer 的初始对象处理重新建账，和单个容量条目重建是两条恢复路径。",
            "L48_这个修复怎样验证_目前有哪些实际验证记录": "核心对照是：磁盘总量 100 GiB、已有 PVC 占用 60 GiB，删除重建容量条目后再申请 50 GiB，可用量应为 40 GiB，Filter 应拒绝。此外还验证重复重建不会重复扣减、PVC 先到仍能补回占用，以及条目缺失期间 PVC 变化后按最新明细记账。",
            "L49_选中节点和磁盘后写到哪个对象_Reserve和PreBind怎样衔接": "Filter 把 PVC UID 到磁盘的映射保存在 CycleState，VolumeBinding 也按候选节点保存需要动态供应卷的 PVC。Reserve 需要先由 VolumeBinding 复制 PVC、写 selected-node 并建立绑定缓存，再由本插件在这些副本上写 disk-selected、预留本地容量。到 VolumeBinding 的 PreBind，才把 PVC 更新提交给 API Server，并等待存储侧创建和绑定。节点和磁盘选择沿同一份绑定状态传递，实际修改的是 PVC 对象。",
            "L50_两个StorageClass共用一块盘_评分是否等于物理盘剩余比例": "不一定，Score 按 SC 分别汇总容量，只减去当前 Pod 对这一类的请求。假设一块 100 GiB 空盘同时支持 A、B，两者都用 Spread，分别申请 40、20 GiB，加权剩余比例是 (0.6×40+0.8×20)/60，约 0.667，原始分为 67；实际共用盘只剩 40 GiB。这个分数表达按 SC 汇总的容量偏好。Filter 按物理盘累计两个卷的 60 GiB 试算占用，逐卷判断能否放下。",
            "L51_同一轮Unreserve调用两次_会不会重复归还容量": "评分版本的 reservedMap 记录哪些盘扣减成功，但归还后没有清除此标记；如果对同一 CycleState 重复调用 Unreserve，存在再次把容量加回去的路径。要让这条回滚幂等，归还时还要清除相应成功标记。补账版本则按 PVC UID 删除 AssumeMap 条目，重复删除同一 UID 不会重复归还字节。",
            "L52_为什么不在删除容量对象时把PVC明细也清掉": "容量对象被删除只表示这份容量描述失效，PVC 的申领和磁盘数据可能仍然存在。一起清掉 PVC 明细会丢失仍有效的申领依据，重建时更难恢复正确余额。保留按 UID 记录的明细，在容量 Add 创建入口按节点和磁盘汇总，既能处理条目重建，也能处理 PVC 先于容量对象到达的顺序。",
            "L53_别只讲评分公式_你实际怎么验证价值和取舍": "我用实际插件做过一组连续请求对照：三台节点各剩60Gi，先放40、10、10、40、10、10Gi的卷。关闭存储评分后剩10、10、40Gi，启用Spread后剩10、20、30Gi，后续各要20Gi且必须跨节点的副本，可接纳数从1个变成2个。这是人工输入、模拟绑定的组件验证。随后我换了请求大小和评分权重，找到了Spread反而放不下后续大卷的反例，所以最终保留分散和装箱的业务选择，没有把容量均衡说成普遍最优。",
            "L54_讲一个实际代码问题_不要只说幂等和最终一致性": "容量条目被删掉再重建时，PVC 明细还在，新条目的已用容量却从零开始。即使同一个 PVC 再来一次事件，去重逻辑算出的增量也是零，丢掉的占用仍补不回来。我把补偿放在条目创建入口，按节点和磁盘汇总已有 PVC 明细，填好占用后再让容量查询使用这个条目。修复后这份账表达的是总容量扣除已申请容量和在途预留后的余额。",
            "L55_Score返回比例还是分数_和NormalizeScore有什么区别": "Score 针对单个候选节点返回整数分，剩余容量比例只是中间结果；Spread 使用 round(100R)，Binpack 使用 round(100(1−R))。NormalizeScore 再拿到本插件对全部候选节点的分数列表，按本轮最大最小值调整。例如原始分为 20、35、50，就映射成 0、50、100，全部同分则保留原值；最后由框架结合其他插件权重计算总分。",
            "L56_CSI已经上报容量_为什么调度器还要记录PVC占用": "在容量修复对应的多盘实现里，上报的是磁盘总量，调度器还要扣除已经分给 PVC 的容量，以及正在调度、尚未完成的预留。PVC 申请了 60 GiB，即使应用只写入 10 GiB，也不能把剩下的 50 GiB 当成未分配容量再次承诺给其他卷。容量条目重建时漏掉已有 PVC，占用就会被低估，所以需要从保留的 PVC 明细补回。",
            "L57_这部分是你做的吗_NormalizeScore具体怎么打分": "评分这部分是我实现的。先澄清一点：打分发生在 Score，NormalizeScore 不产生分数，只做归一化。我的实现是标准的 min-max 拉伸，分两步：先遍历一遍把每个分数裁剪到 0 到 MaxNodeScore 之间，同时记录本轮的最小值和最大值；再按 (score - min) * MaxNodeScore / (max - min) 线性映射。这里有一个必须处理的分支：最小值等于最大值时直接返回，不做拉伸。一是分母为零，二是所有节点同分说明这个插件对本次调度没有区分度，硬拉成 0 和 100 会凭空制造偏好。",
            "L58_从PVC申请容量到0到100的分数_完整过程是什么": "分四步。第一步在 PreFilter，遍历 Pod 的待绑定本地 PVC，按 StorageClass 把申请量累加进 CycleState，因为一个 Pod 可能挂多个卷、用不同 StorageClass。第二步在 Score，对每个 StorageClass 取该节点的总量和可用量，可用量要再减掉本轮已经假定分配出去的部分，然后算剩余比例，也就是可用除以总量。第三步把各 StorageClass 的比例按申请量加权平均，得到这个节点对本 Pod 的综合剩余比例。第四步按策略决定方向，Spread 直接用这个比例，Binpack 用 1 减去它，None 直接返回 0，最后乘 MaxNodeScore 取整。举个数：某节点对应 StorageClass 总量 100 GiB、可用 60 GiB，Pod 只申请这一类，Spread 下比例 0.6，原始分就是 60；同样场景走 Binpack 就是 40。",
            "L59_为什么按申请量加权_直接对各StorageClass取平均不行吗": "不行，会被小卷稀释判断。假设一个 Pod 申请 StorageClass A 100 GiB、StorageClass B 1 GiB，节点上 A 很紧张、B 很宽松，简单平均会把两者算成同等重要，选出一个放不下大卷的节点。按申请量加权，权重就落在真正决定成败的那一类上。这里的取舍是：加权反映的是容量压力，不反映业务重要性，如果某类卷在业务上更关键但申请量小，这个公式表达不出来，需要另外引入权重来源。",
            "L60_Score阶段输出的是比例吗_和NormalizeScore的本质区别在哪": "输出的不是比例。比例只是中间量，Score 接口返回 int64，我返回的是对比例四舍五入后乘 MaxNodeScore 的整数分，框架要求这个值落在 MinNodeScore 和 MaxNodeScore 之间，返回浮点比例会被判越界。这里有三个实现细节：常量用 framework.MaxNodeScore 而不是硬编码 100；用 math.Round 而不是直接截断，避免系统性偏低；乘之前先把比例夹在 0 到 1，因为可用量减去在途预留后理论上可能为负。至于和 NormalizeScore 的本质区别，是视野不同。Score 每个节点调用一次、可以并行，只能看到当前节点，算出的是绝对值；NormalizeScore 每个插件调用一次，入参是全部候选节点的分数列表，能看到全局分布。这个区别之所以重要，是因为不同插件的原始分分布可能差很多，我这个插件可能所有节点都落在 70 到 75，另一个插件铺满 0 到 100，直接加权求和的话我这个插件事实上被削弱了，配置的权重就不是真实生效的权重。归一化就是把窄区间拉开，让插件之间的权重可比。实现上 NormalizeScore 不是独立扩展点，要通过 ScoreExtensions 接口挂上去，不需要归一化时这里返回 nil；而且它是原地修改传入的分数切片，返回值只是 Status。",
            "L61_除了Score和NormalizeScore_还接触过调度框架哪些扩展点": "按调度周期和绑定周期分开说。调度周期是同步串行的，决定放哪：QueueSort 决定队列顺序，全局只能有一个插件生效；PreFilter 做前置计算并写 CycleState，也可以直接判定不可调度提前结束，我在这里汇总 PVC 申请量；Filter 逐节点过滤硬条件，我在这里按盘试算容量并选盘；PostFilter 只在 Filter 全部失败后触发，抢占就实现在这里；PreScore、Score、NormalizeScore 负责打分；Reserve 和 Unreserve 成对出现，在调度器缓存里占住资源，后续任何阶段失败都会回调 Unreserve 回滚；Permit 可以让 Pod 进入等待或被拒绝，gang 调度就是靠它，凑不齐 minMember 就 wait，凑齐了统一放行。绑定周期是异步的，不阻塞主调度循环：PreBind 做绑定前准备，本地卷的实际绑定就在这一步；Bind 写 nodeName；PostBind 做善后。我自己实现的是 PreFilter、Filter、Score 加 NormalizeScore、Reserve 加 Unreserve 这几个。其中 Reserve 和 Unreserve 这一对最容易出问题，因为调度器是乐观并发的，Assume 之后到真正 Bind 成功之前有一段窗口，这期间必须在缓存里把容量占住，否则并发调度的下一个 Pod 会读到过期余额；同时回滚必须幂等，我们评分版本的 reservedMap 在归还后没有清除成功标记，重复调用 Unreserve 存在把容量重复加回去的路径，补账版本按 PVC UID 删除预留才不会重复归还。另外我对比过 Extender 和 Framework：Extender 是 HTTP 回调，介入点只有 Filter、Prioritize、Bind 那几个，每次都有网络开销，好处是不用改调度器二进制；Framework 插件要编译进调度器，但扩展点细得多，有 CycleState 跨阶段传状态，而且像 Permit 这种能力 Extender 做不到。HAMi 走的是 Extender，Volcano 直接做了自定义调度器。"
          },
          "提问分层与答案索引": {
            "说明": "按面试官读简历条目后的推进顺序整理：验真、算法、正确性、价值、广度，前一层答不稳通常不会往下问。箭头后为本节 L 编号，括号内是这题必须给出的具体材料。",
            "第1层_验真_问题是否真实存在": [
              "总空闲容量够为什么副本部署不了 → L01（三节点 10、10、40 GiB 的例子要能张口就说）",
              "原来的调度器到底知不知道容量 → L02（原来有容量硬检查，缺的是候选节点之间的排序）",
              "你是不是重写了一套调度器 → L03（强调在平台已有插件上做两项增量，不是新写调度器）"
            ],
            "第2层_算法_公式与实现": [
              "从 Pod 提交到卷可用的完整链路 → L05",
              "一个 Pod 多个卷怎么避免重复使用容量 → L06（同一份 diskAssumeMap 累计）",
              "评分公式是什么、现场算一个 → L07、L58",
              "为什么用剩余比例不用绝对空闲量 → L08（承认可能选中绝对空闲更少的节点）",
              "为什么按申请量加权而不是简单平均 → L59（100 GiB 与 1 GiB 被稀释的例子）",
              "Score 返回的是比例还是分数、和 NormalizeScore 什么区别 → L55、L60（本质答视野差异）",
              "节点间 Spread 和节点内 Balanced 有什么区别 → L10（两层策略最容易被问混）"
            ],
            "第3层_正确性_并发与边界": [
              "Filter 通过之后容量被别人占了怎么办 → L13（锁内复检再扣减）",
              "多个磁盘只预留成功一部分怎么回滚 → L14、L51（主动交代 reservedMap 重复归还不幂等的路径）",
              "上报延迟或 Scheduler 重启会不会覆盖预留 → L15（两版行为不同，先说明是哪一版）",
              "容量数据缺失是忽略打分还是拒绝调度 → L25（Filter 判不可行，Score 返回 0 分不等于剔除）",
              "记账漏记的完整故障时序 → L41（总量 100、已有 60、误算成 100 的那组数字）",
              "PVC 明细还在，为什么等 Informer 再发事件补不回来 → L42（去重后差值为零）",
              "补偿放在哪里、怎么防止重复扣或混账 → L43、L44、L45（Add 创建分支、UID 去重、同一把写锁）"
            ],
            "第4层_价值_最容易翻车": [
              "副本从 1 个到 2 个是怎么来的 → L19（给完整请求序列和权重配置）",
              "装箱让 60 GiB 大卷放下的代价是什么 → L20（跨节点小副本可用节点变少）",
              "有没有负收益的例子 → L21（必须主动给反例，不等对方问出来）",
              "是不是挑了有利顺序 → L22（说明基线只关闭存储评分，其余输入相同）",
              "能不能说提升了利用率或扩容成功率 → L23（直接说没有线上统计，只有组件级对照）",
              "这个修复实际验证过什么 → L48"
            ],
            "第5层_广度": [
              "除了 Score 和 NormalizeScore 还接触过哪些扩展点 → L61（按调度周期与绑定周期分段，落到自己实现的五个）",
              "和 GPU、训练、推理怎么配合 → L26 至 L31（逐题说明本插件边界在单 Pod 卷放置）",
              "节点和卷很多时开销如何、下一步做什么 → L36、L38（按 SC 维护有序树加哈希表，索引只缩小候选范围）"
            ],
            "答题前必须先声明的前提": "评分与容量补账分属两个版本，Reserve、Unreserve、周期同步和容量来源的行为都不同。被问到这些点时先说明按哪一版回答，不要把两版混进同一条执行链路，参见本模块口述边界中的 Reserve 版本差异。"
          },
          "实践案例": {
            "分散评分收益和退化都保留": {
              "实践类型": "实际插件组件对照；人工容量/请求，模拟最终绑定，2026-09-07。",
              "现象与判断": "总剩余容量相同，后续大卷或跨节点副本的接纳情况仍可能不同；需要验证评分是否改变放置，也需要找收益不成立的输入。",
              "实际操作": "固定三节点容量和相同CPU/内存请求，只切换关闭评分、Spread、Binpack，调用实际 Filter/Score/NormalizeScore/Reserve 等代码；再改变卷顺序、容量、权重并保留负例，三轮独立进程重复。",
              "观察与结果": "权重CPU:LocalPV=1:1000，前置[40,10,10,40,10,10]Gi后，关闭评分剩[10,10,40]Gi，Spread剩[10,20,30]Gi，后续20Gi跨节点副本从接纳1个变成2个。改成[40,15,15,40,15,15]Gi后，基线还能放一个30Gi卷，Spread一个也放不下；权重10:1时原收益例也消失。",
              "口述回答": "我用实际插件做过一组连续请求对照：三台节点各剩60Gi，先放40、10、10、40、10、10Gi的卷。关闭存储评分后剩10、10、40Gi，启用Spread后剩10、20、30Gi，后续各要20Gi且必须跨节点的副本，可接纳数从1个变成2个。这是人工输入、模拟绑定的组件验证。随后我换了请求大小和评分权重，找到了Spread反而放不下后续大卷的反例，所以最终保留分散和装箱的业务选择，没有把容量均衡说成普遍最优。"
            },
            "容量条目重建后重放PVC也补不回占用": {
              "实践类型": "已落地的容量记账修复。",
              "现象与判断": "LocalStorageCapacity触发内存磁盘条目删除重建后，pvcUsageMap中PVC明细仍在，但新条目的DiskUsage从零开始；相同PVC重放的增量为零，不能自动修回聚合。",
              "实际操作": "在容量map的Add创建分支，持同一把写锁创建条目，按NodeName+DiskName汇总仍有效的PVC请求量，填好DiskUsage再发布；已有条目只更新总量，保持占用与预留。磁盘条目尚未建立时的PVC重放也一并处理：pvcUsageMap中已存在该UID时只在请求量变化时更新sizeBytes，支持扩容且不重复累加。",
              "观察与结果": "100Gi总量、60Gi既有申领、重建后新申请50Gi用于解释前后余额，是手算示例。",
              "口述回答": "容量条目被删掉再重建时，PVC 明细还在，新条目的已用容量却从零开始。即使同一个 PVC 再来一次事件，去重逻辑算出的增量也是零，丢掉的占用仍补不回来。我把补偿放在条目创建入口，按节点和磁盘汇总已有 PVC 明细，填好占用后再让容量查询使用这个条目。修复后这份账表达的是总容量扣除已申请容量和在途预留后的余额。"
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
        "参考回答": "平台需要持续托管用户的 Agent，并记录调用量和积分，支持后续资源治理。我实现了两条链路：将镜像和运行配置部署到 Kubernetes，并用 Watch 加周期 List 同步状态；通过 Kafka 异步记录调用明细和聚合统计，处理重复事件与失败补偿。",
        "项目状态": "两条主链路在仓库中有实现。面试按代码描述已完成能力，并明确并发配额、跨系统故障恢复、删除状态与计量幂等的一致性边界；完整闭环需要结合后续设计和实际验证，不能只依赖注释或旧答稿。",
        "面试身份边界": "本人实现 Agent Server 生命周期编排、Kubernetes 运行状态回写，以及 Kafka 驱动的调用和发布事件计量。Kubernetes 负责 Pod 调度和副本维护，Agent 镜像负责业务执行；当前仓库不能证明本人实现了完整请求网关或底层模型推理引擎。",
        "与MaaS的对应关系": {
          "服务管理": "Agent Server 的镜像、Deployment、Service、副本和运行状态管理，可以对应 MaaS 在线模型服务的生命周期控制。运行对象相似，具体模型加载、推理执行和性能优化需要由模型服务或推理引擎实现。",
          "计量治理": "Kafka 调用事件、MongoDB 明细和 MySQL 汇总，可以复用为 MaaS 用量治理的基础；当前字段以 Agent 调用、用户和积分为主，尚未形成 Token 用量与多租户模型成本账本。",
          "额度对接": "仓库 LiteLLMClient 根据会员等级构造 team_id、max_budget 和 budget_duration，并调用外部 LiteLLM 更新接口；这能证明额度配置对接，不能证明本仓库实现了网关路由、实时预算扣减或 Token 结算。",
          "扩展口径": "maas_interview_extensions 中的 Token 账本、持久幂等、实时预算、优雅缩容和推理扩容回答属于设计演进与通用原理，不能作为已经在 TP 落地或验证的经历。"
        }
      },
      "个人职责": {
        "个人实现范围": [
          "Agent Server 的 Secret、Deployment、Service 编排，以及创建、更新、副本调整和 Pod 日志查询",
          "通过 Watch + 周期 List 同步 Kubernetes Deployment 运行状态，并清理从 Deployment 枚举到的无数据库归属资源及同名 Service",
          "Kafka 事件消费、Redis eventId 去重、MongoDB 明细、MySQL 聚合、ChainedTransactionManager 配置与事务边界、失败重试和 DLQ"
        ],
        "职责边界": "本人实现两条主链路中的业务编排与处理逻辑，底层复用 Kubernetes、Kafka 和数据库。",
        "证据使用边界": [
          "本人贡献以简历和现有业务代码为依据；实现、上线配置、故障注入和效果统计分别表述。",
          "代码缺少请求级幂等、完整创建恢复记录、持久化事件账本、自动回放等机制时，先说明当前分支，再说明如果改进会怎样设计。",
          "通用 Kubernetes 能力不等于当前平台模板已接入；未找到 StartupProbe、生产隔离策略或高级日志参数时，不说已实现。",
          "MaaS 的 Token 用量、费用归集和推理性能回答保留设计语气；Token 预算、GPU 容量和 CPU/内存配额分别解释。"
        ]
      },
      "interview_core": {
        "k8s_mysql_sync": {
          "主题": "Agent Server 执行环境与生命周期管理",
          "参考回答": "平台把用户镜像和运行配置部署成 Kubernetes 服务，并将实际副本和运行态回写产品数据库。创建时检查账号额度和重名，按需准备镜像 Secret，再创建 Deployment、Service，最后登记 MySQL；接口成功不等待 Pod Ready，业务可用需要正确的 Readiness 与启动逻辑。Watch 提供及时更新，周期 List 重新读取 Deployment 当前快照，并清理从 Deployment 枚举到的部分无归属资源。当前失败分支有补偿，但未持久化创建/清理步骤，也有删除通知、创建登记窗口和补偿覆盖范围上的局限；更完整的恢复需要操作状态、资源归属和双向对账。",
          "链路说明": {
            "一句话主线": "用户提交镜像与运行配置 → 后端校验账号额度与重名，编排 Secret、Deployment、Service 并登记 MySQL → Kubernetes 异步启动 Pod → Watch/List 回写运行态；当前补偿和清理覆盖部分失败分支，后续可补可恢复操作与完整对账。",
            "链路概览": [
              "1. 请求受理：读取账号，接收镜像、端口、启动参数、环境变量、CPU/内存、副本和可选 Readiness/Liveness 路径，检查额度和重名，并以 serverName-userId 生成资源名。",
              "2. 资源编排：按需创建镜像拉取 Secret，再创建 Deployment 和 ClusterIP Service，最后登记用户归属、期望配置、DNS 与初始状态到 MySQL；接口不等待 Pod Ready。",
              "3. 异步运行：Deployment Controller、Scheduler 和 kubelet 创建、调度并启动 Pod，已配置的 Readiness 决定是否作为 Service 就绪后端；未配置探针不能据默认就绪认定模型可用。",
              "4. 状态同步：Watch 与周期 List 回写实际副本、可用副本及摘要状态；周期仅枚举现存 Deployment，并清理对应无数据库归属的 Service/Deployment，删除通知和缺失资源对账仍有缺口。"
            ],
            "完整链路_8阶段": [
              "1. 接收镜像及运行参数，读取账号，并检查账号 CPU/内存额度和同账号重名。",
              "2. 以 serverName-userId 生成稳定 Kubernetes 名称；这统一查询与更新目标，但不等于整个创建请求已幂等。",
              "3. 提供镜像仓用户名和密码时创建 imagePull Secret，否则模板引用预设拉取 Secret。",
              "4. 创建 Deployment，填入镜像、参数、环境变量、CPU/内存 requests/limits、副本和可选 Readiness/Liveness；创建 ClusterIP Service。",
              "5. 在 MySQL 登记归属、期望配置、DNS 和初始运行态；该事务不能回滚已完成的 Kubernetes 操作。",
              "6. Kubernetes 异步调度、拉取镜像并启动 Pod；Readiness 与模型加载/预热逻辑决定业务是否可以接流量。",
              "7. raw Watch 持续回写 status.replicas、availableReplicas 与摘要状态，当前不按通知 type 单独处理 DELETED，也未保存 resourceVersion。",
              "8. 周期 List 重新读取现存 Deployment 并校正摘要，针对无数据库归属者删除同名 Service/Deployment；独立 Service/Secret 孤儿、数据库有行但无 Deployment，以及创建登记窗口需进一步完善。"
            ],
            "创建成功后的状态": "接口成功表示同步编排和数据库登记成功返回，不等待 Pod 就绪；可选 Readiness 需要正确反映应用加载和预热。未配置 Readiness 时 Kubernetes 默认就绪行为不等于业务模型可用，Watch/List 的 availableReplicas 摘要也不能代替真实请求检查。",
            "失败闭环": "Deployment 创建异常会尝试删除 Secret；Service 创建异常会尝试删除 Deployment/Secret；MySQL insert 返回非正数会删除 Service/Deployment/Secret。补偿失败仅记录日志，API 超时结果不明、insert 抛异常、事务提交失败和崩溃不一定进入这些分支；周期可再处理仍有 Deployment 的部分孤儿。后续设计需持久化操作与步骤，按实际状态幂等恢复，并避免按名补偿删除其他请求的对象。",
            "同步与清理范围": "同步范围是当前 namespace 的现存 Deployment，回写实际副本、可用副本和摘要；清理范围是从这些 Deployment 查到的无业务归属资源及其同名 Service。它不回滚 spec，不独立清理 Secret 或只有 Service 的孤儿，不纠正数据库有记录而 Deployment 已消失的方向，也没有创建中保护窗口。"
          },
          "各类数据分别以哪里为准": {
            "MySQL": "保存用户归属、额度关系、产品查询所需的期望配置，以及页面展示所需的实际副本、可用副本和运行状态。",
            "Kubernetes": "Deployment、Service、Secret 是实际运行资源；Deployment spec 表示已经下发到集群的运行期望，判断实际副本和运行状态时以 Deployment status 为准。",
            "同步方向": "业务请求负责把用户期望写入 MySQL 并编排到 Kubernetes；Watch + List 主要把 Kubernetes 运行态回写 MySQL，并清理失去数据库归属的受管资源。",
            "一致性语义": "MySQL 事务不能回滚 Kubernetes 副作用；当前有部分逆序补偿和单向运行态同步、孤儿清理，但所有故障窗口尚未闭环。稳定命名也不等于请求级幂等，演进需要持久化操作状态、受管资源身份、可恢复补偿和双向对账。",
            "为什么仍需MySQL": "Kubernetes 不承载完整的账号归属、权限、额度和产品查询模型；MySQL 负责业务主数据与聚合查询，Kubernetes 负责运行资源和实际状态。"
          },
          "关键实现与取舍": {
            "直接KubernetesClient": "业务后端直接编排标准 Kubernetes 对象，能够直接接入账号校验、额度和数据库流程；补偿、状态同步和残留资源清理由业务层实现。",
            "Watch加周期List": "Watch 提供低延迟但可能断线，List 提供当前全量快照。两者组合追求最终状态收敛，不依赖每个事件只处理一次，也不试图还原所有中间事件。",
            "状态语义": "平台 RUNNING 由实际副本 status.replicas 与 availableReplicas 得出，不是基于完整 rollout 或目标副本完成条件。Running、Ready、Available 有不同语义，模型服务需要业务 Readiness 和加载预热状态补足判断。",
            "资源生命周期": "创建依次准备 Secret、Deployment、Service，再登记数据库；部分失败分支有逆序删除，但补偿只记录失败，API 结果不明、事务提交失败和进程崩溃仍有窗口。更新先 replace K8s 再写数据库，周期同步不回滚 spec；Deployment/Service 删除404视为已完成，Secret删除未作同等处理。",
            "Pod日志": "当前实现按 Pod 名全量读取日志，再在内存按行分页；没有暴露 container、follow、previous 和时间过滤参数。replicaLog 未使用 agentServerId 做账号或 Pod 归属校验，需要补充授权检查，不能声称已有三层隔离。"
          },
          "失败与补偿闭环": [
            "当前补偿按具体失败分支执行：Deployment 异常删 Secret，Service 异常删 Deployment/Secret，insert 返回非正数删 Service/Deployment/Secret；不能概括为任意异常都会自动逆序恢复。",
            "数据库异常、事务提交失败或进程崩溃后，周期 List 能再次处理仍有 Deployment 的无归属资源；只有 Service 的孤儿和 Secret 残留无独立扫描闭环，补偿持续失败也未保存任务进度。",
            "Watch 重连未携带旧 resourceVersion，周期 List 补当前快照而非历史事件；当前忽略通知 type，DELETED 可能把最后快照误写成现存状态，需要独立删除处理和数据库反向核对。",
            "更新先 replace Deployment/Service 再写 MySQL，失败不能由数据库事务撤销 K8s spec；周期只同步运行态，同份请求重试是一种恢复手段，但还要核对实际配置与操作结果。",
            "孤儿查询限定 namespace 并按名称解析用户归属；数据库异常不作为不存在处理，但创建事务未提交和同名 UID 重建仍有误删窗口。改进需操作状态、创建保护和 UID 删除前置条件；API 响应超时先读取确认。"
          ],
          "高压追问参考回答": {
            "Agent平台是什么业务": "这是一个 Agent 开发、发布和托管平台，支持 Prompt、Workflow、Framework 和第三方 Agent。开发者可以提交镜像部署 Agent Server，再把它关联到 Agent 的调用地址；平台还维护发布审核、调用次数、用户统计和开发者积分。我的 TP 经历主要集中在运行环境生命周期和异步调用计量，Pod 调度、副本维护由 Kubernetes 完成。仓库可以确认积分、会员和 LiteLLM 调用预算相关能力，不能直接把它说成按积分兑换 GPU 卡或完整 Token 账单系统。",
            "用户提交源码_二进制还是镜像_平台怎么部署": "用户提交已经构建好的镜像，以及端口、CPU、内存、副本数、启动参数和可选探针路径，平台不负责源码构建。后端读取当前账号、检查资源额度和同账号重名，再按需创建镜像 Secret，随后创建 Deployment 和 ClusterIP Service，最后登记 MySQL。Service 的 selector 关联 Pod 标签，平台保存集群内 DNS、端口和主接口路径供关联 Agent 使用。接口成功不等于镜像已经拉取或进程可用；配置了 Readiness 时需通过探针，未配置时 Kubernetes 默认就绪语义也不能证明模型已经加载完成。",
            "用户已经有镜像_为什么不自己部署Kubernetes": "平台把镜像托管、账号归属、Agent 发布、调用计量和资源额度管理连起来。开发者只需提交运行配置，平台统一生成资源、维护入口和运行状态，并处理业务记录与 Kubernetes 资源之间的失败窗口。当前仓库能确认调用统计、积分与会员预算管理，我不会把这些直接说成 GPU 资源兑换功能。是否自行部署取决于用户是否愿意维护这些配套能力，底层运行机制仍然复用 Kubernetes。",
            "AgentServer和普通Kubernetes服务有什么区别": "运行层仍是标准的 Deployment、Service 和 Secret，区别在 Agent 产品控制面。平台增加账号归属、资源额度和 Agent 引用关系，例如正在被 Agent 使用的服务不允许删除。服务端口或主接口路径变化时，还要重新计算访问地址并同步给关联 Agent。用户操作的是这些产品能力，底层继续复用 Kubernetes 的运行机制。",
            "底层都是Deployment和Service_为什么不直接用ArgoCD": "Argo CD 可以负责 Deployment 和 Service 的声明式发布，但账号额度、Agent 引用和删除保护仍要由业务代码处理。当前实现由业务接口直接调用 Kubernetes，创建资源后登记业务记录，再持续同步运行状态，没有再引入 Git 作为发布入口。这让用户操作与业务校验衔接得更直接，代价是资源补偿和状态同步需要自己维护。",
            "请讲一次AgentServer从请求到可用的完整链路": "用户通过创建接口提交镜像和运行参数，后端获取账号，计算每副本 CPU、内存乘副本数的占用，并检查额度和重名。按需创建镜像 Secret，随后依次创建 Deployment、ClusterIP Service，最后把归属、配置和 DNS 写入 MySQL；关联 Framework Agent 时组合 DNS、端口和主路径作为 requestUrl。Kubernetes 再异步完成调度、镜像拉取和容器启动，配置的 Readiness 决定是否可以作为 Service 就绪后端。Watch 和周期 List 将实际副本、可用副本与摘要状态回写页面。明确失败的分支有逆序补偿，但异常、响应超时和进程崩溃不能都视为已经补偿成功；周期清理目前也只覆盖从 Deployment 枚举到的孤儿 Deployment/Service。",
            "这是标准K8sController吗": "这是一套由业务后端直接调用 Kubernetes API 的资源编排服务，再通过 raw Watch 和周期 List 同步并校正状态。它没有使用 CRD、Informer cache、workqueue 和按 key 执行的标准 reconcile。",
            "Kubernetes和MySQL分别以谁的数据为准": "用户归属、额度和产品查询模型以 MySQL 为准；实际运行资源和副本状态以 Kubernetes 为准。两边共同保存的期望字段通过业务操作写入，再由后续核对逐步保持一致；不存在一份能够代表所有数据的唯一记录。",
            "K8s和MySQL能原子吗": "不能，mysqlTransactionManager 只能管理数据库事务，不能撤销已经提交给 API Server 的操作。当前创建链路依靠顺序创建和部分失败分支的逆序删除，周期同步则校正运行态并清理部分孤儿。这里仍有数据库提交前被扫描、API 超时结果不明和补偿失败等窗口，不能说最终一致性已经在所有故障情况下得到保证。更完整的方案是先持久化操作 ID、期望配置和 Creating/Deleting 状态，再由可重试的后台流程核对真实资源并推进步骤。",
            "为什么Watch之后还要List": "Watch 断开期间可能漏掉变化，所以周期 List 用来重新核对当前状态。当前 Java Watch 没有保存断线前的 resourceVersion，重连以后不能靠它补齐全部历史事件。List 重新读取 Deployment 的副本和运行状态，修正数据库里的摘要；历史事件回放不属于这一步。",
            "K8s成功而MySQL失败怎么办": "创建时 MySQL insert 返回非正数会显式删除 Service、Deployment 和 Secret；insert 抛异常、数据库提交失败或进程崩溃可能绕过该补偿分支。周期 List 能再次处理有 Deployment 而无数据库归属的资源，但不会独立扫描 Service、Secret，也缺少创建登记窗口保护。更新时 K8s spec 不会随 MySQL 事务回滚，周期任务仅回写运行态。更完整的方案是先持久化操作状态与期望配置，对结果不明的步骤先读取核对，再依据操作归属和 UID 重试或补偿，并对数据库和各类受管资源做双向对账。",
            "fullSync能保证什么": "当前 fullSync 枚举专用 namespace 中现存的 Deployment，回写副本和摘要状态；没有数据库归属时再删除同名 Service 和 Deployment。它不是双向配置 reconcile，不会恢复旧 spec，也不能发现数据库有记录但 Deployment 已消失、只有 Service 的孤儿或孤儿 Secret。每次同步可以继续处理尚未清理的 Deployment，但代码没有持久化重试次数、清理任务或创建宽限期。扩大保障范围需要同时枚举业务记录和各类受管资源，并依据操作状态、归属 ID 和资源 UID 做差集核对。",
            "为什么还要MySQL": "Kubernetes 擅长管理运行资源，但不适合承载账号、权限、额度和产品查询关系；MySQL 提供稳定业务模型，Kubernetes status 再同步回来供页面展示。",
            "RUNNING等于可服务吗": "不等于，Pod phase=Running 只说明 Pod 已绑定节点且容器已启动或正在启动、重启，不代表应用已经准备好。Pod Ready 决定它是否可以进入普通 Service 的就绪后端；Deployment availableReplicas 统计持续满足 minReadySeconds 的就绪副本。当前平台 RUNNING 是把 Deployment.status.replicas 和 availableReplicas 做比较得出的摘要，没有核对完整 rollout 条件。即使副本数满足，模型、依赖或实际请求仍可能失败，需要业务探针和请求指标补充判断。",
            "创建接口返回成功等于PodReady吗": "不等于，创建接口成功只表明该次同步链路完成了 Kubernetes 对象创建和数据库登记，镜像拉取、调度、容器启动仍是异步过程。当前模板可选配置 Readiness HTTP 探针；没有配置时，容器默认就绪行为不能证明模型加载和预热已经完成。Watch/List 将 availableReplicas 回写产品页面，但不会把接口阻塞到全部副本 Ready。若产品要求等待可调用，应提供异步操作状态或独立就绪查询，并设定等待时限和明确的失败原因。",
            "为什么创建顺序是K8s在前MySQL在后": "当前顺序是先让 API Server 接受 Secret、Deployment 和 Service，再登记业务对象，创建过程的资源错误可以直接返回给调用方。代价是最后数据库写入失败时，前面创建的资源需要逆序清理。也可以先在数据库登记 Creating 状态再异步创建资源，那样要多维护一套创建状态和恢复流程；当前实现采用了前一种顺序。",
            "为什么直接用JavaClient而不是Helm或CRD": "每个 Agent Server 只对应少量标准对象，生命周期又和账号、额度、MySQL 事务及产品接口紧密耦合，直接使用 Java Client 可以在一条业务链里完成校验、编排、补偿和日志查询。Watch 状态同步和跨系统补偿由业务服务自行实现，因此这套能力定位为业务编排服务，没有采用标准 Operator 模式。",
            "用户可以提交自定义镜像_怎么做安全隔离": "当前仓库能确认账号归属校验、按账号的 CPU/内存额度、容器 requests/limits 和镜像拉取 Secret；所有 Agent Server 都在 agent-server namespace。模板使用 default ServiceAccount 和空 securityContext，仓库没有证据证明已落地按租户的 RBAC、NetworkPolicy、镜像签名扫描或沙箱运行时，因此这些不能说成生产既有能力。面向不可信镜像，我会补最小权限且不挂载无用 API 凭据、准入限制高权限配置、非 root 和 seccomp、网络隔离，以及租户 ResourceQuota/LimitRange。高风险用户可进一步使用独立节点、集群或沙箱运行时；具体方案要根据威胁模型和成本选择。",
            "重复请求会不会重复建资源": "同账号重名检查和 serverName-userId 稳定资源名可以限制同名资源的数量，但不能保证整个创建请求幂等。当前没有持久化请求幂等键、创建操作状态或对 AlreadyExists 的独立处理；并发请求和 API 超时后重试仍可能遇到冲突。更危险的是失败分支按名称删除，没有核对本次创建的 UID，可能删掉另一请求已创建成功的资源。改进应以账号加幂等键创建唯一操作记录，重试返回同一操作结果，遇到已存在资源先核对归属与配置，补偿只处理本次操作拥有的资源。",
            "孤儿清理怎么防误删": "当前扫描专用 agent-server namespace，从资源名末尾解析 userId，再按 serverName 和 userId 查询 MySQL；只有更新无匹配且归属查询为零才删除同名 Service/Deployment。查询异常会停止当前同步路径，这能避免把数据库故障直接当作不存在。仍有两个明显窗口：创建资源后数据库事务尚未提交，扫描可能提前判为孤儿；同名资源删除重建后，按名删除缺少 UID 前置条件。演进时应先持久化操作状态和不可变归属 ID，对 Creating 资源保留等待窗口，并在删除前再次核对状态和实际 UID，使用 UID precondition 防止删除后来重建的对象。",
            "删除K8s成功但MySQL删除失败怎么办": "这时数据库记录仍在，但 Service/Deployment 已不存在；周期 List 只扫描现存 Deployment，无法修复这个方向。删除接口返回失败后，调用方重试同一删除请求；Kubernetes 的 404 被视为资源已删除，随后再次删除 MySQL 记录，直到整次删除完成。",
            "多实例会不会重复写": "会，多实例可以各自收到 Watch 并执行周期 List，覆盖更新同一行只能让相同值重复写入不再累加，不能避免旧快照晚到覆盖新状态。清理也可能并发执行，404 视为成功只解决部分重复删除问题，不能解决创建窗口和同名重建误删。当前没有按对象串行队列、leader election 或数据库写入 fencing 机制。要完善可以把变更和周期核对统一转成对象 key，再由有明确处理权的一方读最新状态、按业务版本和资源 UID 更新数据库；多副本处理时还要防止旧处理者继续写入。",
            "更新AgentServer的完整链路是什么": "平台先读取原有业务记录，按旧用量和新用量重新校验账号 CPU、内存配额；镜像仓凭据变化时更换 Secret，然后用完整 YAML replace 同名 Deployment 和 Service，最后更新 MySQL 期望配置，并重新计算所有关联 Agent 实际调用该服务的完整访问地址。Deployment 的 Pod template 变化会交给 Kubernetes RollingUpdate；接口成功只表示新期望已接受，不表示新 Pod 已 Ready。",
            "更新后新镜像起不来会自动回滚吗": "当前不会自动回滚。接口替换 Deployment 和 Service 后就更新业务记录，Watch 和周期 List 只同步运行状态，没有等待完整 rollout 后自动恢复上一版的逻辑。新镜像起不来时，需要检查 Pod 状态和日志，再提交已知可用镜像的更新请求。",
            "raw_Watch从哪个resourceVersion开始_410_Gone怎么办": "Watch 是 API Server 持续发送变化通知的 HTTP 流，断开后客户端需重新建立。当前代码没有保存或携带上次 resourceVersion，也没有专门处理 410 Gone；异常时等待五秒重建，再依赖周期 List 补当前态。更完整的流程是先 List 得到集合 resourceVersion，从该位置 Watch，并保存处理位置；历史失效遇到 410 时重新 List 建立快照，再从新版本继续 Watch。resourceVersion 是 API 给出的不透明令牌，不应自己转整数大小比较，也不能拿 Deployment generation 当作副本状态的新旧顺序。",
            "Watch和周期List同时回写会不会旧状态覆盖新状态": "会，Watch 和周期 List 各自读取快照、覆盖同一行，旧的观测结果可能晚写入数据库。当前没有统一的每对象处理顺序，周期同步只能在后续继续修正，不能消除这个窗口。要改进，可以让两条路径只把同一个对象 key 入队，由同一处理者重新读取最新状态再回写，并防止多实例同时处理同一对象。Deployment generation 只跟配置变更有关，不能单独判断哪份副本状态更新。",
            "怎么判断AgentServer更新真正完成": "当前平台主要比较 status.replicas 和 availableReplicas，没有把期望 spec.replicas、observedGeneration 和 updatedReplicas 全部纳入判断，因此旧副本都可用也可能提前显示 RUNNING。完整 rollout 应确认控制器已观察最新 generation、目标数量的新副本都已更新且可用，以及旧副本已退出。还要检查 Progressing 的 reason 和失败原因，不能只看 Progressing=True 或 Available=True，因为它们不等价于本次更新已经完成。当前 getDeploymentStatus 还把 ProgressDeadlineExceeded 当作 condition.type 判断，实际通常是 Progressing=False 的 reason，需要作为实现局限说明。",
            "Readiness_Liveness和StartupProbe分别负责什么": "Readiness 判断是否接新流量，失败一般使 Pod 退出 Service 的就绪后端；Liveness 失败达到门限会让 kubelet 重启该容器；StartupProbe 用于容忍较长启动时间，在它成功前不执行 Readiness 和 Liveness。当前 DTO 和模板只支持可选 Readiness、Liveness HTTP 路径，没有 StartupProbe 配置。两类探针的时序参数在模板中固定，用户不能通过当前接口自由设置；failureThreshold=100 也需要结合业务解释其发现故障的延迟。若迁移到模型服务，应新增 StartupProbe 与可配置启动预算，Readiness 覆盖模型加载和预热，Liveness 只检查可通过重启恢复的进程失活，避免把下游短暂异常引发成重启风暴。",
            "containerPort_ServicePort_targetPort和进程监听端口是什么关系": "containerPort 是 Pod 规格中对容器端口的声明，本身不会启动进程；Service port 是客户端访问入口，targetPort 是 Service 转发到 Pod 的目标端口。当前三者使用用户提交的同一端口值，但最终要求镜像内 Agent 进程确实监听该端口且绑定 0.0.0.0 或 Pod IP；如果只监听 127.0.0.1，Service 转到 Pod IP 后仍无法访问。",
            "Pod日志查询具体支持到什么程度": "当前日志接口接收 agentServerId、replicaName 和分页参数，但服务实现直接使用 replicaName 查询 Pod 日志，没有利用 agentServerId 再校验账号归属或 Pod 与服务的关系。Kubernetes 日志调用的 container、follow、previous、时间过滤等参数均为空，实际是一次拉取日志后在内存中按行分页。副本列表查询会先按账号获取 Agent Server，但不能据此认为直接调用日志接口也已被授权。完善时应重新校验账号和业务资源归属、核对 Pod UID/ownerReferences 或可信标签，再显式支持容器选择、tailLines、sinceTime、previous 和流式 follow，并限制单次返回量。",
            "举一个生命周期管理里实际处理过的问题": "Agent Server 创建跨 Kubernetes 和 MySQL，我按已经完成的步骤做补偿：Service 创建失败就删除之前建好的 Deployment 和 Secret，业务登记明确失败时再把 Service 一起逆序清理。另一个实际问题是 Watch 可能先看到 Deployment，数据库还没有写入业务记录，所以这时只返回，不立即删除资源。周期 List 再核对业务归属，处理残留的 Deployment 和 Service。创建与扫描并发时仍需保护登记窗口，这个检查本身不构成跨系统事务。",
            "更新副本时账号额度怎么计算_实际代码做了什么": "更新服务时，账号新占用按“当前总量减去这个服务的旧占用，再加更新后的占用”计算。数据库记录每副本 CPU 和内存，计算服务占用时要乘副本数，避免把一次扩容当成又新建了整个服务。当前上限是单账号 2000m CPU、4096Mi 内存，超过就直接拒绝，通过后才修改 Kubernetes 资源。",
            "Kubernetes已经有部署和扩缩容_你是不是只封装了一层API_增量在哪里": "底层确实复用了 Kubernetes API，Pod 调度、副本维护和容器重启都交给 Kubernetes。我增加的是 Agent 平台的业务规则，以及业务数据和运行资源之间的衔接。例如删除服务前检查是否仍被 Agent 引用，有引用就拒绝删除；创建时校验账号额度、登记用户归属，运行资源创建后数据库登记失败还要补偿清理。运行后通过 Watch 和周期 List，把实际状态同步到用户的 Agent 列表。",
            "两个请求同时通过账号额度检查_会不会超额": "有这个并发窗口。当前是查询账号总占用后做判断，两个请求可能读到同一份剩余额度，分别通过后合起来超限。要限制并发超额，可以在数据库里按账号原子预留额度，创建失败再释放预留，资源成功后把预留转成正式占用。",
            "为什么自己写Watch和周期List_不用Informer": "当前同步对象主要是单个 namespace 的 Deployment，raw Watch 回写运行态，周期 List 补当前快照，因此实现比较直接。代价是重连、资源版本续传、删除通知、错误事件和并发旧写都需要自己维护，当前代码并没有全部覆盖。Informer 可以复用 List/Watch、缓存和重连机制，事件回调只入队，业务工作者按 key 重新读取当前状态后处理。周期业务对账仍然有价值，因为 Informer 解决资源观察，不自动解决数据库登记、跨系统补偿和孤儿归属；其 resync 也不能直接等同于重新从 API Server 做全量 List。",
            "你负责服务部署_请求转发还是模型推理": "我主要做平台控制面的运行环境生命周期和调用计量，负责账号校验、额度、标准 K8s 资源编排、状态同步及失败处理。创建 Agent Server 后平台保存 ClusterIP Service 的集群内 DNS、端口和主路径，Framework Agent 引用它形成 requestUrl。真正的 Agent 进程运行在用户镜像中，模型推理实现不属于这一段 Java 编排代码。当前仓库能证明服务地址登记和计量入口，不能据此说我实现了独立 Gateway 的流式代理、路由算法或 vLLM 推理引擎。",
            "Agent平台和MaaS是什么关系_经验怎么迁移": "Agent 平台提供 Agent 的开发、发布、托管和调用，MaaS 通常围绕模型 API 和推理服务；两者在服务生命周期、身份额度、状态同步、调用计量上有共同问题。我的代码经验可迁移到 MaaS 的控制面，例如部署与更新服务、同步运行态，以及异步记录用量。当前 TP 仓库的计量主要是调用次数和开发者积分，并通过 LiteLLM 管理会员调用预算，没有完整的 Tenant/Project/Model Token 成本账本。GPU 调度、推理性能优化和 Token 账单可以作为目标场景设计讨论，需要明确是已有实现之外的扩展。",
            "Deployment创建成功_Service创建失败怎么办": "当前实现捕获 Service 创建异常后，按名称删除已经创建的 Deployment，再删除镜像 Secret，并向用户返回部署失败。若删除也失败，代码只是记录日志，不会同步阻塞到资源确实消失。周期任务可以再次发现没有业务记录的 Deployment 并尝试清理，但只剩 Service 或 Secret 时没有完整扫描路径。还必须区分明确失败和超时结果不明：超时可能表示 Service 已经创建，恢复流程应先查询核对实际对象，再决定补偿。",
            "补偿删除一直失败_谁来负责收尾": "当前 rollback 方法只做一次删除并记录日志，没有持久化补偿任务、重试上限或失败状态。周期 List 会再次尝试清理仍能从 Deployment 枚举到的无归属资源，但这不覆盖所有残留资源，也没有自动升级人工处理的闭环。设计上应保存操作 ID、步骤、最后错误和下次重试时间，由后台流程按退避继续推进，404 视为完成而权限、网络等错误保留诊断信息。超过业务时限后告警或转人工，但操作记录仍保留，避免接口返回失败后无人跟踪残留。",
            "平台重启后怎么发现遗留资源": "启动时会启动 Watch，定时任务再 List 当前 namespace 的 Deployment，并按资源名回查业务记录。它能发现部分 K8s 有资源但 MySQL 无记录的情况，不能恢复创建流程的每个已完成步骤，也不能独立覆盖 Secret、Service 或数据库有记录但 Deployment 不存在。更完整的恢复应从持久化未完成操作入手，逐步读取实际资源，继续创建或补偿；同时枚举数据库和各类受管 K8s 对象做双向核对。资源应携带受管标识、不可变业务 ID 和创建操作 ID，恢复时依据 UID 确认自己处理的是同一实例。",
            "创建与周期孤儿扫描并发_为什么会误删": "当前创建先写 Kubernetes，最后才 insert MySQL，而且数据库事务提交发生在方法返回之后。周期 List 如果恰好在这个窗口读取到 Deployment，却还看不到已提交业务记录，就可能按孤儿删除它；Watch 路径发现无记录直接返回只能保护 Watch，保护不了周期清理。名字里带账号 ID 也不能消除这个时间窗口。设计上可以先提交 Creating 操作和归属，再创建 K8s 资源，扫描识别创建中状态并延后清理，删除前再核对操作状态和资源 UID。",
            "接口超时后重试_怎么区分没创建和已创建": "客户端超时只说明没拿到结果，不能推断服务器或 API Server 没有完成操作。当前稳定资源名使重试指向同一对象，但缺少请求操作记录和 AlreadyExists 的恢复分支，直接重建再按名补偿可能处理错已有资源。设计上由客户端或入口提供稳定幂等键，以账号加幂等键保存唯一操作，重试先返回或继续同一操作。对 K8s 超时则读取同名资源并核对业务 ID、配置和 UID，明确不存在再创建，明确属于本次操作再推进，避免盲目删除。",
            "Deployment被删除后平台怎么知道_当前代码有缺口吗": "Watch 返回的通知包含 ADDED、MODIFIED、DELETED 等类型，但当前循环没有看 item.type，直接把 item.object 的最后状态回写。DELETED 携带的是被删对象的最后快照，仍可能含有可用副本，因此平台可能继续显示旧状态。周期 List 只遍历当前仍存在的 Deployment，也不会主动把缺失的数据库行改成 NOT_FOUND。完善时应处理 DELETED 并核对对象 UID，避免旧删除通知影响同名新资源；周期再从业务记录反查真实对象，用明确 404 纠正缺失状态，并区分用户删除中和意外丢失。",
            "更新镜像怎么保证服务可用_当前模板保证了什么": "当前模板配置 RollingUpdate、maxUnavailable=25%、maxSurge=25%，并可选择 Readiness，Kubernetes 据此逐步替换副本。它能控制副本更替节奏，但业务可用依赖正确探针、旧副本健康、足够冗余和新副本可调度资源。新旧版本并存期间还需要接口和状态兼容，模型冷启动时要提前准备加载与预热时间。当前服务没有等待完整 rollout、灰度流量或失败自动回滚闭环；升级 MaaS 可以增加发布状态、启动超时、灰度验证和恢复上一版流程。",
            "缩容时正在执行的请求或流式响应怎么办": "当前更新只是修改期望副本数，没有实现应用层请求排空；模板提供三十秒 terminationGracePeriodSeconds，但它不自动保证长请求执行完成。更完整的流程是让待退出实例停止接收新请求，等待入口的后端更新传播，同时由应用处理 SIGTERM 并等待已接收请求完成。流式响应可能长于宽限期，应根据业务设置排空时限、入口超时和连接关闭策略，必要时保留原实例到流结束。到期仍无法完成需要明确中断语义，不能声称换到另一副本就能无缝继续同一生成过程；这一块属于后续设计。",
            "模型加载很慢_怎么避免容器反复重启": "当前模板没有 StartupProbe，只提供 Readiness/Liveness 路径，若 Liveness 启动预算不合适，模型加载期间可能被误判失败。演进时用 StartupProbe 容忍加载与预热，在它成功前不让 Liveness 介入；Readiness 在模型可服务之后再成功。Liveness 主要判断进程是否失活，把下游短暂不可用当作存活失败可能造成反复重启。启动预算应根据实际加载时间和资源条件配置，同时记录拉镜像、调度、加载、预热各阶段耗时，区分启动慢和真正失败。",
            "配额预留_正式占用和失败释放怎么设计": "当前只有读取数据库总量后判断，没有独立预留状态，两个并发请求可能一起超额。改进可以在同一数据库事务内创建唯一操作并对账号额度原子扣减或锁住额度行，把新用量记为预留；创建资源成功后将预留转为正式占用。超时不能立即释放，因为资源可能已经创建，应先核对并确认清理完成，再幂等释放对应操作的预留。对卡住的预留设置过期检查和后台对账，释放依据操作状态而非单纯时间到期，避免额度已释放但资源仍在。",
            "扩容缩容失败_配额应该什么时候增减": "当前更新按账号总用量减旧占用加新占用检查，但 K8s 先更新、MySQL 后更新，失败时账面和实际配置可能不同。扩容应先预留增量额度，资源更新确认后转正式占用；更新失败需要核对是否仍有新增副本，清理或撤销后再释放预留。缩容也不能只因修改期望就立即认为运行资源已经释放，要约定按期望预算还是实际占用计账，并核对终止中的副本。滚动更新的 maxSurge 会临时超过目标副本数，物理容量和 namespace quota 要另留发布余量，不能只看业务的目标配置总量。",
            "CPU内存_GPU配额和Token预算有什么不同": "CPU、内存的 requests 用于调度和资源预留，limits 控制容器使用边界，账号额度是平台准入规则；当前模板把每副本 requests 和 limits 设成相同值。GPU 配额通常限制可申请设备数量或特定切分资源，显存与共享策略需要结合设备方案，当前 Agent Server 模板没有 GPU resource 字段。Token 预算限制一段时间内的调用用量或费用，是请求侧消耗统计，不是 Kubernetes 运行资源。三者可以共同约束租户，但需要各自明确计量单位、周期、并发扣减和释放规则，不能用一个CPU/GPU quota替代Token预算。",
            "KubernetesResourceQuota能解决按账号并发超额吗": "ResourceQuota 是 namespace 范围的准入额度，能限制该 namespace 可创建 Pod 的资源总量；它不会自动识别平台数据库中的账号。当前所有账号共用 agent-server namespace，因此单个 ResourceQuota 只能兜底这一 namespace 的总体占用，不能直接表达每账号上限。若采用租户 namespace，可以配合 ResourceQuota、LimitRange 和平台原子预留做两层限制，但还要处理 Deployment 已创建而 Pod 因 quota 不足无法创建的状态。数据库额度控制业务预算，K8s quota 控制集群准入，两边都需要对账和错误展示。"
          },
          "实践案例": {
            "更新副本时重新核算账号额度": {
              "实践类型": "本人实现；数值为代码门限，不是线上效果统计。",
              "口述回答": "更新服务时，账号新占用按“当前总量减去这个服务的旧占用，再加更新后的占用”计算。数据库记录每副本 CPU 和内存，计算服务占用时要乘副本数，避免把一次扩容当成又新建了整个服务。当前上限是单账号 2000m CPU、4096Mi 内存，超过就直接拒绝，通过后才修改 Kubernetes 资源。"
            },
            "创建部分成功与Watch先到的处理": {
              "实践类型": "本人实现中的具体失败分支和异步事件处理；没有故障注入执行记录。",
              "口述回答": "创建跨 Kubernetes 和 MySQL，当前 Service 创建失败会删除 Deployment/Secret，insert 明确返回失败再把 Service 一起清理。Watch 如果先看到 Deployment 而数据库还未登记，会直接返回，避免在 Watch 路径立即当孤儿删除。周期 List 再按数据库归属清理仍能从 Deployment 枚举到的残留。这里不能夸大为完整恢复：创建与周期扫描并发时可能误删尚未提交的资源，补偿失败只记录日志，独立 Service/Secret 残留也未全部覆盖；操作状态与 UID 核对是后续完善方向。",
              "补偿边界": "insert 返回非正数有显式清理，insert 抛异常或事务提交失败不等于已走补偿；周期仅枚举 Deployment，没有创建中标记、登记宽限期或按 UID 删除条件。API 创建响应超时需读取确认结果，同名请求冲突后不能盲目删除其他操作的资源。"
            }
          }
        },
        "kafka_mysql_mongodb": {
          "主题": "Agent 调用计量与资源治理链路",
          "参考回答": "我把 Agent 调用和发布事件接入 Kafka，将在线请求和统计处理解耦。消费端保存 MongoDB 明细并更新 MySQL 汇总，用 eventId 加 Redis 缓存抑制短期重复，业务失败做有限重试和失败留存。当前版本的计量口径是调用次数、用户和开发者积分；Redis、双库和 offset 分开提交，所以不能称为严格幂等或跨库强一致。进一步完善应持久化唯一事件身份、让每个投影在本地事务内幂等更新，并建立可靠失败交接与对账重建。",
          "链路说明": {
            "一句话主线": "上游调用或发布事件 → Kafka → 字段校验与 Redis 短期重复抑制 → Mongo 明细与 MySQL 汇总分别写入 → Redis 完成标记；部分业务失败进入有限重试和失败留存。当前版本不具备已验证的严格幂等、可靠失败交接或双库原子提交，新的唯一账本与投影方案属于改进设计。",
            "链路概览": [
              "1. 事件入口：调用与发布进入不同 Topic，生产发送异步；接口返回不等于 broker 已确认，在线请求不等待双库统计完成。",
              "2. 消费与去重：解析并校验后查询 Redis；调用与发布的校验和坏消息分支不同，缓存只能抑制短期重复。",
              "3. 核心写入：Mongo 保存调用、用户和积分文档，MySQL 更新累计统计；旧调用/用户明细缺 eventId，不能据此保证无损重建。",
              "4. 成功标记：业务执行后记录24小时 processed key；数据库、Redis和offset分开提交，CTM默认代理下的自调用不启动注解事务。",
              "5. 失败处理：业务异常最多总计尝试三次，等待5秒、10秒；耗尽后保存FailedMessage并异步尝试DLQ，交接确认与自动回放仍需完善。"
            ],
            "完整链路_7阶段": [
              "1. 上游提供 eventId；调用 API 转发请求到 Kafka，当前生产端异步 send，失败回调只记录日志，API 立即返回，不构成已确认送达承诺。",
              "2. 调用与发布分别由独立 Topic 和 Consumer Group 处理。仓库 Listener 签名意图批量消费，具体 batch、AckMode 和 consumer 参数由外部 Nacos kafka-business.properties 决定，不能从仓库确认生产值。",
              "3. 调用消费者要求 eventId 非空；生命周期消费者未同等校验。两者检查 Redis processed key，相同 ID 被短期抑制；Redis 故障放行，24小时TTL及检查后写入都留下重复窗口。",
              "4. Mongo 保存调用、用户和开发者积分，MySQL increment/upsert 更新累计汇总；usage是调用次数，revenue对应积分。调用和用户明细无 eventId，重试重新生成文档 ID；积分有 eventId 但无唯一约束。",
              "5. processMessage 的事务注解通过同Bean方法直接调用，在本地默认代理模式下不被拦截。即使修正为代理入口，CTM也可能部分提交；increment更新0行仅warning，仍可能继续标成功。",
              "6. 业务完成后写Redis成功标记；数据库、标记与offset没有同一事务。消费者吞掉的异常不能让容器自动知道批次未完成，实际offset推进需结合外部容器配置核对。",
              "7. 抛出的业务异常总共尝试3次，等待5秒、10秒，耗尽后插入FailedMessage并异步尝试DLQ；send Future未确认，文件降级非高可用。调用坏JSON可中断批次，参数拒绝等路径未进入可靠恢复队列。"
            ],
            "提交与恢复条件": "本版本 Redis、MongoDB、MySQL 与 Kafka offset 分开提交；CTM 自调用路径默认不经事务代理，increment 返回 0 只 warning，坏 JSON 和 DLQ 异步发送也有未可靠交接路径。AgentUsage/AgentUsers 没有 eventId，不能作为保证可无损重建的事实账本。设计建议是唯一原始账本、每个存储独立幂等、本地事务内提交投影 receipt 与聚合、可靠失败队列和对账恢复。"
          },
          "组件职责": {
            "Kafka": "解耦事件生产与统计写入、吸收突发并支持消费者独立扩容；它不负责 MySQL、MongoDB 等外部副作用的原子提交。",
            "Redis": "以 eventId 做快速重复抑制，降低相同消息重投的重复处理；它不是持久化事件账本，也不能单独提供严格业务幂等。",
            "MongoDB": "当前保存调用、用户和积分等文档明细并支持时间聚合查询。AgentUsage/AgentUsers缺少eventId且可能因重试重复，不能直接称为可靠事实账本；改进以稳定唯一事件ID保存原文与规则版本，使投影可核对和重放。",
            "MySQL": "保存关系型业务聚合与稳定查询结果；原子自增可以防止并发写丢更新，但同一事件重复消费仍可能重复累加。",
            "ChainedTransactionManager": "配置组合 MongoDB/MySQL 本地事务，但当前 Consumer 内部自调用 processMessage 在默认代理模式下不会被事务拦截。即使移到代理入口生效，CTM 也可能部分提交且已弃用，不能提供双库原子性。",
            "重试与DLQ": "短暂故障由有限重试吸收；重试耗尽后先保留失败现场，再尝试投递 DLQ。DLQ 保存排查和补偿入口，其发送结果与失败记录需要分别跟踪。"
          },
          "一致性与失败处理": {
            "重复消息": "当前先查 Redis，再写 Mongo/MySQL，最后标记 processed；检查与业务结果不在同一事务内，不能确保并发或崩溃后的事件只算一次。改进时以持久唯一事件身份保存原始事实，再在每个目标库的本地事务内把投影处理记录与聚合更新一起提交；Redis 仅缓存最近完成结果，恢复分别补缺失投影。",
            "Redis不可用": "Redis 查询异常时，链路继续处理核心计量，重复消息可能再次写入双库。这是在计量可继续处理与重复抑制之间的取舍；Redis 检查、业务写入和 processed 标记分开执行，严格幂等需要持久化唯一事件键与对应事务机制。",
            "双库部分提交": "本地默认代理模式下processMessage自调用不启动注解事务，任何已完成写入都可能独立保留；若修正代理入口，CTM仍存在提交阶段部分成功。MySQL 0行只warning也会被上层当成功。建议每个库自行幂等，对各投影完成状态分别确认并修复，不能把“任一库已有eventId”当成整条链路已完成。",
            "Offset推进": "Kafka offset 与数据库提交不在一个原子事务。当前 batch Listener 在成功处理或把失败交给失败记录/DLQ 后都会正常返回，因此 offset 按部署时的容器 ack 配置推进；这条链路的坏消息恢复依赖失败记录与 DLQ，而不是假设抛异常后 Kafka 必然重投。",
            "消息顺序": "Kafka 只保证同一分区顺序，当前生产者没有显式 Agent 业务 key，且调用和发布跨 Topic，不能依赖它们按业务顺序到达。需要顺序时按稳定 key 分区并校验业务版本；跨 Topic 的前置依赖通过暂存重试或幂等补建统计基础行处理。",
            "失败交接范围": "可重试业务异常耗尽后插入Mongo FailedMessage并异步尝试DLQ，Future未确认；本机文件降级不是高可用。调用坏JSON可能中断整批，生命周期解析失败只跳过，参数拒绝等不进入可靠失败队列。改进需保留来源topic/partition/offset及原文，可靠隔离契约错误，确认失败接管后才按容器策略推进，交接失败不能吞异常。"
          },
          "高压追问参考回答": {
            "这是不是计费系统": "这是 Agent 调用计量、开发者积分统计和资源治理链路，不涉及财务扣费和出账。它统计调用量、独立用户和开发者维度数据，为资源配额及成本治理提供数据基础。",
            "请具体讲一下Kafka计量链路": "一条调用事件进入消费者后，先校验 Agent 和 eventId，检查是否已经处理，再按新老用户写入用户、积分和调用明细，最后更新 MySQL 的收入、用户数和调用数。正常处理完才写入有效期 24 小时的 Redis 完成标记。抛异常时最多尝试三次，耗尽后保存 FailedMessage 并尝试发到 DLQ，发送采用异步调用。这条链路按事件推进明细和汇总，重复处理与部分成功还需要单独核对。",
            "为什么要Kafka": "Kafka 把调用事件统计从在线请求里拆出来，业务请求不用逐次等待 MongoDB 和 MySQL 的统计写入。消费者可以按数据库承载能力处理积压，流量突增时先由队列缓冲。代价是统计存在延迟，而且消费重试可能重复执行，所以需要处理去重、失败记录和双库部分成功。",
            "为什么同时用MongoDB和MySQL": "这套设计按数据模型和查询方式分工：MongoDB 保存逐条调用、用户和积分明细，MySQL 维护产品页面使用的调用量、用户数和收入汇总。明细保留每次业务发生的记录，页面查询则直接读取汇总，不必每次重新扫描明细计算。MongoDB 承接文档形式的明细，MySQL 承接关系型业务查询；如果只需要固定字段的调用计数，用一套 MySQL 也可以。双库的成本是同一事件要分别写入，部分成功时需要核对明细和汇总，处理补写或重算。",
            "Redis_eventId等于严格幂等吗": "不等于。当前代码先查 Redis，写 Mongo/MySQL，再写 24 小时标记。两个消费者可以同时查到未处理；写库后进程崩溃、Redis 标记失败或过期，也会再次处理。MySQL 原子 increment 不丢并发更新，但同一事件执行两次仍加两次。更可靠的方案是在负责聚合的 MySQL 本地事务内，先插入带唯一键的事件处理记录，只有首次插入者才更新聚合，两者一起提交；Mongo 明细则按同一稳定 eventId 做唯一写入或从可靠账本异步生成。每个存储自行幂等，跨库依靠重试与对账最终收敛，不能称为一次跨库原子提交。",
            "eventId由谁生成_业务重试能换吗": "当前调用 API 从上游请求直接读取 eventId，仓库不能证明上游实际生成规则。同一计量事件重发、消费重试和 DLQ 回放应保留原 ID；新的真实调用用新 ID。eventId 标识一条计量事实，requestId 标识逻辑请求，attemptId 标识一次真实执行尝试，三者不要混用。两个 Topic 共用 processed key，要么约定 eventId 全局唯一，要么把租户、事件类型或 Topic 纳入去重命名空间；同 ID 不同内容应比对 payload hash 并报警，不能静默当重复。当前生命周期消费者还缺 eventId 非空校验，这会使缺 ID 的不同消息共用一个 Redis key。",
            "换成SETNX就够了吗": "不够。先 SETNX 再写数据库，进程挂在中间会留下未完成占位，重投可能被跳过；先写数据库再 SETNX，仍有重复窗口。TTL 能让占位释放，但释放后原处理者可能继续执行，Redis 锁也不能替代数据库幂等。我的改进方向是把唯一事件处理记录与该数据库的聚合更新放在同一个本地事务里：首次插入成功才更新，失败一起回滚，唯一冲突表示这个投影已完成。事务的锁和异常处理要以数据库实际语义为准，而不是先查再插；跨库分别执行可重试的幂等写入。",
            "ChainedTransactionManager是2PC吗": "不是。CTM 顺序启动、提交多个本地事务，没有 2PC 的 prepare 协议；先提交的库已经成功后，另一个库仍可能提交失败，因此会出现部分提交，Spring 官方也已将它弃用。若要保留双库，我会采用可靠原始账本加异步幂等投影，明确提供最终一致性；CTM 不应被介绍成双库严格原子事务。",
            "CTM在生产链路里怎么生效": "这个版本注册了 CTM，在 processMessage 上加了 @Transactional，但 processWithRetry 在同一个 Bean 内直接调用它。按当前默认代理模式，这条自调用不会经过事务拦截，因此注解不会在这条路径开启事务。改进可以把数据库业务方法移到独立 Bean 后通过注入对象调用，或用 TransactionTemplate 明确事务范围；每次重试应开启新事务。修正入口后，CTM 仍有部分提交风险；更强的保证需要让每个存储的处理记录和该库副作用处于同一个本地事务。以上针对当前仓库调用路径，历史生产版本和事务验证结果不能从这份代码推断。",
            "Mongo成功MySQL失败怎么办": "当前 MySQL 抛异常会触发整条处理重试，但 Mongo 明细的 ID 每次重新生成，部分写入后重试可能再插一份；CTM 自调用路径也不能提供预期回滚。另一个确定的遗漏是 increment 返回 0 只记 warning，随后仍可能标记 processed。改进先把预期更新一行却为 0 作为异常或明确的待处理状态，再让明细和每个聚合投影使用稳定事件键。Mongo 明细已确认存在不代表 MySQL 也完成，恢复应继续完成缺少的投影，不能因为任一存储发现 eventId 就整条跳过。",
            "原子自增等于幂等吗": "不等于。原子自增解决并发更新的丢更新；重复消费仍会把同一条调用加两次。建议在 MySQL 事务内插入 UNIQUE(projection_name,event_id)，只有首次插入成功才 increment 并一同提交。这里的 projection_name 区分调用量、积分等独立投影；同一事件有多个积分类别时还需包含奖励类型或业务操作键，不能只用一个 eventId 把合法的不同副作用全部去掉。",
            "Offset什么时候提交": "仓库的 @KafkaListener 接收 List，意图是批量消费，但实际 batch 配置、AckMode 和 enable.auto.commit 来自 Nacos 导入的 kafka-business.properties，仓库没有其内容，所以不能声称使用手动提交或确定提交时点。两类 Listener 会捕获异常后正常返回，DLQ send 也没有等待结果；容器可能按正常完成推进 offset。设计上应明确约定：核心数据已持久化，或失败消息已可靠移交恢复通道后，才允许推进；同一分区不能提交跨过尚未处理或尚未交接的较小 offset。数据库成功后 offset 前崩溃仍会重投，必须由存储幂等吸收。",
            "Kafka能保证消息顺序吗": "只保证同一分区的追加顺序，不保证跨 Topic、跨分区的业务顺序。当前 MessageProducer.sendMessage(topic,message) 没有提供 Agent 业务 key，因此不能默认同一 Agent 的事件有序。仓库的发布和调用在不同 Topic，还可能出现调用先到、统计行尚未创建，increment 返回 0 的情况。若某些状态确实要求顺序，可按稳定 Agent key 分区并使用业务版本校验；跨 Topic 依赖仍需暂存重试、创建幂等的统计基础行或状态机处理，不能只靠相同 key 解决。",
            "一批消息中间有一条坏消息怎么办": "当前调用消费者的 JSON 解析发生在单条业务 try 之外，坏 JSON 会落入 batch 最外层 catch，中断后面的记录并让 Listener 正常返回；生命周期消费者对解析错误记录日志后跳过。两者都没有把这类坏消息可靠写入现有 FailedMessage/DLQ。改进应保留原始 payload 和 topic/partition/offset，以它们作为缺少 eventId 时的恢复身份；不可恢复的格式或契约错误可靠隔离，短暂依赖故障做有限重试。失败交接不成功时要向容器暴露失败或暂停，不能吞异常后把整批视为成功。具体采用逐条 Listener 或批次失败索引由所用 Spring Kafka 版本和容器配置决定。",
            "重试和DLQ能保证不丢吗": "当前不能。业务总共尝试 3 次，等待 5 秒、10 秒，是线性递增，代码注释的 exponential backoff 不准确。耗尽后先插入 Mongo FailedMessage，再异步 send DLQ；没有等待或监听 send 的 Future，日志中的 sent 不等于 broker 确认。FailedMessage 插入或同步 send 抛异常时只降级写本机 /tmp 文件，文件不具备高可用恢复保证。改进可以把失败记录作为可靠恢复队列：记录 PENDING、错误、来源 offset 与原始消息，后台调度器发送 DLQ，收到确认后再标记 SENT，失败继续重试并告警。重复 DLQ 投递仍复用原 eventId，后续恢复也要幂等。",
            "MongoDB索引怎么设计": "当前 AgentUsage、AgentUsers 和 DeveloperPoints 对 agentId、timestamp 分别建了单字段索引，可以支持单字段过滤；当前索引不是 agentId+timestamp 复合索引。组合身份加时间范围查询是否需要复合索引，要用真实查询和 explain 的扫描量、排序情况决定。",
            "独立用户数是严格distinct吗": "当前不是。新用户判定查询的是 DeveloperPoints 是否已有该 Agent/user 的调用奖励，而不是向独立用户唯一表原子登记；不同事件并发时可能都判为首次。关闭 pointsEnabled 后不再写积分，该判定还可能持续认为用户首次出现。建议把独立用户事实与积分功能开关分开，在与用户聚合相同的 MySQL 事务中插入 UNIQUE(agent_id,user_id)，只有首次插入者增加总独立用户数。累计独立用户、每日活跃用户和当天新增用户是不同统计口径，要先明确日期、时区和唯一键。",
            "怎么发现少算或重复": "先确定统计口径和数据完整性，再比较可靠事件集合与各投影完成记录。现有 AgentUsage/AgentUsers 没有 eventId，Mongo 本身可能含重试重复，不能直接把它的总条数当作真实调用数；DeveloperPoints 虽有 eventId，同一事件可产生新用户和调用两类奖励，要按 eventId 加奖励类别核对。结合 Kafka 留存原文、上游请求日志和 FailedMessage 可辅助定位，但如果原始唯一身份已经丢失，就只能确认差异或做有说明的近似修复。新设计应持久化稳定 ID 和来源位置，使每个事件能逐项核对。",
            "数据库变慢会怎样": "消费者处理耗时上升、Kafka lag 增长，页面统计延迟也增大。先同时看每分区 lag、最老未处理事件年龄、消费耗时、重试比例，以及 MySQL/Mongo 的连接池等待、锁等待、慢查询和写入延迟。数据库已满载时加消费者会放大压力，应限制并发、修复慢查询与热点、在幂等前提下批量写入；消费端 CPU 有瓶颈且数据库有余量时才扩并发。增加实例仍受分区数和热点分区限制。长业务处理加 Thread.sleep 还会逼近 max.poll.interval.ms，需控制单次 poll 记录数和处理时间，设置有界重试；不能仅把超时调大当作容量改进。",
            "调用事件和发布生命周期事件的Consumer有什么不同": "调用消费者累计 usage、首次用户和开发者调用积分；发布消费者处理首次发布、审核通过和发布时间等副作用，两条链路使用不同 Topic 和 Consumer Group。共同使用 eventId/Redis 和有限重试，但校验与失败分支不同：调用有 eventId 非空校验且坏 JSON 可能中断批次，发布解析异常被单条跳过且 eventId 非空未校验。它们还跨 Topic，发布统计行与调用 increment 存在到达顺序风险，所以不能把两个消费者描述成完全相同的统一可靠事件框架。",
            "ConsumerGroup_partition数和消费并发度是什么关系": "普通 Consumer Group 中，每个 partition 同一时刻分配给一个消费者，一个消费者可以负责多个 partition。比如四个分区、两个消费者时，每人可以处理两个；扩到四个消费者后，重平衡可以改成每人一个，增加并行处理能力。消费者数超过分区数时会出现空闲实例。实际吞吐还受热点分区和数据库写入能力限制。",
            "消费中发生rebalance会不会重复计量": "会。数据库已写成功、offset 尚未提交时崩溃或 rebalance，新消费者会从已提交位置重新读，同一事件可能再次执行。数据库重复操作必须由稳定 eventId 和本地事务内的唯一处理记录吸收；Redis TTL 不是充分保障。处理线程长期 sleep 或数据库卡顿也会影响 poll 间隔，增加 rebalance 风险；调优要同时看消息处理耗时和容器配置。Kafka 对消费位点的管理不会回滚已经发生的外部数据库副作用。",
            "DLQ消息怎么重新消费": "仓库能确认失败记录和 DLQ 投递尝试，未找到自动回放服务。回放前按错误原因决定修配置、补业务依赖还是丢弃非法消息；保留原 eventId、事件时间和原始规则版本，新增 replayId 记录这次修复操作，控制速率并观察结果。对当前数据，要先确认各存储完成到哪一步，不能把全部消息再次走非幂等 increment。建议链路使用每个投影的持久完成记录，回放只补缺失投影；Redis key 存在或已过期都不能作为最终依据。",
            "Redis去重key过期后旧消息重放怎么办": "当前过期后会再次写入，旧调用量和积分可能重复增加；24 小时只是缓存抑制窗口。建议把 eventId 去重依据放到持久账本及每个目标库的处理记录，保存周期至少覆盖 Kafka 留存、DLQ 最长回放周期和业务审计要求，再讨论归档。归档、删除后的旧事件回放要有版本和重建策略。Redis 可以短路最近已完成事件，但重建指定投影时要绕开这一缓存，否则会把合法的重算也拦掉。",
            "MongoDB明细和MySQL聚合不一致时怎么补算": "先确认 Mongo 明细是否可信，不能默认它是事实账本。当前 AgentUsage/AgentUsers 缺 eventId，重试生成新 ID，可能同时存在重复和遗漏；仅根据 agent、user、时间接近不能无损识别同一调用。若 Kafka 留存或上游日志仍有稳定 eventId，应重建去重后的原始事件集合，再按固定规则生成投影；证据已丢失时明确无法保证精确恢复。重建选择截止位点或数据快照，在临时表中生成新版本并核对；在线消费者继续写时用增量补齐到切换点，避免覆盖新数据，也不能在旧聚合上无条件追加。当前 MySQL 是累计聚合，没有每日明细桶，不能直接宣称已有固定窗口覆盖实现。",
            "别只说Kafka加Redis_一条调用具体怎么计量和处理失败": "一条调用事件进入消费者后，先校验 Agent 和 eventId，检查是否已经处理，再按新老用户写入用户、积分和调用明细，最后更新 MySQL 的收入、用户数和调用数。正常处理完才写入有效期 24 小时的 Redis 完成标记。抛异常时最多尝试三次，耗尽后保存 FailedMessage 并尝试发到 DLQ，发送采用异步调用。这条链路按事件推进明细和汇总，重复处理与部分成功还需要单独核对。",
            "数据库成功但还没写Redis标记就挂了怎么办": "当前重投会再次落库，Mongo 新 ID 会新增文档，MySQL 再 increment，所以会重复。Redis 标记即使写得更早也会制造漏处理窗口。建议由 MySQL 本地事务把唯一 eventId 处理记录和统计更新一起提交：重投时唯一键已存在就不再更新；Mongo 使用稳定文档 ID 或唯一 eventId 插入。这两个库不是一次原子提交，恢复必须分别判断是否完成，缺哪个补哪个。Redis 改为成功结果缓存，标记失败不影响持久幂等。",
            "两个消费者同时发现eventId不存在怎么办": "当前 Redis hasKey 是检查后执行，两者都可能继续写；eventId 相同是否落到同一 Kafka 分区也未由当前生产者显式业务 key 保证。建议把并发裁决交给数据库唯一约束：一个事务插入处理记录成功并更新聚合，另一个等待后看到唯一冲突，读取已提交结果并结束。若赢家回滚，输家仍可重试处理。不要先查数据库再插作为保障，唯一约束与聚合更新必须在同一个本地事务里。",
            "Redis故障时继续消费还是停止": "当前查询失败返回未处理，标记失败只记日志，是 fail-open，能继续消费但会增加重复风险。若已有持久事件唯一键与幂等投影，Redis 故障只降低去重缓存效率，可以继续并告警；当前版本尚无这个基础时，继续处理与重复风险必须明确，必要时暂停计量消费者等待恢复，Kafka 缓冲容量和保留周期要覆盖故障。不能同时说 Redis 无关紧要又依赖它确保不重复。",
            "如何设计既能重放又不跨库误称原子的计量链路": "建议先选定权威原始账本：消费事件以稳定 eventId 唯一写入，只有确认持久化后才推进入口 offset。之后独立生成 Mongo 查询明细和 MySQL 汇总；在 MySQL 每次事务中，写 UNIQUE(projection_name,version,event_id) 的处理记录并更新聚合，一同提交，重复时不再计数。投影处理状态可扫描账本或由可靠 outbox 分发，失败反复重试；不能先提交账本后随手 send 一次通知就假设一定被投影。这里提供的是可重放的最终一致性，各存储只有本地原子性，账本持久化的 write concern、复制和归档也要有明确约定。",
            "Mongo明细已经有eventId为什么不能直接跳过整个消息": "因为一个 eventId 的明细存在只证明明细写过，不证明 MySQL 聚合或积分投影已提交。如果 Mongo 成功后 MySQL 失败，下一次整体跳过就永远少算。正确做法是以事件账本判断是否收到，再分别按投影完成记录补齐副作用；同一事件的不同奖励或不同聚合也要分别记录。幂等标记的含义必须明确到它对应的持久结果。",
            "DeveloperPoints有eventId就一定能精确重放积分吗": "也不能。它没有 eventId 唯一约束，一个调用还可同时生成新用户和调用两类奖励；重试可能重复插入，需按事件及奖励类别辨别。更大的问题是当前奖励金额依赖处理时的会员等级和倍率，每日五次判断查询的是当前日期，旧事件延迟到达或回放时语义可能变化。建议保存事件发生时间、奖励类型、规则版本和采用的等级/倍率；把首次用户身份与每日额度做原子登记，重放沿用原计量决策或明确进行规则迁移，不能按今天的状态再次算一遍。",
            "为什么不能把MySQL更新时间当事件时间": "Kafka 消费可能延迟、重试或回放，更新时间是处理时间，不能代表调用实际发生时间。当前消息已有 timestamp，统计日桶应按事件时间和明确时区生成；超过窗口的迟到事件采用补写或更正流程，并保留 receivedAt/processedAt 衡量计量延迟。新增日聚合时要约定窗口 [start,end)，避免相邻窗口边界重复。",
            "DLQ发送失败还能提交原消息offset吗": "只有消息已经被另一条可靠恢复路径持久接管，才可以。当前 FailedMessage 是 Mongo 留存，但没有确认的恢复调度器，DLQ 异步 send 结果也没处理，不能说已完成安全交接。改进可让原消息处理依赖确认的失败队列落库，并用后台任务重发 DLQ；或者等待 broker 确认后再让容器推进，超时或失败继续重投。两种方案都保留稳定身份与来源位点，并用幂等吸收交接成功但 offset 未提交的重复窗口。",
            "在线API返回成功就证明调用事件已经进Kafka吗": "当前不是。MessageProducer 用 KafkaTemplate 异步发送，回调失败只 log；agentInvocations API 发起发送后立即 return true，成功响应不是 broker ack。另一个代码细节是构造了转换 timestamp 的 AgentInvocationsMessage，却实际序列化原始 Request，需校验线上时间格式契约。计量要求可靠接收时可等待有界发送确认，或先持久化 outbox 再由后台发送；不能阻塞业务过久时定义响应为 accepted 并给可查询状态。生产者幂等可减少 broker 层重试重复，不替代业务 eventId 和外部存储幂等。",
            "怎么验证计量可靠性而不是只讲组件": "围绕故障窗口验证：并发提交同 eventId；Mongo 成功后 MySQL 失败；数据库成功后 Redis 失败和 offset 前崩溃；坏 JSON 夹在批次中间；DLQ 异步发送失败；跨 TTL 的回放；重复 eventId 携带不同内容。检查结果不只看 Listener 有没有异常，而是比对唯一事件数、每个投影 receipt、聚合值和恢复队列状态；对账指标包括积压年龄、重复拒绝数、缺失投影、失败交接未完成数和修复耗时。这是建议验证计划，仓库没有证据表明这些验证已做。"
          },
          "实践案例": {
            "一次调用的计量和失败交接": {
              "实践类型": "本人实现；未找到生产事故或失败注入原始记录。",
              "口述回答": "一条调用事件进入消费者后，先校验 Agent 和 eventId，检查是否已经处理，再按新老用户写入用户、积分和调用明细，最后更新 MySQL 的收入、用户数和调用数。正常处理完才写入有效期 24 小时的 Redis 完成标记。抛异常时最多尝试三次，耗尽后保存 FailedMessage 并尝试发到 DLQ，发送采用异步调用。这条链路按事件推进明细和汇总，重复处理与部分成功还需要单独核对。",
              "核对中发现的具体问题": "当前版本查 Redis 后写双库再标记，存在并发和崩溃重复窗口；AgentUsage/AgentUsers 无 eventId 且重试生成新 ID；processMessage 同 Bean 自调用不触发默认代理事务；MySQL increment 0 行只 warning；调用坏 JSON 会中断批次且 Listener 正常返回；DLQ Future 未检查。以上为仓库静态核查，未声称已发生生产事故或已上线修复。"
            }
          }
        },
        "rca_investigation_agent": {
          "主题": "面向异常事件的只读取证、预算受控 RCA 调查循环",
          "参考回答": "RCA Agent 接收一次异常事件后，Planner 先生成候选根因和可验证 Probe，再按信息价值选择下一项只读工具查询。工具结果统一写入 Evidence Ledger，并更新各候选的支持证据、反证和置信度；证据满足完成条件时输出 RCA，预算耗尽、工具不可用或证据仍不足时转 ESCALATED，由人工继续调查。",
          "调查循环": [
            "Planner 根据当前异常上下文生成候选假设，以及能够区分这些假设的只读 Probe。",
            "调度器综合信息价值、查询成本、依赖关系和剩余预算选择下一项 Probe。",
            "Tool Gateway 按允许的 DAG 和权限边界执行查询，限制工具、参数、超时和最大调用次数。",
            "每份结果先进入 Evidence Ledger，记录来源并关联到对应假设，再更新支持、反证和置信度。",
            "证据达到完成条件时进入 COMPLETED；预算、时间或工具能力不足时进入 ESCALATED。"
          ],
          "冲突证据": "支持与反对同一假设的证据都保留在 Evidence Ledger 中，降低对应置信度，并优先选择能够区分冲突解释的新 Probe；结论同时保留支持与反对证据，单一日志作为候选线索继续核查。",
          "安全边界": "调查工具默认只读，并受允许列表、参数校验、超时、调用预算和 DAG 依赖约束；RCA 输出提供根因判断与证据关联，不直接执行破坏性修复动作。",
          "与LangGraph_Jenkins的分工": "LangGraph/Jenkins 是独立的部署审批和状态恢复链路，JenkinsClient 读取 Queue/Build JSON 状态。RCA 调查则通过只读 Probe 收集证据、更新候选假设，两条链路分别处理部署状态与异常根因。"
        },
        "maas_interview_extensions": {
          "主题": "从 TensorOpera Agent 平台迁移到 MaaS：职责边界、Token 计量、服务质量与资源治理",
          "参考回答": "我的 TP 经历可以对应 MaaS 的服务管理和用量治理：已实现 Agent Server 的 Kubernetes 生命周期与状态同步，以及 Kafka 驱动的调用明细和聚合统计，仓库还对接了 LiteLLM Team/预算配置。当前主要计量调用次数与开发者积分。如果扩展到 MaaS，我会增加可信 Token usage、逐次推理尝试、稳定事件身份、归属与价格快照，支持幂等明细、维度聚合和对账；在线 Gateway 单独实施限流与预算预占。推理侧则按 TTFT、ITL、排队、KV Cache 和资源容量管理服务质量、扩缩容与发布。Token 财务计费和推理引擎优化属于新增设计，不能当作这段 TP 实习已经完成的工作。",
          "实现边界": {
            "当前仓库可确认": "Agent Server Deployment/Service 生命周期、CPU/内存配额检查、Watch与周期List运行态同步、Kafka调用和生命周期消费、MongoDB明细与MySQL聚合、Redis快速去重、失败记录/DLQ尝试，以及LiteLLM Team/预算配置接入。",
            "当前不能据此宣称": "完整Tenant/Project/Model Token成本账本、严格实时预算、财务出账、多模型网关内部实现、推理引擎调优或TTFT驱动自动扩缩容已经在TP完成。已有代码的幂等与跨库失败窗口需按原项目问答解释。",
            "新问题的回答口径": "除明确指出的TP已有能力，其余统一用“如果扩展到MaaS，我会……”说明设计、故障窗口和验证方法。厂商字段、指标名、缓存与计价口径随接口和引擎版本核对，客户收费政策不能从技术事实直接推导。"
          },
          "术语速记": {
            "用量": "一次实际调用发生的输入、输出及其他计量数量。",
            "成本": "供应商采购或自托管资源消耗的成本，可能需要分摊。",
            "客户计费": "根据合同价格、折扣和业务政策产生客户应付金额。",
            "预算准入": "请求执行前限制可用额度；异步统计本身不能保证实时不超额。",
            "请求与尝试": "一个requestId可关联多个attemptId；同一事件重投复用eventId。"
          },
          "高压追问参考回答": {
            "Agent平台与MaaS是什么关系_你负责部署还是转发或模型推理": "当前 TP 项目管理的是 Agent 的开发、发布和运行环境；MaaS 把模型能力通过稳定 API 提供给用户，还要管理访问、路由、用量和服务质量。Agent 可以调用一个或多个 MaaS 服务，两者在服务生命周期、权限、额度和异步计量上有共通能力。我在 TP 负责 Agent Server 的 Kubernetes 生命周期与运行态同步，以及 Kafka 调用计量和开发者积分统计。仓库还有 LiteLLM Team 和预算配置接入；它不能证明我实现了 LiteLLM 内部网关、Token 成本账本或模型推理引擎。如果扩展到 MaaS，我会把现有部署和计量经验迁移到控制面与用量治理，再对接成熟推理引擎。",
            "MaaS与普通后端基础设施有什么不同_控制面与数据面如何划分": "如果扩展到 MaaS，我会把创建服务、版本发布、配额配置和状态同步放在控制面，把鉴权、模型路由、流式转发和推理执行放在在线数据面。控制面短时故障时，已经运行的服务应尽量沿用最近生效的配置；在线请求不应逐次依赖部署数据库。普通后端经验可以复用幂等、限流、Kafka 和对账，但模型服务新增了 GPU 和显存约束、模型冷启动、长连接、Prefill/Decode、KV Cache 以及按 Token 计量。同样一次请求的上下文与输出长度不同，算力、显存占用和执行时长也不同，所以只按QPS衡量容量会失真。",
            "如何把TP调用次数计量演进为MaaS的Token计量链路": "如果扩展到 MaaS，我会从 Gateway 或推理服务取得可信的实际 usage，生成稳定的计量事件，经可靠交接进入 Kafka；消费者先幂等保存逐次调用明细，再生成按租户、项目、模型和时间聚合的视图。当前 TP 已有 Kafka、明细和聚合框架，但 usage 主要是调用次数，revenue 在对应消费逻辑中是开发者积分，不能直接叫 Token 账单。新增 Token 计量时，我会分开保存用量、计价版本和计算结果，监控缺失事件、未知 usage、重放和对账差异。用量归集和成本核算是基础模块；实际出账、折扣、税费、收款和退款还需要独立业务规则。",
            "MaaS计量事件需要哪些字段_这些字段分别解决什么问题": "如果扩展到 MaaS，我会让事件至少包含 eventId、schemaVersion、eventType、eventTime、requestId、attemptId、tenantId、projectId、user或key标识、Agent标识、请求模型与实际模型版本、provider、deploymentId、状态及结束原因、输入输出和缓存 Token、usageSource、usage是否完整、priceVersion。eventId用于同一事件重投去重，requestId关联一次逻辑请求，attemptId区分每次实际上游调用；归属和模型字段用于核算和排障。租户身份从鉴权结果获得，不能直接相信用户提交的tenantId。计量主要需要数量与关联标识，完整Prompt等文本内容属于另一个按权限和保留策略管理的数据流。",
            "输入输出Token与缓存Token如何分类_怎么避免重复计数": "如果扩展到 MaaS，我会保留厂商原始usage，并由适配层转换为平台的标准分类，再按各分类的价格计算。不能把所有厂商的 input_tokens 当成同一种口径：例如有的接口缓存读Token是输入总量的子集，Claude接口则把普通输入、缓存创建和缓存读取分列。标准化后可以保存普通输入、缓存读取、缓存创建、输出以及必要的细分，但必须注明哪些字段互斥、哪些只是总量的子集。总量和分类不能再加一次。推理Token等细分项也要核对是否已经包含在输出Token里；字段缺失应记为未知而非一律为0。计量使用对应模型的Tokenizer或权威usage，不能用字符串长度换算。",
            "流式请求中途断开怎样确定实际用量_没收到最终usage是否记零": "如果扩展到 MaaS，我会分别记录客户端连接结束、上游是否真正停止以及usage是否确认。客户端断开不代表上游已经停止生成；能取消时把取消传到上游，并在服务端记录取消完成情况。若拿到推理引擎的最终计数或供应商结果，按权威记录保存；若只能拿到部分输出，则先记部分观测值和usage未完成，后续查询或对账补齐，估算值单独标记。流式API可能把usage放在最后一个chunk，中断后这个chunk可能收不到，因此不能把缺失usage当作零。SSE chunk可能含多个Token，也不能按chunk数计量。用量事实与客户收费政策分别处理，中断请求按预先确定的政策结算。",
            "requestId_attemptId_eventId有什么区别_重试时哪些保持不变": "如果扩展到 MaaS，我会把requestId用于一次逻辑调用，把attemptId用于该逻辑调用的一次实际推理尝试，把eventId用于一条确定的计量事件。Gateway超时后向另一个后端重试，通常仍关联同一requestId，但产生新的attemptId；同一条事件重新投递则必须复用eventId。不同attempt可能都真实消耗了Token，不能只按requestId去重把成本吞掉。外部客户端的业务幂等重试还需要独立的idempotencyKey及结果记录，不能假设它天然携带同一个requestId。同一次attempt先上报部分值、再上报最终值时，用版本更新或明确的增量/修正事件处理，不能把两个累计快照直接累加。",
            "模型价格变了历史成本怎么算_价格快照应该何时确定": "如果扩展到 MaaS，我会在计量记录中保存实际模型版本、provider、服务档位、币种、计量单位、价格版本及其有效区间，再保存具体计算结果。先由业务规则明确价格绑定请求受理时间、开始执行时间还是其他时点；异步消费和迟到回放都按这个约定找对应历史价格，不能用今天价格改写昨天成本。价格表要保留旧版本，折扣或后续修正以可追踪的调整记录表达。金额计算要统一单价计量单位和定点精度，用Decimal或最小货币单位累计，并明确按单条还是汇总后舍入。Token计量、供应商成本和客户售价可以共享数据基础，但属于不同计算口径。",
            "Tenant_Project_Model成本如何聚合_用户换项目后历史归属怎么办": "如果扩展到 MaaS，我会把调用发生时从鉴权得到的tenantId、projectId和用户/Key身份写入明细，同时保存请求模型别名与实际模型版本，按时间桶和这些维度幂等聚合。历史成本应读取事件里的归属快照，不能每次拿用户当前所属项目回填，否则转项目会改写历史。项目维度缺失时进入明确的未归属桶，并监控其比例；一个请求或attempt按规则归属一次，复杂共享成本另做分摊记录。完整归属放在数据库，在线监控指标只保留受控维度；eventId、requestId和大量用户ID不直接作为Prometheus标签，避免时序数量随请求增长。",
            "自托管GPU服务没有供应商Token单价_内部成本如何核算": "如果扩展到 MaaS，我会先明确这是实际资源成本、内部成本分摊还是对客户报价。自托管成本通常包含GPU运行时长、机器及存储网络等，不是简单套用外部模型每Token单价。可以先按模型部署和时间窗口归集资源成本，再按Token、执行时长或其他约定权重分摊到请求；固定预留容量、空闲资源和无法归属部分单独展示。并发批处理下请求共享GPU，不能把每个请求的端到端时长都当成独占GPU时长再相加。模型规模、精度、上下文、并行方式及利用率都会影响单Token实际成本；分摊公式属于核算规则，需要记录版本。",
            "Gateway为什么同时限制RPM_TPM和并发_长请求怎样做准入": "如果扩展到 MaaS，我会用RPM限制请求频率，用TPM约束Token吞吐，用并发限制同时执行的长请求，再限制上下文和最大输出长度。同样QPS下，长上下文或长输出可能持续占用更多算力与KV Cache，单靠请求计数不够。调用前估算输入与允许的输出，原子预占对应配额；结束后按真实usage修正，多实例共享一致的限流状态。租户、Key和模型层级都可以设置额度，拒绝时返回明确原因与可重试提示。请求没有max_tokens时，采用平台允许的输出上限或可解释的保守估计。不能假设输出恒定很短，多实例也不能各自拥有完整的租户额度。",
            "异步Kafka计量能直接保证实时预算不超额吗_怎样预占和释放": "如果扩展到 MaaS，我会把实时准入和异步账本分开：Gateway在开始请求前按价格及允许的最大用量原子预占预算；结束后以实际费用替换预占，释放未使用部分；Kafka负责可靠记录、聚合和对账。只检查已消费累计金额会有延迟窗口，多个并发请求可能一起通过并超额。预算预占记录需要状态、requestId和恢复流程，防止进程退出后永远占用；未确认上游终止时也不能只因TTL到了就释放，让仍在运行的请求再次突破预算。预算存储故障时，按业务要求选择拒绝、可核对的降级或受限宽限，并记录剩余风险。TP仓库接入LiteLLM预算配置不等于实现或验证了这套严格预算机制。",
            "Gateway返回成功后进程挂了_UsageEvent怎样避免漏记": "如果扩展到 MaaS，我会先确定事件产生位置和可恢复的数据来源。自有推理服务可以先持久化执行结果和usage，再经outbox或可恢复发布器交接Kafka；外部模型调用则尽量保存调用尝试、上游requestId和获得的usage，并通过供应商查询或对账补缺。Kafka生产确认与HTTP返回不是一个原子事务，不能承诺进程在任意时刻退出都天然没有漏记。在线链路选择等待持久化确认，会增加延迟；异步缓冲则要接受并量化恢复风险。消费端持久化eventId唯一约束或等价幂等键，聚合可从明细重建。Outbox只解决本地记录到消息的可靠发布，不能把外部推理副作用纳入本地事务；供应商调用结果仍需要单独记录、查询或对账。",
            "QPS不高为什么TTFT仍然很慢_先从哪里定位": "如果扩展到 MaaS，我会先用requestId把网关、路由和推理服务的耗时连起来，区分鉴权/网络、队列等待、模型冷启动、Prefill以及首次输出的耗时。QPS只反映请求次数，少量超长Prompt也可能需要很重的Prefill；GPU容量小、路由倾斜或大量长输出占着KV Cache时，也可能排队。先看按模型和输入长度分组的P95/P99 TTFT与排队分布，再看running/waiting、吞吐、缓存命中和GPU状态。低GPU利用率也可能来自网络、CPU预处理、跨卡通信或引擎阻塞，GPU可能正在等数据。根据分段证据定位，再选择补副本、改路由或调引擎配置。",
            "TTFT_ITL_排队时间_KVCache分别观测什么_怎么避免混淆": "如果扩展到 MaaS，我会约定测量位置：客户端或Gateway的TTFT是从请求开始到首次输出，包含它们看到的网络和排队；引擎TTFT可能有不同起止点。严格逐Token的ITL是相邻生成Token的时间间隔。TPOT常见口径是同一测量位置的（请求总耗时−TTFT）÷（输出Token数−1），表示首Token之后的平均耗时；输出Token数≤1时这个口径没有定义，需要约定为缺失或其他明确值。平均值可能掩盖中途卡顿。排队时长和waiting请求数反映等待执行的压力，KV Cache占用反映推理上下文状态占用了多少缓存容量，它不等于整张卡全部显存占用。观测时同时保留请求长度、输出长度、成功/取消状态，并看延迟分位数。一个chunk含多个Token时，输出事件间隔与严格逐Token间隔不完全相同，指标需遵循引擎版本定义。",
            "MaaS应该按CPU_GPU利用率还是队列扩容_多指标如何落地": "如果扩展到 MaaS，我会按模型分别压测容量，用TTFT/ITL的SLO判断服务质量，用队列长度、队列增长、运行并发和Token吞吐判断负载，再结合GPU、KV Cache和可用GPU容量。CPU可用于网关等模块，GPU利用率用于辅助判断推理负载；都不直接等同剩余服务能力。扩容可以用校准后的队列或并发指标驱动HPA或专用控制器，同时设置最小热副本、扩容速率、缩容稳定窗口和最大容量。HPA配置多个指标时选择它们建议副本数的最大值，指标需要相应适配接口。KV占用高时，先分清活跃请求缓存压力与可回收缓存；路由倾斜、单请求过长或外部依赖慢也不能仅靠加副本解决。",
            "模型冷启动很慢怎么应对突发流量_Pod启动后何时接流量": "如果扩展到 MaaS，我会拆开镜像拉取、权重下载、GPU加载、引擎初始化和预热，先找最长阶段，再采用镜像与权重预缓存、节点本地缓存、最小热副本及按趋势提前扩容。Pod Running仅说明容器在运行，模型加载和探测通过后才能标为Ready。startupProbe给初始化过程足够时间，readinessProbe表达是否可接流量，livenessProbe检测需要重启的失活，避免把合法慢启动当成故障。突发流量超过热容量时，要配合有上限的排队、准入拒绝或业务允许的其他模型路由；新增Pod没有可用GPU或模型还没Ready时，不能把它计入在线容量。",
            "GPU推理请求失败怎么区分业务问题_显存OOM和节点故障怎么处理": "如果扩展到 MaaS，我会先看是否与输入长度、并发和特定模型相关，核对容器日志、推理错误、GPU温度/ECC/驱动错误、Node Condition以及同节点其他服务。CUDA显存OOM不自动代表GPU损坏；可能需要限制上下文或并发、调整KV Cache预算、分批和模型配置。模型加载失败、镜像拉取失败和应用异常也分别处理。确认设备或节点故障时，先停止把新请求路由到异常实例，按业务策略重试或降级，再由节点自愈链路隔离和维修。容器重启通常仍在原Pod与原设备上，驱动或硬件故障可能反复；是否迁移、GPU重置或整机维修应依据故障类型。",
            "推理超时是否可以直接重试_流式已返回一半还能换模型吗": "如果扩展到 MaaS，我会区分尚未输出与已输出的请求。还未向客户端输出时，可在限定尝试次数和总超时内重试，同时给新上游调用新的attemptId，并尽量取消旧尝试；旧上游可能已执行，所以要保留其用量。已经输出一部分时，透明切换可能拼接出不一致内容，也可能重复执行工具，通常应终止当前流并明确失败，让协议或业务决定如何续接。切换到其他模型前要检查用户允许的模型、上下文和工具能力，降级不能悄悄改变能力约定。工具副作用需要各自的业务幂等机制，requestId只能关联链路，不能保证工具操作不重复。",
            "模型升级如何灰度发布_回滚和成本追踪怎么关联版本": "如果扩展到 MaaS，我会把对外模型别名与实际模型版本、权重和引擎配置区分开，用受控比例将请求路由到新版本，并按版本比较成功率、TTFT/ITL、输出质量及资源占用。新版本先加载预热并Ready，再逐步加流量；旧版本停止接新请求后等待在途流式请求结束，不能只改镜像就宣布完成。回滚时停止新版本新增流量并恢复路由，已有请求按超时和排空规则处理。计量明细保存实际执行版本和对应价格版本，才能解释升级前后的成本变化。Kubernetes滚动更新主要管理副本和可用性，模型行为仍要通过业务评测与线上观测确认。",
            "MaaS资源配额和Token预算有什么区别_多租户如何隔离": "如果扩展到 MaaS，我会分别管理部署资源额度和调用额度：CPU/内存/GPU额度控制可部署的资源量，Token/RPM/并发和金额预算控制在线调用消耗，两者约束不同。自托管整卡GPU分配通常不自动限制每个请求的显存，MIG等分区能力、时间共享或引擎级并发控制要按硬件和方案另行配置。租户身份通过鉴权绑定，运行环境结合namespace、RBAC、ResourceQuota、网络策略、Secret权限和Pod安全配置；共享推理服务再实施公平排队和并发隔离。Namespace提供资源组织和权限范围，多租户隔离仍需这些机制共同生效，高敏感或不可信镜像可采用更强的独立运行边界。"
          }
        }
      },
      "resume_bullets": [
        {
          "bullet_title": "Agent Server 执行环境与生命周期管理",
          "ownership": "个人实现",
          "业务问题": "面向多账号 Framework Agent 的容器化运行需求，建设标准化的 Agent Server 执行环境与生命周期管理能力。",
          "简历原文口径": "基于 Kubernetes Deployment / Service / Secret 编排镜像、启动参数、环境变量、CPU/内存、副本及可选 Readiness/Liveness 探针；通过 Watch + 周期 List 回写实际副本、可用副本和摘要状态，按业务归属清理枚举到的孤儿 Deployment/Service，支持创建、更新、副本调整及基础 Pod 日志查询。",
          "面试展开重点": "先讲用户镜像到 Service 可用后端的链路，再按追问说明创建部分成功、超时不确定、稳定命名的幂等边界、Watch/List 与删除事件、探针和滚动更新、账号额度并发窗口。优雅缩容、持久创建任务和加强隔离作为设计演进。"
        },
        {
          "bullet_title": "Agent 调用计量与资源治理链路",
          "ownership": "个人实现",
          "业务问题": "面向 Agent 调用及发布生命周期事件的异步计量需求，建设多账号 Agent 的调用计量与资源治理链路。",
          "简历推荐写法": "基于 Kafka 异步消费调用与发布事件，以 MongoDB 保存调用、用户和积分明细，MySQL 维护聚合统计；使用 Redis eventId 抑制短期重复消息，并实现有限重试、失败记录和死信投递，为资源配额与成本分析提供调用计量数据。",
          "面试展开重点": "先讲调用事件的实际落库与聚合，再解释 Redis 去重窗口、双库部分成功、CTM 自调用、offset 配置来源、坏消息和 DLQ 交接、明细不具备唯一事件键时的对账限制。持久账本、幂等投影和 Token 成本归集作为设计演进。"
        }
      ]
    },
    {
      "company": "硕士毕业论文",
      "role": "硕士研究课题（进行中）",
      "period": "2026",
      "focus": "MoE 在线推理的 KV 容量感知准入与服务性能权衡：从保守最大长度预留，推进到已有未来峰值准入在原生异步运行时中的适配，研究更早入场怎样影响完整请求服务。",
      "resume_positioning": {
        "简历当前标题": "MoE 在线推理的 KV 容量感知准入与服务性能权衡",
        "当前内容概括": "在原生 vLLM 0.26 上适配已有 MC-Benchmark 未来峰值准入，处理页粒度、异步在途输出与物理释放时机，建立从实际准入动作到完整请求延迟、吞吐和输出差异的证据链。",
        "参考回答": "我研究单卡 MoE 推理中，怎样在有限 KV 容量下接纳请求。我先比较了固定并发和按最大长度预留，随后把已有的未来峰值准入方法适配到 vLLM 异步调度，利用旧请求完成后释放的空间提前接纳新请求。难点是分清已经调度、输出已经返回和物理页真正可复用这几个时刻。一组四次运行的开发对照中，相对全额预留基线，两次候选的平均首 token 等待分别下降约 8.8% 和 15.7%，同时生成速度变慢，说明容量利用和服务体验要一起评价。",
        "研究介绍参考": "新请求入场后，KV 会随着生成增长；按每个请求的最大长度同时预留能控制容量，却可能让本来可安全入场的请求继续排队。已有 MC-Benchmark 按未来轮次检查占用峰值，把旧请求完成后的释放算进去。我在 vLLM 0.26 的单卡异步路径上做了组件适配，按真实已计算和本轮已调度的位置计页，并保守处理在途输出和延迟释放。固定 GPU KV 为 64 GiB、同一引擎按全额预算/MC/MC/全额预算比较，每次 384 条请求全部完成且没有抢占；两次 MC 分别实际放宽 166、167 条请求，平均 TTFT 下降 8.80%、15.70%，平均 TPOT 则由约 45ms 升到约 48ms。未来峰值原则属于已有工作，我完成的是运行时适配与服务代价测量，当前还没有成立的新算法贡献。",
        "结果范围": "当前主例为 2026-10-09 的 C v25/r02：OLMoE-1B-7B-0924/BF16、单张 RTX PRO 6000、vLLM 0.26 原生 async 深度2、GPU KV 64 GiB/host KV 16 GiB，单引擎开环请求实验。一个共享 engine 的开发 ABBA 块，输出质量未测；不代表 HTTP 全链路、多卡专家并行或独立确认。早期 RTX 5090 同步实验另作历史案例。"
      },
      "project_background": {
        "问题": "新请求的 prefill 与已有请求的 decode 共享计算和 KV 容量。只看当前空闲页可能忽略后续增长，把所有请求的最大需求同时预留又可能增加排队。旧请求会完成并释放容量，能否利用这段时间关系，需要结合真实执行与物理释放边界判断。",
        "目标": "在相同资源和外部请求条件下，评价准入对首 token 等待、生成流畅度、完整完成时间及服务吞吐的影响，并记录受益和受损请求。当前没有可靠应用 SLO，保留探索性取舍。",
        "MoE联系": "当前以 OLMoE 为实验模型；KV 增长、异步记账和准入取舍也存在于 Dense 推理。当前证据没有证明专家负载感知的独立价值，不能将本例称为 MoE 特有算法。专家路由只有在解释了普通指标无法解释、且可通过动作改善的损失后，才值得作为新信号加入。"
      },
      "current_primary": {
        "name": "未来释放感知的 KV 准入：从容量可行到完整服务取舍",
        "更新日期": "2026-10-10；当前主例的实验与判决日期为2026-10-09",
        "status": "C v25/r02 已完成、分析并归档。已有 MC 组件适配获得有代价的开发支持；未发现正确近邻之后的重要执行失信，当前新增信用控制器尝试已收束，原创论文贡献尚未成立。",
        "个人实现": "在原生首次 prefill 分配入口实现完整声明预算门控，并复用 MC 未来峰值检查；对异步路径核对真实驻留页、computed+本轮scheduled 的页起点、placeholder/在途批次及退休边界。守卫不满足时保持普通预算判定，只有原生容量与并发允许、未来峰值也可行时才放宽。关联记录放宽、成功分配、确认准入、首prefill与逐请求结果。",
        "复用能力": "MC-Benchmark 的未来轮次 KV 峰值原则；vLLM 的 KV 分配、FCFS运行顺序、Attention/MoE kernel、异步执行和原生抢占恢复。本地实现是有条件的组件适配，没有完整复现 MC-SF 的排序与理论系统。",
        "关键认识": "声明的最大占用之和、某一时刻的实际占用、未来轮次占用峰值是三个量。未来峰值可以比最大需求之和小，但更早接纳会改变合批执行和服务时间；容量可行并不自动保证完整请求更快。",
        "下一步": "作为待评估的演进线索，先明确真实服务更重视首响应、生成流畅度还是整请求完成，再判断正确 MC 对照之后是否仍存在可控制的重要损失。值得检查的是容量允许时，当前执行负载是否提供了有价值的延后准入信息，并与调好的固定并发、全额预算和已有 MC 比较。只有出现简单指标不能解释的决策差异，才考虑专家信号或新的联合策略。",
        "下一步状态": "上述演进是研究建议，未启动新实验或控制器。旧5090的90%/95%显存对照不再作为当前待办；历史记录保持原有证据范围。",
        "判定": "已有未来容量原则已能利用真实机会；当前剩余 TTFT/TPOT/flow 取舍尚未被证明对应重要应用损失。后续投入取决于具体目标、强简单基线之后的残差和可改变它的动作，不能把适配成功或更多准入次数直接当成创新。"
      },
      "experiment_facts": [
        {
          "实验": "C v25/r02：已有未来峰值准入的原生异步组件适配",
          "阶段": "当前主例；2026-10-09完成，2026-10-10知识库核对",
          "设置": "OLMoE-1B-7B-0924 BF16、单张PRO 6000、vLLM0.26、async深度2、FCFS、token量子1024、完整在途cap256；固定GPU KV64GiB=32768页、host KV16GiB。384篇文章按0.1s间隔到达，允许EOS、输出上限1024。共享engine按F/MC/MC/F执行，F为物理已用加未分配声明增长的全额预算，配对为01/00与02/03。",
          "实现": "普通预算拒绝后，先检查旧请求完整驻留且每轮单token decode、在途批次和页状态一致；用computed+本轮scheduled计占用起点，以声明剩余输出加两个非空调度轮次保守延后释放；未归还的非live物理页继续计费。通过未来峰值检查才允许原生首次prefill分配，不改变旧请求、victim或恢复规则。",
          "动作结果": "四次均384/384完成，共1536次请求执行，失败、拒绝、超时、未完成和抢占均0。两次MC的实际放宽、成功分配、确认准入和首prefill对应166/167个请求；active观测峰180→217，物理已用页峰28520/28538→32737/32736，未超过32768页。放宽指原本在该时刻被F拦住的请求更早入场，不是多完成166/167条请求。",
          "服务结果": "MC相对各自F：平均TTFT−8.80%/−15.70%，输出token吞吐+1.57%/+2.35%，到达后完成排空−1.38%/−4.27%；平均flow+2.89%/−0.35%，TTFT P95+19.15%/−5.70%，每请求最大gap的P95−3.27%/+13.95%。平均TPOT由44.904/45.174ms升至47.657/47.661ms。",
          "成本与输出": "F控制wall为0.832/0.854s，MC为1.879/1.788s，已经计入完整服务时间。输出总量配对变化+0.635%/−0.598%，序列不同185/194条、长度不同9/7条；各臂95.8%—97.4%达到输出上限，自然任务质量未测。",
          "解释": "一个开发ABBA块支持已有MC适配能真实利用未来容量，并呈现更早准入与生成速度的取舍。结果不是等输出工作量加速、统计显著性证明、全域安全证明或原创算法收益。两个F也无抢占，不能把本组描述为MC消除了抢占。"
        },
        {
          "实验": "并发上限与延迟反馈",
          "设置": "短输入/输出各 128 token；平稳与突发到达分开；同引擎静态 cap 8/12/16/24/32。两轮共 64 个正式 episode、2048 次请求执行，复用同一 32 条源文本。",
          "结果": "突发场景的新反馈优于旧规则，但 4 组都低于同期最好静态点；平稳场景仅一次胜出，重复未保持。",
          "解释": "同期最好静态点仍然领先。部分首次降 cap 时 waiting 已为 0，动作前也可能已有 batch 分叉，分析时要分别看动作是否有可调请求和动作前轨迹差异。",
          "阶段": "历史：RTX 5090 同步请求级基础实验，与当前 PRO 6000 异步组分开解释"
        },
        {
          "实验": "Prefill token 预算",
          "设置": "固定 cap 8 和编译容量 1024；引擎排空后切换预算。16 条请求，输入 128/2048 混合、固定输出 128；全短输入作无动作差异对照。",
          "结果": "256 相对 1024：平均 TTFT 增加 32.06%/34.56%，整批时间增加 9.05%/9.14%。512 相对 1024：ITL p99 下降 29.9%/29.5%，TTFT 增加 6.36%/7.80%，整批时间增加 2.12%/2.38%。",
          "解释": "较小预算平滑了部分生成间隔，但长 prompt 需要更多次推进，首 token 和完成时间付出了代价；探索到 512 后停止继续扫描。",
          "阶段": "历史：RTX 5090 同步请求级基础实验，与当前 PRO 6000 异步组分开解释"
        },
        {
          "实验": "短 prompt 优先",
          "设置": "固定 cap 8、token 预算 1024；仅对已到达、从未执行的 WAITING 请求按输入长度稳定排序。FCFS 对照承担相同队列操作和采集开销。",
          "结果": "3 个完整回传 block：短请求 TTFT 下降 14.06%/10.38%/9.45%，长请求增加 3.09%/6.44%/5.19%；整体平均请求延迟仅下降 2.52%/0.22%/0.03%，长请求完成尾部增加。",
          "解释": "排序把收益集中到短请求，三个 block 的整体平均延迟改善从 2.52% 降到接近零。当前实现只重排等待队列，持续到达场景还需补防饥饿机制；第 4 个反向重复的原始数据未回传，结果使用前 3 个完整 block。",
          "阶段": "历史：RTX 5090 同步请求级基础实验，与当前 PRO 6000 异步组分开解释"
        },
        {
          "实验": "保守 KV 容量上限",
          "设置": "32 篇自然文章，3072 输入/1024 输出、50ms 到达间隔。该次 live KV 池 7677 块，每块 16 token；每请求最多 256 块，保守 cap=floor(7677/256)=29。",
          "结果": "有效 cap 29 相对 cap 16：吞吐增加 3.75%/4.53%，TTFT p99 增加约 4.45/4.28 秒，请求平均 TPOT 的中位数增加约 5.10/4.96ms。",
          "解释": "cap 29 是该布局下的保守容量边界，不是性能最优值。早期 cap 32 在 decode 增长触边后被守卫中止，随后单独补跑完整原生恢复基线。",
          "阶段": "历史：RTX 5090 同步请求级基础实验，与当前 PRO 6000 异步组分开解释"
        },
        {
          "实验": "完整原生抢占恢复基线",
          "设置": "同一长上下文负载、显存预算 0.90；四个新引擎按 native32→safe29→safe29→native32 执行，全部正式请求与预热记录已回传。",
          "结果": "native32 较 safe29 吞吐高 16.99%/17.08%；TTFT p99 约 0.76/0.80 秒，对照约 17.90/18.11 秒。每轮两条受抢占请求暂停约 2 秒和 4.5 秒；safe29 最大 ITL 约 0.108 秒。native32 的请求平均 TPOT 中位数也更差。",
          "解释": "零抢占会付出准入等待和吞吐代价。一次 4.47 秒间隔中，约 4.36 秒在首次重算调用之前，约 0.116 秒是包含其他工作与采集的重算调用首尾跨度。",
          "阶段": "历史：RTX 5090 同步请求级基础实验，与当前 PRO 6000 异步组分开解释"
        },
        {
          "实验": "同并发、不同 KV 资源预算",
          "设置": "固定 cap 32，比较 0.95 与 0.90。当前单组布局下，32 条声明最大长度同时驻留需 32×256=8192 块；每次读取真实块数。",
          "结果": "历史记录中首次尝试在模型加载前因 GPU 资源冲突退出，执行请求数为0；当时已准备重试包。本次未追认后续结果，不作为当前待运行任务。",
          "解释": "8192 块是该布局下同时保留全部最大长度的充分条件，不是请求完成的必要条件。增加显存预算与同资源算法优化分别评价。",
          "阶段": "历史：RTX 5090 同步请求级基础实验，与当前 PRO 6000 异步组分开解释"
        }
      ],
      "evaluation_method": {
        "时钟与延迟": "使用同一 host 时钟。TTFT=首 token 返回−计划到达；请求延迟=完成−计划到达；平均 TPOT=(末 token−首 token)/(输出 token 数−1)，仅在输出超过 1 个 token 时定义。queue_s 只记录提交滞后，原生队列等待另看首次调度记录。",
        "吞吐与达标": "区分输出token吞吐与完成请求吞吐，分母包含等待及约定的后台排空。当前C v25的+1.57%/+2.35%是输出token/秒；历史native32对safe29的约17%是请求/秒。goodput计完整观察区间内完成且满足事前阈值的请求，全部到达、失败、拒绝、超时和未完成保留。",
        "尾延迟": "同时看 TTFT、每请求平均 TPOT、所有 token 间隔的分布和每请求最大 ITL。当前小样本 p99 接近最慢请求，用来描述这组实验中的尾部行为；生产尾延迟需要更大的独立请求样本。",
        "SLO": "每组使用自己的冻结口径，不跨组拼接。当前C v25没有可靠应用SLO，保留TTFT[2,5,10,20,40]s、flow≤120s及maxgap[0.25,0.5,1,2]s的探索性20格；四档gap均不限制达标，所以20格不是20项独立成功。早期5090的200ms/9ms或5s/200ms仅是对应组参考阈值。",
        "对照": "固定模型、精度、外部到达、输入、声明输出上限与资源；策略真实推进自身队列、KV和生成。当前组共享engine，按F/MC/MC/F运行并固定01/00、02/03配对，控制wall保留在完整时间中。策略可以产生不同内部轨迹，不能把逐请求差值当成相同前态的因果干预。",
        "基线与重复": "比较强简单方案与正确适配的最近邻。当前主例只有一个开发ABBA块；1536次请求执行不是1536个独立系统重复，也没有独立负载确认。历史最好静态点用于诊断，不能当已部署的无代价oracle。",
        "输出与质量": "当前组允许自然EOS，但绝大多数达到1024输出上限。报告输出总量、长度、序列和终止原因差异；尚未评估自然任务质量，不能宣称相同计算量或质量等价。"
      },
      "high_pressure_answers": {
        "题目写专家负载感知_现在实际做到哪一步": "现在已经完成已有未来峰值准入方法在 vLLM 异步调度中的组件适配，并做了完整请求对照。两次候选平均 TTFT 分别下降约8.8%和15.7%，同时平均 TPOT 变差。专家负载还没有形成独立增量证据，因此当前题目定位为 KV 容量感知准入与服务性能权衡，原来的专家负载感知是研究设想。",
        "你自己写了什么_哪些复用vLLM": "我实现了实验侧准入门控、已有 MC 峰值检查的页粒度和异步适配，以及请求级采集分析。适配会核对已经计算和本轮已调度的 token 位置、真实页数、在途批次及释放边界，条件不满足就保持普通预算判定。KV 分配、模型 kernel、异步执行和抢占恢复复用 vLLM，未来容量原则来自已有 MC-Benchmark。",
        "为什么不直接做一个复杂的反馈控制器": "已有方法已经能实现一部分准入收益。当前 MC 适配让更多请求提前入场，同时提高了平均 TPOT，并增加部分请求的等待代价；还没有证明这些代价需要一个新控制器才能解决。我会先明确服务目标，并与调好的固定并发和正确的近邻比较，找到有价值的决策差异后再增加状态和规则。",
        "这些问题Dense模型也有_为什么研究MoE": "当前主例使用 OLMoE，但 KV 增长、准入和异步释放也存在于 Dense 模型。当前结果支持的是这个 MoE 运行域中的容量与服务取舍，没有证明 MoE 特有贡献。专家路由要能提供普通长度、并发和 KV 信息之外的有效决策，才有依据把方法称为专家感知。",
        "当前创新点和收益分别是什么": "当前已经成立的是运行时组件适配和完整服务取舍的测量，新算法创新尚未成立。已有 MC 方法经适配后，相对全额预算两次配对的平均 TTFT 降低8.80%和15.70%，输出token吞吐提高1.57%和2.35%；平均 TPOT 上升，尾部和完成时间也没有一致改善。未来峰值原则属于已有研究，我的工作是把它接到真实异步运行时并测清收益与代价。",
        "吞吐提高17%_能写成你的算法加速吗": "这 17% 是原生 cap 32 相对保守 cap 29 的配置对照结果。两组都复用 vLLM；前者完成更快，但少数请求承担秒级暂停。我的工作是补全原生恢复基线并测清这种权衡。",
        "下一步具体做什么": "先确定实际服务对首响应、生成流畅度和完成时间的要求，再判断已有 MC 之后是否仍有重要、可通过准入改善的损失。一个待验证线索是：KV 容量允许接纳时，当前执行负载是否仍提示应该暂缓入场，并且这种判断能否超过调好的固定并发。这个方向尚未启动，也没有预先认定需要专家信号。",
        "论文上线了吗_单卡结果能说明什么": "目前是单卡研究实验，当前主例来自 PRO 6000 上原生 vLLM 的异步引擎请求对照。它能说明这个配置下的准入动作、容量使用和服务取舍；论文中心贡献尚未成立，也没有生产上线结果。HTTP 全链路和多卡专家通信需要各自的证据。",
        "一个推理请求从用户到返回经过什么_你改哪一层": "典型服务经过网关、实例选择、分词、引擎等待、prefill、逐步decode和结果返回。当前改动在引擎的首次prefill分配入口，决定队首新请求是否可以入场，模型执行与旧请求调度复用原生逻辑。TTFT和完成时间从外部计划到达开始计算，包含引擎外等待和提交滞后；测量仍是in-process实验。",
        "prefill和decode有什么区别_为什么会相互影响": "Prefill 处理输入序列、建立 KV，并提供生成首个新 token 所需的输出；后续 decode 逐步生成新 token。长 prefill 一次处理的 token 多，和 decode 共用 GPU 时可能拉长生成间隔；把它拆小又会延后长请求的首 token，预算需要兼顾两端。",
        "KV缓存到底存什么_为什么不缓存Q": "KV 保存各层 Attention 中历史 token 的 Key 和 Value，让新 token 能直接读取历史上下文。每步的新 Query 用来查询这些历史 K/V；历史 Query 不参与这次查询，所以常规自回归 decode 不需要把它们长期保留。",
        "continuous_batching和chunked_prefill是一回事吗": "不是。Continuous batching 在迭代边界移除完成请求、接纳新请求；chunked prefill 把一个长输入拆成多步计算。两者可以一起使用，前者决定哪些请求参与，后者控制长输入每步推进多少。",
        "MoE的一层怎么计算_专家是什么": "常见 MoE 用多个前馈网络作为专家。Router 为每个 token 选少数专家，dispatch 把 token 按专家分组，专家计算后再按路由权重 combine 回原 token。多卡专家并行时，分发和汇聚还可能跨设备通信。",
        "只激活少数专家_为什么模型仍然很占显存": "每个 token 只计算少数专家，降低的是当次计算量。在全部权重驻留的部署里，未被这个 token 选中的专家仍占显存；不同请求还可能使用不同专家。专家权重与动态增长的 KV 共同争用显存。",
        "专家负载具体看什么_卡利用率不够吗": "先看每层各专家收到的 token 数、最大值相对均值、实际分组形状和执行时间；多卡还要看承载设备及通信量。GPU 利用率只有整体忙闲程度，同样很忙可能分别受计算、KV 读取或通信限制，不能直接当专家拥塞。",
        "专家负载不均_一定会拖慢请求吗": "不一定，还要看负载如何映射到设备和 kernel，以及最慢部分是否阻塞后续步骤。单卡分组执行与多卡等待最慢设备的机制不同；要把专家负载与同一步执行耗时、请求推进关联起来，才能判断它是不是瓶颈。",
        "TP_EP和多副本有什么区别_当前用了哪个": "TP 把同一层的张量计算拆到多张卡，EP 把不同专家放到不同卡并按路由分发 token；独立模型副本则各自服务请求。它们可以组合，分别引入通信、状态和容量成本。当前原生实验使用单张 GPU 和单个引擎。",
        "cap_token预算_KV容量分别限制什么": "cap 限制运行请求数，token 预算限制本步安排的 token 总量，KV 容量限制可保留的上下文。例如 cap 8、已有 6 条普通 decode，每条推进 1 token 后剩余预算可用于 prefill，但新接纳还受剩余请求名额和 KV 约束。",
        "把cap从32降到16_已有请求会被停掉吗": "我的准入限制只阻止新请求加入，已有请求继续推进。如果还运行着 24 条，就等它们自然完成，降到 16 以下再补新请求。降 cap 本身不会截断 running 列表；KV 压力下的原生抢占仍按引擎机制处理。",
        "怎么证明参数真的起作用_反馈为什么会来不及": "看真实 running、waiting、每步 token 数和首次阻止新接纳的时刻。有一次第一次降 cap 时 waiting 为 0，当下没有请求可被拦住；已经运行的请求及 KV 又不会随目标值下降立刻消失，所以写入参数和缓解拥塞之间可能有明显延迟。",
        "新反馈优于旧反馈_为什么还不算赢": "因为同期最好静态配置仍然更好。突发场景中新反馈优于旧规则，但四组都低于同引擎最好静态点；平稳场景的一次胜出也没有重复出来。当前选型应以强静态基线为参照，再解决反馈的触发时机和实际动作效果。",
        "为什么要同引擎比较_并发变小不就更快吗": "逻辑请求数变小不保证实际执行成本同比下降，编译、图捕获档位和 padding 都可能改变执行形状。我比较 cap 或 token 预算时尽量保持同一引擎与 KV 池，避免把初始化配置差异混入策略收益。",
        "小prefill预算的实测结果怎样": "固定 cap 8 时，256 相对 1024 预算使 TTFT 增加约 32%–35%、整批时间增加约 9%。512 虽让 ITL p99 下降约 30%，仍增加 TTFT 约 6%–8%和完成时间约 2%。它保护了部分生成间隔，但没有带来这组负载的整体加速。",
        "短prompt优先具体改哪里_效果怎样": "只重排已到达、从未执行的 WAITING 请求，按输入长度稳定排序；已经开始的 prefill 和 decode 保持原生处理。三个完整结果中，短请求 TTFT 降约 9%–14%，长请求增加约 3%–6%，整体平均延迟改善从约 2.52% 降到接近零。",
        "短prompt优先就是SJF吗_长请求会不会饿死": "它只是用输入长度近似成本，真实总服务时间还取决于生成长度和合批执行，因此不等于已知总时长的 SJF。当前纯排序没有防饥饿保证；持续短请求到达时，在线方案需要最大等待、老化或有限排序窗口。",
        "prefill都完成了_为什么decode还会缺KV": "Prefill 只建立输入部分的 KV，后续每生成 token 还会增长。本次 32 条长请求就是在 prefill 完成后逐渐触到容量边界；只检查刚入场时有多少空闲块，会漏掉后续增长。",
        "KV显存怎么估算": "对常规等形状的全注意力层，可近似按 2×层数×KV 头数×每头维度×元素字节数×缓存 token 数计算，2 代表 K 和 V。GQA/MQA 应使用 KV 头数；实际还要计入块对齐、缓存共享和运行预留，多卡按本地存储布局计算。我的容量实验直接读取引擎实际可用块。",
        "cap29怎么来的_为什么不是32": "该次实际 KV 池有 7677 块，每块 16 token。每条请求最大 3072 输入加 1024 输出，需要 256 块，因此 floor(7677/256)=29；32 条同时保留最大长度需要 8192 块。这个计算限定于该单组布局、无前缀共享及额外 lookahead 的配置。",
        "线上不知道输出长度_按什么预留": "当前用请求声明的最大输出数作为上限，不需要事先知道实际EOS位置。全额预留把每条请求的最终最大需求同时算入；未来峰值检查则在旧请求持续推进的条件下，把它达到上限后可释放的容量算进去，并保守处理异步延迟。这里没有用事后真实输出长度指导在线准入。",
        "PagedAttention已经解决碎片_为什么还要限制准入": "分页让 KV 不必占连续的大块地址，并减少存储浪费，但总物理容量仍有限。所有活跃请求的上下文继续增长时，仍可能放不下；我的准入限制处理的是这类总容量与等待问题。",
        "KV不足时抢占什么_是杀掉Pod吗": "本实验中的原生抢占暂停的是引擎内的请求，回收其 KV 供其他请求使用，后续再通过重算恢复上下文。模型服务进程和 Pod 继续运行，已返回的 token 也不会撤回；用户感受到的是后续生成暂停。",
        "零抢占不是更好吗_为什么cap32反而吞吐更高": "保守 cap 29 避免了抢占，却让末尾三条请求更晚入场，TTFT 接近 18 秒。cap 32 允许原生恢复，整批吞吐约高 17%，但两条请求暂停约 2 秒和 4.5 秒。选择取决于业务更能接受首 token 前等待还是生成中停顿。",
        "4_5秒是不是全在GPU重算": "不是。一次最长间隔约 4.47 秒，其中约 4.36 秒发生在首个重算调用之前，重算调用首尾跨度约 0.116 秒。后者还包含其他请求工作与采集，需要另外测 kernel 时间才能得到纯 GPU 重算成本。",
        "提高gpu_memory_utilization是不是就解决了_为什么不直接设满": "提高它会改变可用资源预算，实际KV还取决于权重和运行开销，必须读取真实块数。当前主例直接固定GPU KV为64GiB，两种方法使用相同资源，考察已有空间怎样被不同准入规则利用。旧5090的90%/95%对照属于历史计划，不能拿它替代当前同资源的结果。",
        "TTFT_TPOT_ITL分别是什么_排队算进去了吗": "TTFT 是到达至首 token 的时间；ITL 是相邻 token 的返回间隔；平均 TPOT 是首末 token 时间差除以输出数减一。我的到达、提交和返回使用同一 host 时钟，TTFT 包含提交滞后与引擎等待；queue_s 这个字段只代表提交滞后。",
        "平均TPOT和token_p99正常_为什么用户还卡几秒": "平均值会摊薄停顿，把所有 token 间隔合在一起的 p99 也可能漏掉少数极端请求。cap 32 的 pooled-token ITL p99 约 28ms，但仍有请求暂停约 4.5 秒，所以还要单列每请求最大 ITL 和受抢占请求。",
        "吞吐_goodput_SLO达标率怎么区分": "吞吐看每秒完成多少请求或输出多少 token，先说清单位；请求 goodput 只计入同时满足预定延迟要求的完整请求。达标率看全部已到达请求里有多少达标，失败、拒绝和未完成要保留，避免只留下容易完成的请求。",
        "SLO怎么定_换阈值会不会换一个赢家": "换阈值可能改变赢家，所以当前没有把探索阈值称为客户SLA。C v25保留事前固定的TTFT、完整完成时间和最大生成间隔组合，同时报告连续延迟分布；其中四档间隔阈值都未限制达标，20格改善不能算20项独立证据。后续需要从具体应用确定主目标，不能看完结果再选有利阈值。",
        "16或32条请求测p99有意义吗": "有意义，它接近这组请求的最慢样本，适合观察极端停顿。重复运行同一批文档用于检查稳定性；要估计生产尾延迟分布，还需要独立文档、不同到达过程和更大的样本。",
        "同样输入和seed_为什么还要检查动作前轨迹": "同步驱动的 host 时刻如果跨过某个到达点，本步能看到的请求集合就会不同。已有对照在首次降 cap 之前已经出现 batch 差异，所以需要分开记录动作前变化和动作后的影响。",
        "为什么不能固定route_trace离线换cap": "cap 会改变入场顺序、batch 形状和后续执行状态，原轨迹里的未来路由未必仍成立。我的不同策略各自真实推进队列、KV 和生成；若做回放上界，也必须处理动作后的状态变化。",
        "输出token不同_性能对比还有效吗_质量保持了吗": "当前可以描述这些策略在同一外部负载下的实际系统表现，尚不能证明等工作量加速或质量保持。C v25两对输出总量分别变化约+0.64%和−0.60%，185和194条请求的输出序列不同；允许EOS后仍有超过95%的请求达到上限。质量需要对应任务评估，不能只用输出长度或相同seed替代。",
        "最好静态cap事后才知道_实验是不是不公平": "事后最好静态点用于判断简单配置的性能水平。部署比较时，应该先在校准数据选定 cap，再到独立运行中比较动态策略。256 之后继续探索 512 属于参数探索，独立确认需要另用冻结配置和新的运行。",
        "in_process计时能代表HTTP服务延迟或GPU计算时间吗": "只能代表本实验驱动观察到的引擎请求时间。真实服务还要加入网关、分词、网络与流式输出开销；host 包围一次调用也可能包含等待和其他工作，纯 GPU 执行需要设备侧计时或 profiler。",
        "请求还没运行_怎么知道它会访问哪些专家": "准确路由要在对应层的 hidden state 进入 router 后才知道。首次准入只能利用当时可见的输入信息或历史统计；prefill 后可以观察已发生的专家访问，用于预测 decode，但预测需要测误差。首次准入的特征只能取自决策时已经可见的信息。",
        "专家特征怎样证明比普通负载有用": "固定同一个动作和资源预算，先比较长度、running/waiting、KV 和近期延迟，再加入当前专家负载，最后检验额外路由历史。除了预测误差，还要真实执行决策，比较完整请求收益并计入采集开销；数据按独立请求或到达过程切分。",
        "专家遥测延迟或丢失_怎么处理": "设计上给信号带时间和执行窗口，过期或无法对齐就回退到普通队列、长度和 KV 规则。决策不等待遥测返回，缺失也不当成零负载；恢复时加滞回，避免频繁切换。",
        "持续过载怎么办_一直排队还是拒绝": "在线方案应同时限制等待请求数、待处理 token 量和等待时间，超过预定边界就拒绝新请求。已接纳请求继续按契约处理；评价时记录全部到达、拒绝、超时和 goodput，同时评价接纳能力与延迟。",
        "多个租户怎么公平_高优先级能一直插队吗": "先约定服务份额与资源配额，再在租户内处理优先级和等待时间。按请求数轮询会忽略长度差异，按 token 数也需要考虑 prefill/decode 成本；高优先级仍受额度约束，并检查低优请求的最大等待。",
        "快超时的请求是不是一定优先": "应比较截止时间减去当前时间和预计剩余成本，还要判断是否仍来得及完成。首 token 前可按 TTFT 截止；只有业务明确约定最大 token 间隔时才有逐 token 截止，平均 TPOT 不能直接当成逐 token 硬期限。",
        "prefill干扰decode_为什么不直接PD分离": "跨 GPU 分离能分别配置两阶段资源，但会增加 KV 传输、阶段排队和两边负载不均。单卡预算实验先回答共置条件下的权衡；若比较分离方案，需要保持总资源可比，并把传输和等待都计入请求。",
        "多副本选短队列还是前缀命中_运行中能迁移吗": "新请求应比较预计排队加未缓存 prefill 成本，缓存命中只是其中一项。运行中迁移还需要搬运 KV、确认目的端容量并协调切换；它能使用其他副本余量，但有带宽和停顿成本。",
        "HPA扩容了_为什么还需要引擎内调度": "HPA 增加服务副本，引擎调度安排当前副本的请求，生效时间不同。扩容还要等待 GPU 资源、模型加载和实例就绪，当前队列不会因副本目标增加立即清空；联动时可用快速准入应对短时压力，用较慢扩容补长期容量。",
        "Kueue_Kubernetes调度器_你的cap分别管哪一步": "Kueue 管作业何时取得配额，Kubernetes 调度器给 Pod 选节点，引擎 cap 管模型已运行之后接纳多少请求。集群有空闲 GPU 不代表当前副本有 KV；总共空 8 张卡，也不意味着一个要求同节点 8 卡的 Pod 能放下。",
        "节点自愈或Pod重启_推理请求会自动接着生成吗": "通常不会，进程内 KV 和未完成请求状态可能已经丢失。服务层需要约定重试、已返回内容与重复输出的处理；想继续同一请求，还需要显式保存或迁移状态。引擎内部抢占能够重算恢复，不等于节点重启也具备同样能力。",
        "和训练调度有什么关系_能直接用于训练吗": "共同点是资源准入和负载管理，状态与目标不同。在线推理关注单请求 TTFT、生成间隔和 KV；分布式训练还要考虑整组 worker、梯度同步、优化器状态和 checkpoint，当前请求调度不能直接替代训练调度。",
        "已有Orca和PagedAttention_你的工作多了什么": "我复用连续合批和分页KV，当前重点是把已有MC-Benchmark的未来容量检查适配到真实异步运行时。它要将逻辑输出、在途计算和物理页释放对应起来，并在条件不符时保守回退。我的已完成工作包括这部分适配及完整请求对照，未来峰值准入原则本身来自已有论文。",
        "Sarathi说切分prefill有效_你的小预算结果矛盾吗": "不矛盾，chunk 大小本来就在吞吐、TTFT 和生成间隔之间取舍。Sarathi-Serve 按目标与硬件成本选预算，我的结果说明在这组 cap 8 负载下继续减小预算，保护部分 ITL 的同时增加了首 token 和完成时间。",
        "FCFS_SJF_SRPT_MLFQ在LLM里怎么区分": "FCFS 按到达顺序，SJF 看预计总服务成本，SRPT 看剩余成本并允许抢占，MLFQ 根据已获得的执行服务调整队列优先级。LLM 通常知道输入长度、不知道最终输出长度，合批又会改变成本；短 prompt 优先只是其中一种近似。",
        "TetriInfer和FastServe与你的排序有什么区别": "TetriInfer在有限窗口内排序，并结合阶段分离和输出长度预测；FastServe使用迭代级抢占、多级反馈队列和KV换入换出。我的早期短prompt实验只重排从未执行的等待请求；当前C主线保持FIFO，只决定队首请求是否进入首次prefill。频繁切换运行请求还要计入KV搬运、重算和暂停成本。",
        "VTC每token记账就是每token抢占吗_和DRF一样吗": "不是。基本 VTC 累计客户端获得的 token 服务成本，主要在有准入空间时选择服务不足的客户端；其服务量公平不等于每个请求延迟相同。DRF 比较多种资源上的主导占用份额，解决的是另一层的资源分配。",
        "缓存命中最大_为什么不一定TTFT最短": "长 prompt 如果已有大量前缀缓存，实际剩余计算可能少于短 prompt；但命中副本的排队也会进入 TTFT。SGLang/RadixAttention 提供前缀复用，实际路由还需兼顾负载和最大等待，避免只追命中率。",
        "专家和KV联合管理_WiSP不是已经做过了吗": "WiSP已经研究专家分页与expert/KV预算，联合管理本身已有先例。当前C主线固定权重与KV预算，比较全额预留和未来峰值准入，不包含专家权重回收。若研究权重与KV的联合分配，需要把加载代价与容量收益一起计入完整请求，并核对WiSP等最近邻的覆盖范围。",
        "ELDR也用专家特征_FluxMoE也加载专家_你的区别在哪": "ELDR利用prefill专家签名选择decode worker，FluxMoE处理权重分层和加载；当前主例则是单引擎KV准入，直接最近邻是MC-Benchmark。机制层次不同只说明比较对象不同，不能据此证明创新。专家特征尚未在我的当前主线中形成独立收益。",
        "为什么有的论文说预取有用_有的说没有": "取决于搬运能否被计算覆盖，以及预取是否挤占必需数据的带宽。WiSP 与 FluxMoE 的并发、存储层次和流水方式不同；专家预测准还不够，必须把无效预取、等待和新增 KV 收益一起算。",
        "DistServe_Llumnix_Lina_Aurora都要做性能基线吗": "应先选动作与资源条件接近的基线。DistServe做阶段分离，Llumnix做跨实例迁移，Lina/Aurora涉及专家部署或通信；当前C主线直接比较全额预算与已有MC组件适配。涉及对应多卡动作时，再在匹配资源与服务目标的条件下比较，不能只因都属于推理系统就把所有方法堆成基线。",
        "以前研究RCBA_为什么现在讲请求调度": "RCBA 停在协议和 evaluator 设计阶段，目标是判断路由相关专家长尾怎样放大成请求等待。我后来把动作转到原生引擎中实际可执行的准入、预算和队列上，直接观察队列、KV 和完整请求结果，先确认哪些约束在主导延迟。",
        "动态batch的A_B_C_D四组对照怎么做": "四组从同一目标请求的 token、position 和深拷贝 KV 开始：A 单独运行；B 保持批宽，其余行放目标的独立副本；C 恢复原同批请求；D 保持 C 的批宽、各行长度和 padding，换成等长的不同请求。每组重复并交替顺序，核对状态一致且 KV 存储不别名。",
        "动态batch差异定位到哪里_最终token变了吗": "六个选定事件里，A/C 的首次可见差异在第 0 层 Attention 输出，C/D 在 MoE 输出。A/C 同时改变批宽等多个因素，根因需要进一步拆分。4 个事件各继续 2 个 teacher-forced 步骤，共 8 个事件-步，logits 差异保留 8/8、专家选择差异 6/8，预测 token 变化为 0/8；这部分测量采用固定喂入 token。",
        "原生后端没有复现差异_是不是推翻了旧实验": "两组干预范围不同。原生局部探针固定零索引 layer 3、16 行和目标位置，36 次调用中专家内部布局改变而目标输出逐位相同；旧实验还改变批宽及上游 Attention/KV 状态。进一步比较需要在原生后端还原相同的完整 decode 状态。",
        "Oracle有什么用_为什么不能拿事后最好静态点当Oracle": "Oracle 用理想信息估计同一动作集合、容量和依赖约束下的收益上界，帮助判断还有多少空间。事后最好静态点只覆盖固定配置，不覆盖所有动态动作；真正的动态上界还要重新处理动作后的队列、KV、路由和成本。",
        "最后没有稳定收益_论文还能写什么": "当前可以围绕未来容量原则在真实异步运行时中的适配、代价和边界继续组织材料，但能否形成论文取决于新增知识的力度。已有一组开发结果说明更早入场伴随生成速度代价，还需要重要的应用问题、可信解释和必要的独立证据。候选失败或实验数量本身不能保证形成表征论文。",
        "如果同时要求首token快和生成不停顿_现有两档该选哪个": "要按同一业务要求检查两档的完整结果。以早期5090为例，假设要求TTFT在5秒以内、最大ITL在200ms以内，safe29的TTFT接近18秒，native32又有约2秒和4.5秒暂停，两档都不满足。这说明该组配置存在取舍，不能靠选一个更好看的指标决定；当前PRO 6000的F/MC也应按明确应用目标比较，不能套用旧组阈值或宣称已经解决同一个问题。",
        "17%的吞吐按请求还是Token统计_固定了哪些比较条件": "这组 17% 使用请求每秒，计算为完成请求数除以整批总完成时间，准入等待也计入。两组都是 32 条相同请求，输入 3072 token、输出长度 1024 token，按 50 毫秒间隔到达，固定模型、精度和显存预算，比较并发 32 与保守并发 29。原报告首组为约 1.382 对 1.181 请求/秒，重复为约 1.373 对 1.172 请求/秒，增幅约 16.99% 和 17.08%。这是该固定负载的配置对照结果。",
        "使用现成压测工具还是自己写的工具_Codex做了什么": "用的是基于原生 vLLM 写的实验脚本，Codex 辅助脚本开发和运行。脚本控制请求到达、准入和指标采集，推理实际在 vLLM 中执行。除完成时间外，还关联每条请求的 token 时间、KV 占用和抢占记录，分析某次暂停发生在哪个阶段。",
        "vLLM已经有并发参数_你的研究是不是只调参": "早期包含并发和预算参数对照，当前已经实现了读取请求状态与物理页的准入门控，并适配已有未来峰值检查。固定cap只限制数量，未来峰值方法会利用各请求的上下文、剩余声明输出和完成释放时机；这需要运行时接线，但实现复杂度本身不能证明新颖性。现阶段准确定位是已有方法适配和服务取舍研究。",
        "为什么选择未来释放感知准入作为当前主线": "这条线已经连上了问题、方法、实现和完整结果：保守预留会限制入场，已有未来峰值方法可以利用旧请求完成后的空间，异步实现需要核对实际释放边界，最终还要看首响应收益是否伴随其他代价。它适合讲清我怎样从基础调参推进到运行时建模和策略评价；当前不把它当作已完成的新算法论文。",
        "全额预算和未来峰值检查有什么区别": "全额预算把已占用物理页、已接纳请求尚未分配的声明增长和新请求的完整需求一起计入。未来峰值检查按后续轮次估计：旧请求还会增长，但完成后会释放，新的需求不一定与所有旧请求的最大需求同时出现。因此总声明可能大于容量，实际每个时点仍可能容纳；必须满足持续推进和释放边界等条件。",
        "MC来自哪里_你完整复现了原论文吗": "未来峰值检查来自Online Scheduling for LLM Inference with KV Cache Constraints中的MC-Benchmark。当前保留FCFS，将这项容量判定做成vLLM运行时组件，补齐分页、异步在途和保守释放边界。没有完整复现论文里的MC-SF排序或其全部实验与理论设定，所以结果定位为组件适配。",
        "异步适配具体难在哪里": "异步时，CPU看到的输出数、已经调度的token数和GPU相关工作完成状态可能不同。直接用prompt加已返回输出估页可能漏掉在途位置；把请求逻辑结束当成页立即可复用也可能过早借出容量。我用computed加本轮scheduled对齐当前实际页数，并核对在途批次、placeholder和释放边界，资格不符就不新增未来容量借用。",
        "为什么释放延后两轮_所有异步系统都适用吗": "两轮来自当前固定的单卡、FIFO批次队列深度2和退休边界适配，用来保守覆盖在途返回及延迟释放。模型中的单位是非空调度轮次，不能换成固定毫秒。这是当前源码与部署条件下的处理；改变队列深度、推测解码或调度顺序后必须重新检查，也不能用本次零抢占代替全域安全证明。",
        "旧请求不能继续decode时怎么办": "当前只有旧请求都完整驻留、每轮能安排一个普通decode token，且在途状态和页数一致时才允许未来峰值放宽。出现prefill、terminal-inflight或其他不满足条件的状态，就保持普通全额预算判定。这个回退只影响新的借用，不撤销已经接纳的请求；守卫回退也不能直接叫执行失信。",
        "166和167次放宽是不是多服务了这么多请求": "每次仍然是384条请求全部完成。166和167表示两次候选中，有这些不同请求在全额预算原本会拒绝的时刻，通过未来峰值检查获得了实际分配，并匹配到首次prefill。它证明方法改变了入场时机，不能单凭这个数说明完整服务更好。",
        "KV使用更充分_为什么TPOT反而变差": "更早入场增加了同时活跃的请求，当前观测峰由180升到217；同一GPU要在不同合批负载下服务它们。两次候选平均TPOT都由约45ms升到约48ms，同时控制wall也增加。这与容量复用带来的计算竞争相容，但当前没有单事件对照把计算竞争与控制开销的各自贡献分离出来。",
        "本组零抢占_能说MC消除了抢占吗": "两次MC和两次全额预算都没有抢占。当前比较回答的是同样零抢占时，未来容量能否让请求提前入场，以及这种改变的服务代价。历史原生async组的抢占数属于另一组策略与运行结果，不能拼到这里声称MC减少了相同数量的抢占。",
        "控制器开销有没有算进去": "算进去了。当前两次全额预算的控制wall约0.83和0.85秒，MC约1.88和1.79秒，完整服务计时没有把它们扣掉。插桩计时能描述开销规模，但缺少关闭观测的因果臂，不能直接相减得到可实现的性能收益。",
        "平均TTFT更好_是不是大多数请求都更好": "并非如此。当前两对中TTFT下降和上升的请求分别为171/213、218/166，完成flow下降和上升分别为104/280、162/222。均值可以由少量较大的改善拉动，所以我同时看全人口分布和受损请求；这些逐请求计数也不是独立运行重复。",
        "下一步为什么研究服务代价_这不就是已知调度问题吗": "容量和服务时间的取舍本身已经是已知问题，加入耗时信息并不自动新颖。当前只把它作为筛查线索：明确应用目标后，检查容量相近的状态是否需要不同准入动作，以及简单并发上限和已有方法是否已经足够。如果找不到具体的新约束或可控制残差，就没有理由再包装一个感知控制器。"
      },
      "followup_sequences": {
        "研究问题与本人贡献": [
          "为什么选择未来释放感知准入作为当前主线",
          "你自己写了什么_哪些复用vLLM",
          "MC来自哪里_你完整复现了原论文吗",
          "当前创新点和收益分别是什么",
          "这些问题Dense模型也有_为什么研究MoE",
          "题目写专家负载感知_现在实际做到哪一步",
          "下一步具体做什么",
          "论文上线了吗_单卡结果能说明什么"
        ],
        "请求链路与基础概念": [
          "一个推理请求从用户到返回经过什么_你改哪一层",
          "prefill和decode有什么区别_为什么会相互影响",
          "KV缓存到底存什么_为什么不缓存Q",
          "continuous_batching和chunked_prefill是一回事吗",
          "MoE的一层怎么计算_专家是什么",
          "只激活少数专家_为什么模型仍然很占显存",
          "专家负载具体看什么_卡利用率不够吗",
          "专家负载不均_一定会拖慢请求吗",
          "TP_EP和多副本有什么区别_当前用了哪个"
        ],
        "准入预算与排序实现": [
          "cap_token预算_KV容量分别限制什么",
          "把cap从32降到16_已有请求会被停掉吗",
          "怎么证明参数真的起作用_反馈为什么会来不及",
          "新反馈优于旧反馈_为什么还不算赢",
          "为什么要同引擎比较_并发变小不就更快吗",
          "小prefill预算的实测结果怎样",
          "短prompt优先具体改哪里_效果怎样",
          "短prompt优先就是SJF吗_长请求会不会饿死"
        ],
        "KV容量与抢占恢复": [
          "prefill都完成了_为什么decode还会缺KV",
          "KV显存怎么估算",
          "cap29怎么来的_为什么不是32",
          "线上不知道输出长度_按什么预留",
          "PagedAttention已经解决碎片_为什么还要限制准入",
          "KV不足时抢占什么_是杀掉Pod吗",
          "零抢占不是更好吗_为什么cap32反而吞吐更高",
          "4_5秒是不是全在GPU重算",
          "提高gpu_memory_utilization是不是就解决了_为什么不直接设满",
          "如果同时要求首token快和生成不停顿_现有两档该选哪个"
        ],
        "指标与实验可信度": [
          "TTFT_TPOT_ITL分别是什么_排队算进去了吗",
          "平均TPOT和token_p99正常_为什么用户还卡几秒",
          "吞吐_goodput_SLO达标率怎么区分",
          "SLO怎么定_换阈值会不会换一个赢家",
          "16或32条请求测p99有意义吗",
          "同样输入和seed_为什么还要检查动作前轨迹",
          "为什么不能固定route_trace离线换cap",
          "输出token不同_性能对比还有效吗_质量保持了吗",
          "最好静态cap事后才知道_实验是不是不公平",
          "in_process计时能代表HTTP服务延迟或GPU计算时间吗"
        ],
        "专家信号与上下游设计": [
          "请求还没运行_怎么知道它会访问哪些专家",
          "专家特征怎样证明比普通负载有用",
          "专家遥测延迟或丢失_怎么处理",
          "持续过载怎么办_一直排队还是拒绝",
          "多个租户怎么公平_高优先级能一直插队吗",
          "快超时的请求是不是一定优先",
          "prefill干扰decode_为什么不直接PD分离",
          "多副本选短队列还是前缀命中_运行中能迁移吗",
          "HPA扩容了_为什么还需要引擎内调度",
          "Kueue_Kubernetes调度器_你的cap分别管哪一步",
          "节点自愈或Pod重启_推理请求会自动接着生成吗",
          "和训练调度有什么关系_能直接用于训练吗"
        ],
        "相邻方法与取舍": [
          "已有Orca和PagedAttention_你的工作多了什么",
          "Sarathi说切分prefill有效_你的小预算结果矛盾吗",
          "FCFS_SJF_SRPT_MLFQ在LLM里怎么区分",
          "TetriInfer和FastServe与你的排序有什么区别",
          "VTC每token记账就是每token抢占吗_和DRF一样吗",
          "缓存命中最大_为什么不一定TTFT最短",
          "专家和KV联合管理_WiSP不是已经做过了吗",
          "ELDR也用专家特征_FluxMoE也加载专家_你的区别在哪",
          "为什么有的论文说预取有用_有的说没有",
          "DistServe_Llumnix_Lina_Aurora都要做性能基线吗"
        ],
        "历史实验与研究追问": [
          "以前研究RCBA_为什么现在讲请求调度",
          "动态batch的A_B_C_D四组对照怎么做",
          "动态batch差异定位到哪里_最终token变了吗",
          "原生后端没有复现差异_是不是推翻了旧实验",
          "Oracle有什么用_为什么不能拿事后最好静态点当Oracle",
          "最后没有稳定收益_论文还能写什么",
          "吞吐提高17%_能写成你的算法加速吗",
          "17%的吞吐按请求还是Token统计_固定了哪些比较条件"
        ],
        "未来释放感知准入_当前主线": [
          "全额预算和未来峰值检查有什么区别",
          "线上不知道输出长度_按什么预留",
          "异步适配具体难在哪里",
          "为什么释放延后两轮_所有异步系统都适用吗",
          "旧请求不能继续decode时怎么办",
          "166和167次放宽是不是多服务了这么多请求",
          "KV使用更充分_为什么TPOT反而变差",
          "本组零抢占_能说MC消除了抢占吗",
          "控制器开销有没有算进去",
          "平均TTFT更好_是不是大多数请求都更好",
          "输出token不同_性能对比还有效吗_质量保持了吗",
          "下一步为什么研究服务代价_这不就是已知调度问题吗"
        ]
      },
      "related_work": {
        "使用范围": "相邻方法速查，用于比较机制、动作层次和实验设置。",
        "Orca": "逐迭代合批、selective batching、按最大长度预留；对应并发准入与保守容量基线。",
        "PagedAttention_vLLM": "分页与共享 KV，降低存储浪费；容量仍会随并发上下文增长而耗尽。",
        "Sarathi_Serve": "切分 prefill 并按预算与 decode 合批；比较 ITL、TTFT 与吞吐的共同代价。",
        "TetriInfer": "阶段分离、有限窗口排序和输出长度预测；比当前首次入场排序多了窗口、预测器和跨实例放置。",
        "FastServe": "迭代级抢占、多级反馈队列及 KV 换入换出；成本包含搬运和恢复。",
        "VTC": "按客户端累计的输入/输出 token 服务成本安排准入；保证对象是服务量，不是逐请求 SLO。",
        "DRF": "按多资源主导份额分配资源；与 VTC 的时间累计服务量是不同的公平口径。",
        "DistServe": "prefill/decode 分别部署和配资源，以满足延迟要求的吞吐为目标；计入 KV 传输和两侧排队。",
        "Llumnix": "新请求分发与运行中 KV 迁移，使用其他实例的空闲容量；区别于单实例 KV 地址碎片管理。",
        "SGLang_RadixAttention": "按前缀组织并复用 KV；命中率、排队和饥饿需要同时处理。",
        "WiSP": "专家分页与专家/KV 预算分配；低并发 PCIe 条件下预取可能争抢必需加载带宽。",
        "FluxMoE": "权重分层、压缩与按需加载流水；加载是否被计算覆盖取决于其并发与存储条件。",
        "ELDR": "用 prefill 专家签名和负载选择 decode worker；专家特征与跨实例路由已有直接先例。",
        "Lina": "专家复制、映射及分布式通信调度；需要多卡运行域比较，批次耗时与在线 TTFT 分别解释。",
        "Aurora": "基于流量和设备成本安排专家共置及通信顺序；附件核对的主要证据为仿真。",
        "MC_Benchmark": "当前直接最近邻：Online Scheduling for LLM Inference with KV Cache Constraints，§5.2与附录C中的FCFS未来峰值准入；本地保留FCFS并适配容量检查。论文的MC-SF另有按预测输出长度排序，当前未完整复现。原文：https://arxiv.org/html/2502.07115v5"
      },
      "historical_research_lines": {
        "混合精度_RankLane": "已做质量结构、单卡局部微基准和冻结域上界评估。在 p_return≤20%、codec 等成本按零计的乐观条件下，相对 uniform FP8 的最优端到端改善上界为 4.1667%，因此停止该固定执行方案。",
        "动态batch执行一致性": "OLMoE BF16、单卡自定义 cached-decode runtime 的 A/B/C/D 测量及后续传播，记录中间数值与专家选择变化；固定喂入 token 的 8 个事件-步中，预测 token 变化为 0/8。原始源事件 capture 未保留，当前记录覆盖后续测量过程。",
        "RCBA": "历史候选，停在协议与 evaluator 设计阶段：判断路由相关专家长尾是否经运行屏障放大为完整请求等待。J 为平均请求完成延迟，H 为移除合法屏障后的相对改善，A 为原路由与去相关路由的 H 差；当时计划门槛为 H≥10%、A≥3 个百分点。",
        "其他早期候选": {
          "StableBatch": "局部 exact Oracle 在 16 个新请求上恢复 57/84 的 route distance；两个可观测选择器得分均为 −7，低于 shuffle 的 −4，停止该选择器路线。",
          "SemanticFence": "fresh proxy coverage 约 60.39%，但 witness 执行 5 对中 4 对不安全，计入原型开销后净投影为负，转为离线核验。",
          "JoinStream": "单卡微基准 4 组中 3 组出现可重叠窗口，安全收益为 0/4，冻结当前方案。"
        },
        "5090_准入预算与等待排序": "早期cap29/cap32、prefill预算和短prompt排序的结果保留在experiment_facts中，使用原有硬件、指标和样本范围。约17%是原生配置的请求吞吐对照；90%/95%显存预算在旧记录中只有启动失败与重试准备，未在本次追认结果。它们不再代替当前C v25主线，也不与PRO 6000异步结果合并。"
      },
      "evidence_sources": {
        "使用说明": "路径相对工作区根目录；用于核对事实，正式口述按问题选取机制与结果。当前主例只使用下列C v25/r02证据，不将其他线的收益或旧资源状态混入。",
        "当前结果与判决": "MOE_C_admission_20261004/research/C_admission_20261004/RESULTS.md；顶部研究卡与v25/r02完成段",
        "异步适配源码": "MOE_C_admission_20261004/research/C_admission_20261004/async_mc.py；_old_rows、before_allocate、after_allocate、confirm_admitted",
        "完整指标": "MOE_C_admission_20261004/research/C_admission_20261004/analysis/westd-async-mc-r02.json",
        "动作与固定配对及输出差异": "MOE_C_admission_20261004/research/C_admission_20261004/analysis/westd-async-mc-r02.comparison.json",
        "已有方法": "[MC-Benchmark 原文 §5.2/附录C](https://arxiv.org/html/2502.07115v5)",
        "全工作区背景": "docs/research/AI_Infra_研究现状与组织复盘_20261009.md；是导航与评审解释，最新实验事实以各线结果入口为准"
      }
    }
  ],
  "safe_answer_patterns": {
    "usage": "按当前问题检索以下概念，统一遵守 system_short_rules.runtime_use_instructions.answer_rules。长链路与备查说明提供可选细节，输出只保留当前结论所需内容。",
    "langgraph_jenkins_project_qa": {
      "LangGraph是谁选的_为什么选择它": "LangGraph 的选型是我参与比较并主导的。当时的部署流程需要多轮补充信息、查询工具、生成配置，再等待人工确认；用户修改参数后，还要回到相应步骤继续。普通函数也能串起来，但对话状态、分支跳转和审批暂停都需要自己维护。LangGraph 可以把这些步骤定义成节点，用条件边控制跳转，通过 interrupt 暂停并等待确认，比较符合我们的交互流程。我主要看中这套状态和流程控制能力，代价是需要维护状态字段、节点输入输出，以及恢复时如何处理已执行的外部操作。",
      "哪些交给模型决策_哪些必须由代码控制": "模型负责理解需求和生成候选参数，执行条件与外部操作由代码控制。比如用户说“把某台设备升级到最新版本”，模型可以提取设备名、升级意图和版本要求。设备名对应哪个 IP、最新版本具体是什么，要调用资产和版本接口查询；查询失败或升级条件不满足时，代码直接停止。生成的部署参数还要经过人工确认，审批通过后才进入 Jenkins 执行节点。请求超时时，代码根据已经保存的请求标识查询 Queue、Build 状态，再决定如何继续；执行结果也以这些接口返回的状态为准。"
    },
    "generic_concepts": {
      "k8s_controller_informer": [
        "Operator 用资源对象表达期望状态，再通过控制循环把实际状态持续收敛过去。以出口规则为例，Informer 感知对象变化后，把资源 key 放进 WorkQueue；worker 取出 key，读取当前规则及关联资源，检查需要补建、更新或清理什么。执行结果写回状态，暂时失败则退避重试。由于同一个对象会被重复处理，外部动作需要幂等，并能根据持久化状态继续推进。",
        "经典 List/Watch 流程先 List 获取资源快照和对应的 resourceVersion，再从该版本 Watch 后续增量。Informer 更新本地缓存并通知事件处理器，处理器快速把资源 key 放入 Workqueue，由 worker 读取当前对象并调和。连接断开后可以从最近版本续接；历史版本过期时重新 List，再建立 Watch。Workqueue 对 key 去重，并提供失败重试和退避，避免把耗时工作放在事件回调中。",
        "Workqueue 保存需要重新调和的对象 key，相同 key 可以合并。worker 处理时读取对象当前状态，因此队列无需保留每次中间版本。",
        "Reconcile 读取 API 对象与可观察的外部状态，比较期望和实际差异，再执行下一步；事件用于触发这次检查。",
        "为什么 Controller 必须幂等：同一对象可能因重试、Watch 重连、进程重启或周期同步被重复处理，实现思路是先观察当前状态再执行最小必要动作，关键步骤持久化，外部请求结果不确定时先查询结果不能直接重发。",
        "resourceVersion/乐观并发控制：防止多个写者静默覆盖同一对象，旧版本写入会冲突，调用方需重新读取计算；它不能自动解决业务层冲突。",
        "Leader Election 协调多个副本的执行权，减少并发调和冲突。旧 Leader 发出的外部请求可能仍在执行，新 Leader 接手后先读取任务状态并查询实际结果，再决定是否继续操作。",
        "At-least-once 与 exactly-once：Controller 调和通常是 at-least-once，跨 Kubernetes、云 API 和 SSH 很难实现 exactly-once；工程上应让普通请求可以安全重试，同时避免重启等破坏性结果重复发生。"
      ],
      "alibaba_interview_gap_qa": {
        "Controller有状态和无状态怎么区分": "看关键任务状态能否在实例替换后恢复。无状态 Controller 把期望和执行进度保存在 API 对象或外部系统，进程重启后重新读取并继续处理；内存缓存和队列可以重建。如果正确性依赖实例本地的持久状态，就还需要设计状态复制、恢复和实例身份。",
        "Controller有内存队列和InformerCache还算无状态吗": "算。Informer cache、workqueue、limiter 和正在执行的 goroutine 都是可重建的运行时状态，只要它们不是唯一事实源，进程重启后能够从 API 对象和外部实际状态恢复，就仍然是逻辑上的无状态 Controller。",
        "Controller应该用Deployment还是StatefulSet": "多数 Controller 使用 Deployment，关键任务状态存放在 API 对象或外部系统，实例可替换。需要多副本时结合 Leader Election 或明确的任务分工；实例确实依赖稳定身份或本地持久化状态时，再考虑 StatefulSet。",
        "单副本等于有状态吗_多副本等于无状态吗": "都不等于。单副本进程如果重启后能从 API 对象恢复，仍然可以是无状态；多副本进程如果共同依赖本地不可恢复状态，也可能是有状态。副本数描述部署可用性，有无状态描述恢复和替换边界。",
        "LeaderElection解决什么_没解决什么": "Leader Election 解决多个 Controller 副本中谁执行有副作用的调和，降低同时派单或同时修改外部系统的冲突。它不保证 exactly-once，不恢复旧 Leader 的内存，也不自动解决新旧版本兼容；外部动作仍要依靠持久化任务、幂等键和实际状态查询防重。",
        "多副本Controller怎么滚动升级": "先保证 CRD、任务状态与外部协议兼容，再通过 Deployment RollingUpdate 逐步替换实例。新旧版本共存时，通过 Leader Election 协调执行权；新 Leader 读取未完成任务和外部实际状态后继续处理。旧实例发出的远端操作结果不确定时，先查询结果，再决定下一步。",
        "把状态持久化到APIServer准确吗": "更准确地说，状态保存在 Kubernetes API 对象的 spec/status 中，经 API Server 写入 etcd。Controller 通过 API Server 读写这些对象，API Server 提供校验和并发控制，etcd 承担持久化。",
        "Deployment从提交到Pod运行经过哪些组件": "API Server 接收 Deployment，完成认证、鉴权和准入后写入 etcd。Deployment Controller 创建 ReplicaSet，再由 ReplicaSet Controller 创建 Pod。Scheduler 为未绑定的 Pod 选节点并提交绑定；目标节点的 kubelet 协调卷和镜像准备，通过 CRI 请求运行时创建 Pod sandbox 和容器，运行时调用 CNI 配置网络。容器启动后，kubelet 更新状态并执行已配置的探针。",
        "Kubernetes集群为什么不能无限扩大": "节点和 Pod 增多会同时放大 API 请求、List/Watch 通知、etcd 写入和调度计算，控制面与插件的处理能力都会成为约束。5000 节点对应经过验证的一组规模条件，实际承载量还取决于 Pod 密度和对象变更频率。继续扩大需要控制面扩容与参数调优，或拆分集群，代价是增加跨集群管理成本。",
        "有状态应用一个Pod故障会怎样": "取决于它的角色、数据是否持久化、复制协议和剩余副本是否还能形成 Quorum。从副本故障可能只降低冗余，Leader 故障通常需要重新选主；Kubernetes 能重建 Pod 和重新挂载 PVC，但数据一致性、主从关系和业务可用性由应用自身保证。",
        "有状态应用怎么跨集群迁移": "重点是先迁状态，再切流，最后下线旧集群。先在目标集群准备运行和存储环境，通过复制、备份恢复或存储迁移追平数据；切换前控制最后写入窗口并确认不会双写分叉，再切 Service/DNS/入口，验证目标集群后才停止源集群，同时保留回滚条件。",
        "节点自愈后的Pod重调度是集群内还是集群间": "当前节点自愈链路里的 Drain 和 Pod 重建发生在同一 Kubernetes 集群内，上层工作负载 Controller 重新创建 Pod，本地 Scheduler 再选择健康节点。跨集群迁移需要额外的多集群控制面先选择目标集群，不是单集群 Scheduler 自动完成。",
        "多集群调度和单集群调度怎么分层": "多集群控制面根据集群容量、地域、GPU 类型、队列、数据位置和故障域完成 Cluster Selection，再把 workload 下发到目标集群；目标集群内部 Scheduler 完成 Node Selection，kubelet/Device Plugin 再完成设备分配。Karmada 更偏多集群资源传播和调度，Volcano 更偏单集群批任务 Queue、Gang 和抢占。",
        "SLI_SLO_SLA和错误预算是什么_节点自愈如何关联": "SLI 是实际测量的服务指标，SLO 是指标的目标和统计窗口，SLA 是服务约定及未达标责任。错误预算是 SLO 允许的失败范围：假设一个窗口有 100 万次有效请求，成功率目标为 99.9%，允许失败 1000 次；已经失败 800 次，就消耗了 80% 的预算。窗口和请求量变化时需要重新计算；如果最近错误率为 1%、允许错误率为 0.1%，预算消耗速度就是正常额度的 10 倍。预算紧张时，团队按约定减少高风险变更并优先修复可靠性。GPU 自愈侧看故障发现、维修和资源回池耗时；训练侧看任务恢复时间与进度损失，推理侧看成功率和延迟。维修动作本身造成的业务中断也要计入对应指标。",
        "StatefulSet如何滚动升级_怎样灰度和处理失败": "StatefulSet 提供稳定的副本序号、网络身份和每副本存储关联，Pod 重建后 UID 会变化，但可以继续使用对应的 PVC。以默认 OrderedReady 和滚动更新并发为例，修改模板后从最大序号开始逐个替换，等待当前 Pod Running、Ready，配置了 minReadySeconds 时还要满足稳定时间，再继续更新前一个。三个副本 app-0、app-1、app-2 可以先设 partition=2，只让 app-2 使用新模板，验证后再依次降到 1 和 0；低于 partition 的副本即使重建也保持旧版本。OnDelete 策略则不会主动替换已有 Pod，需要删除后才按相应模板重建。有状态升级还要检查副本是否追平、剩余成员能否维持 Quorum、是否需要切换 Leader，以及新旧版本和数据格式是否兼容；Ready 只证明配置的就绪条件通过，PVC 保留也不会自动完成数据复制或迁移。StatefulSet 自身滚动更新不通过 Eviction 受 PDB 约束，因此必须自行控制更新并发和应用恢复门槛。新版本一直不 Ready 时先停止推进并排查启动、挂载和应用恢复；默认有序更新可能卡在坏 Pod 上，回退模板后还需要视情况重建已使用坏配置的 Pod。重建前确认副本和数据安全，镜像回退无法撤销已经发生的不兼容数据迁移。"
      },
      "pdb_custom_workload_qa": {
        "PDB是原生机制吗_通过什么识别一组Pod": "对，PDB 是 Kubernetes 原生的 PodDisruptionBudget，用于限制主动驱逐造成的中断。它通过 spec.selector 匹配同一个 Namespace 下的一组 Pod，再根据健康状态和 minAvailable 或 maxUnavailable 计算允许的中断预算。\n\n比如五个副本配置 minAvailable: 4，五个都健康时允许驱逐一个；预算不足时，Eviction 会被拒绝。我们的 Drain 遵守这个结果，等待预算恢复。PDB 负责限制中断，替代 Pod 由工作负载 Controller 创建；它不能阻止硬件故障等非自愿中断。",
        "五个副本保留四个_两个Drain并发会不会都驱逐成功": "准确地说是至少保留四个健康副本，五个都健康时，最多允许主动中断一个。假设这五个 Pod 都 Ready、受同一个 PDB 保护，配置 minAvailable: 4 后，disruptionsAllowed 是 1。\n\n我们调用 Eviction，API Server 会先通过带资源版本校验的状态更新扣减这一个预算，并记录在途驱逐，再推进 Pod 删除；此时再次驱逐另一个健康 Pod，就会因为预算不足被拒绝，通常返回 429。不需要等第一个 Pod 真正退出，预算才被占用。两个 Drain 并发时竞争的是同一份预算，发生更新冲突的请求需要读取最新状态后重试，不能重复使用已经扣掉的额度。\n\n等替代 Pod 真正 Ready，并被 PDB 控制器观察、更新出新的可中断预算，Drain 才继续。如果替代 Pod 一直 Pending，或者模型还没加载完、Readiness 没通过，就继续等待，不能为了推进维修改成直接 Delete。这个约束针对经过 Eviction 的主动驱逐，不能保证同时发生硬件故障时也始终有四个健康副本。",
        "模型服务是自定义CRD_自愈需要读取它的replicas吗": "自愈组件不需要直接读取每一种模型 CRD 的 replicas，Drain 提交 Eviction，由 PDB 的预算检查决定能否驱逐。PDB Controller 计算期望副本数时，需要能找到工作负载的 scale 子资源；如果 Pod 实际由 Deployment 管理，就按这个归属处理。自定义控制器直接管理 Pod 且没有 scale 时，可以使用整数 minAvailable；百分比 minAvailable 和 maxUnavailable 需要期望副本数，maxUnavailable 还要求对应同一个控制器。模型是否完成预热、剩余副本是否能承担流量，则要由业务确认。",
        "PDB会感知模型负载_预热或Checkpoint吗": "PDB 主要依据 Pod 的 Ready 状态和副本预算判断主动驱逐，不会主动读取模型请求量、GPU 利用率或 Checkpoint 状态。四个 Ready 副本是否能承接全部推理流量、训练状态是否已保存，需要上层业务或训练平台判断。副本数量约束不能代替这些业务条件。",
        "驱逐和直接删除Pod有什么区别_PDB保护哪条路径": "两者最终都可能终止 Pod，区别在删除前的准入检查。通过 Eviction API 驱逐时会检查 PDB，预算不足就拒绝；直接 Delete Pod 不经过这一步。正常 Drain 应走 Eviction，遇到 PDB 阻塞就等待或调整维修时机，改成直接删除会绕过这层保护。"
      },
      "kubevirt_gpu_passthrough_qa": {
        "KubeVirt解决什么_VM_VMI和virt_launcher是什么关系": "KubeVirt 让虚拟机也能通过 Kubernetes API 声明、调度和管理。VM 保存虚拟机配置与启停策略，VMI 表示一次具体的虚拟机实例。KubeVirt 为 VMI 创建 virt-launcher Pod，由 kube-scheduler 选择节点，再由节点上的组件配合 libvirt/QEMU 启动虚拟机。",
        "虚拟机怎么把GPU透进去_谁选择具体设备": "以整卡 PCI 直通为例，节点先启用硬件虚拟化和 IOMMU，由管理员或节点自动化把目标 GPU 从原驱动解绑并绑定到 vfio-pci。设备插件发现并上报可分配资源，VMI 声明所需 GPU 类型后，Scheduler 选择节点；常规 Device Plugin 路径下，kubelet 的设备管理器选择具体设备并调用插件 Allocate。KubeVirt 再通过 libvirt/QEMU 把设备作为 host device 挂入 Guest，Guest 内安装对应 GPU 驱动后使用这张卡。",
        "Guest读写QEMU模拟设备的MMIO经过什么链路": "在启用 EPT 的 Intel KVM 中，Guest 读写模拟设备的 MMIO 地址时，相关 EPT 表项会让访问退出到宿主机 KVM。KVM 根据访问地址和映射信息进入 MMIO 模拟；需要用户态设备处理时，通过 KVM_RUN 返回 `KVM_EXIT_MMIO`，携带地址、长度和读写信息。QEMU 按地址找到设备的 MemoryRegion，并调用对应的读写回调。读操作还要把设备返回的数据交回 KVM，随后再次调用 KVM_RUN 完成指令模拟并恢复 Guest 执行。",
        "写MMIO设备寄存器后具体做什么": "动作由该寄存器的设备语义决定。比如写 doorbell 寄存器，Guest 已经在内存里准备好请求描述符，QEMU 的写回调据此读取请求并提交后端 I/O。操作完成后，设备模型更新完成状态，并在需要时通知 Guest 中断。",
        "MMIO会触发哪种VMExit_EPTViolation与Misconfiguration区别": "EPT Violation 表示当前 EPT 映射不允许这次访问，Misconfiguration 表示表项编码违反硬件格式要求。Intel KVM 可以主动用非法权限组合标记缓存的 MMIO：缺少映射的首次访问可能触发 Violation，建立特殊表项后再访问可以触发 Misconfiguration，具体取决于表项和缓存配置。需要 QEMU 处理时，返回用户态的退出类型是 KVM_EXIT_MMIO。",
        "Guest普通缺页和EPTViolation有什么区别": "如果缺的是 Guest 物理地址到 Host 物理地址的二阶段映射，会触发 EPT Violation，由 KVM 查找内存 backing 并补建映射。如果缺的是 Guest 自己页表中的虚拟地址映射，产生的是 Guest 的页故障，由 Guest 内核处理。两种情况对应不同层次的地址转换。",
        "KVM拿到GPA后怎么区分RAM与MMIO": "KVM 会根据 GPA 查找已注册的 memslot，判断它是否有可用的内存 backing。QEMU 通过 `KVM_SET_USER_MEMORY_REGION` 等接口把 Guest RAM 的地址范围及用户态 backing 注册给 KVM。像用户态模拟设备的 MMIO 区间没有普通 RAM backing，KVM 会进入相应模拟处理，需要 QEMU 处理时再返回用户态。",
        "QEMU怎么根据GPA定位具体设备": "QEMU 根据 GPA 查自己的 AddressSpace，定位覆盖这个地址的 MemoryRegion。设备创建和 BAR 映射时，已经把 MMIO 范围及读写回调注册进这个地址空间。找到区域后，再计算区域内偏移并调用对应回调，就能执行那个设备寄存器的操作。",
        "MMIO读操作返回Guest前要做什么": "要把设备读出的值交给 KVM，完成 Guest 原来那条读指令。QEMU 将返回值填进 kvm_run 的 MMIO data，再调用 KVM_RUN；KVM 据此完成指令模拟并更新目标寄存器等状态。比如 Guest 执行的是 `mov eax, [mmio_addr]`，就需要让 EAX 得到设备返回值，再继续执行。",
        "VMCS是什么_MMIO读值放在哪里": "VMCS 是 Intel VT-x 管理 Guest/Host 状态以及 VM Entry、VM Exit 控制的结构。MMIO 读出的数据先通过 kvm_run 交给 KVM，KVM 更新被模拟指令对应的 Guest 状态，再按 VMCS 的控制恢复运行。VMCS 不保存全部通用寄存器，相关寄存器还需要 KVM 软件维护。",
        "QEMU模拟设备怎样向Guest发中断": "设备模型通过配置好的虚拟中断路径通知 KVM，再由 KVM 向目标 vCPU 投递中断。以 MSI-X 为例，QEMU 根据设备的消息地址和消息数据形成中断请求，KVM 按中断路由和虚拟 APIC 状态处理。Guest 允许接收时，CPU 交付相应向量，进入 Guest 驱动注册的中断处理程序。",
        "KVM向vCPU注入中断有哪些路径": "Intel KVM 的 CPU 侧有软件 VM-entry 注入，以及 APICv 提供的硬件辅助交付。软件路径把中断向量写入 VM-entry 注入信息；Guest 暂时不能接收时，Interrupt Window 用于等待可注入的时机。启用 Virtual-Interrupt Delivery 后，硬件根据虚拟 APIC 的待处理中断状态完成交付。Posted Interrupt 可以通过描述符和通知向正在运行的 vCPU 投递中断，减少为投递中断而退出 Guest 的开销。",
        "直通设备DMA怎么写入Guest内存": "Host 为设备建立 IOMMU 映射，把设备使用的 DMA 地址转换到 Guest RAM 对应的 Host 物理页。传统没有 vIOMMU 的直通场景中，Guest 驱动通常使用 Guest 物理地址作为 DMA 地址，QEMU/VFIO 为这段地址空间建立映射并固定 backing 页。设备发起 DMA 时，由硬件 IOMMU 校验权限和完成地址转换，数据直接进入对应物理页。",
        "直通设备中断怎样送到vCPU": "常规路径通过 VFIO 和 KVM irqfd 把设备中断事件接入 Guest 的中断路由，再投递给目标 vCPU。支持并启用中断重映射和 Posted Interrupt 时，可以让硬件把设备中断关联到目标 vCPU 的 posted-interrupt 描述符。目标 vCPU 正在 Guest 中运行时，硬件可直接通知并交付，减少宿主机软件中转。",
        "DMA期间为什么需要pin页面_IOMMU映射还不够吗": "需要持有 DMA backing 页的 pin 引用，保证 IOMMU 映射有效期间物理页持续存在，不被正常回收、换出或随意迁移。IOMMU map 决定设备地址对应哪一页，page pinning 保证这页不会被重新分配给其他用途。结束使用时，先确保设备停止访问并撤销 DMA 映射，再释放 pin。",
        "传统VFIO直通是否需要pin全部Guest内存": "在这里讨论的传统 VFIO 直通、没有 Guest DMA 映射通知的场景，通常要把设备可访问的 Guest RAM 整体映射并 pin 住。Host 不知道 Guest 驱动随后会选择哪些页做 DMA buffer，需要覆盖这些可能的访问。通过 vIOMMU 映射通知或协同 DMA 跟踪得到具体范围后，才可能缩小 pin 的范围。",
        "VFIO是什么_和QEMU及IOMMU如何分工": "VFIO 是 Linux 把设备访问、DMA 映射和中断控制提供给用户态的一套框架，QEMU 可以通过它把物理设备分配给 VM。QEMU 管理设备在 Guest 中的呈现，VFIO 提供宿主机侧设备访问接口，IOMMU 负责 DMA 地址转换和隔离。正常数据传输可以直接由 Guest 驱动控制设备完成，减少用户态设备模拟的参与。"
      },
      "gpu_scheduler_deep_dive": {
        "参考回答": "传统 Device Plugin 路径下，GPU Pod 先由 Scheduler 选择节点，绑定后由节点侧分配具体设备。调度时经过 PreFilter、Filter 和 Score，选中节点后在缓存中 Assume，再执行 Reserve、Permit 和后续绑定；失败时撤销相应预留。默认调度器主要按扩展资源数量及已配置约束处理 GPU，不会自动理解显存共享或完整互联拓扑。要按这些条件选择节点或具体卡，需要额外的调度插件和设备分配状态。",
        "失败追问": "Filter 无解就记录 Unschedulable 原因并等待相关事件激活；Reserve 失败后按逆序调用全部已启用 Reserve 插件的 Unreserve；Permit 拒绝或超时、PreBind 失败、Bind 失败也都执行 Unreserve 和 ForgetPod。越靠近 Bind，越可能已经写了 Pod 标注、CR 或外部系统，所以不能只相信内存回调，还要用 Pod UID 等幂等键、状态查询和 Controller 对账处理部分成功与 Scheduler 崩溃。",
        "调度队列参考回答": "Scheduler 内部通常把待调度 Pod 分成三种状态：activeQ 负责立即尝试，backoffQ 负责失败后的限速重试，unschedulableQ 负责等待相关集群事件。新 Pod 进入 activeQ，失败后根据退避和事件竞态进入另外两个队列；Node、Pod、存储或资源变化只激活可能受益的 Pod。这个设计兼顾吞吐、防热循环和事件驱动重试。",
        "公平性参考回答": "优先级决定当前谁先，公平性决定多个租户长期各拿多少。常见做法是层级 Queue 加最低保障、最大上限、权重、借用和回收；多资源场景可以用 DRF 比较 dominant share。GPU 还要处理型号不可替代、Gang 大小和 Checkpoint 成本，所以不能只按卡数平均。",
        "抢占参考回答": "高优 Pod 没有可行节点时，PostFilter 在候选节点上模拟移除低优 Pod，再执行 Filter，判断抢占能否让它放下。选定受害者后删除这些 Pod，记录 nominatedNodeName，等待资源真正释放后重新调度和绑定。PDB 在受害者选择中是尽力遵守，不能保证阻止抢占。节点标签或设备型号不匹配通常无法靠删除 Pod 解决，同节点的 Pod 反亲和冲突则可能因此解除。",
        "现场编码参考回答": "我会把题目给定的 Pod 需求、Node 资源模型和硬约束映射到 Framework：PreFilter 解析并缓存，Filter 做无副作用的可行性判断，Score 做归一化排序，需要临时占用才实现 Reserve/Unreserve，需要 Bind 前写状态才用 PreBind。编码时区分不可调度与系统错误，最后覆盖空值、边界、并发、重复回滚和阶段失败。",
        "二面连续追问路由": "跟随当前追问回答，分别选用成组准入、提交补偿、索引或设备分配等相关材料；需要解释原因时补一个具体对象或例子，不预先展开下一层问题。"
      },
      "k8s_service_networking": [
        "Service 给客户端提供稳定的 ClusterIP 和端口，后端可以是 Pod，也可以是 Controller 管理的外部 VM。以 kube-proxy 的 iptables 模式为例，它监听 Service 和后端变化，将映射写入节点的 Netfilter 规则。请求命中入口后，规则选择后端并执行 DNAT，由路由继续转发；连接跟踪让同一连接的后续报文沿用该后端。业务 VM 也可以作为访问这个入口的客户端。\n\n追问：kube-proxy 是在用户态帮每个请求转发吗？\n在这里的 iptables 模式下，它负责生成和同步规则，实际报文匹配、地址转换和转发由内核完成。",
        "普通 Service 靠 selector 匹配 Pod，由内置 EndpointSlice/Endpoints Controller 自动生成后端；selectorless Service 没有这个自动生成过程，后端地址必须由自定义 Controller 显式维护。",
        "ExternalName 类型 Service 只做 DNS CNAME，不参与转发，没有 ClusterIP、没有 Endpoints，kube-proxy 完全不介入；因此它无法根据后端健康状态及时移除流量，也无法对多个后端做负载均衡。",
        "kube-proxy 和 Cilium 的 Service 接管能力可以按配置并存。安装 Cilium 后，仍需核对 kube-proxy 模式、Cilium 的 kube-proxy replacement 与 Socket LB 状态，并查看对应入口的实际转发表；Partial 状态或存在 Cilium Service/BPF 表，都不能单独证明所有请求已由 eBPF 转发。"
      ],
      "tls_sni_vs_host_header": [
        "通用原理。SNI 属于 TLS 层，在 Client Hello 里携带，发生在加密通道建立之前，服务器靠它在握手阶段选择要返回哪张证书。",
        "Host Header 属于 HTTP 请求，用来在同一 IP 和端口上区分目标站点。使用 HTTPS 时，它随请求在 TLS 通道建立后发送；普通 HTTP 则没有前面的 TLS 握手。",
        "两者信息可能指向同一个域名，但解决的是不同阶段的问题：SNI 解决'握手前该给哪张证书'，Host Header 解决'握手后这个请求该给哪个站点'。",
        "反向代理场景里，如果连接地址用的是已校验的 IP、身份信息（SNI/Host/证书校验）用域名，两者要刻意拆开配置，避免代理在请求过程中重新解析域名从而绕过前置的地址白名单校验。"
      ],
      "virtual_networking_qa": {
        "Underlay和Overlay流量怎么区分": "以 VXLAN 为例，线上报文带有外层 IP、UDP 和 VXLAN 头，VXLAN 头里的 VNI 标识所属虚拟网络。Underlay 按外层 IP 把报文送到目标隧道端点，端点再根据 VNI 选择虚拟网络并处理内层报文。Underlay 和 Overlay 描述的是同一封装报文的承载层与租户逻辑层。",
        "VXLAN在报文哪一层_VNI是什么": "VXLAN 封装在外层 UDP 里，UDP 后面是 VXLAN 头，再后面是原始以太网帧。外层 UDP 的标准目的端口是 4789，实际部署也可配置其他端口。VNI 是 VXLAN 头中的 24 位虚拟网络标识，对端隧道端点据此选择对应虚拟网络，即使不同租户的内层 IP 重叠也可以分别处理。",
        "OVS流表的快慢路径怎么区分": "以 OVS 的内核 datapath 为例，报文命中已有流表后直接执行转发动作；未命中时通过 upcall 交给用户态计算处理动作，并可把结果安装到 datapath。后续同类报文就能在快路径处理，减少反复进入用户态的开销。",
        "为什么Service提供iptables_IPVS等多种数据面实现": "主要是在兼容性、规模和维护复杂度之间取舍。iptables 直接使用成熟的 Netfilter 机制，但 Service 和后端增多时，规则同步及首包匹配成本会增加。IPVS 提供专门的四层虚拟服务表和多种负载均衡算法，同时需要内核模块及相关规则配合，适配 Kubernetes Service 语义也有维护成本。较新版本还提供 nftables 模式，因此选型应结合实际 Kubernetes 版本、内核能力和规则规模。"
      },
      "linux_cpu_isolation_qa": {
        "Linux容器底层靠什么隔离和限制资源": "容器是共享宿主机内核的一组进程，运行环境主要由 Namespace、cgroup 和独立 rootfs 组成。Namespace 隔离进程看到的 PID、网络和挂载等视图，cgroup 控制和统计 CPU、内存等资源。容器运行时准备这些环境后启动进程，进程的系统调用仍由宿主机内核处理。",
        "限制容器最多使用4核_和固定运行CPU有什么区别": "使用 CPU controller 的 quota/period 限制总 CPU 时间，cgroup v2 对应 cpu.max。比如设成 `400000 100000`，表示每 100 毫秒提供 400 毫秒 CPU 时间预算，平均相当于 4 核算力。如果要求只在指定的 4 个 CPU 上运行，再通过 cpuset 配置允许使用的 CPU 集合。",
        "CFS的throttle是什么_何时恢复": "Throttle 是 CPU 时间预算不足时，暂停该组继续参与调度。某个 CPU 上的组队列用完本地 runtime、又无法从组的共享预算取得额度时，内核会节流这个队列，即使里面还有可运行任务。周期预算补充后，满足条件的队列才能继续运行；嵌套 cgroup 还受祖先组预算约束，父组被节流时子组也不能继续运行。",
        "层级cgroup的sched_entity分别代表什么": "中间层的 sched_entity 代表一个子组，叶子层的 sched_entity 代表具体任务。假设层级是 root、A、B、任务 P，那么 root 的队列选择代表 A 的实体，A 的队列选择代表 B 的实体，B 的队列再选择 P。每一级都只在自己的队列里比较可调度实体，逐层向下才选到最终执行的任务。组织这些组关系的内核对象是 task_group。",
        "同一cgroup在多个CPU上如何组织": "同一个 task_group 在各 CPU 上分别关联自己的 cfs_rq 和组调度实体，任务进入它所在 CPU 对应的组队列。比如 A 组的任务同时运行在 CPU0 和 CPU1，就分别由两个 CPU 上的 A 组队列参与调度。sched_group 用于调度域里的 CPU 负载均衡分组，cgroup 的层级调度使用 task_group。",
        "CPU有rq_为什么每个cgroup还有cfs_rq": "CPU 的 rq 管理该 CPU 的整体调度状态，非根 cgroup 在这个 CPU 上另有自己的 cfs_rq。子组通过一个 sched_entity 挂在父级 cfs_rq 中，自己的 cfs_rq 再容纳下一级组或任务。rq 还包含其他调度类的状态，cfs_rq 专门服务于公平调度。",
        "多个CPU如何共同消耗同一cgroup的quota": "Quota 是整个 cgroup 共享的时间预算，各 CPU 的 cfs_rq 分批领取 runtime，再按本地实际执行时间扣减。假设 quota 是 200 毫秒、period 是 100 毫秒，四个 CPU 同时满负荷运行，理想情况下约 50 毫秒就会消耗这份预算。共享池没有额度时，各 CPU 已领取的本地额度仍可使用；本地也耗尽、无法补充后，才节流相应队列。",
        "CFS在哪里扣runtime_层级组如何记账": "CFS 在 update_curr() 中统计当前实体本次执行的 delta_exec，带宽记账逻辑再从所在 cfs_rq 的 runtime_remaining 中扣除这段时间。本地额度不足时尝试向共享池领取，领不到就触发该组队列的节流。层级调度还会更新任务对应的祖先组实体，使上层组预算也覆盖这段执行时间。"
      },
      "k8s_scale_and_cache_qa": {
        "大规模集群如何管理Operator的订阅和调和压力": "我会先看各 Operator 订阅了哪些资源、产生多少调和和 API 请求，再确定需要收缩的范围。能在 List/Watch 阶段用 selector 限定的资源先限定，读操作优先使用 Informer 缓存，写操作只在状态变化时提交。调和设置并发上限和失败退避，事件处理器过滤无关变化，减少重复工作。实例故障通过 Leader Election 接管；单个 Leader 确认达到吞吐瓶颈后，再按互不重叠的资源范围分片。",
        "Calico_Cilium节点Agent全量Watch怎样减轻压力": "先区分本节点私有状态和跨节点必须共享的网络状态，本地 Pod 配置可以按节点过滤，远端 endpoint 的地址、身份和策略关联信息仍要保留。Calico 可以通过 Typha 汇聚订阅、缓存并向 Felix 分发，减少每个节点直接访问 API Server。Cilium 可按版本评估 CiliumEndpointSlice，把远端 endpoint 信息精简合批同步。EventHandler 过滤只减少后续处理，已经接收和缓存的数据流量还在，因此要优先优化订阅内容和状态分发方式。",
        "一次扩容几百个节点怎样避免List_Watch启动风暴": "我会先控制节点接入或 Agent 初始化同步的实际并发，把集中启动分批展开，并在重连时使用退避和随机抖动。客户端 QPS/Burst 要结合实例总数设置，单个客户端有限流并不代表全局请求量受控。对大量重复的全局状态，优先复用 Typha 或相应的聚合分发能力，减轻 API Server 重复读取和发送的压力。Ready 是初始化结果，单独延迟 Ready 不会减少已经发出的 List 请求，限流点必须放在实际启动和请求入口。",
        "大量List_Watch请求的控制面瓶颈在哪里": "大量初始 List 会集中消耗 API Server 的对象读取、序列化、内存和网络资源，即使命中 watch cache，这些开销也存在。批量扩容还会产生 Node、Pod 等对象的创建和状态更新，写请求进一步消耗 etcd 的共识和磁盘能力。如果继续无限增加并发，排队和超时会引发重试或重新 List，进一步放大压力。我会结合 API 请求延迟与限流、API Server CPU/内存，以及 etcd 提交和落盘延迟定位实际瓶颈，再控制对应阶段的并发。",
        "APIServer哪些请求从缓存返回_哪些请求访问etcd": "对使用 etcd 持久化的 Kubernetes 资源，通过校验并实际提交的创建、更新和删除最终都要写入 etcd。WATCH 通常由 API Server 的 watch cache 分发增量，各个客户端不需要分别直连 etcd。LIST 在缓存能提供所需一致性和版本的配置下可以从缓存返回，普通 GET 则需要单独看 resourceVersion 和具体实现。缓存服务减少了存储访问，但 API Server 仍需处理和发送响应。",
        "GET_LIST何时真正读etcd_以Kubernetes_v1_34为例": "以启用 watch cache 的 Kubernetes v1.34 为例，单对象 GET 不带 resourceVersion 时仍走 etcd，resourceVersion=0 则允许从缓存读取任意可用版本。对最新一致性的 LIST，满足功能和 etcd 版本条件时，可以确认缓存已经追平后从缓存返回对象数据。关闭资源缓存，或某些精确版本、分页 LIST 所需的缓存快照不可用时，会进入存储读取路径。历史版本已经过期时还可能返回 410，因此不能把所有缓存无法满足的请求都理解成能自动回源补齐。"
      },
      "kubeflow_argo_workflows_qa": {
        "Kubeflow是什么_与Kubernetes和PyTorch有什么关系": "Kubeflow 是运行在 Kubernetes 上的一组机器学习平台组件，提供开发环境、训练任务管理和流水线等能力。比如 Trainer 管理分布式训练任务，Pipelines 串联数据处理、训练和评估，Katib 做超参数搜索。Kubernetes 负责底层 Pod 的调度和运行，Pod 里的 PyTorch 执行训练计算。各组件可以按需求单独使用，不必为了提交训练任务安装整套平台。",
        "Trainer和Pipelines分别管什么": "Trainer 管一次分布式训练中的进程部署和任务状态，Pipelines 管多个阶段的先后依赖和输入输出。比如数据处理完成后提交训练，训练成功后再评估模型，这个顺序由 Pipelines 推进；训练阶段可以调用 Trainer 创建和管理训练任务。两者可以组合，也可以只使用其中一个。",
        "PyTorchJob和TrainJob有什么区别": "PyTorchJob 属于 Kubeflow Training Operator v1 的框架专用 API；Trainer v2 使用统一的 TrainJob，并引用可复用的训练运行配置 TrainingRuntime 或 ClusterTrainingRuntime。v1 通常在 PyTorchJob 里定义 Master、Worker 的副本和 Pod 模板；v2 把通用运行配置放进 Runtime，单次任务填写训练镜像、命令和规模等参数。这里的 v1、v2 指训练组件的版本，示例配置要与实际安装的 CRD 对应。",
        "PyTorchJob提交以后谁创建Pod_比普通Job多做了什么": "在 Training Operator v1 中，Controller 监听 PyTorchJob，按副本配置创建训练 Pod 和所需 Service，配置分布式启动所需的地址等信息，并根据副本状态更新任务状态。Scheduler 为这些 Pod 选择节点，训练镜像中的启动命令再拉起 PyTorch 进程。普通 Job 也能运行训练脚本，PyTorchJob 额外提供了训练角色、通信配置和整项训练的状态管理。训练代码仍要正确初始化进程组并执行训练循环。",
        "Kubeflow和Volcano的Gang调度有什么区别_怎样配合": "Kubeflow 的训练组件创建和管理训练任务，Volcano 根据队列、资源和整组放置条件决定这些 Pod 什么时候能调度。以 Training Operator v1 的 Volcano 集成为例，启用集成后，Operator 会为训练任务创建 PodGroup，Volcano 根据 minMember 等条件进行成组调度。假设固定八个 Pod 才能训练，当前只能容纳五个，就让这组任务等待，避免部分成员先占着 GPU。即使总 GPU 数够了，单节点容量、亲和性或拓扑约束也可能让整组无法放置；获准调度后，各进程还需要完成启动和通信初始化。",
        "用了Kubeflow以后Worker故障能自动恢复训练进度吗": "训练进度能否恢复，取决于训练程序保存的 Checkpoint 和配置的任务恢复策略。普通同步训练中，一个 Worker 退出可能导致整组通信失败；重新拉起 Pod 后，还需要按框架要求重建相关进程和通信组，再加载完整保存的模型、优化器和步数等状态。Kubeflow 的任务状态和重试配置解决的是任务管理，进度恢复必须落实到训练程序。没有可用 Checkpoint 时，重启可能从头训练。",
        "KubeflowPipelines和ArgoWorkflows怎么选": "需要机器学习实验管理、模型与数据产物追踪、组件复用时，可以优先考虑 Kubeflow Pipelines；主要需要容器任务的依赖、重试和执行状态时，Argo Workflows 更直接。比如同一模型反复比较数据版本和训练参数，Pipelines 提供的实验和产物记录更有用；通用批处理和运维流程通常只需工作流能力。Kubeflow Pipelines 的 Kubernetes 后端可以使用 Argo 执行，两者在产品能力和执行层面有分工。选型还要考虑团队是否愿意维护额外的元数据、存储和平台组件。",
        "Pipelines缓存什么_数据变了为什么可能仍命中缓存": "Pipelines 可以复用匹配输入和组件执行配置的历史输出，避免同一步骤重复计算。假设步骤只接收一个固定的数据路径，而路径里的内容被覆盖，缓存未必能感知这次变化。我会把数据版本或内容摘要作为显式输入，并固定训练代码和镜像版本；依赖实时外部状态或必须重新执行的步骤则关闭缓存。复用时也要保证历史输出仍然可访问。",
        "ArgoWorkflows是什么_已经有Job和CronJob为什么还需要它": "Argo Workflows 用 Workflow 这个自定义资源描述多个任务的依赖、参数和执行结果，由 Controller 推进执行。Job 适合完成一项批处理，CronJob 在指定时间创建 Job；需要先处理数据、再训练、成功后评估，并给不同阶段设置重试时，Argo 可以直接表达这些关系。只有一个定时脚本时，CronJob 就足够；多阶段流程才需要承担额外工作流组件的维护成本。",
        "Workflow_WorkflowTemplate_CronWorkflow分别是什么": "Workflow 记录一次工作流执行，包括输入、步骤和运行状态。WorkflowTemplate 保存可以反复使用的流程定义，提交时传入不同参数，就能生成不同的 Workflow。CronWorkflow 按时间规则创建 Workflow，适合每日数据处理或周期评估。跨 Namespace 共享定义时还可以使用 ClusterWorkflowTemplate，具体执行仍发生在选定的 Namespace。",
        "ArgoWorkflows和ArgoCD有什么区别": "Argo Workflows 执行有开始和结束的任务流程，例如训练、评估和模型打包。Argo CD 持续比较 Git 中的资源声明和集群状态，按同步配置把应用调整到期望版本。两者可以配合：Workflow 产出模型和部署配置，Argo CD 根据已经审核的声明发布服务。长期运行的推理副本随后由 Deployment 或相应服务控制器维护。",
        "提交Workflow以后具体怎么执行_谁决定下一步": "Workflow Controller 读取 Workflow 的入口和依赖关系，为当前可以执行的容器任务创建 Pod。Scheduler 选择节点，kubelet 和容器运行时启动容器，Argo 的执行器收集声明的输出参数和文件产物。Controller 根据任务结果更新 Workflow 状态，依赖和条件满足后再创建后续任务。Argo Server 提供 API 和界面，推进执行的是 Workflow Controller。",
        "Steps和DAG有什么区别_怎样设置条件分支": "Steps 按阶段组织执行，同一阶段里的步骤可以并行；DAG 为每个任务单独声明依赖，适合存在多条分支的流程。假设清洗数据后并行执行训练和数据质量检查，最后评估依赖这两项成功，DAG 可以直接表达这个关系。依赖成功与否可以通过 depends 条件表达，业务条件则可以用 when 判断，例如评估分数达标后才进入发布。",
        "步骤之间怎样传参数和模型文件_能直接读上个Pod目录吗": "小的配置值和指标通过 Parameters 传递，数据集、模型等文件通过 Artifacts 或共享存储传递。使用对象存储产物时，上游上传文件并记录位置，下游按声明下载；使用 PVC 时，需要满足访问模式和节点挂载条件。两个独立 Pod 的本地目录和 emptyDir 不会自动共享。大模型文件留在对象存储或卷里，Workflow 只保存引用及必要元数据。",
        "某一步失败怎样重试_会把整个流程重新执行吗": "可以在失败步骤的模板上配置 retryStrategy，设置重试次数、适用错误和退避时间，已经成功的上游通常不用重新执行。临时网络故障适合有限重试，参数错误和持续 OOM 应先修正原因。重试耗尽后，后续是否继续由依赖条件和失败策略决定。训练步骤重试会再次执行启动逻辑，能否接着上次进度训练，还要看它是否加载 Checkpoint。",
        "重试会不会重复提交训练或重复发布模型_怎么处理": "会，外部操作已经成功而响应丢失时，直接重试可能再提交一次。如果由我设计，会用跨重试保持不变的业务任务 ID 生成固定的训练资源名。超时后按这个名字查询，核对任务标识和配置；对象已存在就继续观察，重新创建返回 AlreadyExists 时也接续同一个任务。模型发布则使用固定版本和请求标识查询执行结果，避免因响应丢失重复发布。",
        "WorkflowController重启会不会从头执行": "正常接管时，Controller 会重新读取 Workflow 的执行状态和关联 Pod，继续处理未完成步骤，已经运行的 Pod 不会因为 Controller 进程重启就自动停止。执行进度保存在持久化对象中，启用节点状态卸载时还需要相应数据库。接管后的具体动作要结合已记录的进度和实际资源状态判断。调用外部系统后尚未记录结果的步骤，仍需要用业务请求标识查询上次执行结果。",
        "Argo怎样调用Kubeflow训练_创建成功就算训练成功吗": "Argo 可以用 resource 模板创建 PyTorchJob 或 TrainJob，并按对应版本的状态字段配置成功和失败条件，确认训练成功后才进入评估。创建请求成功只表明训练对象已经写入，训练 Pod 可能还在排队。也可以让步骤调用 Kubernetes API 并持续观察同一个训练对象，成功后把模型位置传给评估步骤。提交步骤重试时要继续查询已有任务，避免生成另一份训练。",
        "Argo的parallelism能限制GPU用量或实现Gang吗": "parallelism 限制工作流或任务的执行并发，GPU 数量还要由 Pod 的资源申请、配额和调度规则控制。尤其是一个 Argo 步骤创建了八个训练 Pod 时，子任务不会自动按八个 Pod 计入这个工作流的并发限制。需要在训练任务所在的 Namespace 和队列设置 GPU 配额，并通过 Volcano 等调度器处理成组放置。限制同时运行多少个步骤，无法保证一次训练所需的整组资源已经可用。",
        "工作流操作集群用谁的权限_多租户怎样限制": "用户能否提交 Workflow，由用户身份和 RBAC 决定；工作流步骤访问 Kubernetes API，则使用配置的 ServiceAccount 权限。比如一个步骤需要创建和观察 PyTorchJob，就给专用 ServiceAccount 授予对应 Namespace 内这些资源的必要权限。提交入口还要限制用户选择高权限 ServiceAccount、挂载敏感凭据和运行特权 Pod，否则仅限制谁能创建 Workflow 仍可能被绕过。网络访问和产物存储再分别用网络策略及存储凭据控制。",
        "工作流一直不往下走_怎样区分Argo问题和调度问题": "先看当前步骤有没有创建 Pod，把依赖没满足和 Pod 已创建后卡住分开。没有 Pod 时，检查依赖、暂停和并发限制，再看 Controller 事件中是否有权限或配额导致的创建失败。Pod Pending 时，查看调度事件中的 GPU 不足、亲和性或 PVC 绑定问题；Pod 已运行却没有完成，就查任务日志和产物上传状态。步骤提交的是外部训练任务时，还要继续看那个训练对象、PodGroup 和训练 Pod 的状态。",
        "场景设计_怎样串起数据处理_训练_评估和推理发布": "我会把数据处理、训练、评估和发布定义成有明确输入输出的步骤，用 Argo Workflows 或 Kubeflow Pipelines 管理依赖。数据处理产出固定版本的数据，训练步骤提交训练任务并等到成功，再把模型位置和版本交给评估。评估达到阈值并完成需要的审批后，发布步骤更新推理服务版本，等待就绪和必要的请求验证通过才记为发布成功。每次执行关联数据版本、镜像版本和模型版本，失败时可以定位具体步骤，发布异常时按保留的旧版本回退。"
      }
    }
  }
}