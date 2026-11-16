# 故障覆盖与恢复验收标准

适用于逐机制从已验收的单点执行扩大到多目标。资格绑定到机制、版本、模型实现、实际 Native API、Worker/rank/device、构建及预先声明的 workload。状态分别记录单点、多目标、每次匹配、rank 隔离和整组恢复。[机制总账](unified_fault_inventory.md#9171f3b-全局审计)

## 单点资格和现有接口

当前本地原型保留 `QualifiedNativeSession` 的单个合格目标和一次尝试。重复健康调用须定位到同一模块、实际执行位置、slot、cache 注册代次和已加载构建。先取得正常 reference，再比较临时输入 D2D 的原存储哈希、完整原生及模型输出，核验 OFF。Fault39 分别验收 self-only、mat2-only、dual。原生运行尝试开始即消费这一次机会，故障异常依赖已记录的状态恢复。

`CandidateSet`、`TargetIdentity`、`CoverageReport` 和 `TargetLease` 位于 `fault_injection/coverage.py`。`QualifiedTargetSet` 在 CPU 按 lease 和模式核对 D2D 证明、调用和预算，部分恢复时保留待恢复租约。结果为 `execution_authorized=false`。新增资格从现有 Provider/Profile、ModelFaultBackend 和 NativeSession 增量接入。生产多目标与每次匹配 capability 随各 Provider 实机结果授予。[CPU 测试](../tests/test_coverage_planning.py)

## TargetIdentity 和 CandidateSet

每条实际调用记录 PID、Worker、rank、逻辑 device、实际加载的模型实例、模型对象、模块对象、逻辑角色、算子 API、执行边界、executor、cache 注册代次、张量布局和构建签名。调用另带独立 call ID、forward ID、实际请求映射及阶段。对象和运行时身份随相应生命周期失效。

| Provider | 机制专用证明 |
| --- | --- |
| 20、39 | typed GWS、executor/cache generation、当前 Run 形参→live slot、临时 buffer 和原输入存储 |
| 26 | 同 forward producer、allocator、writer、原正常 out、受害残差及真实 consumer |
| 共享量化 | 实际 quant method、scheduler request/row/prefix、健康输入位置及内容 |
| 28、33 | 真实调用策略、typed/serialized tiling、实际 launch。33 另需 page table、KV、请求行和 cache 生命周期 |
| 15、17、18 等 | 实际对象和调用语义，再按机制验证 view/base 写读、Conv3d 几何或模块返回与消费边界 |

结构阶段列候选。运行观察记录支持、待验证、歧义、无法观察、资源受限等原因。完整 CandidateSet 需要对指定请求窗口内的活跃调用给出独立发现结果，动态分支未执行时记录未观察。请求身份缺失时保留 `UNPROVEN_REQUEST_MAPPING`。

`unique` 选取唯一经证明目标，歧义时拒绝。一次性多目标实测先显式选择有界目标。 `all_eligible` 指合格目标集合的选择范围，`every_match` 指每次符合授权且实时合法的调用触发。两项分别验收。每次执行仍核对目标、owner、generation、构建、rank、请求、真实资源和策略。

## 覆盖率和传播

同一个 run/build 下，报告模型、rank、采集窗口、prefill/decode、请求集合、目标及策略。计数分别记录独立发现、结构候选、合格、策略选中、实际命中、透传、拒绝、失败、无观察、资源受限及停止。不同 rank 或 stage 单独可查。

| 指标 | 分子与分母 |
| --- | --- |
| 目标覆盖率 | 经实际命中并对账的计划目标数 ÷ 预先确定的计划目标总数 |
| 调用覆盖率 | 经实际干预并对账的合格调用数 ÷ 从独立 observer 获得的合格调用数 |

独立观察来源可为实际 native trace 或独立 workload observer。未观测或分母不完整时报告未知或部分观察。只有按照真实 Worker/call ID、目标身份、当前 lease、owner 和 generation 关联的回执才能计入。跨 rank、缓存复用、以及同形状不同层须能区分。OFF 意外命中单独计数。

首次异常前为 `PRE_FAULT_PROPAGATION`。首次异常后的调用进入 `POST_FIRST_ANOMALY`，例如生成 token 变化造成下游输入差异。关闭与恢复为 `OFF_RECOVERY`。按各阶段核对调用归属和分母，基线只与可比目标和输入对照。

## 开放顺序和多目标资源

| 当前机制 | 第一阶段允许运行范围 | 多目标及每次匹配的后续条件 |
| --- | --- | --- |
| 20 WeightNz | 发布版可在固定专用同步样板内运行。新 scoped 路径先做唯一单目标。 | 首个候选。先多个目标串行受控，各自 D2D、槽位、cache 和 owner 证明，逐次重新验收每个 hit 与整组恢复，再评估每次匹配。 |
| 20 QMM | 独立新构建先 `unique`。 | 单独核验 QMM x1、scale、有效 out 和临时存储，额外覆盖同形状模板与 cache 共用。 |
| 39 MatMul | 独立包按已验证选择策略和 budget，生产首轮 `unique`。 | 第二个候选。保持 ABI v1，先受控协调多目标。记录同次 self/mat2 临时 buffer、逐 rank lease、请求和整组恢复。 |
| 26 QMM/FIA | 现有固定 Worker 依实际 producer/consumer 和当前分配门禁。 | 每一发生条件都需当前 forward 证据。QMM 非邻接明确透传。写入消费者与异常恢复完整前，每次匹配的执行策略不能扩大。 |
| 共享量化 | 目前固定 replay 只选择首个合格量化点。 | 传播后输入可能不再符合健康哈希。完成独立逐目标参考、动态请求映射和跨阶段策略前，维持单目标。 |
| 33、28 Native 策略 | 仅精确匹配的固定 prefill/decode 和构建作用域。 | 复核动态 shape/page、实际 launch、当前 cache 与 stream 能否区分，再做有界整组恢复。进程级 scope 需明确串行和隔离边界。 |
| 15、17、18 与其余机制 | 目前仅各自已审查入口和历史范围。 | 先完成新入口的真实执行和 OFF。25、41 等需建立可用执行位点。 |

每个目标绑定 owner、generation、资源 lease、执行次数、停止状态和回收状态。用有界资源池计入 host/device 临时字节、并发数、stream/event 及持续时间。预算不足在执行前拒绝并记录。首个多目标样板使用串行、限定目标与固定请求，之后独立验证并发、跨 Worker、错误注入与组恢复。仅当全部目标确认可恢复时授予该作用域的全量资格。

## 恢复语义

停止新注入 → 等待实际在途调用 drain → 恢复机制状态 → 释放本次 owner 资源 → 在同进程复跑健康基线 → 确认恢复。Provider 实现自己的释放和恢复，Controller 汇总真实 Worker 回执。记录各阶段的起止、身份、等待原因、实际 B/F/OFF、资源释放以及未恢复条目。出现异常、超时或遗失 Worker 回执时保留 owner/generation 和待恢复租约并隔离。整组通过要求每个目标释放且基线可复核，并验证同进程另一 owner 或机制的状态。

## 独立 B/F/OFF 证据

使用现有 `compare_model_capture.py`、`compare_msprobe.py` 和试验 `compare_pair.py`。原始 B/F 同配置采 `statistics/mix`。对齐实际 step/rank、输入、model/build、collector、结构调用顺序与栈，公式为 `abs(F-B) > 1e-8 + 1e-5*abs(B)`。保留 null、NaN、Inf、不可比、缺步和采集边界不足。比较器独立从原件找首差，再把执行证据关联到可定位的范围。

26 QMM 的 rank1 step0 双方 NaN 目前阻断“最早变化”判定。下一次同时保留该字段的自然语义、首个可见差异和独立日志。33 优先在真实 IFA 输出边界提高原始可观测性。目前下游 mean 的差异只能定位到传播范围。每个 B/F/OFF 回执都保留原件位置、SHA、比较器身份、未观测项和同进程 OFF 对照。发布只收录摘要和索引。

[现有比较器](../fault39/tools/compare_model_capture.py) · [生产接线与测试矩阵](unified_model_injection.md#生产接线规划) · [现有实机证据](../experiments/provider_adaptation_20261116/README.md)


## 隔夜验收补充

本文件既有资格规则保留；2026-11-16新增[分级证据、独立分母和17条机制覆盖边界](overnight_coverage_acceptance_standard.md)，不改变现有单点执行门禁。
