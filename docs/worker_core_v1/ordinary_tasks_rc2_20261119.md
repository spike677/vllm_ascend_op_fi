# RC2 普通任务建设（开发中，尚未冻结发行）

RC1.1 已单独发布；这里的开发改动不回写 RC1.1 包。蓝区测试部署与黄区现役部署分别验收。

## 公共约束

`ordinary_admission.ordinary_route_report` 是能力查询和创建共同的普通路由判定。
缺绑定显示 pending，实际调用始终待观测；候选/绑定不授予执行资格。当前普通入口为
18、39、Batch、已通过a3实机的8/17及待验的22；默认执行还须匹配部署验收准入。其他八项中的未接入部分返回明确
`ORDINARY_TASK_ADAPTER_PENDING`，不会回退 C1/group/coverage。研究实现仍保留为显式
`create_research_provider` 或直接类调用，不接受用户配置的研究模式开关。
15 当前部署不适用、25 不支持、28 legal 历史保留、28 denominator F/OFF 未完成均关闭。

三层报告为 deployment_identity、mechanism_compatibility、actual_call。身份问题排序
优先于未观察到消费者；当前公共层已接路由/绑定检查与串行 eager 聚合前提。各 Native
加载器的精确 SDK/ABI/源文件/加载库校验继续生效，但其历史独立错误尚未全部汇总到
三层报告，不能宣称这一部分已全覆盖。

`require_serial_eager` 一次列出所有缺失/不匹配字段与期望/实值；RPC 在有界
`prerequisites` 字段返回。CPU custom GEMM/consumer 使用独立只读测试身份，
不再覆盖生产版本或源码常量；设备张量、真实 Native 加载和租约在该上下文被拒绝。

普通 Batch 去掉 qwen2/qwen3 名称白名单。仍须通过实际动态 W8A8 方法链、INT8 权重、
TP1、真实 scheduler→embedding→RMSNorm→Q/scale→QMM 关系，继承六批协议。
新模型名通过结构候选检查不等于适配验收，未知 producer/consumer 仍拒绝。

## Fault8（124 a3 两作用域通过，新增门禁/大小边界待补验）

部署侧调用 `install_restore8_task_binding`，用户只选 `FaultConfig(8,'restore_offset')`。
不填层名。真实 normal forward 中，复用 `NaturalMoETrace/MoERestoreTrace` 核对
专家选择、路由、MLP、combine 的来源和对象关系；首个合格 unpermute 自动选定。
F 在同一调用的原始 I/P 上进行成组映射后调用一次真实算子，不改模型/router，不重放算子。
未选调用保持原参数；不是全部 MoE 层注入，也不宣称全目标验收。

沿用健康索引完整排列、未预先施加 router 权重、完整成组 CPU 参考精确对应原算子的
原有边界。它适用于既有 DeepSeek-V2-Lite-W8A8 TP1 串行 eager 来源组合，其他结构待验。

固定四个真实请求依次为 normal/fault/off/health，名称不选阶段。用户结果在
`response.provider_state.evidence`（兼容旧 serving 字段）；包含实际请求、位置、配对
映射、原算子执行次数、数值变化和恢复。数值摘要明确是“实际输出与精确未移动组参考”
比较，不伪称同一返回值被事后改写。首 token 是否变化另报，不强制文本变化。

正常清理 finish→disable→unregister→detach→remove_tensor_serving_binding；失败先
recover（必要时真实 OFF/health）或 abort，再 clean/unregister/detach/remove。
故障已开始但未恢复时不能清掉所有权。a3 已验证同 Worker 两个独立作用域；新增准入和
公共返回大小边界还需补验，不能把历史结果直接当作新门禁的实机结果。

共享普通请求实现新增 evidence 别名；失败前的原始异常先保存。成功 abort 释放且未实际
注入时，历史失败保留为报告，不再误当活跃失败阻塞 clean。RMS 14 项回归通过。

## 进展与门槛

| 能力 | 当前普通任务进度 | 蓝区/黄区测试/黄区现役 |
|---|---|---|
|18|既有四请求；维护公共结果/清理|既有 Source126；用户转述黄区两作用域通过；现役未验|
|39 output_proxy|既有四请求；RC1.1 诊断与路由修复|既有 Source116；用户转述黄区两作用域通过；现役未验|
|Batch|既有六批；结构而非模型名判定|既有 Source118；黄区缺 W8A8/bundle；现役未验|
|8|新四请求，两个独立作用域|124 a3通过；黄区未验；新增门禁/大小边界待补验|
|17|新四请求，两个独立作用域|124 a3通过；统一完成回调后的路线待补验；黄区未验|
|22|四请求代码与40项定向CPU检查完成|普通入口NPU未验，默认关闭|
|20、26 两变体、33、41|历史机制保留，普通任务待接入|不可沿用历史机制结果宣称新普通入口通过|
|28 denominator|正常关联与 opt-in Execute 借用已保留|F/OFF 未完成，关闭|

定向检查：前提/结构/统计边界 56 项通过；CPU 构建身份与 39 52 项通过；8/RMS/MoE
复用 20 项通过；公共路由与四种普通路径 75 项通过。都是 CPU 检查，部分选择有重叠，
不能相加作为唯一测试数量。基线中的历史冻结 SHA 检查尚未重写或冒充当前验收。

## Fault17（124 a3 两作用域通过，接口统一修订中）

`install_vision17_task_binding` 绑定已验证来源的 Qwen3-VL-2B、TP1 BF16、串行 eager，
单图像、16×16 非重叠 patch Conv3d。不是纯文本 Qwen3-0.6B 的可用故障；没有视觉路径则拒绝。
四个请求按到达次序 normal/fault/off/health；统一流程为 `advance(internal_id)` →
调度器真实等待完成 → `query()`。实际logits返回回调自动完成本次证据与资源收集；
query仍只读，不运行模型、不推进阶段。宿主不再为17调用collect_replay。

normal 从实际 encoder 调用选择单一投影；F 只把原 Torch Conv3d 参数 padding 从 [0,0,0]
变为 [0,7,7]，原算子执行一次。上下层独立 observer 检查其余参数/完整输入不变、真实输出与
首个消费者变化。OFF/health 都需要重新执行真实 encoder，并精确恢复 encoder/hidden/logits。
公共缓存清理只在无在途请求时执行，不替代 encoder 执行证据。多投影、Native ABI、E5 不在范围内。

`provider_state.evidence` 返回实际阶段、匹配位置、B/F 完整输出身份、消费者变化、logits
变化和恢复。正常 finish→disable→unregister→detach→remove_vision17_task_binding；
失败 abort/recover 后须完成实际 OFF/health 再 clean/unregister/detach/remove。不同任务
必须新 owner/provider，允许同 Worker 完整解绑后建立第二作用域。

本轮 124 Fault8 a2 已跑完首组 normal/F/OFF/health：配对组映射改变 26,592/28,672 个元素，
最大绝对差 2.859619140625，hidden/logits 变化，OFF/health 精确恢复，首 token 未变。
但随后验收器回执超过 16 MiB，整体记失败，第二作用域未跑；不能算普通任务交付通过。
改动收敛观察到一个实际 MoE producer/consumer 链，保持原始大小上限，准备独立重测。

新增 CPU 定向检查：普通 8/17、共享 RMS 生命周期、路由及缓存控制 32 项通过；
17 的实际 Conv/请求上下文加普通路径组合 35 项通过，两次选择有重叠。NPU 待新构建。

## 部署验收准入（RC2 新约束）

`deployment_acceptance.py` 将三种状态分开：`ACCEPTED_DEPLOYMENT`、
`IMPLEMENTED_NOT_ACCEPTED_FOR_DEPLOYMENT`、`NOT_IMPLEMENTED`。能力查询和创建共同
使用该结果。binding/结构候选不等于验收；未匹配组合返回
`ORDINARY_ROUTE_NOT_ACCEPTED_FOR_DEPLOYMENT`。已知组合按实际包版本、CANN安装信息、
模型配置文件身份与当前内存配置、实际模型类、dtype、并行及 eager 条件匹配；各机制的
源码/ABI/权重/真实调用检查仍独立执行。这个记录不授予本次目标资格。

Batch 的新模型即使结构可兼容，也没有自动进入已验收集合；Qwen3.5/3.8、DeepSeek-V4
等必须完成自己的实际验收。CANN 不同版本同样不会自动继承组合验收，这不是将底层
8.x/9.x ABI 适配改回单版本白名单。RC1.1 冻结包不受本次开发变更影响。

可信蓝区验收器可以显式安装 `install_blue_acceptance_permit`：限定现有 Worker/model、
机制、attempt、源码清单 SHA、进程与线程。公共 FaultConfig 没有此参数；许可不绕过
机制兼容/源码/ABI/本次调用检查，也不会把一次试验写成 accepted。结束后先释放 owner，
再移除许可。CPU模拟 binding 仅接受纯CPU参数/缓冲，报告 CPU_FIXTURE_ONLY，不能借此
取得NPU准入。7项新增准入测试与普通路由/8/17/22/RMS生命周期合计33项CPU通过；新门禁
尚待124按实际组合补验。

## 8/17 a3 原始结果与回执大小复核

源提交 `721b1240b66351c50293ce2f9550687b3a975329`，清单
`b10d87d21b6db7c12d8b6bd68f938316ee8e28f19d729b4a6f042192cd18dda5`。
124卡5、CANN8.5.1、TP1/BF16/串行eager，标准vLLM调度器实际执行。
[逐作用域总结](../../experiments/worker_core_v11_20261118/ordinary8_17_20261119_summary.json)
保留请求ID、命中、效果、恢复、原始回执SHA及位置；没有新增统计或完整张量。

- 8：同Worker两个独立任务，共8个真实请求；每个F只选一条实际MoE链。
  输出26,592/28,672元素变化，max_abs_diff=2.859619140625；hidden/logits变化，
  首token/文本未变；OFF/health精确恢复，binding移除且资源为0。
- 17：两个独立任务，共8个真实请求；实际Torch padding [0,0,0]→[0,7,7]，
  输出/首消费者/logits变化；首token未变；OFF/health精确恢复，binding移除且资源为0。
  只验收单投影Torch参数，不是Native launch参数或全目标覆盖。

8的a2超限发生在验收器write(host_receipt.json, client.snapshot())：累计Host历史的
16MiB阈值，未写出该文件。原始单次provider_state未单独保留，不能虚报其精确大小。
a3逐份实际回执复核：单次provider_state最大565,122字节，evidence最大282,111字节；
两个完整Host回执15,950,324与15,951,092字节。17的完整验收文件13,117,415字节主要为
重复RPC历史，单次provider_state最大150,357字节。大小为json.dumps(sort_keys=True)
默认ASCII转义与分隔符后的UTF-8字节数；文件自身缩进大小另外报告。

新增公共边界：evidence最多512KiB，单次provider_state最多2MiB。超限保留原始私有
观察和所有权控制字段，返回ORDINARY_EVIDENCE_LIMIT_EXCEEDED、complete=false、
字节数/SHA及payload_omitted；不截断成成功，不提高旧16MiB验收阈值。
后续注入被阻止；实际abort/clean/unregister仍可释放资源。Host历史累计大小与该边界
独立，长历史可使用已有cursor/stream接口。50项CPU定向检查通过，包括实际RPC超限
后拒绝execute、清理可完成；NPU新增边界验收待下一次8/17回归。

## 17 统一宿主完成接口的修订

旧a3机制/恢复结果保留，但本次接口行为单独补验：普通ActualVision17Request在真实
logits返回、回调计数退出后完成collect/close/校验。正常与失败路径仍要求实际encoder、
上下层Torch参数、真实消费者以及OFF/health；缓存跳过不能代替正常encoder。
CPU56项通过，含两个作用域、失败恢复、缓存跳过拒绝、只读query与第五请求拒绝。
新路线NPU未完成前，17组合标记UNIFIED_RETURN_COMPLETION_NPU_PENDING，默认不可执行；
蓝区新验收器显式安装限定机制的许可。公共宿主无需新增方法或故障特例。
