# RC2 普通任务建设（开发中，尚未冻结发行）

RC1.1 已单独发布；这里的开发改动不回写 RC1.1 包。蓝区测试部署与黄区现役部署分别验收。

## 公共约束

`ordinary_admission.ordinary_route_report` 是能力查询和创建共同的普通路由判定。
缺绑定显示 pending，实际调用始终待观测；候选/绑定不授予执行资格。当前普通入口为
18、39、Batch 和待实机验收的 8、17。其他八项中的未接入部分返回明确
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

## Fault8（CPU 已验证，124 待验）

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
故障已开始但未恢复时不能清掉所有权。CPU 已验证同 Worker 的两个独立作用域；NPU
验收未完成前，能力表不能写成蓝区通过。

共享普通请求实现新增 evidence 别名；失败前的原始异常先保存。成功 abort 释放且未实际
注入时，历史失败保留为报告，不再误当活跃失败阻塞 clean。RMS 14 项回归通过。

## 进展与门槛

| 能力 | 当前普通任务进度 | 蓝区/黄区测试/黄区现役 |
|---|---|---|
|18|既有四请求；维护公共结果/清理|既有 Source126；用户转述黄区两作用域通过；现役未验|
|39 output_proxy|既有四请求；RC1.1 诊断与路由修复|既有 Source116；用户转述黄区两作用域通过；现役未验|
|Batch|既有六批；结构而非模型名判定|既有 Source118；黄区缺 W8A8/bundle；现役未验|
|8|新四请求 CPU 及真实 producer 拒绝测试通过|新普通入口 NPU 待验；黄区未验|
|20、22、26 两变体、33、41|历史机制保留，普通任务待接入|不可沿用历史机制结果宣称新普通入口通过|
|28 denominator|正常关联与 opt-in Execute 借用已保留|F/OFF 未完成，关闭|

定向检查：前提/结构/统计边界 56 项通过；CPU 构建身份与 39 52 项通过；8/RMS/MoE
复用 20 项通过；公共路由与四种普通路径 75 项通过。都是 CPU 检查，部分选择有重叠，
不能相加作为唯一测试数量。基线中的历史冻结 SHA 检查尚未重写或冒充当前验收。

## Fault17（普通接口 CPU 已通过，NPU 待验）

`install_vision17_task_binding` 绑定已验证来源的 Qwen3-VL-2B、TP1 BF16、串行 eager，
单图像、16×16 非重叠 patch Conv3d。不是纯文本 Qwen3-0.6B 的可用故障；没有视觉路径则拒绝。
四个请求按到达次序 normal/fault/off/health；每次 `advance(internal_id)` 后真实等待完成，再
`collect_replay()` 收集该次结果（此名称是既有 Host API；不会执行重放）。query 不代替 collect。

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
