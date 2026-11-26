# 第二owner拒绝放大：r8.2最小候选

> 11-26晚更新：冻结r8的准确ALREADY_INSTALLED→OWNER_MISMATCH→exit5路径已在A2 TP2复现；候选用同探针返回exit4，未向竞争owner发binding rollback，原owner不变并正常恢复。详见 `r8_owner_rollback_a2_20261126.md`。C1“missing_permit回滚成功但binding残留”在冻结r8两次独立实例均未复现，黄区仍缺带owner的审计和中间RPC，不能把本C2修复当成C1根因修复。以下保留CPU候选形成时的证据与实现边界。

分支 `codex/rc3-r8-2-owner-refusal-20261126`，从已交付工具提交fad902e3开出。**仅候选，不冻结、不打新标签、不交付新wheel、不要求黄区替换运行时。** r8.1继续以e38c4113为登记基线；已交付Native/N2工具包原样保留。

## 新证据与归属

负责人本轮附件SHA `4c83cf1c4de3a51eb9feede3c5e6fa4d7fe5c338a3b0370928c3d23653aaf0a0`。黄区 `yellow-a1-fresh-audit-bff9af4b/A1新实例审计与第二次500来源说明_脱敏.txt`，29717字节，SHA `b59cba2e489532c47942d11f9081dd48f9c596370be3cdc80121a4996e81ff1e`。

新实例开跑前rank0/1的leases/owners/bindings/wrappers均为0，随后missing_permit用例后留下binding。此前“可能来自旧实例前几次运行”的候选来源不适用于这次新实例。仍须证明missing_permit实际走直接RPC还是ProductClient：缺少该用例原始reply、client.recovery、cleanup_evidence和逐RPC owner/generation/序号，不能把拟议脚本当成实际执行证据。

后续新owner正对照的路径已经明确：binding attach被拒 → 回滚遇到原owner → PRODUCT_ENABLE_RECOVERY_REQUIRED/exit5。它与上轮情况A并不矛盾：A描述既有残留，B描述新owner拒绝被清理错误放大。此缺陷已有CPU独立复现，下面准备最小候选；实机具体关联仍待补证。

## 修改边界

唯一运行时改动：`src/demoncat/runtime/vllm_ascend/worker_extension/product_host.py`。

在真实binding attach调用处，仅对以下条件记录无事可做的清理：

1. 同次attach返回完整rank集合，全部是typed `PRODUCT_BINDING_REFUSED`，拒绝码全部精确为 `CONTINUOUS_BINDING_ALREADY_INSTALLED`。
2. 该拒绝码在现有 `install_continuous_binding` 中位于创建/写入binding之前。不是把“所有异常”或“没有成功回包”当成未写入证明。
3. 每个子client的本地账本明确为attachment=None、ownership_retained=False；字段缺失、未知或仍有attachment均不走此分支。

满足时：保存原始逐rank拒绝，bound/enabled/pending关闭，不调用drain、Worker detach或binding rollback。原拒绝继续抛出，由现有产品适配返回exit4和原拒绝码。后续公开disable/detach幂等成功，不碰原owner的binding、租约、warmup或异步读取者。

这里 `resources_zero` 只指**本次enable没有创建资源**，不是Worker全局审计。回执增加 `resource_scope=CURRENT_ENABLE_ATTEMPT`、`global_resources_zero=null`，并保持restored=false、canary_exact=null。原owner占用仍然存在，黄区结束审计若非零必须停槽，不能把这个返回当作全局清场成功或产品验收通过。

部分rank成功、丢回包、重复rank、超时、内部错误、其他未知拒绝码仍走原来的全rank恢复路径；清理失败仍exit5并保留责任。attach成功后prepare因缺许可而拒绝，必须正常回滚，不能走无事可做分支。直接Worker rollback的OWNER_MISMATCH保护完全不变。

没有把所有pre-attach拒绝码加入白名单；本候选只处理已证明的ALREADY_INSTALLED路径。也没有修补尚未定性的missing_permit泄漏源。

## CPU复现和回归

冻结r8上先运行新增6项测试：2项失败，均准确复现ALREADY_INSTALLED被OWNER_MISMATCH放大成exit5；其中一项经过真实Worker扩展、product_binding和ContinuousBinding安装守卫，保留原owner的活跃lease作为对照。模型结构校验使用CPU夹具，不宣称模型推理实测。

候选后新增第7项晚阶段错误边界测试；相关7个测试模块共 **43/43通过**，见 `r8_2_owner_refusal_candidate_cpu_20261126.json`。覆盖world_size=1/2共同路径、原owner对象/租约不变、disable/detach幂等、部分成功、未知或不完整回复、账本未知、attach后缺许可回滚、后续阶段同名异常不能误用此分支，以及已有恢复失败重试、bootstrap和persistent产品Host回归。

扩大到所有product测试时，Windows的torch导入在c10.dll依赖处报WinError126，尚未开始执行该扩大测试集，不能写成通过。A2/NPU验证本轮未运行，用时0。本候选不以CPU通过替代最终A2门禁。

## 影响与后续门禁

所有公共/外部产品能力共享ProductClient，因而都经过修改的控制入口；仅完整ALREADY_INSTALLED拒绝场景改变，成功attach/金丝雀/注入/正常disable与机制代码不变。Host文件属于implementation_identity取样范围，未来r8.2身份SHA会改变，不能冒用r8.1的旧实现身份登记。

定稿前请黄区保存同次运行：标签与加载源码SHA、Worker PID/rank、两个owner及generation、每次RPC序号/动作/返回体、首次enable异常（不要只留下finally的覆盖异常）、client.recovery、cleanup_evidence、前后审计及同引擎正对照。RPC序号在宿主日志记录，不向不支持该字段的公共RPC payload硬塞参数。

若missing_permit实际走ProductClient且自动回滚确有泄漏，再按原件补修；若走直接RPC，黄区修自己的生命周期。禁止用私有 `_binding('rollback')` 在finally掩盖缺陷。证据对应后再做A2 TP2全拒绝/部分成功/未知回复与缺许可的真实控制回归，原owner不受损、引擎存活；之后才决定r8.2冻结。r8.1仍只修改登记表。
