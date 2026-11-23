# 产品模式登记方案（待负责人批准，不进入r7）

目标：产品组合经过审查登记后，正常使用不需要临时验收许可。当前r5.1只是continuous_acceptance证据登记，r6/r7候选仍使用PRODUCT_ACCEPTANCE_ONLY；本方案不更改这些冻结合同，不自动开放任何组合。实施安排为r7之后的独立版本，负责人批准前只做设计。

## 登记键与证明

每条记录按机制命名变体、实现版本、execution_mode=product、证据/源码版本和适用域建立不可变记录，至少包含：

- TP/world_size（TP1/TP2/TP8分别登记）、DP/PP/EP及实际专家分布/请求归属规则；不以“world_size较小”推断可借用。
- 硬件family和芯片原名、各rank逻辑设备及rank映射。物理设备地址不作为跨实例通用键，原始设备身份仅保留审查证据。
- vLLM/vllm-ascend/torch/torch_npu/Python/CANN实测版本，实际加载源码指纹与必要的ELF/ABI身份；Native机制绑定库SHA、构建收据/recipe和已加载CANN。9.1/9.2不互用。
- 模型config SHA、量化描述SHA、全部权重索引文件名→SHA；只有模型名不够。缺失必需文件不能用空字典或默认值替代。image_digest按既定4.9约定为可选，未观测显式null，不作为隐式通配其他身份。
- 模式、eager/编译配置、同步调度、prefix cache、MTP/KV connector、dtype、量化类型；多模态范围区分文本/视觉。
- 部署参数：max_model_len/max_num_batched_tokens/max_num_seqs，以及prompt长度、生成长度、单chunk、批大小和位置布局的已验收范围。默认使用精确部署参数；运行请求只在显式允许区间内。区间必须来源于实测，不自动把某个点扩展为闭区间。
- 变体参数及其版本（如33的控制/故障分组及≥99%控制证据、15的ND别名语义、Batch的decode批>=2）；用户不能覆盖内部参数。
- 验收模式/产品模式、CLI接线版本、两个owner、所有rank覆盖/消费者/恢复、金丝雀协议版本、负向检查、原件/提取SHA、审查器SHA、审查结论与证据等级。文本证明明确原件保留黄区，不称蓝区原件审查。

通过规范化JSON计算record_sha256。动态pid/thread/owner/request_id是运行租约身份，不进入可复用登记键；每次运行仍验证其一致，不能用稳定登记绕过动态守卫。

## 与现有许可和生命周期的关系

沿用现有prepare/register/check/enable/disable/recover/unregister/detach与多rank合同，不新增状态机。可信部署bootstrap选择两种互斥模式：

1. PRODUCT_ACCEPTANCE_ONLY：候选实测必须持有原有scope许可，许可不能创建/修改登记记录，也不代表正式可用。
2. PRODUCT_REGISTERED（拟议名称）：仅在全部rank的实时身份与发行内已审查product记录匹配时，替代“临时许可存在”这一准入条件；结构/源码、请求范围、消费者、恢复、全实例单活和租约守卫照常执行。普通用户无安装许可或写登记表入口。

不自动由mode=acceptance降级到registered，也不由匹配失败降级到宽松许可；模式选择由可信部署配置明确。故障被disable后仍执行字节恢复、资源清零、部署金丝雀；产品登记不能把恢复失败变成功。

## 拒绝码（拟议）与审查结果

|条件|拟议拒绝码|
|---|---|
|没有该模式/机制/变体的登记|PRODUCT_COMBINATION_NOT_REGISTERED|
|某必需字段未观测|PRODUCT_REGISTRATION_IDENTITY_MISSING（附字段路径）|
|硬件/软件/制品/源码/Native身份漂移|PRODUCT_REGISTRATION_IDENTITY_MISMATCH（附期望/实测差异）|
|跨TP/DP/EP借用|PRODUCT_REGISTRATION_PARALLEL_SCOPE_MISMATCH|
|部署参数超出记录或缺少范围依据|PRODUCT_REGISTRATION_DEPLOYMENT_RANGE_MISMATCH|
|请求超范围|NOT_INJECTED + 原有具体范围原因，计数不伪装eligible成功|
|登记摘要/签入内容损坏|PRODUCT_REGISTRATION_RECORD_INVALID|
|验收模式缺许可|保留当前provider报告中的ORDINARY_ROUTE_NOT_ACCEPTED_FOR_DEPLOYMENT，附缺许可检查项|

公开错误不得暴露机器地址/私有路径。所有rank在必达控制边界汇总，一个rank拒绝则整组不能enable；条件性算子命中处不插入collective。运行中身份改变按现有统一失败恢复路径处理。

## 必做负向测试

CPU和专用实机组合分别验证：未登记组合拒绝；把TP2记录用于TP1/TP8拒绝；每项必需字段逐项null拒绝；量化描述/某个权重索引SHA改变拒绝；额外索引缺失/增删拒绝；CANN或Native库替换拒绝；实际源码改变拒绝；同名模型不同config拒绝；参数边界内/外行为；模式混用拒绝；登记文件篡改拒绝；跨rank身份不一致整组拒绝；恢复失败保留owner。可选image_digest=null必须按4.9规则允许，而非扩大其他字段缺失例外。

正对照必须证明：同一正式登记组合不安装临时许可仍能正常attach/enable/真实请求/disable/detach，两owner、全rank覆盖及恢复成立；验收模式同组合没有许可仍拒绝。缺许可负例与未知profile、未知机制、类型错误分别测试。

## 审查、登记与发行流程

黄区通过主项目CLI按候选版完成产品验收 → 本地审查/提取文本（含完整owner关联及负向检查） → Codex核对canonical SHA、所有字段与结果一致、源码和变体对应 → 缺项/矛盾退补，不登记 → 负责人确认范围 → 从候选发行基线开维护分支只登记获批组合 → CPU身份漂移测试、安装校验、新发行清单和标签 → 黄区独立安装做无临时许可正对照。首版功能本身需要实现与完整门禁；之后仅数据登记可走维护分支。

登记前A3 Qwen TP1/TP2与DeepSeek TP8+EP分别列状态；蓝区A2不能授A3资格。product_closed的前置条件为：13项交付范围中适用项产品证据、黄区自有14项最终状态、CLI单活接线、登记后正式使用闭环全部完成；15开发中、B组未回传、33未过A3控制等缺口如实保留。
