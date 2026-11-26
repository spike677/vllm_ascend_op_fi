# RC3 A3 候选验收契约（准备稿，未发行）

此文交负责人保留，未向黄区发送。RC2.1/RC2.2 标签和包不变。蓝区已通过的组合不能直接登记为 A3；候选包尚未冻结，以下接口存在不代表 A3 通过。

## 范围和阻塞

| 能力 | 当前蓝区依据 | A3 专用实例待验证 |
|---|---|---|
|18.api_half|A2、0.26.0+empty/Ascend0.26.0rc1/CANN9.1、社区Qwen3.8 W8A8、预热后TP2两个作用域|现役制品、A3源码身份、TP1及TP2；当前只验独立AscendGemmaRMSNorm返回，不宣称融合Q/K RMS覆盖|
|22.wrong_row|同组合，真实融合mRoPE目标Q/K末行变化，TP2双作用域恢复|现役A3 TP1/TP2；合法错行机制，不宣称整数溢出或所有层覆盖|
|39.output_proxy|社区out_proj量化，不符合指定BF16 MatMulV2|需现役实际BF16 out_proj和新栈Native接线；本候选不执行39|
|Batch/17/41|新栈第二批仍待适配；Batch已完成独立V2算子和group库构建检查|不得沿用0.18或算子级资格|

## 两套独立实例

TP1与TP2分别使用v0.26.0rc1-a3专用实例，记录实际image digest、芯片名、CANN运行时及文件来源、完整Python包版本、权重config和quant描述SHA。目标参数：eager、async_scheduling=False、prefix_cache=False、MTP=None、KV connector=None、image/video上限均0、max_num_seqs=1、max_model_len=1024、max_num_batched_tokens=1024、chunked_prefill=True。实际调度仍必须证明单请求一个chunk，不能只看预算。两作用域期间不得重启Worker或重新加载模型。

先核对实际源码，profile `v026_installed_c9c9432b` 是固定Python源码字节集合，不是A3镜像资格。A3字节不匹配时停止并回传源码身份，不改runner属性、不把缺字段填1、不覆盖旧profile。身份相同也仍需A3独立许可与验收。

每套实例先运行三个无注入、同输入fresh prefill；要求最后两次完整hidden/logits精确相同。冷启动首轮单独记录，不把它并入故障归因；不满足则停止。当前blue probe是TP2且安装BLUE许可，**不能原样在黄区运行或改标签伪装成A3**。

## 可信部署安装接口

所有函数在Worker进程及其实际执行线程上调用。使用已有部署认证回调和 `configure_deployment_acceptance(worker, source='YELLOW_DEDICATED_INSTANCE', authorize_operator=...)`；回调由可信控制面提供，不能用用户传入函数或无条件True。许可通过 `install_deployment_acceptance_permit` 安装，填写实际operator、owner/attempt、mechanism和候选源码清单SHA。它仅允许测试，不创建已验收登记。

| 项 | 实际函数和必要参数 |
|---|---|
|18 binding|`new_stack_rms18.install_rms18_newstack_binding(worker, prompt_token_ids=..., weight_manifest=..., workload_sha256=..., environment_sha256=..., group_contract=...)`|
|22 binding|`new_stack_rms18.install_rope22_newstack_binding`，参数同上|
|TP1|group_contract=None；`HostAdapter` + `WorkerFaultClient`，只允许唯一实际Worker，不从用户配置选择rank|
|TP2|`MultiRankTaskClient`生成当前group_contract；每rank各安装binding；normal后由协调器提交角色，18用`commit_newstack_rms18_role`、22用`commit_newstack_rope22_role`|
|移除binding|`serving_tensor_binding.remove_tensor_serving_binding(worker, mechanism)`，必须租约已清零|
|移除许可|`deployment_authorization.remove_deployment_acceptance_permit(worker, authority=..., operator=...)`，不得有残留owner|

weight_manifest来自实际制品，不填社区权重SHA代替现役。新栈binding只接受已固定源码与token-only路径；用户不填层名/rank。18自动选择首个实际独立RMS返回；22自动选择首个实际full-attention融合调用。完整融合消费者与非目标行守卫必须发生。

## request_plan 与两作用域

每个作用域新owner/generation；同组Worker不变。prepare → register → check → enable后，读取query的request_plan与next_phase，按返回计划循环，不硬编码请求数。当前18/22计划为normal/fault/off/health。

每轮由宿主：确认队列空闲 → `sleep(level=0, mode='wait')`并确认暂停 → enqueue固定prompt → 从实际output_processor提取内部ID → advance(内部ID) → `wake_up(tags=['scheduling'])` → 等待真实完成 → query。这是暂停调度，不卸载权重；调用binding或advance不得代跑模型。内部/外部ID及实际request_state必须一致。复用同一prompt时每轮仍需fresh内部ID及独立KV状态。

complete且没有failure后：finish → disable → unregister → detach → 移除binding → 移除许可，记录所有rank的hooks/wrappers/leases/inflight为0。失败进入统一recover；需要OFF/health时使用新请求完成恢复，不把容器退出当成精确恢复，不自动重试F。

## 效果和负向检查

18：注入rank的实际RMS返回减半及真实消费者；guard rank本地返回不变；分别记录每rank完整hidden/logits变化。22：原QKV/权重/positions/cache不变，目标Q/K行变化，非目标Q/K与V/gate不变，真实Attention消费融合返回；guard未修改。两项OFF/health全部rank完整精确恢复，token是否变化单独报告，不作为通过条件。

无许可、错误源码身份、外来Worker/owner、重复请求ID、视觉参与或partial chunk须拒绝；仅在空闲负向检查，不修改现役模型来构造错误。CPU已有反例不能代替A3实际准入检查。

每项结束回传summary全文，包含：来源YELLOW_DEDICATED_INSTANCE、证据等级YELLOW_TEXT_ATTESTED、真实硬件/软件/模型/TP、源码和包SHA、两作用域ID、实际选点、目标/guard效果、OFF/health、资源清零、请求阶段与映射、安装检查。22必须包含Q/K或rotary行效果字段，不只回传hidden/logits。原件留黄区，附每份文件名/字节数/SHA。审查通过后再登记与发行RC3。

尚缺的交付件：A3实际源码身份审查、正式可信binding RPC封装、TP1候选宿主执行器、A3候选发行清单与安装检查。本文不把这些写成已提供；不要求负责人现在转发运行。
