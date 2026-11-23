# 产品 RPC 契约（r5）

Qwen验收组合：v0.26.0rc1-a3 / CANN9.1.0 / Qwen3.8现役W8A8，TP1与TP2分别登记。蓝区A2/TP2结果不授予A3资格。只支持开启期间全部请求，产品候选五项为18、39 output_proxy、22、28、17。

## CLI与控制通道

CLI主进程须保存一个可信`ProductClient`会话及owner/generation；不可每次CLI调用新建owner。调用跨进程时由主项目已有服务持有会话。HTTP executor把`collective_rpc(method,args,kwargs,timeout)`转为POST `/collective_rpc`；仅可信管理网络可以访问，不向普通推理用户开放。

|CLI|ProductClient|实际Worker调用|
|---|---|---|
|`demoncat fault list`|`capabilities()`|`demoncat_worker_identity`，`demoncat_product_binding` action=`capabilities`|
|`demoncat fault enable`|`attach()`（首次）→`enable()`|binding `attach`、`arm_canary`/`canary`；现有`demoncat_attach`、`demoncat_fault_rpc` prepare/register/check/enable|
|`demoncat fault status`|`status()`|binding `status`，按cursor分页，对齐各rank内部request ID|
|`demoncat fault disable`|`disable()`→`detach()`|既有fault RPC disable/unregister/detach；binding status、reset_canary_observer、arm_canary/canary、最后detach|

`demoncat_product_binding`本身**没有**enable/disable action；这两个CLI动作必须经过ProductClient复用已有生命周期，不能绕过租约。宿主不需要用户请求before/after回调，不给用户请求加标记。部署bootstrap提前配置已审查profile与许可；不得用实验脚本的任意callable bootstrap暴露成公众接口。

```json
{"method":"demoncat_product_binding","args":["status",{"operator":"<trusted principal>","profile":"<reviewed profile>","cursor":0}],"kwargs":{},"timeout":120}
```

通用payload为operator/profile。attach另含scope={"mode":"all"}及ProductClient根据真实Worker census生成的group_contract（owner、generation、members完整身份及SHA、tp、机制、ALL_RANK_EVERY_ELIGIBLE_CALL）。status另含非负cursor；其他未知字段拒绝。主项目不得自行编造内部ID、成员身份或source SHA。

返回值：HTTP层包装`results`为每rank结果；capabilities为PRODUCT_CANDIDATE且product_qualified=false；attach为惰性binding安装结果；status为rank、requests（每页最多1条）、cursor、more、active_request_id、failure。每请求记录覆盖observed/eligible/attempted/applied/changed/consumed/restored/failed、实际site、变化元素、非有限值、范围拒绝理由、内部ID；外部ID不可得时明确UNKNOWN。ProductClient按ID汇总，disable返回区域恢复/资源清零/canary_exact三项，全部成立才允许detach。

主要错误：PRODUCT_RPC_OPERATOR_NOT_AUTHORIZED、PRODUCT_PROFILE_UNKNOWN、PRODUCT_RPC_UNKNOWN_FIELD、PRODUCT_BINDING_NOT_ATTACHED、PRODUCT_STATUS_CURSOR_INVALID、PRODUCT_CANARY_UNSTABLE_IN_SIX、PRODUCT_RANK_INTERNAL_ID_MISMATCH、PRODUCT_RECOVERY_NOT_PROVEN、PRODUCT_CONFIRMED_RECOVERY_REQUIRED。Worker异常由已有HTTP/RPC错误封装返回；不得把异常或空rank结果当作成功。遇RECOVERY_REQUIRED保留owner并走既有recover，不能通过创建下一owner隐藏未恢复状态。

## 金丝雀与A3步骤

1. 安装r5三wheel，校验source-review、安装源码、量化描述和全部权重索引SHA。记录三个调度参数；更换镜像需新组合。专用实例eager、同步调度、无prefix cache/MTP/KV connector，串行max_num_seqs=1、上下文1024、生成1～4token、单chunk。17另用已验收64×64图像路径，不能作为文本项运行。
2. 可信部署固定canary文本与真实服务tokenizer得到的tokens。部署主机通过`/v1/completions`发送文本canary，采样temperature=0、max_tokens=4、min_tokens=4、ignore_eos=true；Worker只在已arm的观察阶段匹配真实token序列。17须用真实processor展开的图文tokens，通过`/v1/chat/completions`发送固定图像canary，不能用未展开文本tokens替代。
3. 开启前最多6次canary、每次600秒，连续两次完整hidden/logits有限且精确稳定。enable完成后发送三个不同用户prompt（17两张不同图片），不调用逐请求RPC；各请求生成4token。再发max_tokens=5的范围负例：推理正常、不注入、记录原因。
4. status读取每请求所有rank覆盖及实际消费者，eligible失败必须0；disable后新canary完整指纹与开启前精确相同，区域恢复和资源零均通过。detach后同组Worker换owner再做独立第二作用域，不重启Worker。
5. 回传产品review summary全文及原回执文件名/字节/SHA、五个身份字段、安装检查；许可不是登记。性能倍数只记录，不作为门禁。正式推理部署预算900秒，预热600秒；HTTP客户端超时应覆盖预算与传输，控制RPC另设超时。验收模式仍用normal/fault/off/health，勿与产品模式混用。

## Native构建（验收模式41/26fia/20）

只在v0.26.0rc1-a3/CANN9.1.0隔离容器构建当前`fault_injection/native_adapters/postwrite`（以发行源码清单实际路径为准），使用`tools/build_native_regions.py --source <postwrite-source> --sdk <actual-cann-sdk> --output <new-owned-output>`；记录library SHA、receipt和编译源SHA。机制名41.tail_column、26.fia_destination、20.buffer_substitute。旧fault26.fia.abi0.v1/fault20.qmm.abi0.v1 recipe不适用，不能用9.2.1构建物替代9.1组合。本命令仅构建和CPU替身测试，不代表A3注入通过。

## 2026-11-23 黄区17启动补充

2026-11-23调整：17验收默认使用内置64×64 RGB图片，像素为 `(4*x,4*y,4*((x+y)%64))`。`--image` 是可选覆盖；产品配置 `canary.image=null` 使用同一内置图，回执记录像素SHA。用户图片仍来自 `/v1/chat/completions` 请求并由实际processor处理，不用金丝雀图替换用户图。r5已通过的黄区固定图片结果仍有效。A3 TP2启动17服务前设置：

```sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

未设置时黄区观测到 `Invalid thread pool`，属于启动环境问题。此补充不修改已冻结r5包；TP1/TP2仍各自验收。完整图像SHA门禁维持原决定，不擅自恢复。
