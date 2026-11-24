# r7.3 交接补充：最小复测与7项外部接线

本补充发布在开发文档，不修改冻结包。r7.3已发布：源码d010fb78，发行1b816dd6，标签demoncatpfi-rc3-preview-20261124-r7.3；原任务书“未打标签/推送”为过期状态。
开源main已按负责人授权先快进7ffd9a60再到1b816dd6；黄区master不能据此认为已经升级，仍需回传审查、黄区CLI冒烟。

本轮再次核对远端main、r7.3分支与标签解引用均为1b816dd65cd81d4231036919d07831dfbd96968e。发行目录DELIVERY-SHA256SUMS的293个文件逐项核验无不符，清单本身SHA256为d663a78c8dab927ef82c6dd0391b021f5fd5d331c7fa5be1a24cfd0b822d27a6。标签包含从d010fb78构建的发行物，不将仅源码提交误作含包的发行提交。

## source-delta路径审查

依据r7.3发行source-delta.json。三个机制目录316文件完全一致，但共享Worker接线发生变化。
全部公共项均通过product_binding（配置/attach）、product_host（跨rank身份）、registry/runtime（注册/能力）、continuous_binding（site绑定）、product_request（请求/forward收尾）。
内置机制不进入external.factory或sampler分支，但该分支的新增位于共享请求处理函数中，必须复测共同控制/消费/恢复路径。

|公共项|实际路径/原因|A3 TP2要求|
|---|---|---|
|18|Torch norm hook；共享model forward收尾|两个owner全闭环|
|22|融合rope；共享请求映射/收尾|两个owner全闭环|
|39op|投影返回与消费者；共享binding/request|两个owner全闭环|
|39mr|投影重算；共享binding/request|两个owner全闭环|
|28|attention重算；共享binding/request|两个owner全闭环|
|17|视觉结构与请求门禁；共享binding/request|两个owner含内置图/用户图范围|
|Batch|并发decode映射；共享binding/request|两个owner并发正例及B1负例|
|15|ND别名改向与消费者；共享binding/request|两个owner，changed计调用、changed_elements单列|
|41|Native KV写入；共享租约/恢复|两个owner，匹配已审查Native库|
|26fia|Native residual目的缓冲区；共享租约/恢复|两个owner，匹配已审查Native库|
|20|Native输入覆盖；共享独占消费/恢复|两个owner，匹配已审查Native库|

因此最小集为11项全部（含强制18、41、Batch、15），没有完全绕过本次Worker接线且能免复测的公共项。
r7.2通过证据保留，不作废；r7.3复测作为新包接线回归，不自动授予新范围。TP1不是必要条件，不再要求TP1 Batch复测。

## 七项黄区适配

|黄区能力|v1 stage|必须证明的边界|
|---|---|---|
|activation/faulty_requant|model|瞬态激活重数量化，实际消费者读取且可恢复|
|operator_input_precision|model|真实算子输入区，独占消费窗口，布局/dtype合法|
|value_inject|model|真实瞬态张量区，不能指向Parameter/cache|
|operator_output_value|model|返回区的真实消费者与同stream读取完成|
|fake_requant|model|瞬态Q/scale及真实QMM消费；持久scale不属于此项|
|activation/param_perturb|model|名称不代表可写持久参数；仅实际瞬态激活允许，Parameter拒绝|
|exponential_perturb|sampler|真实probs.div_(q)消费的噪声q，正温度/已审查源码，设备读取后恢复|

这些是适配边界，不是蓝区替黄区实现或验收七种机制；只有示例已在A2验证。
参考external_provider_r73.md和发行附带external_gain_example.py / external_sampling.ExponentialNoiseSite源码。

逐rank可信接线顺序：审核适配器文件SHA、实际模型/runner/软件库身份及source-review许可身份→注册实际factory对象ExternalSite（stage明确、region_kind=transient）→configure_product_profiles(mode=PRODUCT_ACCEPTANCE_ONLY)→现有可信安装许可→ProductClient.attach/capabilities/enable→真实HTTP请求→status→disable（drain、区域恢复、资源零、金丝雀）→detach→移除许可。
每rank使用同一审查SHA；用户不能提交Python路径/回调/任意数值/rank。黄区将命名机制映射到可信会话，再使用python -m demoncat.cli fault list/enable/status/disable --socket "$SOCKET"，enable另传--fault-id；不是公共包自动注册黄区CLI。
持久weight_scale、int8参数、KV区域返回EXTERNAL_V1_PERSISTENT_WEIGHT_OR_KV_RESTORE_UNPROVEN；不得把region_kind虚报transient绕过。
跨rank覆盖八项计数、消费者、区域字节恢复、资源零、金丝雀、真实owner/调用序号、三负例全部保留。

33仍需独立A3取证，不纳入11项放行；V4和8暂缓验收（负责人决定）。本补充仅交负责人转交。
