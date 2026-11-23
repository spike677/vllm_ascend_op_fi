# r7候选交付：N批、八项产品回归、r6.1维护

标签 `demoncatpfi-rc3-preview-20261123-r7`；源码 `cc9928b5cdffbf6aace73f7b3e3be61c243be082`。保留r5/r6/r6.1冻结物；候选不授予黄区资格。负责人分批转交，蓝区不直接操作黄区。

- [完整发行目录](demoncatpfi-rc3-preview-20261123-r7/)：三个wheel、源码tar、source-review、release-manifest、SHA256SUMS和独立安装检查。
- 版本：demoncatpfi `0.4.0rc6`，worker-extension / fault39-core `0.3.0rc6`。
- 许可填写 **source-review.json** SHA256：`31eee2d6a36e269daf8005a0627d70cd6e85a9c5495a48ec08b1ea1b1cbe4071`。
- [CPU门禁](../docs/worker_core_v1/r7_freeze_cpu_20261123.json)：1163项，1161通过、2项废弃研究入口带理由跳过，无失败。
- [124无设备安装](demoncatpfi-rc3-preview-20261123-r7/install-check.json)：387文件逐字节相同，无Worker、无NPU初始化。外围安装记录校验见DELIVERY-SHA256SUMS。

## 交付范围和实机证据

|范围|候选内容|实际验收组合及限制|
|---|---|---|
|N批|41、26fia、20产品入口|A2/Ascend910B4、0.26/CANN9.1、社区Qwen3.8 W8A8、TP2；同组Worker两作用域、消费者、恢复和资源清零；A3尚待执行|
|原八项回归|18、22、39op、28、17、Batch、39mr、33|同一A2组合，逐项真实产品请求回归；[N批及回归证据](../docs/worker_core_v1/r7_native_a2_20261123/README.md)保留实际窗口和各项范围，不能推广到任意形状/模型|
|r6.1维护|Batch串行B1金丝雀|[新增复核](../docs/worker_core_v1/r61_batch_validation_20261123.json)替代旧并发金丝雀作为新恢复门槛：正常3轮、关闭后2轮；真实故障Batch仍为并发decode|
|保持关闭|15、8产品、DeepSeek相关未验收组合；33 A3|15独立开发不进入本次冻结；33 A3仍先launch观测及控制重算≥99%，不借用A2分组；D/P材料后续单独审查|

没有为了发行重跑未修改的全部NPU矩阵：复用已保留的A2真实回执、Native源码一致性记录；新增Batch协议单独实机重验，其他入口完整CPU回归。该记录不意味着所有实测都来自同一个源码SHA，实际manifest在每项证据中保留。

## N批A3执行单已定稿

[N批A3执行单](../docs/worker_core_v1/r7_native_a3_draft_20261123.md)保留历史文件名，内容为正式候选步骤。先TP2、再TP1；每项同组Worker两个owner。复用已审查的黄区库 `d9aa0d9488c5c1e8f7fca10c297e4968687ceb6a24866d8346bd33a58afce274`，不重建、不用A2回执代替A3。

配置和attach核对A3原名Ascend910_9382、TP、库SHA以及实际映射的ACL/runtime归属于CANN9.1.0。未知或9.2.1拒绝；不能给runner打补丁。主项目安装公开binding，用户仅attach/enable/status/disable/detach，所有rank持续命中。真实区域字节、消费者、资源和新请求金丝雀必须完整；缺许可、未知profile、用户库路径、超范围请求均做负例。

使用本包 `review_product_mode.py --tp 2 --a3-native`（TP1改1）及 `extract_product_record.py`，分页原样回传，原件留黄区。各工具SHA在source-review内。旧工具包r2不替代r7运行时和审查器。

## A组登记与后续

r6.1接线沿用逐条发送金丝雀的说明；D4/D5由新协议复测取代，旧失败原因未定。缺许可负例中的actual_call待观测项本身不污染负例，其他部署字段仍须匹配。

A组登记等待TP2六份完整提取文本、两侧有效负向检查，以及owner—原回执SHA对应表。收到后检查各页规范化SHA、作用域、计数、消费者、恢复与身份一致性；缺页或错误SHA不登记，不能用文字“已通过”替代。当前不新增A组组合登记。

产品无临时许可登记仅有[待审批方案](../docs/worker_core_v1/product_registration_design_20261123.md)，不在r7实现。已登记验收记录也不等于产品模式免许可。
