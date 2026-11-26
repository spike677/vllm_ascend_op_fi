# 11月26日追加审查：身份、负例和第二owner

> 后续新实例证据更新：初始全rank审计零，残留发生在当前missing_permit期间，不再沿用旧实例残留来源推断。新owner拒绝被放大成exit5已由CPU独立复现并准备r8.2候选；同次实机owner/generation/RPC序号及missing_permit实际路径仍待补。最新结论见 `r8_2_owner_refusal_candidate_20261126.md`、`yellow_three_slot_review_20261126.md`。不修改已交付工具包中的历史文档副本。

依据：`yellow-r8-negative-review-correction-e2931928/r8负例清理问题与15_22结论更正_原文.txt`，12871字节，SHA `a77349c88ac3137144eb6556f485f18d714a7e91989626689ae0a91c5b455929`；rank0沿用已存档 `yellow-r8-review-progress-b4133357` 文本。组合仅 A3 Ascend910_9382 / Qwen3.8 W8A8 / 0.26 / CANN9.1.0 / TP2。

## 登记草稿

`r8_1_expected_ranks_draft_20261126.json` 已填18.api_half、22、15、39op、39mr、28、17共七项的 expected_ranks。六项复用负责人提供的同部署文本身份，17使用独立视觉身份；不是从rank0改一个rank号生成rank1。14份身份逐个通过 validate_identity；跨rank公共字段一致，rank/逻辑设备分别保留实际值。

每项 evidence.negative_controls 仍为 null、executable=false，草稿不能通过 validate_record，且不在运行时加载位置。维护分支仍从 r8发行 e38c4113 起步，无登记表改动。补齐负例和其余证据后才可冻结r8.1；不能由“身份齐”推成“已登记”。17效果/产品原件的请求与图像关联仍按此前待补项保留。

15页面 SHA 已更正为 `5db767de906cbe0acd535e95cc0dec586e9e85f06f210f20fcb51459945ef6d2`；22页面 SHA 更正为 `b46142ed48fbbe828ad87c2d6c13fffdc29026d0ef4ee10075b20f7aa294c4a6`，与先前规范化核对一致。over4仍未达到产品流程完成条件，不进入这七项。

Native 的 d9aa 构建元数据已生成，见 `native_build_metadata_d9aa_20261126/README.md`。这是可核对的原A2构建身份，不替换黄区旧回执中的 null；41/26fia/20仍需放置sidecar后双rank实读，负例合格后才登记。

## 第二owner：情况A已定性，拒绝被放大的问题单独保留

追加材料：黄区 `r8负例第二次500定位与Batch更正_原文.txt`，5347字节、SHA `34fdadee334dcbaf24127445441b750139b577c8388839dd8d8c544d734175c6`；负责人本轮裁定原文34269字节、SHA `178f7ccdb748f74de4a1b341a7e6d738b6e6b53be72d05cb1e40cb8b4adff78f`。黄区确认 already_enabled 分支使用同一个 client。

复核冻结r8的 product_host.py（SHA `eed69a5853386405fb84f966271f01c3baf89893b03b9a6c9ed90b007fd23a9a`）：enable 的重复调用守卫在 try 前，直接拒绝 PRODUCT_NEW_ATTACHMENT_REQUIRED，不调用Worker、不设置待回滚标志。首次 _enable 在binding attach前设置 bound=True；其后失败会置 _enable_failure_pending，回滚成功才清除此标志。disable 的 _rollback_enable 分支说明前一次enable及其回滚未完成，而不是成功enable后的重复调用。

结合本次原始栈中的其他owner拒绝及既有binding，**第二次500定性为情况A：用例首次enable前已存在其他owner的binding，尚未走到重复enable。** 不再把A/B本身列为待判定。各rank错误码不同不能单独证明泄漏发生在哪个rank或哪一步；TypeError、remove_permit失败或之前用例中途退出只是来源候选，须新实例逐用例审计定位。

残留及负例适配器清理由黄区负责。第一次enable被拒后，回滚碰到其他owner，最终变成exit5的问题由codex单独跟踪。本轮不修改公共runtime，不放宽remove_permit、不强制移除其他owner的binding。

需同一次运行的最小证据：

1. owner0/owner1、generation、Worker pid、rank及每次RPC调用序号；第二owner enable前的binding归属、enabled/lease/inflight状态。
2. 第二owner enable的结构化拒绝及它是否实际创建过自己的任何资源。
3. 后续回滚目标owner/generation、调用序号、逐rank返回体和CLI退出码；不能混接两次运行的日志。
4. 回滚后owner0的原binding/租约未受损、引擎存活；按owner0自己的关闭流程能完成恢复。

新实例若不再出现首次enable拒绝后exit5，登记为r8.2待办，不阻塞合格的本轮负例登记；若再出现，按上述同次运行逐RPC证据做最小修复：确实没有本owner资源可清理时幂等成功，保留最初契约拒绝码、CLI exit4，不回滚别人的资源。部分rank已创建本owner资源时仍必须清理并证明恢复，不能把真实恢复失败吞成exit4。修复须补CPU和受影响A2回归，不能塞进只改登记表的r8.1。未提供完整证据的负例不记通过。

## 当前负例结论

文字写“缺许可、未知项通过”尚不能替代逐rank返回体、exit4、资源清零、引擎存活和后续正对照原件。三项负例完整证据仍待回传；“remove_permit严格拒绝”自身不是公共缺陷。Batch同组成冷启动不稳定的蓝区新证据见 `r8_2_batch_no_injection_20261126/README.md`，不据此注册黄区Batch。
