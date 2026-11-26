# 11月26日追加审查：身份、负例和第二owner

依据：`yellow-r8-negative-review-correction-e2931928/r8负例清理问题与15_22结论更正_原文.txt`，12871字节，SHA `a77349c88ac3137144eb6556f485f18d714a7e91989626689ae0a91c5b455929`；rank0沿用已存档 `yellow-r8-review-progress-b4133357` 文本。组合仅 A3 Ascend910_9382 / Qwen3.8 W8A8 / 0.26 / CANN9.1.0 / TP2。

## 登记草稿

`r8_1_expected_ranks_draft_20261126.json` 已填18.api_half、22、15、39op、39mr、28、17共七项的 expected_ranks。六项复用负责人提供的同部署文本身份，17使用独立视觉身份；不是从rank0改一个rank号生成rank1。14份身份逐个通过 validate_identity；跨rank公共字段一致，rank/逻辑设备分别保留实际值。

每项 evidence.negative_controls 仍为 null、executable=false，草稿不能通过 validate_record，且不在运行时加载位置。维护分支仍从 r8发行 e38c4113 起步，无登记表改动。补齐负例和其余证据后才可冻结r8.1；不能由“身份齐”推成“已登记”。17效果/产品原件的请求与图像关联仍按此前待补项保留。

15页面 SHA 已更正为 `5db767de906cbe0acd535e95cc0dec586e9e85f06f210f20fcb51459945ef6d2`；22页面 SHA 更正为 `b46142ed48fbbe828ad87c2d6c13fffdc29026d0ef4ee10075b20f7aa294c4a6`，与先前规范化核对一致。over4仍未达到产品流程完成条件，不进入这七项。

Native 的 d9aa 构建元数据已生成，见 `native_build_metadata_d9aa_20261126/README.md`。这是可核对的原A2构建身份，不替换黄区旧回执中的 null；41/26fia/20仍需放置sidecar后双rank实读，负例合格后才登记。

## 第二owner：等待情况B证据，不改运行时

已收到文字栈：rank0 `PRODUCT_ROLLBACK_OWNER_MISMATCH`、rank1 `CONTINUOUS_BINDING_ALREADY_INSTALLED`，经 ProductClient.disable → _rollback_enable → binding rollback。它说明错误被放大，但**不足以确认**究竟是前一scope未清理的情况A，还是前一owner合法占用、第二owner被拒后错误回滚他人资源的情况B。原文对旧binding的解释不是独立证明。

按负责人决定，本轮不立项修改公共runtime，不放宽 remove_permit、不把其他owner的binding强制移除。黄区负例适配器先按既定生命周期自行修正。

需同一次运行的最小证据：

1. owner0/owner1、generation、Worker pid、rank及每次RPC调用序号；第二owner enable前的binding归属、enabled/lease/inflight状态。
2. 第二owner enable的结构化拒绝及它是否实际创建过自己的任何资源。
3. 后续回滚目标owner/generation、调用序号、逐rank返回体和CLI退出码；不能混接两次运行的日志。
4. 回滚后owner0的原binding/租约未受损、引擎存活；按owner0自己的关闭流程能完成恢复。

若这些确认B：再做最小修复，使未创建本owner资源时回滚幂等成功，保留enable的契约拒绝码、CLI exit4；绝不回滚owner0的资源。若确认A：由黄区修复scope0关闭/移除顺序。情况未明不将exit5当合格负例，也不提前填写negative_controls=true。

## 当前负例结论

文字写“缺许可、未知项通过”尚不能替代逐rank返回体、exit4、资源清零、引擎存活和后续正对照原件。三项负例完整证据仍待回传；“remove_permit严格拒绝”自身不是公共缺陷。Batch同组成冷启动不稳定的蓝区新证据见 `r8_2_batch_no_injection_20261126/README.md`，不据此注册黄区Batch。
