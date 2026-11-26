# r8 黄区后续文本审查（11月26日17时批次）

组合仍为 A3 / Ascend910_9382 / 0.26.0rc1-a3 / CANN9.1.0 / 现役Qwen3.8 W8A8 / TP2。来源为 `yellow-r8-review-progress-b4133357` 和 `yellow-r8-negatives-cleanup-guard` 两份原文；原件留在仓库外。前者30280字节，SHA `e97bcd4d01dc5af871e0fe4fed61881b9a90ccf218a30ecd34f71e37367fb271`；后者3411字节，SHA `05f573e2731cf91b535b7079fb39a09375404f91a5d49ee60f8b9a7adbc4ac22`。逐页计算结果见同名JSON。

## 审查结果

| 项目 | 结论 | 登记前还缺什么 |
|---|---|---|
| 22.wrong_row | 页面verify与产品提取哈希均通过；两个owner、两rank各192次，failed=0；top5变化、OFF与三项恢复通过 | rank1完整身份、有效负例、文件SHA表勘误 |
| 15.alias_detach | 页面verify与产品提取哈希均通过；两个owner、两rank各192次，failed=0；OFF与三项恢复通过 | 同上；仍标B档ND别名断开，不声明私有格式条件 |
| 18.variance_over4 | 效果页面verify通过；产品提取哈希不符，且内容明确 `review_passed=false / PRODUCT_RUN_INCOMPLETE` | 黄区补CLI映射后走真实CLI产品流程；重新运行提取并原样回传，不能修改哈希字段强行通过 |

over4 的CLI缺项属黄区接线，当前不能凭效果页覆盖产品流程失败，也不能登记。其产品提取声明SHA `24f19498…` 与回传JSON规范化计算结果不符，属于额外的文本完整性问题。

文件SHA表混淆了原件SHA、页面文件SHA和extract_sha256：22把effect原件SHA当页面SHA；15页面SHA为`?`；over4标题还漏掉原件SHA开头的`5`。本地从收到的页面JSON按工具的 canonical + LF 计算出：

| 页 | 字节数 | 本地重建 canonical + LF 的SHA（不是黄区文件实测） |
|---|---:|---|
|22|2395|b46142ed48fbbe828ad87c2d6c13fffdc29026d0ef4ee10075b20f7aa294c4a6|
|15|2402|5db767de906cbe0acd535e95cc0dec586e9e85f06f210f20fcb51459945ef6d2|
|over4|2437|22f3de5597d94fbdbab7e08eaaa617735b7dae654b0561d2c1be962aa5445fca|

这些值用于黄区核对文件表，不能替代黄区对真实文件运行sha256sum。三页内部extract_sha256都已通过；不因标题笔误否定页面内部完整性，但登记证据表必须消除矛盾。

## 身份与剩余门禁

三份完整身份全部是rank0/逻辑设备0。文本与17视觉身份通过 `validate_identity` 字段校验；Native身份明确缺 `build_metadata_sha256` 和 `build_metadata`，校验返回 `NATIVE_BUILD_REGISTRATION_PENDING`。不能复制rank0并改rank号填rank1；Native需补真实库对应的既有构建元数据及两rank观测。

标签、许可、工具与审查器SHA已给出，和冻结r8匹配。耗时仅为文件时间推估，记录而不作性能结论。外部Conv3d已说明运行在视觉实例，但选择了文本身份文件，属黄区身份采集接线错误；外部项继续仅受控许可，不进入公共免许可表。

目前累计14份效果页校验通过，其中11个公共页、3个外部页；公共over4的产品流程仍失败。**完整登记通过数仍为0**：缺有效负例及rank1身份，Native另缺构建身份。冻结r8、r8.1登记分支和加载中的注册表均未变更。

## 负例清理问题

本轮missing_permit错误链为黄区适配器在未安装许可时无条件remove，使 `BLUE_ACCEPTANCE_PERMIT_NOT_BOUND` 覆盖原始负例结果。公开契约的remove对不存在的许可明确拒绝；不修改公共运行时把未知许可清理伪装成成功。

黄区自行修复适配器时应保留以下契约：

1. 保留最初attach/enable拒绝和后续cleanup异常两个字段，后者不能覆盖前者。
2. 许可安装/移除按实际每rank结果及本次所有权记录；仅移除本次确实安装的许可。部分rank安装成功而宿主RPC失败时仍需核查、清理已成功rank，不能只靠一个宿主布尔值判断全组未安装。
3. `client.recovery is None` 不得直接下标读取；`not client.bound` 也不是资源清零证据。负例须实际查询所有rank的owner/租约/hook/在途请求，确认引擎存活，并完成同引擎正对照。
4. 恢复不确定时保留recovery_required，停止新流量；不能为跑完负例强制detach或吞掉清理错误。

该回传尚无有效exit4、拒绝码、存活、资源零和同引擎正对照整套证据，故仍是“负例未完成”。按既定分工，不替黄区改适配器，不新增补充包。

## Batch与33

黄区Batch的scope1文字称组成相同且OFF精确，但scope0仍失败；文字列出的baseline frame0 batch7与off frame4 B1不构成同次完整调度轨迹。因此不据此推出scope0是相同组成，也不将其改判通过。蓝区r8.2已增加逐rank、逐轮实际调度记录，另行完成A2复现与恢复验证。

33三形状×两rank的六条采集文字报告已齐，128/512 key为5000000000010200203、block_dim24；2304 key为5100000000010200203、block_dim10。采集完整性与tiling解释是两道门：当前六行仍报 `TILING_BOUNDARY_UNPROVEN`，`serializer_validated=false`，故N2未完整，不能进入控制重算或登记。不用key或blockDim反填未知tiling字段；保留现有原件等待ABI/边界证明。
