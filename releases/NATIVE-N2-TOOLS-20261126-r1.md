# 黄区 Native sidecar 与 N2 独立交付索引

**开发线工具，不属于 r8 冻结包。** 由负责人转交黄区；不覆盖r8、任何旧工具包或标签，不修改公共runtime/许可/登记表。没有NPU任务，不中断黄区当前测试。

源码提交：`dfc4b37ced28a483dc7817f25e88f7e3c3f9220a`（开发分支 codex/rc3-r8-2-effects-20261126）。四个核心文件保持57b6ca76版本字节；sidecar依据原r7开发阶段A2构建，三个N2工具只读处理现有原件。

- [ZIP](native-n2-tools-20261126-r1/native-n2-tools-20261126-r1.zip)：**24794字节**，SHA256 **`7d1966a5acdbc03da5df58e693a3775858a7b38e04a2d7e2a629e83c56bd6e8b`**。
- [逐文件 manifest](native-n2-tools-20261126-r1/manifest.json)：SHA256 `91755fd9f2a9f7e2cbeae7c3c20ccf24389c04b7c7e2d2c27e2ca8f271ae1dcd`。
- [DELIVERY-SHA256SUMS](native-n2-tools-20261126-r1/DELIVERY-SHA256SUMS)：SHA256 `dfed2fccd7f25b9dbb93509d16db20808a64cf2ef88df3cf3347ab1729707fb9`。
- [执行说明](native-n2-tools-20261126-r1/README.md)：sidecar放置、两rank身份、N2离线命令和回传范围。
- [独立解压验证](native-n2-tools-20261126-r1/cpu.json)：Python3.12.14、15/15测试通过；CLI两次输出一致，19条JSONL、最长1415字节；修改一个工具字节时校验失败。只有标准库，不依赖本仓库或r8源码。NPU用时0。

包内包括全部依赖、说明、测试、`verify_bundle.py`、manifest和SHA256SUMS。取发行目录后先 `sha256sum -c DELIVERY-SHA256SUMS`；解压后 `sha256sum -c SHA256SUMS`、`python -I -B verify_bundle.py`。不要只下载一个N2文件混用旧同名模块。

|核心文件|SHA256|
|---|---|
|dcat-native-build.json|c61f1a27a78e27d65cb714f786a5f1015d64754550460027097ad0b04a973ea9|
|tools/extract_fault33_launch_host.py|93d0504f82cbbd7a247fe23803dbeef4c7706afcffc50938ff86341ffaa426df|
|tools/extract_fault33_existing.py|2c84f71cae4ef00e6b98e94974e5bd57faa7f2edc5f144aa179a16ab4983606c|
|tools/review_launch_capture_v2.py|66ee176da78971091b2e2108d7f296123bc6abeb5b909862f99a0fb38268743f|

Native仅匹配d9aa0d94…库、Qwen / CANN9.1.0，不能借给CANN9.2.1组合；放置文件后要重新读取两个rank的身份，不能改写旧null回执。N2只完成完整host运输，tiling null、控制重算和资格仍关闭，不作为D/G前置。

## owner定性与r8.1下一步

第二次500已通过冻结r8执行路径及黄区同client说明定性为**情况A**：already_enabled第一次enable即遇到已有其他owner binding。残留来自哪个用例或哪次未完成运行仍需新实例逐用例审计。首次拒绝放大成exit5另记：新实例不再出现则保留r8.2待办；若再出现，凭同次逐RPC证据做幂等清理最小修复，不吞真实恢复失败、不清别人资源。完整说明在包内docs/owner-review.md。

截至本次交付，接收目录最新仍是第二次500的文字纠正；未收到重启后完整负例原件，也未收到实际放置sidecar后的两rank身份。七项expected_ranks草稿保持已有14份身份，negative_controls仍null，运行时注册表不填通过。负例到达即按owner/generation、PID、调用序号、逐rank前后审计、返回码、引擎存活和同引擎正对照审查；合格后填七项，Native三项补齐实际身份及相关证据后加入。从e38c4113开出的r8.1只改登记表，届时再冻结，不发布空登记版。
