# RC3预览r7.1交付（2026-11-23）

r7.1 = r7 + Fault15 B档产品候选。旧r7/r6/r6.1标签和包均未修改。今晚只Qwen3.8 A3 TP2，由负责人转交；不启动TP1、DeepSeek或33 A3注入。

- 源码：`3f242e0453235dbdbc2ebbb9b35786694b3ae947`；发行标签：`demoncatpfi-rc3-preview-20261123-r7.1`。
- 三wheel版本：demoncatpfi 0.4.0rc6.post1；Worker/Core 0.3.0rc6.post1。
- 许可source_manifest_sha256填本包 **source-review.json** 的SHA：`77e30525212d2d1e8f46313452bacdd46c81ecb751ce2e052e4d72dabbf6f7e2`，不是release-manifest SHA，不沿用r7。
- CPU门禁1169项：1167通过，2项废弃研究入口按既有理由跳过；零失败/错误。
- 124无设备安装：388文件逐字节一致，无Worker/NPU初始化、无网络。见包内install-check.json/execution.json。
- 新A2 TP2模型回归267.825740秒：15两个owner、每owner每rank192次匹配，共768次；真实消费者、字节恢复、活动资源清零、OFF/health完整hidden/logits精确恢复均通过。333个实测runtime文件与发布源码一致。隔离已解除。

## 文件交付

目录：`releases/demoncatpfi-rc3-preview-20261123-r7.1/`。三个wheel、源码tar.gz、source-review.json、release-manifest.json、SHA256SUMS、无设备安装回执、r7-file-comparison.json、DELIVERY-SHA256SUMS均需保留。

|文件|字节|SHA256|
|---|---|---|
|demoncatpfi-rc3-preview-20261123-r7.1.tar.gz|3161922|`4aecf8549f820e8df942a98ff344d25fa4df267f63f9168afbd536fec9799c84`|
|demoncat_fault39_core-0.3.0rc6.post1-py3-none-any.whl|49186|`947ddad561fe691e810e7f03ec8af9f43ad16ea26cb061fed1cc069607a20b90`|
|demoncat_worker_extension-0.3.0rc6.post1-py3-none-any.whl|607563|`0b9f11982d5b3d1acb03b50125319f1d1b87101cd47cd86c28088936d954f21e`|
|demoncatpfi-0.4.0rc6.post1-py3-none-any.whl|1045343|`f9aa5da6c7041dc5d640e0ddf04a8b43ca8ba28f5f9598f5378a4a7d503cda56`|

```sh
cd "$RELEASE"
sha256sum -c SHA256SUMS
sha256sum -c DELIVERY-SHA256SUMS
python -m pip install --no-index --no-deps --target "$PREVIEW_SITE" wheels/*.whl
python "$SOURCE/tools/check_rc3_preview_install.py" --site "$PREVIEW_SITE" --release "$RELEASE"
```

源码先在新目录解压本包tar.gz，SOURCE指向解压根目录；服务PYTHONPATH和wheel必须来自同一版本，独立安装不覆盖旧冻结环境。使用已有服务的受信扩展和session daemon。

## 与r7逐文件比较

包内r7-file-comparison.json可重算：`python tools/compare_r71_r7.py --head 3f242e0453235dbdbc2ebbb9b35786694b3ae947 --output /tmp/r71-comparison.json`。它比较不可变r7提交c50efb81与本源码的Git blob/文件模式，列出每个变更文件双方字节数/SHA256。

- 3124个受跟踪文件完全相同；原有24个site实现全部未改。
- 28个变更文件明细随包。15新增site、release_unmodified、共享候选/工厂、15专属准入及回执接线是必要实现改动；版本元数据、测试、比较器、执行单与证据另列。
- 因共享接线、版本号和文档必然变化，**不能声称除新site外所有文件完全相同**。能证明的是：未列出的文件逐字节一致，原有机制site和Native源码未改；所有例外公开，不以文档结论掩盖差异。
- r7既有机制实机结果在此未改范围内沿用；15有本次新回归，不把A2结果记作A3通过。

## 黄区执行

1. [TP2隔夜总单](../docs/worker_core_v1/tp2_overnight_runbook_20261123.md)：所有线参数、可信许可、两owner、串行金丝雀、三项负例、review/extract。
2. [15 A3 TP2执行单](../docs/worker_core_v1/alias15_a3_execution_20261123.md)：语义为**ND视图别名断开，不含私有格式触发条件**。黄区CLI须显式补入`15.alias_detach`映射；r7.1不会自动改黄区cli.py。入口是`dcatpfi`，绑定本线socket，不能给用户层/rank/裸地址参数。
3. [33快速判定](../docs/worker_core_v1/paged33_a3_fastlane_20261123.md)：两rank实际launch/分组/控制≥99%审查前A3仍关闭。满足后另准备r7.2候选，不改本包。

15 A3运行时仍核验实际芯片、TP2、CANN9.1映射文件、源码、BF16 ND活跃写入/消费链；普通产品资格不自动开放。8对Qwen不适用；DeepSeek产品仍关闭。黄区提取原样文本到达后逐份判“通过/不通过/需退补”，不要求传原件、不补缺失身份。

## 记录限制

性能倍数未测不推算，不作门禁。实机服务退出仍有第三方resource_tracker信号量/共享内存告警，记录在窗口摘要；注入租约/钩子/inflight清零已分别核验。冻结不代表product_closed，不改变免临时许可方案。
