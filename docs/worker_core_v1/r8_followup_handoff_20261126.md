# r8 后续五项交接（2026-11-26）

本次交付是开发线工具、元数据、诊断原件和登记草稿，不改冻结r8，不发新适配器包，不发布未完成的r8.1登记。

| 项目 | 本次结果 | 下一步 |
|---|---|---|
|Native构建身份|d9aa… 的 dcat-native-build.json 已按原构建证据生成；989字节、SHA c61f1a27…73ea9|黄区核对库SHA后放到实际库旁，再回传双rank observe_identity；负例通过后登记|
|N2|新增 launch_host 变体，传完整host/holder，tiling仍null|黄区用已有六条原件离线运行；不阻塞D/G，不直接进入33控制重算|
|Batch|A2/0.26/Qwen3.8 W8A8/TP2 三轮无注入完成；同组成下首轮第0请求不同，后两轮精确|保留基线不稳定结论；具体内核与黄区OFF原因未证明，不放宽恢复协议|
|第二owner|已有栈，尚不能确认情况B|待同次运行owner/调用序号/资源归属证据；未改runtime|
|r8.1草稿|七项14份rank身份校验通过，expected_ranks已填|negative_controls保持null，未加载、未登记|

交接入口：

- Native：`native_build_metadata_d9aa_20261126/README.md` 和同目录 `dcat-native-build.json`。
- N2：`fault33_n2_launch_host_20261126.md`；工具为 `tools/extract_fault33_launch_host.py`，依赖 `tools/extract_fault33_existing.py`、`tools/review_launch_capture_v2.py`。按文档从同一开发源码根运行，不混入旧目录中的同名模块。
- Batch：`r8_2_batch_no_injection_20261126/README.md`、原始baseline JSON、独立review JSON、隔离原件。
- 身份与负例：`yellow_r8_identity_owner_review_20261126.md`、`r8_1_expected_ranks_draft_20261126.json`。

验证：相关CPU **68/68**；Native生成命令实际运行且重复生成一致；442个公共机制/运行时文件对照r8全部字节相同。文件逐项SHA和测试模块见 `r8_2_identity_n2_checks_20261126.json`。A2新对照源码提交5f1dc24f，清单52c2b0cd…868f8；没有把旧冒烟当作新工具的NPU证据。新增Native/N2提取不运行NPU。

本次A2窗口北京时间17:38:31–17:42:47，约256秒，已结束并解除隔离。蓝区诊断不替代A3正式资格；现有冻结标签、wheel和运行时登记表未改。
