# RC3 预览发行更正：使用 r2

请使用标签 `demoncatpfi-rc3-preview-20261121-r2` 和目录 `releases/demoncatpfi-rc3-preview-20261121-r2/`。

不要使用标签 `demoncatpfi-rc3-preview-20261121` 中的预览目录：Git 的 JSON 换行归一化改变了 source-review.json、release-manifest.json、regression-review.json 的原始字节，造成 SHA 清单不一致。问题在推送后从 Git 对象读取文件复核时发现。旧标签和目录保留为失败发行记录，不改写、不覆盖。

r2 使用已在 124 隔离无设备容器安装并核验的原始文件，不改变 wheel、源码包、source-review 身份或蓝区验收结果。目录明确设置 `-text` 保留字节，并从已提交 Git 对象重新核对每个文件的 SHA。三个 wheel 的版本号保持不变，因为二进制内容没有变化；r2 是发行容器更正，不是机制升级。

具体交付范围、局限、许可、安装与 A3 步骤见 `RC3-PREVIEW-HANDOFF-20261121.md`，其中发行标签和目录以本更正为准。许可仍填 `source-review.json` 原字节 SHA256：`b1f39306b093d160ff57c0554a33de1d2ad89935c21e05aa165c31a04968586c`。

仅 28 和 Batch 已通过蓝区 A2/0.26/Qwen3.8 W8A8/TP2 公共持续入口；A3 尚待验收。没有向黄区发送消息。
