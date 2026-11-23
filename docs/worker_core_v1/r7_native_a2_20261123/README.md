# r7 A2 Native 产品模式及共享模块回归

范围：Ascend910B4 / A2，0.26 A2 镜像 c9c9432b…，CANN 9.1.0，社区 Qwen3.8 W8A8，TP2。真实 localhost HTTP 用户请求由 Worker 自动发现；不以实验张量夹具替代模型。A3、TP1 Native、DeepSeek 和 33 A3 仍关闭。没有修改冻结版本。

## 构建及接线

`build.json` 为本次 `tools/build_native_regions.py` 的蓝区构建收据。三个 C++ CPU 测试和导出符号检查通过，链接真实 ACL，构建时没有映射设备。库 SHA `d9aa0d9488c5c1e8f7fca10c297e4968687ceb6a24866d8346bd33a58afce274`；源 SHA 已与当前 C++ 逐文件核对。r7 不修改 C++，A3 原构建是否可用仍待其完整材料审查。

新增受信部署配置接线：`native_library/native_sha256 → profile → product_binding_rpc attach → install_continuous_binding → NativeRegions`。用户 RPC 不允许增加库路径字段。Native 初次通过前只经明确的 BLUE 验收通道执行，已有公开候选不因此自动扩张。Worker 的实际 SoC/TP 和选中 CANN toolkit 必须匹配；模型、execute、调度源码由已有 source-bound 检查继续限制。

## 实测约定

每项两个 owner、同组 Worker；每个 owner 三个不同真实文本请求，各 prefill+3 decode，另一个五 token 范围外负例。请求前连续两次金丝雀完整 hidden/logits 有限且一致；关闭后新金丝雀精确恢复。记录所有合格命中、真实消费者设备读取完成、区域原值/恢复 SHA、所有 rank 的资源清零。Native 正例额外保留 source/after 或 KV 页/槽区间证据。

正式回执同时包含无许可不能准备、未知 profile 拒绝、用户 payload 不能指定库路径三项负向检查。审查器独立检查原始回执，不接受单一 status=passed。41 的非 paged prefill 明确不适用；decode 的所有合格调用仍持续触发。changed 小于 applied 不代表失败。

## 证据组织

- `review_*.json`：从原始 HTTP 回执独立重算的结果、回执 SHA、审查器 SHA。不上传完整统计/张量原件。
- `window_*.json`：每个窗口开始/结束的绝对时间、用时、隔离解除及原始记录 SHA。
- 完整回执/隔离日志保留在本地 `.codex-tmp/r7-live/` 及远端本项目独立窗口目录；公开总结不包含其他团队容器名称。
- 早期 41/26 单项窗口用于验证接线，正式候选以增加完整负向检查的联合窗口结果为准。

使用已有资源锁和最多35分钟有界清理器，仅作用卡2/5、只停止独占这两卡的外部容器，不删除数据、不改 Docker/驱动/CANN。每个窗口 finally 停止并 join 清理线程，恢复派发能力；不持续占卡。

## 发布边界

N 批的 A2 产品通过及原有八项回归通过后，才加入对应 A2 候选。黄区 Native 构建审查通过之前不冻结 r7、不发出 A3 执行请求。[A3 草稿](../r7_native_a3_draft_20261123.md)单独保留。D/P 结果不并入本次资格。

## N 批正式结果

三项在 `r7-native-all-a2` 窗口全部通过；窗口335.58秒，已解除隔离。每项同组 Worker 两个owner、六个正例用户请求、两个范围外请求；缺许可/未知profile/用户Native路径均拒绝。每rank六个正例总计：41 288次合格消费，26fia 384次，20 7296次；applied=consumed=restored，failed=0。完整摘要见 `native_summary.json`。三项现已加入限定蓝区A2/TP2的候选，A3未放行。
