# 33：K3 审查与旧文件补提交付

独立补充件；不覆盖 r2，不修改 r7.4.1 标签、wheel 或许可。由负责人转交，蓝区不直接联系黄区。

- ZIP：[`fault33-r2-k3-supplement.zip`](fault33-evidence-candidate-20261124-r2-k3/fault33-r2-k3-supplement.zip)
- SHA256：`6896c0a316cb96a8f0a1aede81ef17198906771e72d228c50b43a30d78748ce9`
- 逐文件身份：同目录 `manifest.json`；外层 `SHA256SUMS`。
- 原 r2 采集器源码、库不变；补充旧文件提取工具和审查文档。43 个 payload 文件，独立解压目录 41 项 CPU 测试通过。首次检查发现缺少只读控制重算依赖，已补入冻结版本原字节并重测；不需要从开发分支临时拷文件。

## 按此顺序使用

1. 新目录解压，校验 SHA。先读 `docs/worker_core_v1/yellow_intake_20261124_fault33_window1.md`：已有三形状原件保留，先离线补提；没有真实 key 或唯一对应关系时不进解码。
2. 执行 `docs/worker_core_v1/fault33_n2_existing_capture_20261124.md`：JSON Pointer 只映射原件字段，`tools/extract_fault33_existing.py` 只读文件、stdout 输出。字段缺失返回 exit 2 和 null；不补 A2 默认值。
3. ABI/离线布局完整步骤在 `docs/worker_core_v1/fault33_candidate_windows_20261124.md`，工具 `tools/fault33_candidate_schema.py`。旧文件不伪造 r2 sidecar。新的必要采集才用 r2；沿用文件不等于沿用 serializer 结论。
4. 回传 N2 后审查真实结构。当前候选没有 `outerSplitParams/fdParams`，不直接等同已有 coreInfo/splitInfo。不满足五项条件就停止；本包不开放 33，不下发未经审查的控制重算参数。

包内另附完整 Qwen 正式测试总单 `releases/RC3-PREVIEW-R7.4.1-FORMAL-TEST-PLAN-V2-20261124.md` 和 `docs/worker_core_v1/yellow_adapter_recipes_20261124.md`。这两份取代之前缩窄的复测范围；其他旧 intake / 验证报告保留为历史记录。

Python ≥3.10。离线提取只依赖标准库，先设置 `PYTHONPATH` 为解压目录。控制重算测试另需 torch；真实 NPU 控制仍需对应 CANN/torch_npu 组合与审核后的参数。本补充件无新增 NPU 资格。
