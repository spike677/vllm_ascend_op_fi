# 黄区交付索引 2026-11-26 r1：r8

本索引交负责人转发，不直接联系黄区。负责人已决定立即冻结候选81643582，A3登记为空，按新正式执行单开始验收。覆盖A3 / Qwen3.8现役W8A8 / 0.26 / CANN9.1 / TP2；BLUE结果不授予A3资格。

- 源码提交：`81643582d2983b574b97208ddb3530312b167f07`。
- 发行提交：标签 [`demoncatpfi-rc3-preview-20261126-r8`](https://github.com/spike677/vllm_ascend_op_fi/tree/demoncatpfi-rc3-preview-20261126-r8) 指向的提交；执行 `git rev-parse demoncatpfi-rc3-preview-20261126-r8^{commit}` 核对。源码提交与发行提交分开记录。
- [交接单](https://github.com/spike677/vllm_ascend_op_fi/blob/demoncatpfi-rc3-preview-20261126-r8/releases/demoncatpfi-rc3-preview-20261126-r8/HANDOFF.md)；[A3正式测试执行单](https://github.com/spike677/vllm_ascend_op_fi/blob/demoncatpfi-rc3-preview-20261126-r8/releases/demoncatpfi-rc3-preview-20261126-r8/r8_a3_formal_acceptance_20261126.md)；[r8.1免许可增量单](https://github.com/spike677/vllm_ascend_op_fi/blob/demoncatpfi-rc3-preview-20261126-r8/releases/demoncatpfi-rc3-preview-20261126-r8/r8_incremental_acceptance_20261126.md)（后者只在登记发布后生效）。
- [发行目录](https://github.com/spike677/vllm_ascend_op_fi/tree/demoncatpfi-rc3-preview-20261126-r8/releases/demoncatpfi-rc3-preview-20261126-r8)，附三个wheel、源码tar.gz、完整docs、验证摘要、manifest和SHA清单。
- [DELIVERY-SHA256SUMS](https://github.com/spike677/vllm_ascend_op_fi/blob/demoncatpfi-rc3-preview-20261126-r8/releases/demoncatpfi-rc3-preview-20261126-r8/DELIVERY-SHA256SUMS) SHA256：`44dd6d886c36d1b7c200431161e03017fe9b50031c7610cf66d7b99780cdf9ec`。取完整目录后在该目录执行 `sha256sum -c DELIVERY-SHA256SUMS`。
- 源码tar.gz：3599632字节，SHA256 `14e38c28a8d58991af9073bdba49d4f2588f56d684852c463719fd411f164861`。
- 许可使用 **source-review.json** SHA256：`8568f8a4b59fbfc98060238bbd4acf526f0056d67ddd8eb2f91835dddcc311f5`。

|wheel|字节|SHA256|
|---|---:|---|
|`demoncat_fault39_core-0.3.0rc7-py3-none-any.whl`|49130|`854ba3fb0ce47e0b2952a38ebe5e804ad01bcd0df4674b856c4c9d363cdc9c8b`|
|`demoncat_worker_extension-0.3.0rc7-py3-none-any.whl`|647919|`5ba51c1b96f22ba662362e2f305941e27f994202dd2fb31b169c54ceb2d6a500`|
|`demoncatpfi-0.4.0rc7-py3-none-any.whl`|1050026|`1a6c3d916d67c6f3ff47b285c9d34f02f559aa4c1e498e57234bc5e8515d6782`|

## 验证和范围

[安装检查](https://github.com/spike677/vllm_ascend_op_fi/blob/demoncatpfi-rc3-preview-20261126-r8/releases/demoncatpfi-rc3-preview-20261126-r8/install-check.json)：124上无设备、无网络容器，三个wheel和406个Python/JSON文件逐字节一致，未初始化NPU；[执行记录](https://github.com/spike677/vllm_ascend_op_fi/blob/demoncatpfi-rc3-preview-20261126-r8/releases/demoncatpfi-rc3-preview-20261126-r8/execution.json)。

[CPU](https://github.com/spike677/vllm_ascend_op_fi/blob/demoncatpfi-rc3-preview-20261126-r8/releases/demoncatpfi-rc3-preview-20261126-r8/cpu.json)：1360项，1358通过、2项既有研究入口带理由跳过；登记更新相关18项通过；正式单涉及的效果/bootstrap/打包检查27项通过。规范执行代码字节相同的证明随包，不把不同来源清单SHA混写。[A2回归](https://github.com/spike677/vllm_ascend_op_fi/blob/demoncatpfi-rc3-preview-20261126-r8/releases/demoncatpfi-rc3-preview-20261126-r8/validation/README.md)：16/16通过，同组Worker双owner，全rank消费者、恢复、资源及负例；Batch首轮失败原件保留，同源码完整复跑通过。安装后免许可18双owner和未登记22负例另外通过。实际A2组合为Ascend910B4 / 0.26 / CANN9.1 / 社区Qwen3.8 W8A8 / TP2。

[shared-byte-comparison](https://github.com/spike677/vllm_ascend_op_fi/blob/demoncatpfi-rc3-preview-20261126-r8/releases/demoncatpfi-rc3-preview-20261126-r8/shared-byte-comparison.json)和[影响分析](https://github.com/spike677/vllm_ascend_op_fi/blob/demoncatpfi-rc3-preview-20261126-r8/releases/demoncatpfi-rc3-preview-20261126-r8/r8_shared_path_impact_20261126.md)：r7.4.1既有481个文件中474不变，7个共享准入/控制改动；机制/site/observer/Native字节不变。冻结件与已测81643582候选的运行时和wheel相同。

## 执行与后续

本次A3产品登记0条；公共13个命名项与3个外部示例均用验收许可，18三档分开。四个TP2槽0–7卡，Batch B8、视觉spawn和内置图、Native精确库及实际CANN映射、39op双owner控制证据均按执行单。保持黄区CLI，不要求重写宿主或新增参考适配器。

正式文本到达当天审查。通过组合从本发行提交开维护分支发布r8.1，只修改登记表，附免许可冒烟；未通过/缺证不登记。外部示例仍走受控许可。33和D包等待各自证据，作为后续增量，不阻塞此次r8。所有旧标签/包不改。
