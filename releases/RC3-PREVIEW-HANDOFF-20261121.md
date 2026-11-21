# RC3 预览包交付：公共持续 28 / Batch

负责人转交材料；未向黄区发送。目标是 A3、0.26、Qwen3.8 W8A8、TP2 的独立验收实例。不是全部 12 项能力完成，也不是 A3 已验收。冻结的 RC1/RC2 发行物保持原样。

## 发行身份

- 标签：`demoncatpfi-rc3-preview-20261121`。
- 源码：`0f30107d378ad08c1bb51d56c2d3c8b374f143d5`；源码包包含 962 个经清单登记的文件。
- 发行目录：`releases/demoncatpfi-rc3-preview-20261121/`。
- 三个 wheel：demoncatpfi `0.4.0rc1`、demoncat-worker-extension `0.3.0rc1`、demoncat-fault39-core `0.3.0rc1`。
- 许可的 `source_manifest_sha256` 必须填写 **source-review.json 原始字节**的 SHA256：`b1f39306b093d160ff57c0554a33de1d2ad89935c21e05aa165c31a04968586c`，不是 release-manifest.json 的 SHA。
- 每个发行文件以 `SHA256SUMS` 校验。源码/三个 wheel 的哈希和字节数另见 `release-manifest.json`。

在蓝区 A2 的隔离无设备容器中重新安装三个 wheel，核对 348 个已安装 Python/JSON 文件的字节与源清单一致；检查通过，无 Worker 创建、无 NPU 初始化。原始结果为 `install-check.json`。这项检查不能替代模型验收。

## 蓝区验收范围

组合为 Ascend910B4（A2）、CANN 9.1.0、vLLM 0.26.0 / vllm-ascend 0.26.0rc1、torch 2.10.0+cpu / torch_npu 2.10.0.post4、社区 Qwen3.8-27B W8A8、TP2。

|能力|公共入口、同 Worker 两作用域|每 rank 每作用域故障覆盖|相对正常请求耗时|
|---|---|---|---|
|28 torch denominator replay|通过|16 个 full-attention 层 × 4 次实际执行，64 次 applied/consumed/restored，failed=0|1.4265× / 1.3416×|
|Batch NPU shared scale|通过|304 个实际独立量化点 × 4 次执行，1216 次 applied/consumed/restored，failed=0|6.3440× / 6.2005×|

两项均验证 prefill + 3 次 decode，全 rank 持续触发；真实消费者同 stream 完成后恢复区域字节，租约/钩子/在途请求清零；新 OFF/health 请求完整 hidden/logits 精确恢复。无许可实际创建被拒，耗尽后 advance 被拒。原始私有回执保留在蓝区，公共独立复核报告随源包提供。

Batch scale 与参考精确一致；最终 NPU 回执每 rank 约 124 万个确定性 Q 样本中差异为零。仅检验样本，不声称完整 Q 无差异。CPU FP32 路径对照为 6.0880× / 6.4494×；当前数据没有证明 NPU 路径更快。正常请求实验上限 120 秒，性能约束详见验收步骤。

28 是分母块重放的大误差效果变体，不复刻历史 kernel。Batch 只含独立 DynamicQuant→QMM，不涵盖融合量化。其他 10 项没有借本预览取得持续入口资格；已有代码是依赖或历史实现，不等于预览可执行范围。

## 黄区执行与回传

按源包 `docs/worker_core_v1/rc3_preview_acceptance_20261121.md` 执行。包括安装校验、可信 binding、YELLOW_DEDICATED_INSTANCE 许可、真实内部请求 ID、预热、四阶段、两个作用域、负向检查和纯文本独立总结。部署源码指纹不一致则拒绝，不能修改指纹跳过审查。

黄区原件留黄区，只回传独立总结、安装检查全文、回执文件名/字节数/SHA。通过审查后才可登记 A3。Codex 没有向 openchamber 发送消息。

## 测试与已知限制

关键公共入口/部署准入/独立复核/只读 inspect/预览包装等定向 CPU 测试通过，最后一次修正后 34 项通过。历史全量测试没有全绿：与基线比较共存 199 项失败及 11 项错误，详见 `regression-review.json`；另三个新增或未分类项已定性并通过定向复测。不能把这次预览说成旧研究路线或全部历史能力的全量合格发行。

卡窗口 a1–a6 的绝对时间、状态和隔离恢复记录在 `blue-window-audit.json`。均已结束并解除隔离，没有停止外部容器。a2 的原始 NPU scale 失败保留；修正后以 a5（28）/a6（Batch）为最终验收，不覆盖失败证据。
