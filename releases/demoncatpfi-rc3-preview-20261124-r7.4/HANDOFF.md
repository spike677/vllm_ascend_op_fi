# RC3 preview r7.4（2026-11-24）

标签 `demoncatpfi-rc3-preview-20261124-r7.4`，源码提交 `d43e57106d7d8dd262d66bceef9c0c455065b241`。只增加外部 provider v2 持久区域路径和33采集器v2；r7.3.1、r7.3和其他冻结标签不变。

许可 source_manifest_sha256 使用 **source-review.json** 的 SHA256：`87f6a3ed3fd3e1ddd5e2321d2e2aa2f2e8348a8e40de9f5483b399b791c9f646`，不是 release-manifest.json。

## 门禁与验收范围

- CPU完整门禁1282项：1280通过、2项旧研究入口带理由跳过；随后审查器加强owner关联，6项补充回归通过，并重新审查最终真实KV回执。
- 124 A2 / Ascend910B4 / vLLM0.26.0、vllm-ascend0.26.0rc1 / CANN9.1.0 / Qwen3.8社区W8A8 / TP2：weight_scale和KV16层各两个owner，同组Worker、真实消费者、字节恢复、资源清零和金丝雀通过。weight示例未采同输入fault hidden，不宣称模型传播。
- KV每owner/rank写入16次、16元素改变、48次真实FIA读取；同输入fault hidden变化。缺许可、请求范围负例、真实HTTP取消后FINISHED_ABORTED释放恢复通过。未消费即取消分支仅CPU覆盖。
- 398个已安装源码文件逐字节一致；无设备映射、network=none、未创建Worker、未初始化NPU。
- 472个原有运行时/机制源文件与r7.3.1逐字节一致。新文件及工具/文档/打包元数据差异见source-delta.json；版本号变化不伪装成逐字节一致。
- 三个wheel：demoncatpfi0.4.0rc6.post5，worker-extension和fault39-core0.3.0rc6.post5。

## 接入与限制

完整发行目录含docs、examples、源码包、三个wheel、清单和SHA。先核验：

```sh
sha256sum -c DELIVERY-SHA256SUMS
python -m pip install --no-index --no-deps wheels/*.whl
```

可信部署按docs/worker_core_v1/external_v2_handoff_20261124.md注册persistent_*适配器、逐rank源码SHA、许可和真实入口排空/页面释放绑定。实例只支持唯一受控HTTP提交者、B1、4生成token、prefix cache关闭；不支持未知消费者或异步恢复。普通v1/公共路径不变。已有v1不安全持久区域拒绝码保持不变。

weight/faulty_requant、weight/param_perturb、kv_update_misdirect、kv_rope_quant_loss需要黄区自己的适配器和部署验收；本包A2示例不授予A3资格。区域覆盖、派生缓存、取消和恢复限制见r74_v2_acceptance_20261124.md。

33本包仅交付完整捕获采集器，**A3注入仍关闭**。公开源码候选解码和三形状控制检查在r7.5开发线推进，不以A2结果替代A3。D包与8仍等待D0/D1实际回传。r8不随本包发行。

完整原回执保留蓝区私有目录，Git仅含审查摘要和SHA。最终KV回执SHA：ca73ddbb02ff093d4d4611b37ec204b08c7a608d9a2ead2547c83e3df5718d33。窗口与隔离记录见r74_v2_windows_20261124.json；各窗口结束均已解除有界隔离，未修改Docker入口、驱动或CANN。

由负责人转交黄区，不自动发送或登记黄区组合。
