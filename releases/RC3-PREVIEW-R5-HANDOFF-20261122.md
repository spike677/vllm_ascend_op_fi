# RC3预览r5交付单

标签`demoncatpfi-rc3-preview-20261122-r5`。源码提交`d4ae552df2b4fd82641bb39f234d2071dcccaed5`。沿用r4十项验收模式能力；产品模式候选五项18、39op、22、28、17。本轮不是A3登记发布，product_closed=false。33和V4新代理没有加入资格。r4及更早冻结物保持不变。

- 发行目录：[r5](demoncatpfi-rc3-preview-20261122-r5/)，含三wheel、源码包、source-review.json、release-manifest.json、SHA256SUMS、安装和测试回执。
- 版本：demoncatpfi 0.4.0rc4；demoncat-worker-extension与demoncat-fault39-core 0.3.0rc4。
- 许可source_manifest_sha256必须使用**source-review.json原始字节SHA256**：`eff3ec54d292deec219555f32fc928278c30c23541e3ef794ef7f89f152a2fff`。
- 验收栈：Qwen现役W8A8 / v0.26.0rc1-a3 / CANN9.1.0；TP1、TP2分别验收登记。换镜像/模型制品/三个调度参数就是另一个组合。

1071项CPU回归0失败0错误、2旧研究skip，最终相关56项复核通过（两组有重叠，不累计为独立测试总数）。124无设备、无网络隔离安装通过，378个安装源文件逐字节一致，不创建Worker或初始化NPU。

蓝区A2/0.26/CANN9.1/社区Qwen3.8W8A8/TP2：五项Worker自动请求发现HTTP验收通过；同组Worker两作用域，每作用域三prompt、一个范围负例；17两张图片。39mr修复后验收模式两个作用域通过：每rank每作用域704次符合条件的调用，完整消费恢复，OFF/health精确，证据329629字节未超过512KiB。蓝区通过不代表A3通过。

正式阶段默认900秒（部署参数1～900），预热600秒/最多6次，性能倍数不作门禁。运行套件逐项独立；失败恢复证明通过后才继续，否则保留quarantine。身份登记要求量化描述SHA、全部实际safetensors索引文件名/SHA和max_model_len/max_num_batched_tokens/max_num_seqs。r4后续全文回传只审查结果、不新增登记；等r5带身份字段重跑。

接线：[产品RPC契约](../docs/worker_core_v1/product_rpc_contract.md)、[产品入口](../docs/worker_core_v1/rc3_product_usage.md)、[修复说明](../docs/worker_core_v1/r5_repairs_20261122.md)。验收模式沿用[r4步骤](../docs/worker_core_v1/rc3_preview_r4_acceptance_20261122.md)，将发行目录/三wheel/许可SHA全部改为本r5身份；正式预算以本页为准。可用`tools/run_continuous_suite.py`，不要遇首项失败后丢弃其余项。

41/26fia/20须在v0.26.0rc1-a3/CANN9.1隔离容器运行tools/build_native_regions.py，源码fault_injection/native_adapters/postwrite。不要使用旧fault26.fia.abi0.v1或fault20.qmm.abi0.v1 recipe，也不要使用9.2.1构建物。回传构建receipt/library SHA。未确认构建物与部署相符不得注入。

产品CLI用可信ProductClient控制attach/enable/status/disable/detach；用户直接访问推理API，禁止逐请求before/after钩子和next-N。r5仅候选，先在专用实例按RPC契约验收。回传各summary全文、原回执名称/字节/SHA、身份字段、安装检查、Native构建证明。未直接向黄区发送，由负责人转交。
