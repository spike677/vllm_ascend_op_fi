# RC3预览r4交付单

r4可交负责人转发。新增17、41、26 fia_destination、20，累计包含原r3六项共十项；33未通过不纳入，DeepSeek新代理8另行适配。A3尚未验收，product_closed=false。原r3标签及字节不变，未直接向黄区发送。

- 标签：`demoncatpfi-rc3-preview-20261122-r4`
- 源码提交：`d72d8ad584bfda7d607855a27701a641084262a6`
- 发行提交：`cbbc2421b6ac28a26da72eb806d616195135d9a4`
- 发行目录：`releases/demoncatpfi-rc3-preview-20261122-r4/`，三个wheel、源码tar.gz、source-review.json、release-manifest.json、SHA256SUMS、测试及安装回执。
- wheel：demoncatpfi `0.4.0rc3`，worker-extension与fault39-core均 `0.3.0rc3`。
- 许可 `source_manifest_sha256` 取 **source-review.json原始字节SHA256**：`ab4155c3ff445026ae7bd97d2cebd6daaf86df1b7155fe11a4a1fe08b2eae421`。不是release-manifest SHA，也不是r3值。

1026项CPU范围测试、0失败/错误、2旧研究skip；首轮本地缺Pillow的错误未计通过，项目隔离环境补依赖后完整复跑。124无设备隔离安装通过，368个安装源文件与清单逐字节一致，无Worker/NPU初始化。所有发行文件已从Git暂存对象核对SHA；发布后保持原字节。

执行：[r4 A3 TP1/TP2步骤](../docs/worker_core_v1/rc3_preview_r4_acceptance_20261122.md)。基础参数必须重新对齐[r3补充页](../docs/worker_core_v1/r3_parameter_handoff_20261122.md)。不同能力可分专用实例并行，同能力两作用域必须同组Worker；TP1/TP2分别登记。

17必须带视觉占位符、mm_processor_cache_gb=0、新multimodal_uuid，固定64×64图片。按负责人决定默认关闭预期图片SHA比对；仍记录实际文件/像素SHA与尺寸，只是不声称与此前原件做了预期身份核验。可选参数支持完整SHA比对。原件后续已找到完整SHA的事实不改变本决定。

41/26/20须黄区先在87隔离-a3/CANN9.1容器构建Native库；[构建步骤](../docs/worker_core_v1/native_regions_a3_build_20261122.md)不依赖patchelf。源码/SDK/工具链/库SHA回传审查后才用于验收。torch_npu.get_device_name当前逻辑设备原值作为硬件依据，npu-smi泛称仅参考；A2库不授予A3资格。

公开绑定、许可、request_plan、四阶段四token、两作用域、负向检查、消费者事件、字节恢复、完整OFF/health及资源零要求不变。回传安装检查和各机制summary全文、原回执文件名/大小/SHA、覆盖与性能、异常/完整错误码，Native附构建收据。通过审查后仅登记实际A3/Qwen3.8/0.26/TP1或TP2专用实例组合，不代表现役对外服务原样配置。
