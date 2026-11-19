# Qwen3.8公开制品、访问检查与预算

目标是开发用社区W8A8/A2/0.26/TP2；不是黄区现役制品或A3验收。M1继续本地Qwen3-0.6B BF16/0.18/TP2，不等待本项。

## 来源、身份和许可

[vLLM-Ascend官方模型教程](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.8-27B.html)的Atlas800I A2/A3示例明确引用[Eco-Tech/Qwen3.8-27B-w8a8](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-27B-w8a8)。这是普通W8A8制品，不使用310p、MXFP8或C8变体。官方CI的A3名称不能据此推断其内部镜像/制品字节与本仓相同。

模型卡声明`Apache License 2.0`、基模型`Qwen/Qwen3.8-27B`；卡内部分文字和配置名仍写Qwen3.6，原件保留不替其纠正。该仓未单列LICENSE文件，声明来自已校验README。基模型BF16仓有Apache许可证，但本轮没有下载BF16权重。

[固定逐文件清单](../../experiments/worker_core_v11_20261118/qwen38_public_model_manifest_20261119.json)记录每文件revision、大小、SHA；ModelScope返回的Revision是逐文件最后修改提交，不冒称统一仓库提交。下载按各自固定revision和SHA验证，不能只根据master名称认为一致。

- 10个safetensors合计30,366,594,248字节（约28.28GiB）；仓库所有列出文件30,389,967,433字节（约28.30GiB）。
- config SHA：`191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`，恰与黄区回传相同，但不足以证明权重一致。
- quant描述SHA：`ae7835acbab2264a4e9940e1b45507fdc80be75b527fec2dbc21f0701c7c178b`。
- 描述字符串计数：W8A8_DYNAMIC=1843，FLOAT=585；动态计数包括1个model_quant_type元字段。黄区报告1321/759，未收到原始描述，不能逐项补齐差集或假设相同计数口径。
- 本仓48个linear_attn.out_proj.weight都为动态量化，黄区观察为BF16 MatMulV2/FLOAT。39需分别适配实际自然调用，不能把社区quant路径验收替代黄区BF16路径。
- norm、linear conv1d、A_log/dt_bias、embedding/lm_head、视觉patch_embed为FLOAT；Q/K/V/o和MLP投影等为动态量化。视觉和MTP还有独立量化项，计数不等于运行时算子调用次数。

## 实际访问与下载

2026-11-19，Windows和124对ModelScope的BF16/W8A8元数据API、昇腾社区首页均HTTP200。只能证明相应接口连通，不能保证大权重CDN可完整传输。Windows的小文件已按SHA落在D盘；未写C盘缓存，未传权重到124。

下载目录：`D:\DemonCATPFI\models\Eco-Tech\Qwen3.8-27B-w8a8`。公开原始元数据接收目录：`D:\DemonCATPFI\intake\public\20261119\qwen38`。黄区原始材料仍在独立yellow intake，不混入公开清单。

本轮先尝试固定revision全文件读取，再尝试严格Content-Range校验的分段读取。首分片都遭CDN读取超时；第二次保留4,980,736字节`.partial`，尚无任何完整权重分片通过SHA。1MiB范围探测曾成功HTTP206，不能将其当大文件下载完成。已校验配置等10个小文件；保留失败收据，未自动无限重试。后续可从校验过的偏移续传，最终必须全文件SHA一致后才更名。

## 124磁盘与0.26门禁

本轮/home可用66,168,741,888字节。仅扣社区完整制品30,389,967,433、自有构建10GiB及保留15GiB，给镜像解压量的上限剩**8,935,228,855字节（约8.32GiB）**。镜像完整解压总量尚未知，不能转移模型或pull。

按更保守的下载峰值另外保留压缩层6,718,398,624字节，展开空间只余2,216,830,231字节；已测三层+第4层部分展开就超过此值。因此保守峰值预算已不足，后续需额外磁盘预算；不能依赖删除其他任务数据或擅改Docker。Windows D约562GiB空闲，能容纳社区制品，不改变124门禁。

[0.26流式测量](v026_gate_evidence_20261119.md)3层通过，第4层EOF停止、后13层未尝试；官方仅找到CANN9.1与HDK25.5家族依据，宿主精确driver/firmware配套未证明。未pull、未无设备安装0.26、未新CANN构建、未NPU实测。
