# 黄区参考适配器（开发中，独立于冻结 r7.4.1）

除第12行的持久权重行构造外，本目录条目均按**现象复现，非原机理**交付。transient 工厂通过 r7.4.1 的 `ExternalSite` 注册；第12行通过 v2 `PersistentSite` 注册为 persistent_weight。不修改共享产品实现，不赋予任何黄区资格。黄区审查并用自己的部署身份注册后，仍须正式产品测试和登记。库行号不是公共故障编号。

下表是语义说明，不是状态表。实测状态以补充件manifest及a2-review.json为准；开发线当前状态见docs/worker_core_v1/fault_library_coverage_20261124.json。CPU测试不能替代A2两个owner的产品验收。只有附有实机结果和独立复核的条目才能进入参考交付 ZIP；开发目录不等于可交付清单。

|库行|固定命名变体/当前实现|采用的语义|黄区需要核对|
|---|---|---|---|
|9|`row9_scale_half`、`row9_scale_quarter`|原生 DynamicQuant 得到 scale；同一 x 按 scale×0.5/0.25 重新量化；nearest-even、int8 范围 [-127,127]；原 QMM 读取原 Q/scale 存储|原描述中的 scale 是 DynamicQuant **输出、QMM 的 pertoken_scale 输入**，不是 x。需核对原语舍入和饱和约定|
|10|`row10_scale_double`、`row10_scale_quadruple`|同上，scale×2/4；报告新量化零值与原生零值数量|这是 per-token scale 固定放大，模拟分辨率下降，不声称由真实离群 token 或 batch 变化引起|
|7|`row7_mlp_last_row_zero`|结构发现 MLP，实际输出最后一个 token 行置零|贡献丢失代理，不证明 MoE 超容量；若消费者原地改写受保护区域则拒绝|
|8|`row8_decoder_last_row_zero`|结构发现 decoder，层间 hidden 最后一行置零|模拟有效行被零 padding 替代，不是公共8，也不证明 CP 原生错误|
|12|`row12_lm_head_collapse`、`row12_lm_head_swap`|v2 改写真实 lm_head 的固定 token 10/11/12 权重行：后两行取首行，或交换前两行；仅拥有完整目标组的词表分片写入|必须证明无派生副本、无 tied embedding、实际权重读取与采样消费；未知路径或跨分片组拒绝。旧 logits 列移位不交付|
|3|`row3_gate_fp16_round`|SwiGLU gate 的另一种计算/舍入顺序|需核对实际 act/mul 是否融合；保留配置项说明，不能用任意输出缩放替代|
|16|`row16_conv3d_compute_fp16`|真实 Conv3d x/weight 只读，FP16 重算后改写实际输出|不是 ConvTranspose；必须图像请求且证明 encoder 读取|
|32|`row32_conv3d_input_e4m3`、`row32_conv3d_input_e5m2`|真实 Conv3d 输入模拟 FP8 舍入后回到原 dtype；nearest-even、渐进下溢、有限值饱和，保留非有限值|必须有真实图片、Conv3d 和 encoder 消费。旧 FP16 round-trip 不作为本轮达标实现|
|19|`row19_bf16_reverse_splitk`|BF16 matmul 反向 split-K FP32 累加|社区量化投影不等同现役 BF16 out_proj；不能擅自解量化后称为原 BF16 matmul|
|34|`row34_projection_mantissa5`、`row34_projection_mantissa3`|真实 down_proj BF16 输出保留5/3位尾数，实际融合 add_rms_norm_bias 消费后恢复|模型路径的现象代理，不改变权重或量化配置，不制造假加法；与19单独登记。旧视觉 bias 路径保留历史证据|

9/10 每次实际调用同时租用 Q、scale 的活跃对象；同 stream 等待原 quant scheme 的所有 QMM 读取完成，再恢复两块原始字节。输入 x 不改，保存其指纹。`changed` 是有字节变化的调用次数，`changed_elements` 是 Q 和 scale 中变化的元素数之和；饱和、裁剪、零值另列。

7/8 不按用户层名选点。16/32 注册 vision stage；新34 注册 model stage，要求自然融合残差消费者；旧34 bias 变体仍属历史 vision stage。19 固定最后一行前1024列，K块128，合法尾块包含在内。命名变体不提供用户数值、rank 或层名参数。

## 隔离 A2 验收

使用测试容器中 `/source` 的源码、`/model` 的社区 Qwen3.8 W8A8 和 `/evidence` 结果目录；Python 路径包含 `src`、仓库根目录、`fault39/src` 和 `experiments/worker_core_v11_20261118`。仅可信隔离环境启用测试用 collective RPC。`bootstrap.py` 是验收包装，不是对外未鉴权服务。

```bash
python3 -B reference_adapters/product_probe.py \
  --mechanism 18 --reference row9_scale_half --negative-regression
python3 -B -m reference_adapters.review_quant /evidence \
  --output /evidence/reference-review.json
```

第10行替换为 `row10_scale_double`；其他固定档须各自通过两个 owner 验收，不能借用另一个档的结果。这里 `--mechanism 18` 仅选择测试工具已有的串行文本调度配置，实际注册/回执机制是 `external.row9_scale_half` 等，不是公共18。

验收包含真实 HTTP 请求、4个生成 token、同组 Worker 两个 owner、缺许可/超范围/重复开启负例、原始区域恢复、资源清零、前后金丝雀。侧录由 `DCAT_EXTERNAL_SITE_EVIDENCE=/evidence` 打开；独立审查将侧录的 rank/request 与公共回执逐项对齐。原始实机数据留在验收目录，交付物只附审查总结和文件身份。

配置项说明仍见 `docs/worker_core_v1/fault_library_coverage_20261124.md`：库3的黄区原语默认值未提供；库19 `CLOSE_MATMUL_K_SHIFT`、库37 `ASCEND_ROUNDING_MODE` 的黄区报告均为未设置，不能推断为0或1。本包固定变体不设置这些全局环境变量。

## 社区与现役矩阵乘边界

本轮A2社区量化描述SHA为 `ae7835acbab2264a4e9940e1b45507fdc80be75b527fec2dbc21f0701c7c178b`。实际读取到1843项W8A8_DYNAMIC、585项FLOAT（另有元数据）；`linear_attn.out_proj`、`self_attn.o_proj`、`linear_attn.in_proj_a/in_proj_b` 和视觉block的投影权重均标为W8A8_DYNAMIC。FLOAT矩阵候选排除norm、conv、embed后只有 `lm_head.weight`。

这不同于黄区曾报告的1321/759项以及BF16 `linear_attn.out_proj`。库19只匹配实际BF16权重和真实dispatch操作数，不能把社区量化out_proj写成BF16。新库34只改原生投影已经产生的 BF16 输出，要求实际融合残差消费者；不会自行解量化权重或添加假的 bias/residual。最终范围以实际调用回执为准。

## 第19行的独立效果判定

分别报告 native→正序参考（control）、正序→逆序（rule）、native→实际替换值（changed）的Hamming元素计数，不把控制漂移称为顺序效应。由逐调用计数得出可归于顺序变化的实际改写元素下界 `max(0, rule-control)`；同时校验 `abs(rule-control) <= changed <= rule+control`。每个owner都需有正下界，所有rank仍须完成应用、消费和恢复；分片数据不同，不要求每rank都出现新的数值变化。下界不是精确交集计数。

A2实测每rank24调用。rank0：changed=2、control=2、rule=0；rank1：changed=4、control=6、rule=8，可归因下界2（两个owner各1）。因此现象很小且只在rank1证明，不能写成所有rank均存在顺序差异。全rank执行和恢复通过；不授予A3资格。
