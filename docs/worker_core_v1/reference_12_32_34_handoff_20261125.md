# 库12、32、34：独立参考件接入边界

开发线材料；最终可交付档位只以新补充件 manifest 和 a2-review.json 为准。冻结 r7.4.1、旧补充件、旧标签不变。A2 / vLLM0.26 / CANN9.1 / 社区Qwen3.8 W8A8 / TP2 的通过记录不授予 A3 资格；黄区仍须审查实际部署、可信注册、许可和正式产品验收。库行号不是公共故障编号。

## 12：v2 持久 lm_head 权重行构造

固定部署变体：`row12_lm_head_collapse` 将 token10/11/12 的行统一为 token10 的原行；`row12_lm_head_swap` 交换10/11，保留12。用户不能传 token、数值、地址或层名。只在拥有完整目标组的词表分片改写；其他rank核对本地完整权重未变。目标组跨分片时当前实现拒绝，不默默改成一部分。

不是旧的 logits 列移位。使用现有 v2 生命周期和 `PersistentRegions`：独占入口排空、设备读者结束、备份实际3行连续区域、原位写入、开启期间各真实请求读到权重、关闭时排空并按同一对象/分配代恢复原字节。持久写入只计一次；`reader_observations` 单独记录 prefill/decode 每次读取。`changed` 是有变化的写入调用次数，`changed_elements` 是权重区域变化元素数；不把持续读取次数伪计为重复写入。

可信绑定需要证明：未量化 BF16 ND lm_head、原始权重直接被实际 linear/mm 读取、无 tied embedding、无未审查派生副本/预打包、词表分片边界真实。A2包装类中的第三方源码SHA只对本次源码有效；A3不匹配必须先审查，不能删除校验或将新SHA当默认。原生量化 lm_head、reduce_sample、未覆盖的缓存/融合路径均拒绝。

每owner的第一token必须对比同一hidden下正常/故障目标logits：坍塌档三列等于原首列；交换档等于原对应两列交换。记录完整logits SHA、目标概率、真实sampler读取同一logits对象的证据。token可以不变，必须有目标分布变化。两个owner用同组Worker；缺许可、超范围、重复enable应结构化exit4。OFF字节恢复、资源清零、新请求hidden/logits逐位金丝雀三项分别通过。

隔离A2执行入口（完整源码及已审查模型挂载由执行器提供）：

```bash
DCAT_REFERENCE_ROW12=row12_lm_head_collapse python3 -B -m reference_adapters.lm_head_product
python3 -B -m reference_adapters.review_lm_head /evidence/persistent_product.json --output /evidence/independent-review.json
```

交换档使用新的evidence目录并把环境变量值换为 `row12_lm_head_swap`。该包装类是隔离验收示例，不是可直接对公网开放的RPC；黄区必须保留自身鉴权。

## 32：真实 Conv3d 输入的 FP8-like 舍入

`row32_conv3d_input_e4m3` / `row32_conv3d_input_e5m2` 是现象代理。输入round-trip回原BF16 dtype，原算子和权重不改；固定nearest-even、渐进下溢、有限饱和：E4M3最大448，E5M2最大57344。保留NaN/Inf，不通过添加偏置制造变化。真实图片路径发现Conv3d，实际算子读取改写输入，等待同stream读取完成后恢复；encoder继续读取真实输出。

```bash
python3 -B reference_adapters/product_probe.py --mechanism 17 --reference row32_conv3d_input_e4m3 --negative-regression
python3 -B -m reference_adapters.review_reference /evidence --receipt product_http_17.json --output /evidence/independent-review.json
```

每个档单独两个owner，不能借用另一档资格。`--mechanism 17`只是验收工具的图片调度配置，登记机制是 `external.row32_conv3d_input_e4m3` 等；不是公共17。

## 34：真实 BF16 投影输出到融合残差消费者

`row34_projection_mantissa5` / `row34_projection_mantissa3` 在原生投影已经产生的 BF16 输出上截断低尾数位，非有限值不变。实际 down_proj 返回值必须就是MLP返回值，且被 `_C_ascend::npu_add_rms_norm_bias` 读取；元数据view不算消费。不是反量化权重，不增加假bias或加法，不宣称复刻图融合机理。社区量化投影的输出边界不等于黄区BF16权重投影，A3必须观察实际链路。

```bash
python3 -B reference_adapters/product_probe.py --mechanism 18 --reference row34_projection_mantissa5 --negative-regression
python3 -B -m reference_adapters.review_reference /evidence --receipt product_http_18.json --output /evidence/independent-review.json
```

`--mechanism 18`只选择串行文本调度，实际机制为 `external.row34_projection_mantissa5` 等。该路径为model stage；不得套用旧34的vision stage说明。

## 控制拒绝与交付身份

32/34使用T2的ReferenceControlMixin，缺候选、身份漂移、超范围在控制路径返回exit4和原拒绝码；真实forward/kernel错误不伪装为正常跳过。新ZIP同时携带此mixin和必要数值helper，逐文件核对实机source_manifest SHA；仅比较工厂SHA不够。原始回执和私有数组留在124或黄区，Git仅提交审查摘要、原件SHA、源身份和代码。
