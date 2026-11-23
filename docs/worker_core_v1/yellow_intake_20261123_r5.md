# 2026-11-23 r5黄区提取文本审查

14段提取文本逐份重新verify，全部consistent=true、registration_ready=true（负责人最终决定方案B）。原始材料在仓库外接收目录，完整summary及回执留黄区；仅按黄区文本提取证明登记。原文SHA及逐rank八计数见[逐份审查记录](yellow_r5_extract_review_20261123.json)。不传播机器地址、容器名或硬件序列号。

唯一缺失字段是image_digest，其他新增字段均齐全；原文null及extract_sha256不改。方案B将其记录为“未观测（4.9 非必需）”，不再等待补充声明，不进入身份比对；其余必需字段缺失仍拒绝。14份均匹配A3/Ascend910_9382、0.26/0.26rc1、torch2.10.0+cpu、torch_npu2.10.0.post4、CANN9.1.0及回传现役制品config、量化和全部索引SHA。

|TP|机制|verify/登记完整性|作用域|消费者/恢复/负向|两作用域耗时倍数|
|---|---|---|---|---|---|
|1|28.fia.denominator_replay|通过|2个独立owner|全部通过|2.4816/2.0868|
|1|quant.batch_shared_scale|通过|2个独立owner|全部通过|12.5084/4.9721|
|1|18.api_half|通过|2个独立owner|全部通过|8.7236/4.9537|
|1|22.wrong_row|通过|2个独立owner|全部通过|1.2699/2.9923|
|1|39.output_proxy|通过|2个独立owner|全部通过|8.1674/9.3465|
|1|39.matmul_range|通过|2个独立owner|全部通过|5.2270/5.0449|
|1|17.spatial_padding|通过|2个独立owner|全部通过|0.9936/0.9918|
|2|28.fia.denominator_replay|通过|2个独立owner|全部通过|2.3437/2.3534|
|2|quant.batch_shared_scale|通过|2个独立owner|全部通过|9.7181/13.1475|
|2|18.api_half|通过|2个独立owner|全部通过|5.2429/5.0585|
|2|22.wrong_row|通过|2个独立owner|全部通过|1.3547/6.3178|
|2|39.output_proxy|通过|2个独立owner|全部通过|9.0062/10.0162|
|2|39.matmul_range|通过|2个独立owner|全部通过|23.7000/9.1755|
|2|17.spatial_padding|通过|2个独立owner|全部通过|1.0106/0.9895|

Batch 831/832、39mr 509～511/512是applied成功但无新增变化；changed<=applied合法，不算失败。所有eligible=attempted=applied=consumed=restored，failed=0。耗时只记录，不作门禁或性能结论。

r5.1从a76f33cf维护线仅登记14条组合，不合入r6机制改动。登记是验收证据，不自动授予产品模式资格；r5持续入口仍需原有可信作用域许可。

B组脚本未进入r5/r6源码包是打包遗漏，独立工具补交，不改冻结标签。请负责人转交：暂停以未知结构的旧脚本做正式标定，W4A8 NaN诊断和decode计数可继续；旧标定是否可转换需核查完整字段，不能预先断定作废或通过。

Native仅一行构建OK，没有构建身份，仍未审查；见[回传清单](native_build_text_return_20261123.md)。17历史固定图结果有效。

发行已完成：[r5.1交付单](https://github.com/spike677/vllm_ascend_op_fi/blob/demoncatpfi-rc3-preview-20261123-r5.1/releases/RC3-PREVIEW-R5.1-HANDOFF-20261123.md)，发行提交0e6ded808774111f20fdfb4bb63dcac0ef78a0e9；源6bc79f824607cf4a3bfac16c9021aaef0bce5cd6。48项维护CPU检查、1122个源码归档成员和124无设备安装378文件核对通过。开发线的方案B提取工具单独交付，不混入维护源码。
