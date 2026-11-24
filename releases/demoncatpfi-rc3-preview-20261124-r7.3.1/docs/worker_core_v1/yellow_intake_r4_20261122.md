# 2026-11-22 r4 / V4-0731 文本接收与调整决定

原文保留在仓库外接收存档；本页不记录机器地址、容器名或硬件编号。证据等级 YELLOW_TEXT_ATTESTED；不能当作蓝区原件审查。

## 原件身份

|材料|字节|SHA256|
|---|---:|---|
|01_r4包安装与验收结果_原文.txt|7739|62c7494b5e70e2a0f319b0a49a995ffc947c52068ec1fd88c81199184d984878|
|02_B组推理对照结果_原文.txt|7520|90d2a98198035732e7d900208ef91b7d775486763ff6c5fe0f38fe7394126e1f|

## r4 结论与缺项

黄区报告 A3 / Ascend910_9382 / 0.26 / 现役 Qwen3.8 W8A8 / TP2 的 28、Batch、18、22、39 output_proxy 两作用域通过。收到安装检查全文及五份 summary 的文件名、大小和 SHA；**未收到五份 summary JSON 全文**，不能审核逐阶段、每 rank 的效果、恢复、参数和资源是否一致。本次不新增已验收组合，不发布登记结论。

登记前必须有运行时量化描述 SHA、全部实际权重索引的文件名与 SHA，以及 max_model_len、max_num_batched_tokens、max_num_seqs。现役文本声明：

- quant_model_description.json：d3725f719225567a6cd0fd3e340860082e36c3643dfe0f32d64e3172547bf793。
- quant_model_weights.safetensors.index.json：512d41c5403b1fe0a65861ffe70da9a84a1e93a534e132fd31a01fdc32844ae9。

两值是待与各份运行时 summary 核对的现役制品身份，不硬编码为蓝区社区制品观测值。config 相同不能代替制品核验。补充文本由负责人转交：五份 summary 全文、逐份运行参数身份、39mr 失败后的统一恢复与资源计数。

39mr 的 622989 字节超 524288 限制属于证据导出缺陷，不能据此认定完整模型机制验收通过。修复为逐调用核对后按站点汇总及完整有序序列 SHA，不提高限制。TP1 首请求超过正式120秒为预算问题；预热独立600秒/次、最多6次，正式阶段仍120秒。各机制独立运行、失败清理后再继续；未恢复不能复用同一 Worker。

## V4-0731

收到黄区 V4-Flash-0731 W8A8 / v0.32.0rc1-a3 / A3 / TP8 + EP / DP1 / 4096 的文本。2500token 长输入有限、跨进程 token/logprob 一致；短输入 Q 跨进程 token 不同，短输入 P token 一致但 logprob 有差异。长输入尚缺同进程重复与换序证据，不能据此登记产品恢复能力。

实际候选路径为 SparseAttnSharedkv、Compressor、CompressorMetadata、MC2 Dispatch/CombineV2、HcPre/HcPost、InplacePartialRotaryMul、RmsNorm、QuantMatmulWeightNz、DynamicQuantV2。旧 V4.1 的 LightningIndexer/SparseFlashMla 准备不能直接当作这些调用的覆盖证明。仍缺部署源码指纹、decode 独立计数、实际加载 CANN 来源；目录或符号链接的版本不代替运行时版本。

## 负责人本轮决定

33 选择 A：限时获取真实 kernel launch、分块和累加顺序，再决定能否原生等值重算。执行窗口约定2小时工作时间、NPU最多30分钟；不扩大非原生等值替换范围、不人为放大误差。旧 split8/16 失败原件保留；A 不能完成时列证据缺口，不登记。

身份字段是登记前提。耗时本轮只记录，不做分析优化或改变精度/恢复标准。产品范围仅开启期间全部请求，Worker 从实际调度识别请求；宿主不参与逐请求开始/结束。冻结 r4 不改；修复和新验收进入下一候选，未通过默认关闭。
