# 模型结构、实际调用与故障验收分层

本轮实际构造并运行了 Transformers 5.5.4 的6种小型CPU模型：Qwen3 dense、Qwen3 MoE、
GLM、DeepSeek-V2 MLA+MoE、DeepSeek-V3 MLA+MoE，以及Llama对照。
全部使用随机初始化权重和`eval()/inference_mode()`，执行B2带padding的prefill及带KV缓存的decode。
没有下载预训练权重，也没有训练、反向传播或优化器步骤。

这是**真实模型实现的CPU适配验收**，不是6种预训练模型的NPU故障闭环。
逐模型结果、实现源码SHA、实际Attention参数、模块角色与调用次数见
[model_matrix.json](../experiments/overnight_20261116/model_matrix.json)，
可执行入口为[model_matrix_probe.py](../experiments/overnight_20261116/model_matrix_probe.py)。

| 实际实现 | 结构及角色 | 实际CPU调用与恢复 | 当前原生Provider边界 |
|---|---|---|---|
| Qwen3ForCausalLM | 独立Q/K/V、GQA、dense MLP gate/up/down | 两层各prefill/decode一次，共4个共享Attention调用；透明路由与OFF精确恢复 | 当前小几何不满足FIA固定kernel；已有0.6B NPU证据需另按具体Provider引用 |
| Qwen3MoeForCausalLM | router与packed experts分开识别，不把dense gate当router | 实际MoE前向、Attention路由及OFF通过 | 没有因此证明CANN MoE dispatcher、unpermute、QMM或Fault8命中 |
| GlmForCausalLM | 独立Q/K/V、融合gate_up、down、规范化层 | prefill/decode/透明路由/OFF通过 | 只验收该内置GLM实现；不是所有GLM版本或remote-code的证书 |
| DeepseekV2ForCausalLM | MLA的KV latent投影、Q投影、router、experts和shared experts | 实际MoE+MLA前向、prefill/decode/OFF通过；QK与V维度不同 | 当前FIA适配器明确拒绝该几何；不能只因共享注册表可用就执行固定Qwen几何kernel |
| DeepseekV3ForCausalLM | 除上述结构外，识别Q latent和Q展开投影 | 实际MoE层从第0层开始；不是只测dense替代层；透明路由/OFF通过 | 原生MLA、expert融合与分页KV路径均须独立适配 |
| LlamaForCausalLM | GQA与dense MLP | 同样的CPU控制与恢复通过 | 辅助对照，不替代Qwen/GLM/DeepSeek三个主要家族 |

每个CPU路由调用仍使用实际模型的原始eager Attention函数、mask、scale和缓存；没有改写返回张量。
原生FIA门禁对上述小配置全部返回unsupported，报告保留具体原因。
`AttentionInterface`能够注册只证明控制接口可用，不能证明任意算子、形状、精度或硬件合法。

## 自动发现的实现

- [model_discovery.py](../fault_injection/model_discovery.py)从实际模型class、config和实现源文件记录架构身份。
  checkpoint名称不参与家族选择；实现与config家族冲突会明确标为`FAMILY_METADATA_CONFLICT`。
  自定义/remote实现保留`CUSTOM_IMPLEMENTATION_UNVERIFIED`，不能按名字继承内置模型资格。
- 角色仍由模块对象、兄弟关系、权重形状和投影几何共同证明，保留别名及歧义。
  新增`ATTENTION_Q_LATENT_PROJ`和`ATTENTION_KV_LATENT_PROJ`诊断，避免遗漏MLA压缩投影。
- [runtime_discovery.py](../fault_injection/runtime_discovery.py)继续负责真实dispatcher/native调用观察。
  本轮没有将静态角色直接转换为ACLNN slot，也没有新增第二套registry。
- 候选名只是诊断信息；自动选点仍须通过实际operator、typed formal、storage、scope及无损门禁。
  模型用户不填写层名，部署人员仍需提供匹配版本和实际预加载构建。

## 124模型资产与尚未完成项

本轮只读盘点发现`Qwen3-0.6B`、`Qwen3-32B-w8a8-nopdmix`、
`DeepSeek-R1-Distill-Qwen-32B-W8A8`、`DeepSeek-V2-Lite-W8A8`配置文件，SHA见
[environment_matrix.json](../experiments/overnight_20261116/environment_matrix.json)。
其中Distill-Qwen实际`model_type=qwen2`，其此前26/共享量化的通过结论属于Qwen架构，
**不作为DeepSeek-V2/V3 MLA/MoE的NPU验收**。

本次资产目录中未取得GLM预训练模型及已附着的GLM Worker；CPU GLM结构通过仍只记CPU。
DeepSeek-V2配置存在不等于当前Worker已加载，更不等于Fault8/20/39在其自然native路径中可用。
本轮可用性、共享Worker实测及后续故障结果由总闭环报告另外记录，不用本页的CPU结果填充NPU矩阵。

适配状态分别记录`MODEL_STRUCTURE_VERIFIED_CPU`、`MODULE_ROLE_VERIFIED_CPU`、
`CPU_SHARED_ATTENTION_OBSERVED`和`CPU_TRANSPARENT_ROUTE_RECOVERY_VERIFIED`；
`NATIVE_PROVIDER_VERIFIED`、`MODEL_FAULT_VERIFIED`须来自另一份真实设备回执。
