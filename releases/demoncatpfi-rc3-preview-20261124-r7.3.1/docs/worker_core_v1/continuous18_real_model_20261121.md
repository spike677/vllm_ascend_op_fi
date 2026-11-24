# 18：Qwen Gemma / 融合 QK 公共持续验收

组合：A2 Ascend910B4，CANN9.1.0，vLLM0.26.0/vllm-ascend0.26.0rc1，torch2.10+cpu/torch_npu2.10.post4，Qwen3.8-27B社区W8A8，TP2，eager纯文本。源码清单 `bba4a8df8e23a1e8788f7c68db608ddcb3cb1a8a13caee432c717794ddac6f11`。精简原件审查见 `experiments/worker_core_v11_20261118/continuous_public_18_review_20261121.json`。

同组Worker两个作用域、公共Runtime四阶段、每请求prefill+3decode，均通过独立比较器（包含真实无许可创建拒绝）。每rank每作用域161个实际输出点、644次observed/eligible/attempted/applied/changed/consumed/restored、failed=0。129个GemmaRMSNorm模块输出及16层融合Q/K的32个输出均实际出现；没有要求从未调用的q_norm/k_norm子模块人为执行。

方法是每次归一化实际输出的最后token、首个元素乘0.5；所有符合该适配器条件的调用持续触发，不等于改写输出的每一个元素。融合归一化/旋转后的QK位于同一真实返回边界，统一标为缩放效果代理，不宣称改变了kernel内部精度模式。实际消费者为DynamicQuant、unified_attention_with_output，以及最终hidden选择的aten::index。强引用对象租约保持原存储存活，同stream消费者完成后恢复该元素，逐次原始字节哈希相等。最终norm的消费者在模型forward返回之后，观察器保持到真实选择操作，不以Python返回代替消费。

normal/fault/off/health所有hidden/logits均有限；fault发生变化，OFF/health完整指纹精确恢复。各rank租约、钩子、在途资源清零，作用域许可清理完成。性能分别1.8884825×、1.8466447×，不扩展至其他长度/输入/模型。

## 实测记录

a13在预热拒绝 `FUSED_NORM_PARENT_UNPROVEN`，未执行故障：插件将AscendQwen3NextAttention.forward赋给原类，类名没有变化。修正为从真实模块结构发现并审核实际forward文件SHA；没有放宽源审查或忽略缺失父调用。a13结束为2026-11-21 12:48:42.042943 UTC，隔离已恢复。

a14成功窗口：2026-11-21 12:49:08.358153至12:55:46.378706 UTC，398.021秒（含模型加载，不是kernel活跃时间）。沿用卡2/5最长2100秒有界清理器；窗口结束已停止并恢复派发能力，未改Docker/驱动/CANN，无外部容器stop记录。

登记仅为开发线BLUE_COMBINATION，不外推A3。r2冻结包不变，不发布r3；22另行独立验收。
