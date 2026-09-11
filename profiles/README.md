# 配置、示例与能力描述

| 路径 | 类型 |
| --- | --- |
| [capabilities](capabilities/README.md) | 能力manifest，保留native源码引用，不是新服务许可 |
| [discovery](discovery) | 历史模型发现profile，不能跨epoch执行 |
| [distortion_probe_suite_v1.json](evaluation/distortion_probe_suite_v1.json) | 内容为冻结NORMAL_DIAGNOSTIC_SUITE_V2、16题和阈值；文件名兼容 |
| [qwen06b_demo_source_suite.json](evaluation/qwen06b_demo_source_suite.json) | 已有MT-Bench首轮8题公开源快照，有revision/hash，没有模型响应或私有部署 |

单卡使用prepare工具从source选择82/86/149，生成新512-token suite并重算本地tokenizer长度。source原预算1024不等于新演示默认预算。`demo_scope`保留转换/来源hash，不修改原题/阈值。

deployment、key、lease、epoch、TargetPlan是会话产物，不提交这里。配置意图可复用，实时绑定不可跨服务继承。详见[产物合同](../docs/reference/artifact_io_contracts_zh.md)。
