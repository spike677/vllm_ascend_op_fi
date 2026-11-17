# 未通过尝试及边界

所有新尝试以新目录执行；失败原件未覆盖。下列目录均相对于124的
`/home/worker-positive-20261117`。完成状态和回执哈希见
[尝试清单](summaries/attempts.json)。本表解释与结论有关的失败，不把重试成功
改写为“首次即成功”。

| 尝试或边界 | 观察及处理 |
|---|---|
| 默认 ND / 原 PTA 路径 | 模型模块被调用不等于 WeightNz 命中。此前226个模块事件的原生事件为0，保留该结果。新通过限定为明确披露的 controlled PTA/NZ 部署。 |
| `readonly20-pta-a1` | 仅主库/nn代理未覆盖实际查找顺序。补齐math代理以及已加载句柄的实际 dlsym/dladdr 验证。 |
| `readonly20-pta-a2/a3` | 更早的 custom opapi 句柄被拒绝。实查发现其实际函数解析到同一已验证代理；只在函数地址、文件、哈希和状态符号全部相同时接受，不以custom库名豁免。 |
| `serving20-qwen-a1` | Callable RPC收到WorkerWrapperBase，Extension命名RPC绑定真实Worker。部署安装入口显式取真实Worker，生产身份门禁未放宽。 |
| `serving20-qwen-a2/a3` | overlayfs映射设备号与stat设备号不一致导致环境指纹拒绝。将同一镜像的7个库按原字节复制到本轮私有目录，再只读绑定到容器原位置；没有忽略映射身份或修改主机CANN。 |
| `serving20-qwen-a4` | 默认绑权输出头是ND，实际输出投影没有WeightNz事件。没有挑一个近似候选冒充目标。 |
| `serving20-qwen-a5` | 真实权重NZ转换保持逻辑值，但绑权Embedding输出NZ，后续AddRmsNorm只支持ND，基线失败。增加B/F/OFF相同的Embedding ND部署适配，每次验证完整逻辑值一致。 |
| `serving-qwen-statistics-a2` | 同Worker的step2/step6物理KV槽位不同，独立首差为`_npu_reshape_and_cache.0.forward/input_kwargs/slot_indices`，Max77→141。此结果不支持MatMul首差。正式重采使用独立Worker同phase6，不改统计字节或过滤该算子。 |
| `closure-batch-a1/fault20-baseline` | 无损资格后额外reference被原生状态机拒绝；尚未执行fault。增加只允许同一目标、完整输入/输出相同的只读复查，保留已经消耗的fault预算，并增加原生回归测试。 |
| `closure-secondary-a1/fault20-smollm` | SmolLM2绑权Embedding和输出头是两个模块。只对输出头注册ND适配不足，故在注入前AddRmsNorm失败。后续按实际Parameter身份和VocabParallelEmbedding类型识别全部相关模块，不要求用户层名。 |
| QMM与WeightNz不同 | 有WeightNz验收不能授予QMM资格；真正DeepSeek量化模型的head在量化说明中为FLOAT，不能把这个输出投影标成QMM。独立只读QMM探索结果单列。 |

历史11月16日的15/17/18严格统计 `ALIGNMENT_MISMATCH`、Fault26 rank1写前NaN
和未知初始化来源均原样保留；本次没有修改比较器去绕开它们。本轮统计仍然只
能排除之前**可观测**字段的超阈值变化，不能将null解释为相同。

本地CPU预检的首轮曾因测试导入路径和新增math代理夹具不完整失败；修正后完整
556项通过。`python-source-a10` 的错误处理改动在CPU测试发现回调尚在执行时
不应触发drain，未部署到NPU；a11修正为仅保存原始失败原因，正常返回后恢复。
