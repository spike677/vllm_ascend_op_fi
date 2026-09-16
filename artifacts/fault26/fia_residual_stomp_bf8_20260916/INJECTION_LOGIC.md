# Fault26 注入逻辑：FIA wrong-destination residual stomp proxy

## 点位

冻结runtime拦截 `aclnnFusedInferAttentionScoreV3GetWorkspaceSize`，将FIA输出tensor绑定的目标地址由原输出A改到当前live residual R。真实FIA执行写入R，而不是CPU直接制造随机tensor内容。

拓扑跟踪用上一AddRms输出与后续AddRms输入的关系确认residual。R不是任意邻近allocation。

## 执行与恢复

1. 验证设备、输出/victim大小及当前residual生命周期门禁，保存R原始内容。
2. 临时将FIA输出指向R内合法的目标区域，执行真实FIA。
3. 将FIA结果复制回原输出A，维持算子正常输出路径；R仍携带错误内容。
4. 等待匹配的后续AddRms消费R并完成，再恢复原始内容，readback核验hash。

本次参数：device0、output8192B、residual10240B、offset0、all_hits=true、max_hits=1024。rank1不执行注入。all-hit只针对满足既有门禁的调用，不是所有API、rank或shape。

## 不能声称什么

这不是FIA保持原输出地址后自然越界写入相邻内存，不要求A/R物理邻接，也没有证明历史kernel的OOB根因。它复现的是“算子输出目的地错误导致live residual被破坏并传播”的效果链。

本次447次applied中只有441次具备完整consumer+restore证明；6次恢复成功但consumer证明未成立，应保留为缺口。模型输出异常不抵消这些缺口，也不能用恢复原storage替代新请求recovery测试。
