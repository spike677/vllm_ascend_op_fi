# 未闭合项及下一步

这里记录本轮已探索后的缺口；不是泛用支持承诺。每条完整方案见17张能力卡及探索台账。

| 范围 | 有证据的阻塞 | 最小下一步 |
|---|---|---|
| 20 WeightNz | 真实Qwen3 serving两轮0原生事件；模块调用存在 | 只读确认实际MatMul API和cache-hit路径，适配真实调用而非按层名强绑 |
| 20 QMM | TP2各rank仅第一轮3事件；第二轮0，PROFILE_REQUIRED | 补跨缓存生命周期观察和明确profile；再由Host提供排他完整模型回放，不能任取一个候选 |
| 39 range | 新共享Worker真实拒绝缺scheduler replay；同步七阶段要求不同真实请求ID | 现有Host实现可信调度回放或审查异步阶段接口，保留owner、单次、完整存储/输出门禁 |
| 15/17/18统计 | 全局调用链结构不齐；15消费者统计null | 先定义、测试并审查允许干预的非张量参数/新增调用对齐规则；不能删除记录强行通过 |
| 15模型路径 | 仅构造私有格式view写入原base消费 | 自然模型中寻找实际写入消费者链，找不到保留机制级 |
| 17/18模型路径 | 构造Conv3d/标准RMSNorm；自定义Norm不直接支持 | 使用实际适用视觉/Norm模型，分别验证调用、消费者和恢复 |
| 26 residual | 写前目标NaN，分配初始化来源未证 | 不修改原采集，补allocation生命周期观察和健康输入完整链 |
| 26 FIA / 33 | 已有历史实际推理Provider，共享Worker宿主接线缺 | 实现部署factory和可信replay，分别保留writer/copyback、page/KV/launch证明 |
| 共享量化 | 已有请求行映射和全量统计，缺每请求scale/Q影响 | 两rank逐行scale/Q辅助核对；独立首差仍只读原始统计 |
| 28三变体 | FA模型桥、新FIA router匹配vendor验收缺 | 在各自已验证形状/构建完成接线；合法切分和分母错误单独验收 |
| 8 / 22 | 8找回源码但依赖缺；22仅旧证据无实现本体 | 恢复可审查源码与Host依赖，再证请求、路由/行选择及真实消费者 |
| 25 / 41 | collective组或dispatcher/cache消费者尚未证 | 先完成真实只读边界观察；无边界则不注册Native Provider |
| 39 output | 历史post-Run代理存在，新Host合同未定义 | 专属typed output/storage proof与fresh-request OFF，不套用range合同 |
| GLM/DeepSeek NPU / A3 / 未安装CANN | CPU结构通过不等于这些环境的NPU故障通过 | 获得实际兼容模型/设备/镜像后逐组合验收，不按品牌/大版本插值 |

没有把这些缺口改成用户手写层名，也没有为达到数量降低单次/unique资格门禁。
