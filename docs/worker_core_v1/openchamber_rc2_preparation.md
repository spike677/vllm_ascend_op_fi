# 给 openchamber：RC2 测试部署验收与登记

用户已确认 RC1.1 合入黄区 master，18/39 各两个作用域通过；不重复请求无关回归。原 RC1/RC1.1 标签和包不覆盖。RC2 的新增部署准入按来源和硬件独立登记；未收到 RC2 的 14 项实值和回执前，黄区 18/39 默认关闭。

|能力|本次黄区安排|蓝区范围|
|---|---|---|
|18 api_half|现有 Qwen3-0.6B BF16 测试部署，许可下两个独立作用域|Source126 普通请求|
|39 output_proxy|同上，须原 postwrite Native 身份与实际恢复|Source116 普通请求|
|22 wrong_row|同上，真实首条 fused RoPE/最终 reader|124 a1 两作用域|
|8 restore_offset|本轮不要求黄区准备或验收|DeepSeek-V2-Lite W8A8 a5 两作用域|
|17 spatial_padding|本轮不要求黄区准备或验收|Qwen3-VL-2B 图像 a5 两作用域|
|Batch|黄区保持待配置，不要求当前准备权重或 bundle|已有动态 W8A8 单投影；a1/a3两个独立 NPU 作用域通过|

串行 eager：TP/PP/DP=1、enforce_eager=True、async_scheduling=False、enable_chunked_prefill=False、enable_prefix_caching=False、max_num_seqs=1；不启用 speculative、DBO 或上下文并行。保留机制自己的源码、ABI、实际调用检查，不能伪填蓝区版本满足门禁。

1. 按 RC2 冻结发行清单安装三个升级版本 wheel，校验 SHA 与实际安装源码。
2. 可信 Worker bootstrap 配置 `YELLOW_TEST_DEPLOYMENT` 和部署操作员授权回调；按 [验收许可接口](deployment_acceptance_rc2.md) 安装限定机制/本次源码清单的许可。FaultConfig 不提供许可开关。
3. 每项独立运行两个 owner 作用域，每作用域 normal/fault/off/health 四个真实请求，统一 advance→等待完成→query。请求名称不决定阶段；第五请求须拒绝。结束 finish→disable（适用时）→unregister→detach→移除 binding。
4. 保留失败首因，按原恢复接口清理；容器重启不能代替 OFF/health。
5. 从两份实际 Host 回执生成有界总结，文字回传供 Codex 审查；下一 RC2.x 登记对应黄区组合，之后才默认允许。

请回传：源提交、源码清单 SHA、三个 wheel SHA、实际安装校验结果；原 13 项字段加 A2/A3；具体芯片型号；两作用域每请求内部 ID/阶段、效果摘要、恢复和资源状态。蓝区无法直接连接黄区；尚未收到的信息保留待确认，不能由历史脚本替代。

黄区测试部署验收与现役专用实例分开登记。`product_closed` 仍以 openchamber 的真实主项目验收为准；不把小模型 TP1 eager 结果扩展到现役 TP8/DP2/EP/图模式。
