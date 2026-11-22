# RC3 第一批增量候选（未冻结）

11-22 最新决定已批准立即准备 r3，不再等待 r2 回传。r3 六项范围与 TP1/TP2 验收步骤见 [新版步骤](rc3_preview_r3_acceptance_20261122.md)。下表为 11-21 历史增量记录。**本清单不改变r2的28/Batch限制，也不授予A3资格。** 用户配置仍不接受rank、层名或地址。

|能力|蓝区组合与结论|下一发行候选|黄区|
|---|---|---|---|
|28 denominator torch|A2/0.26/Qwen3.8社区W8A8/TP2，公共两作用域通过|r2已有|等待r2文本回传|
|Batch torch|同上，公共两作用域通过；immediate为6.20–6.34倍，forward延迟核对为5.93–5.97倍|r2已有immediate；forward优化已在a17复核，待下一包审查|等待r2文本回传|
|39 output_proxy|同上；304真实投影×4，1216命中/rank/作用域，完整恢复|加入下一候选|未验收|
|18 api_half|同上；129模块+32融合QK输出×4，644命中/rank/作用域，完整恢复|加入下一候选|未验收|
|22 wrong_row|同上，16融合层×4，64命中/rank/作用域，672 rotary行改变，完整恢复|加入下一候选|未验收|

39/18/22可信安装复用 `install_continuous_binding`，mechanism分别为 `39.output_proxy`、`18.api_half`、`22.wrong_row`；MultiRankTaskClient continuous=True，真实group_contract，部署许可、normal/fault/off/health四阶段，每阶段4生成token，同组Worker两作用域。绑定本身不授予资格；A3未审查组合仍关闭。实际负向创建、耗尽后advance、最终许可和资源清理必须独立检查。

18和39的新源码尚不在r2 wheel中，不能只修改r2配置声称可用。下一发行须把通过项加入新的包范围声明、重新审查源身份、安装检查和SHA，保留r2所有标签和字节。39故障阶段允许机制预期的非有限输出，其他阶段必须全有限；18/22所有阶段全有限。消费者和区域字节恢复证据不可用“Python函数已返回”代替。

V2-Lite 28 MLA(a16)、8(a20)、18(a21)、22(a22)、39 output_proxy(a25)已通过各自声明范围内的公共两作用域，分别为108、104、220、108、332 eligible命中/rank/作用域。18融合KV内部norm未覆盖，8单组decode应用但无新变化，39只覆盖实际BF16 ND模块输出；为第二批准备保留独立组合，没有将它们写成A3/DeepSeek-V4.1资格。11-22 补充：39 matmul_range 已完成 A2/0.26/Qwen3.8社区 W8A8/TP2 公共两作用域及源码审查；33 模型传播未达标，保持关闭；Native 内存阶段仍待验收。进度总表另列实验/公共结果。
