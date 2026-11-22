# RC3 第一批增量候选（未冻结）

## 2026-11-22 后续开发线

r3已发布原六项，冻结标签和包不变。本轮新增候选为17视觉、Native41、Native26 fia_destination、Native20，均已在A2/CANN9.1/0.26/Qwen3.8社区W8A8/TP2通过公共同组Worker两个作用域；33模型传播不达标，继续关闭。Native三项的每rank每作用域eligible分别48、64、1216，全部消费/恢复，failed=0。对应报告和限制见[总表](continuous_progress_20261121.md)。尚未把这些后继实现打入新预览包，r3工具仍拒绝其黄区验收入口。

A3 Native先按[native_regions_a3_build_20261122.md](native_regions_a3_build_20261122.md)独立构建并审查产物，再由后继包提供机制步骤；不得用A2 DSO或旧r3参数绕过缺失实现。TP1、TP2在87分别验收，社区模型与现役权重的候选数量不要求相同。DeepSeek MC2 CPU准备和稀疏MLA结构夹具不纳入可执行候选。

## 早期增量记录

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
