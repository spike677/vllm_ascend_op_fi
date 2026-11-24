# 黄区r7.2过程回传审查（2026-11-24）

来源：负责人提供的r7.2独立安装、公共TP2产品流程、负例和探测结果文字，原件保留仓库外接收目录。环境为黄区A3 Qwen3.8专用实例；不当作蓝区A2证据。本页不收录机器地址、硬件编号、容器名。

|材料|审查结论|用途|
|---|---|---|
|公共11项两个owner汇总|需退补：过程文字不含逐组合可复算提取及完整身份/owner/rank/原件链|历史结果保留；最新决定改在r7.3.1正式测试后回传一页总结，不要求重交旧四页|
|未知profile HTTP200+PRODUCT_BINDING_REFUSED/PRODUCT_PROFILE_UNKNOWN|通过负例口径核查；HTTP200本身不是缺陷|直接RPC结构化拒绝可接受；CLI须exit4|
|缺许可实际被拒、宿主ValueError被CLI映射5|不通过旧CLI映射；已定性公共类型契约+黄区CLI接线|r7.3.1提供ProductContractRefused/product_control_reply；黄区采用映射后验证|
|非串行Batch金丝雀导致EngineDeadError|不通过引擎存活要求；公共维护修复|r7.3.1前置检查及控制拒绝锁存，不能仅要求换脚本掩盖|
|CLI99 exit4、清理与后续正例汇总|通过过程口径核查，不替代逐组合登记|最新部署级负例统一跑一次|
|rank1回包|接线问题结案 + 单次未复现保留|黄区自建rpc_pair错配一处；随后22轮正常。不能再把蓝区故意抛异常实验当作黄区已证明根因|
|外部适配器11项CPU编译/检查|需退补实际产品证据|七项瞬态走v1；四项持久走获批开发v2。CPU导入和factory门禁不是注入验收|

报告给出的计数不能单独作为通过依据。changed<applied（包括18、39mr、15、Batch）允许，表示部分写入没有新增数值变化；仍须证明eligible无失败、消费者、区域字节恢复、资源清零、金丝雀与正确rank/owner关联。

四项新设计 init_routing_cutoff、cp_padding_restore、conv_transpose_padding、faulty_requant_rounding：当前仅有源码/CPU检查文字，无实际注入最终结论。库行9、10、13、19、23、34、35、39继续逐项探测；两套列表可能重叠，不重复计算成果。Codex不替黄区填写其自有机制结论。

最新任务已覆盖旧的“V4/8暂缓、v2仅设计、33限时结案”：V4仅为8及DeepSeek专属条目重启，v2开始开发，33需补齐A3取证与适配。其余公共V4项仍暂缓。

当前不生成YELLOW_DEDICATED_INSTANCE登记。不向黄区直接发送消息或要求执行；补充项由负责人转交。
