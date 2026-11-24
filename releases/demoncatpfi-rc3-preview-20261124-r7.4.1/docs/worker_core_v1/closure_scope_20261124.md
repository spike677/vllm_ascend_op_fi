# 收口台账：2026-11-24最新决定

公共13项：18、22、39op、39mr、28 denominator、17、Batch、15、41、26fia、20、33在A3 Qwen3.8 TP2完成注入测试与登记；8在A3 DeepSeek V4-0731 TP8+EP完成。TP1不是必要条件，已有结果保留。

不交付4项：25、26 residual、28 FA legal、28 FIA legal。15为B档ND视图别名断开，不含私有格式触发条件；changed为调用次数，changed_elements单列。

|事项|当前结论与下一步|
|---|---|
|r7.3.1|控制拒绝维护与统一效果工具；A2回归后冻结；黄区master从本版直接开始|
|公共11项黄区r7.2|收到过程汇总，缺逐组合可复算原件提取；不以汇总计数代替登记；正式结果改为r7.3.1一页总结|
|33|A3尚关闭，必须补齐身份/serializer/sparseMode3/完整捕获及≥99%控制检查；缺材料升级负责人，不再限时结案|
|外部v1七项|公共模型/采样示例A2已验证；黄区各适配器仍需实际注入测试|
|外部v2四项|最新批准开发；weight/faulty_requant、weight/param_perturb、kv_update_misdirect、kv_rope_quant_loss；未验收前关闭|
|四项黄区新增设计|init_routing_cutoff、cp_padding_restore、conv_transpose_padding、faulty_requant_rounding：仅收到源码/CPU检查汇总，无最终注入证据|
|故障库41行|以fault_library_coverage_20261124.md为唯一当前口径；库行号不是公共编号。库8=CP padding、库17=转置卷积padding；27→公共15、41→公共20、39→公共39mr。C类含新增30→18方差误差代理，待实测；不以公共17冒充原转置卷积机理|
|HCCL与黄区其余条目|仅引用黄区实证；统一工具需要预加载路径真实回执适配，不伪造公共计数|
|D包/8/第13行|负责人已批准第13行router量化噪声变体与公共8一起进入D包；0.32路线附外部provider v1，依赖D0/D1与噪声审查；其他公共V4项仍按原范围，未验收不登记|
|r8|最终r7.x+D包+审查通过登记+可信bootstrap+CLI helper+免许可；冻结取决于真实证据，不以A2代A3|

## 映射范围补充决定

2026-11-24映射范围补充决定：第37行由D14“仅配置说明”改为“配置项说明＋faulty_requant舍入命名变体测试”；第7行仅在Qwen用MLP部分token行置零，不做V4 combine扩展。蓝区维护公共说明与接入边界，黄区实现自有变体并实测。三项范围均已确认，验收状态仍待实证；见[d_package_scope_20261124.md](d_package_scope_20261124.md)。

## 负例口径

产品CLI缺许可、未知profile/名称：exit4并保留拒绝码，引擎存活；直接RPC允许HTTP200+结构化拒绝body。未知profile原HTTP200不能判为缺陷。重复enable不得破坏已有会话；重复disable必须安全。资源清零和同引擎正对照均必要。

## rank1结案

最新黄区回传确认一处为自建rpc_pair脚本把rank1 capabilities与前次probe错误配对；另一次瞬态未复现，后续22轮调用链全部对齐。按“接线问题已定位 + 单次未复现保留记录”结案，不宣称修复了公共ProductClient错位缺陷。蓝区公共路径99次调用/198回包与落盘一致。另做的未捕获异常队列污染实验是独立风险证据，不能套为黄区事件根因。

## 审查

结论只有通过/不通过/需退补。核对组合身份、owner/provider owner、rank、请求与调用关联、八项计数、消费者、三项恢复、原件SHA。changed<applied允许；image_digest按4.9非必需可空；耗时只记录。seq若只是宿主顺序，必须标注，不能当成wire回包对齐证据。当前不新增YELLOW_DEDICATED_INSTANCE记录。

## B1—B6蓝区补齐

两个18固定gain档、module-input和DynamicQuant参数例已通过A2/0.26/CANN9.1/Qwen3.8 W8A8/TP2两个owner。状态诊断与15风险边界、排空复测、41行台账及33只读取证补充件已整理；见[r7.4.1蓝区收口](r741_b1_b6_closure_20261124.md)。A3资格和r8/D包不由本轮自动完成。
