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
|故障库9/10/13/19/23/34/35/39|逐项探测与最终结论待黄区回传；有真实注入点才注入，不用公共包有无注册代替判定|
|HCCL与黄区其余条目|仅引用黄区实证；统一工具需要预加载路径真实回执适配，不伪造公共计数|
|D包/8|V4仅为8与DeepSeek专属黄区项重启；其他公共V4项仍暂缓|
|r8|最终r7.x+D包+审查通过登记+可信bootstrap+CLI helper+免许可；冻结取决于真实证据，不以A2代A3|

## 负例口径

产品CLI缺许可、未知profile/名称：exit4并保留拒绝码，引擎存活；直接RPC允许HTTP200+结构化拒绝body。未知profile原HTTP200不能判为缺陷。重复enable不得破坏已有会话；重复disable必须安全。资源清零和同引擎正对照均必要。

## rank1结案

最新黄区回传确认一处为自建rpc_pair脚本把rank1 capabilities与前次probe错误配对；另一次瞬态未复现，后续22轮调用链全部对齐。按“接线问题已定位 + 单次未复现保留记录”结案，不宣称修复了公共ProductClient错位缺陷。蓝区公共路径99次调用/198回包与落盘一致。另做的未捕获异常队列污染实验是独立风险证据，不能套为黄区事件根因。

## 审查

结论只有通过/不通过/需退补。核对组合身份、owner/provider owner、rank、请求与调用关联、八项计数、消费者、三项恢复、原件SHA。changed<applied允许；image_digest按4.9非必需可空；耗时只记录。seq若只是宿主顺序，必须标注，不能当成wire回包对齐证据。当前不新增YELLOW_DEDICATED_INSTANCE记录。
