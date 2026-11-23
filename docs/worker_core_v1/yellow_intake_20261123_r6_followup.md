# r6 A组、Native全文及故障库映射接收复核

2026-11-23。原件位于负责人指定的Git外存档；文件SHA及逐页机器复核见 [yellow_r6_native_review_20261123.json](yellow_r6_native_review_20261123.json)。不记录机器地址、设备编号或容器名；不把黄区文本当作蓝区实测。

## A组：目前不能登记产品组合

黄区声明r6独立安装384文件通过，TP2七项通过、TP1六项通过且Batch恢复失败。完整产品提取仅收到TP2 18的3页：scope0/rank0、scope0/rank1、scope1/rank0；原文称“四页”但scope1/rank1未出现。

三页的规范化extract SHA均匹配；每页eligible=attempted=applied=consumed=restored=1932，changed=1931，failed=0，资源清零及canary_exact为true。审查器SHA `605dfce0…`匹配已交付 `yellow-tools-20261123-r1`，不是r6冻结包旧审查器的字节；不混淆这两个来源。缺许可和未知profile两项负向明确尚未执行，不能补true。owner仅有文字中的短前缀，不当作完整owner身份。

**由负责人转交的补传清单**（本页不发消息）：

1. TP2 18缺失的scope1/rank1原样提取页；该组合两个scope、各rank完整owner及对应回执SHA。
2. 其余13个组合的提取：TP2的22/39op/39mr/28/17/Batch；TP1的18/22/39op/39mr/28/17/Batch。TP1 Batch保留失败提取，不要求把失败改成通过。
3. TP1、TP2分别补缺许可和未知profile负向结果，并绑定机制/许可/profile、包与回执身份。没有跑过的保持未执行。
4. 各组合完整硬件原名、实际软件栈、模型config/量化描述/全部权重索引SHA、参数范围；不能由TP2 18示例补齐其他组合。
5. 黄区CLI接线只有源码SHA，没有源码正文；不能据此声称蓝区已审查CLI实现。保留黄区“按ProductClient接线”的文本声明层级。

每页原样JSON、≤4KB，完整回执留黄区；不要求summary全文。审查通过后才能在后继版本登记产品组合。当前产品登记新增0个。

## TP1 Batch：恢复失败已确认，根因未定

开启前6次金丝雀逐位相同；关闭后8个prompt的第1步hidden哈希均不同且有限，黄区审查结果为 `PRODUCT_RUN_INCOMPLETE`。这足以说明该轮恢复证明不成立，不足以断言注入残留或调度变化。按 [TP1诊断执行单](batch_tp1_recovery_diagnostic_20261123.md)区分；不重启原失败进程来覆盖原证据，不降低精确恢复标准，不在未定性时开r6.1修复分支。

## Native：接受构建身份文本，运行资格仍需检查

Native行原文含Markdown传输转义 `\\~`，不是合法JSON。仅在解析副本中移除此转义、保留原件不变后，规范化SHA严格等于声明的 `5f863898a877f6dd0f181949e66cfcce07e3a7673fba5632a594dc61d21863d2`。没有改任何身份值或补默认值。

- 5个源文件完整SHA与r5、r6、当前开发线均一致；C++未改。
- SDK版本文件正文与其SHA匹配，声明9.1.0；g++11.4.0，10条构建/CPU/符号检查返回0。
- 库 `d9aa0d94…`、29088字节；原receipt `ff19425f…`、9057字节；recipe为 `postwrite-stream-object-v2`，来源r5源码提交已提供。
- `cann_runtime=null`及`NO_DEVICE_BUILD_NO_RUNTIME_OBSERVATION`符合无设备构建事实。接受为黄区文本构建证明，不改成运行时9.1.0、不代表A3 NPU验收已过。A3启动时仍须核对实际加载CANN、库SHA、源码与许可身份。

N批可进入A3候选接线/执行版准备，直接复用已审查库，不要求重建。r7尚需完成候选打包和相同发行门禁后才能冻结；构建文本通过不自动改变现有A2限定代码。

## B组与映射

本次没有正式噪声、全rank只观测、decode重新计数结果，不把89算子或旧logprob汇总提升为证据。固定卡组上的标定与1.5安全系数保持原计划；不从蓝区操纵黄区设备。故障库20→公共20，28→公共28 denominator，38→黄区HCCL；3/37作为黄区配置项说明引用，不另开发注入。
