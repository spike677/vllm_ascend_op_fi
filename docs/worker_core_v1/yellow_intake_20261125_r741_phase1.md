# 黄区 r7.4.1 Y1 安装与阶段1回传审查

来源：负责人提供的 Y1 交付安装与阶段1纯文本汇总，原文保存在仓库外接收目录。原文2870字节，SHA256 `9eeb561b898daa96c26a353c237e029bf69200f964f03da94ac28dba7ee7d6a7`。本记录不复制机器地址、设备编号、容器名或黄区内部目录。原件在黄区；此回传不是完整运行回执或逐项提取页。

## 安装结论与逐项状态

最新[控制证据更正](yellow_intake_20261125_control_evidence_correction.md)：26fia明确无新失败，补配置后通过声明待审查；39op失败stdout未存档，需要宿主补日志后定点复测；Batch金丝雀组内Barrier存在接线疑点，尚不能定exit5根因；V4启动受阻，不算D0/D1证据。

最新收尾回传另见[阶段1/Y4/Y5处理记录](yellow_phase1_y4_y5_resolution_20261125.md)：18.variance_under4和15新增通过声明；Batch在enable失败，根因待内层body；41仍为harness漏映射；26fia“产品缺陷”与上一轮补配置通过存在待核对的运行归属。下表保留初次汇总，不代表后到结果。负责人所报10项通过暂记汇总，未取得逐项清单及正式文本前不扩展登记。

黄区声明交付索引及传输397/397、发行清单352/352通过；三wheel版本分别为0.4.0rc6.post6、0.3.0rc6.post6、0.3.0rc6.post6。交付引用为 `yellow-delivery-20261125` / `21bb22a5`；此交付标签与公共r7.4.1源码/发行身份分开记录。许可只给了SHA前缀，不能作为完整身份校验值。

|机制|黄区汇总结论|Codex本次审查状态|
|---|---|---|
|18.api_half|PASS，644 obs/elig/app/cons/rest，changed643，failed0|需补逐owner/rank提取页，未登记|
|22.wrong_row|PASS，64，两rank两owner，failed0|需补逐owner/rank提取页，未登记|
|39.matmul_range|PASS，1216；changed511/512与508/512，failed0|需补计数归属及效果页，未登记|
|28.fia.denominator_replay|PASS，64|需补逐owner/rank提取页，未登记|
|17.spatial_padding|PASS，1/1，图片，三项恢复true|需补逐owner/rank提取页及输入身份，未登记|
|20.buffer_substitute|PASS，832|需补Native区域/运行库身份及提取页，未登记|
|39.output_proxy|scope0通过，scope1 disable失败|缺陷候选，需退补现场，不登记|
|41.tail_column|disable exit5|缺陷候选，需退补现场，不登记|
|26.fia_destination|configure HTTP500|未定性，需退补现场，不登记|
|Batch、15、18两方差档、三个外部示例|未跑|保持待测试；外部适配器由黄区自行闭环|

changed小于applied本身不是失败；两者分别统计产生新数值变化和实际应用的调用。耗时只记录、不作门禁。

## 六个声明通过项需要补充的文本

每项提交现有一页效果工具和产品提取工具生成的文本，不回传原始数组、不要求summary全文。两个owner、两个rank必须可以逐条对应；工具分页的文本原样保留，不能手工把计数拼成一个“通过”标记。

1. 机制/命名变体、产品模式、发行标签/源码提交/完整许可清单SHA、审查器与提取器SHA、审查结论、提取文本自身SHA。
2. 两个owner、generation及每个rank的对应关系，同组Worker的身份连续性；内部请求ID、prefill/decode边界及请求范围。每个owner/rank八项计数、changed_elements和不适用原因。
3. 真实消费者证据，区域字节恢复、资源清零（租约/owner/hook/wrapper/inflight）和新请求精确金丝雀；三者分别给结果。
4. 效果工具的top-5 logprob变化及其请求/阶段对应关系。top-k logprob不是完整raw logits；没有采集的字段填null，不补造，不要求token必须改变。
5. 该次运行实测身份：硬件原名与逻辑设备、TP/并行参数、软件及实际CANN、config/量化描述/全部权重索引SHA、参数范围。Native项另给已审查库SHA与实际映射CANN依据。image_digest可按既定方案B保留null，不以镜像备注代替实测字段。
6. 原始回执与summary的文件名、字节数、SHA；两个owner与原件的对照表；负例的拒绝码/exit4、拒绝后引擎存活与正对照，以及资源清零。

可用于：确认黄区声明的安装进度、安排缺陷定位和退补；暂不能用于 `YELLOW_DEDICATED_INSTANCE` 登记。缺失身份不从历史启动脚本、A2结果或默认值推填。此次没有新增任何A3登记。

## 两个缺陷候选

### Y1-OWNER：39op/41第二owner关闭

原报错：`ContextRefused` / `PRODUCT_AUTOMATIC_WRAPPER_OWNER_CHANGED` / `PROVIDER_ACTION_FAILED` / `recovery_required=true` / exit5。

源码位置为共享 `product_observer.AutomaticRequests.close()`：核对runner上的 `execute_model`、`sample_tokens` 是否仍为当前观察器所安装的**同一个函数对象**，不一致就保留归属并拒绝拆除。此码不是Native库错误。r7.4.1产品39op实际走 `Projection39Site + RegionLeases` 的Torch改写；41走Native KV区域，两者共用自动请求观察器。

CPU验证：正常关闭旧owner、安装新owner后再次幂等关闭旧观察器，不会移除新包装器。若旧包装器未关闭，新owner在install时即以 `PRODUCT_AUTOMATIC_BOUNDARY_ALREADY_OWNED` 拒绝。由此不能仅凭scope1 disable失败就归因于“未detach”；也不能反向认定黄区驱动无错。运行期间包装器被替换、并发清理/诊断接线、实际所加载代码等仍需现场核对。

退补字段：scope0从enable到disable、unregister、runtime detach、binding detach、remove_permit的实际顺序及回包；scope1的同样调用序列；失败的完整双rank body（worker/rank/owner/generation/action/request序号）；驱动与可信RPC包装源码SHA和相关调用片段。若有现场只读观察，记录每个runner边界当前函数与期望包装器的对象身份、module/qualname、闭包observer/owner，以及变更发生在哪两个控制边界之间。不要为了观测先删除或覆盖包装器。

### Y1-CONFIGURE：26fia HTTP500

`configure_product_deployments → load_product_deployment → configure_product_profiles` 对20、26fia、41共用Native准入。此阶段核对路径、文件SHA、设备/TP、部署身份和实际CANN，尚未构造 `NativeFIA26Site` 或执行注入。缺少traceback时不能认定FIA、Native写入或owner切换为根因。

退补范围：失败configure前30秒至后60秒的API异常栈和全部Worker stderr（rank、PID只用于黄区本地关联）；准确RPC方法/动作、脱敏后JSON字段及字段类型、Worker返回body、退出码和健康检查；三wheel/实际加载文件SHA；部署JSON与许可/source清单完整SHA；库路径可读性、字节SHA、CANN挂载目标和实际映射身份。输出有界分页，保留异常类型与调用栈完整性，不以“HTTP500”一行替代。由负责人转交，本轮不直接联系黄区。

## 后到现场补充（覆盖前文中被更正的汇总）

负责人随后提供9148字节补充文本，SHA256 `3696b01f08cacad8c23eb9f0375a1d390df15a1c0b2a3762dd5034fccf24cc19`。原文留在附件/接收目录，不提交原件。最新判断如下：

- **41**：黄区明确更正为harness的 `--mechanism` 缺41，`invalid choice`发生在测试前。此前“disable exit5”是误报。记为**黄区CLI/harness接线问题、尚未验收**，不是公共运行时失败；黄区自行补映射并测试。
- **26fia**：早期forkserver信号异常叙述与后来服务器RPC均200、无Worker异常栈的记录缺少同次运行关联，不能拼成“FIA＋预加载库信号冲突”的根因。最新明确拒绝码为 `PRODUCT_NATIVE_REVIEWED_LIBRARY_REQUIRED`；补齐库配置后黄区声明两scope、两rank各64/64、failed0且恢复通过。暂定为**部署配置/可信bootstrap接线问题，黄区声明已恢复**，仍需逐项效果/身份页才能登记。拒绝码可由库路径类型/绝对路径、未匹配审查SHA等触发，不能仅凭该码确认是哪一个字段原值错误。
- **39op**：scope0恢复、detach及许可撤除均有具体文本；scope1 enable、两rank预热及四请求status成功，disable两rank均到达provider并报包装器对象归属失败。它支持“不是缺许可/没有进入推理”，但**不证明ticket owner错配**。详见[源码语义分析](y1_phase1_fault_analysis_20261125.md)。当前仍为待定位，不接受建议中的“permit复用已登记owner”作为修复。

补充中39op原回执字节数带问号；41日志的字节数栏不是有效整数。二者均不能用作完整原件身份。26fia原回执SHA已给出，但仍未提供可复算的两个owner/rank效果页。具体补充需求收敛为39op包装器边界证据、26fia成功页及各声明通过项的一页总结；不重复索要已经收到的错误码与清理顺序。

## 后续

先在原r7.4.1运行时上进行A2复现，结果另见 [Y1复现与维护记录](r8_y1_defect_review_20261125.md)。确认公共缺陷才在r8做最小修复，保留原标签。黄区其它阶段继续；仅有recovery_required的实例先隔离并按安全恢复步骤处理。
