# 阶段1收尾、Y4交付与Y5原件缺失：核对及处理

来源：负责人2026-11-25本聊天纯文本回传，无新附件原件；不为聊天摘要编造运行回执SHA。目标组合为黄区A3/Qwen3.8/0.26/CANN9.1/TP2，具体身份仍以每次正式文本为准。本记录不修改冻结包、不新增适配器补充件、不授予A3资格。

后到[控制证据更正](yellow_intake_20261125_control_evidence_correction.md)已明确：26fia没有补配置后的新失败，撤销“新产品缺陷”疑问；39op旧失败stdout未保存，改为黄区补归档后复测；Batch组内Barrier若位于run_canary则不符合串行B1契约，但exit5仍需内层错误。下方保留收到本轮摘要时的审查过程。

## 1. 阶段1状态：通过声明与缺陷定性分开

- 新增18.variance_under4：回传644、changed644、failed0、两个owner。新增15.alias_detach：回传64、changed26/64与64/64、failed0、两个owner。记为黄区声明通过、待正式效果/身份页；15的changed小于applied符合计数语义，不是失败。八项计数仍需逐owner/rank对应，不能从摘要推填。
- “10项通过”作为黄区汇总原样保留，待完整机制/变体/owner清单去重，不把命名变体与公共编号混算，也不自动变成10个登记。
- Batch确实是enable失败，与39op scope1的disable归属异常分开跟踪。当前只能确定宿主脚本在`run_cli(..., 'enable')`返回非零后抛出`MAIN_CLI_FAILED:5:`，不能确定底层故障位置。
- 41仍为harness漏映射，未进入机制测试，由黄区修正接线。
- 本次“26fia产品缺陷”与上一份“补齐Native路径/SHA后两个scope通过”冲突。缺少新的错误阶段/回执对应，不能覆盖上一结论为已确诊公共缺陷。需注明是旧问题汇总还是新失败；新失败按新运行证据另行审查。
- 39op scope1异常未定性；外部三项未开始。已在其它项通过的原件继续保留，不要求全组重跑。

## 2. Batch：先保留内层错误，再决定修复位置

`YELLOW_PRODUCT_DEPLOYMENT_FILES_CONFIGURED`只证明该步配置返回；每rank一条permit是正常形状的可能解释，不证明重复安装或有效许可。`acceptance_granted=false`不是许可失败的充分依据。空stderr也不等于无错误：CLI的结构化错误可能在stdout或session daemon中。没有这些内容时不能认定是金丝雀、许可、共享运行时或模型故障。

公共`ProductClient._enable`的顺序为：binding attach → arm_canary/宿主run_canary/读取canary，至多六轮 → prepare → register → check → enable → status。失败在宿主enable这一行，不等于Worker已经执行enable，也不证明任何故障注入已发生。Batch要求故障前连续三轮稳定的**串行B1**金丝雀，真实用户故障批次才并发；关闭后两轮串行金丝雀。并发金丝雀已有明确拒绝码`PRODUCT_BATCH_CANARY_REQUIRES_SERIAL_B1`。

由负责人转交黄区，先从现有失败原件提取，不立即重新enable：

1. 同一次CLI的完整stdout JSON、stderr、returncode；保留body的status/code/reason/error_type/refusals和每rank字段。宿主`run_cli`应先保存stdout/stderr再抛外层异常，不能只保存`MAIN_CLI_FAILED`。此处是黄区宿主诊断改动，不修改公共运行时。
2. session daemon完整异常链及此次enable前30秒至失败后60秒的Worker stderr；原件保留本地，回传脱敏文本、文件SHA和运行别名。
3. 最后成功的控制动作和失败动作的双rankbody；已有的canary observations（protocol、all_finite、warmup_valid、output/schedule比较结论）及实际run_canary顺序，勿从空状态补0。
4. enable失败后的cleanup/recovery、每rankowner/hook/lease/inflight、服务健康。缺少清理证据不能把同实例直接用于下一个owner。
5. 同一次包/实际安装源码/部署/许可SHA；脱敏部署中Batch的max_num_seqs、canary prompt列表与生成长度。许可条目分别对应哪个rank，是否仅是同名scope而非重复安装。

定性与处理：

|内层证据|处理|
|---|---|
|双rank结构化契约拒绝，CLI却exit5|检查黄区CLI是否使用body.exit_code；若公共类型转换丢失，作为r8公共修复；不要把所有RuntimeError统一改exit4|
|非串行金丝雀拒绝|黄区将run_canary逐条发送并等待真实完成，保持故障批次并发；拒绝后检查清理和引擎存活|
|PRODUCT_CANARY_UNSTABLE_IN_SIX|审查有限性、同输入与实际调度；记录基线不稳定，不放宽精确恢复或清零状态冒充修复|
|部分rank失败、内部异常、传输超时或回滚不完整|exit5可能符合契约；按完整traceback定位，恢复未证明时隔离该实例|
|缺少内层证据|未定性，不提交猜测性运行时补丁|

安全清理沿用[r8 Y1维护记录](r8_y1_defect_review_20261125.md)：先停止该实例新请求并drain，保留原session/owner走恢复；必要时退役同组API/Worker，不能直接删wrapper或租约、不能用remove_permit替代恢复。

本次在r8开发线复跑`test_product_control_refusal`与`test_product_batch`，unittest执行20项全部通过（含导入的site测试）；证明已有CPU契约路径，**没有复现或解决黄区这次exit5**。环境没有pytest，改用unittest，未安装新依赖、未占NPU。

## 3. Y4：既有r3已交付，解决交接版本问题

已读源码、解包并校验既有汇总ZIP；421个manifest成员的字节数/SHA全部匹配，汇总与四个目标内包CRC检查通过。GitHub远端冻结标签`yellow-delivery-20261125-r3`解引用为`a9ddc3f4d69578e6811bd1da420e6b87e6020fbd`。因此“r7.4.1原包没有”成立，“仍需Codex开发补件”不成立。

负责人转交已有[r3交付索引](https://github.com/spike677/vllm_ascend_op_fi/blob/yellow-delivery-20261125-r3/releases/YELLOW-DELIVERY-INDEX-20261125-r3.md)及[汇总ZIP](https://github.com/spike677/vllm_ascend_op_fi/raw/refs/tags/yellow-delivery-20261125-r3/releases/yellow-delivery-20261125-r3/yellow-delivery.zip)。汇总8132026字节，SHA256：

`62fdd31b10fc8f3b0b2751bb85f8a1e31e11653a1e11645d4eccf2392231802b`

以下路径均相对汇总ZIP解压根目录，分别独立解压；不把多个同名`reference_adapters`目录盲目叠加，也不覆盖r7.4.1安装源码。

|用途|既有内包|SHA256|
|---|---|---|
|12：lm_head坍塌/交换两档|releases/yellow-reference-lm-head-20261125-r1/reference-lm-head.zip|5cd4aaeedb5920d6e0f7138753b8fc99d92b0cf071a4df2345c8cd470df0c7f1|
|32：E4M3/E5M2|releases/yellow-reference-vision-fp8-20261125-r1/reference-adapters.zip|075954a73aab8a1b953d0dd43b03b0fa07eb46a8a2fef0c5b80585e9ea5efd87|
|34：尾数5/3|releases/yellow-reference-projection-round-20261125-r1/reference-adapters.zip|9e40e3d2cfa224accc1c4332b2760787fa4416b1fab92bb2f8bc258d08457217|
|T2：控制拒绝guard|releases/yellow-reference-control-20261125-r1/reference-control.zip|c79a41de6fe08fb0b72c57eddcf4d1130f654c51f56c08a09ba194cc1678f0d1|

按内包HANDOFF和`docs/worker_core_v1/reference_12_32_34_handoff_20261125.md`核对真实site与源码身份后接入。T2包含`reference_adapters/control_guard.py`和`dcat_reference_control_guard.py`，只处理参考适配器控制拒绝，不修复公共Batch。12使用v2持久权重协议，不用旧transient列错位路径替代；32/34为现象代理。A2审查不是A3资格。后续适配器修改、CLI映射和本地测试归黄区，不再新增蓝区补充包。

## 4. Y5：保留六条门禁，补采缺失原件

已核查`extract_fault33_existing.py`：先检查calls长度等于6，再读取bin，最终要求`{128,512,2304} × {rank0,rank1}`全部覆盖。两条方案直接触发`THREE_SHAPES_TWO_RANKS_REQUIRED`；六条中填不存在文件也不能得到正常“缺字段”输出。此前只提2304两条的建议不能作为此工具执行方案，现更正。

处理决定：**不把两条变体当作N2完成，不降低三形状门禁。**现有2304两rank原件封存，先用已有`review_launch_capture_v2.py`分别审查完整性并记录SHA/缺项；7296条数量本身不证明完整、无丢弃或关联唯一。这可形成局部证据索引，但不能进入窗口二控制重算。

128/512两组双rankbin已删除且无法找回，必须补采缺失形状，不能从API JSON重建bin/header/tiling，也不能复用2304复制成六条。补采前锁定采集器与配套审查工具：旧extractor只接受已审查`1fecdc87…`来源，不可把r2新采集器产物冒充旧来源；r2按其binding sidecar与完整性规则审查。跨版本证据分别保留身份，若不能证明同一部署、目标模块、唯一API/launch关联和同一候选布局，则采用一次新的三形状完整窗口，不能强拼六条。

补采原件在审查完成前不删除；每形状独立目录/清单，保存真实API记录、双rank映射、launch/header、完整性审查和源码/构建/加载身份。pointers指向真实JSON键；blockDim来自80字节header；未审定tiling和未观测block_table有效范围保持null。六条数量满足后仍需字段边界、三形状自洽及每组控制重算≥99%等既定门禁，不把数量检查当作放行。

具体采集方法沿用r3中`fault33_candidate_windows_20261124.md`与`fault33_n2_existing_capture_20261124.md`，本次只更正原件缺失后的路径，不重发工具、不让黄区手工拼造N2。由负责人安排黄区针对性只读补采，其它正式测试继续。

## 当前完成与待补

已解决Y4“实现未交付”的定位，已更正Y5两条提取的不可执行建议，并验证现有Batch控制CPU契约。Batch真实根因和26fia新旧汇总冲突仍需同次运行文本，不能宣称修复完成。新增通过项进入待审查队列；没有新增正式登记、没有冻结r8、没有改变机制文件。
