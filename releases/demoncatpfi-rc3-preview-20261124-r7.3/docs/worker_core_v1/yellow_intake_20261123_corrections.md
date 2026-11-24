# 黄区纠正与后续计划：接收记录（2026-11-23）

来源：负责人指定的 Git 外存档 `yellow-corrections-followup-72d6a44c/黄区纠正与后续计划记录_原文.txt`，8987字节，SHA256 `de43682fe06888cb0940d717bc4f6c3fb536c0bb33cda6ab85af87d6e36baafa`。随附任务文本23164字节，SHA256 `8b25e64c0c44af10391d281ce1068d4cb64d8ff5fabe013b836981e16d6e2950`。原文未修改、不复制进Git；本页不传播机器地址、硬件编号或容器名。

## 材料性质与字段完整性

本批是黄区工作安排及分类纠正的文字汇总，可用于职责/范围对齐与后续验收计划，**不是新增实机验收或组合登记证据**。黄区所称接收目录是其本机路径；蓝区本次检查没有同路径目录，也没有15份P0原文。

未提供逐项新运行回执、实际软件/模型制品完整身份、两个作用域每rank计数、消费者/恢复/负向结果及其文件SHA；不能据汇总将15项标为产品通过。P1未完成的分类按“待探测”保留，后续以黄区P1结果为准，不要求现在重传15份原文。DeepSeek正式噪声、全rank只观测、decode重计数和Native完整提取仍待回传。

## 已记录的决定

1. 总表32条=公共历史17条+黄区自有15条。公共12项交付、5项不交付；公共结论及证据见 [public_17_final_status.md](public_17_final_status.md)。黄区最终五种状态按环境分别记，不能把计划或历史实验记为产品通过。
2. 黄区量化类改按模型适配处理；05/08先核查Conv2d与视觉Conv3d/linear适用性，不提前关闭。重复项按实际机制和注入点处理；float_matmul合并39mr为黄区已声明决定，其余条件式合并仍须P1确认。operator_inspect若只读则为诊断工具。
3. 黄区自有15项的开发、探测、验收全部归黄区；Codex不接手。外部接口契约暂缓，不进入r7。不向黄区发送执行消息。
4. 黄区CLI唯一入口，公共走ProductClient，自有走黄区Worker扩展。全局同时一个故障，由黄区CLI协调两边状态；这强于“同模块不共存”。验收需包括公共/自有交叉冲突和重复enable/disable的结果。
5. 黄区优先级仍为r6 A组、B组标定/只观测、Native步骤；自有P1只读利用空档，产品验收在前置结果通过后执行。r7 N/D/P主线不改。

## 蓝区源码核对与两处澄清

**数量按版本固定。** 只读解析冻结标签 `demoncatpfi-rc3-preview-20261123-r6`：`MECHANISMS=12`，`PRODUCT_CANDIDATES=8`。开发提交 `5deb8c50`：仍12个公共机制，产品候选11个（新增Native三项，A2限定）。原文“11已作废”仅适用于此前误称r6数量，不能用于否定r7已完成的A2结果。

**不要把黄区字段误写为公共API字段。** 冻结r6的 `src/` 和 `fault_injection/` 中未找到 `demoncat_status`、`registered_faults`、`module_owners` 这些接口/字段名。实际 [ProductClient.status](../../src/demoncat/runtime/vllm_ascend/worker_extension/product_host.py) 返回 `enabled/requests/recovery/canary/cleanup/product_qualified`；Worker [product_binding](../../src/demoncat/runtime/vllm_ascend/worker_extension/product_binding.py) 的status为按rank的请求观测、cursor、active_request_id及failure。黄区名称若属于内源Runtime，须由黄区CLI映射，不应声称它们由r6公共status直接提供。

这不改变“黄区CLI负责互斥、Codex不新写对接代码”的决定；单次查询不是跨客户端原子互斥证明。黄区需在唯一CLI控制通道串行化占用/释放并验证跨路径冲突，查询失败或恢复未完成不能当空闲。蓝区没有黄区CLI源码与并发测试，不能在此宣称互斥已经验证。

本轮只增加文档及只读核对，不启动NPU、不改准入/实现、不修改冻结包、不新增接口契约、不将本文作登记依据。r7 N批A2及八项回归结果见 [r7报告](r7_native_a2_20261123/README.md)；A3 Native冻结门禁仍等待完整材料审查。
