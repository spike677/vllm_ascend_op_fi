# 33 A3补证与适配：不再限时结案

本页覆盖旧 `fault33_a3_execution_addendum_20261124.md` 的时限和结案段落；技术检查保留。A3 Qwen3.8 TP2 / v0.26.0rc1-a3 / CANN9.1，33保持关闭直到控制重算通过。

现有材料只给出serializer缺失、sparseMode=3、7928条且最大serial8191；没有能证明匹配源码的完整已加载operator/host库身份。**不能从同名CANN版本、A2 tiling或master分支猜出A3布局。** 当前阻塞归属：负责人协调部署方/软件包提供者提供下面的来源链；Codex据实匹配后实施适配。

1. 在实际Worker进程记录 `/proc/self/maps` 中对应operator/host库的解析路径、SHA256、ELF build-id及运行时CANN来源。先确认真正加载，不能只hash目录里某个同名库。
2. 对应发行包/安装清单/构建收据与其匹配的ops-transformer或CANN源码commit/source包身份、逐文件SHA。建议从该已安装包的供应方取得source manifest、host tiling serializer、key注册和kernel模板。若发行物不附源码，由负责人协调可合法取得的匹配源码；Codex不将未知包视为匹配。
3. 原始API参数、真实Q/K/V与mask/序列边界/页映射保留黄区；文本回传参数和SHA即可。沿匹配host条件逐步证明 sparseMode=3 →实际key→模板→serializer→S2。scheMode和sparseMode不可混用。
4. 旧采集器两处8192上限和其他过滤都可能丢记录。用每rank独立短窗口或经审核的扩容/过滤采集器，完整报告seen/selected/written/filtered/oversize/open_error/short_write/capacity_drop，结束flush并核对文件header长度。没有drop计数不能宣称捕获完整。不得只增上限或从截断stdout补造raw。
5. 取得完整实际分组后，对固定真实Q/K/V的完整目标输出做控制重算，逐位一致≥99%，同合法key集合。通过才进入两个owner正式注入测试，真实模型输出变化、消费者、区域/资源/金丝雀恢复分别成立。

回传按rank/窗口分≤4KiB文本，含源文件与原件SHA、总段数、每段SHA、不可观测项。缺任何一项则结论“需退补，升级负责人”，不结案、不改阈值。适配将作为r7.4新增项；若必须改共享请求路径，要列出受到影响的公共能力及最小复测集。
