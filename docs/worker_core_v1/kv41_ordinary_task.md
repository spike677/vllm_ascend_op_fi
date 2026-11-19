# 41普通任务：真实KV尾列的单点推理作用域

本次从既有Postwrite41WorkerProvider抽取普通入口，保留Native合法地址尾列机制、源码/调度/活KV对象/原始consumer恰一次检查。用户仍只给FaultConfig(41, 'tail_column')，无层名、地址或proof参数。部署用install_fault41_task_binding；旧research binding不能进入普通路由。

阶段为normal、normal_repeat、lossless、fault、off、health六个真实请求。前两次健康观察用于原机制稳定选点，lossless保留无损Native门槛；F只对首个实际稳定writer/cache/consumer目标作用一次，不代表所有匹配点。请求ID由正常vLLM enqueue获得，advance只登记；实际root回调自动结束阶段，query不collect、不推进、不执行推理。完整28层等数量由实际模型决定，不写死用户层名。

OFF保留真实原调用并验证完整root hidden精确恢复，随后移除所有机制hooks/API/Native lease；health使用独立只读首decode观察器，核对同Worker/model、真实scheduler输入前缀及完整root hidden哈希。health不重新安装故障机制；结束移除观察器。Native效果同时保留changed_elements、selected_elements、实际consumer和restore/release回执。

范围限制：provider只对root hidden作完整哈希对照，**不声称采集logits**。实机探针另核对正常、OFF、health及注销后完整生成token/text；普通入口不默认启用msprobe，不产生完整张量文件，E5保持未评估。若需要统计独立首差，使用独立采集验收，不能用命中日志替代。

两个作用域使用同一Worker、各自新的Runtime owner/provider、独立观察/无损/F/OFF/health。关闭后精确恢复不能由退出容器替代。F不重试；失败保留原始原因并最多接入新的OFF/health恢复。连续相同原因失败两次停止该项。

当前开发状态：CPU公共Host/Runtime及真实Torch writer/consumer契约测试已通过；**新普通路由的NPU验收待执行，默认无已验收部署**。可信部署许可只开放此次验收，不产生接受记录。旧Source60证据和研究协议原样保留；此开发不写入冻结RC2，也不混入仅登记的RC2.1。
