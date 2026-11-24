# 41普通任务：真实KV尾列的单点推理作用域

本次从既有Postwrite41WorkerProvider抽取普通入口，保留Native合法地址尾列机制、源码/调度/活KV对象/原始consumer恰一次检查。用户仍只给FaultConfig(41, 'tail_column')，无层名、地址或proof参数。部署用install_fault41_task_binding；旧research binding不能进入普通路由。

阶段为normal、normal_repeat、lossless、fault、off、health六个真实请求。前两次健康观察用于原机制稳定选点，lossless保留无损Native门槛；F只对首个实际稳定writer/cache/consumer目标作用一次，不代表所有匹配点。请求ID由正常vLLM enqueue获得，advance只登记；实际root回调自动结束阶段，query不collect、不推进、不执行推理。完整28层等数量由实际模型决定，不写死用户层名。

OFF保留真实原调用并验证完整root hidden精确恢复，随后移除所有机制hooks/API/Native lease；health使用独立只读首decode观察器，核对同Worker/model、真实scheduler输入前缀及完整root hidden哈希。health不重新安装故障机制；结束移除观察器。Native效果同时保留changed_elements、selected_elements、实际consumer和restore/release回执。

范围限制：provider只对root hidden作完整哈希对照，**不声称采集logits**。实机探针另核对正常、OFF、health及注销后完整生成token/text；普通入口不默认启用msprobe，不产生完整张量文件，E5保持未评估。若需要统计独立首差，使用独立采集验收，不能用命中日志替代。

两个作用域使用同一Worker、各自新的Runtime owner/provider、独立观察/无损/F/OFF/health。关闭后精确恢复不能由退出容器替代。F不重试；失败保留原始原因并最多接入新的OFF/health恢复。连续相同原因失败两次停止该项。

当前开发状态：CPU契约及124普通入口两个作用域已通过独立审查，仅登记该蓝区A2/Qwen3-0.6B BF16/TP1串行eager组合。许可自身未创建接受记录。旧Source60证据和研究协议原样保留；此开发不写入冻结RC2，也不混入仅登记的RC2.1。


## 124 ordinary41-a3独立审查

源码105fb2c10abfcacc35c40fe098450c04e02a5981，清单2e723e2cb8513bd01e690cb9f2972dbf4015c8986476c75b3dd7a89f2e86f87d。
运行0.18/torch2.9/torch_npu2.9/CANN8.5.1、910B4(A2)、Qwen3-0.6B BF16，单物理卡5。Native构建c62b223b未修改；原库SHA8876f2b99e70a9df8201b6ae0ace2cc4be8702afcf6703f9b72d1dda3bc26622。

同一Worker的两个独立Runtime owner均完成六请求；每次F在实际首个稳定Attention live V slot改16个BF16元素（8 KV头各尾部2列），实际consumer恰一次。两次F均改变完整root hidden，OFF及移除机制后的health恢复基线完整hash。全部请求生成token均为[576,4226]，文本相同。最终Native lease、owned API、hooks和inflight均为0，Host DETACHED，binding和验收许可移除。

[独立总结](../../experiments/worker_core_v11_20261118/ordinary41_a3_20261119_summary.json)保留原件SHA、两owner/permit、14项实际部署、每阶段hidden hash及限制。stdlib reader完成124项核查，8种损坏回执均被拒绝。只提交总结，不提交新统计或完整张量。

a1在门禁YAML依赖缺失处停止；a2在源包与控制器清单衔接处停止。两者均未启动NPU容器，原控制记录保留。a3修正准备问题后正常完成，不把前两次标为通过。

这是普通单点机制验收，不是全目标、FIA tiling、越界写或msprobe首差验收。观测元数据中的native_launch_verified=false仍保留，不借用33的观察器状态授予FIA原生资格。冻结RC2仍为六项，41只在后继开发线登记，后续发行独立安排；黄区RC2.1只改黄区登记，不混入本实现。
