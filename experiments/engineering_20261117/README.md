# 17 条故障工程整合：2026-11-17

本轮从冻结提交 `cae9cdf4685543df98598bbc2cf0f3b9892b7635` 创建
`codex/17-fault-engineering-20261117`，保留原工作区、历史实现和四条公共 Worker
限定验收。正在完成新增机制的源码、部署及实机对账；不宣称 17/17 或 C3 已通过。

## 入口和实际边界

- [公共 Native Build Manager](../../fault_injection/native_build/README.md)：
  发现实际头文件、符号、ELF、工具链、输入和加载指纹；七个算子专属 recipe。
  39 MatMul、20 QMM、26 FIA/新 QMM 状态接口和 ACL postwrite 已在124完成新构建。
  编译成功不授予 NPU 资格。
- [Tensor / MoE / RoPE / Collective Provider](PROVIDERS_C.md)：
  8、15、17、18、22 的构造 NPU 推理模块完成真实效果和 OFF；25 只接受真实
  同输入通信的差异样本，无样本时拒绝。Python hook 状态与 Native 状态分开。
- [20 的自然 QMM 中间投影](FAULT20_QMM.md)：
  从真实 embedding→norm→quant method 的调用选择投影，核验 INT8/scale/native slot，
  不要求用户填写层名，不把 floating 输入冒充量化后的 x1。
- 26 residual 新增专属 Worker Provider 和 `dcat26q_state_v1`；同一 forward 的
  QMM、活 residual、邻接存储、consumer 和恢复仍由各自门禁确认。
- 39 output proxy / 41 tail column 使用独立 public ACL 写入、设备读回和字节恢复，
  与 MatMul range / 历史 dispatcher 分开；当前 Cache IndexCopy 路径属于构造机制范围。
- [33 的真实 Worker 调用核查](FAULT33_WORKER_AUDIT.md)：28层实际 TND FIA 的
  writer/live KV/page/slot/consumer 已核对。历史 BNSD IFA 构建不自动适配这条路径。
- 28 FA 历史源码恢复到 `native_execution_strategy/native`；FIA legal/replay 各自
  从固定 c77c0829 源码打补丁，增加完整 host/kernel 新构建器。新工件须重新验收。

公共身份增加 Python hook、Tensor alias、Operator API、Kernel launch、Collective
五种实际边界，继续共用 CandidateSet/Coverage/Recovery。它们是观察身份，不是资格证书。
模型角色增加 MoE router/experts；scheduler 的请求数、token 行数和缓存容量分别记录。
Extension 不创建 Worker，不替换 `execute_model` 或 `_prepare_inputs`。

## 验收和统计边界

执行、干预、模型效果、恢复、独立统计分别记录。构造模型不会升级为预训练模型；
加载库、实际调用和 Native 故障命中也分开记账。C1 不升级为 C2/C3。

`statistics/mix` 比较固定 `rtol=1e-5, atol=1e-8`，baseline为参考；独立比较只读
原始 dump/stack/construct、配置和采集器身份，不读注入日志、策略或完整数组。
`null`、NaN、Inf、结构差异与不可比项保留。当前新增 Tensor 机制的严格调用链
对齐仍有拒绝，不能用机制日志替它指定全局首差。

第一次 postwrite 采集把运行后的 `Torch.equal` 验证混入了模型统计，导致 B/F 结构
不同。原采集保留；新 `model_capture.py` 只在真实模型前向内开始/停止采集，
运行后的比较、存储恢复和租约释放在采集窗口外。这个修复不改原始统计字节或阈值。

所有新原始统计、完整数组、构建二进制保留在124：
`/home/17-fault-engineering-20261117`。GitHub只发布实现、总结、路径及SHA。
每个 `source-a*`、`*-a*` 都是新目录；每批最多8 case，共享锁与逐次空闲门禁有效。
没有恢复过期的 Docker 入口隔断，也未停止外部任务。

## 可复核入口

| 用途 | 脚本 |
| --- | --- |
| CPU 多套回归和跳过项 | `run_cpu_validation.py` |
| 精确源码快照 / 独立 wheel | `package_source.py` / `build_packages_124.py` |
| 无设备 Native 构建 | `build_manager_124.py` |
| 有界NPU控制器 | `run_cases_124.py` |
| Tensor机制五阶段 | `statistics_probe.py` / `tensor_provider_probe.py` |
| 真正双rank collective | `collective25_probe.py` |
| 26标准Worker | `writer26_worker_probe.py` |
| postwrite B/F及真实HF prefill | `postwrite_probe.py` |
| 严格独立统计 / 拒绝诊断 | `analyse_batch_124.py` / `inspect_statistics_alignment.py` |

当前未验证组合保持 UNVERIFIED：A3环境未取得，黄区现役内源宿主不可访问；
不能将124的910B4/A2、CANN8.5.1构造或特定模型结果推广到它们。Native 8.x/9.x
能力按实际 recipe/符号/构建和执行证据分别开放，20/26私有ABI的范围仍受限。

本轮完成记录、失败门禁及后续任务将与阶段提交一起更新；基线27节点与历史证据不改写。
