# 17 条故障工程整合：2026-11-17

本轮从冻结提交 `cae9cdf4685543df98598bbc2cf0f3b9892b7635` 创建
`codex/17-fault-engineering-20261117`，保留原工作区、历史实现和四条公共 Worker
限定验收。本轮已完成第一阶段代码发布及新增实机对账；不宣称 17/17 或 C3 已通过。

## 已核实的本轮结果

| 验收范围 | 结果 | 限制 |
| --- | --- | --- |
| 标准 Worker 自然推理 | 新增 26 residual 的 C1；20 增加独立 QMM 路径的 B/F/OFF | 公共 Worker 有限定正向范围的机制数 4→5，20 两条路径仍只算一个机制；不是全部层/请求 |
| 构造 NPU 推理 | 8、15、17、18、22、39 output、41 七个机制的干预和 OFF 通过 | 不升级为预训练模型或公共 Worker |
| 新 FIA 小模块 | legal / denominator replay 两个新包各通过 stock/OFF 桥接、B/F重复稳定及 OFF 精确恢复 | 完整输出有差异，原生统计没有可观测首差；未授予独立 launch、模型或公共 Worker 资格 |
| 独立 statistics/mix | 39 output、41 的完整结构对齐及首差通过 | 其余新增实验保留结构拒绝、NaN 和更早 slot 变化，不用命中日志替代首差 |
| Native Build Manager | 五个新构建回执：39 MatMul、20 QMM、26 QMM/FIA、ACL postwrite | 构建回执自身不授予 NPU 资格 |
| CPU 回归 | Windows 777 项：775 通过、2 项跳过；Linux 7 项门禁测试通过 | Linux 补测了 Windows 无权限的真实软链；Windows Gloo 项仍未执行 |
| 全量覆盖 | C2=0/17、C3=0/17 | 未开放 all_eligible / every_match |

首阶段提交为 `deb5de1e3efd893620f7202676a15b88aa17d5cf`，本轮实现和实机总结提交为
`cb5b0dc120ce6c80f1469bb8b7248da27a7bb258`。实际验证的源码快照、安装包、构建和比较器均另附 SHA，不能用
当前工作树替换已经执行的版本。详见 [27 节点与能力矩阵](CAPABILITY_MATRIX.md)、
[CPU 回执](cpu_validation_summary.json)、[Linux 门禁回执](fia_cpu_contracts_summary.json)、[20 QMM](FAULT20_QMM.md)、
[26 residual](FAULT26_RESIDUAL_REPORT.md)、[39 output / 41](POSTWRITE_REPORT.md)。
新 FIA 的失败登记、官方索引修复、B/F/OFF 及统计负结果见
[新工件报告](FAULT28_FRESH_BUILD.md)。
[发布校验清单](publication_manifest.json) 保留本分支相对冻结基线的源码/总结逐文件 SHA，
并核对 Git 暂存字节，避免换行转换改变已审查身份。

[Tensor/MoE/通信报告](TENSOR_COLLECTIVE_REPORT.md) 记录五项构造推理及真实双 rank 校准的
通过范围与失败。 [构建与证据管理摘要](infrastructure_summary.json) 列出五个 Native recipe
实际构建的输入/工件指纹、两个 FIA 新源码构建及一次真实保存运行的 EvidenceManager
对账；后者只整理原有运行，不冒充新增验收。

20 的统计首候选为 `NPU.npu_quant_matmul.0.forward/output/0` Max `25.75→15.6875`，
但 B 自比遇 NaN，整段不可比。26 的首候选是更早的物理 KV slot `61→77`，不能归因于
residual。39 output 的独立首差为 `Torch.mm.0.forward/output/0` Max `8→+Inf`；
41 为 `Tensor.index_copy_.0.forward/output/0` Max `4→2`。每项完整限制见各自报告。

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

## 仍未关闭的门禁

- 25 实际双 rank 同输入校准没有发现不同输出，因此没有故障重放样本和 F；进程退出
  曾挂起，不能将局部 OFF 相等记成整组 drain 通过。
- 33 自然 Worker 的 TND FIA 与历史 BNSD IFA 不同，需要自己的 Native adapter。
- 39 output 在实际 HF Qwen 前向只观察到 `linear/matmul`，没有当前接受的 `mm`，
  无损门禁拒绝且未注入。需要独立证明这些 producer/consumer，不能更换名称绕过检查。
- 28 新 host/kernel、已有实验资格和公共 Worker 接线分别验收；新构建不继承旧证书。
- Tensor/MoE/RoPE 的自然预训练链、所有机制的 C2/C3、A3及黄区现役 Host Adapter
  尚未完成。环境缺失和尚待开发的适配不能合并为 PASS。

按计划的有界探索规则，保留每轮拒绝及准确下一步；没有降低门禁、改变机制或增加
输出扰动来补数字。基线27节点与历史证据不改写。
