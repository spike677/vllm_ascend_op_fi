# 公共 Worker Core V1：本轮交付与验收边界

本轮已将实际调用发现、候选资格、请求绑定、Provider 生命周期和恢复回执接入同一公共 Worker 接口，并完成新的自然推理链路验证。**不是17/17验收通过**：严格的机制计数仍为 **C2 6/17、C3 6/17**。可调用工厂、自然调度入口、单点效果、完整覆盖和独立统计定位分别记录。

## 已交付的接口

- 用户使用 `FaultConfig(fault_id=..., variant=...)`，不填写层名、指针或内部算子位置；可信 Host 从实际返回的 Worker 身份选择路由，不能默认第一个返回值就是 rank0。
- `WorkerFaultClient` 经公共 RPC 完成 prepare/register/check/enable/advance/query/finish/disable/unregister。RPC 只设置或收集状态；推理由 vLLM 的正常 enqueue/scheduler 执行。
- CandidateSet 保留完整实际调用清单及不支持、未证实的候选。单目标 C1 的自动选点不会缩小 C2 分母。
- 当前注册表列出17条机制，13条有可调用工厂；其余三条28变体及33明确等待公共 RPC 适配。部分旧工厂仍依赖可信同步 replay，不能把13个工厂称为13条已验收的自然 Worker 路径。
- 原 Native ABI 和故障算法保留。新增实际进程/请求/对象/存储/代次绑定、源与库指纹、只读 query、明确 collect、失败保留 owner、真实 readback 后释放。

具体用法见 [Host 合同](../../docs/worker_core_v1/host_contract.md)、[公共框架说明](../../docs/worker_core_v1/README.md)，每条机制的实际状态见 [17条矩阵](C2_C3_MATRIX.md)。不是以模型名字白名单代替实际结构检查；跨模型仍需满足每个适配器的真实算子、布局、缓存及生命周期约束。

## 本轮新增实机结果

| 路径 | 实际模型与范围 | 结果与限制 |
|---|---|---|
|8.restore_offset，公共 Core source66|DeepSeek-V2-Lite-W8A8，TP1、固定 fresh prefill|26个目标 C2、52次 C3；原件独立复核、OFF和同 Worker 健康请求通过。没有新 E5 采集。|
|18.api_half，公共 Core source66|Qwen3-0.6B，TP1、BF16 fresh prefill|113个目标 C2、226次 C3；含57次原生 RMSNorm 和56次融合/custom RMSNorm。公共入口独立复核通过，不重复增加机制计数。|
|26.fia_destination，公共 Core source67|预训练 SmolLM2，TP1、固定 fresh prefill|每请求保留30次实际调用，29条可解析链；自动单点一次 F 改变14743个残差字节，首 token 从2/EOS变为30/`.`，OFF恢复。C1，非C2/C3/E5。|
|26.residual_stomp，公共 Core source68|DeepSeek-V2-Lite-W8A8，TP1、固定 fresh prefill|每请求保留135次实际调用；自动选中一处，一次 F 改变31个残差字节，writer输出保持正确，logits改变，OFF恢复；生成token始终185。Native构建源与Worker源分别复核。C1，非C2/C3/E5。|
|39.output_proxy，公共 Core source68|Qwen3-0.6B，TP1、14-token fresh prefill|113次实际 GEMM、28个合格消费者对；自动选中 gate_up_proj→SwiGLU，一次 F 改变logits及首token576→0，OFF/健康请求恢复576。独立复核真实 Native 租约和注册归属通过。C1，非39.matmul_range、C2/C3/E5。|
|quant.batch_shared_scale，公共 Core source64|DeepSeek-V2-Lite-W8A8，B8 first decode|完整135次调用/阶段、共945个projection parent；一次 F 改变71个Q字节和3个scale字节，logits改变，OFF恢复，8×2个生成token不变。只有2个位置/阶段证实请求行映射，其余133个保留未解析；C1。|

这批 Worker 结果限于真实测试的 **CANN8.5.1、Ascend910B4/A2、vLLM0.18镜像**。9.1的单算子与collective校准是另一范围；没有据此授予9.x Worker、A3、并发多Worker、任意模型或连续生成正确性。

详细的原件位置、SHA、消费者链和独立审查限制见 [增量报告](WORKER_INCREMENT_20261118.md)。完整 tensor 数组没有公开，也没有把运行时 hash 回执描述成离线重算张量。最后普通健康请求在卸掉观察器后只核对同 Worker 和 token/text。

## 修复及测试

- 真实 MoE 嵌套 dispatch mode 的异常退出现已按实际栈保留/清理；before-pop、after-pop、无效退出、外来栈及重试有 CPU 反例。source66正常结果不能追溯证明旧异常分支安全。
- Fault39 校验当前 GEMM PrivateUse1 注册、原始权重对象和只读 NPU 消费者；同 Library 或其他 Library 覆盖注册均拒绝。真实生产 Torch2.9 的无设备检查验证了两种覆盖痕迹，不能只依赖旧注册集合。
- Native open/apply/restore/release 与原有 live readback 均保存原始返回；没有增加 Native 调用或改变数值、恢复决策。缺少 release 回执不会伪造成成功。
- 量化模型加载身份支持实际 safetensors 索引名称；要求唯一索引准确覆盖全部规则分片，拒绝缺失、多余、越界和符号链接，不重命名权重、不反序列化模型以凑匹配。
- source68 的三个分发包无网构建/安装及SDK observer检查通过；同生产镜像36条 CPU 合同通过。可搬迁包的源码Core214条及installed-only Core214条覆盖通过，生产模块没有源码回退。不同批次测试有重叠，不累加成总测试数。

## E5与未完成项

新公共26/39入口的诊断张量读取尚未与完整 `statistics/mix` 窗口隔离，当前显式拒绝该入口的 `--statistics`。**本轮不以这些诊断回执宣称 msprobe 独立首差定位通过**。历史 B/F statistics 和固定比较器保留不改；null、NaN、无法对齐及先出现cache索引差异均按原样记录。

剩余工作按实际缺口推进：26/quant/39.output 的 C2/C3；15的自然私有格式视图写回链；17的标准多模态 Worker 请求/消费者链；28的实际 Worker TND 适配；33原版与私有正常版一致性；25的合法可重放差异样本。没有正常桥接或自然样本时不制造 F，也不以CPU替代设备验收。

## 内源交付

导出器输出 `contracts / worker_extension / providers / native_build / adapters / tests / integration_guide / compatibility_matrix` 八个索引，源码只保存一份。三个原分发根保持不变；没有vendor、权重、原始statistics、私有张量或预编译Native库。

黄区 Registry/Controller 的实际源码仍不可见，因此交付的是可实现的 HostAdapter 合同、源码和测试，黄区集成状态为 **PENDING**。主控的模型部署 prepare/probe 位于完整仓库，不能把精简runtime/CPU包误当即用推理服务。

最终导出清单和包校验见本目录的 `integration_bundle_delivery.json`；复现命令及源码审核清单随同提交。蓝区原始实验和二进制继续留在124，Git只提交审查后的实现、测试、摘要和哈希。
