# Provider开发与验收合同

本合同复用现有`NativeSession`、`QualifiedNativeSession`、`ModelFaultBackend`、
`TargetIdentity`、`CandidateSet`、`CoverageReport`和`RecoveryReport`，不新增Registry、
Controller或生产CLI。17张[独立能力卡](overnight_fault_cards/README.md)定义各机制专属动作。
宿主内源接线仍是`INNER_FRAMEWORK_INTEGRATION_PENDING`。

## 结构、调用和实际执行

模型结构发现只生成候选及别名。资格判断必须使用实际operator、dtype、shape、format、
mask/cache语义及精确build identity。Qwen、GLM、DeepSeek名称不能代替实际路径。
DeepSeek-R1-Distill-Qwen属于其真实Qwen架构，不能作为DeepSeek-V3/MLA/MoE验收。

Provider实现应维持以下职责，不得用返回值扰动替代不相干机制：

| 阶段 | 必须证明 |
|---|---|
| install/observe | 当前进程已有匹配preload；真实模型对象、scope owner和API；未调用NPU或加载任意库的inspect必须明确 |
| check | 实际调用符合本机制语义，已绑定worker启动身份、rank/device、model/module实例、call和storage生命周期 |
| execute | 原始算子恰好执行规定次数；唯一允许干预符合专属语义；未匹配调用明确透传/拒绝 |
| finish | 实际native完成、数值观察和临时资源释放；prepare/hit不是完成证明 |
| abort/disable | 停止新干预，等待已在途执行；保留未完成恢复的owner与对象引用 |
| clean/detach | 核对hook/registry所有权并恢复；无法恢复则隔离，不覆盖foreign owner、不全局reload权重 |

LD_PRELOAD由serving启动环境在Worker创建前设置；Extension只连接并核验已经加载的运行库。
运行时不能为补充注入临时重启Worker、补加载interposer或修改vendor安装。
Worker重建、PID复用、模型重载均令旧owner/generation失效，不自动re-arm。

## 专属修改边界

| 机制 | 允许动作与关键证据 |
|---|---|
| 8.restore_offset | paired索引/权重整组映射；真实token_combine独立分母、原始映射合法、token_unpermute实际消费 |
| 15.alias_detach | 私有格式转换形成独立storage再构造view；初值正确、copy_正确、原base更新丢失、同base消费者 |
| 17.spatial_padding | 实际Conv3d参数处改padding；输入/weight不变、shape数学边界及真实下游消费 |
| 18.api_half | 标准RMSNorm模块返回×0.5；明确API效果代理、hook观察顺序及全部符合条件调用 |
| 20.buffer_substitute | typed formal→真实Run slot→temporary input；WeightNz与QMM分别构建、观察和lossless，不互授slot证明 |
| 22.wrong_row | 原生IndexSelect返回后的合法source行copy；真实RoPE cache/positions/consumer；不是32位地址溢出 |
| 25.reduce_scatter | 实际group/成员/sequence/stream/分片；只能观察真实数值次序差异或重放真实同输入样本；无样本不随机加噪 |
| 26.residual_stomp | 实际active allocator邻接、QMM writer、当前forward残差consumer和完整恢复；不能用地址靠近推断邻接 |
| 26.fia_destination | FIA V3 output destination/copyback及当前forward残差consumer；正常FIA返回可完全一致 |
| quant.batch_shared_scale | 实际DynamicQuant归约范围；scheduler request→token row→scale/Q映射，两rank分别证明 |
| 28.fa.legal_tiling | FA真实host合法策略及实际launch；M-only/ratio-only/joint归因分开 |
| 28.fia.legal_tiling | FIA健康模板间M/S2切换；不选择分母重放key，不承诺数值一定超阈值 |
| 28.fia.denominator_replay | native online-softmax分母贡献重复；重复次数、模板、合法key覆盖及真实输出 |
| 33.paged_decode | actual page table/KV value逆映射、存活页和request归属、SInner策略及actual launch |
| 39.matmul_range | 有限输入临时替换→真实Matmul范围异常；不直接写Inf、不宣称已证明accumulator类型 |
| 39.output_proxy | 真实Matmul输出存储的显式非有限写回；独立合同，不混用range Native ABI |
| 41.tail_column | 实际dispatcher/cache writer之后的已证明尾列→同storage consumer；未命中dispatcher时只读阻塞 |

## 无损与恢复

`register`不得默认施加有损动作。先执行与目标相同的actual operator/profile的无损门禁；
无损回执必须绑定build、owner/generation、模型与请求/forward上下文，不能只比版本字符串。
相同文本或Top-1不证明全部中间值相同；应明确完整bytes/hash、全logits、token/text或摘要的验证范围。

`disable`不等于撤销已经传播到KV/cache/请求状态的损害。故障请求与fresh-request OFF应分开。
对于15，重建base再做OFF；26等真实原地修改还须先证明原storage精确恢复；
25不能只让一个rank跳过collective；41共享cache归属未知时不得猜测恢复地址。

## 独立取证

保留原生`statistics/mix`三件套、配置、collector身份、实际step/rank和未观测项。
比较固定`abs(F-B) > 1e-8 + 1e-5*abs(B)`，baseline参考；先从原始记录计算首差，
再关联native证据。`null`是不可观测，双方NaN仍不可比较；统计汇总相同不等于张量相同。

本轮新的原始采集留在124，只公开路径、hash和派生分析。其他agent只取总结不能重新算原始首差，
必须取得列明的原件。辅助完整tensor检查不作为独立统计比较输入。

离线[`evidence_contract.py`](../experiments/overnight_20261116/evidence_contract.py)检查报告内部
一致性（历史scope挪用、缺OFF、CPU冒充NPU、阈值变化、NaN遗漏、零分母等）。
它既不是签名认证，也不能令调用者提交的JSON自动获得生产执行资格。

## Fault25的可执行设计边界

[`collective25_design.py`](../experiments/overnight_20261116/collective25_design.py)只运行CPU设计模拟：
成员顺序决定scatter shard，global rank不等于local index，每个贡献恰好一次。
FP32的`[1e8,1,-1e8]`按两种合法加法次序得到0或1，解释数值机制，不构造真实HCCL证据。

真实实现的下一步是观察已存在Worker的communication group与实际backend/API，核验对应版本签名、
rank成员、collective序号、SUM/dtype/send/receive count及stream完成。同输入多次执行无差异时保留
`NUMERICAL_NONDETERMINISM_NOT_OBSERVED`。只有真实采得并绑定完整输入/group/layout身份的差异
可进入单独的replay设计；当前设计函数即使收到完整记录仍以`NotImplementedError`拒绝arm。
缺块、重复rank贡献、错误scatter区间属于另一机制，不允许作为25的通过结果。
