# Fault39 source101：C2 会话预算停止，未进入 C3

本次只读独立复核的结论是：**实测在第25次独立 C2 故障请求开始时触发会话预算停止。已完成24/28个故障目标并完成恢复，但未到累计 C3，未复现 `POSTWRITE_APPLY_4`，也未闭合五条原快照反事实链。** 不将预先允许的退出码2称为诊断成功，不继承source84或其他scope的C2资格。

实际模型为Qwen3-0.6B，标准vLLM TP1 Worker，BF16 eager、相同输入的新prefill，物理卡5。probe与Native新构建均绑定source101：`417d43ccc85b58554d243791546f9d484fdcd8f1`。本次没有采集statistics/mix，因此E5仍未通过。

## 第一个停止原因

不是由顶层 `SESSION_STOPPED` 字符串猜测。审核器从完整2715条ledger重新生成call/ticket顺序、实际目标身份和admission原因：

| 原始位置 | 实际调用 | 原始decision | 说明 |
|---|---|---|---|
| sequence2712，`forward:26:1` | 第0层 `self_attn.qkv_proj`，`vllm.unquantized_gemm.default` | `SESSION_BUDGET_EXHAUSTED` | 首次预算停止，非故障目标，没有Native应用 |
| sequence2713，`forward:26:2` | 第0层 `self_attn.o_proj` | `SESSION_STOPPED` | 已处于STOPPING，非故障目标，没有Native应用 |
| sequence2714，`forward:26:3` | 第0层 `mlp.gate_up_proj` | `SESSION_STOPPED` | 首个随后遇到的eligible调用；抛出保留的有界会话停止错误 |

首个停止发生于第2713个非OFF调用，`max_calls=3164` 尚未耗尽。原件中 `max_seconds=1200`；已重哈的 `CoverageSession.begin` 在这个分支只检查“调用数超限或会话elapsed达到上限”。因此，**会话1200秒时间预算耗尽是绑定源码与完整计数的分支排除推断**。原件没有保存精确的session elapsed采样，不能写超了多少秒。

这与probe软截止1500秒、queue硬截止1800秒是三个不同边界。本轮没有修改或延长任何预算。容器原状态为exit2、非OOM，controller保持`CASE_FAILED`；不能归因为Native非有限值guard，也没有证据归因为外部kill。

## 已发生的部分与未完成项

| 项目 | 原件独立复核结果 |
|---|---|
| 实际分母 | 每个完整forward 113个projection调用，28个eligible目标 |
| 原6前置 | 完成；独立检查真实请求、Native、恢复和调用链 |
| 全目标lossless | 28/28完成；逐目标完整字节摘要一致、真实租约释放 |
| 独立C2 F | 24个目标完成，每目标一个真实成功apply、值变更和实际consumer；对应第0–23层 `gate_up_proj` |
| 本次未测目标 | 第24–27层 `gate_up_proj`，不能从原计划或旧scope补入 |
| 请求 | 32次正常协议请求中31次完成、1次被预算停止；另2次恢复请求完成 |
| Native | 54个唯一租约，所有apply返回0；重复原件不重复计数；没有apply4 |
| 原件完整性 | 162个metadata原件、1,295,691,209字节逐文件重哈；150个被引用原件再次核对 |
| 调用链 | 28个完整inventory、15,848个实际forward frame核对；失败forward的3个实际调用保留为partial |
| live C2 proof | 本轮未签发、未消费 |
| C3 | 未开始，累计臂Native原快照缺失 |

全部28个逻辑目标的恢复记录、Native restore/release/live、原公共owner detach、实验trace关闭、模型OFF/health、注销后普通health与binding removal相互对应。完整hidden+logits bundle与完整logits分别在各自摘要域比较，不混用。容器清理记录另行保留，没有拿它替代同进程模型恢复。

## 三条诊断实际说明什么

记录的完整逻辑span均为 `[14,6144]` BF16：86,016元素、172,032字节。

| 真实观测 | 原快照来源 | 结果 |
|---|---|---|
| 健康请求的预注册第1层位置 | 原目标的fresh `read_span` | 86,016个有限值 |
| 单前序F的第0层原值 | 同一个实际Native `lease.original` | 86,016个有限值；实际apply仅一个元素变更、返回0 |
| 同一单前序F请求的第1层位置 | 原目标的fresh `read_span` | 86,016个quiet NaN；首个索引0、位模式`0x7fff` |

单前序请求的全部Native原件只包含一次故障写入，producer、实际原存储consumer、请求和完整span摘要均关联。健康fresh摘要可重建并匹配lossless原张量摘要；前序Native原快照也与实际execution原张量摘要匹配。这支持**该输入下单个前序故障会传播出非有限值**。

但是，第1层的单前序fresh读取不能冒充累计C3中同一lease的原快照。累计两条记录没有发生，无法完成其完整SHA对应，也不能确认本轮guard拒绝。历史source84的失败仍保持其原结论和限制。

## 身份、实现与复算

机器摘要见 [output39_source101_partial_summary.json](output39_source101_partial_summary.json)。主要原件身份：

- result：`70ef7235b1661fcc9d5a1467bcc892747e8b78e82a53642aa5071870cc8c7ed8`
- 三条diagnostic：`b17566a1da05cde78e41aebcf71001ba9173ff7f795982d89da87332afbd77b8`
- source manifest：`6c6b544f1709f6235961858d175bcffd6494ba47b7bbc556cf50cca49ebd715c`
- Native DSO：`9261d0867f822c90d0d179457e41bc9a2b132bc39e643b61fe5d47a3fd81424f`
- build key：`05e6a510efa3a8ac8f4a7c3f77056af2027137a3d90440cc36ec31f90af1ccbf`

source101冻结1072份源码全部重哈；38份入口/运行依赖及13份编译依赖与原manifest、loaded source和build inputs对应。原六个Native函数体SHA未变。build inputs规范JSON的SHA重新得到同一build key；新构建和实测记录相同镜像ID、CANN8.5.1及GCC11.4身份。

本地实际读取的是源码和metadata。DSO、SDK头文件/库、编译器及模型权重的实际字节没有下载，相关SHA和ELF身份仍是原构建/运行回执，不能声称在本地独立重哈了它们。

使用 [review39_refusal_prefix.py](review39_refusal_prefix.py) 复算本次**不完整前缀**；原 [review39_refusal_originals.py](review39_refusal_originals.py) 的五条完整门槛未降低。按完整原件目录运行：

```sh
python /review-source/experiments/worker_core_v11_20261118/review39_refusal_prefix.py \
  --evidence-root /original-attempt/output39-fixed-c3-refusal-diagnostic \
  --metadata-bundle-root /original-metadata-bundle \
  --result-sha256 70ef7235b1661fcc9d5a1467bcc892747e8b78e82a53642aa5071870cc8c7ed8 \
  --source-root /original-source101 \
  --source-manifest-sha256 6c6b544f1709f6235961858d175bcffd6494ba47b7bbc556cf50cca49ebd715c \
  --source-commit 417d43ccc85b58554d243791546f9d484fdcd8f1 \
  --native-sha256 9261d0867f822c90d0d179457e41bc9a2b132bc39e643b61fe5d47a3fd81424f
```

原件一致性的`consistent=true`只表示**已发生的不完整执行和恢复可复核**，不表示故障诊断、C2、C3或E5通过。本提交只包含审核器、CPU合同和总结；不包含原RPC/Native数据、Pod/OCI内容、私有张量或权重。

最终原件审核为96,961项检查、源码83项检查、metadata/build/controller 182项检查，均无未解决issue。这些是检查断言数量，不是独立NPU用例数量。新增真实CPU public-driver预算停止合同及负例，连同完整反事实reader和旧group回归共 **53 tests passed / 0 skipped**；不与先前重叠套件累加。
