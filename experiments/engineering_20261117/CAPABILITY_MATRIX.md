# 本轮台账与能力矩阵的读法

[`program_ledger.json`](program_ledger.json) 保留冻结计划的 **10 个公共节点 + 17 个机制节点**。
来源为 `deb5de1e3efd893620f7202676a15b88aa17d5cf` 中的验收矩阵，按其依赖关系排列；
原矩阵及历史证据不改写。每个节点都有任务 ID、工作流、依赖、文件归属、交付物、
测试入口、退出条件、阻塞处理和提交字段。`tests.required` 是要执行的检查，不能当成通过回执。

`commit.source_stage` 记录本次整理的源码阶段，`publication_commit=null` 表示尚未登记
本轮发布提交，不使用假 SHA。主控发布后可以据实补充提交关联。文件按主工作流独占；
同工作流内多个机制引用同一公共文件不意味着允许并发编辑。20 的中间 QMM 边界及
必要派发由主控明确委派 B，机制集成责任仍保留 C，单独记录这项委派。

## 六阶段与四种证据范围

| 阶段 | 达标内容 | 当前台账如何避免扩大结论 |
| --- | --- | --- |
| S1 ENGINEERED | 实现、Provider、配置、最小复现完整 | 有源码或 CPU 用例不等于完整工程验收；不声称 17/17 S1 或 M1 已完成 |
| S2 ADAPTED | 自动识别实际模型、角色、算子、环境 | 依据真实对象和实际调用；不根据模型名或 CANN 主版本推断 |
| S3 NATIVE_CLOSED | 原生执行、一次干预、效果成立 | 逐条 `scope` 记录；构造 NPU 实验不升级为自然模型或公共 Worker |
| S4 RECOVERY_CLOSED | OFF、drain、资源释放、恢复 | 局部 OFF 不能代替多 rank 整组清理；未知恢复状态保留 |
| S5 GENERALIZED | 逐模型、环境组合验收 | 单模型单算子通过不覆盖整个 Qwen/GLM/DeepSeek 家族 |
| S6 FULL_COVERAGE | all_eligible、every_match、整组恢复 | 本轮 C2/C3 仍为 0，配置参数不能授予资格 |

范围分为 CPU 合约、构造 NPU 推理模块、预训练模型、标准 vLLM Worker。
这些范围不可互相替代。S3/S4 可以先在有限范围取得证据，之后再完善 S1/S2；
台账不靠一个顺序等级掩盖这种区别，也不把观察器成功当成故障注入成功。

历史四条公共 Worker 机制的范围逐字保留在 `historical_snapshot`：20 的 WeightNz
输出投影、26 FIA 的有限 fresh-prefill、DeepSeek 首个 B8 decode 的一个共享量化投影、
39 MatMul 的一个输出投影。20 的自然 QMM 是另一条资格路径，不能借用 WeightNz 结论。
新回执只追加到 `tests.current_reviewed_scopes`，不会修改旧记录。

## 本轮需要特别区分的结果

| 机制 | 实现或当前观察 | 仍不能推出的结论 |
| --- | --- | --- |
| 8、15、17、18、22 | 构造 NPU 的实际调用、干预、输出变化和 OFF 已有本轮回执 | 自然预训练调用、生产调度和公共 Worker；最新 C5 统计为 `ALIGNMENT_MISMATCH`，没有独立首差 |
| 20 | 新 a3 已通过 DeepSeek 自然 dynamic-QMM 单点 Worker 的实际调用、干预、logits 变化和 OFF | QMM 与历史 WeightNz 分别记录；B 自比 NaN 导致 E5 整段仍未通过，不能扩大到其他投影 |
| 25 | 两 rank HCCL 校准执行；没有取得同输入的不同原生结果 | 没有可注入样本或故障命中；退出挂起也不能算整组恢复 |
| 26 residual | 新标准 Worker 的实际 writer、活 residual、消费者和恢复按独立总结追加 | 历史 runner 包装或新单点通过都不授予所有层/请求覆盖；NaN、不可比项不能删除 |
| 26 FIA、共享量化、39 MatMul | 保留历史限定的公共 Worker 正向证据 | 其他模型、版本、布局或请求阶段自动通过 |
| 28 的三个变体 | 新 host/kernel 构建及合法性校验分别处理 | 构建成功不继承历史 NPU 资格；归约错误扩展不冒充历史合法切分机制 |
| 33 | 自然 Worker 的 28 层 TND/page16 调用及真实缓存映射被观察到 | 历史 BNSD IFA 构建不兼容该自然路径；只读诊断不是 B/F/OFF |
| 39 output proxy、41 | 构造 NPU 下 public ACL 写入、读回、恢复与独立统计首差 | 41 不是自然 KV dispatcher；输出代理不冒充算子内部根因 |

执行/干预/效果/恢复与 msprobe 结论分开保存。独立首差报告只使用原始
`statistics/mix`、配置、调用栈、模块关系和采集器身份，固定
`abs(F-B) > 1e-8 + 1e-5 * abs(B)`。运行命中与完整数组证据不参与首差选择。
`FIRST_OBSERVABLE_STATISTICS_DIFFERENCE` 与 `whole_dump_comparable=false` 可以同时存在：
前者只说明可观测前缀存在差异，后者仍禁止声称整段严格比较通过。null、NaN、Inf、
不齐调用链、无差异和下游首差均保留。

## 泛化矩阵

台账列出 17×3=51 个模型家族目标格，以及 17×4×2=136 个环境目标格。
家族为 Qwen、GLM、DeepSeek；环境版本为 8.5.1、9.0.0、9.1.0、9.2.0，分别交叉 A2/A3。
**这些格的含义是完整泛化能力验收，不是“是否存在过一个成功样例”。** 因而当前均保留
`UNVERIFIED`，同时可在同机制的限定回执中看到已通过的模型/算子实例。

没有 A3 实机、现役黄区宿主源码及完整平台测试，就不能开放相应格。
也不能将未安装 vLLM 的某个 9.1 镜像推断为“9.x 不支持”。具体适用性由实际接口、
ABI、源码、工件、加载身份和运行证据决定；没有证据时保留未验证，不滥用不适用。

## 重生成、复核与追加回执

```text
python experiments/engineering_20261117/generate_ledger.py --self-test
python experiments/engineering_20261117/generate_ledger.py --check
```

生成器固定核验冻结矩阵 SHA，不依赖当前可写矩阵的偶然内容。没有 Git 对象的源码包可
用 `--matrix <冻结矩阵文件>`，但仍必须逐字匹配冻结 SHA。输出固定排序、UTF-8、无运行
时间戳；相同输入可得到相同字节。自测包含依赖环、缺节点、历史范围篡改、矩阵重复、
模型名推断支持、未审查 C2/C3、构造实验伪装 Worker 等拒绝用例。

生成器默认读取本目录内已经审查的 `tensor_collective_summary.json`、
`postwrite_constructed_summary.json`、`residual26_worker_summary.json` 和
`fault20_qmm_worker_summary.json` 和 `attention28_fresh_summary.json`；不存在的文件不会生成通过结论。
额外总结可用 `--review-summary <仓库相对路径>`，必须来自已审查的运行记录。
这里是证据索引工具，不会为运行时签发资格，也不替代原始回执复核。

除已有 postwrite 专用适配外，新总结通过 `ledger_scopes` 显式提供：

```json
{
  "ledger_scopes": [{
    "mechanism": "26.residual_stomp",
    "scope_id": "唯一的真实运行和限定作用域",
    "scope_kind": "NATURAL_PRETRAINED_WORKER",
    "scope_description": "实际模型、API、版本、硬件、布局和请求阶段",
    "entrypoint": "standard_vllm_worker",
    "public_worker_qualified": false,
    "pretrained_model": true,
    "E1_execution_observed": false,
    "E2_intervention_observed": false,
    "E3_effect_observed": false,
    "E4_off_and_local_recovery": false,
    "runtime_evidence": {"path": "实际运行回执路径", "sha256": "实际文件SHA256"},
    "independent_statistics": {"status": "UNVERIFIED", "whole_dump_comparable": false},
    "all_eligible": false,
    "every_match": false
  }]
}
```

示例中的布尔值不是验收预设，必须逐项按真实回执填写。公共 Worker 资格需要真实
预训练 Worker 范围且 E1–E4 均已核对；独立统计单独记录。后续若要开放 C2/C3 或
整格泛化，必须修改审查规则并补齐对应矩阵，不能只改这个 JSON。

本轮只发布实现与总结。新的统计原件、完整私有张量和二进制留在124。
