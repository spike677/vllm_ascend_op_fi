# 目标选择合同 V1：手选、全部可绑定与参与 rank

实现入口：[selection.py](../../src/ascend_fi/inspect/selection.py)、[plan_builder.py](../../src/ascend_fi/inspect/plan_builder.py)、[target_resolver.py](../../src/ascend_fi/inspect/target_resolver.py)。操作示例见[手选／显式全选手册](../usage/fake_requant_manual_all_targets_v1_zh.md)。

本扩展是选择与可解释预览，不新增注入器，不改变 FakeRequant 数学。复用 `FAKE_REQUANT_TARGET_PLAN_V1`，新增选择描述版本 `EXPLICIT_TARGET_SELECTION_V1`；旧无 selection 的计划仍由原合同分支验证。

## 1. 输入与身份

输入是完整的 `MODEL_TARGET_REPORT_V1` worker 报告数组，不是任意模块字符串列表。报告 checksum 必须通过，枚举必须完整且未截断。计划构建要求 LIVE 来源、明确参考源码 hash、当前受支持模型适配状态和 eager 证据；离线读取保存的 LIVE 报告不能证明其仍属当前运行实例。

目标 identity 为本次 worker 身份加 canonical module name。服务 epoch、runtime fingerprint、model fingerprint 与输入观察来源一起约束绑定。对比其他实例时允许地址不同；执行时必须重新取得 descriptor 回证，不把“同一个文件”当作“同一个实例”。

| 坐标 | 用途 | 禁止替代 |
| --- | --- | --- |
| global rank | 显式 rank 策略及参与组 | 物理 device ID、进程号 |
| TP rank | 核对当前纯 TP 映射 | 任意 DP/PP 组的 global rank |
| device/local rank | 部署与观测上下文 | 自动解释为用户指定 global rank |
| worker epoch | 当前实例身份 | 旧 IDENTITY PASS 文件 |
| module name | worker 内规范目标路径 | 模糊正则或同名跨 worker 合并 |

## 2. 手选与全选语义

`resolve_selection(mode='manual', roles, layers, module_names, rank_filter)` 是普通入口。手选至少提供 roles 或精确 modules；仅 layers 不足以声明要修改何种模块。

```text
候选行 = role 任一匹配
       AND layer 在显式集合
       AND module name 在显式精确集合
```

未提供的过滤维度不限制；用户提供的列表不能空、重复或类型错误。指定 module/layer/role 在作用范围缺失时明确拒绝。手选命中 blocked 条目即失败，不自动忽略它并宣称完成。

`mode='all_bindable'` 显式将范围内的全部合格目标加入一个计划。blocked 条目写入 `excluded`；结构缺口另列，不冒充“模型所有数学位置已覆盖”。每 worker 至少有一个可绑定目标，否则拒绝。

`all_bindable` 与 `ranks=all` 独立：前者扩大目标集合，后者明确选择参与 rank。省略 CLI rank 仍为 `[0]`，全目标不会隐含全 rank。没有基于 L2、严重程度或敏感度的自动排名。

## 3. 当前参与 rank 的支持边界

`participating_ranks()` 从 descriptor 核对：

- worker 报告数与 TP size 一致；global rank 和 epoch 唯一、非缺失。
- PP=1、DP=1、EP=false；当前只支持单个完整纯 TP 组。
- global ranks 必须恰为 `0..worker_count-1`；显式给出 TP rank 时必须等于 global rank。
- `ranks=all` 解析为该组，不指向其他 DP 副本或整机所有 NPU。
- 任一必要 worker 缺失、重复或无法证明归属，整次选择拒绝，不静默缩小。

当前 owner 安装同一组精确模块名，所有参与 worker 都须有完整一致的可绑定集合，包括未被 rank_filter 选中有损执行的 worker。异构目标集合、不同 selector、global!=TP 和非连续 global ranks 均明确不支持；不声称已实现任意分布式拓扑。

每 worker 同名 canonical module 是不同执行对象，不跨 rank 去重。单 worker 逐字相同报告行可去重并记录 `deduplicated`；同名但内容不同报告拒绝。多物理别名目前由 resolver 的 `SHARED_MODULE_ALIAS_UNSUPPORTED` 阻塞，不能借去重猜测共享对象关系。计划还拒绝父子模块同时入选。

## 4. 可绑定不是“看见 float”

沿用原 resolver 门禁：真实已停止观察、同 epoch/runtime/session、受支持模块实现、明确 activation 参数语义、唯一浮点输入、可解析的参数路径、无输入对象别名；dtype 为 FP16/BF16/FP32，正维度二维连续 `torch.strided`，不是 meta tensor，特征维与当前 partition 证明一致。

计划只接受明确 selector，不回退 `first_float`。观察 shape/stride 来自真实调用；绑定中的 token 维可为动态正数，但 feature size/dtype/layout 等合同不因“全选”放宽。不同 worker/目标 selector 异构时当前单参考 owner 拒绝。

## 5. 计划字段和预算口径

| 字段 | 含义 |
| --- | --- |
| `selection.intent` | 原始 mode 范围、roles/layers/modules、显式 ranks 或 all |
| `selection.participating_global_ranks` | 已证明参与该纯 TP 请求的 worker ranks |
| `selection.targeted_global_ranks` | 实际允许有损变换的 ranks |
| `selection.per_worker_modules` | 每个参与 worker 要求安装/核对的精确集合 |
| `selection.excluded` / `deduplicated` / `structural_gaps` | blocked、重复报告处理和结构缺口说明 |
| `workers[].targeted` | 该 worker 是否执行有损变换；不是是否存在 |
| `prospective_arm` | 待执行配置，非当前已 arm 状态 |
| `request_window.per_worker_max_hits` | 注入 worker 的有限上限；未注入 worker 的有效注入预算为 0 |
| `theoretical_sum_of_targeted_worker_caps` | 所有注入 worker 上限之和，不是共享计数器 |
| `plan_sha256` | 包含选择字段的完整关闭计划校验值；编辑后不可沿用旧许可 |

上限计算保持原保守公式：

```text
每 worker cap = M × (max_tokens + max_prompt_tokens + 16) × 2
预算范围 = PER_WORKER_ARM_LIFETIME
M = 该 worker 的目标模块数（当前同质 TP 集合）
cap <= 2^21，目标模块数 <= 512
```

实际 hit/candidate/reject 仍由每 worker 分别提供，再合计。上限不是目标剂量，不填充短请求，不触顶清零，不把一个 rank 的额度转给另一个 rank。每个正式请求由既有流程单独 arm，完成后 finally disarm/remove。

模式仍是 `REFERENCE_REQUANT`、provider 为 `tensor_absmax`、interval=1、有限倍率 `0 < m <= 20`。倍率 1 不是 IDENTITY。扩大多个模块和 rank 是组合条件，不支持独立因果归因，也不保证病态单调加重。

## 6. 预览与执行门禁

CLI [inspect_model_targets.py](../../tools/inspect_model_targets.py) 的 `target_preview.md/json` 只读解释，不 arm。每行保留 rank/epoch、layer/role/module、输入 selector/语义/dtype/layout/feature、真实观察、角色证明、指纹和 blocker。

状态不是同一个布尔值：

```text
discovered → observed → bound → selected
                             ↓
                   SAME-EPOCH IDENTITY
                             ↓
                     实际 hit / effect
```

预览固定标记 `identity=NOT_RUN_BY_PREVIEW`、`hit=NOT_MEASURED_BY_PREVIEW`。生成的计划为 DRAFT、enabled=false，执行仍通过现有 owner/TargetPlanExecutor 完成 rebind、独占检查、精确安装集合、IDENTITY 和各 worker arm 确认。

候选与 hit 相同证明目标调用窗口，但没有独立 scheduler 执行步来源就不能认证完整请求窗口。模拟 worker RPC 通过只认证选择/编排逻辑，不能写成 Ascend 多 rank 现场 PASS。

## 7. 结束策略与报告状态

“同进程 B/F/R”和“单次有损后停止实例”是不同实验策略，不能改成两个故障实现。后者保留前置 IDENTITY、正常 disarm/remove 尝试，然后停止本 session 自有 worker；正式有损后的文本恢复未测，不虚构 recovery 输出。

如果后续启动新 worker 并验证正常，称为重启后对照；只有同 worker 实际移除后再次推理并验证相等，才能记录同进程恢复。任何新实例都须重新绑定与 IDENTITY。[hook 与清理证据边界](../architecture/hook_mechanism_compatibility_v1_zh.md)

Dashboard 是既有 `EvaluationResult` 的只读展示，产物索引是证据指针，不替代 raw 数据或升级未验证状态。本轮没有现场全 rank 测试；本地无 PyTorch 的情况下也不把替身测试冒充真实 torch 模块执行。历史 Qwen rank0 实测、当前离线选择测试、未来现场验收必须分栏记录。
