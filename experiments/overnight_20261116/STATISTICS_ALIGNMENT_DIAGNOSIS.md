# Fault15/17/18 原始 statistics/mix 对齐诊断

三条机制的本次原始采集都保留为 **严格独立首差未通过**：每个实际 `step0–2/rank0` 返回 `ALIGNMENT_MISMATCH/structure`，组级为 `INCOMPLETE_COMPARISON`，全局首差为 `null`。下述局部统计诊断解释能观测到的内容，不覆盖严格结果，也不代替机制级 B/F/OFF 的另外验收。

只读取原始 `dump.json`、`stack.json`、`construct.json`、采集配置、采集器身份；没有读取命中日志、运行时完整张量、生成响应或验证器结果。所有数值使用固定公式 `abs(F-B) > 1e-8 + 1e-5*abs(B)`，相对变化用 `abs(F-B)/abs(B)`，基线为零时不计算相对变化。`null` 保持不可观测。

| 机制 | 原始结构不齐 | 局部可见证据 | 本次不能声称的内容 |
| --- | --- | --- | --- |
| 15 | F 多出 `NPU.npu_format_cast.1.forward`，格式参数为 `2`；两个 `Torch.equal` 的非张量 bool 输出也改变 | 写后检查 API 的输入统计和 bool 结果改变 | 不能从本次统计唯一定位别名断裂；私有格式消费者输入/输出四项均为 null |
| 17 | `Functional.conv3d.0.forward/input_kwargs/padding/1/value` 和 `/2/value` 均 `0→7` | 同栈、同结构的 Conv3d 模块输出及后续 add 发生统计变化 | 不能把改变 padding 的 API 当作同参数对照，也不能称为完整序列独立首差 |
| 18 | F 增加 `Tensor.__mul__.0.forward` 和 `.1.forward`，各自乘数 `0.5` | RMSNorm 模块记录仍相同，后续 add 的输入/输出幅值变为一半 | 不能声称 msprobe 首差已落在 RMSNorm 输出，或是 RMSNorm 原生计算先产生差异 |

这三组中，各组自身的三个 `dump.json` 字节哈希一致，以下数值在三个 step 都出现。step 是真实采集步数，不解释为三个 decode token。

## Fault18：返回处理发生在模块统计之后

两条 `Module.a/b.RMSNorm.forward.0/output/0` 的 B/F 可观测统计均为 `Max=1.7051955461502075`、`Min=-1.7051955461502075`。输入和权重的可观测统计也没有变化。Mean/Norm 均为 null，因此不能据此声称完整输出相同。

F 新增两个乘法记录，原始栈首帧是 `module_faults.py` 的返回 hook：`return output * 0.5 if self.arm == 'fault' else None`。这是原始采集栈中的信息，没有另外读取注入器日志。乘法在 B 中没有配对记录，故严格比较失败。

共有的 `Tensor.__add__.0.forward` 的调用栈、非统计描述和 construct 项可以逐项对应：

| 字段 | B | F | 绝对差 | 相对变化 |
| --- | ---: | ---: | ---: | ---: |
| `input_args/0/Max`、`input_args/1/Max` | 1.7051955461502075 | 0.8525977730751038 | 0.8525977730751038 | 50% |
| `input_args/0/Min`、`input_args/1/Min` | -1.7051955461502075 | -0.8525977730751038 | 0.8525977730751038 | 50% |
| `output/0/Max` | 3.410391092300415 | 1.7051955461502075 | 1.7051955461502075 | 50% |
| `output/0/Min` | -3.410391092300415 | -1.7051955461502075 | 1.7051955461502075 | 50% |

六项均超过固定容差，但只能称为**可配对消费者记录中的统计变化**。完整序列存在未配对乘法，因此不输出全局独立首差。它与“模块输出被返回处理改写，再被消费者使用”的现象相符，无法证明 RMSNorm 内核本身计算错误。

## Fault17：API 参数变化与模块输出变化分别记录

`Functional.conv3d.0.forward` 的 input/weight Max/Min 相同，padding 从 `[0,0,0]` 变成 `[0,7,7]`，因此该 API 的完整非统计记录不能对齐。工具保留参数差异及同路径数值的算术差，只把后者标为 `record_pairing_accepted=false`，不纳入通过的局部配对集合。

`Module.patch.Conv3d.forward.0` 的原始栈、construct 项、输入/参数的非统计描述可以对应，其可观测输入/权重统计未变：

| 字段 | B | F | 绝对差 | 相对变化 |
| --- | ---: | ---: | ---: | ---: |
| `Module.patch.Conv3d.forward.0/output/0/Max` | 0.5635518431663513 | 0.24107059836387634 | 0.322481244802475 | 57.22299531318963% |
| `Module.patch.Conv3d.forward.0/output/0/Min` | -0.4876461327075958 | -0.39287781715393066 | 0.09476831555366516 | 19.43382899962225% |
| `Tensor.__add__.0.forward/output/0/Max` | 1.563551902770996 | 1.2410706281661987 | 0.32248127460479736 | 20.624916514333916% |
| `Tensor.__add__.0.forward/output/0/Min` | 0.5123538970947266 | 0.6071221828460693 | 0.09476828575134277 | 18.496645831859757% |

这些字段及 add 的两个对应输入字段均超过固定容差。证据支持“Conv3d 的调用参数变化，与模块输出和消费者的统计变化同时出现”，不能据此宣称同输入同参数的原生 Conv3d 精度缺陷。

## Fault15：私有格式输出不可观测，检查 API 有现象

本次原始 dump 没有独立 `as_strided`/`copy_` 数值记录。私有格式转换输出、`Tensor.__add__.0.forward` 输入和输出的 Max/Min/Mean/Norm 均为 null；后者原始栈显示 `result = base + 1`。所以不能用本次 statistics 给该消费者选出数值首差，也不能把 null 当作 B/F 相等。

原始栈表明 `Torch.equal.3.forward` 检查 `base_matches_expected_update`；`.4` 检查 `base_target_retains_prewrite_values`。这两条检查 API 出现在原始采集内，不是另读成功日志：

| 原始字段 | B | F | 解释边界 |
| --- | ---: | ---: | --- |
| `Torch.equal.3.forward/output/0/value` | true | false | 写后匹配预期更新的检查结果改变；非张量字段变化导致记录无法严格对齐 |
| `Torch.equal.3.forward/input_args/0/Min` | -32 | 0 | 同路径统计值确有变化，绝对差 32；相对变化 100%，记录配对仍不接受 |
| `Torch.equal.4.forward/output/0/value` | false | true | 目标区域仍保留写前值的检查结果改变；同样不能绕过结构门禁 |
| `Torch.equal.4.forward/input_args/0/Max` | -32 | 136 | 绝对差 168；相对变化 525%，只作算术诊断 |
| `Torch.equal.4.forward/input_args/0/Min` | -32 | 17 | 绝对差 49；相对变化 153.125%，只作算术诊断 |

该现象与原张量更新丢失相容；别名身份、写入合法性及消费者完整值仍需要机制验收的独立运行时证据。本次统计不能独立唯一证明这些事实。

## 复核、文件和限制

- [15 摘要](diagnostic15_summary.json)、[17 摘要](diagnostic17_summary.json)、[18 摘要](diagnostic18_summary.json) 保留严格重算结果、原始三件套哈希、实际步数、未配对 API、参数变化、可配对数值和不可观测字段。它们是诊断摘要，不是新采集原始 dump。
- [离线诊断工具](diagnose_statistics_alignment.py) 不接受命中日志或完整张量参数。只有操作名、出现次数、完整非统计描述、逐操作解析后的原始栈和 construct 项都满足要求，才列入有限局部配对。相同 API 的调用次数不同会拒绝按序号推定配对。
- 原始三件套和完整诊断保留在 `private/stat{15,17,18}_{baseline,fault}` 及 `private/diagnostic*_details-a3.json`，不进入公开提交。原始文件未重写。Fault18 重新计算的严格报告与远端既有 `independent18-a1.json` 完全一致。
- [CPU 回归](test_diagnose_statistics_alignment.py) 的 9 项通过：未配对 API、标量参数变化、同 API 次数变化、栈/construct 不齐、null/NaN/零基线、固定阈值、B 自比、摘要不把被拒记录的算术差转成首差、配置不一致时拒绝全部局部配对。CPU 诊断测试不算 NPU 验收。
- null construct 只表示没有记录关系，不能独立证明别名或真实 producer-consumer 身份；相同统计也不能证明完整张量相同。本诊断不认定模型架构、预训练质量、自然推理链覆盖或多模型原生支持。

从仓库根目录，在持有原始数据的环境复算，例如：

```sh
python experiments/overnight_20261116/diagnose_statistics_alignment.py \
  --baseline experiments/overnight_20261116/private/stat18_baseline \
  --fault experiments/overnight_20261116/private/stat18_fault \
  --rtol 1e-5 --atol 1e-8 \
  --output /tmp/diagnostic18-summary-new.json \
  --private-details-output /tmp/diagnostic18-details-new.json
```

工具拒绝覆盖已有输出文件。严格比较仍调用既有 `fault39/tools/compare_msprobe.py` 和 `compare_model_capture.py`，没有更改其阈值、过滤范围或验收规则。
