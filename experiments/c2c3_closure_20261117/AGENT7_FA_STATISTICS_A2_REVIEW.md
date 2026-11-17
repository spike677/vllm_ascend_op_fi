# Agent 7：FA 新采集 B/F 的独立 statistics 复核

**固定 `rtol=1e-5、atol=1e-8` 下没有超阈值差异，因此本轮不能授予 E5 首差定位，也没有达到0.1%的统计可观测性目标。** 原始统计有4项精确变化，保留10项null不可观测。这个结果不等于完整张量没有变化。

摘要：[agent7_fa_statistics_a2_summary.json](agent7_fa_statistics_a2_summary.json)。程序：[review_fa_statistics_124.py](review_fa_statistics_124.py)。本阶段先独立完成并冻结统计结论，只读取以下两组的 `config.json`、`collector_identity.json` 和 `observation/step0/rank0/{dump,stack,construct}.json`：

- B：`fa28-statistics-vision17-census-a2/fa-fresh-b8-joint-baseline/run/2-baseline_repeat`
- F：`fa28-statistics-vision17-census-a2/fa-fresh-b8-joint-fault/run/2-fault_repeat`

没有读取 root 的先行结论、运行时命中、tiling/profile、完整张量或任何预期首差位置。冻结比较器按原始JSON十进制数字比较，再由另一份直接Decimal遍历独立交叉核对；两者得到相同的变化数量与位置。比较式固定为 `abs(F-B) > 1e-8 + 1e-5*abs(B)`，baseline为参考，不加epsilon、不降低阈值。

| 复核项 | 结果 |
| --- | --- |
| 原始配置 | 两组均 `statistics/mix`、`summary_mode=statistics`、`precision=high`；仅输出目录不同 |
| 对齐 | 原始张量结构、算子顺序、调用栈、模块关系、collector identity全部对齐 |
| 数据范围 | 1个实际API记录，10个tensor统计记录；仅真实step0，代表重复的单算子prefill采集，不是decode token或完整模型 |
| 统计字段 | 每组30项finite、10项null，NaN/Inf/unsupported均0 |
| B自比 | 精确变化0、超阈值变化0 |
| B/F | 精确变化4、超阈值变化0，`first_exceedance=null` |
| 首个精确变化 | `NPU.npu_fusion_attention_v3.0.forward/output/0/Mean`，低于容差，不能称为合格的首差定位 |

4项精确变化如下，B/F实值保留原始JSON精度：

| 原始位置 | 字段 | B | F | 绝对差 | 固定容差 | 超阈值 |
| --- | --- | --- | --- | --- | --- | --- |
| `output/0` | Mean | `6.687331080001968e-7` | `6.68727906827371e-7` | `5.2011728258e-12` | `1.0006687331080001968e-8` | 否 |
| `output/0` | Norm | `118.3081044305463` | `118.30810443395829` | `3.41199e-9` | `0.001183091044305463` | 否 |
| `output/2` | Mean | `745.0740354663358` | `745.0740355780945` | `1.117587e-7` | `0.007450750354663358` | 否 |
| `output/2` | Norm | `1247311.8798017192` | `1247311.8799793604` | `0.0001776412` | `12.473118808017192` | 否 |

`output[0]` 的 Max均为`0.9296875`，Min均为`-0.9453125`；Norm相对变化约 **`2.884e-9%`**。`output[2]` Norm相对变化约`1.424e-8%`。均未达到0.1%；Mean不参与幅度达标。本轮不能继承历史输入、旧变体或其他构建的统计幅度结论。

10项null逐项保留在摘要的 `independent_decimal_crosscheck.unobservable_fields`：mask的Mean/Norm；空`output[3]`的四项；`output[4]`和`output[5]`的Mean/Norm。它们没有按相同数值计入。bool mask的Max/Min原字节为JSON布尔值，只在明确`torch.bool`的这两个极值字段解释为1/0；其他dtype的布尔数值仍拒绝。摘要中的可比标记仅表示已提供的可观测字段能够比较，不表示全部张量值已比较或null已变成可观测。

两份collector identity字节一致，记录489个包文件身份，披露使用 `attention-msprobe-cpu-fp64-v1`：运行时替换 `PytorchDataProcessor.get_stat_info`，对支持的完整浮点张量执行阻塞CPU拷贝后以FP64求统计；bool、integer、empty等回交原实现。adapter SHA256为`1e66f220c1098a71eca9611bbc0cb582a088e7e47819269a5a8dbf3a8bb50b48`。这不是声称使用完全未替换的默认统计方法；本阶段核对的是两组原始身份记录的一致性，没有借运行时日志选择首差。

| 身份 | SHA256 |
| --- | --- |
| B dump | `ccfb4b0eb2738be4d50954a4fc10f877c86c576d9d6f2b5b407df0fe26e8073c` |
| F dump | `180c67028971cdbe81b0fbc9f2e1b9c464cea65e4a5b6d973c9c8299cc054b43` |
| 相同collector identity | `d6299a080c0d5e6a69ccbcc53402ec1defcb0d562934fe3ccd79b536ca5328d9` |
| 冻结比较器 | `d5633600f8bde98590c4eaf367eb16d76d4e6c9ad5b0cb2f82d434ae6a377ca2` |
| 独立审计程序 | `fe6e242c52f040492876d62c5d99010898330fd96eabd47cea2eee23040be140` |
| 独立摘要 | `31d61e61d578b69ab088e9a56a2fe3a3fba1239a85b599f988fbbdf11765662c` |
| 完整统计审计报告，留124 | `98212a752ec990f10e6f60f9e236fb06d30c016fee8c07934559461280b4c347` |
| B自比报告，留124 | `26a6d7d8c723ee7c6a3057fdb461bf2a2efadecd1c9b29f741006308ec0bb02f` |

独立产物位于 `/home/17-fault-c2c3-20261117/agent7-fa-statistics-a2-review-a1`，公开摘要与远端摘要逐字节一致。8项CPU交叉计算回归通过，覆盖固定容差、严格边界、零参考、Mean排除、null/NaN、类型明确的bool极值、更早算子、结构不齐和Inf迁移。没有在本提交上传原始B/F三件套或完整张量。

真实launch、策略与重复稳定性需要另行审核，证据入口与本阶段隔离；这些结果不能覆盖本轮“无超阈值统计首差”的结论。C2/C3和完整模型验收均保持未授予。
