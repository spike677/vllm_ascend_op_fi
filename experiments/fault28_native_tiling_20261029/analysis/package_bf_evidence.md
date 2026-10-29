# Fault28 high B/F 证据打包工具

`package_bf_evidence.py` 已完成 15 项 CPU 临时样本检查，结果见 [package_bf_cpu_checks.json](package_bf_cpu_checks.json)。真实 `bf_statistics_high_a1` 的 capture 与 collector 来源 schema 也已只读预检。**尚未生成正式包**；必须由主流程完成独立科学验收并提供最终 acceptance 后才可调用 `create`。

工具只返回 `VERIFIED_PACKAGE_INTEGRITY`，不会自行生成科学 PASS。最终 acceptance 必须为 `PASS_FAULT28_NATIVE_TILING_AND_HIGH_STATISTICS`，固定 attempt 为 `bf_statistics_high_a1`，并明确：

- `native_msprobe_first_difference_pass`、`mechanism_validation_pass`、`formal_runtime_consistency_pass` 均为 true。
- `collector_variant="task_local_msprobe26_1_fp32_high"`，对应 stage 内部版本 `task_local_fp32_high_v1`。
- `official_default_msprobe_exact_location_pass=false`、`historical_defect_reproduced=false`。

acceptance 的 `source_reports` 必须逐一绑定下列八个**包内相对路径**及原文件 SHA；不允许引用自身或任意外部路径：

| 包内路径 | 内容 |
| --- | --- |
| `summaries/off_compatibility.json` | 已通过的 stock→private OFF 对照 |
| `summaries/mechanism_validation.json` | ON/guard/recovery 原生机制诊断 |
| `summaries/formal_runtime_crosscheck.json` | 当前 high attempt 的值透明性检查，必须绑定本包 B/F result SHA |
| `summaries/independent_msprobe_step0.json` | high step0 原生独立首差推断 |
| `summaries/independent_msprobe_step1.json` | high step1 原生独立首差推断 |
| `summaries/official_default_msprobe_step0.json` | 原官方默认 a2 step0 的定位失败报告 |
| `summaries/official_default_msprobe_step1.json` | 原官方默认 a2 step1 的定位失败报告 |
| `summaries/collector_validation.json` | collector 来源、CPU 和实机透明性独立验证 |

另以 `--summary execution_acceptance.json=...` 提供根验收原文。高精度推断报告必须与包中 config 和六份原生三件套逐文件 SHA 相符，首个被观测差异须是 SDPA.1 输出，28 层覆盖完整；不能把旧官方默认报告的下游 `Tensor.pow.7` 重新标为成功。

正式输出只能是 `artifacts/attention_tiling/fault28_native_fa_high_statistics_bf_20261029/`。每 arm 的原生统计主数据为 config、receipt、两个 step 的 dump/stack/construct，共 8 份 JSON；实际运行时记录、56 份调用 JSON、collector provenance 和 host tiling JSONL 另作证据。只包含 B/F 采集；冻结 source/setup 中的其它校准用例是源码配置，不是第三组采集数据。

collector 只公开五份来源文件和许可证：stage manifest、原安装 source manifest、精确 patch、原 processor、patched processor、`LICENSE.MulanPSL-2.0`。patched 文件从 stage 的真实 `packages/msprobe/core/dump/data_dump/data_processor/pytorch_processor.py` 按原字节复制。完整 stage 库存与已安装 RECORD 清单保留，验证只有这一模块改变；实际 result/receipt/provenance 的 module 路径、方法源码路径、所有 SHA 和批准 case 均必须匹配。原 wheel 归档缺失会如实保留，仅记录 direct_url 中的 wheel SHA，不声称重新哈希了 wheel。

禁止 `.pt`、张量 `.bin`、模型权重、`private_replay`、cache、编译库、大日志、第三组采集和任意额外文件；原生 Tensor JSON 也不得携带原值数组或文件载荷引用。host `raw_hex` 是 tiling 元数据，保留为 JSON。原始 JSON、源码和 patch 不重写；`.gitattributes` 使用 `* -text` 防止提交转换字节。清单覆盖所有发布文件，清单本身除外以避免自引用。

调用形态（实际报告路径由主流程提供）：

```bash
python package_bf_evidence.py create \
  --attempt /path/to/bf_statistics_high_a1 \
  --collector-stage /path/to/collector_high_a1 \
  --collector-license /path/to/LICENSE.MulanPSL-2.0 \
  --summary off_compatibility.json=/path/to/off.json \
  --summary mechanism_validation.json=/path/to/effect.json \
  --summary formal_runtime_crosscheck.json=/path/to/formal_runtime_crosscheck_pinned.json \
  --summary independent_msprobe_step0.json=/path/to/high_step0.json \
  --summary independent_msprobe_step1.json=/path/to/high_step1.json \
  --summary official_default_msprobe_step0.json=/path/to/default_step0.json \
  --summary official_default_msprobe_step1.json=/path/to/default_step1.json \
  --summary collector_validation.json=/path/to/collector_review.json \
  --summary execution_acceptance.json=/path/to/acceptance.json \
  --output /path/to/artifacts/attention_tiling/fault28_native_fa_high_statistics_bf_20261029
```

可选 `--archive` 只能指向同目录同名 `.tar.gz`。目标必须是新目录/文件。下载后可完全离线验证：

```bash
python tools/package_bf_evidence.py verify --package .
python tools/package_bf_evidence.py verify --archive /path/to/fault28_native_fa_high_statistics_bf_20261029.tar.gz
```

离线验证拒收缺失、额外、被篡改的文件、链接、路径穿越、错误 collector、未绑定或夸大结论的 acceptance；它不执行包内模型或 collector，也不把 SHA 完整性当作独立数值因果证明。
