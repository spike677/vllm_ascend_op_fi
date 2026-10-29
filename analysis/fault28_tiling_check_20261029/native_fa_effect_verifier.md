# Fault28 原生 FA ON / guard / recovery 诊断验收

`private_on_validation_a2` 的真实数据已通过 `PASS_NATIVE_FA_EFFECT_DIAGNOSTICS`，报告见 [native_fa_effect_a2_report.json](native_fa_effect_a2_report.json)。B1 guard 两次不变；B8 ON 两次均由有序目标哈希推导出 layer1 output 首差，logits 改变、argmax 仍为 3516；同一进程关闭后所有目标层、raw 与 logits 恢复。该结果仅覆盖运行时与原生 tiling 诊断。后续 [独立 msprobe 统计分析](msprobe_inference_a2/README.md) 的首个被观测差异落在下游 layernorm `pow`，因此不能将本诊断通过等同于 msprobe 精确故障点定位通过。

17 项 CPU 检查用于验证分析器能拒收错误证据，不作为额外实机复现数据。

分析器 [verify_native_fa_effect.py](verify_native_fa_effect.py) 的前提是独立 stock→private OFF 报告严格通过，且其中 OFF result SHA 与本次实际对照一致。输入固定为 `private_off_compat_a2/fa_off_b1`（1 forward）、`fa_off_b8`（2 forwards），以及 `private_on_validation_a2/fa_on_b1_guard`、`fa_on_b8`、`fa_recover_b8`（各 2 forwards）。目录名可通过 CLI 指定。

每层 target Q/K/V 输入与输出按实际顺序比较，先推导 `first_observed_value_difference`，再与预先约定的 layer1 output 位置比较。分析器不会把预期位置或命中记录写入实际首差结果。上游层已变化、选中层输入先变化、只有元数据变化、只有命中而没有输出/logits 变化，都会失败。

验收分为以下独立证据部分：

- 数值：layer0 目标输入输出不变，layer1 完整 batch 的 Q/K/V、mask 和 scalar 参数不变，layer1 目标输出首次变化并传播至 logits。ON 重复两次的目标值一致。原生格式 0→2 规范化只在同调用前后值/几何/身份保持、B/F 轨迹一致时允许。
- 公开 tiling：actual native 日志中的每个 FA launch 与真实 API 窗口、调用序号对应；选中层 public-context raw 必须等于实际 launch 的 declared-prefix。105 个公开 POD 自有字段与 typed 记录一致，完整 raw 字节另外检查。两次 ON 的 M=64、S2 base=128、ratio=4、group=512；两个 forward 的其余 54 次 native raw 均与 OFF 同层完全一致。
- 合法归约：先核对固定 kernel 源码 SHA，再移除 mode、eligible、applied、token 等 selector 字段后调用 `derive_query_trace.py`。每个真实 query 只归一个 core、合法 causal keys 恰覆盖一次、目标 16 heads 的归约组从 1024 变为 512，precision/mask/合法 key 集合保持。它是源码推导的 CPU oracle，不是硬件执行轨迹或数值等价证明。
- 控制与恢复：B8 ON 两个唯一 callback 都 eligible/applied、status=0；B1 ON 两个 callback 不匹配，56 次 native raw、所有目标层输出和 logits 均等于 B1 OFF。同一 recovery 进程的 forward0 ON 与独立 ON 运行一致，forward1 OFF 的全部目标层、logits 和 raw 均恢复到 OFF 对照。私有库实际加载记录与同一 vendor manifest/关键库 SHA 另外验证。

整体成功状态仅为 `PASS_NATIVE_FA_EFFECT_DIAGNOSTICS`，`mechanism_accepted` 固定为 false。最终 B/F 验收必须由主流程另行结合独立的原生 msprobe 首差分析；本工具不读取 msprobe，也不能替代 msprobe 定位。target 哈希不能扩张为所有 batch 行输出相等，logits 变化也不要求 argmax 必须变化。

示例（路径需指向实际 a2 冻结源、manifest 和已通过的 OFF 报告）：

```bash
python3 verify_native_fa_effect.py \
  --root /home/fault28-native-tiling-20261029 \
  --off-run private_off_compat_a2 \
  --on-run private_on_validation_a2 \
  --stock-off-comparator /path/to/verify_stock_off.py \
  --extractor /path/to/extract_fa_log_profile.py \
  --trace-oracle /path/to/derive_query_trace.py \
  --source-root /path/to/a2/source/checkout \
  --schema /path/to/a2/fa_pod_schema.json \
  --vendor-manifest /path/to/a2/vendor_manifest.json \
  --off-compat-report /path/to/passed_off_report.json \
  --output new_effect_report.json
```

全部操作是标准库 CPU 只读分析，不导入 torch、不启动 NPU、不改写原始日志。输出路径必须不存在；未满足门槛时退出码为 1，并保存实际错误。检查记录见 [native_fa_effect_cpu_checks.json](native_fa_effect_cpu_checks.json)，测试源码为 [test_native_fa_effect.py](test_native_fa_effect.py)。
