# Fault28 stock → private OFF 独立兼容性门槛

`private_off_compat_a1` 的严格结论是 **FAIL_STOCK_OFF_COMPATIBILITY**。B1 通过；B8 的两个 forward 都记录 `eligible=false`，不满足运行前要求的 B8 资格门槛。不得将数值相等替代完整门槛，也不得将这次 OFF 校准称为故障注入通过。

正式派生报告：[stock_off_compat_a1_verified.json](stock_off_compat_a1_verified.json)。这是只读分析结果，没有重写原始 result、native call、host event、原生日志或 stock profile。早期诊断报告 `stock_off_compat_a1.json` 保留。

| 检查 | 实测 |
| --- | --- |
| B1 28 层；B8 两个 forward 共 56 层的目标 Q/K/V 前后逻辑 SHA、shape、dtype、几何 | 全部与各自相同 batch 的 stock 对照一致 |
| 84 个目标 attention 输出 SHA | 全部一致 |
| 3 次目标 next-token logits SHA、argmax | 全部一致；argmax 均为 3516 |
| 3 次 layer1 public-context raw 与 stock declared-prefix | 1464 字节逐字节一致 |
| 105 个公开 POD 自有字段；host typed 映射 | 精确一致；float 字段按 FP32 打包后比较 |
| 实际 key / block / workspace count | `22619644976` / 20 / 1，一致 |
| OFF schedule / stock schedule | OFF 为 0；stock profile 无此项，比较状态为不可观测 |
| OFF workspace vector / stock vector | 只记录 OFF；stock 的 API 总 workspace 含辅助算子，不能与 kernel 向量等同 |
| 原生日志加载记录和文件 SHA | 实际私有 opmaster/proto 加载路径与 launcher/manifest/现存库 SHA 一致 |
| OFF 控制 | 全部 callback status 0、policy_applied false；B1 eligible false 正确，B8 eligible false 失败 |

比较器先独立比较逐层数值，再读取 host 控制事件；不会利用注入命中记录指定模型首差异。本次未观察到目标数值差异。读取的是保存的逻辑 SHA，未重新读取私人张量 `.bin`，因此不能从这些报告推导数值误差大小或完整 batch 的逐元素值。OFF layer1 另有完整 batch 输入的前后 SHA 自洽检查，但 stock 的目标行 SHA 不能扩张为完整 batch 对比。

Q/K 的实际格式 0→2 转换仅在逻辑 SHA、几何和同一次调用内对象/存储身份保持时允许；两种运行必须具有相同的转换轨迹。跨进程不会比较对象地址或 PID。原生模型模块源码 SHA、模型配置、FP16、causal/scale/GQA 参数、已知 token 前缀和执行选项均需一致；采集探针源码版本不同会明确记录。

完整 raw 相等同时覆盖嵌套库结构的字节；105 字段解码并不冒充这些嵌套库结构的逐字段解码。私有 op_api 的优先加载路径和文件 SHA 作为附加证据保留，实际 opmaster/proto 加载与 selected public-context 回调单独验证。API 窗口与同一进程的 FA launch 可关联；设备 launch 的工作线程不伪称为调用线程。

标准库用法（在能读取原始证据和 vendor staging 的主机运行，无 NPU 计算）：

```bash
python3 verify_stock_off.py \
  --root /home/fault28-native-tiling-20261029 \
  --stock-run hf_stock_2048_a3 \
  --off-run private_off_compat_a1 \
  --schema /home/fault28-native-tiling-20261029/build_off/fa_policy/fa_pod_schema.json \
  --vendor-manifest /home/fault28-native-tiling-20261029/build_off/fa_vendor_stage_a1/vendor_manifest.json \
  --output new_report.json
```

输出必须为新文件；通过退出码 0，严格失败退出码 1。后续 vendor 重建必须指定对应的新 manifest，不能沿用旧库 SHA。`test_stock_off.py` 的 17 项 CPU 检查覆盖输入改写、输出差异、错误位置/causal、非法格式转换、POD 错误及 a1 的资格失败不可被数值通过掩盖。结果见 [stock_off_cpu_checks.json](stock_off_cpu_checks.json)；这些是分析器回归，不是额外实机测试。
