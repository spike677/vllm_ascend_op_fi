# FIA 图模式的独立分析与设备粒度核验

本轮分开运行统计定位、执行策略核验和完整数值比较，最后连接证据。首差推断不以“应该是 FIA”、期望 profile 或命中日志作为定位输入。

`graph_evidence.py` 只读取原生 `dump.json/stack.json/construct.json`、采集配置与采集器身份，以及实际 FX/GE 图和原生 GE CSV。它不读取 case 的 mode、host 命中日志、预期 profile 或故障位置标签。

比较分为两条证据链：

1. 同 Batch 的 B/F 统计按实际 GE 依赖关系对齐；逐项保留输入、输出统计、metadata 和不可观测项。原生 GE CSV 的首个可见差异独立记录。
2. `GraphBoundary` 的原生 module 输出统计通过实际 `NetOutput` 边映射到生产者。该子图的 slot0 直接返回 FIA output0，slot1 返回真实 `o_proj` 消费者。候选按图依赖排序，较早的 Data/input 差异优先，不按 tuple 顺序或预期故障名指定首差。这是输出别名的定位推断，没有把 module 记录改名为 eager NPU API。

```bash
python analysis/graph_evidence.py \
  --baseline /path/to/B --fault /path/to/F \
  --output /path/to/fresh_report.json
```

该命令可读取发布包中保留原字节、缩短文件名后的数据。`source_mapping.json` 记录原路径、发布路径及 SHA-256。统计不能证明张量逐元素相等；跨 Batch 的聚合统计也不能直接比较同一 query。

`compare_targets.py` 在实机私有目录读取完整目标输出，覆盖全部 16 heads × 2,048 queries × 128 dimensions，单独检查消费者、重复、差异位置和固定位置的 CPU FP64 参考。它不参与 msprobe 首差定位。完整张量不进入发布包。

`public_plan.py` 独立解码实测 SDK serializer layout 的全部 65 个范围，将 63 个逻辑字段与真实 public host raw 对照，并检查无 SplitKV、完整 M 分核和合法 key 覆盖。`native_plan.py` 关联真实 host 选择、CANN JIT 编译、GE 静态任务及三次实际图迭代；不会把静态注册列表当执行。`audit_case.py` 进一步确认编译器实际收到的 tiling raw/key 与公开 host 记录一致，并将实际任务绑定到编译二进制 SHA。

本轮在真实 `SoftmaxFlashV2Compute` 的 `SoftMaxShapeInfo` 和实际调用之前采集了设备端 `columnCount/actualColumnCount`。`device_trace.py` 解析官方 `show_kernel_debug_data` 的原始 stdout，与相同 owner PID 的 runtime `ParsePrintToLog: PrintInfo:` 行作多重集合完全相等检查，拒绝编译器printf格式串、其他进程、重复和缺失记录。19组各42条、共798条，七个固定query的覆盖及唯一形态重数3通过。仅采request0/query-head0，不表示所有请求或query都做了标量trace。

图printf异步刷出，原逐次时间窗检查失败保留。最终只验收进程/图级汇总；`individual_scalar_to_graph_call_pairing_proven=false`，不能据重数3声称每条已配对到单次replay。公开host raw不是直接读取的设备入口参数，原生完整日志、标量缓冲与二进制仍需远端核查。N同时影响BMM1、Softmax及BMM2，实测列数也没有唯一分离某条Reduce指令的误差贡献或记录指令树。

原 FP32 统计轮的严格首差验收失败必须保留。M256 的完整输出已有 81 个差异，但 FIA 的 FP32 四统计没有分辨它；GE CSV 还存在六位显示精度限制。新增的 CPU FP64 采集器对所有真实浮点张量统一计算四统计，不按算子、shape 或注入模式筛选，不改变原张量及元数据。`stage_collector64_cpu.py` 保存从官方 processor 出发的完整补丁和包指纹；`observer_neutrality.py` 对比两轮所有计算输出，验证观测变化没有改变计算。

本轮沿用上述同一FP64采集器；新的`observer_neutrality.py`比较无设备printf和有printf的23组/69对调用，完整目标字节、整批/逐请求哈希与实际host raw/key全部相同。`join_measured_chain.py`在独立首差推断后连接实际Batch/M、设备列数和完整数值；主B/F固定同B8、同causal规则，排除跨Batch消费者自身差异。验证范围是当前版本的受控FIA→o_proj图子图。

`verify_public_bundle.py`检查全部发布文件SHA及原字节映射，重解码ABI并核对798条标量，随后构造物理盲验目录：只放配置、采集器身份、统计与FX/GE图/CSV，排除case/result、host、策略、设备和数值报告。两个F的两步均重新推断首差为FIA output0 Mean/Norm。

```bash
python analysis/verify_public_bundle.py \
  --bundle /path/to/public/data \
  --blind-work /path/to/new-empty-review \
  --output /path/to/new-review.json
```

```bash
python -B -m unittest discover -s analysis -p test_graph_evidence.py -v
python -B -m unittest discover -s analysis -p test_native_plan.py -v
```

`run_tests_cpu.py`固定运行`test_device_trace.py`、`test_graph_evidence.py`、`test_native_plan.py`、`test_package_evidence.py`。27项在无设备Linux容器通过、无跳过，覆盖输入先变、缺失数据、空输出、null、任意生产者、图输出别名、外部/额外launch、伪造列数、缺失/重复标量和路径逃逸。CPU成功不能替代NPU验收。step0/1是相同prefill子图两次回放，不是两个模型token。

`package_evidence.py`只导出审查后的统计、图、源码/配置、元数据和日志行提取，保持原生证据字节。正式导出1,514个文件、31,108,919字节，1,492份原始文件有逐项映射；23组含46套dump/stack/construct。完整张量、原始标量缓冲、库和完整DEBUG日志不发布。
