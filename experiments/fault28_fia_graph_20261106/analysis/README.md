# FIA 图模式的独立分析

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

动态 N 宽度是由已核验 key/raw 与实际 kernel 源码推导的执行计划，不是设备内 trace；公开 host raw 也不是直接读取的设备入口参数。N 同时影响 BMM1 的 N、Softmax 的 Reduce 及 BMM2 的 K，因此没有唯一分离某条 Reduce 指令的误差贡献。

原 FP32 统计轮的严格首差验收失败必须保留。M256 的完整输出已有 81 个差异，但 FIA 的 FP32 四统计没有分辨它；GE CSV 还存在六位显示精度限制。新增的 CPU FP64 采集器对所有真实浮点张量统一计算四统计，不按算子、shape 或注入模式筛选，不改变原张量及元数据。`stage_collector64_cpu.py` 保存从官方 processor 出发的完整补丁和包指纹；`observer_neutrality.py` 对比两轮所有计算输出，验证观测变化没有改变计算。

```bash
python -B -m unittest discover -s analysis -p test_graph_evidence.py -v
python -B -m unittest discover -s analysis -p test_native_plan.py -v
```

这些测试验证输入先变、缺失数据、空输出、null、任意算子生产者、图输出别名、外部进程及额外 launch 等反例。CPU 测试成功不能替代 NPU 验收。最小 API/module 实验中的 step0/1 是相同 prefill 子图的两次回放，不是两个模型 token。
