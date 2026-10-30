独立统计分析只在整个 attempt 成功并清理完成后执行。原比较器
`compare_replay_statistics.py` 保持冻结版本；批量工具为每一对保留完整报告、输入 SHA 和报告 SHA。

固定计划共 88 对，不按结果挑选：discovery 同 batch 独立重复 4 对；每个受控目标 28 对，
包括各 case 的 0→1、0→2、0→3、1→2，两个独立 repeat 的四个 step，以及 stock/private native。
目标为 query_tile、key_group、joint。B2/B4 仅 discovery，不进入 controlled 比较。

```bash
python3 -B analysis/compare_matrix_statistics.py \
  --group discovery --discovery-attempt outputs/discovery_a4 \
  --output outputs/independent_statistics/discovery_a4

python3 -B analysis/compare_matrix_statistics.py \
  --group query_tile --discovery-attempt outputs/discovery_a4 \
  --controlled-attempt query_tile=outputs/CONTROLLED_QUERY_TILE_ATTEMPT \
  --output outputs/independent_statistics/query_tile_REPORT
```

同样分别运行 key_group 和 joint；或者 `--group all` 并提供三个
`--controlled-attempt target=path`，一次输出全部 88 对。默认 CPU 时间上限 120 秒，
输出目录必须全新，不能与原 attempt 目录重叠。脚本不启动容器、设备调用或任何调度。

首差只来自 config/dump/stack/construct，completion/receipt/collector 和 result 的有限字段
仅作完整性门槛。host/hit/profile 日志、私有张量、策略意图、资源计量不用于定位。
完整 source stack 不丢帧或改写；stock/private 无法对齐时保留 INCONCLUSIVE。
主 attention 输出与五个辅助返回分开，seed/offset 的变化不升级为主输出故障。
null 不当作相等；有限统计一致仍不证明张量相等。Off 的统计恢复也不证明逐位恢复或根因。

`test_compare_matrix_statistics.py` 使用明确标注的 CPU 数据验证全部 88 对和 SHA 绑定，
包括 source stack 不匹配与 null 情况；不代表真实 SDK/schema 或硬件验收。
