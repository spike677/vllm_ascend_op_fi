# 参考依赖与研究边界

当前必要运行依赖：[fake_requant_reference](fake_requant_reference/README.md)。按固定Git提交纳入运行源码与原5项数学/config测试，不合入其他研究driver，不依赖研究分支工作树当前内容。

| 历史路线 | 保留位置／边界 |
| --- | --- |
| Fault15 | fault15_scale_granularity，统计粒度研究 |
| Fault20 | fault20_fast_validation/source下的l4_buffer_substitute_allhit_v1、l4_metadata_lifecycle_proxy |
| Fault26 | 同source下readonly_trace与fia_writer_proxy；受控stomp不自动证明物理OOB |
| Fault39 | 共享native和reports；自然nonfinite与output proxy分开 |
| Fault41 | 外部实验FUNCTIONALLY_COMPLETE WITH_KNOWN_MSPROBE_OBSERVABILITY_LIMITATION；canonical提交能力仍有边界 |

见[故障索引](../docs/faults/README.md)。本轮不搬native、不把外部probe变成产品、不升级manifest。canonical为D盘正式仓库；旧C盘clone/nested worktree只读保留。其他独有修改去向见[发布盘点](../docs/releases/source_baseline_v1/README.md)。
