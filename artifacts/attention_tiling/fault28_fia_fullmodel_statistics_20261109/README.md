# 故障28：正常FIA与故障FIA两组statistics/mix

**当前只交付两组msprobe采集：正常FIA B和故障FIA F。** 两组都显式接入真实FIA完整Qwen3图，未将模型默认SDPA作为B。原始采集于2026-11-09在124卡5完成；2026-11-10按要求收敛交付并重新盲验，没有重新占用NPU。

| 组别 | 目录 | 实际FIA策略 |
| --- | --- | --- |
| 正常FIA B | [model_b8_baseline](data/cases/model_b8_baseline/) | 同causal规则，M512 |
| 故障FIA F | [model_b8_fault_r1](data/cases/model_b8_fault_r1/) | 同causal规则，M128 |

两组同模型、同B8输入、同请求顺序、同2048-token prefill、同采集器和配置。正常B是本受控实验的M512基线；两组均沿用已披露的私有vendor及共同causal尾块规则，不代表未经适配的HF默认路径或历史盘古原始策略。

[两组选择](data/primary_BF.json) · [实际接入与调用链](../../../experiments/fault28_fia_fullmodel_20261109/README.md) · [两组验收摘要](data/reports/acceptance.json) · [独立首差复核](local_fault_first.json) · [发布字节与覆盖复核](local_public_verification.json)

## 采集内容

统一 `task=statistics, level=mix, summary_mode=statistics, precision=high, async_dump=false`，同步、无张量筛选。每组保留step0/1、rank0的原生 `dump.json`、`stack.json`、`construct.json`、配置、采集器身份及完成回执，共4套统计三件套。step是同一prefill图的回放，不是decode token。

两组均包含全部原生图CSV、实际FX/GE图、全部28层真实Q/K/V/FIA输出及logits的GraphBoundary统计。统一CPU FP64四统计增强及其[补丁和指纹](data/collector/)原样披露，未按故障位置选择张量。主交付没有`.pt`等完整张量。

每组含1次warmup和2次真实后端回放；合计6次完整图执行、168次FIA节点执行、1,344个层/请求/执行组合。28/28层、8/8请求的实际节点、共享kernel二进制身份、原生任务/runtime实例化及168份FIA迭代CSV交叉核验，见[两组覆盖报告](data/reports/coverage/)。一次host回调不计为28次运行时命中。

## 首差及传播

所选B/F的第0层全部请求Q/K/V完整数组逐元素及字节SHA一致，Attention改变1,429/33,554,432个元素。该层全部请求末query仍相同；末query差异从第1层Attention出现。最终8/8请求首token logits改变，最大绝对差0.0546875，Top-1均未变。[原始完整比较报告](data/reports/fault_r1_complete.json)保留，私有完整张量未上传。

独立推断只读原始统计、配置、采集器身份与真实FX/GE图/CSV；物理盲验目录没有case/result、host、策略或完整数值报告。所选F两步均按图依赖找到**第0层FIA output0**的Mean/Norm首差，见[原推断](data/reports/fault_r1_msprobe.json)和[重新盲验](local_fault_first.json)。原生module仍是GraphBoundary；输出叶子`$[0][4]`通过实际NetOutput映射FIA，没有把module记录改名为eager API。

默认六位GE CSV首个可见差异仍在`Mul_13`输入1，之前326个无唯一CSV节点保留为gaps；FIA首差依赖已披露的高精度统计及真实输出别名。null、空LSE和缺失记录保持不可观测；四统计相同不证明逐元素相同。共享Data的重复CSV原件保留，只在同执行组完整行相同才等价合并，冲突继续拒绝定位。

```bash
python data/analysis/verify_public_bundle.py \
  --primary-bf-only \
  --bundle data \
  --blind-work /path/to/new-empty-review \
  --output /path/to/new-verification.json
```

## 交付范围和来源

`data/`共9,486个文件、51,822,225字节，9,475份原始文件保留来源映射。原始dump/stack/construct和CSV字节未改；[source_mapping.json](data/source_mapping.json)记录原路径、公开路径、长度与SHA，[SHA256SUMS](data/SHA256SUMS)覆盖全部导出文件。[publication_scope.json](data/publication_scope.json)明确两组选择、父提交和新增整理材料；新增摘要不是新的NPU采集回执。

本次不附B1、独立F、OFF、无别名或默认SDPA采集组；完整10组历史矩阵及其控制结论仍在[原发布提交2aa4432](https://github.com/spike677/vllm_ascend_op_fi/tree/2aa4432ce92a2c878f6fb6ce3a648b1049df4b48/artifacts/attention_tiling/fault28_fia_fullmodel_statistics_20261109)。历史源清单指纹保持原样，当前仅附两组运行配置。没有重写历史提交。

范围是CANN9.1、910B4卡5、Qwen3-0.6B/FP16/TP1/B8/2048-prefill的显式注册FIA完整模型图；不是任意形状、图模式KV-cache decode、历史盘古默认策略或业务质量验收。M引起的宽度同时影响BMM1/Softmax/BMM2，尚未唯一归因到某条Reduce指令。公开数据不能重读124上私有完整张量、原始大日志或机器码。
