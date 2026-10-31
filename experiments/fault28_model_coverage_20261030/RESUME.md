# 交接状态：正式验收完成，发布准备中

2026-10-31，`formal_stream_a1` 的 B1/B8 完整模型联合验收已通过：
`PASS_MODEL_ALL_ELIGIBLE_NATIVE_STRATEGY_COVERAGE`。
联合报告 SHA256：`16e82c3e072442839465a4e2b604d891430ead77bc2aa018a8f7ce9fe6fed8cf`。

**不要重跑 NPU；后续仅进行证据审查、打包和 GitHub 发布。** 自有容器已移除，设备锁已释放；未停止外部服务。
GitHub 本轮模型证据发布尚未完成，不能将此前算子 replay 的发布当作本轮模型发布。

## 已验收结果与范围

| 项目 | 实际结果 |
| --- | ---: |
| 正式原始 SDPA 调用 | 560 |
| 符合条件的受控 prefill 实际切换 | 112/112 |
| 受控 decode 原 API 透传 | 112 |
| 受控 layer/request 映射对 | 504 |
| 模型层数 / 不同请求数 | 28 / 8 |

全部五轮含280次 prefill：112次受控切换，168次 calibration/native/off 保持原生策略；
另有280次 Q1/KV2049 decode，全部明确列为证书范围外。B8一次原生调用承载8个请求，
504个映射对不是504次 kernel 调用。B1/B8分别在各自模型进程完成五轮。

四组 native/fault 的原始 msprobe 独立首差均为 prefill 第0层 SDPA output Mean；
后层变化区分上游传播。两次 controlled 持续命中且重复稳定，off 后各层、各请求输出及完整 logits 恢复。
统计保留 null，不把相同摘要当逐元素相同。定位限于框架 SDPA 输出边界，不唯一认定 kernel 内部 Reduce、
准确率恶化或历史 vLLM FIA 缺陷复现。decode 直接调用原 API，没有经过 Session 的 unsupported 分支。

## 保留的历史运行审计

- `pilot_a1`：旧采集代56次校准调用，不计入正式560次。
- `formal_a1`：原采集器重复扫描增长日志前缀而变慢，由本实验 STOP 主动结束；保留67次已返回调用，
  尚未开启故障，不能计作正式验收。原日志2072084876字节，SHA256
  `c669e9c527c9064b7410e2f1c70f3f1a8e4722c2bb69def9826429ed410e4212`。
  STOP已归档为该 attempt 的 `STOP.saved`；自有容器清理完成，没有外部服务操作。
- 新顺序 reader 在旧实采日志的66个窗口上与原解析结果逐字段等价；原 `actual_launch` 精确字符解析保留。
- `pilot_stream_a1`：新代56次校准调用通过独立准入，报告SHA256
  `37e73c33cf47b3023404d17d471ae3831d0219ecde0be6291a7efeaa2994e465`。
- `formal_stream_a1`：使用与新pilot相同的47个冻结执行源，正式560次调用完成并联合验收通过。

上述总计739次已返回原始 SDPA 调用，不是739个全部模型算子或 kernel；仅最终560次属于正式矩阵。
旧代 source 快照、部分采集、停止记录和完整原日志继续保留，不用新源码覆盖旧证据。

## 发布阶段交接

1. 依据已通过的正式联合报告审查公开统计、逐调用元数据、来源与独立推断报告；保留 high collector 修改前提及限制。
2. 校验公开子集清单、原始 JSON/源码 SHA、README 和相对链接；完整 private logits 和巨型原生日志不混入主统计包。
3. 在已隔离的发布分支完成提交与 GitHub 推送，再更新发布状态。不得因此重启采集、修改已冻结原始证据或核发新的核心证书。

`local_source_manifest.json` 是当前本地发布审查清单，**不是实机执行快照**。
实际执行身份以各 attempt 的 `source_manifest.json`、`source/`、completion 及联合报告为准。
实现说明见 [analysis/IMPLEMENTATION.md](analysis/IMPLEMENTATION.md)，最终范围与结果见 [README.md](README.md)。
