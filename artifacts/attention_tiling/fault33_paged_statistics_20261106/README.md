# 故障 33：原生分页 Attention 策略与统计证据

本次在 124 的 910B4 物理卡 5、CANN 9.1.0 上验证合法执行分块改变造成的数值差异。实现与调用链见[实验说明](../../../experiments/fault33_paged_tiling_20261105/README.md)。这是当前版本的受控机理实验，不能替代历史 MindIE RC2 B080、ATB PA 或特定业务事故的复现。

| 目录 | 内容与验收边界 |
| --- | --- |
| [stock_discovery](stock_discovery/index.json) | 18 次原生调用；物理页大小和 Batch 对照，保留无差异结果 |
| [rejected_a3_bridge](rejected_a3_bridge/README.md) | a3 的守卫未满足，拒绝启动 F；保留原始拒绝证据 |
| [build_a4](build_a4/README.md) | 匹配源码、CPU 构建、私有包、公共 serializer ABI 的指纹与回执 |
| [operator_a4](operator_a4/INVENTORY.json) | 10 次 launch 的运行与输出审计；其中 B/F 各 2 step 共 4 份原始 statistics/mix 三件套，以及配置、原生日志摘录与冻结源码 |
| [stock_model_pilot](stock_model_pilot/README.md) | 真实 Qwen3 28 层的无注入预检；不作为 B/F 结论 |
| [reports/replay_a4_acceptance.json](reports/replay_a4_acceptance.json) | 算子 B/F、重复、关闭恢复、独立统计的联合验收 |
| [model_a4](model_a4/INVENTORY.json) | 四个干净模型回合的 4 份原始统计三件套、运行和逐层映射、实际 launch 摘录及相同冻结源码 |
| [reports/model_a4_acceptance.json](reports/model_a4_acceptance.json) | 完整模型、全部符合条件调用覆盖、重复、恢复及独立首差的联合验收 |
| [reports/public_model_verification.json](reports/public_model_verification.json) | 从公开原始统计重算四对首差，并复核 112 条 launch、112 份公共 raw 与完整哈希记录 |
| [resource_prelaunch](resource_prelaunch/) | 四次启动前资源回执，与控制器执行窗口的资源记录分开 |

算子级为 `PASS_CONTROLLED_REPLAY_AND_STATISTICS`：固定 B8、长度 16,384、128-token 物理页及全部输入，切换真实 host 的 KV 计算分块 `8192 → 1024`，原生代码重算所有依赖 tiling，记录循环 `2 → 16`。Attention 有 5,371 / 16,384 个元素变化，最大绝对差为 0.000030517578125；8 个请求均出现差异。固定策略重复逐 bit 稳定，同进程 OFF/ON/ON/OFF 恢复完整输出及公共 host raw。两对 B/F 的 msprobe 独立首差均为 `NPU.npu_incre_flash_attention.0.forward` 输出。

模型级为 `PASS_ADAPTED_MODEL_AND_STATISTICS`。真实 Qwen3-0.6B 的四组 B/F/F/OFF 共 112 次 decode IFA，56/56 次 F 均切换，覆盖全部 28 层和 8 请求，受控层/请求组合为 448。两次 B/F 的独立首差均在第 0 层原生 IFA 输出 Mean/Norm；首层完整 QKV 哈希相同，后续真实 `o_proj` 和全部请求 logits 出现变化。完整 logits 有 1,035,472 / 1,215,488 个元素变化，最大绝对差 0.1171875，Top-1 变化为 0/8。F/F 和 B/恢复的完整 logits 及逐层输出哈希一致。模型使用显式分页 decode 适配器，非自然服务调度；该结果不证明业务质量下降。

采集配置为 `task=statistics`、`level=mix`、`summary_mode=statistics`，使用已披露的任务私有 high FP32 统计归约采集器；它不改变模型张量和计算。原生 dump/stack/construct/config 按原字节保存，null 保留为不可观测。输入使用完整逻辑和物理 SHA 判等，分页映射另作逐元素反查；算子输出及模型私有 logits 按全部元素复读比较，模型中间层使用全量与逐请求 SHA。公开的是哈希和结果，不发布完整张量。

`operator_a4/INVENTORY.json` 可用以下命令独立验证全部原始公开文件的大小、SHA 和允许范围：

```sh
python3 -B experiments/fault33_paged_tiling_20261105/analysis/package_evidence.py \
  --verify artifacts/attention_tiling/fault33_paged_statistics_20261106/operator_a4
python3 -B experiments/fault33_paged_tiling_20261105/analysis/package_evidence.py \
  --verify artifacts/attention_tiling/fault33_paged_statistics_20261106/model_a4
```

报告单独位于 `reports/`，不改写原始 bundle 的清单。公共 serializer 解码的是 host 最终 raw 的 72 个字段；`kernel_argument_raw_observed=false`，没有把 host raw 宣称为设备入口参数。保留完整输出无差异、原生分支变化混杂和唯一 Reduce 指令归因不足的结果，不根据期望 profile 或命中日志指定 msprobe 首差。

公开文件足以独立重算统计首差，例如：

```sh
python3 -B experiments/fault33_paged_tiling_20261105/analysis/compare_model.py \
  --baseline artifacts/attention_tiling/fault33_paged_statistics_20261106/model_a4/model/model_baseline/observation/step0/rank0/dump.json \
  --fault artifacts/attention_tiling/fault33_paged_statistics_20261106/model_a4/model/model_fault_r0/observation/step0/rank0/dump.json \
  --output /tmp/fault33-model-bf.json
```

模型每对 trace 均有 2,510 个节点、178 个统计字段不可观测。公开包能复核完整中间层及逐请求哈希；logits 的逐元素差值由服务器私有原始输出复读产生，公开包不能单靠哈希重新计算这些数值，独立公开复核报告明确保留此界限。

卡 5 的 ATK 自动派发曾中断采集。中断 attempt 保留失败回执，其部分结果不参与正式验收；用户已授权的启动前精确 ATK 清理另有 `manual_launch` 回执。控制器的 `external_container_mutations=false` 只描述自身持锁执行窗口。未修改队列结果文件，未占用卡 2。
