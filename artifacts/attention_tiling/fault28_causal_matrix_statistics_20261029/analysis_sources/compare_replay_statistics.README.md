# 单调用 replay 的独立 msprobe 统计比较

`compare_replay_statistics.py` 只从原生 `config.json`、`dump.json`、`stack.json`、`construct.json` 确定首个可观测的 API 统计差异。它不读取 host/profile/hit 日志、私有 tensor 或完整目标输出，不以策略标签指定故障点。

当前冻结的 `replay.py` 配置为 `task=statistics`、`level=mix`、`summary_mode=statistics`、`tensor_list=[]`、`data_mode=["all"]`、`rank=[]`、`step=[0]`、同步采集。`PrecisionDebugger.start()` 位于实际 `ReplayModule` 调用前，`stop()/step()` 位于调用完成后、输入/输出读取与 FP64 比较前。收据要求一个原生支持的 `NPU.npu_fusion_attention_v3.*.forward` 节点及 replay module；它不代表 CANN 内核覆盖证明。

旧 `analysis/fault28_tiling_check_20261029/analyze_msprobe_native.py` 的完整性门槛面向多层 HF SDPA，不适用于这里的一次原生 API replay。本工具没有修改该工具、现有因果分析器、冻结 runtime 或 setup。

## 输入与完整性门槛

每个 case 目录需要：

```text
config.json
receipt.json
collector_provenance.json
result.json
observation/step0/rank0/dump.json
observation/step0/rank0/stack.json
observation/step0/rank0/construct.json
```

另传该 case 所属 attempt 的 `completion.json`。completion 必须已完成、case 返回码为 0、结果 SHA 对应且自有容器已清理。工具只从 `result.json` 使用 `status`、`case_id`、`receipt_sha256`，其余字段即使与完整性元数据在同一文件内，也不参与定位。receipt 绑定 config、collector provenance 和原生三件套的 SHA/bytes。此检查不替代独立资源审计。

high 模式要求公开披露 `task_local_fp32_high_v1`，实际 collector 原 SHA 为 `e981aa6f…`、修改后 SHA 为 `3aa7ede0…`、stage manifest 为 `743354fe…`，并核对实际 module/code 路径。它明确是修改版 collector，不能称未修改官方实现。本工具核对收据中的已知指纹与实际加载声明，不重新读取整个 collector stage；完整 stage 文件核验由采集流程及独立部署证据承担。也支持两边均使用已审官方 `official_low` 的比较。B/F collector 或配置不同会抑制全局首差结论。

统计目录内出现 `.bin/.pt/.pth/.npy/.npz/.safetensors` 或符号链接会被拒绝。完整输出只留在独立私有比较目录，不进入统计包或 GitHub；工具不会读取这些文件。

## 调用

在实际采集完成之后，对**相同 batch 大小**的一对 case 运行，例如：

```sh
python analysis/compare_replay_statistics.py \
  --baseline outputs/primary_part1/base_b8_r0 \
  --fault outputs/primary_part2/m_only_b8_r0 \
  --baseline-completion outputs/primary_part1/completion.json \
  --fault-completion outputs/primary_part2/completion.json \
  --output outputs/statistics_reports/base_vs_m_b8_r0.json
```

上述 attempt 名是调用示例，应替换为实际路径。输入目录均只读，输出报告必须是新文件。重复 0/1 应分别比较；本工具不汇总或挑选更显著的一次。

```sh
python analysis/test_compare_replay_statistics.py
```

CPU fixture 报告写入本目录 `replay_statistics_cpu_checks.json`。fixture 不是 NPU 或故障复现证据；测试不启动容器，不导入计算框架。

## 如何解释结果

对齐依据包括 API 名、完整原生调用栈（含路径、行号和源代码文本）、construct 祖先以及 dump 顺序。缺失、重排、不兼容元数据、不同配置或不同采集器都会得到 `INCONCLUSIVE_ALIGNMENT_OR_METADATA`，不输出全局首差。工具不静默排除诊断节点；如果 `snapshot/raw_bytes/reference_queries` 出现在采集栈中，则标为采集范围异常。

`first_observed_api_difference` 是首个**可观测统计差异 API 记录**，`first_difference_boundary` 区分其输入或输出。如果更早的其他 API 有差异，就报告那个 API，不会强制定位到 FA。module 输出属于完成后的汇总记录，另行报告，不混入 API 首差顺序。

返回值 0 是 Attention score；返回值 1–5 分别为 softmax_max、softmax_sum、softmax_out、seed、offset。报告分别给出 `attention_score_output_observed_difference` 和 `auxiliary_output_observed_differences`。只有辅助返回变化时，不能把它称为主 Attention 输出故障。

- `ALIGNED_OBSERVED_API_STATISTICAL_DIFFERENCE`：对齐后发现 API 输入或输出统计不同。
- `ALIGNED_AVAILABLE_STATISTICS_MATCH`：可用有限统计一致；不证明张量逐元素相等。
- `ALIGNED_NO_OBSERVED_API_DIFFERENCE_WITH_UNOBSERVABLE_STATISTICS`：没有观测到 API 差异，存在 null、缺失或相同非有限统计。
- `ALIGNED_MODULE_DIFFERENCE_WITHOUT_API_DIFFERENCE`：只在 module 汇总记录中看到差异，无法给出 API 首差。
- `INCONCLUSIVE_ALIGNMENT_OR_METADATA`：比较边界或配置不兼容，不能形成首差结论。
- `INVALID_CAPTURE_EVIDENCE`：采集完成性、格式、披露或文件绑定检查失败。

`null` 与 `null`、`null` 与有限数都记作不可观测，不转换为零、不认为数值相等。相同 Max/Min/Mean/Norm 也可能掩盖真实张量差异。输入统计一致不足以证明 Q/K/V/mask 字节一致；API 输出首差不足以证明物理故障点、具体内核指令或 tiling 根因。

B1/B8 的整批 shape 不同，不能用整张量统计直接定位同一目标请求。需要目标切片和全序列因果分析时，使用另外的私有输出分析流程，不能把它的结果倒灌为本工具的独立 msprobe 推断。

退出码：`0` 表示对齐的统计分析完成（包括没有差异或不可观测），`2` 表示对齐/元数据不确定，`1` 表示输入证据无效。任何退出码都不代表故障机制验收通过，报告始终保留 `mechanism_accepted=false`。
