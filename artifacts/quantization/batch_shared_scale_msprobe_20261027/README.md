# Batch 共享量化尺度故障：msprobe 实机数据

本数据集采集了 **DeepSeek-R1-Distill-Qwen-32B-W8A8、TP2** 上的两个真实实验条件：`letter_counts_B2` 和 `even_count_B4`。使用 msprobe 26.1.0 的 `PrecisionDebugger.save(..., save_backward=False)`，原生格式为 **`debug.json` + `.debug.pt`**，属于 `tensor/debug` 采集，并非全模型自动 API dump。

共 **20 个 rank/session、140 份张量文件、20 份 `debug.json`**。张量已与独立保存的原始调用数据逐字节核对；健康、故障重复和恢复检查均通过。

## 下载与数据组成

按实验阶段分为五个 ZIP；每包包含 **2 个条件 × 2 个 rank = 4 个 session**，共 **28 份 `.debug.pt`、4 份 `debug.json`**，以及对应的 session 配置和映射记录。

| 数据包 | 阶段含义 |
|---|---|
| [H.zip](data/H.zip) | 启用 msprobe 的健康采集 |
| [H1.zip](data/H1.zip) | 健康重复 |
| [F.zip](data/F.zip) | 目标行共享尺度故障 |
| [F2.zip](data/F2.zip) | 同一故障重复 |
| [R.zip](data/R.zip) | 关闭故障后的恢复 |

每个 session 的 7 份张量依次覆盖：量化输入 `input_x`、原生 `native_q`/`native_scale`、实际送入消费者的 `returned_q`/`returned_scale`、原消费者输出和下游 logits。`debug.json` 提供张量名称、dtype、shape 和数据文件引用；session 记录提供条件、阶段、rank、真实请求行和 token 位置。读取时按包内文件定位引用，保留张量原始 dtype。

## 采集与故障边界

目标层为 `model.layers.31.mlp.down_proj`，仅绑定该层第一个“所有指定请求实际在场”的 **pure decode** 调用。请求与输入行的对应关系来自真实 scheduler 元数据，不根据 batch 顺序猜测。运行使用 eager、seed=42、temperature=0；目标请求最多生成 1024 个 token，配对请求最多 3 个。

F/F2 在真实 W8A8 消费链上受控替换目标行的配对 Q/scale；原生量化器仍实际执行，其他请求的 Q/scale 保持原样。共享 absmax 使用同次调用全部有效行的真实激活，范围限于 **每个 TP rank 的本地特征分片**；未人为放大激活，也未增加跨 rank 最大值归约。consumer 输出是局部 TP 贡献，logits 已经过后续模型计算。

独立进程的 B 对照不导入 msprobe，但保留原始调用观察。**B/H 的输入约束、9 份原始 dtype 的旁证张量，以及完整 token IDs、文本、结束原因和 top-5 logprobs 均一致**。H/H1、F/F2、H/R 也一致。此结论限于实测数据；msprobe 存在进程级包装行为，不能据此声称没有任何全局副作用。B 对照不计入上述 140 份 msprobe 张量。

## 实测结果

G2 表示目标量化 code 改变且重建误差增加；G3 表示与该故障对应的目标下游 logits 变化可重复，并通过恢复对照。

| 条件 | rank | 目标 scale 比值 | 改变的目标 code 数 | logits 最大绝对差 | G2 / G3 |
|---|---:|---:|---:|---:|---|
| letter_counts_B2 | 0 | 1.0822785 | 2683 | 0.625 | 通过 / 通过 |
| letter_counts_B2 | 1 | 1.0656935 | 2411 | 0.625 | 通过 / 通过 |
| even_count_B4 | 0 | 1 | 0 | 0.8359375 | 未成立 / 未成立 |
| even_count_B4 | 1 | 1.0729928 | 2538 | 0.8359375 | 通过 / 通过 |

`even_count_B4` 的 rank 0 保留原生 Q/scale，其 logits 仍通过 TP 后续计算受到另一 rank 的影响，不能算作该 rank 的局部量化损伤。

两个条件的完整生成序列分别从第 32、121 个输出 token 开始分叉（从 0 计），但 **健康与故障运行的最终答案仍正确，且最终答案区相同**。因此本轮证实了可重复的中间数值变化和下游传播，未证实最终答案错误；格式是否严格合规与数值正确性分别判定。

本实验是受控的 Q/scale 配对替换，**不构成原生量化器存在跨序列尺度污染缺陷的证据**。采集与绑定证据止于实际 torch_npu API，不额外宣称已追踪 ACLNN 内核符号。

## 默认状态

实验默认 **DISARMED**。故障需要显式启用参数 `--allow-fault` 和环境变量 `BATCH_OUTLIER_ENABLED=1`，并通过 B/H、H/H1、实际输入绑定及两 rank 目标校准门禁。复现必须使用新输出目录，保留原始采集文件。


## 审计支持与离线校验

`audit_support.zip` 保存独立 B 与采集运行的原始 result/control、逐代控制记录、调用观察 JSONL 和旁证张量，以及 9 份冻结模型 Python 源、必要 legacy Python 源和 CPU 校验器。该包按白名单收录，不包含控制器、服务清单、环境脚本或主机运维日志。`prior_result.json` 是原字节保留的前序矩阵结果，只提供这两个条件的选择背景；不能计入本轮 140 份 msprobe 张量。

五个阶段包保留 182 个不同原生文件：140 份张量、20 份 debug.json、20 份 session manifest 和 2 份 rank 配置。每个阶段包重复包含相同字节的两份 rank 配置。所有包内路径均以同一采集根目录为基准，必须解压到同一目录。原始记录中的 `/audit/...`、`/opt/...` 等容器路径保持原样；CPU 校验器按已记录的目录边界重定位，无需访问原机器。

`capture_index.csv` 仅移除了四个绝对路径列和 PID，保留包内相对路径与其他字段；`paired_metrics.csv` 保持原字节。`verification_summary.json` 是不带本机路径的范围、计数、比较结果和透明性状态摘要。`SHA256.json` 列出所有发布文件、ZIP 成员及其原始证据 SHA256；它自身不作递归自哈希。原始证据成员的哈希与已验证归档清单逐一匹配。

先运行仅依赖 Python 标准库的完整性检查：

```sh
python verify_publication.py --root .
```

然后将六个包解压到一个新的共同目录（放在发布目录外，避免与发布文件清单混淆）：

```sh
python -m zipfile -e data/H.zip ../batch_shared_scale_msprobe_unpacked
python -m zipfile -e data/H1.zip ../batch_shared_scale_msprobe_unpacked
python -m zipfile -e data/F.zip ../batch_shared_scale_msprobe_unpacked
python -m zipfile -e data/F2.zip ../batch_shared_scale_msprobe_unpacked
python -m zipfile -e data/R.zip ../batch_shared_scale_msprobe_unpacked
python -m zipfile -e audit_support.zip ../batch_shared_scale_msprobe_unpacked
```

以下校验需要 Python 和 NumPy，可在 CPU 上执行；不导入 torch、torch_npu、msprobe，也不加载模型。输出文件必须不存在：

```sh
python ../batch_shared_scale_msprobe_unpacked/verify_msprobe.py --on-run ../batch_shared_scale_msprobe_unpacked/evidence/model_capture01 --off-run ../batch_shared_scale_msprobe_unpacked/evidence/model_B01 --source-dir ../batch_shared_scale_msprobe_unpacked/source_versions/model_B01/model --output ../batch_shared_scale_msprobe_cpu_review.json
```
