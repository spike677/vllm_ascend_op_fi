# 私有格式 as_strided：CANN 9.1 真实推理 msprobe 补采

2026-10-26 在 124 服务器完成真实 Qwen3-0.6B 推理补采。原生 msprobe 的 statistics/mix JSON 结构与已提交历史数据兼容，张量原始 dtype 标签集合一致。独立 native 观测确认了 **私有格式权重 → as_strided 返回视图 → matmul 读取**；本次仍未建立该视图或其派生视图的原地写回链，故维持**历史/条件候选，暂不实现注入**。

## 环境和实际执行范围

| 项目 | 实测记录 |
|---|---|
| 设备 | 124，Ascend 910B4，独占当时空闲的物理 NPU 2，进程内 npu:0 |
| CANN | 9.1.0；安装根 /usr/local/Ascend/cann-9.1.0；内层版本 V100R001C11SPC001B243 |
| 软件 | torch 2.10.0；torch_npu 2.10.0.post4；vLLM 0.23.0+empty；vLLM-Ascend 0.23.0 |
| 采集器 | mindstudio-probe 26.1.0，导入模块 msprobe |
| 模型 | 真实预训练 Qwen3-0.6B，BF16、TP1、eager、max_model_len=256 |
| 触发条件 | allow_internal_format=True；**显式设置非默认 weight_nz_mode=2**；不能推及所有默认模型配置 |
| 对照 | 两个独立完整模型进程，msprobe OFF / ON；两组均保留同一 native 身份观测器；未注入故障 |
| 请求 | “The capital of France is”，每组 cold、warm 各一次，每次生成 8 token |
| msprobe 范围 | 仅 ON 组 cold 请求前三次真实 AscendModelRunner.execute_model，step0–2/rank0，调度 token 数分别为 5、1、1；forward step 不等于生成 token 序号 |

模型执行完成、测试容器退出并释放设备的时间为北京时间 17:28:04，早于本轮 17:49 截止时间。73 次资源采样未发现测试期间物理卡 2 被其他容器占用；结束快照显示该卡空闲，原有五个服务的容器 ID、镜像和创建时间保持一致且仍为运行状态。这是资源与状态证据，不是其他业务延迟或性能测量。

## 与历史已提交数据的对照

历史基准固定为提交 `9bc4ad238b2619ae1bb6c7e7762fa42c369602b4` 中的 [Fault22 statistics 交付](../../fault22/wrong_row_output_proxy_steps012_20260929/README.md)。该历史故障仅用于对照数据格式，不能据此关联故障机制。

| 检查项 | 本次结果与边界 |
|---|---|
| 原生文件 | 9 份 JSON：3 step × 1 rank × dump/stack/construct；历史各组为 3 step × 2 rank，共 18 份 |
| 采集形式 | 两者均为 statistics、mix；Tensor 字段包含 type、dtype、shape、Max、Min、Mean、Norm、requires_grad；本次无需新增 schema 字段 |
| dtype 标签 | 两者均出现 BF16、FP32、INT8、INT32、INT64、BOOL；本次记录数分别为 7962、423、84、248、453、6，数量和形状无需与历史相同 |
| 数据含义 | 原生统计 JSON，没有全量 .pt/.npy Tensor；dtype 是被观测张量的原始类型，不是 JSON 数字的存储类型 |
| 工具版本 | 本次为 26.1.0；所选历史包未记录工具版本，因此不能声明采集器版本一致 |
| 原始数值 | 每步 112 个目标线性层的 BF16 权重保留 dtype/shape，但 Max/Min/Mean/Norm 全为 **null**，不能当成零或权重数值相同 |
| 实验关系 | 本次是健康推理的 msprobe OFF/ON 对照；历史是 TP2 的 B/F 故障对照，本次不构成 B/F 注入实验 |

详见 [格式核查](audit_msprobe_format_attempt2.json) 和 [算子核查](audit_msprobe_operator_review_attempt2.json)。原生 JSON 按字节保留，未将自定义 JSONL 转换或伪装成 msprobe。

## 真实调用链与最后一环

每个采集 step 有 113 个 Functional.linear：112 个来自 `vllm_ascend/ops/linear.py:58`（经 apply:106），另一个为 LM head。msprobe 的 dump/stack 能确认真实模型调用及 dtype/shape，但没有 NPU 私有格式、TensorImpl 或 Storage 身份字段，也没有显式 as_strided dump 条目；这些需要独立 native 证据。

native 按真实推理 scope、原 F.linear 权重身份、Storage 和几何信息绑定事件。OFF、ON 各确认 **1792** 条精确链：112 个权重、cold 896 次、warm 896 次。链上输入和返回视图的实际格式均为 29（FRACTAL_NZ），共享原权重 Storage；返回视图作为 `aten::matmul` 的 `other` 参数被读取，matmul 输出使用新 Storage。ON 的前三个 msprobe scope 只覆盖其中 **336 条**（每步 112），不能将全部 1792 条算入 msprobe 采样。

代表事件：ON `native_on/native_pid750.jsonl:616–622`；OFF `native_off/native_pid164.jsonl:387–393`。一个 ON 样例为 BF16 权重 `[4096,1024]`、stride `[1024,1]`、offset 0，返回视图 `[1024,4096]`、stride `[1,1024]`、offset 0。具体身份、消费者与源文件 SHA256 见 [ON native 核查](audit_native_on_attempt2.json)、[OFF native 核查](audit_native_off_attempt2.json) 及 [采样范围对齐](audit_attempt2_token_and_capture_scope.json)。

已记录的精确链上，直接或容器 schema 写目标为 0，存活派生别名边为 0，原 linear 区间外的已跟踪写目标也为 0。解析、配对和相关身份核查未发现错误或截断。但是 native 回调仍包含没有 schema 的事件，隐藏设备写入和数值写回正确性未被证明；**未观察到写回不等于整个框架不存在写回**。本次 FRACTAL_NZ 也不能替代历史 NC1HWC0/NDHWC/NDC1HWC0/Z_3D 的格式覆盖。

OFF/ON 两个请求的全部生成 token 和文本一致，均生成 ` Paris. The capital of France is also`。这只支持记录范围内的输出一致，不证明完整 logits、中间张量或无观测器基线等价。原始缺陷是否仍存在、目标链是否具有可注入写回点，仍待证。

## 文件、保存与校验

- `msprobe_statistics_steps012_rank0.zip`：9 份未经改写的原生 msprobe JSON。
- `native_identity_logs.zip`：第二轮 OFF/ON 原生身份日志、对应 worker 控制记录和摘要、模型结果，共 8 个文件；与 msprobe 分开保存。
- `msprobe_config.json`、`capture_bridge.json`、`capture_source/`：实际采集配置、scope 记录及桥接实现。调用真实 PrecisionDebugger.start(model=runner.model, rank_id=0)、stop、step；不替换模型计算。
- `audit_*.json`：格式、调用点、native 身份、输出和资源核查。审查记录中的绝对路径是原始证据定位信息。
- `environment_version_evidence.json`：本次环境采集记录中的 CANN 安装文件内容、哈希及格式枚举节选，明确标注派生来源；模型结果另记运行时软件版本。
- `manifest.json`：本发布目录文件及两个 ZIP 的逐成员 SHA256；自身不列入自身校验。
- `preservation/`：完整原始包收据、244 文件清单和本地校验结果；第一轮采集缺口单独保留。

第一轮 msprobe 采集成功，但 native 日志目录缺失导致身份原始日志未落盘，不能据此作“没有写回”的负面判断。修复独立采集目录后，第二轮重新运行两个完整模型进程，补齐身份日志；本目录主数据均来自第二轮，第一轮未删除或覆盖。

两轮全部原始记录保存在本地 `C:\Users\admin\Documents\demoncatpfi\outputs\as_strided_msprobe_20261026\delivery_attempt2\raw_evidence.zip` 与服务器 `/home/as-strided-msprobe-20261026/delivery_attempt2/raw_evidence.zip`，244 个文件、压缩后 11,887,198 bytes。原包 SHA256 为 `738a7334890127185a42f1ce8fa80b2a5ce5a2cc6463cee579cc26627c63e4d0`，ZIP CRC 及逐文件 SHA256 已通过校验。Git 交付选取本故障相关统计、身份与审查证据，不包含模型权重、运行库或完整宿主机/业务服务日志。
