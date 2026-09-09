# 正常输入用例库与真实双卡验证

## 目标与边界

这轮不修改 FakeRequant、Fault20 或模型计算算法。复用 Research 已验证的 R1：MLP_DOWN 一个模块族、全部 64 层、rank0、REFERENCE_REQUANT、tensor_absmax、scale_multiplier=0.25、max_hits=4096。全部参数从真实 E1 manifest 读取并校验 hash，不从此文档构造。

4096 是每次 arm 的上限，不是每条短请求必须耗尽的配额。本轮不发送补齐配额的额外请求；命中、变化元素和实际层覆盖只取测量请求自身的 query。rank1 必须无命中，并有 rank_filter 拒绝证据。分开记录 NOT_HIT、HIT_NO_NUMERICAL_CHANGE、MUTATION_CONFIRMED 和 MECHANISM_UNKNOWN。

## 一个用例数据源，两个独立版本概念

唯一维护源仍是 [固定用例 JSON](../../profiles/evaluation/distortion_probe_suite_v1.json)。文件名为兼容旧路径保留；当前内容版本以 JSON 内 `version` 为准。

- 原内容版本：DISTORTION_PROBE_SUITE_V1，hash `abfc3c9a26597e3eefc7204306553dd381353c45dfe5b3d135632bbd8856d502`。
- 当前协议版本：NORMAL_DIAGNOSTIC_SUITE_V2，hash `e5d853bfb947a4082b77392a55d0c5eb9435ce265f3eade94780ee536ac8bf86`。
- 元数据视图版本：NORMAL_DIAGNOSTIC_CASES_V1。`normal_cases.catalog()` 从同一个固定文件导出，不形成另一个可编辑的 prompt 数据源。

每个导出用例包含 case_id、category、语言、prompt、context、generation_config、正常性质、适用检测器、reference/validator、来源、revision、license 说明及内容 hash。全库也有 hash。16 条均是项目编写的工程 smoke 输入，不冒充外部 benchmark；不需要联网下载。

### 为什么升级协议

2026-09-09 的真实 Fault OFF 校准发现：裸 `/v1/completions` 输入会把模型置于继续补全文本的形态。例如 gold 问题答出 Au 后续写下一道题；yes/no 问题自然出现重复。这不是故障证据。

在任何故障测试之前保留 V1 部分校准记录，最小修订为 `/v1/chat/completions`、单条 user message、`enable_thinking=false`。一条独立 clean 检查返回 `Au`、finish_reason=stop。16 条 prompt、参考答案、温度、seed、max_tokens、检测阈值全部不变。单元测试把 V2 去掉协议元数据后恢复为原 V1 hash，防止夹带调参。

模型只接收 prompt 的 user message、固定生成参数和对话模板选项。答案、检测阈值、故障参数不放进消息。

## 校准与误报边界

Fault OFF 串行运行两遍全部 16 条用例。每个 case × detector 分别保留 ELIGIBLE、NOT_APPLICABLE、BASELINE_FAILED、BASELINE_UNSTABLE。要求正常性质稳定，不要求自由文本逐字一致。失败和不适用项不删除、不缩小总样本分母。正式周期开始前把校准 hash 固定，不能事后筛选有利样本。

离线 fixture 与 live 证据分离。正常 JSON 的标点和重复字段不是刷屏/循环；合法列表不因格式重复而报错；简短但完整的答案不是提前终止。无独立不完整证据的长度缩短只要求复核，不能只凭比例称为 early stop。措辞不同不代表病态，未触发规则也不代表语义正确；开放文本标为 SEMANTIC_REVIEW_REQUIRED。

## 真实执行与恢复

1. 在现有 NPU_LOCK 内核对空闲物理卡 2、5，18031 端口、模型、rootfs、runtime hash 和磁盘。
2. 调用历史已验证 summary launcher；不创建新模型启动参数，不打开逐算子日志。
3. 两轮 clean 校准、冻结，再做 disabled-hook 与 IDENTITY 无损控制。
4. 三个完整独立 cycle：clean → baseline 全集 → install → 各请求 arm/request/query/finally-disarm → remove → recovery 全集 → health/clean check。
5. 每周期落盘完整 raw 请求、原始 chat response、实际机制 query。任何失败保留原 attempt，不自动覆盖重试。
6. 原请求的 token IDs 不存在时保持 null；token pieces 单独标识。
7. 同一批 `live_raw.json` 再载入同一 evaluator，在线与离线结果需精确一致。
8. 最终停止本任务实验服务，使用原无扩展 launcher 恢复服务；确认健康后释放锁。重启恢复与周期内同进程恢复分别报告。

监督脚本有 12 小时总期限、3 小时校准期限。继承 12 GiB 空闲磁盘下限，新增本次 session 4 GiB 产物限额、20000 客户端事件上限。native metadata/buffer 日志仍为 `/dev/null`；不删除任何历史日志。

## 三项独立结论

| 结论 | 所需证据 |
| --- | --- |
| 工具在线验证 | 完整采集、机制校验、分类、汇总、恢复、可追溯及 replay 一致 |
| R1 效果 | 如实报告有无明确可见退化；文本分叉/PPL/轻微格式差异不单独算明显扭曲 |
| 可交付复现用例 | 同一正常输入、同一配置，多轮同一异常，有请求自身机制证据且解除后恢复 |

旧 Robust Visible Distortion 的“两种 phenotype / 广度”门禁保留为独立指标，不强迫本轮 R1 产生两种病态。

## 实现入口

- [正常用例导出/校准](../../src/ascend_fi/evaluation/distortion/normal_cases.py)
- [既有 RPC 客户端适配与实测](../../tools/run_normal_diagnostic_live.py)
- [既有 launcher 会话包装](../../tools/distortion_service_session.py)
- [串行监督及最终恢复](../../tools/supervise_normal_diagnostic.py)
- [离线 fixture](../../tests/evaluation/normal_detector_fixtures.json)

这些入口不实现新的 fault controller。RPC 仍委托已有 E1 client，故障仍由原 FakeRequant extension 实现。源文件 hash、模型/tokenizer hash、PID、CANN/torch/vLLM 版本与实际 device/rank 映射另存 service_startup_audit/startup_details；结果未完成时不得提前标为 PASS。
