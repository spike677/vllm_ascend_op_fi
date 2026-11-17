# Agent 7：FA a5 完整序列与门禁范围复核

**a5 的固定单算子 B/B-repeat/F/OFF 序列通过独立复核：真实策略改变、完整输出发生变化、同进程 OFF 精确恢复。尚未授予 stock bridge、自然模型、标准 Worker、E5、C2 或 C3。** 本次没有 `statistics/mix` 采集，不能据此宣称 msprobe 首差、0.1% 统计幅度或历史 Batch→M→causal Reduce 链路通过。

公开摘要：[agent7_fa_a5_sequence_summary.json](agent7_fa_a5_sequence_summary.json)。审计程序：[review_fa_a5_sequence_124.py](review_fa_a5_sequence_124.py)。原始记录和全部张量只留在 124；审计没有启动容器或执行 NPU 调用，只读文件、只读镜像元数据，并执行已经核验身份的 CPU POD decoder。

| 检查 | 独立结果 |
| --- | --- |
| 固定调用 | `torch_npu.npu_fusion_attention_v3`，B8、FP16、2048，构造 Q/K/V；单算子，不是完整模型 |
| 实际执行 | 四次 `aclrtLaunchKernelWithHostArgs`，每次一个匹配 kernel、两次成功同步；共 4 launch / 8 sync |
| 内核绑定 | 实际注册 ELF SHA256 `967265862973a761ae7de2c7b54600e246f067898cd2c97f297aa9a12e784eb0`，function binding 和实际 key `22619644976` 对齐 |
| 最终参数 | 从 1720-byte 实际参数中，按已编译 FA ABI 的 pointer offset 200、data offset 216 取出 1464-byte POD；四次均与已保存 raw 完全一致 |
| 公共 POD | Python 解码和原构建 C++ decoder 的 205 个字段逐项一致；原 host 投影/完整 raw 也一致；OFF 不依靠 host 命中日志 |
| 策略 | B/B-repeat/OFF：M128、nRatio8；F：M64、nRatio4；S2 base 始终128，因此核内分组1024→512；blockDim16、scheduleMode0 |
| 完整输出 | 四次 × 6 个 tensor leaf，共24文件、335,544,384 bytes；全部重算哈希、检查有限值并逐字节比较 |
| 恢复 | B-repeat 与 B 的完整 tuple 全部字节一致；无计划、无 host 日志的 OFF 同样全部字节一致 |

F 的 `output-0` 共 1,112 个元素变化，最大绝对差 `7.62939453125e-06`；`output-2` 共 519,936 个元素变化，最大绝对差 `0.000244140625`。其余四个 leaf 不变，全部输出有限。这是完整张量离线核验的辅助结果，**不是 msprobe statistics 比较结果**。本次只有一次 F，不授予重复 F 的稳定性，也不将 joint 策略归因为 M-only。

固定输入规则的四个完整哈希已用不依赖 torch/NPU 的 CPU 实现独立重算，匹配 `inputs.json`。冻结 probe 在每次调用后读取完整设备输入并与初始哈希比较，四次均记录成功；没有另存每次 post-call 输入字节，因此不能把本次离线审计说成重新读取了这些设备输入。原 API 返回值直接交给完整 payload 读取器；未修改结果。

851 个 vendor 产物、source-a33 的609个文件、source-a34 的618个文件、镜像5个SDK头及 observer DSO 均通过身份检查。实际容器命令证明 `/driver` 挂载 **source-a33**，它是已经执行的 probe 与 reader；source-a34 是主控声明使用的门禁快照。controller 自身与 queue contract 的哈希在两份快照中相同，原回执 `frozen_scripts/source_manifests` 数组为空，所以本审计不凭共享脚本哈希补造唯一的启动时快照身份。原始源码、命令与后续复核的边界分别保留。

原 a2 的 decoder envelope 失败、a3 的 `aclInit Resource_Busy` 和 a4 的进程阻断没有被 a5 的成功追溯改写。新构建的 stock 原包无损桥接尚未完成；本次不沿用旧构建的任何资格。

## 保留 checkpoint 的门禁审查

对最初 `947375d4` 的只读审查确认：原始门禁回执 SHA256 `048704bfd59cf5a18728b257e4617f4aabdb5dcacd4d6fd1df60d8d60d26640a` 未变，公开 target 摘要是原件的精确子集，两个旧 checkpoint blocker 完整保留，canonical proof SHA256 `c501de94b2621a7348cd58c67c83f1deb71c021c77626b73545a8c0b206aa9c7` 可重算，OCI 清单摘要和最终门禁相符。

`NO_LIVE_OWNER_CORROBORATED` 只表示该观测窗口内各来源没有发现当前 owner。它不证明 kubelet 内部 allocation/pending-admission 已清空，不证明驱动能够初始化，也不建立设备独占。代码和回执保留这三项 `false`。门禁回执存有 API/RPC 的计数、scope 摘要及字节哈希，没有包含可供本审计重新解析的完整 API/RPC 响应；这里独立复核的是代码、回执关联和哈希，未声称离线重放了每一份原始 API/RPC 响应。Kubernetes 的内部设备状态还受 source readiness 和 pending admission 影响，见 [固定版本设备管理实现](https://github.com/kubernetes/kubernetes/blob/v1.28.15/pkg/kubelet/cm/devicemanager/manager.go)。

审计发现 `947375d4` 的默认 `review_stale=True` 未绑定指定 UID/checkpoint/attempt，范围大于文档中的单次授权。已反馈并由 `d7e2cda8`（root集成 `8339b2a2`）与 queue `e2a94f31` 修正：默认拒绝，显式 opt-in 必须绑定准确 Pod UID、checkpoint SHA256、fresh attempt 绝对路径；scope 文件保留，解析拒绝重复字段和过大/不规范文件，换 owner/hash/attempt 或未知状态继续阻断。

a5 是修正前 source-a34 的历史执行。本审计核实它前后门禁实际只处理已批准的卡5目标 UID `33991897-e3ed-4910-aa2c-835b9fe682a0`、checkpoint `b570d0ea0d0280157a26d48c536a6aacff8cdf4fb55037f3b3882e85c1395414`，未扩展 owner。不能把未来的 scope 修正写成 a5 当时已经执行的代码。

另只读核验后续 `fa28-statistics-vision17-census-a1/controller.json`：SHA256 `5f387a6ec796e91e3e3ccc4057fbbd6980bbc590fb65ba5d94848d68ec7a8f63`，状态 `DEVICE_BUSY`、`idle=false`，新 allocation 以 `STALE_ALLOCATION_CHECKPOINT_SCOPE_MISMATCH` 保留阻断，尚无 command/start 记录。没有借旧 UID 授权移出新 owner。此处不公开无关 Pod、OCI 或 Job 内容。

原始门禁44项CPU合同通过；收紧后的门禁与queue共61项CPU合同也在本审计方通过。新增8项本审计回归通过，包含scope错误、摘要篡改、缺少tuple leaf、形状变化、NaN和正负零不能冒充字节精确恢复。测试集合存在扩展/重叠，不相加为独立机制验收数。

## 身份与交付

| 原件/独立产物 | SHA256 |
| --- | --- |
| a5 controller | `8302e00c12f98a5b37040163179fc196ef36c89af7a75f2442b5af8eb3a4437e` |
| a5 result | `e919244c2a615ec71413fe859fd28a4c26ecd929941a1e0a1191bb292832eea6` |
| 独立摘要 | `b53fcc0a498096c96f3946b730bbd2ff130789bae81fcc0ea0d37411cfeadb03` |
| 独立审计脚本实际执行版本 | `e760107a08607ba1313448ae13b6bd3f01bb415ab18e3aeb0c24759223f82060` |
| 原构建 completion | `7189c077dad9411e53240d6a0cc9190477d00433d5b1325833c0fcb26c7cac5d` |

远端独立产物：`/home/17-fault-c2c3-20261117/agent7-fa-a5-review-a1/summary.json`。本目录公开 JSON 与其逐字节一致。保留完整 payload 只用于本次辅助核验；后续 E5 必须另采原始 B/F `statistics/mix` 并按固定容差独立定位，不能由这份辅助报告替代。
