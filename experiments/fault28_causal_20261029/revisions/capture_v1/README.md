# 故障 28：固定 Q/K/V 的原生 FA 因果验证

当前状态：**CPU 准备完成，新增 NPU 实验尚未执行**。2026-10-29 的启动尝试在资源检查阶段退出：124 卡 5 上的 `gyl_test` / 8002 推理服务仍占用设备。用户要求保留服务、等待卡 5 释放；本实验不会终止该服务，也不会抢占设备。

本线程已设置每 10 分钟复查，设备占用未变时保持安静；只有交接清单就绪且空闲门禁通过后才继续执行。自动化编号为 `28-5`，完整验证结束后停用。

本目录承接此前已发布的特定 FA 路径实验，补查 M 与 nRatio 分别产生什么作用。既有模型传播结果不作为本次因果矩阵的运行结果。本次不宣称复现历史 FIA 默认策略或历史事故。

## 固定输入与实验矩阵

从此前真实模型 layer1 捕获中冻结 batch0 的 Q/K/V。B1 使用该序列，B8 使用八份逐字节等值副本；保留原 V 的 stride、FP16/BNSD、GQA、mask、scale 和精度属性。每次调用前后检查完整输入 SHA、对象/存储身份、版本及几何信息，排除上游 batch 输入差异。

| 阶段 | 组合 | 调用数 | 判定用途 |
| --- | --- | ---: | --- |
| 原生桥接 | stock B1/B8，各重复两次 | 4 | B8 目标输出绑定此前真实捕获；B1/B8 差异据实记录 |
| 主因果矩阵 | B1/B8 × base(128,8)、m_only(64,8)、r_only(128,4)、joint(64,4)，各重复两次 | 16 | 分离 M、nRatio、batch 及交互效应 |
| 次级受控 batch 策略 | B1/B8 × batch_m、batch_r、batch_joint，各重复两次 | 12 | host 根据实际 batch 选择合法候选策略，验证受控 Batch→tiling 链 |

次级策略的 B1 均使用 base；B8 分别使用 M-only、ratio-only、joint。它们属于新构造的策略选择，不能被描述成 SDK 自带的历史错误。

主端点是完整目标输出 `[16,2048,128]`，覆盖所有 query、head 和 channel。CPU 用精确 FP16 整数单位分析主效应与交互，不只检查最后一个 query。预先固定十个 query 的 FP64 参考，参考误差不要求必须增大。六个 native 返回分别保存元数据、SHA 与统计；辅助返回变化不自动等于主输出变化。

## 实现与运行门槛

- `source_policy/`：匹配 CANN 9.1.0 的官方 host 源，固定 commit `c77c0829f1f0f0b8f882db1d9b051a758b605472`。只改变 M/nRatio selector，后续 core、workspace、BMM 和 Softmax tiling 仍由真实 host 生成；kernel 源不改动。
- `vendor_exec.py`：导入框架前校验并加载本任务独立 vendor 包；同一二进制运行所有 profile。
- `replay.py`：调用公开的 `torch_npu.npu_fusion_attention_v3`，每个进程只执行一次目标 native 调用。原 layer1 只是输入来源，不包装成正在执行模型 forward。
- `controller.py`：本计划固定传入 `--card 5`；保留资源锁、启动前/调用前后空闲检查、日志和时间上限，以及仅清理本任务容器的约束。每批最多八个 case。
- `analysis/`：依据实际原生日志、完整 raw tiling 与公共 POD 独立核验执行策略，再检查完整输出的因果对比。profile 名称及注入命中日志不能替代执行证据。

`analysis/verify_base_bridge.py` 单独检查 stock4 与 private base4，用于继续 cross profile 前的中途门槛；`analysis/index_attempts.py` 只创建跨批只读消费者使用的符号链接索引，保留原始回执和来源 SHA。符号链接本身不改变目标文件权限，分析流程仍必须只读。

先运行 4 次 stock bridge，再运行 4 次 private base。private base 必须与**相同 batch** 的 stock 完整目标输出和实际 raw tiling 字节一致。该门槛通过后才继续 M-only / ratio-only / joint；cross profile 的 CPU 编译与静态资源检查通过，不等于真实 GetTiling 或 kernel 已通过。最后执行预注册的次级策略。

源码派生的 computed masked 范围、合法 key 分组和动态 Vec/Softmax shape 可以支持机理分析，但不等于硬件浮点指令轨迹。联合变化不能独立证明 M 为唯一根因。若没有输出差异，应报告该输入下未观察到效应；若重复不稳定，报告无法确定归因。

## 采集与证据

两侧使用 `task=statistics`、`level=mix`、`summary_mode=statistics`。保留原生 `dump.json`、`stack.json`、`construct.json`、配置与完成回执。沿用此前披露的任务内 FP32 高精度统计 collector；它只改变统计计算，不改变模型/算子输入输出。默认半精度统计曾漏掉 FA 首差的限制继续保留。

完整目标输出 `.bin` 只用于私有逐元素因果验证，不进入主统计包或 GitHub。msprobe 的数值首差必须独立读取原生采集推断；`null` 只能计为不可观测。运行前准备报告不能当作 msprobe 实采完成回执。

`analysis/compare_replay_statistics.py` 为本次单调用 replay 提供独立统计定位，使用原生 dump/stack/construct 对齐及配置、采集回执、collector 披露做完整性检查；不读取 host 命中日志、期望策略或私有输出定位。主 Attention 输出与五个辅助返回分开报告；跨 batch 的整张量形状不同不能形成目标请求首差。其 16 项 CPU fixture 不是实际采集证据，使用方式见 `analysis/compare_replay_statistics.README.md`。

当前可复核的 CPU 材料：

- `source_policy/build_evidence_a1/ready_summary.json`：独立 host 包构建、安装 staging、scope 与公共 1464 字节/205 字段 POD 检查。
- `replay_cpu_final.json`：B1/B8 六项输入一致性及固定 query 参考检查，`npu_calls=0`。
- `outputs/cpu_prepare/real_vendor_readiness.json`：32 个 case、真实 vendor 的 670 个文件以及 collector 清单校验。
- `analysis/cpu_checks.json`：独立分析器的 CPU 合成回归；不是 NPU 故障证据。
- `analysis/stock_scope_cpu_checks.json`：区分 stock 阶段与实际 `operator_replay` 调用阶段，拒绝 stock 携带注入 scope/host 事件。
- `analysis/replay_statistics_cpu_checks.json`：16 项独立统计定位 CPU fixture，零 NPU 调用。
- `outputs/stock_bridge_a1/completion.json`：资源门禁退出，未创建运行容器，case 列表为空。

预注册 SHA256：`7bda4d97baad8d3be15d565b405b8a22ea8017284d08262f9b8eb3c848a6ee64`。输入 manifest SHA256：`509123dc725e351d6b10840919c2640b327a13eecceba2ca4a888c4a094f8e62`。vendor manifest SHA256：`3d3ca8e0bc12003e19c6f3f4cfba21775e516385fe5807b10d59c6bb53cb1785`。

远端任务根目录为 `/home/fault28-causal-20261029`；旧实验根目录、系统 SDK 和既有发布材料均不覆盖。下一次实际运行使用新的 attempt 名称。详细源码、scope、资源边界见 `source_policy/BUILD.md`，独立分析命令及结论限制见 `analysis/README.md`。
