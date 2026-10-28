# 故障 15：每次满足条件均触发的 B/F 统计证据

状态：**持续条件命中的机制级 B/F 与统计证据验收通过**。本目录来自一次完整的正式运行 `npu_allhit_a2`；控制器状态为 PASS，独立验收为 PASS_ALL_ELIGIBLE_MECHANISM_AND_STATISTICS。打包没有重新执行 NPU，也没有更改原生 JSON 或验收门槛。旧单次版本及其证据保留，本目录是新的独立交付。

本机制正确编号为**故障 15**；此前单次实验曾误标为故障 27。历史单次源码与原始采集、完整证据包保留原路径、schema 编号和原始字节；公开说明文档另加编号纠错提示。本次新方案使用故障 15，不回写旧原始证据。

B 使用原始 NZ29 BF16 base 的真实 as_strided 视图。F 每次满足注册对象、实际格式/几何和指定入口条件时，执行 base→ND2 格式转换，再调用原始 as_strided；没有首命中后禁用或第二次匹配报错的策略。临时张量创建时等于该次原 base，写后视图与临时存储更新，原 base 保留旧值。

每组在同一进程和持续注入器下运行 **2 个真实 forward × 每个 base 连续 3 次调用**。两个注册 base 提前创建并强引用，forward 之间使用不同 base，同一 forward 的三次调用不重置 base。固定几何为 `[16,16]`、初值 arange(256)，view `[8,8]`、stride `[16,1]`、offset 17；三次写入分别为 `-64-j`、`-128-j`、`-192-j`，`j=0..63`，每次都由同一原 base 计算 `base+1`。

F 的 6 次符合条件调用全部触发，B 的故障命中数为 0；两组都有 6 次匹配的原生调用。每次写入，B 原 base 的 64 个目标元素相对自身前态全部改变，F 原 base 相对自身前态不变；每次消费者 B/F 差异恰好在这 64 个目标位置，192 个非目标位置不变。每 forward 另有不同调用位置的已注册 base 只读视图，以及同一选中入口的未注册 NZ base 写入/消费两个 guard；每组 4 次 guard 均原生透传，未注册 base 的写值为 `-16-j`。

第一个调用前，两端新 base 的精确值一致；本次原生语义模块首个被观测数值差异在写后读取原 base。第二、第三次调用前，B/F 已因前次写入具有不同值；已观察到的 ReadBaseBefore 差异属于继承传播，不能冒充新的更早故障点。逐次命中验收依靠每次实际新视图的存储关系及各端自己的 pre/post 转移，不能仅凭累计 hits 或持续输出差异。按 step、invocation 语义模块和调用栈对齐的实际结果见 [native 报告](reports/acceptance.native.json)，精确状态与别名检查见 [joint 报告](reports/acceptance.joint.json)。

| case | 首个被观测数值差异的语义阶段 | 首个对齐 API 数值差异 B / F |
| --- | --- | --- |
| f0_c0 | read_original_base | NPU.npu_format_cast.15.forward / NPU.npu_format_cast.17.forward |
| f0_c1 | read_base_before | NPU.npu_format_cast.24.forward / NPU.npu_format_cast.29.forward |
| f0_c2 | read_base_before | NPU.npu_format_cast.39.forward / NPU.npu_format_cast.49.forward |
| f1_c0 | read_original_base | NPU.npu_format_cast.15.forward / NPU.npu_format_cast.17.forward |
| f1_c1 | read_base_before | NPU.npu_format_cast.24.forward / NPU.npu_format_cast.29.forward |
| f1_c2 | read_base_before | NPU.npu_format_cast.39.forward / NPU.npu_format_cast.49.forward |

call 0 的 API 首差异来自共同观察器写后读回，不能把该诊断格式转换本身当作故障发生位置。原生 dump 节点数：B step0/step1 = 87/87，F = 102/102。

此结果属于构造的最小机制实验，不是模型级调用链通过，也不证明历史 CANN 缺陷或生产事故已复现。统计值不能独立唯一证明别名根因；存储身份、真实写入与精确小规模值来自单独的共同运行观察器，见 [joint 报告](reports/acceptance.joint.json)。

## 数据分区与原始路径

- `statistics/B/`、`statistics/F/`：每组 config、receipt 和 step0–1/rank0 的 dump、stack、construct，共 8 个原始 JSON。配置为 task=statistics、level=mix、summary_mode=statistics，无 tensor_list 全张量采集。step 对应两个真实微基准 forward，不是模型 token 步数。
- `B_msprobe_statistics.zip`、`F_msprobe_statistics.zip`：各自仅打包上述 8 个 JSON，ZIP 内路径从 config.json/observation 开始；内容与解压目录逐字节相同。
- `runtime/B/`、`runtime/F/`：完整小规模逻辑值/BF16 位值、存储身份及单独 control；这些不是 msprobe 统计推断的输入。
- `reports/`：正式 B/F 验收、健康先行门槛、控制器完成/源码指纹、本次资源策略与逐次检查、资源前后快照、每组原始日志。不包含失败 attempt 或缓存。
- `source/`：这次运行冻结且经 SHA 核对的 9 份源码与参考，包括只读结束检查器。版本：mindstudio-probe=26.1.0, torch=2.10.0, torch-npu=2.10.0.post4。

冻结的 acceptance_contract.json 仍保留运行前设计状态 pending；本次实测状态以 [acceptance.json](reports/acceptance.json) 为准，不回写冻结文件以维持 source_manifest 对应的原始 SHA。

默认 Python mix API 列表未包含 torch.as_strided / Tensor.copy_；正式运行使用官方 `register_custom_api` 接口额外观察这两个真实调用，并逐项恢复原函数。没有修改安装包或拼造 dump；实际覆盖门槛见 native 报告。结束检查通过容器完整 ID 和 completion、source_manifest、冻结检查器的 SHA 绑定本次运行，确认自有容器已移除；较晚的资源快照可能已变化，不要求整台主机没有其他合法任务，也不据空闲快照声称设备独占。

config/dump 中 `/evidence/B`、`/evidence/F` 等历史路径，以及 receipt 中的相对引用均保持原样。receipt 的 runtime/control 哈希对应本目录 `runtime/{B,F}/`；native_files 的相对路径对应 `statistics/{B,F}/`。它们不代表需要访问原机器。所有来源映射见 [source_mapping.json](source_mapping.json)。不含 `.pt`、模型权重或其他张量二进制。

已附[警告来源审查说明](reports/warning_note.md)；原始 B/F 日志保留所有警告。

## 校验

```bash
python package_evidence.py --verify .
```

[SHA256.json](SHA256.json) 覆盖其自身以外的全部交付文件，ZIP 也校验。PASS_PACKAGE_INTEGRITY 只表示文件完整性；机制结论来自原始验收报告。`source/analyze_alias.py` 的原始命令要求每组 config、runtime、control、receipt 与 observation 同目录；本包为清晰分区调整了目录，原文件字节未改。
