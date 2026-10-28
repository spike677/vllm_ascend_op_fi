# 故障 27：视图别名断开机制的 B/F 统计证据

状态：**机制级 B/F 与统计证据验收通过**。本目录来自一次完整的正式运行 `npu_a5`；控制器状态为 PASS，独立验收为 PASS_MECHANISM_AND_STATISTICS。打包没有重新执行 NPU，也没有更改原生 JSON 或验收门槛。

B 使用原始 NZ29 BF16 base 的真实 as_strided 视图。F 只在受控视图入口执行 base→ND2 格式转换，再调用原始 as_strided；它不修改写入或消费结果。两组随后执行相同 view.copy_ 和 original_base+1。共同观察器与独立整数索引参考确认：F 的原 base 保留旧值，消费者差异对应 64 个目标元素，192 个非目标元素保持不变。

固定几何为 base `[16,16]`、初值 arange(256)，view `[8,8]`、stride `[16,1]`、offset 17，实际写入 `-64-arange(64)`；原 base 消费者为 `base+1`。B/F 都是逻辑 rank0、一个真实微基准 forward 的 step0，不是模型 token 步数。

机制差异是构造视图时失去原存储别名；写入后才出现原 base 的数值差异。原生统计按语义模块对齐得到的首个可观测数值差异为 `Module.read_original_base.ReadOriginalBase.forward.0`。完整 trace 的观察器节点、未配对节点和不可观测统计限制见 [native 报告](reports/acceptance.native.json)；不要将缺失/null 统计判为相等。

此结果属于构造的最小机制实验，不是模型级调用链通过，也不证明历史 CANN 缺陷或生产事故已复现。统计值不能独立唯一证明别名根因；存储身份、真实写入与精确小规模值来自单独的共同运行观察器，见 [joint 报告](reports/acceptance.joint.json)。

## 数据分区与原始路径

- `statistics/B/`、`statistics/F/`：每组 config、receipt 和 step0/rank0 的 dump、stack、construct，共 5 个原始 JSON。配置为 task=statistics、level=mix、summary_mode=statistics，无 tensor_list 全张量采集。
- `B_msprobe_statistics.zip`、`F_msprobe_statistics.zip`：各自仅打包上述 5 个 JSON，ZIP 内路径从 config.json/observation 开始；内容与解压目录逐字节相同。
- `runtime/B/`、`runtime/F/`：完整小规模逻辑值/BF16 位值、存储身份及单独 control；这些不是 msprobe 统计推断的输入。
- `reports/`：正式 B/F 验收、健康先行门槛、控制器完成/源码指纹、资源前后快照、每组原始日志、停止授权执行记录（如有）。不包含失败 attempt 或缓存。
- `source/`：这次运行冻结且经 SHA 核对的源码与参考。版本：mindstudio-probe=26.1.0, torch=2.10.0, torch-npu=2.10.0.post4。

默认 Python mix API 列表未包含 torch.as_strided / Tensor.copy_；正式运行使用官方 `register_custom_api` 接口额外观察这两个真实调用，并逐项恢复原函数。没有修改安装包或拼造 dump；实际覆盖门槛见 native 报告。资源结束快照确认本实验容器已移除，不要求整台主机没有其他合法任务。

config/dump 中 `/evidence/B`、`/evidence/F` 等历史路径，以及 receipt 中的相对引用均保持原样。receipt 的 runtime/control 哈希对应本目录 `runtime/{B,F}/`；native_files 的相对路径对应 `statistics/{B,F}/`。它们不代表需要访问原机器。所有来源映射见 [source_mapping.json](source_mapping.json)。不含 `.pt`、模型权重或其他张量二进制。

已附[警告来源审查说明](reports/warning_note.md)；原始 B/F 日志保留所有警告。

## 校验

```bash
python package_evidence.py --verify .
```

[SHA256.json](SHA256.json) 覆盖其自身以外的全部交付文件，ZIP 也校验。PASS_PACKAGE_INTEGRITY 只表示文件完整性；机制结论来自原始验收报告。`source/analyze_alias.py` 的原始命令要求每组 config、runtime、control、receipt 与 observation 同目录；本包为清晰分区调整了目录，原文件字节未改。
