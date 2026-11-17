# Agent 7：公开视觉模型准备链独立复核

**已独立确认固定版本模型文件完整，以及真实图片的 CPU 预处理执行完成。尚无这条视觉链路的模型前向、NPU 故障、B/F/OFF、E5、C2 或 C3 验收。** 本次不改变已有机制资格或计数。

复核摘要为 [vision_preparation_summary.json](vision_preparation_summary.json)，校验程序为 [review_vision_preparation_124.py](review_vision_preparation_124.py)。程序仅用 Python 标准库读取既有文件；唯一外部命令是只读 `docker image inspect`。没有创建容器、导入模型运行时、执行 NPU 计算或重跑处理器。公开材料不含模型权重、原始图片或张量载荷。

| 已检查内容 | 独立结果 |
| --- | --- |
| 官方来源 | `Qwen/Qwen3-VL-2B-Instruct`，revision `89644892e4d85e24eaac8bacfd4f463576704203` |
| 完整文件 | 10 个固定文件全部重新流式读取，共 4,266,640,306 bytes |
| 权重 | 4,255,140,312 bytes；SHA256 `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`，匹配官方 LFS 元数据 |
| 其余 9 个文件 | 尺寸、SHA256 和官方 Git blob SHA1 全部匹配；原始下载、后续本地复核、远端发布三份清单一致 |
| 发布文件 | `model.safetensors` 与保留的 `.upload-a1` 当前仍为同一已验证 inode；读取前后 fd 与路径身份、尺寸、mtime/ctime 均稳定 |
| 权重头 | 625 个 tensor header；patch weight 为 BF16 `[1024,3,2,16,16]`，bias 为 BF16 `[1024]`；仅解析头，未构造模型或载入权重 |
| 冻结源码 | `source-a33` 的 609 个文件与原 manifest 全部匹配；manifest 基线 `498325a85bab978af1d40f89859c2f0a4e22453d` |
| CPU 处理器 | 原始 a3 退出 0；镜像内 Transformers 4.57.6 的模型源码与 processor 源码、只读 driver 库、输入图片和 9 个配置文件的哈希均匹配原回执 |

CPU a3 使用默认公开 processor，对固定自然图片和文本 `Describe this image in one sentence.` 执行预处理，未改 patch 几何或 processor 配置。原回执记录 `input_ids=[1,901]`、`attention_mask=[1,901]`、`pixel_values=[3536,1536]`、`image_grid_thw=[1,3]`，对应 dtype 与摘要见 JSON。这些输出形状及摘要与进程日志的终端 JSON 一致；本审计没有重新计算图像张量，因此不将它称为第二次处理器数值复现。

**CPU a3 的 `actual_npu_calls: 0` 不能解释成运行时 API 零调用。** 进程日志实际保留了 `halGetDeviceInfo`、`rtGetDeviceCount`、`aclrtGetDeviceCountImpl` 失败和设备数量探测告警。已执行脚本和控制命令支持的结论是：未执行显式 NPU 计算，未构造模型或加载权重。原命令使用 `runc`、`--network none`、无 device 映射，driver 仅只读挂载；脚本要求输出张量位于 CPU。原字段是写入回执的常量，不是运行时 API tracing 计数。

原 a3 实际执行脚本 SHA256 为 `f2a857c8f215aa66ee72f9b36e331dfb137ed7a00ea0b690969faa4cff9a3987`。后续 `1a68fd5` 将未来脚本表述改为 `explicit_npu_compute_invocations=0` 和 `runtime_discovery_may_be_attempted=true`，没有重写 a3 原件；本审计使用的仍是 a3 冻结脚本。a1/a2 的退出 1 与 `libascend_hal.so` 缺失也均保留，没有改写为成功。

文件发布验证器 `verify_vision_remote_a1.py` 校验固定 metadata、文件清单、官方 LFS/Git blob、读取稳定性，再通过不覆盖已有目标的 hardlink 发布。本次进一步独立重读了发布后的全部字节，确认最终文件与原回执一致。这能证明本次已完成准备的文件身份；不能由此保证未来没有写入者修改该 inode。初始下载回执中的 helper 哈希是完成时记录，不能补造为下载启动时脚本身份；当前验证器和本次审计各自记录自身身份，与这个历史限制分开。

原件仍位于 124 的 `/home/17-fault-c2c3-20261117`。独立产物为 `agent7-vision-preparation-a1/summary.json`；本地公开 JSON 与该产物逐字节一致。

| 证据 | SHA256 |
| --- | --- |
| 独立摘要，本目录 `vision_preparation_summary.json` | `97b647b9973a0aa6c6b8f0f8d564418439f513751b30925cd046b5bd94a6fd20` |
| 本次审计程序，实际远端执行版本 | `50de79ce3e88e2494c39a0e7e30384d739dd2a4f918ccc5365b37d2ea265c370` |
| 官方固定 metadata | `2fcf3612194e166fae921a35892bce9f424b6095a4edf24fc1e5ebfbf5eb7395` |
| 原始下载回执 | `b4b6f6c8fdba3823e60230cb410ee3496c747f5b8723b6bbdb730e9aac18500a` |
| 本地文件独立重验回执 | `2f3dea862ec5acdc23823dad83986e3bad91931d3160dbdc60d872e83be2e409` |
| 远端文件发布回执 | `854c618c173e073df60330e09b1fdd3c3a0816adcd4e5e2e87698ac9d9590faa` |
| source-a33 manifest | `5330df14ad254a2d93638417634d347b8fd53e248480ecb4fd5f57f0951124a9` |
| a3 controller | `6c80f543731924ff4db557a9c68d6867f6a050247332e7de9d76d3ad3c82dbc5` |
| a3 processor 回执 | `2af508bd8b19e027909cd97a6b5b7f0c04adf74849348318ae7674df631370f8` |
| a3 process log | `a7698599fd51bad1a4f76009316274b4af12c05ce20d13cabdb3b9c1d4b747b5` |

8 项 CPU 单元检查通过，覆盖精确文件集合、官方 revision、官方 Git blob、LFS 与回执共同错误时的拒绝、同字节哈希/解析，以及设备探测告警不能被 `actual_npu_calls:0` 掩盖。未把这些 CPU 检查计入任何机制的 NPU 或 Worker 验收。

后续仍须在自然模型推理中验证实际目标调用、故障触发及效果、B/F/OFF 恢复和独立 `statistics/mix` 首差。处理器输出或模型文件完整不能替代上述任一环节。
