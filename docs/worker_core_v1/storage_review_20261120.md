# 124 存储整理评估与授权执行记录（2026-11-20）

以下第一部分为 06:56 UTC 的历史只读检查；后续授权删除见文末。该次检查未执行删除或 NPU 作业。以下容量是观测值或条件候选上限，不是删除清单；当前新增可直接执行的删除项为 **0**。

## 占用与历史回收

2026-11-20 06:56:38 UTC 的 `du -x -B1`，按实际分配空间计量、不跟随目录符号链接，并由 du 去重硬链接：

| 范围 | 分配字节 | GiB | 结论 |
|---|---:|---:|---|
| `/home/haoran/experiments` | 65,188,515,840 | 60.711536 | 混有模型、依赖、源码、原始证据，不能整体整理 |
| 其中 fault28 | 35,616,083,968 | 33.170063 | 主要日志之外仍有构建/输入/有效证据 |
| operator_validation | 9,640,341,504 | 8.978268 | 仅 MiniCPM-V-2 模型目录就占 6,878,461,952 B，必须保留 |
| fault33 | 4,885,745,664 | 4.550205 | 不能按实验结束推断无依赖 |
| `/home/17-fault-c2c3-20261117` | 79,122,833,408 | 73.688881 | 正在使用，全部保护；容量仍增长 |

两大范围合计约 134.40 GiB，不代表可回收量。07:00:34 UTC `/home` 可用 89,927,446,528 B。此前完成的 821 文件、29,931,037,238 B（27.875451 GiB）已删除且完成回执核对，不能再次计入本轮候选。

## 备份映射与边界

备份位于 `D:\DemonCATPFI\server-backups\124-20261119`。本轮以只读方式打开 `manifest.sqlite`，363 卷仍为 VERIFIED；原计划文件逻辑字节合计 233,142,945,174，压缩卷字节合计 122,565,733,342，另有 65 个复用文件。完成回执、原清理回执和独立 reconciliation 一致。既有验证是成员集合、类型、大小、gzip CRC、链接元数据和源稳定性；没有新增逐文件 SHA256，本轮也没有重新逐卷解压或实施恢复演练。ACL/xattrs 不在原备份范围。

对下述 2,209 份条件候选逐文件查询 manifest：当前大小和纳秒 mtime 全部与原记录相符，映射到 52 个 VERIFIED 卷；本轮逐卷核实这 52 个实际归档文件存在且大小与数据库一致。它们与已删除的 821 文件不重叠。

## 条件候选：历史 outputs 日志

原清理方案仅选择 fault28/fault33 的 outputs 下 >=100 MiB 日志。因此还有大量低于阈值的历史日志残留，不能误认为原清理遗漏执行。

本轮扫描 fault28/fault33 的常规 outputs 日志，先排除已发现符号链接目标或目标目录覆盖的 28 份日志（逻辑 898,601,744 B，分配 898,662,400 B），得到 **2,209 份条件候选**：逻辑 **10,539,406,523 B**，分配 **10,544,091,136 B（9.819950 GiB）**。这些文件 nlink 均为 1；这是逐文件分配空间合计，不是实际释放保证。仅 >=10 MiB 子集有 177 份、逻辑 9,727,487,898 B。

| 所在实验 | 条件候选分配字节 |
|---|---:|
| fault28-fia-graph-20261106 | 4,595,138,560 |
| fault28-fia-grain-20261109 | 2,379,198,464 |
| native-execution-strategy-20261030 | 1,760,342,016 |
| fault28-native-tiling-20261029 | 561,160,192 |
| fault33-paged-tiling-20261105 | 541,130,752 |
| fault28-model-coverage-20261030 | 431,431,680 |
| fault28-causal-20261029 | 164,216,832 |
| fault28-m-only-model-20261109 | 111,464,448 |
| fault28-eager-20261110 | 8,192 |

**9.82 GiB 不是可直接回收量。** 尚未证明每份日志的 native runtime、host tiling、statistics/mix 或异常复现信息已有完整独立保留。需要区分纯进程/编译 stdout 与原始运行证据，并核验历史分析脚本、报告、索引和之后计划对具体文件的引用。保留后者；只有依赖闭环且经明确批准的逐文件子集才能成为后续执行计划。

### 日志内容抽样后的保留分类

每份日志最多读首尾各 32 KiB，结合文件路径分类；这是保守筛查，未做完整日志语义复核。含 tiling、statistics/mix、fault/runtime 标记，或属于 native_logs/native_profile/PROF/profiler 的文件均保留。没有命中不证明是纯 stdout。

| 分类 | 文件数 | 逻辑字节 | 分配字节 | 处理 |
|---|---:|---:|---:|---|
| 原生运行/tiling/统计/性能采集证据线索 | 2,195 | 10,539,306,296 | 10,543,972,352 | 保留，待语义及引用复核 |
| 仅发现编译标记，纯编译 stdout 未确认 | 1 | 96,128 | 98,304 | 保留，待语义及引用复核 |
| 仅发现进程标记，纯进程 stdout 未确认 | 1 | 2,381 | 4,096 | 保留，待语义及引用复核 |
| 未分类 | 12 | 1,718 | 16,384 | 保留，待语义及引用复核 |

因此，9.82 GiB 应理解为已核对备份的历史日志检查范围；绝大多数已有原始证据线索，不能作为直接可回收目标。`cpu_inventory_a1.log` 虽出现编译标记，名称和抽样不足以排除其为库存报告；也继续保留。

另发现 haoran 范围 170 个归档（逻辑 1,387,723,391 B）及 45,636 个 pyc（逻辑 1,184,986,671 B），虽然元数据与原备份一致，也未列为可回收：归档可能是唯一交付/源码依赖，pyc 混在第三方环境且未证明源文件齐全。c2c3 内所有日志、归档与 pyc 均继续保护。

## 已证实的保留依赖

- 今日 source-tp2-*/source-tnd*/source-remaining*/remaining-m1*/remaining-tnd*/installed*，以及 tnd28-borrow-build-20261120-a1/build/run 和 vendor 全部保护。
- `build_tnd33_124.py` 及今日多份 source 中的 `attention28_fa_build_124.py` 明确把 `/home/haoran/experiments/fault28/fault28-fia-graph-20261106/dependencies_a4` 挂到 `/input-dependencies`。该目录分配 150,806,528 B，是当前跨目录依赖，不能随着“旧实验”删除。同目录 build/python_deps_a1/a2/a4、source_a1/a2/a4 未证明可重建且不再被引用，继续保留。
- `source-ordinarybatch-20261119-a2` 控制器与 queue，`packages-v11-source111-a1/cann851/attention-preflight`、`v11-image-mapped-libraries-a2/a3`、刚被 TP2 使用的 native-v11-postwrite39-source101-a1 缓存及其 libdcat_postwrite.so 均保留。
- 当前发布源码 `experiments/worker_core_v11_20261118/prepare_quant_ordinary_case.py:17` 明确引用 `product-qwen06-w8a8-cpu-a5/model`。其量化权重与 a4 权重是同一 inode `64770:28895052`、nlink=2、逻辑 1,952,163,304 B。删除 a4 的一个链接不会释放这份权重空间。
- Qwen3-VL-2B-Instruct 的 `model.safetensors` 和 `model.safetensors.upload-a1` 同 inode `64770:28878946`、nlink=2、逻辑 4,255,140,312 B。不能把两者误算成双份模型；`.part` 的 25,951,640 B 也未批准删除。
- `/home/gyl/models/Qwen3-0.6B`、所有模型、输入/QKV、有效 statistics/mix、源码、报告及后续构建依赖必须保留。`/home/batch-scale-dynamic-msprobe-20261023/python_packages` 是明确保护对象。
- 实际发现 `fault28-causal-20261029/outputs/derived_primary_index_a3` 的绝对符号链接指向同实验旧日志与控制器；28 份受其目标覆盖的日志已从条件候选排除。还存在 experiments 指向 `/home/batch-scale-dynamic-msprobe-20261023`、`/home/fault28-visible-r2-20261113` 等目录的链接；本轮没有跟随它们计入目录占用或扩大整理范围。

## 增量备份缺口

06:20 完成审计的 64,465 条新增记录合计逻辑 2,124,587,148 B，未包含在原备份、未删除。

06:57:38–06:58:53 UTC 的扫描中，与 manifest 原顶层路径对比，c2c3 已有 **99 个新增顶层项，包含 61,400 个常规文件，合计逻辑 3,019,259,617 B（2.811905 GiB）**。这是新增顶层项内常规文件的统计，**不是全范围增量清单**：不含旧顶层项中后来新增/修改的所有小文件；与 64,465 条“所有类型条目”不可直接按数量相减。服务器持续写入，该值只是观测下界。

主要缺口：tnd28-borrow-build 为 1,250,617,435 逻辑 B；remaining-m1rope22 为 582,184,161 B；remaining-tnd28stock 为 193,942,541 B；remaining-m1-a2 为 174,500,612 B；remaining-tnd28borrow-a3 为 163,404,377 B；remaining-m1output39 为 152,152,757 B。所有这些路径继续保留。后续如要求备份，应在任务稳定边界重新列出完整元数据差异，并对增量做成员/类型/大小/链接/CRC及源稳定性验证，不能沿用原 BACKUP_COMPLETE 宣称当前全量覆盖。

## 动态使用检查与未决证据

06:56:38 UTC 观测到 `c23-tnd28-m128-control-1911578` 容器运行；扫描末尾 `docker inspect` 的 122 个现存容器中已无该短生命周期容器。07:00:34 UTC 也未观察到运行容器。这只说明瞬时状态变化，不说明后续任务没有依赖。现存退出容器 `gyl_test` 挂载整个 `/home`。当前扫描未发现进程命令行/cwd命中两个目标根；不以此推断可删，也没有把未打开文件当作无依赖。

本轮记录 240 条相关符号链接、5,405 条有限范围脚本/配置引用，扫描无 OS 错误。引用扫描范围是目标根顶层控制器及今日 source/ordinarybatch/source-remaining 树的 <2MB 文本文件；排除 vendor/.git/build 等。没有穷尽所有历史脚本、容器内路径、动态字符串构造、外部机器计划和 mmap/打开文件。因此，条件候选仍需内容/依赖复核，服务器任何移动或删除仍需新的具体授权。不能用文件名、目录日期、退出容器、备份存在或未发现引用单独作删除依据。

## 私有审计材料

详细路径清单及扫描脚本仅保存在 `D:\DemonCATPFI\worktrees\fault28-native-fa-publish\demoncatpfi\.codex-tmp\rc2-20261119`，未加入 Git：

- `storage-review-scan.py`：只读远端元数据/链接/有限文本引用/容器采集程序。
- `storage-review-remote.json`：现场扫描记录，不含 tensor 文件内容。
- `storage-review-candidate-map.json`：71,637 份日志/归档/pyc 的元数据及备份 part/reuse 映射；后缀候选不等于删除候选。
- `storage-review-historical-output-logs.json`：历史 outputs 日志子集。
- `storage-review-log-subset.json`：条件候选、符号链接保留项和新增顶层项统计。
- `storage-review-log-content-samples.json`：日志首尾关键词分类，不能替代完整语义复核。

## 授权执行记录

时间采用 UTC（北京时间加 8 小时），以下来自实际回执，不重复删除。

| 操作 | 时间 UTC | /home 前可用 B | 后可用 B | 净增 B |
|---|---|---:|---:|---:|
| Qwen3-32B-w8a8-nopdmix 与别名 | 11-20 09:11:06 | 58,130,685,952 | 100,904,173,568 | 42,773,487,616 |
| CATMonitor 38 tags / 20 IDs | 11-20 09:12:18–09:12:46 | 100,904,005,632 | 102,641,094,656 | 1,737,089,024 |

模型 `/home/haoran/models/Qwen3-32B-w8a8-nopdmix` 的 du 为 42,773,396,798 B；按授权先移除 `/AIdata/haoran/Qwen3-32B-w8a8-nopdmix` 别名，再核查无硬链接、打开进程、符号链接引用及独立容器挂载。未归档。

所有现存容器均未引用选定镜像；逐项 `docker rmi` 均返回 0。未 prune，未删除其他项目容器或数据。原简报“37 标签”漏计一项，本次按全部 38 条 rmi 回执更正。镜像共享层的标称容量不能相加当作回收量，df 净增也可能受同时写入影响。

| 实际删除 tag | Image ID |
|---|---|
| `ghcr.io/spike677/catmonitor-stress-npu:arm64-v0.3.6-stress-job-a3-cann91-20261015-r1` | `sha256:1dc65fe1465fd6f8475a2638d1c0d04aa07ee1a9b94e1eab6621267fbb04c869` |
| `catmonitor-delivery/stress-npu:arm64-a3-cann91-5a3537c-20261014` | `sha256:22b81e2a449dfcb61bdb024ba6a5ce5cf0c9e40f0a0d1a18b41eb7b831ac933d` |
| `ghcr.io/spike677/catmonitor-stress-npu:arm64-a3-cann91-candidate-5a3537c-20261014` | `sha256:22b81e2a449dfcb61bdb024ba6a5ce5cf0c9e40f0a0d1a18b41eb7b831ac933d` |
| `catmonitor-delivery/framework:a3-cann91-models-20261014` | `sha256:27b408c0a463d5b7b36c8428e5a35209c4a627818e386ee362cce89799f1d816` |
| `catmonitor-idle/control:20261015-r1` | `sha256:35e4dfa32f9f2ace237d8bab439562551df4013434a6a2f2df14c9d4ca0858ed` |
| `ghcr.io/spike677/catmonitor-stress-npu:arm64-v0.3.6-a2-5a3537cfc0cd` | `sha256:441578d7c49c3773a86a0625b6a1f99fd517668ffb905d822bc462dbfd799735` |
| `catmonitor-npu:latest` | `sha256:46c9ece05b5bb674c813aed12ca5941dbf8d9b0576c51a5d33b8bcfd5fcff1f0` |
| `ghcr.io/computing-availability-tools/catmonitor-npu:latest` | `sha256:46c9ece05b5bb674c813aed12ca5941dbf8d9b0576c51a5d33b8bcfd5fcff1f0` |
| `ghcr.io/computing-availability-tools/catmonitor-npu:v0.3.6` | `sha256:46c9ece05b5bb674c813aed12ca5941dbf8d9b0576c51a5d33b8bcfd5fcff1f0` |
| `ghcr.io/computing-availability-tools/catmonitor-npu:v0.3.6-arm64` | `sha256:46c9ece05b5bb674c813aed12ca5941dbf8d9b0576c51a5d33b8bcfd5fcff1f0` |
| `ghcr.io/ggboom12138/catmonitor-npu:latest` | `sha256:46c9ece05b5bb674c813aed12ca5941dbf8d9b0576c51a5d33b8bcfd5fcff1f0` |
| `ghcr.io/ggboom12138/catmonitor-npu:v0.3.6` | `sha256:46c9ece05b5bb674c813aed12ca5941dbf8d9b0576c51a5d33b8bcfd5fcff1f0` |
| `ghcr.io/ggboom12138/catmonitor-npu:v0.3.6-arm64` | `sha256:46c9ece05b5bb674c813aed12ca5941dbf8d9b0576c51a5d33b8bcfd5fcff1f0` |
| `catmonitor-jobparam/control:20261014-r3` | `sha256:498506bf63df6e007cd73d1ebfacc1e5e65a0a66bd7418bffa3e202e2b3961f8` |
| `catmonitor-demo/stress-cpu:model-merge-20261013` | `sha256:4ff205552b84498eacc270b42ee5169d87c2e381d21c69d85ef67197412b7691` |
| `ghcr.io/spike677/catmonitor-stress-npu:arm64-v0.3.6-stress-defaults-a2-cann851-20261015-r2` | `sha256:5b97aee15006b1f46ae1d9fca45dfa4dfcb70f9bb5e4563ce7f0cfd8b9576bd2` |
| `ghcr.io/spike677/catmonitor-stress-npu:arm64-v0.3.6-stress-job-a2-cann851-20261015-r1` | `sha256:5b97aee15006b1f46ae1d9fca45dfa4dfcb70f9bb5e4563ce7f0cfd8b9576bd2` |
| `catmonitor-generic:latest` | `sha256:625039c983619c1bd90e1fb68dc8a91a728e05d77999c71efa86dc0f36b0dfe6` |
| `ghcr.io/computing-availability-tools/catmonitor-generic:latest` | `sha256:625039c983619c1bd90e1fb68dc8a91a728e05d77999c71efa86dc0f36b0dfe6` |
| `ghcr.io/computing-availability-tools/catmonitor-generic:v0.3.6` | `sha256:625039c983619c1bd90e1fb68dc8a91a728e05d77999c71efa86dc0f36b0dfe6` |
| `ghcr.io/computing-availability-tools/catmonitor-generic:v0.3.6-arm64` | `sha256:625039c983619c1bd90e1fb68dc8a91a728e05d77999c71efa86dc0f36b0dfe6` |
| `ghcr.io/ggboom12138/catmonitor-generic:latest` | `sha256:625039c983619c1bd90e1fb68dc8a91a728e05d77999c71efa86dc0f36b0dfe6` |
| `ghcr.io/ggboom12138/catmonitor-generic:v0.3.6` | `sha256:625039c983619c1bd90e1fb68dc8a91a728e05d77999c71efa86dc0f36b0dfe6` |
| `ghcr.io/ggboom12138/catmonitor-generic:v0.3.6-arm64` | `sha256:625039c983619c1bd90e1fb68dc8a91a728e05d77999c71efa86dc0f36b0dfe6` |
| `catmonitor-delivery/control:arm64-5a3537c-20261014` | `sha256:65ea0e1a39b191300f5c1af7c3ebafbc36c33b6a74fc8f393c7c764f0679bb33` |
| `ghcr.io/spike677/catmonitor-npu:arm64-candidate-5a3537c-20261014` | `sha256:65ea0e1a39b191300f5c1af7c3ebafbc36c33b6a74fc8f393c7c764f0679bb33` |
| `ghcr.io/spike677/catmonitor-npu:arm64-v0.3.6-a2-5a3537cfc0cd` | `sha256:65ea0e1a39b191300f5c1af7c3ebafbc36c33b6a74fc8f393c7c764f0679bb33` |
| `catmonitor-jobparam/stress-cpu:20261014-r3` | `sha256:7011860d574e8765556362003be8ffaec7bbeaf177cc2639b369cd42c0be83bf` |
| `ghcr.io/spike677/catmonitor-stress-cpu:arm64-v0.3.6-stress-defaults-20261015-r2` | `sha256:758577a1677ff407d126635ee113ab35b5d99302aab10b3aefcc08c37033259a` |
| `ghcr.io/spike677/catmonitor-stress-cpu:arm64-v0.3.6-stress-job-20261015-r1` | `sha256:758577a1677ff407d126635ee113ab35b5d99302aab10b3aefcc08c37033259a` |
| `catmonitor-ci/stress-npu:arm64-f67fb4a-lint-20261016` | `sha256:77dfd6a8e477de5964303a6b1da225e6d4939799139720b25fbb04faa7a731c1` |
| `catmonitor-ci/stress-cpu:arm64-f67fb4a-lint-20261016` | `sha256:8392907782fc76b48528eaf9751addca6413f8089fe4a6d965f5ce9288f573e1` |
| `catmonitor-demo/control:npu-20261012-r5` | `sha256:8b179d3f9a3b785c9762982696498b3901560c67a319790bf93b33fd241b2318` |
| `catmonitor-ci/control:arm64-f67fb4a-lint-20261016` | `sha256:97d07eaa052c2084bf22c1878613df45e8a7a2004e4d8f3939b75b3352bee5e7` |
| `ghcr.io/spike677/catmonitor-npu:arm64-v0.3.6-stress-defaults-20261015-r2` | `sha256:9858f14b8b562a21e0429e222064998e260085dc9d58c07519879b462b806acc` |
| `ghcr.io/spike677/catmonitor-npu:arm64-v0.3.6-stress-job-20261015-r1` | `sha256:9858f14b8b562a21e0429e222064998e260085dc9d58c07519879b462b806acc` |
| `ghcr.io/spike677/catmonitor-stress-cpu:arm64-v0.3.5-stress` | `sha256:ed2a91fefd06033ebd75cd47786624d8e0edc1b3234d70310bbe60b47660d586` |
| `catmonitor-idle/stress-npu:20261015-r1` | `sha256:f965bfeac14463838290391f099f3acf62453c3f3136c78b7322a39a8e1a00a9` |

完整 df、du、mtime、引用核查及 stdout 原件：`D:/DemonCATPFI/server-backups/124-storage-20261120`。

- `model-preflight.json`: SHA256 `678b413c00189602eeec3ff6c1f464989622f77535f868eaf91069f609210fe3`
- `model-delete.json`: SHA256 `f90ec273f3193494fe0a2cd3c5925bf08be035cc01fe97fcf395dfed96b6b2ee`
- `images-preflight.json`: SHA256 `ac3655cc2d68f7d8a5c9c0cfffa683ed7a70064f2d36038ac4a0964084d0c4ff`
- `images-delete.jsonl`: SHA256 `45d501c3595af1725760a6f0506187307d26ddaba2213b48ab457121ef2fe7c4`
