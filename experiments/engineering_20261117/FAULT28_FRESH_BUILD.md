# Fault28 fresh build 与独立实验资格

本轮两条新 FIA 变体均完成源码构建、索引登记修复和限定小模块的 B/F/OFF 执行。
完整输出存在差异且重复稳定，OFF 精确恢复；**原生 statistics/mix 均没有可观测首差**。
没有取得独立 kernel argument/最终 launch 资格，也没有升级完整模型、公共 Worker 或
全部匹配覆盖资格。本轮有界验证到此结束。

本轮恢复的是可从源码重建的 FA/FIA 实现，历史证据单独保留。新 FIA 包不复用旧
`.o`、host 库、kernel binary 或旧资格。生产 `fault_injection/fia.py` 的证书/SHA 门禁
保持不变；不能替换新包 hash 后便宣称继承了旧模型或 Worker 验收。

FIA 基于独立检出的 ops-transformer
`c77c0829f1f0f0b8f882db1d9b051a758b605472`，分别选择 `legal_tiling` 和
`denominator_replay`。两者使用独立的新源码、host 构建、private vendor 和 matching
kernel；replay 额外包含 `119000000000400000`，不能并入 legal 模式的资格。

`build_fresh.py` 全量构建 host package，再用同次构建生成的 FP16 参数编译匹配 kernel。
依赖从固定 commit 或 source archive 重建。opbase 提交中自带的 14 个测试 dummy `.o`，
只在新 clone 内核对 Git tracked/blob、精确路径、长度和 SHA 后删除；实际配置确认
tests OFF，未知 object 仍拒绝。第二次构建发现缺少 `patch`，第三次由同机 aarch64
GNU patch 的只读挂载补齐，官方依赖 patch 仍照常执行。

实际状态以各新 attempt 的 `evidence/completion.json` 为准。
`CPU_BUILD_ONLY_REQUIRES_NATIVE_QUALIFICATION` 只说明全量源码/host/kernel 构建与
磁盘产物校验完成；`loading_allowed=false`、`npu_executed=false`，未授予生产加载、
独立 launch 或模型资格。CPU 构建不暴露 NPU。

本轮 legal a3 和 replay a1 均已实际完成 fresh host/kernel 构建。新包指纹如下，
完整 SHA、源码/依赖身份和失败尝试见
[`attention28_fresh_summary.json`](attention28_fresh_summary.json)。

| 新构建 | matching keys | kernel SHA256 |
| --- | --- | --- |
| legal a3 | `103000000000000000`、`109000000000000000`、`109000000000400000` | `fc352594d5d02f759cc5936e89ef88e81b449c91ae7f8c1613cb2c6801da3a0a` |
| replay a1 | 上述 3 个及 `119000000000400000` | `4305610d7521f1bc4d34ed2e0074fd0f42782a3f283b1ac24fce877857519b67` |

## 公开 serializer 准备

[`compile_public_schema.py`](../../fault_injection/native_adapters/attention28/fia/compile_public_schema.py)
从新源码公开 tiling 声明生成 projection，使用本次 `compile_commands.json` 中真实 FIA
host 编译条目的 include/define，重新编译 decoder。历史仅导入 decoder 源码，原字节
hash 见相邻 provenance 文件；未导入旧 decoder 可执行文件。新程序执行 sentinel、
serializer roundtrip、数组和错误长度检查，输出 schema 与编译回执。

```sh
python /repo/fault_injection/native_adapters/attention28/fia/compile_public_schema.py \
  --source /build/run/source --sdk /usr/local/Ascend/cann-9.1.0 \
  --out /evidence/schema
```

新 build 保持编译时的 `/build` 路径只读可见；schema 输出使用独立可写目录。该程序
只在 CPU 编译/执行 serializer 检查，不执行算子。

## 新包的最小 NPU 实验

[`fresh_fia_qualification.py`](fresh_fia_qualification.py) 是独立试验工具。执行者显式给出
新 completion 和 schema 的 SHA；工具逐文件核对新 vendor 全目录、长度、内容 hash、
无额外文件、无未经核对的 symlink 和 matching key 集，再 exec 全新进程，仅对该进程配置新
vendor、`ACLNN_CACHE_LIMIT=0`、同步队列及 16/32 核。不修改全局 SDK、旧 vendor、
Docker 入口或外部任务。

初次 bridge 在导入 torch 前因官方 `op_tiling/liboptiling.so` 内部软链被拒绝，未执行
NPU 调用。只读盘点后，校验器只允许该精确路径及字面目标
`lib/linux/aarch64/libcust_opmaster_rt2.0.so`；链接和最终普通文件都必须在同一 manifest
中，长度/SHA 一致且实际内容匹配。目录链接、越界、链式链接、不同目标和未列入链接
仍拒绝；原构建回执没有改写。

stock/OFF 桥接随后完成，但首次 legal baseline 在真正 kernel 调用完成前失败：
`ACLNN 161002: Cannot find tilingKey[109000000000000000] in kernel json`。
该次 `calls=[]`，没有 F 或 statistics 采集，不能用 host 已选 key 抵消失败。

只读核对发现，新旧包的单算子 registry 与 op-info 内容 hash 分别完全一致，但新包缺少
`kernel/config/ascend910b/binary_info_config.json`。本次新编译 metadata 包含正确的
3 个 key；从本次 source 的官方 `ascendc_ops_config.gen_ops_config` 在内存生成配置后，
生成的单算子 registry 精确匹配现有 registry，生成的 summary 索引也与旧有效布局结构
一致。旧索引仅作诊断对照，不是修复工具的输入。

[`derive_registered_vendor.py`](../../fault_injection/native_adapters/attention28/fia/derive_registered_vendor.py)
只在新目录派生已校验的新 vendor。它调用固定 hash 的本次官方源码函数，从新 metadata
生成缺失索引，核对 key 集、输入/输出/属性 registry 及 simplifiedKey/binPath/jsonPath
一致后写入。所有 parent 文件和 host/kernel 字节逐一重验不变，新 completion 记录 parent
SHA、生成器/helper 身份及新增文件。原 fresh 构建和 completion 均不改写；每种变体用
自己的派生目录。未来 `build_fresh.py` 已加入同一生成流程。

```sh
python /repo/fault_injection/native_adapters/attention28/fia/derive_registered_vendor.py \
  --parent-completion /build/run/evidence/completion.json \
  --parent-sha256 ACTUAL_PARENT_COMPLETION_SHA256 \
  --vendor /build/run/stage/vendors/dcat_f28_legal_transformer \
  --source /build/run/source --out /evidence/derived
```

这一步只补注册索引，不重新使用旧 host/kernel。派生后已在 fresh process 重新执行
stock/OFF、B/F/OFF 与统计复核，具体结果见下文；CPU 登记回执本身不代表实机通过。

```sh
python /repo/experiments/engineering_20261117/fresh_fia_qualification.py \
  --completion /build/run/evidence/completion.json \
  --completion-sha256 ACTUAL_COMPLETION_SHA256 \
  --vendor /build/run/stage/vendors/dcat_f28_legal_transformer \
  --public-schema /schema/fia_abi_schema.json --schema-sha256 ACTUAL_SCHEMA_SHA256 \
  --phase baseline --batch 8 --out /evidence/run \
  --experimental-qualification --statistics
```

hash 必须取实际文件，不允许占位执行。replay 使用自己的 completion、schema 和
`dcat_f28_replay_transformer`。每 case 使用 fresh process 和证据目录；资源锁、设备
门禁及每批最多 8 case 仍由外层 controller 执行。

范围为 B1/B8、FP16、BNSD、Hq16/Hkv8、Q/KV2048、D128、causal、`inner_precise=0`。
Q/K/V 使用固定整数模运算生成规则，各请求行相同；记录并复核完整输入 hash。
这是构造的 standalone FIA 小模块，不是完整模型或自然调度请求验收。

| phase | 实际调用 | 核查 |
| --- | --- | --- |
| stock | 2 次，无 private vendor | 正常算子及重复完整输出 |
| off | 2 次，私有源码关闭干预 | 与 stock 完整输出精确相同后才继续 |
| baseline / fault | 各 2 次，首轮预热、第二轮可采统计 | 同输入、重复输出、实际 host M/S2/key |
| off_recovery | 同进程 off → fault → off → off，4 次 | 干预关闭后完整输出精确恢复 |

legal 的 B 为 M512/S2=512，F 为 M128/S2=128；replay 的 B/F 均为 M128/S2=128，
只有 replay key 和核内 denominator 重放分支不同。两条机制各自记录。
legal 输出没有变化也是有效观察，不能为取得差异修改返回值。

B/F 的 `--statistics` 使用本实验未修改的原生 msprobe `statistics/mix`，采真实 step0
的一次 FIA API/module 输入输出，不过滤算子、不改统计字节。各组挂载为相同
`/evidence/run` 可保持配置中 dump_path 一致。完整输出比较、输入 hash、host 审查和
恢复验证在采集窗口外；原始 null/NaN 保留。原始三件套、配置和采集器身份留在各组。
独立首差仍需用固定 `rtol=1e-5/atol=1e-8` 的统计比较器另行复核。

工具核对同 nonce 的 host 记录、serializer 原始字节与公开 schema 的逐字段解析、
host typed 值、实际 M/S2/key，以及映射的 vendor 文件 inode 和 hash。这些不能替代
独立 kernel argument/profiler。全通过时也只返回
`EXPERIMENTAL_EXECUTION_COMPLETED_NOT_LAUNCH_QUALIFIED`，保持
`native_launch_verified=false`、`kernel_argument_raw_observed=false`、
`public_worker_qualified=false`、`model_acceptance=false`。

后续 profiler 须独立关联实际 task/key、kernel 元数据和 raw 参数，不能拿期望 key
或命中日志代替最终 launch。重新完成对应资格后，才能为新包建立新证书。历史 FA/FIA
图模式、完整 Qwen3 传播、全匹配覆盖和旧 statistics 的通过结果不能自动迁移。

新增 CPU 检查覆盖产物篡改/额外文件/旧 object、错误 key 集、越界路径、过早提升加载
资格、legal/replay 区分、公开 serializer 字节不符与错误长度；CPU 测试不计实机通过。

## 本轮最终结果

两条变体各完成派生包的 4 次 stock/OFF 桥接及 8 次 B/F/OFF 调用，最终共 24 次 FIA
返回。修复前另有两组 bridge 共 8 次返回，原件单独保留；首次 legal baseline 的失败
不计完成调用。固定输入为 B8 的合成 Q/K/V，未加载预训练模型。

| 变体 | B/F 实际 host M/S2 | 完整输出 | 重复与 OFF | 独立 statistics 首差 |
| --- | --- | --- | --- | --- |
| legal_tiling | `512/512 → 128/128` | B/F hash 不同 | B/F 各重复稳定，OFF 精确恢复 | 无可观测差异 |
| denominator_replay | `128/128 → 128/128`，host key 切至独立 replay key | B/F hash 不同 | B/F 各重复稳定，OFF 精确恢复 | 无可观测差异 |

每组 host 原始 serializer 字节与本次公开 schema/typed 字段一致，实际加载的新 opapi
和 host 文件通过 inode/hash 核对。以上是 host 选择证据；最终 kernel 参数和 launch
没有独立取证，不能据此授予 `NATIVE_CLOSED`。台账的 E1/E3/E4 保留为本小模块的实际
执行、完整输出变化及实验 OFF；E2 只记录 host 选择，完整 kernel 干预资格为未通过。
生产 `FiaBackend` 门禁保持原样，C2/C3 均为 0。

独立比较固定使用 `rtol=1e-5`、`atol=1e-8`，没有读取 host 日志、命中记录或完整张量。
每组真实 step0 的原始 dump 只含一个 FIA API 记录，比较器展开为 6 条 tensor 记录；
它不是完整模型逐层 dump。四份 B/F 原始 dump 的字节 hash 都相同：
`f27816f7c6c058899170e19044c957edfc92fc6c944f294de53d3bda02701906`。

FIA `output[0]` 的实际原始统计如下，两条变体及两组都一致：

| 字段 | Baseline | Fault | 判定 |
| --- | --- | --- | --- |
| shape | `[8,16,2048,128]` | `[8,16,2048,128]` | 对齐 |
| Max | `0.9296875` | `0.9296875` | 无可观测变化 |
| Min | `-0.9453125` | `-0.9453125` | 无可观测变化 |
| Mean | `null` | `null` | 不可观测，不作相等判断 |
| Norm | `null` | `null` | 不可观测，不作相等判断 |

每组完整记录保留 14 个 null 字段、0 个非有限统计；空 `output[1]` 的四项统计均为 null。
两次独立报告均为 `NO_OBSERVABLE_STATISTICS_DIFFERENCE`，没有首差算子或字段。
B 自比和记录结构可对齐（`whole_dump_comparable=true`），这只说明可观测统计可比，
不表示 null 可观测，也不表示完整张量相等。

因此，本次完整输出 hash 的变化是独立运行时辅助证据；原生统计的 Max/Min 未变，
Mean/Norm 不可用，**仅使用这两组 statistics/mix 无法识别或定位该数值变化**。
不授予 E5 故障定位通过，不修改 null、不调阈值、不继续搜索放大本轮结果。
完整模型/logits 传播、自然请求自动选点、其他形状及版本仍需各自的新验收；历史结果
不被本次负统计结果覆盖，也不能反向替代新构建的缺失资格。

最终构建、派生、输入/输出、原始统计三件套、配置、采集器、比较器和运行回执的路径与
SHA 均列在 `attention28_fresh_summary.json`。本轮只提交实现和总结，新的原始统计、
完整私有输出和二进制保留在 124。
