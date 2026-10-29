# 故障 28 因果矩阵的私有源码构建

本目录构建同一私有 FA host 库，以四个独立的 M/nRatio 组合区分策略因素。它不修改 kernel、不改输入值或精度属性、不读取 executor 私有内存，也不把 profile 名称当作实际执行证据。CPU 构建成功不等于所有组合已通过 NPU 验收；每次必须核原生 callback 返回值、实际启动 raw/key/workspace、逐 query 工作划分与数值结果。

## 固定来源

- 官方 ops-transformer v9.1.0：`c77c0829f1f0f0b8f882db1d9b051a758b605472`，仓库 `https://gitcode.com/cann/ops-transformer.git`。
- opbase：`cd48c600c4b37146f504217eab2bf3956f7fee8b`；ops-tensor：`b24535e30a6f7a3909c53c8b09713a9c0d08e54b`。
- CANN 9.1.0 / V100R001C11SPC001B243 / OPP `20260730_231653901`，固定 CPU 构建镜像 `sha256:8988c3cd3a971a72ebb2e5baae24e0091cde55146f96d2b4c8229448210735b9`。
- FA kernel 及公共头沿用之前安装源码逐文件匹配证据。新增固定 `common/include/op_kernel/util.h`，SHA256 `442648c354814d149ba7f8185e4b16cc7d528d1333f8437f07c0f3c5c93187d9`。

`build_cpu.py` 是本次实际无设备构建驱动。它只在新根 `/home/fault28-causal-20261029` 内写入：从此前固定官方 clone 创建新 checkout，复制独立依赖副本，调用 `apply_policy.py`，冻结所有 host 源 SHA，再启动 4 CPU / 16 GiB / 1800 秒的无 NPU 容器。它保留完整 argv、日志、源码前后指纹、return code 和容器清理记录。旧源码/依赖/SDK 未写入。新环境复建时应从上述官方 commit 获取材料；脚本中的旧本地来源仅为本机固定缓存，不能用移动分支替代。

构建参数、依赖目录及 SDK 路径完整保存在脚本中。关键参数为 `ENABLE_BUILD_PKG=ON`、`ENABLE_BUILT_IN=OFF`、`ENABLE_OPS_HOST=ON`、`ENABLE_OPS_KERNEL=OFF`、`VENDOR_NAME=fault28_causal`、`ASCEND_OP_NAME=flash_attention_score`、`ASCEND_COMPUTE_UNIT=ascend910b`。生成私有包包含原 kernel JIT 源码；未在 CPU 阶段执行 NPU kernel。

## 策略与 scope

只在固定 S=2048、Q16/KV8/D128、FP16/BNSD、BOOL 2048² mask、causal sparse2、inner_precise0、keep_prob1、pre_tokens INT_MAX、next_tokens0、精确 FP32 scale=1/sqrt128 且所有附加 tensor 为空的原生 context 内生效。

| requested profile | actual B1 | actual B8 |
|---|---|---|
| base | 原生 M128/R8 | 原生 M128/R8 |
| m_only | M64/R8 | M64/R8 |
| r_only | M128/R4 | M128/R4 |
| joint | M64/R4 | M64/R4 |
| batch_m | 原生 M128/R8 | M64/R8 |
| batch_r | 原生 M128/R8 | M128/R4 |
| batch_joint | 原生 M128/R8 | M64/R4 |

以上是目标策略，实际字段仍须由 raw 独立核验。自动模式在 C++ 读取 actual context B 后决定，不由 Python 修改 M/R。完整环境契约：

```text
FAULT28_CAUSAL_PROFILE=<上表名称>
FAULT28_CAUSAL_SCOPE_PID=<当前进程 PID>
FAULT28_CAUSAL_SCOPE_TID=<当前 native thread ID>
FAULT28_CAUSAL_SCOPE_PHASE=operator_replay
FAULT28_CAUSAL_SCOPE_BATCH=1|8
FAULT28_CAUSAL_SCOPE_TOKEN=<1..128 ASCII 字母数字或 _ . ->
FAULT28_CAUSAL_LOG_PATH=<任务目录内绝对路径>
```

scope、shape 或属性未匹配时原生透传；不能以缺失日志当作成功。这里是原模型捕获 QKV 的算子 replay，不声称正在执行模型 layer1 forward。

## 静态合法性范围

匹配 host `arch22/flash_attention_score_tiling_general.cpp` 的 S1S2 模板（优先级96）先选 M 和 nRatio，再原生生成 core、workspace、Matmul、Softmax 参数。仅替换这两个 selector 的返回值。`SetMatMulTiling` 失败仍原样失败，实验必须拒收，不能把小尺寸推论代替 SDK 验证。

| M/R | host Softmax shape | 源码派生 workspace bytes | kernel UB bytes |
|---|---|---:|---:|
| 128/8 | 16×1024 | 69206016 | 193536 |
| 64/8 | 8×1024 | 42991616 | 183296 |
| 128/4 | 32×512 | 48234496 | 193536 |
| 64/4 | 16×512 | 32505856 | 183296 |

这些静态形状满足 host `%8` 行、`%64` 列及列≤2048约束，资源不超过既有 M128/R8 路径；两种 cross 组合仍须实际 native GetTiling 与启动成功才算合法运行。`CalcUBSize` 本身返回 true，不能声称它执行了充分硬件容量检查。

核心 kernel `flash_attention_score_s1s2_bn2gs1.h` 的 `ComputeBmm1Tail` 依据实际 computed group 动态计算 Vec1/Softmax 形状，主 Softmax 在设备上重新生成 local tiling，不能直接用 host softmax 字段当作实际归约树。M 改变 query block、部分 query 的 computed masked 范围和动态 vector 形状；nRatio 改变 key group 宽度。L1Reuse 的 task 配对还可能影响 computed 范围。query2047 相同不能代表整个序列相同，源码派生计划也不等于硬件浮点指令 trace。

## 私有安装与验证

包安装目标唯一为新 `vendor_stage_a1`，不使用 `--force`，不覆盖系统 SDK。`write_vendor_manifest.py` 在检查冻结源码后枚举 vendor 全文件和安全链接，记录 SHA/bytes；manifest 位于 vendor subtree 外。实际目标：

```text
/home/fault28-causal-20261029/vendor_stage_a1/vendors/fault28_causal_transformer
/home/fault28-causal-20261029/vendor_stage_a1/vendor_manifest.json
```

运行前独立 launcher 必须逐文件验 SHA。相同最终二进制执行所有 profile；`base` 或移除 scope 恢复原生 selector。公开 POD decoder 的 `--schema`、`--self-test` 提供 1464 字节结构与 205 字段布局，其中两个 TCubeTiling 各50个字段使用 SDK `offsetof`，没有手写私有偏移。与实际 raw 完整字节相等的门槛需在 NPU 后另行核验。
