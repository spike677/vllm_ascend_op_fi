# 自然 FIA 输入 journal：默认关闭的接线候选

本次接线只观察现有 stock TND FIA 路径，不注入 F，不增加公共 Provider、registry、Core 接口或全局 `execute_model` 包装。`StockTndPrefillCensus(..., input_journal=None)` 保持原默认行为；只有新实验探针显式传入经过预启动解析的、确切类型的 `NativeFiaInputJournal`。当前交付是 **CPU 接线合同**，尚未运行这份接线的 NPU 用例，C1/C2/C3/E5 全部为 false。

原 journal 的 ABI、同 owner 资源合同和未知 lineage 边界见 [原候选说明](fia_input_journal.md)。这份新增说明不追改旧候选或 source87 的原件，也不把 source111 的无设备构建升级为实际调用证据。

## 固定运行范围与顺序

实验限于已冻结的 CANN 8.5.1 镜像、Qwen3-0.6B、TP1、FP16 eager、单请求 2048-token prefill。使用普通 vLLM 启动的、同一进程和原生线程中的真实 Worker；另一个进程或 RPC 线程不能重建 journal 冒充预启动对象。配置中的 `28.fia.denominator_replay` 只是已有公共观察入口的机制标识，此探针没有 replay/fault 阶段。

固定计划为 **1 次普通健康 + 2 次观察 + 1 次普通健康**。两次观察沿用 `observe` / `observe_repeat`，每次预期 28 个实际层调用，共 56 个 public API 窗口。原请求映射、scheduler、实际 module 对象和调用分母保留在 Core 原件，不按名称合并，不遗漏不支持的调用。

1. 子进程由 `LD_PRELOAD` 加载实际 observer 和 journal；`journal.preflight()` 在导入/构造 LLM 前执行 NOLOAD 解析及原函数 owner 校验。启动门禁拒绝时不构造 LLM。
2. 标准 Worker RPC 核对同一 PID、process-start、原生线程，以及实际 Worker/model 对象，绑定预启动保存的同一 journal。先执行普通健康请求，再 begin 观察。
3. 精确 `StockTndPrefillCensus` 建立原有 launch scope，使用该 scope 的真实 UUID token；随后 journal.enter，执行原 public FIA 恰好一次，使用原有完成同步，然后 journal.leave。新增代码不添加设备同步，不将 V `contiguous()`，不复制/重排或替换 Q/K/V/mask，不改变 dtype、输入或算子参数。
4. 逐次 `census.stop()` 只结束当前窗口。journal 不会在第一层后全局停用；公开 query 不查询 journal，也不额外同步设备。
5. 两次实际请求完成后，公共 finish/disable/unregister/detach 清理 owned hook/wrapper/lease，精确移除本实例的 deployment。之后才执行 journal.stop、保存 observation、release、保存释放回执，最后通过普通 enqueue 在同一 Worker 上执行健康请求。

窗口内 `forward=1/2` 只是两个观察 phase 的关联索引，operator sequence 来自实际 ordinal+1，boundary sequence 是实际 census 顺序；它们不冒充模型 decode 步数。完整原输入引用一直保留到完成同步和窗口关闭。原 V 的非连续 stride/offset 通过真实对象及 storage/data 地址记录，不能用新分配的同值对象替代。

## 预算、失败和清理

仅一个 queue case、单张卡 2 或 5，最长 1800 秒，48 GiB / 8 CPU；journal 上限为 64 个窗口、128 个 Native 调用、64 个 pending pair、256 个事件和 16 MiB 事件缓冲。没有清空历史腾空间，也不重放失败请求。

原 API 异常先保存并保持为主错误。窗口开始/同步/leave 失败时，保留原输入引用、实际 owner、pending 和失败记录；不 reset 未知 executor。公共 Core ownership 未清除或实际 deployment 尚在时，禁止退休 journal。释放失败可进行一次有界清理重试，第一份原始文件不覆盖；不会因此补一个健康请求或把失败改成成功。全局预加载 wrapper/DSO 保留到独立进程退出，不宣称全局卸载。

公共 owner 因 pending 保留时，实验另外尝试**一次只读失败快照**，不执行 stop/leave/release、`_phase` 或移除绑定。它先核同一 held PID/原生线程/Worker/model、实际映射身份和两组件真实 owner/generation；只有实际 Native `active_calls`、metadata `active_captures/active_getters` 均为零，且前后两组件完整 state 一致，才保存事件。pending executor、scope、live registration 可以非零，原值保留，不解引用 pending executor，也不据快照授予清理权。查询、身份或稳定性检查失败时另存明确拒绝原件，不能用默认零值代替未知状态。

观察请求取得实际 enqueue 映射后立即保存 `completed=false`，wait/collect 失败仍保留该请求与 phase；完成收集后才改为 true。后续 host receipt、最终 result 写盘等错误不会覆盖首个 API/protocol 错误，而记录为 secondary error。若存储本身持续不可写，不能保证新文件落盘；终端退出信息仍保留首错并披露次级写盘错误。

生产新增 FIA GWS/execute 的异常透明合同不扩展到旧四容器 TU：旧 creator/destroy 对原 C++ 异常的 catch→`nullptr`/`161001` 行为仍按原说明披露。本接线没有修改任何 Native 源码或 ABI。

## 原件与独立审查

探针在私有 `/evidence/probe` 下写入：

| 文件 | 内容 |
| --- | --- |
| `input_journal/startup-before-llm.json` | 实际预启动 owner/函数/进程身份，先于 LLM |
| `worker_core_rpc/*.json`、`host_receipt.json` | 完整原公共请求/RPC/生命周期和实际请求映射 |
| `native_calls/<call_id>.json` | 原 stock census 输入前后证据、原调用次数、真实 launch scope 和新增 public window 关联 |
| `runtime/*` | 原 launch observer 记录；不假造 stock POD 或未捕获入口 |
| `input_journal/observation-01.json` | 停止后的 Native/container/public-window 原件及生命周期 |
| `input_journal/released-01.json` | 同 owner 实际释放后的两组件状态；只有释放返回后才保存 |
| `input_journal/cleanup-01.json` | 清理阶段和原错误；有界重试使用 `02` 新文件 |
| `input_journal/failure-observation-01.json` | owner 保留时允许读取的只读部分失败快照，不是清理成功证据 |
| `input_journal/failure-snapshot-01.json` | 上述只读快照的实际状态/引用，或查询、owner、active、稳定性拒绝原件 |
| `result.json` | 原件引用与执行/清理总结，始终不授资格 |

`RECORDED_PENDING_INDEPENDENT_REVIEW` 只说明协议记录完成。探针核对 56 个窗口和实际 Native 计数，不用该状态替代独立逐事件审查。复核应分别连接 public call→真实 token→GWS descriptor/creator→execute→release，并重算各原件 SHA。现有纯 stdlib `fia_input_journal_review.py` 可审 observation/released；它不导入注入器，不把 saved success 当资格。

```sh
python /review/fia_input_journal_review.py \
  --observation /evidence/probe/input_journal/observation-01.json \
  --observation-sha256 ACTUAL_OBSERVATION_SHA256 \
  --released /evidence/probe/input_journal/released-01.json \
  --released-sha256 ACTUAL_RELEASE_SHA256 \
  --build-root /actual/read-only/fia-journal-build \
  --build-sha256 ACTUAL_BUILD_SHA256 --out /review/new-input-summary.json
```

**不能据此证明内部 prepared tensor、Slice/copy 血缘或输入数值的完整传递。** Slice ABI 未提供，保持 unknown/false；本次没有 msprobe/statistics 比较或 E5。健康对照只比较两个观察请求与前后健康的生成 token，不宣称已独立比较文本或 logits。所有真实地址、完整张量及原始运行材料留在私有证据目录，不随代码提交。

## 纯文件准备与导出闭包

Root 冻结新 source、导出安装包并复核门禁后，准备器只读文件并生成一个 queue case，不执行 SSH、Docker 或 NPU：

```sh
PYTHONPATH=/new-source:/new-source/src:/new-source/fault39/src \
python /new-source/experiments/worker_core_v11_20261118/prepare_natural_attention28_input_journal.py \
  --source /new-source --source-manifest-sha256 ACTUAL_NEW_SOURCE_MANIFEST_SHA256 \
  --installed /new-installed --model /actual/Qwen3-0.6B \
  --observer /actual/observer-build --journal /actual/fia-journal-build \
  --journal-build-sha256 ACTUAL_BUILD_SHA256 \
  --card 5 --timeout 1800 --out /new-attempt/cases.json
```

必须保留 source111 编译时的 observer 依赖别名 `/observer/libdcat_attention_observer.so`，以同一已核验目录的只读 bind mount 提供；SHA 为 `bcad8a17191db79b1ab0c89a110ff557b3a3b4243ff8d8f2fe353beeeb3a908b`。准备器重哈 bundle、profile、receipt、26 条 CPU 日志、DSO 及所有文件，要求新的 runtime 对应源码与编译 profile 相同。它不改 build 原件、不借用旧 source 身份，不重复挂载 queue 自带的 `/msprobe_packages`。

新增导出项为本探针、准备器、本文档及两份测试。运行依赖闭包由准备器 `EXTRA`、旧 `prepare_attention28_stock_census.REQUIRED` 和实际 Native `profile.sources` 三者合并；安装包中对应 Python 文件必须与冻结源逐字节相同。没有修改共享 exporter，Root 集成时需把新增文件纳入导出。当前候选对照 source111 profile 的 17 份源文件全部相同，因此不要求重写旧 SDK 原件；是否复用同一实际 DSO仍由 Root 完整重哈和新进程 preflight 确认。

## CPU 验证范围

```sh
PYTHONPATH=src:.:fault39/src:tests:tests/worker_extension:experiments/c2c3_closure_20261117 \
python -m unittest test_natural_attention28_input_journal_probe \
  test_attention28_input_journal test_attention28_stock_census -q
```

本次本地 **34 项通过**，包括旧默认 stock 行为、确切类型/对象拒绝、真实非连续 CPU view、不增 query/sync、原异常/pending 清理、真实公共 Core 的 2×28 CPU 调用、实验 driver 的实际顺序、绑定未移除/公共 ownership 未退时拒绝释放、失败请求和释放重试原件保留，以及纯文件预算/alias/完整 artifact SHA/pre-LLM 拒绝合同。CPU 模型、scheduler、SDK return code 和最终 Native snapshot 中的模拟边界在测试中明确标注；这不是 Qwen/NPU 验收。

最后联合运行上述 34 项与既有 `test_fia_input_journal` 的 37 项，共 71 项通过。二者合计是这次本地测试命令的执行数，不与历史测试重复累计为新的独立覆盖。

独立审查后的失败取证增量再增加 8 项：实际 Core retained+pending 下的生产 Python snapshot 读取、Native 活跃调用、活跃 getter、查询拒绝、foreign owner、完整 container state 漂移，以及 host/final-result 两处 secondary 写盘错误不替代主错误。原失败请求测试同时检查未完成 request mapping 已写入原件。这些测试仍使用明确标注的 CPU Native state/event 返回码接口，未执行 NPU。

增量后的相同联合命令实际运行 **79 项，通过，0 skip**；原 71 项及其日志保留，没有改写历史结果。

此前 Root 报告 source111 独立 SDK 无设备构建的 38 项 Python/导出与 26 项生产 TU C++ 合同通过，属于另一组构建证据，不与本次 34 项相加宣称独立覆盖，也不作为本接线已上 NPU 的依据。
