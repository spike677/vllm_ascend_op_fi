# 33 arch22 S2：新包 normal 后的公共单点 C1 实验

本增量只接入已审的 `model.layers.0.self_attn.attn` 实际对象、其真实祖先调用和首 decode：FP16、TND、Q=1、KV=1025、Hq=16、Hkv=8、D=128、page=16。用户只选择 `FaultConfig(33, 'paged_decode')`，不能填写层名、callback、Native 路径或 proof。可信 deployment 从实际模型结构确认对象；首层是当前实验适配范围，并不是整个模型的 eligible 分母。

每个请求保留 28 次 prefill + 28 次 decode 原调用，所有原调用恰一次。其余 27 个 decode 即使满足相同内核输入条件，仍标为“缺逐目标 logical normal/effect proof”，保留 `input_contract_matches`；不改为不 eligible，也不据首层通过授 C2。C3 必须等待完整 C2 的实际活对象签发和使用。

Native 算法和故障定义沿用 source94 已编译的新 S2 包：在原 SplitCore 之前将 S2 512 改为 256，M 保持 512；所有完整 split/FD 数组、workspace 和最终 POD 均与实际编译的 22528 行 CPU 矩阵连接。此增量不改 C++ recipe/header/kernel，不要求重编同字节 Native 包；它需要最终同源 Python 安装包和 observer，依序执行新包 stock/private normal → 原件审核 → 公共 R/L/OFF/F/OFF。旧 source88 normal 和 source87 census 不重跑、不继承为新包证明。

## 真实入场和恢复

`VerifiedNormalEvidence` 只能由可信 deployment 重读新包两臂原始结果、435 个完整 payload pairs、scope/runtime/registered ELF/最终 POD 后创建；saved PASS 只是文件来源。此对象不可序列化，不从 RPC 接收。运行时再次验证相同模型、源码、安装包、runner、sampling、环境和原始 first-layer 输入。

Reference、lossless、qualification OFF 是同一公共 Runtime private owner/generation 下的三个真实请求。每次保存活 Worker/model/module/impl/cache/input 对象，完整 Q、mask/page metadata、有效逻辑 KV 前后哈希和实际原调用返回。只允许已证明的有效页号重定位及对应表 hash/slot 差异；unused 页、参数、Q、有效 KV 任一改变都拒绝。三次实际 Native tokens、request IDs 和完整输出行共同生成当前进程 gate；F 必须仍持有这些对象和 gate。无全局 execute_model hook；Native 开关仅在实际匹配 API 调用的 TLS token、PID/native TID、owner/generation、phase 范围打开。

五个协议请求分别为 `reference/lossless/qualification_off/fault/recovery_off`。每请求写 87 份完整 payload（hidden、实际 logits 输入、logits、28 层 Attention 输出与有效逻辑 K/V），另保存完整首层 Q 和即时两返回 tensor。F 分别报告：实际 S2 切换、即时首层输出变化、hidden/logits 传播。首层输出有变化而 logits 无变化时，保留单点算子效果，末端传播记为未观察；不新增“logits 必须改变”的机制门槛。策略切换成功但首层输出仍 exact 时，保留选择事实，数值故障 C1 不授。独立 reader 另区分位变化与数值变化。

OFF 必须是同 Worker 的新普通请求，所有 87 个完整 payload 精确恢复。Python 持活输入按 `retained_input_objects/owned_input_references` 单独计数；活动 Native lease 仍按真实 scope/alloc 计数。最终释放前必须实际 drain/stop、Native scope/env 归零，确认 OFF，再清自有 hooks/API/输入强引用。foreign thread/wrapper、未确认 stop、未完成 OFF 都保留公共 owner；所有第一次错误永久保留。

F 不重试。不确定 F 回执只能通过显式公共 `recover` 归档失败请求并开放新 OFF；不确定 OFF 可在实际零 Native scope 后保留旧输入 ledger，重试一个新 OFF。失败路径永不回写为通过。探针最多自动补一次 OFF；之后仍失败就保留 owner 和原件。正常执行 warm + 五协议 + 注销后健康共七请求；若计划 OFF 失败而补 OFF，共八请求。注销后健康只证明 token/text，不宣称 hidden/logits 已观察。

阶段名 `lossless` 具体表示第二次 S2=512 normal-plan 的完整无损重复（原 baseline-repeat），不是执行 S2=256 后覆写或回滚返回输出；R/L/OFF 的实际 Native 回执均保留 512，F 唯一使用 256。

## 主控接线与输入闭包

仅由主控修改共享 registry：把 `33.paged_decode` factory 指向
`demoncat.runtime.vllm_ascend.worker_extension.providers.paged33_s2:resolve_paged_decode_provider`。
它只对 exact `Arch22S2EffectsNativeCall` deployment 选择新 Provider；原无 Native census 仍回落到既有 `PagedDecodeServingProvider`。另三种 28 变体不受影响。

新增生产模块：`paged33_s2_evidence.py`、`paged33_s2_execution.py`、`paged33_arch22_s2_effects.py`、`providers/paged33_s2.py`。新增根控工具：`natural33_arch22_s2_worker_core_probe.py`、`prepare_arch22_s2_single_cases.py`、`review_arch22_s2_single.py`。完整显式闭包是 `arch22_normal_bridge_evidence.REQUIRED + S2_REQUIRED`；包含 shared normal reader、Native 原编译源、公共 Host/Runtime/registry/extension 和调用适配依赖。冻结和 wheel export 必须包含这份闭包；reader 同时核对实际安装包完整文件清单。不得把 root 工作树路径或原始实机数据加入包。

所有路径以下均由主控填写真实新 source/package 和新 normal 原件身份。`BUILD` 仍是已完成的新包 `.../tnd33-arch22-s2-build-a1/build/run`。同源 normal 准备和审核命令见 [normal 文档](attention33_arch22_s2_normal.md)。先完成它的新包两臂，而后执行以下纯 CPU 准备：

```bash
python3 -B "$SOURCE/experiments/c2c3_closure_20261117/prepare_arch22_s2_single_cases.py" \
  --source "$SOURCE" --source-manifest-sha256 "$SOURCE_SHA" --installed "$INSTALLED" \
  --build-root "$BUILD" --completion-sha256 "$COMPLETION_SHA" \
  --observer "$OBSERVER_RESULT" --observer-receipt-sha256 "$OBSERVER_SHA" --model "$MODEL" \
  --stock-run "$NORMAL_STOCK" --stock-sha256 "$NORMAL_STOCK_SHA" \
  --baseline-run "$NORMAL_PRIVATE" --baseline-sha256 "$NORMAL_PRIVATE_SHA" \
  --normal-report "$NORMAL_REVIEW" --normal-report-sha256 "$NORMAL_REVIEW_SHA" \
  --card 5 --timeout 1800 --out "$OUT/cases.json"
```

准备器只写一个 case，不启动设备。主控追加与两 normal 臂完全相同、已审核的生产同字节库只读 mounts（含真实 aarch64 optiling/master）；不可弱化 map dev/inode/hash 检查。准备器保留 `/native` loader alias，另把原 build 顶层同路径只读映射供新包 source/patch/matrix 原件审核；stock/private normal 原件和审核报告均只读映射。主控唯一队列运行该 case。产物位于 case `/evidence/probe`：result、独立逐 RPC JSON、完整 Host history、Native host/runtime/registered ELF/每 token 原始回执及所有原始输出 bins。

实机后只读审查：

```bash
python3 -B "$SOURCE/experiments/c2c3_closure_20261117/review_arch22_s2_single.py" \
  --source "$SOURCE" --source-manifest-sha256 "$SOURCE_SHA" \
  --build-root "$BUILD" --completion-sha256 "$COMPLETION_SHA" \
  --observer "$OBSERVER_RESULT" --observer-receipt-sha256 "$OBSERVER_SHA" \
  --stock-run "$NORMAL_STOCK" --stock-sha256 "$NORMAL_STOCK_SHA" \
  --baseline-run "$NORMAL_PRIVATE" --baseline-sha256 "$NORMAL_PRIVATE_SHA" \
  --normal-report "$NORMAL_REVIEW" --normal-report-sha256 "$NORMAL_REVIEW_SHA" \
  --run "$SINGLE/probe" --result-sha256 "$SINGLE_RESULT_SHA" --out "$REVIEW/review.json"
```

reader 不导入 injector/torch/DSO，只在永久禁止动态加载的 audit hook 之前初始化已记录来源的 CPU NumPy 字节读取。它核对完整公共 owner namespace/phase、原始 RPC 链、三次实际 normal gate 签发使用、五个 Native call/host/launch/ELF/POD、每请求 56 次原调用、435 份完整输出和 OFF/注销/健康。失败原件保留，C2/C3/E5 始终不授。

## CPU 验证与停止条件

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:TORCH_DEVICE_BACKEND_AUTOLOAD='0'
$env:PYTHONPATH='src;fault39/src;.;tests;tests/worker_extension'
& $PY -B -m unittest test_arch22_s2_effects test_core_paged33_s2 test_arch22_s2_probe test_arch22_s2_single_review test_arch22_s2_normal test_arch22_normal_bridge_review test_paged33_logical_binding test_core_attention_serving test_attention28_stock_census test_attention28_review_protocol
```

CPU 中用真实 torch Module、API 原调用和公共 Host/Runtime 完成五请求，但 Native 是明确的模拟 seam，C1/C2/C3 必须 false。负例覆盖错误 owner/phase/final POD/完整数组、页表和 unused 页/数据变化、F 不确定回执、OFF 不确定回执、foreign thread/API、清理失败重试、策略无数值效果、末端 logits 不传播、原始 RPC 篡改及冷进程 reader。CPU 不替代实机。

新 normal 任一原始失败、跨臂 payload 不 exact、当前实际输入不相等价或 source/build/observer 变化时，在 F 之前拒绝。F 发生后，无论数值效果或资格如何均须实际 OFF；OFF/清理不确定则停止并保留 owner。C2 后续单独扩展另外 27 个实际对象的 normal/logical/native/effect 证明，再做全目标一次；C3 必须复用该完整 live C2 proof 做全目标连续两次。此提交没有将这些缺失项改为不可适用。
