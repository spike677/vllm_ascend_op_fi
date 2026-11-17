# FA 原生桥接 v1 契约

桥接使用 `NES_FA_PLAN_JSON`（UTF-8 JSON，不超过 16 KiB）及 `NES_FA_LOG_PATH`（本次证据文件绝对路径）。原生库不会读取模型张量内容，不会修改 raw tiling；它只在固定源码的 S1S2 `CalcNRatio` 末尾同时选择 M/R，后续原生代码重新生成所有依赖 tiling。没有按 B8 触发的分支，也没有命中次数上限。

以下为字段结构；SHA 均为 64 个小写十六进制字符，由注册表和启动器验证所引用材料，而不是由字符串本身证明材料真实。`experimental` 明确表示新桥接尚待实机验收。

```json
{
  "schema": "nes.fa.typed_plan.v1",
  "operator": "FlashAttentionScore",
  "mode": "controlled",
  "scope": {
    "pid": 123,
    "native_tid": 123,
    "token": "call_unique_001",
    "phase": "operator_execution",
    "request_count": 1
  },
  "context": {
    "query_shape": [1, 16, 2048, 128],
    "key_shape": [1, 8, 2048, 128],
    "value_shape": [1, 8, 2048, 128],
    "mask_shape": [2048, 2048],
    "dtype": "float16",
    "mask_dtype": "bool",
    "layout": "BNSD",
    "scale": 0.0883883461356163,
    "keep_prob": 1,
    "pre_tokens": 2147483647,
    "next_tokens": 0,
    "heads": 16,
    "inner_precise": 0,
    "sparse_mode": 2,
    "optional_inputs_absent": true
  },
  "version": {
    "source_commit": "c77c0829f1f0f0b8f882db1d9b051a758b605472",
    "cann_build": "V100R001C11SPC001B243",
    "bridge_abi": "nes.fa.cann9.1.s1s2.v1"
  },
  "requested": {
    "native": {"m": 128, "n_ratio": 8},
    "selected": {"m": 64, "n_ratio": 8}
  },
  "certificate": {
    "id": "fa_cann91_experimental_a1",
    "source_manifest_sha256": "<64 lowercase hex>",
    "build_manifest_sha256": "<64 lowercase hex>",
    "validation_sha256": "<64 lowercase hex>",
    "validation_state": "experimental"
  }
}
```

`native` 模式要求 `requested.selected=null`；`requested.native` 可为 null（发现）或此前实测的整对（检查当前默认选择是否一致）。它允许任意正的显式 request_count 观察；本版其余形状、dtype、layout、mask 和精度属性仍固定如上。C++ 核对真实 QKV shape 是否与显式 request_count 一致，不把第一维自行解释成请求数。

`controlled` 模式要求 native/selected 两对均显式给出且不同，只接受本版范围内 B1/B8 以及 M∈{64,128}、R∈{4,8}。这是已知候选范围，不是新的构建验收证明。B2/B4 可 discovery，但不能因此取得 controlled 支持。C++ 会核对本次原生选择与 requested.native 后原子替换；如果任何条件不合格，两个参数都保持原值。策略标识由 Python 适配器管理，host 只读取 typed 参数。

只有 PID、native TID、phase、token、版本、证书结构、上下文及原生参数全部匹配，才允许 controlled 选择。native 与 controlled 使用同一库和观察逻辑。缺失/错误计划保持原生执行，已进入观测 scope 的拒绝原因写入记录；不得以缺失记录当成功。

证书真实 SHA、注册表认可状态及完整新 vendor 文件 SHA 由启动器/适配器验证，host 核固定源码 commit、ABI、字段及候选边界并原样记录证书。本桥接不内嵌自身最终 SHA，避免自引用；旧构建的验收结果不能给新构建自动授予 validated 状态。

输出为 `nes.fa.host_observation.v1` JSONL：完整 plan、scope 身份、context guard 报告、同次调用 `native_factors`、`selected_factors`、`selection_applied`、`selection_reason`，以及 callback_status、实际 raw/key/block/schedule/workspace、原生 typed 字段（含两组各50项 TCube）。选择成功只表示策略选择被执行，不代表实际策略指纹改变、数值变化或模型效果；这些结论由独立观察前后数据判断。
