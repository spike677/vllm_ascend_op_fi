# 故障 33 · stock 全模型采集预检

124 的 910B4 物理卡 5 执行了 Qwen3-0.6B 的一次完整 prefill 和一次固定输入的 decode，共 8 个请求。28 层 prefill 各调用一次原生框架 SDPA。decode 通过分页适配器调用原生 `npu_incre_flash_attention`，实际日志中有 28 次 IFA launch，与 28 个模型层一一对应，覆盖 224 个请求与层的组合。

msprobe 仅采集 decode 的 step 0。原始 `dump.json`、`stack.json` 和 `construct.json` 均保留在 124 的 `/home/fault33-paged-tiling-20261105/outputs/model_stock_pilot_a1/model_stock_pilot/observation/step0/rank0/`。核查程序从三份原件识别出 28 个 IFA 节点及各自的 Qwen3Attention 父模块。同一份统计自比只用于检验解析与文件绑定。本次预检没有运行故障组，正式 B/F 证据另行归档。

[独立核查报告](review.json)保留原字节，SHA-256 为 `4b9a2bf33c122630909b88c09fe9e612af37f0d582fcf7f1f5566cc3e8127744`。报告记录原始统计与完整日志的 SHA-256、逐层 launch 行号、耗时、NPU 内存统计和清理结果。
