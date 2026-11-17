# Fault33 当前 Worker 接口核对

124 的实际 vLLM Qwen3-0.6B 推理链完成只读核对，结论是
`NATURAL_API_OBSERVED_INJECTION_REFUSED`。这是自然调用边界的诊断验收，
不是 Fault33 注入、B/F/OFF 或公共 Worker 正向准入通过。

本轮使用一个请求、FP16、eager、两 token 生成。实际首 decode 的 28 层全部调用
`torch_npu.npu_fused_infer_attention_score`，`input_layout=TND`；实际 page 大小为 16，
序列长为 15。每层都观察到真实 `DeviceOperator.reshape_and_cache` 返回，随后真实
FIA 消费同一个 live K/V cache。对虚拟引擎、模型 Attention 对象、cache 的 storage/offset、
实际 block table、scheduler 请求行和 slot 做了交叉核验。当前 token 的逻辑位置为 14，
实际物理页为 2，因此 slot 是 `2 × 16 + 14 = 46`，与真实 writer/metadata 一致。

观察前、观察中、移除观察器后的生成 token 均为 `[576, 4226]`，文本均为 ` The answer`。
这仅证明本次短生成在只读观察器下没有变化，不是完整任务准确率或全量 logits 等值证明。

三项调查分别确定了边界：

| 调查 | 实际结果 | 当前结论 |
| --- | --- | --- |
| 8.5.1 镜像的实际 Attention 源码 | 普通 eager decode 走 TND FIA；PA 分支另受图模式条件控制 | 不能将函数表里有 IFA/FIA 名称当作同一个 kernel 接口 |
| 124 自然 Worker 执行 | 28 层真实 TND FIA；page/slot/writer/consumer 对应成立 | 可接自然缓存诊断；当前缺该接口自己的 native 策略适配及 launch 资格 |
| 现有 9.1 镜像的 Python 环境 | `find_spec(vllm)`、`find_spec(vllm_ascend)` 均为空 | 该镜像尚无可直接运行的同版本 Worker；不表示所有 9.x 部署不支持 |

原 Fault33 的证据保持有效且原件保留：它对应独立审查的 CANN9.1 私有 vendor、
`npu_incre_flash_attention`、BNSD、B8/Q1/KV16384/page128/Hq16/Hkv8/D128。
其 HF 路径将实际逻辑 KV 映射到页缓冲区并另验 launch；这与本轮自然 Worker 的
TND/page16 调用不是同一条已适配路径。不能搬用旧 vendor/launch 回执，也不能在 Worker
中重新构造页缓存后声称验证了其自然 writer/storage/slot。

实现位置：

- [`paged33_audit.py`](../../src/demoncat/runtime/vllm_ascend/worker_extension/paged33_audit.py)：
  按实际模型中 `Attention.impl` 自动发现，无用户层名。只在原 API 外安装所属观察器，
  实际算子参数和返回值不变；未替换 `runner.execute_model`。
- [`paged33_worker_audit.py`](paged33_worker_audit.py)：标准 vLLM 临时 Worker、真实 enqueue ID、
  正常推理及清理。未注册新的注入资格。
- [`paged33_worker_summary.json`](paged33_worker_summary.json)：核对后的精简实机回执、
  原始运行结果 SHA256、实际源码 SHA256 和所有观察到的层路径。
- [`test_paged33_worker_audit.py`](../../tests/test_paged33_worker_audit.py)：5 个 CPU 合约测试，
  覆盖等值复制伪装 live cache、页/slot 不符、缺少真实 writer、错误 K/V offset 及参数漂移。

未证明项保留为未证明：cache 的全部数值内容、实际 kernel tiling/launch、Fault33 策略
命中、统计首差和 B/F/OFF 恢复。本轮不是 B/F 数据采集，不新增或伪造两组 statistics。
后续须为当前 TND FIA 编写并验证独立 native 适配器，或部署实际匹配的 9.1 Worker
调用路径后重新完成源、ABI、launch、无损桥接和注入验收；不按主版本号一概拒绝。
