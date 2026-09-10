# Interactive Inspect 双卡实机验收（2026-09-10）

## 结论

已完成真实「列层 → 看参数 → 有限输入观察 → 候选 → DRAFT → 真实 IDENTITY」链路。

状态：`INSPECT_LIVE_READONLY_AND_IDENTITY_VALIDATED`。

**没有执行有损 FakeRequant smoke**：尚未收到本次有损请求输入。不能写成完整有损集成 PASS，也不能把 reference IDENTITY 命中当作故障数值变化。

| 验收项 | 结果 | 证据边界 |
| --- | --- | --- |
| 双 worker 组合 extension / RPC | PASS | 新会话真实加载，没有热 patch 旧评测服务 |
| 列层 / 参数 metadata | PASS | 每 worker 967 个模块、实际 64 层；4 类投影共 256 个/worker |
| 两次真实请求入口观察 | PASS | 早/中/晚层 12 个代表模块；再独立观察全部 64 个 MLP_DOWN |
| stop 与 TTL 清理 | PASS | 两个 worker handles=0；无请求 TTL 测试通过 |
| Observer 输出透明性 | PASS | 两次观察请求与 clean 请求完整回答一致 |
| 现场量化合同 | PASS，范围限定 | 24 个代表描述：18 个静态 W8A8、6 个未量化方法 |
| 选择集 / 已有 owner | PASS | 本次 64 个候选生成的精确集合等于 owner 安装与实际命中集合 |
| 真实 IDENTITY | PASS | rank0=256 hit，changed=0；rank1=0；恢复精确 |
| 实际 factory CLI | PASS | `ascend_fi.inspect.qwen_r1_adapter:factory`，不是占位路径 |
| 有损单请求集成与表型 | NOT_RUN | 待用户输入，不以其它实验替代 |
| QMM 新 live / module→native | NOT_SUPPORTED / UNKNOWN | recorded 不冒充 live；不根据 shape 拼接 |

## 环境与冻结参数

- Qwen3-32B-w8a8-nopdmix，A2 Ascend 910B4，两卡 TP=2；物理 2/5，对应 logical 0/1。
- vLLM `0.18.0+empty`，vllm-ascend `0.18.0rc1`；torch `2.9.0+cpu`，torch_npu `2.9.0.post1+gitee7ba04`；CANN 8.5.1。
- 保留已有 localhost:18031、eager、max-model-len=4096、max-num-seqs=2、memory-utilization=0.90，不变更 fusion 或前缀缓存配置。
- 原 R1：rank0、全部 64 层 MLP down_proj、`REFERENCE_REQUANT / tensor_absmax / multiplier=0.25 / max_hits=4096 / interval=1`。
- 原 reference 扩展 SHA256：`56097835861bb441232663380eed2953acd97f3aa03c49262368dce6defeaca7`。
- IDENTITY 明确临时使用既有无损 mode，不修改 reference 算法。4096 是上限，本次自然命中 256，不填充请求。

## 实际观察了什么

每 rank 先选层 0、32、63 的 qkv/o/gate_up/down 四类代表模块，独占一条普通请求；随后新开窗口观察 64 个 down_proj。两条请求分属不同 observation session，不把单窗口叫多请求隔离。

全量 R1 观察每 rank 128 个事件（64 模块 × 2 样本）。示例：`model.layers.0.mlp.down_proj` 的输入路径为 `args[0]`，BF16，实测首个 shape 为 `[24,12800]`。`phase=UNKNOWN` 保持原样，不凭 shape 或 invocation 猜 prefill/decode。没有保存 activation 数值、做 D2H 或修改 tensor。

观察预算为 512 events / 2 MiB / 180 秒，实际未因预算耗尽漏掉目标。无请求 TTL 单独用 3 秒测试；显式 stop、超时停止后都检查 handles=0。没有证明 scheduler 级并发请求归属，仅在本次独占窗口成立。

## 重要修正：不是整个模型都采用动态 W8A8

第一轮无损机制通过，但 24 项量化描述仍 UNKNOWN。原因不是绕过了 hook：现场类型与最初的动态 W8A8 假设不同。

| 对象 | 本次证明 |
| --- | --- |
| down_proj | `AscendUnquantizedLinearMethod`；不声称其继承的 apply 已映射到某个 native 算子 |
| qkv / o / gate_up 代表投影 | 外层 `AscendLinearMethod`，内层 `w8a8_static.AscendW8A8LinearMethod` |
| input_scale | 静态 per-tensor activation scale |
| aclnn_input_scale / reciprocal | 由 input_scale 沿输入通道重复得到的派生量，供 quantize 使用 |
| weight_scale | per-output-channel 权重尺度，不等于实际 apply 的 deq_scale |
| deq_scale | QMM 反量化参数；BF16 路径为 FP32；不猜一般 checkpoint 中与 input_scale 的数值关系 |
| 模型尺度 FakeRequant | 未绑定 provider，明确不支持；reference absmax 不受该缺口阻断 |

第二轮用新包 clean restart，24/24 描述匹配完整类名及源码 SHA256。动态 W8A8 合同只保留源码审计范围，不声称当前模型执行它。源文件哈希不是对所有版本/底层 kernel 的泛化保证。

## 原样推理输出

输入：`请只回答结果：12乘7等于多少？`

| 请求 | 完整 response | token pieces | HTTP / finish |
| --- | --- | --- | --- |
| 无 hook baseline | `84` | `8`, `4`, `<\|im_end\|>` | 200 / stop |
| hook 安装但 disabled | `84` | 同上 | 200 / stop |
| IDENTITY | `84` | 同上 | 200 / stop |
| remove 后 recovery | `84` | 同上 | 200 / stop |

这里是 token **pieces**，`token_ids=null`。没有有损响应，不能给正常/故障输出造一组对比。IDENTITY 每次覆盖 64 个目标模块，rank0 256 条完整 history，changed=0，rank1=0；同进程恢复。

## 复现命令入口

只在已授权、已独占、加载组合 extension 的新会话中执行：

```bash
python tools/run_interactive_inspect_live.py --session "$SESSION" --stage inspect-identity

python tools/interactive_inspect.py validate \
  --args "$SESSION/acceptance/validate_args.json" \
  --adapter ascend_fi.inspect.qwen_r1_adapter:factory \
  --confirm-lossless --output "$SESSION/identity_cli_result.json"
```

factory 复用现有 `VerifiedOwner`，依赖原已安装 helper/冻结 manifest；并非通用任意 URL 自动注入器。它不创建锁，拒绝不相等的目标集与 selector，确认所有 worker observer 已停止。`validation_run` 只能是短字母数字标识；重复目录拒绝覆盖。

有损入口已提供 `--stage smoke --prompt ...`，但是本轮未执行。后续必须先用新 epoch 的观察计划通过 IDENTITY，再按冻结配置运行一次真实 B/F/R，复用既有 EvaluationResult。单请求不能给出跨轮稳定性结论。

## 版本与私有证据

- 开始基线 `0dc229e36ede5aecaf181cfc40e8b6ae01bfdbd0`，独立 `feature/interactive-inspect-v1`。
- 第二轮 worker 源码包 SHA256：`2f5c6f27131cac9b5f3fde9028162897104232def5e6598c4104a7c712a90091`。
- 正式 factory CLI 客户端包 SHA256：`7b88e14752c81dc9662de056b1ef49461ab62478b6429058fc84b9a2d2230224`。后续收紧的客户端清理/全 worker observer 门禁未热修改 worker。
- 私有证据：`artifacts/inspect/private/live_20260910_v1/` 与 `live_20260910_v2/`；原始源码副本、metadata、完整 request/response、模型路径、进程与运行库 hash 不提交。
- `acceptance/inspect_session_manifest.json` 保存 worker 包/源码、原 runtime、模型 config、CANN 文件及对应映射证据；`identity_cli_result.json` 保留完整 ValidationResult。
- 第一轮与第二轮分别保存，不覆盖失败/UNKNOWN 历史，不改旧长实验结果。

本地回归 146/146 PASS；原 Research FakeRequant 数学回归 5/5 PASS，原扩展 SHA256 未变。源码改动仅 Inspect adapter、验收入口、必要回归与中文文档；没有改 native、FakeRequant 数学、Phase D ABI 或 Evaluation/scorer。

## 最终清理

2026-09-10 03:07:18 UTC（北京时间 11:07:18）监督脚本记录恢复完成。随后再次只读核对：原无扩展服务 `/health` 为 HTTP 200，Fault20 `enabled=0`，原 FakeRequant 方法不再存在，资源锁 `RELEASED`。不是只凭启动命令返回就判断恢复成功。

最终只读 checkpoint SHA256：`7ce843d04793e23cc07b03033e6be88b12cd256fd96a72c1d02a41790e49bc5a`。独立保存于 `artifacts/inspect/private/live_20260910_v2/checkpoint_final.tgz`；解压副本在其 `final_snapshot/` 子目录。保留早期 checkpoint，不覆盖。

本轮不存在残留 armed fault。不等待用户输入而长期占卡；后续有损请求需要重新合法获取资源，并重新绑定新 epoch。
