# Host 驱动的真实请求资格与注入

2026-11-17 增加了实际 serving 请求驱动的 Fault20/Fault39 单目标入口。
它解决同步 RPC 占用 Worker 时无法产生下一真实请求的问题。原同步回放入口和
Native ABI v1 仍保留。验收、失败记录与适用部署见
[124 实验](../../experiments/worker_positive_20261117/README.md)。

## 谁负责执行

标准 vLLM 创建并加载 Worker；安装的 Extension 只管理身份、owner/generation、
资格阶段和原生状态。部署方在自己的临时验收 Worker 内绑定经过检查的 serving
factory。RPC 的 settings 不能携带 factory、Python 代码、层名、地址或任意 slot。
本轮使用 vLLM-Ascend 0.18.0 的 in-process `LLM`，不是一个可以直接控制任意
HTTP 服务的生产适配器。

```text
Host enqueue 返回真实内部 request ID
→ 校验外部/内部 ID 与实际 EngineCore 请求
→ advance(request_id) 返回，Worker 等待正常调度
→ 正常 Worker.execute_model / runner.execute_model
→ 真实模型 root 输入与最后 query hidden states
→ compute_logits 的实际 LogitsProcessor / lm_head 参数
→ 当前原生 API 调用，逐阶段完成资格、注入或 OFF
→ 完整全局词表 logits、同步、原存储与资源状态检查
→ query 回执，Host 发起下一个独立 fresh prefill
```

模型和 LogitsProcessor 使用可移除的自有 module hooks；不会替换 Worker、runner
或模型的核心执行方法。实际 `compute_logits` 调用结构、LogitsProcessor 类型、
Ascend 分派分支、quant_method 身份及完整输入均须符合合同。head 作为真实参数
被观察，不能伪造一次未发生的 `head.forward`。Qwen 和 Llama 的处理不依赖
用户填写层名；新模型是否合格仍由实际调用和 Native 证据决定。

## 生命周期

1. 正常健康请求；确认安装 wheel 的实际导入路径、原生预加载与 Worker 身份。
2. 部署绑定，`attach → prepare → register → check → execute`。
3. 每阶段 `enqueue → advance → wait_for_completion → query`，一次一个请求。
4. `finish → unregister → detach → 移除部署绑定`，再执行健康请求。

`execute` 启动资格过程，不等于已经注入。Fault20 的8阶段为两次 discovery、
reference、lossless、qualification_off、fault、off_observed、disabled_off。
Fault39 的9阶段为两次 discovery、reference、lossless_self、lossless_mat2、
lossless_dual、fault、off_observed、off。每阶段使用新 scheduler ID，固定完整
token/position 输入；旧 ID 不能重用。两种机制都只允许一个实际故障尝试。

阶段之间 Native 暂停且无 inflight/lease，保留同一 owner 和资格。Fault20 的
可选 request ABI v2 区分“paused ownership”和“已注销”；不会以卸载/重装来
重置故障预算。资格后相同目标、完整输入、输出的额外 reference 可以只读运行，
但不能重置已消耗的 fault 次数。Fault39 续接 discovery 只保留同一 generation，
不能从该入口续接注入。

失败会保留原始原因、回执和隔离状态。实际 Run 内的 callback 失败不能在仍
inflight 时强行 drain；待正常执行返回后检查恢复。失败后只有有证据的恢复才
允许清理；不能以清理 owner 代替 OFF 输出证明。Host 只能取消自己真实 enqueue
的请求，Extension 不停止外部任务，也不自动重启 Worker。

## 真实原生边界

Fault20 在 typed GWS 记录当前 formals，绑定实际 executor 与同线程 Execute/Run，
验证活跃存储地址、layout、cache incarnation，而不是使用缓存的旧输入地址。
显式 `DEMONCAT_F20_REQUIRE_TYPED_GWS=1` 的部署针对两种 WeightNz API 关闭上层
PTA 快捷路径，使当前 typed formals 可见；底层 OpExecCache 仍实际执行。
main/nn/math 三个分裂代理和实际动态符号胜出位置都必须通过检查。默认 PTA
快捷路径目前没有资格，不能静默继承 controlled PTA 的结果。

Fault39 在真实 `aclnnMatmul` 调用前核对当前 self/mat2 存储、M/K/N、offset 和
stride；cache 指针还结合 incarnation，避免地址复用混淆。三种无损替换全部通过
后才可执行有限 BF16 临时双输入；真实 kernel 输出单独读回。原输入的完整存储
在 Run 前后校验，temporary buffer 不替代原存储保护。

原生完整输入/输出和 logits 哈希仅用于资格及恢复证明。独立 msprobe 比较器
不读取这些文件，不读取命中日志来指定首差。它只消费两组原始三件套、配置和
采集器身份，以 `|F-B| > 1e-8 + 1e-5*|B|` 比较，保留 null 和不齐结果。

## 当前支持边界

- 实测为910B4/A2、CANN 8.5.1、明确记录的 vLLM/torch-npu 镜像；其他组合另行验收。
- TP1、PP1/DP1、eager、单请求、fresh prefill、同步调度；关闭 prefix/chunked prefill，
  不支持 speculative、multimodal 或异步调度等未经验证的路径。
- 一次自然输出投影是本轮作用域。多个候选、实际调用数变化或请求不一致直接拒绝；
  没有“取第一个候选”或 `max_k` 回退。没有全层、全请求、decode 或 all_eligible 资格。
- Fault20 的 CPU_VERIFIED review manifest 是显式首次实机资格的前提，不能自动改成
  通用 NATIVE_VERIFIED profile。每个新 Worker 仍需完整资格和 OFF。
- 正常 GLM/DeepSeek 推理通过不授予 Native 故障资格。黄区真实 `src/demoncat`
  的 CLI/catalog/bindings 仍需正式接线。
