# r7.3.1 同服务注入效果测试与一页回传

2026-11-24 最新任务：黄区最终包只做一次正式注入测试，结果同时供验收与登记；不要求另跑旧四页提取矩阵。工具不安装许可、不启动 Worker、不注册组合。部署已有的身份、许可、binding、自动请求发现保持有效。

## 运行契约

`tools/product_effect_probe.py` 在同一服务和 Worker 组上创建两个独立 owner。每个 owner 依次发送同一组 prompt 的 baseline、fault、off：temperature=0、seed=42、logprobs=5、4生成token。exponential_perturb 的部署配置改 temperature>0，仍固定seed。关闭调用完成排空、区域恢复和部署金丝雀后才发送off。

API token/top-5 logprob 与完整 logits 是不同证据。工具原样比较 API 返回值，不把 logprob 当作 logits，也不重算缺失的 logits；完整 logits 变化没有真实观察时为null。`off_exact` 比较返回的tokens、text、token_logprobs、top_logprobs规范化字节；这是API字段精确恢复，不是完整张量证明。完整hidden/logits金丝雀由ProductClient原件独立检查。

`changed`是调用次数，允许小于applied；`changed_elements`单独列出。Batch按真实frame去重，不能把每个请求的参与次数相加冒充批次调用数。输出没有大小截断：超过4096字节则拒绝，原件保留。身份缺项原样null，registration_ready=false；一页文本不是自动登记授权。

## 可信部署适配器

部署提供本地Python文件，SHA经负责人审核；普通CLI用户不能传适配器代码。文件导出 `create_adapter(config)`，返回以下接口：

* `identity()`：运行时身份（沿用extract_registration_record.DEPLOYMENT）、mechanism、source_manifest_sha256。
* `new_session()`：同一服务上新owner，暴露owner、worker_identity；attach/enable/status/disable/detach使用现有宿主可信会话。
* `infer(prompts, sampling)`：使用真实HTTP用户请求，返回OpenAI choices。Batch并发发送（max_num_seqs=8）；其他能力遵守实际范围；视觉请求使用内置确定性图或用户图片，不能只发文字替代17。
* `session.evidence()`：从原回执读取各rank八项计数、元素数、请求ID、provider owner、消费者、字节恢复、资源、金丝雀与原件字段路径；不能凭宿主操作成功推断设备效果。

公共ProductClient和外部v1可直接复用 `tools.product_effect_adapter.ProductAdapter(client_factory, infer, identity)`，无需新状态机。它逐项校验原status/cleanup/canary。部署负责给factory提供已鉴权会话和正确的串行Batch金丝雀。示例调用：

```python
from tools.product_effect_adapter import ProductAdapter
def create_adapter(config):
    # 此三项来自黄区现有宿主接线；不能由普通请求提供。
    return ProductAdapter(new_authenticated_product_client, send_real_http, runtime_identity)
```

HCCL预加载路径使用同一工具协议，但必须由黄区适配器读取真实通信回执、跨rank调用与恢复记录。**公共包没有黄区HCCL控制入口，也没有它的A2/A3资格**；不能把公共ProductClient的计数套给HCCL。缺少可审核的映射就报告需退补，不能生成通过的一页总结。v2持久区域尚未纳入r7.3.1。

```sh
# 从解压后的源码根运行；Python>=3.10；PYTHONPATH包含src和源码根。
python tools/product_effect_probe.py --adapter /trusted/effect_adapter.py \
  --adapter-sha256 <审核后的完整SHA> --config /trusted/effect.json \
  --output /evidence/effect-original.json > /evidence/effect-page.json
python tools/product_effect_probe.py --verify /evidence/effect-page.json
# 可在不重新推理的情况下复算一页：
python tools/product_effect_probe.py --summarize /evidence/effect-original.json
```

原件留在黄区；每个机制原样粘贴一段effect-page.json，不手改null、计数、owner或SHA。seq是宿主控制调用序号，明确不是底层wire RPC序号；rank关联仍须依据原始rank/owner/request和Worker身份，不能用seq掩盖队列错位。

## 一次性负例

可信适配器实现 `check_negative(name)`：缺许可、未知名称、已开启时再次开启、重复关闭。复用已有许可安装/移除与CLI入口，不接受工具自行绕过准入。每项返回name、exit_code、code、engine_alive、resources_zero_after_cleanup、positive_control、original_receipt_sha256；重复enable还须original_session_unchanged。重复关闭可幂等成功或结构化exit4，但不得泄漏或破坏引擎。

```sh
python tools/product_effect_probe.py --adapter /trusted/effect_adapter.py \
  --adapter-sha256 <审核后的完整SHA> --config /trusted/effect.json \
  --negatives --output /evidence/negative-original.json
```

该负例只需每个部署/工具组合一次，随正式总结关联SHA。前三项CLI必须exit4且保留拒绝码；直接RPC可HTTP200+结构化body。引擎存活、清理及同引擎正对照必需。模型/kernel异常不是契约拒绝，不能吞掉或自动重试。

## 范围与原件

maintenance实测使用A2/Ascend910B4、0.26、CANN9.1、Qwen3.8社区W8A8、TP2。A2工具验证不等于A3、HCCL或黄区CLI完成。故障本身没有产生新变化（changed<applied）允许；缺失消费者、恢复或eligible失败不允许。若故障导致API不返回可比较logprob，原件保留、结果需退补，不伪造top5。
