# r7.4.1 最小路由改动与验证边界

状态：开发候选，未冻结、未授予新的部署资格。基线为已冻结 r7.4 `abd40cdc`，本分支不修改任何既有发行目录或标签。
负责人已明确批准：18 方差误差命名变体及外部 Conv3d 图片请求的最小路由改动，并要求列明影响项、复测。

## 18 命名变体

`18.variance_over4`：对实际命中的 BF16、连续存储归一化输出全部元素乘以固定 `0.5`。
这是 RMS 均方统计扩大 4 倍时的**输出增益效果代理**，不是原 kernel 的方差计算复刻。
RMSNorm 不是减均值后的方差。原分母含 epsilon 时，精确比值是
`sqrt((mean(x²)+epsilon)/(4*mean(x²)+epsilon))`，不恒等于 `0.5`。
新变体明确记录 `epsilon_exact=false`。用户不能传 gain、层名或 rank。
历史公共17项的编号口径不变；catalog新增一个18的命名变体后，条目数为18，不能把它称为第18种独立故障。
该变体不加入历史15项 ModelFaultBackend 适配清单，避免静态目录误宣称旧宿主已支持。

复用原 `Norm18Site` 的实际模块发现、融合 Q/K 调用识别、源码指纹和作用域。
新消费者模式只在真实非视图运算读取完整改写区域后完成租约；部分读取、覆写受保护区域、非连续输出均明确拒绝。
拒绝发生后保留未完成租约供失败恢复审查，不将其记作成功跳过。
`changed` 是变化调用数，`changed_elements` 是字节表示不同的元素数。
`18.api_half` 的单元素算法文件不变，历史结论不继承给新变体。

## 外部图片路由

受信部署显式注册 `ExternalSite(..., stage='vision', region_kind='transient')`。
注册本身仍要求适配器源码 SHA、完整部署身份和部署侧授权；注册不授予资格。
`stage=model/sampler` 保持纯文本规则；不能以用户图片自动切换 stage。

新 vision 路由复用原视觉部署核验和 ProductRequest 的全部检查：

- 一条当前请求；一张 64×64 图片，实际 grid `[1,4,4]`；不接受视频。
- 当前调度的 encoder 输入、实际 feature、请求位置和 fresh cache 相互对应。
- encoder 输出进入真实 encoder cache，随后同存储 embedding 被模型消费。
- 外部适配器必须在 encoder 返回后完成其自身真实消费者证明和区域字节恢复。
- 外部 vision 报告恢复类型是 `LIVE_REGION_BYTES`；只有原17继续报告参数恢复。

示例 `external_vision_example.py` 发现真实 `visual.patch_embed.proj: Conv3d`，读取输入与权重、仅改写该调用的临时输出，核验输入与权重未被修改。
示例只演示输出增益，不是黄区低精度卷积或转置卷积机制，也不授予黄区适配器资格。
持久权重或 KV 不能通过 transient vision 路由绕过 v1 限制。

## 影响项及复测

|改动位置|影响|要求|
|---|---|---|
|catalog/delivery_scope/product_contract/registry|增加18命名变体的查询、配置|旧ID行为保持；无绑定、无许可仍拒绝；无用户数值参数|
|continuous_binding|新18工厂；显式vision调用原视觉验证|旧18、22、39op、39mr、28、Batch、17、15、41、26fia、20的公共构造与回归|
|product_request|新18收尾；外部vision请求与字节恢复|文本/采样拒绝图片；视觉映射与消费者失败拒绝；原17恢复类型不变|
|product_binding|vision能力查询展示真实image_grid|query不授予资格，阶段由部署注册决定|
|external_provider|允许受信vision stage|未知stage、持久区域、错误源码/部署、无许可等负例仍拒绝|
|新FullNormConsumerMode|完整区域消费者核验|完整/转置读取、视图、部分读取、原地写、0变化、字节恢复|

额外诊断入口 `DiagnosticProductClient` 复用原 ProductClient 控制逻辑。
状态失败时保留实际收到的每 rank cursor、完成数、active、请求ID与计数；不额外RPC、不重试、不修正回包，不掩盖原异常。

## 实机队列

组合：124 A2 / 0.26 / CANN 9.1 / Qwen3.8 W8A8 / TP2，同组 Worker 两个 owner。
2026-11-24 本轮只读检查：卡2无计算进程，卡5仍有 ATK 进程占用。依最新任务“NPU被占用先做别的项”，本轮未停止该任务、未启动NPU验收。

在现有隔离验收容器的 `/source`、`/model`、`/evidence` 挂载约定下：

```sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python experiments/worker_core_v11_20261118/product_http_probe.py \
  --mechanism 18 --variance18 --status-diagnostics
python experiments/worker_core_v11_20261118/product_http_probe.py \
  --mechanism 17 --external-vision --status-diagnostics
```

两条命令必须使用不同 evidence 目录。脚本创建独立临时 Worker，经 HTTP 自动发现请求，执行两个 owner、范围负例、关闭及金丝雀；图片使用内置确定性图。
原始回执先运行 `python -m tools.review_r741_routes <receipt.json>`；它先执行通用产品审查，再核对整输出范围/固定增益声明或图片消费者与只读输入权重。
这只是待执行入口，不是实测证据。必须补齐原公共项回归、NPU消费者/恢复、发布源码一致性和无设备安装检查后才可冻结。
本轮尚未完成模块输入、dispatch参数等其他 B2 示例、V2-Lite router 两owner、A3 33取证以及 r8 新登记整合，均不因本次CPU通过而标为完成。
