# RC3 预览 r3：六项公共持续能力的 A3 验收

由负责人转交，不自行发送黄区。候选为 28 denominator torch、Batch torch、39 output_proxy、18、22、39 matmul_range。33 未通过模型传播验收，不纳入。蓝区证据只覆盖 A2、CANN 9.1、vLLM 0.26.0 / vllm-ascend 0.26.0rc1、社区 Qwen3.8 W8A8、TP2。A3 的 TP1、TP2 必须分别验收、分别登记，不能继承蓝区资格。

## 安装、身份与部署参数

使用本次源包、三个 wheel 和 `source-review.json`；精确提交、标签及许可 SHA 见 `releases/RC3-PREVIEW-R3-HANDOFF-20261122.md`。许可字段 `source_manifest_sha256` 是 **source-review.json 原始字节的 SHA256**，不是 release-manifest.json。在新的空目录安装，保持现役环境和冻结 r2 不变：

```bash
python -m pip install --no-index --no-deps --target "$PREVIEW_SITE" "$RELEASE"/wheels/*.whl
PYTHONPATH="$PREVIEW_SITE" python "$SOURCE/tools/check_rc3_preview_install.py" \
  --site "$PREVIEW_SITE" --release "$RELEASE"
```

回传安装检查全文。安装核验包括发行字节、安装源字节、版本和 import 来源，不创建 Worker、不初始化 NPU，不授予部署资格。

两套 A3 专用实例使用现役 Qwen3.8 W8A8 权重和现役 0.26 镜像；分别 TP1 单卡、TP2 两卡。均为 eager、mp executor、串行请求、关闭异步调度和 prefix cache、不开 MTP、不接 KV connector；纯文本 `limit_mm_per_prompt={'image':0,'video':0}`。开启 chunked prefill，`max_num_batched_tokens=1024`、`max_model_len=1024`；输入由脚本 tokenizer 生成，实际调度必须证明单请求单 chunk。Batch 的 `max_num_seqs=8`，其他项为 1；BF16、ascend quantization、seed 42，4 个生成 token（不提前 EOS）。对外服务原样图模式、异步调度及 MTP 不在支持范围内。

实际模块及 runner 源码由包内审查指纹校验。不匹配时保留完整拒绝码和实际指纹供审查，不修改指纹、复制蓝区身份、补默认字段绕过。若当前 A3 制品不能启动或不能保持基线稳定，停止该组合并回传，不能改用其他模型后宣称通过。

## 可信 binding 与公共宿主循环

可运行入口为 `experiments/worker_core_v11_20261118/continuous_public_probe.py`。`install_continuous_binding(worker, mechanism=..., prompts=..., group_contract=..., arithmetic_device='input', reference_mode='immediate')` 仅安装惰性绑定；组身份来自实际 Worker census。`MultiRankTaskClient(..., continuous=True, world_size=TP, batch_size=...)` 复用现有 Runtime。可信部署使用 `configure_deployment_acceptance` 与 `install_deployment_acceptance_permit`，来源必须是 `YELLOW_DEDICATED_INSTANCE`。脚本的可信 bootstrap 不能暴露为不可信用户的任意 Python RPC。

每项能力内部执行同一组 Worker 的两个独立 owner 作用域。每个作用域先最多六次无注入预热，连续两次完整 hidden/logits 全有限且逐位一致才进入验收。没有达到就拒绝，不使用零初始化掩盖环境缺陷。

宿主读取 `query().response.provider_state.request_plan`，调用 `next_request_phase` 核对 `next_phase`：enqueue → 读取真实内部 ID → advance/advance_batch → 等待真实完成 → query。只按返回计划推进，完成全部阶段且 `complete=true` 才结束；阶段异常不重放故障请求，进入统一恢复。当前计划为 normal/fault/off/health；每请求 4 token，所有 rank 上每次符合条件的 prefill/decode 调用持续触发。

下例中 TP 分别取 1、2，各机制使用新的证据目录，顺序执行：

```bash
for TP in 1 2; do
  for MECHANISM in 28 batch 39 18 22 39mr; do
    PYTHONPATH="$PREVIEW_SITE" python "$SOURCE/experiments/worker_core_v11_20261118/continuous_public_probe.py" \
      --tp "$TP" --mechanism "$MECHANISM" --model "$MODEL" \
      --evidence "$EVIDENCE_ROOT/tp${TP}-${MECHANISM}" \
      --source-manifest "$RELEASE/source-review.json" \
      --deployment-source YELLOW_DEDICATED_INSTANCE || exit 1
  done
done
```

设备映射由专用实例部署设置，TP1/TP2 分别核对实际 rank 与芯片身份。不得向运行中的对外实例发送这些任务。每请求上限 120 秒，超时按失败回传；不自行缩小层范围来补通过。

## 机制边界与覆盖

|机制|本次实现与限制|
|---|---|
|28|full attention 的 torch 分母块重放；block_size=128、块0贡献重复，分子不重复；替换当前请求最后 query 行。不是历史 kernel 的逐字复刻，不覆盖 linear attention。|
|Batch|真实独立 DynamicQuant 的全部调用行共享 absmax/127，交原 QMM；不包含融合量化。scale 精确核对，确定性 Q 样本仅半整数边界允许 ±1；不把样本结果当全量结论。|
|39 output_proxy|实际 BF16 ND 投影输出写 Inf，覆盖量化和非量化输出；新增变化、上游非有限及原值已为 Inf 分开统计。|
|18|AscendGemmaRMSNorm 及已审查融合 Q/K norm 调用，真实下游消费后恢复。|
|22|融合 mRoPE 的目标 rotary 行扰动；summary 包含 rotary 行和 Q/K 输出效果。|
|39 matmul_range|当前仅实际独立 W8A8 QMM 的本地列分片；完整 K 控制值匹配真实输出后，省略末四分之一 K 的累加。非量化分支尚未 NPU 验收，明确拒绝，不借 output_proxy 结论放行。|

社区版 `linear_attn.out_proj` 是量化层，黄区现役版是 BF16。39/Batch 的候选数和命中数可以不同，按真实 eligible/消费/恢复验收，不要求复制蓝区计数。39 matmul_range 必须如实列出非量化分支不适用原因，并至少有真实合格调用；不能将未适配的 BF16 累加路径登记为已通过。

每 rank 输出 observed/eligible/attempted/applied/changed/consumed/restored/failed。eligible 有任何失败即拒绝；无实际命中不通过。必须证明真实消费者读取、同 stream 完成、注入区间字节恢复、资源清零、新 OFF/health 的完整 hidden/logits 与 normal 精确一致。故障请求不要求恢复为从未注入的结果，不要求 token 改变。两个作用域 Worker/model 身份必须相同。

负向检查：无许可实际 prepare 拒绝且无 ticket；阶段耗尽再次 advance 返回 GROUP_PHASES_EXHAUSTED。结束 finish/unregister/detach，再移除许可和 binding，核对许可、租约、tenant、hook、owner、在途请求无残留。异常时保留恢复结果，不能以容器退出当作同进程恢复证明。

## 纯文本回传

每项 TP 组合分别运行，summary 输出完整且不超过 60KB：

```bash
PYTHONPATH="$PREVIEW_SITE" python "$SOURCE/tools/review_continuous_public.py" \
  "$CASE/continuous_public.json" --require-negative-creation
python - "$CASE/continuous_public.json" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);raw=p.read_bytes()
print(json.dumps(dict(file=p.name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())))
PY
```

回传安装检查全文、每项 summary 全文、原回执文件名/字节数/SHA256、rank 八项计数、耗时倍数、异常和完整错误码；原件留黄区。失败时也回传原拒绝信息与恢复状态，不截成“失败”二字。缺失或矛盾不登记。通过蓝区审查后仅登记 `YELLOW_DEDICATED_INSTANCE`（A3、0.26、现役 Qwen3.8 权重、实际 TP1 或 TP2），证据等级为黄区文本证明；工具本身不自动授予资格。
