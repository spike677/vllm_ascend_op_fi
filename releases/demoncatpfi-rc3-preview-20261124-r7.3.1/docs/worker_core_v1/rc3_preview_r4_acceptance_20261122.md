# RC3预览r4：视觉及Native增量验收

由负责人转交，Codex不直接联系黄区。r4包含原r3六项及17、41、26 fia、20共十项；33未通过不纳入，8的DeepSeek新代理另行适配。原r3与更早冻结物保持原字节。新增四项仅有A2/CANN9.1/0.26/Qwen3.8社区W8A8/TP2公共验收，不能外推A3或TP1。

## 安装与统一协议

按r4交付单的新标签、三个wheel和source-review.json安装到新目录；用包内 `tools/check_rc3_preview_install.py` 核对版本、全部安装源字节和来源。许可SHA必须取r4 source-review.json原始字节，不能用r3 SHA或release-manifest SHA。

黄区A组卡8～15可分实例并行能力；TP1/TP2独立验收，同一能力两个作用域必须同组Worker。基础参数按[r3参数页](r3_parameter_handoff_20261122.md)：eager/mp/串行、chunked开启、1024token预算与最大长度、seed42、4token、无异步/prefix/MTP/connector。每次最多6次预热，连续两次完整hidden/logits有限且逐位相同后进入normal/fault/off/health。宿主仍使用MultiRankTaskClient continuous=True、可信install_continuous_binding、request_plan与next_phase循环。

以下命令均只在专用实例执行。变量PREVIEW_SITE、SOURCE、RELEASE、MODEL、EVIDENCE_ROOT指向该版本和独立目录；按部署配置映射设备，TP分别取1和2。源包包含可运行入口及构建工具。

## 17：实际视觉请求

image额度1、video0，`mm_processor_cache_gb=0`、`mm_encoder_tp_mode=weights`、min_pixels4096/max_pixels16384。入口使用 `<|vision_start|><|image_pad|><|vision_end|>` 占位符；每次enqueue生成新multimodal_uuid，每阶段都验证真实encoder调用，decode未执行encoder的调用不补造。

黄区固定图片为64×64，已告知SHA缩写 `29148cb0…a7e37`。按负责人后续决定，默认关闭预期图片SHA比对，不要求补齐缩写；仍读取本地固定图片一次供各阶段复用，并记录实际完整文件SHA和RGB像素SHA。该记录只说明本次使用的图，不声称已与此前缩写对应原件核验。需要比对时可以显式提供可选 `--image-sha256`；发行包、源码和Native库校验不受影响。

```bash
PYTHONPATH="$PREVIEW_SITE" python "$SOURCE/experiments/worker_core_v11_20261118/continuous_public_probe.py" \
  --tp "$TP" --mechanism 17 --model "$MODEL" \
  --image "$FIXED_IMAGE" \
  --evidence "$EVIDENCE_ROOT/r4-tp${TP}-17" \
  --source-manifest "$RELEASE/source-review.json" \
  --deployment-source YELLOW_DEDICATED_INSTANCE
```

padding由0,0,0改为0,7,7，真实patch Conv3d消费；输入/权重不改。蓝区每rank每作用域只有1次真实encoder patch调用，不伪称4次。黄区按实际八项计数与消费/恢复判断，不照抄蓝区计数。

## 41、26fia、20：先构建与审查，再验收

先按[A3构建步骤](native_regions_a3_build_20261122.md)在隔离-a3/CANN9.1容器无设备构建。无需patchelf。回传完整构建收据、SDK/镜像身份、库SHA、各rank真实torch_npu设备名；Codex审查通过后设置NATIVE_LIBRARY/NATIVE_SHA256。不能直接复制A2产物或填A2 SHA。运行镜像set_env.sh保证ACL动态依赖可解析。

```bash
for MECHANISM in 41 26fia 20; do
  PYTHONPATH="$PREVIEW_SITE" python "$SOURCE/experiments/worker_core_v11_20261118/continuous_public_probe.py" \
    --tp "$TP" --mechanism "$MECHANISM" --model "$MODEL" \
    --native-library "$NATIVE_LIBRARY" --native-sha256 "$NATIVE_SHA256" \
    --evidence "$EVIDENCE_ROOT/r4-tp${TP}-${MECHANISM}" \
    --source-manifest "$RELEASE/source-review.json" \
    --deployment-source YELLOW_DEDICATED_INSTANCE || exit 1
done
```

|机制|真实写入与恢复窗口|限制|
|---|---|---|
|41|ScatterPaKvCache后改当前decode V槽尾两列/head，真实FIA读完后字节恢复|直接KV prefill不适用；不宣称kernel越界或邻接；MLA未适配|
|26fia|真实FIA数据写当前请求residual的64个元素，实际只读add-rms消费后恢复|消费端原地更新或别名返回拒绝，不覆盖合法更新|
|20|活跃residual内容覆盖投影真实输入64元素，DynamicQuant读完恢复，QMM消费原生Q/scale|输入内容覆盖，不承诺地址槽替换；仅已审查独立W8A8投影|

可信binding接收 `native_library=[path,sha]`，不是公共用户配置。SHA同时进入部署身份。未知源码、实际输入布局、跨stream或不可证明消费时拒绝，不修改指纹绕过。每rank全eligible持续触发、失败0、真实消费者设备事件、字节恢复、完整OFF/health新请求恢复、资源与许可清零，均需完整通过。

## 负向检查与纯文本回传

入口执行缺许可创建拒绝、阶段耗尽advance拒绝、最终资源清零；两个作用域共用Worker。summary由 `tools/review_continuous_public.py <receipt> --require-negative-creation` 输出，原始回执留黄区。回传安装检查全文、summary全文、文件名/字节数/SHA、各rank八项计数、性能倍数、真实硬件原值、异常/完整错误码；17附完整visual_input字段，Native附构建收据身份。

20按调用逐项核对复制/恢复，然后输出站点聚合计数与完整序列SHA，以满足回执容量；不丢弃失败、不把聚合SHA冒充可从空缺原始列表重算。第三方关闭警告与注入资源零分开记录。登记仅为审查后的 `YELLOW_DEDICATED_INSTANCE`（A3、实际软件/权重、实际TP1或TP2），未验收默认关闭。

## 2026-11-23 黄区17启动补充

2026-11-23调整：17验收默认使用内置64×64 RGB图片，像素为 `(4*x,4*y,4*((x+y)%64))`。`--image` 是可选覆盖；产品配置 `canary.image=null` 使用同一内置图，回执记录像素SHA。用户图片仍来自 `/v1/chat/completions` 请求并由实际processor处理，不用金丝雀图替换用户图。r5已通过的黄区固定图片结果仍有效。A3 TP2启动17服务前设置：

```sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

未设置时黄区观测到 `Invalid thread pool`，属于启动环境问题。此补充不修改已冻结r5包；TP1/TP2仍各自验收。完整图像SHA门禁维持原决定，不擅自恢复。
