# RC3 预览：28 / Batch 公共持续入口

仅交负责人转交，Codex不向黄区发送消息。目标是A3上Qwen3.8 W8A8专用实例、0.26栈、TP2；当前蓝区结论不授予A3资格。不是RC3完整12项发行。冻结RC1/RC2物不变。

## 安装与身份

在隔离测试环境安装预览的三个wheel，保持现役环境不变。发行`release-manifest.json`列出版本和文件SHA。许可`source_manifest_sha256`取**source-review.json原始字节SHA256**，不能取release-manifest SHA。

```bash
python -m pip install --no-index --no-deps --target "$PREVIEW_SITE" "$RELEASE"/wheels/*.whl
PYTHONPATH="$PREVIEW_SITE" python "$SOURCE/tools/check_rc3_preview_install.py" \
  --site "$PREVIEW_SITE" --release "$RELEASE"
```

`PREVIEW_SITE`必须是空的新目录。`SOURCE`是本次源包解压目录，`RELEASE`是发行目录。核验器逐个校验发行文件和安装后的Python/JSON源字节；不创建Worker、不初始化NPU。安装检查不代表推理验收。

实际软件源审查使用包内`v026_installed_source_review.json`及各site源码指纹。A3镜像与蓝区的文件若不匹配，必须保留拒绝并提供摘要供审查，不能复制蓝区镜像标签、修改hash或把缺失字段当默认值。当前普通绑定只支持已经审查的Qwen3.8真实纯文本路径；不表示适用于所有0.26镜像。

## 可信宿主绑定

公开用户配置只接受故障名，不接受rank、层名、地址和源码许可。部署侧使用：

|用途|接口|要点|
|---|---|---|
|组客户端|`MultiRankTaskClient(..., continuous=True, world_size=2, batch_size=...)`|复用既有Runtime；28 batch_size=1，Batch=8|
|范围约束|`fault_injection.rc3_preview_profile.prepare_binding`|只允许28 denominator与Batch，且必须持续组|
|可信安装|`install_continuous_binding(worker, mechanism=..., prompts=..., group_contract=..., arithmetic_device=...)`|输入为真实token ID；组成员来自实际Worker身份；不创建推理请求|
|无故障预热|`observe_continuous_warmup(..., request_ids=...)`，完成后无ID调用收集|最多6次，连续2次完整hidden/logits有限且精确一致|
|验收许可|`configure_deployment_acceptance` + `install_deployment_acceptance_permit`|来源必须`YELLOW_DEDICATED_INSTANCE`，授权回调来自可信宿主；不能用BLUE来源|
|清理|`remove_deployment_acceptance_permit`、`remove_continuous_binding`|finish/unregister/detach后，租约和在途请求清零才能移除|

参考可运行宿主为源包`experiments/worker_core_v11_20261118/continuous_public_probe.py`。它只通过可信bootstrap安装惰性binding/许可，故障控制走公开Worker Runtime与MultiRankTaskClient；不会用实验inject RPC替代公开入口。bootstrap经现有可信executor执行，不能把任意Python callable暴露给未授权用户。

启动参数：TP2、mp executor、eager、异步调度/prefix cache关闭、无MTP/KV connector、image/video上限0；chunked prefill开启，token预算1024且请求长度不超过1024，实际调度再次确认单chunk。28 max_num_seqs=1；Batch=8。4个生成token（prefill+3次decode），无EOS提前结束。参考脚本仅创建独立验收Worker，不接管现役Worker生命周期。

```bash
PYTHONPATH="$PREVIEW_SITE" python "$SOURCE/experiments/worker_core_v11_20261118/continuous_public_probe.py" \
  --mechanism 28 --model "$MODEL" --evidence "$EVIDENCE_28" \
  --source-manifest "$RELEASE/source-review.json" --deployment-source YELLOW_DEDICATED_INSTANCE

PYTHONPATH="$PREVIEW_SITE" python "$SOURCE/experiments/worker_core_v11_20261118/continuous_public_probe.py" \
  --mechanism batch --arithmetic-device input --model "$MODEL" --evidence "$EVIDENCE_BATCH" \
  --source-manifest "$RELEASE/source-review.json" --deployment-source YELLOW_DEDICATED_INSTANCE
```

两条顺序运行，分别使用新证据目录。每项内部使用同一组Worker两个独立owner。目录存在回执时拒绝覆盖。初始模型config/量化描述、硬件芯片原值、安装核验由部署记录保留，不把A2来源登记成A3。

## 请求与判定

宿主循环：enqueue实际请求 → 从输出处理器读取真实内部ID → advance/advance_batch → 等待真实完成 → query读取next_phase，直到complete。normal/fault/off/health四阶段，全部rank故障阶段每个符合条件的调用持续触发，无条件点组通信。绑定和engine/site范围由部署决定，用户不填写层名。

28当前范围：16个full attention层，每调用只替换当前请求最后一个query行；linear attention不属于28。分母重放块0、block_size=128，分子不重复，是大误差torch效果变体，不是历史kernel复刻。实际消费后恢复返回区间与原输出区间。

Batch当前范围：独立DynamicQuant→原QMM，8请求的真实行统一共享scale；不包含融合进其他kernel的量化。NPU scale必须精确等于参考；最多1024个确定性Q样本/调用，仅半整数边界允许±1，报告数量和比例。CPU FP32全量参考作为可选`--arithmetic-device cpu`。抽样零差异不能外推全量零差异。

基线预热不稳定就停止，不通过零初始化或忽略NaN补验收。每请求实验上限120秒；超过则记录不适用/性能失败，不自行缩小层范围后仍声称全覆盖。故障阶段不要求token改变。

必须同时满足：每rank8项覆盖计数完整、eligible失败0、实际消费者同stream完成、注入区域字节恢复、资源0、OFF/health新请求完整hidden/logits与normal精确一致。两作用域保持实际Worker/model身份一致。检查原始回执，不能只看进程退出码。

负向用例：每个rank无许可时实际prepare被拒且无ticket，detach清理；阶段耗尽后再次advance返回GROUP_PHASES_EXHAUSTED；最终许可/Runtime租约/tenant清零。脚本不修改准入登记。

## 只回传文本

原回执和完整数组留在验收环境。逐机制执行：

```bash
PYTHONPATH="$PREVIEW_SITE" python "$SOURCE/tools/review_continuous_public.py" \
  "$EVIDENCE_28/continuous_public.json" --require-negative-creation
PYTHONPATH="$PREVIEW_SITE" python "$SOURCE/tools/review_continuous_public.py" \
  "$EVIDENCE_BATCH/continuous_public.json" --require-negative-creation
```

输出含两作用域各rank的覆盖、hidden/logits效果、恢复、部署和耗时比；原回执SHA可重算，单份输出≤60KB。另回传安装检查全文及原回执文件名/字节数/SHA，不回传模型张量。缺字段、partial rank或任何矛盾均保留，不自行登记。蓝区审查通过后才能形成A3登记；`qualification_granted=false`是工具不自动授权的含义，不代表已验收结果失败。
