# r7.4.1 Qwen线统一正式测试总单 v2（K1）

本单优先于冻结包内旧交接单的“最小复测集/沿用旧结果”段落，也替代此前简版补充单。原标签和包不改写。**所有以下项目均在r7.4.1上正式测试；不能以r7.2未审结回传或旧版本通过结果免测。** 公共项当前没有本轮已审查登记的A3产品组合。

基础：标签`demoncatpfi-rc3-preview-20261124-r7.4.1`，发行`59f6aea5`，源码`fd62c538`；许可source-review.json SHA `9a50e186599fa129d652267e1681c42966cce6603282714ca20f86a78aea9b63`。A3、Qwen3.8现役W8A8、TP2、v0.26.0rc1-a3、CANN9.1。main已经依次快进到r7.4和r7.4.1，无需再执行K2。蓝区示例是A2验证，不替代本表A3验收。

## 完整范围

|项目|机制/挂点|
|---|---|
|18|18.api_half|
|22|22.wrong_row|
|39op|39.output_proxy|
|39mr|39.matmul_range|
|28|28.fia.denominator_replay|
|17|17.spatial_padding|
|batch|quant.batch_shared_scale|
|15|15.alias_detach|
|41|41.tail_column|
|26fia|26.fia_destination|
|20|20.buffer_substitute|
|新增18档|18.variance_over4、18.variance_under4，分别登记|
|新增挂点|模块输入、DynamicQuant dispatch输入、外部视觉Conv3d|
|外部v1|模型输入/输出、采样、视觉，以及黄区相应适配器|
|外部v2|库2、14、40；库21若修改持久weight/scale，必须v2|
|黄区适配器|库3、9、10、11、16、19、21、24、31、32、37；现象复现库7、8、12、34|
|黄区HCCL|黄区独立路径，逐位金丝雀；不能套公共provider计数|

每项同一服务、同组Worker两个owner；每owner依次baseline→fault→off三轮相同输入。disable必须先排空、设备读完、恢复和金丝雀完成。记录token、文本、API logprob变化（不冒称完整logits）、每rank八项计数、changed_elements、消费者、字节恢复、资源和新请求金丝雀。changed≤applied允许，耗时仅记录。v2另列写入数和读取次数，恢复在独占排空/请求页面释放前边界，不能伪造逐请求restored。

负例在这个最终包/实际部署组合上集中跑一次：缺许可、未知名称/profile、重复开启、重复关闭及既定CLI未知编号；关联同一负例回执SHA到各项。拒绝应结构化、CLI exit4（重复关闭可幂等），引擎仍存活且资源无残留。外部注册SHA、v2持久区域、视觉请求等各自安全边界的适配器专项负例仍需覆盖，不能用通用缺许可负例替代。

## 一页效果工具命令

以下是**现有黄区可信宿主适配器**的调用清单；不会替黄区编写适配器或创建CLI映射。`create_adapter(config)`须接已有鉴权session和真实HTTP请求。配置目录中每个文件须写明实际mechanism/命名变体、prompts、temperature、max_tokens和对应宿主参数。case名称只是本地文件名，不能当作已注册external ID。不得把缺失配置补为默认通过。

```bash
export PYTHONPATH="$CODE/src:$CODE:$CODE/fault39/src${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export DCAT_EXTERNAL_SITE_EVIDENCE=/evidence
# 上述环境须在启动两个Worker前传入，之后在宿主export不能追改已运行Worker。
# CODE=已核验r7.4.1源码；ADAPTER=黄区现有可信效果适配器；ADAPTER_SHA=已审核完整SHA。
# CONFIG_DIR和OUT由部署设置；OUT须为新目录，不覆盖旧原件。
run_case() {
  key="$1"
  python "$CODE/tools/product_effect_probe.py" --adapter "$ADAPTER"     --adapter-sha256 "$ADAPTER_SHA" --config "$CONFIG_DIR/$key.json"     --output "$OUT/$key-original.json" > "$OUT/$key-page.json" || return
  python "$CODE/tools/product_effect_probe.py" --verify "$OUT/$key-page.json"
}
run_case public_18  # 18.api_half
run_case public_22  # 22.wrong_row
run_case public_39op  # 39.output_proxy
run_case public_39mr  # 39.matmul_range
run_case public_28  # 28.fia.denominator_replay
run_case public_17  # 17.spatial_padding
run_case public_batch  # quant.batch_shared_scale
run_case public_15  # 15.alias_detach
run_case public_41  # 41.tail_column
run_case public_26fia  # 26.fia_destination
run_case public_20  # 20.buffer_substitute
run_case site_variance_over4
run_case site_variance_under4
run_case site_module_input
run_case site_dynamic_quant_dispatch_input
run_case site_external_vision_conv3d
run_case site_v1_model_output
run_case site_v1_sampler
run_case yellow_row_2
run_case yellow_row_14
run_case yellow_row_40
run_case yellow_row_3
run_case yellow_row_9
run_case yellow_row_10
run_case yellow_row_11
run_case yellow_row_16
run_case yellow_row_19
run_case yellow_row_21
run_case yellow_row_24
run_case yellow_row_31
run_case yellow_row_32
run_case yellow_row_37
run_case yellow_row_7
run_case yellow_row_8
run_case yellow_row_12
run_case yellow_row_34
run_case yellow_hccl
python "$CODE/tools/product_effect_probe.py" --adapter "$ADAPTER"   --adapter-sha256 "$ADAPTER_SHA" --config "$CONFIG_DIR/public_18.json"   --negatives --output "$OUT/negative-original.json" > "$OUT/negative-page.json"
```

HCCL或v2若使用不同适配器，将ADAPTER/ADAPTER_SHA切换为对应已审核文件，命令形式不变。v2按r7.4交接单注册PersistentSite及真实恢复边界；不能用v1绕过持久区域限制。Native41/26fia/20配置必须指向审核过的A3/CANN9.1库及SHA，不能使用蓝区A2构建身份代替。

- 文本：4生成token、单请求单chunk、当前已审核范围。Batch max_num_seqs=8，至少两个并发请求实际同一decode批；另保留B1不注入负例。
- 视觉17、外部视觉、库16/32：真实图片请求，使用内置确定性图片；spawn；不能只发文字。配置保留图像grid/像素SHA。库32输入cast和库16低精度重算不是同一种语义。
- 采样/库24：temperature>0（固定seed），否则没有随机采样调用，不制造调用。
- 库12：compute_logits输出列错位或命名坍塌变体，采样器真实读取；词表248320是此部署观测，不写死到通用接口。
- 15：B档ND别名断开，不含私有格式条件；fault不保证有限或逐位稳定。非有限安全与OFF恢复证据必须单列，不能以HTTP成功推断完整张量有限。

每项回传一页效果文本及原件SHA，完整原件留黄区。结果登记以本次能力＋变体＋部署＋模式为准，故障库行仅备注。33仍关闭：只读材料审查、控制门禁通过后才做r7.5增量单项测试。D包等待D0/D1，最终再整合r8。
