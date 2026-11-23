# r6 黄区A组执行单（负责人转交，不等待B组）

目标：Qwen3.8现役W8A8、A3、v0.26.0rc1-a3、CANN9.1.0；TP1与TP2分别验收七项产品能力：18、22、39op、28、17、Batch、39mr。第八项33仅做取证和控制，不启用A3注入。r5.1是验收登记，不等于产品闭环。此执行单与独立工具补充件不修改r6冻结物。

## A1 独立安装

从tag `demoncatpfi-rc3-preview-20261123-r6`取发行目录与源包；按[交付单](../../releases/RC3-PREVIEW-R6-HANDOFF-20261123.md)核对SHA256SUMS、三wheel并运行check_rc3_preview_install.py。安装目录必须新建，不能覆盖r5.1或现役。期望版本0.4.0rc5/0.3.0rc5，384个安装源文件核对一致；真实回执保存，不手填“384通过”。许可SHA只用r6 source-review.json：`7a411ce29de51a0fecc6dbfe850aeea765f948c6414708360c873dfe22f38bf4`。

```bash
cd "$RELEASE"
sha256sum -c SHA256SUMS
mkdir -p "$SITE" "$SOURCE"
python -m pip install --no-index --no-deps --target "$SITE" "$RELEASE"/wheels/*.whl
tar -xzf demoncatpfi-rc3-preview-20261123-r6.tar.gz -C "$SOURCE"
python "$SOURCE/tools/check_rc3_preview_install.py" --site "$SITE" --release "$RELEASE"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

两TP各开专用实例，硬件设备映射由部署侧确定。不使用蓝区product_http_probe.py的BLUE_COMBINATION bootstrap冒充黄区：该脚本固定TP2。黄区使用下述同一ProductClient公开接口与现役CLI控制通道，按真实TP配置启动。

## A2 真实部署与CLI接线

所有实例eager/mp、关闭异步/prefix cache/MTP/KV connector，chunked开启但单请求单chunk，temperature0。普通文本项max_model_len=max_num_batched_tokens=1024、max_num_seqs=1；Batch改max_num_seqs=8、max_num_batched_tokens=8192；17 image额度1、video0，mm_processor_cache_gb=0，mm_encoder_tp_mode=weights，min_pixels4096/max_pixels16384。17必须spawn，默认内置64×64金丝雀图，用户请求用两张不同图片。

可信部署启动流程调用`configure_deployment_acceptance(source='YELLOW_DEDICATED_INSTANCE')`与`configure_product_deployments`，真实tokenizer/processor提供canary_tokens；安装各机制/本源码SHA的作用域许可。只在管理通道开放既有`demoncat_product_binding`与fault RPC，不能对推理用户开放任意callable或自行改准入。完整JSON字段及函数参数以r6 `product_config.py`和[RPC契约](product_rpc_contract.md)为准。配置source、tp、model与live配置必须一致；Native路径保持null。

主项目CLI映射如下，不新建生命周期：

|用户操作|持久宿主会话|Worker/请求行为|
|---|---|---|
|列出故障|ProductClient.capabilities|读取候选/限制，不授予资格|
|开启|首次attach，再enable|装观察器与租约；canary稳定后开启|
|查看状态|status|Worker自动发现真实HTTP请求，按内部ID/rank归属|
|关闭|drain真实完成，再disable、detach|字节恢复、租约/资源零、新金丝雀恢复|

宿主示意（executor、run_canary、drain由已有管理/HTTP通道实现，不带用户请求before/after钩子）：

```python
from demoncat.runtime.vllm_ascend.worker_extension.product_host import ProductClient
client = ProductClient.from_deployment(executor, deployment_json,
    operator=authorized_operator, run_canary=send_real_http_canary,
    drain=wait_actual_user_requests_finished, timeout=900)
client.attach()
client.enable()
# 用户通过原HTTP推理API提交请求，Worker自行发现和关联。
observations = client.status()
recovery = client.disable()
client.detach()
```

同一CLI控制进程必须保存client及owner/generation，不能每条命令新建会话。两作用域同组Worker、第二次用新owner；原owner恢复证明不完整则停止，不能重启掩盖资源残留。预热上限6次、每次600秒，连续两次完整hidden/logits有限且精确相同。正式阶段900秒；耗时只记录、不优化、不作门禁。

## A3 每TP七项验收矩阵

|机制|正例|负例及关键证据|
|---|---|---|
|18|三条不同文本，每条4token|5token范围外；实际norm消费者、全rank计数|
|22|同上|5token范围外；Q/K或rotary行变化、真实消费者|
|39op|同上|BF16 ND投影，允许真实QBMMv3输出；新增Inf与上游已Inf分开|
|28|同上|真实attention重算及消费者；不把历史kernel复刻作为结论|
|17|至少两张不同用户图片、三次请求|5token范围外；真实encoder调用，不制造decode视觉调用|
|Batch|并发2、4、8请求组，各4token|单请求B1必须不注入；5token负例；记录实际decode批次而非HTTP并发数|
|39mr|三条不同文本，每条4token|5token范围外；完整K控制、错误范围及真实消费者|

各项两个owner，记录每rank八项计数及不适用理由。changed可以小于applied；eligible失败必须0，应用、消费、恢复计数相等。Batch按rank/frame去重，共享调用不能按参与请求重复累加。断言两个作用域Worker身份相同、owner不同；区域字节恢复/租约资源清零/关闭后新金丝雀精确恢复三项均通过。未要求用户token变化。额外负向：缺许可/未知profile/超范围请求不注入且有明确原因。

宿主保存原始client返回值，按r6 `product_http_probe.py`的报告结构组织私有回执（entry=VLLM_HTTP_API、host_per_request_hooks=false，scopes内保留requests/disable/cleanup/detach），不用其固定BLUE启动逻辑。实际部署身份由Worker `observe_deployment`的只读结果单独保存，不能填文档默认值。报告失败原样保存。

## A4 提取回传

使用本次统一工具补充件的`extract_product_record.py`，已补TP1/TP2独立审查；不要用r6包里原固定TP2的review脚本给TP1拼一个虚拟rank。工具只读原回执与实际身份JSON；每scope/rank一页≤4096字节，review_error明确返回，不写登记。

```bash
cd "$TOOLS"
PYTHONPATH="$TOOLS" python -m tools.extract_product_record /private/product-receipt.json \
  --runtime-identity /private/product-runtime-identity.json
```

按TP/机制逐份原样粘贴全部页；完整summary/回执留黄区。另回传安装检查结论/字节SHA、CLI接线文件SHA、缺许可与未知profile负向原始返回的短文本、实际软件版本和CANN读取来源。工具未覆盖的负向不可省略或自动填true。A3产品结论由蓝区复核后登记；r5登记不能替代本项。

## A5 33只取证

按[r6 33步骤](rc3_r6_paged33_a3_acceptance.md)做原生只观测、实际key/核数/S2解析、固定真实QKV控制检查≥99%。A3注入保持关闭，不能复用A2分组或修改设备名。当前只观测probe固定TP2，TP1需按实际几何适配并单独审查，不能拿TP2证据登记TP1；阶段结果不阻塞上述七项。

## A6 CANN路线不可混用

Qwen模型1的Native构建/验收：v0.26.0rc1-a3、CANN9.1.0。DeepSeek模型2：v0.32.0rc1-a3、CANN9.2.1。Native库不能混用，其他CANN构建物按新组合处理。41/26fia/20本轮仍不进入产品验收，待ff19425f构建身份完整审查后另发步骤；B组标定/只观测不影响本A组七项开始。
