# RC3 r6 候选范围与验收入口

本版继承r5，增加已通过A2产品入口的Batch、39mr、33，合入r5独立提取工具和17默认图。原发行标签、wheel、源码包不改；r5.1登记仍独立等待黄区提取文本。本版不是黄区登记发布，product_closed=false。

|能力|验收模式|产品模式候选|A3边界|
|---|---|---|---|
|18、22、39op、28、17|继承已完成候选|A2 TP2已验收|仍须实际产品闭环；17 spawn，默认内置图|
|Batch|继承候选|新增A2 TP2真实并发decode B2～8、B1负例|max_num_seqs=8；实际batch由scheduler决定|
|39mr|继承候选|新增A2 TP2完整K控制与错误累加范围|源/模型/量化身份仍须相符|
|33|A2方向A实测保留|新增A2限定2304 prompt/4token|A3注入关闭，仅取证与控制步骤|
|41、26fia、20|仅匹配且已审查Native构建|关闭，尚未迁移|不可通过配置启用产品路径|
|8及DeepSeek新部署|待实际site/噪声依据|关闭|不能用V2或A2结果授予V4资格|

33方向A的A3步骤见[专页](rc3_r6_paged33_a3_acceptance.md)。12项×环境×模式见[进度表](product_progress_20261123.md)。新增三项实机原件留124，只提交[独立总结](product_r6_20261123/README.md)。所有候选均需可信部署配置及许可，不改变未登记默认关闭的准入。

## r6产品模式复验

专用实例的实际设备映射由部署负责人确定。A2已验证TP2；A3 TP1/TP2独立验收。下列probe自建临时HTTP服务与可信product会话，只用于蓝区专用验收，不能用它接管现役服务。该入口固定TP2、BLUE_COMBINATION及容器路径`/model`、`/evidence`、`/source/source_manifest.json`，没有model或deployment-source命令行参数。部署侧将实际模型、可写新证据目录和源码清单映射到这些路径；不能将此BLUE探针输出改标为A3。黄区产品接线使用现有ProductClient与可信配置API，按[RPC契约](product_rpc_contract.md)单独实现其专用部署启动。三个新增项的蓝区复验入口为：

```bash
python experiments/worker_core_v11_20261118/product_http_probe.py --help
python experiments/worker_core_v11_20261118/product_http_probe.py --mechanism batch
python experiments/worker_core_v11_20261118/product_http_probe.py --mechanism 39mr
python experiments/worker_core_v11_20261118/product_http_probe.py --mechanism 33
# --mechanism batch：并发2/4/8请求组、单请求负例、5token负例；max_num_seqs=8
# --mechanism 39mr：三条真实用户prompt与5token负例
# --mechanism 33：仅已审查A2，三条2304-token prompt与5token负例；A3不能运行此注入路径
python tools/review_product_mode.py /private/product_http_batch.json --output /private/product-summary.json
```

具体参数由probe解析与部署配置约束，不让用户填层/rank/裸地址/数值注入参数。持续触发范围受实测prompt、生成步数、图像与调度限制，范围外明确不注入。当前probe实际核验TP2；TP1不得借用TP2统计作验收。

17无`--image`时统一使用`vision_input.deterministic_image`的64×64 RGB图，像素为(4x,4y,4((x+y)%64))并记录像素SHA。产品金丝雀同图，真实用户图片仍从用户请求读取。A3 TP2设`VLLM_WORKER_MULTIPROC_METHOD=spawn`。r5历史固定图结果有效，不要求重跑。

## 提取工具

源码包含`tools/extract_registration_record.py`、`tools/extract_dsv4_record.py`及`tools/_r5/review_continuous_public.py`。r5提取器固定使用冻结r5审查器，绝不把r6审查器当成r5。原r5补充工具ZIP的SHA和行为保持不变。提取器以Python3.10+标准库运行；仓库内使用`python -m tools.extract_registration_record`。r5十四组合的逐条命令仍在[独立交付单](../../releases/RC3-PREVIEW-R5-EXTRACT-TOOL-20261123.md)。缺失身份保持null，`consistent=true`不等于具备登记资格。完整summary和原始回执留黄区，只回传规范提取文本及SHA。

## 冻结门禁

与r5同口径：提交确定源码→新版本三wheel及源码tar→source-review原字节SHA→CPU回归零失败/错误（既有废弃研究skip逐项列出）→124无设备/无网络安装→逐文件核对安装源码与清单→记录安装执行→发行清单/SHA256SUMS→新发行提交与标签。不能从冻结后工作目录改包。r6仅冻结候选能力，不自动授予A3登记、TP1继承或product_closed。
