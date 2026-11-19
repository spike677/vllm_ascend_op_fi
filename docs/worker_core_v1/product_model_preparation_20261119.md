# 普通 W8A8 验收模型准备

此步骤只准备量化权重，不构成故障命中、推理正确性或主项目接线验收。
124原有小型模型 `/home/gyl/models/Qwen3-0.6B` 为BF16；现成W8A8 Qwen为32B，
本轮没有为迁就权重扩大TP1范围，也没有改变Torch/CANN/vLLM安装。

使用[官方 ModelSlim](https://gitcode.com/Ascend/msmodelslim) `26.0.0`分支的固定提交
`3b3f5151521f0b2a0c1f29937fc735ba4938db43`，执行其
`example/Qwen/quant_qwen.py`：CPU、W8/A8、dynamic、Qwen3全部28层、MindIE格式。
工具的 `setup.py` 标识为 `26.0.0.alpha01`，这里不授予产品版本资格。
仅在独立依赖目录安装accelerate1.12.0、easydict1.13、wcmatch10.1、bracex2.6，
没有安装或替换推理栈的wheel。使用固定镜像 `35f9e569…`、runc、无网络、无NPU设备映射，
原模型和官方源码只读。torch_npu导入会查询设备数量并产生无设备警告；未执行NPU模型算子。

两条固定校准文本是英文天空颜色解释和12+5本书的算术问句；未为故障搜索更换输入。
准备过程保留四次独立目录与失败日志：

|目录后缀|结果|处理|
|---|---|---|
|a1|慢速tokenizer缺少merges.txt|从原tokenizer.json的151387条BPE merges无损生成独立输入文件|
|a2|校准JSONL类型不符合官方接口|改为其原生`inputs_pretokenized`对象字段；文本不变|
|a3|safetensors拒绝tied embedding/lm_head共享存储|仅在CPU序列化边界克隆重复存储，保留全部键并逐值检查相等|
|a4|CPU量化和保存完成，exit0|原权重及配置哈希未变；自有容器已清理|

补齐merges后，校准文本和另四条中英文/空白/特殊token样例的fast/slow token ID完全一致。
序列化只复制`lm_head.weight`的存储，不改变其逻辑值、dtype或量化计算。
这两项是模型准备兼容处理，不是运行时故障注入。

最终私有目录：
`/home/17-fault-c2c3-20261117/product-qwen06-w8a8-cpu-a4/model`。
`model_quant_type=W8A8_DYNAMIC`，配置仍为Qwen3、28层、BF16推理dtype；
q/k/v投影的weight/scale/offset均标记动态W8A8。

|文件|字节|SHA256|
|---|---:|---|
|config.json|1364|9e7fe511a941c0b6d8a3cd6e611945632b00723ff167c9d42ae26f042d221a15|
|quant_model_description_w8a8_dynamic.json|39839|611a5a4feea1b68921179abba823a31951bcdad4b8f4b214b6a9558e817abb82|
|quant_model_weight_w8a8_dynamic.safetensors|1952163304|b27e6b55f62951633f4d91c1f6072a2f936283402bd04eb9fd3a278b2780d93a|

原模型权重SHA256为`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`，
前后均一致。权重、原始回执和日志留在私有目录，不上传GitHub；本页只交付来源和结果摘要。
后续仍须验证标准vLLM加载、实际量化调用和消费者、正常/F/OFF及有界数值效果。

## 固定 vLLM loader 的元数据兼容

Source116实机第一次加载a4在模型构造阶段失败：`KeyError('model.embed_tokens.weight')`，
未注册provider、未执行任何故障阶段。实际描述文件含该条目，问题在读取入口：固定镜像
的`weight_utils.get_quant_config`把非空HF `quantization_config`直接交给
`AscendModelSlimConfig.from_config`作为逐层描述，而a4保存的是量化生成选项。
存在内嵌配置时，loader不再读取独立描述文件。

独立a5目录为`/home/17-fault-c2c3-20261117/product-qwen06-w8a8-cpu-a5/model`。
仅将`config.json`的`quantization_config`完整替换为a4描述文件中的字典；不是零散补键。
新配置SHA256：`bf644c8607476632c4ece502205dbd5b4793723025471c375fe9626688a19816`。
所有其余文件字节/哈希不变；权重使用原只读inode，没有重做量化或改推理wheel。
验收准备器要求完整内嵌描述与独立文件的canonical JSON一致，包含未知新增键、嵌套值及类型。
漏层、多键、不同值或类型均拒绝。a4及其失败原件保留。
具体loader取证和校验见[普通量化运行时说明](quant_ordinary_projection.md)。

Source117在124卡5实际加载a5并完成健康请求及normal B8，自动选中
`model.layers.0.self_attn.qkv_proj`，观察到真实dynamic quant和后续QMM。
reference阶段在Native admission前拒绝，原因为普通前缀缺少实际dispatcher frame；
`admitted_calls=0`、`fault_attempted=false`，清理完成但不算故障或数值OFF通过。
模型加载问题已解除；此失败与模型准备成功分别记录。修复后的验收使用新source和新attempt，
保持a5权重、Native库及输入不变。
