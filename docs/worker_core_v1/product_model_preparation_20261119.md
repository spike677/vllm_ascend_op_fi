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
