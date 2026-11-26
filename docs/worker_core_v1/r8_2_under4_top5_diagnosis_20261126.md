# under4 API top-5 缺项：同次运行诊断

组合：124，A2 / Ascend910B4，vLLM 0.26.0、vllm-ascend 0.26.0rc1、torch 2.10.0+cpu、torch_npu 2.10.0.post4、CANN 9.1.0；社区 Qwen3.8-27B W8A8，TP2，卡2/5。该结论是蓝区复现，不代替黄区 A3 登记。

## 结论与边界

已复现 `API_TOP5_REQUIRED:choice=0:position=0:actual=1`。本次不是完整 logits 出现 NaN/Inf，也不是算子只产出一个候选。故障使不完整字节 token 进入 top-5；completion API 用解码字符串作为字典键，多个 token 的相同字符串发生覆盖。效果工具要求五个键，因而报错。

修复只设置请求的 `return_tokens_as_token_ids=true`，所有 baseline/fault/off 采用相同表示。服务端已有该选项；保留 temperature=0、seed=42、logprobs=5、max_tokens=4，保持至少五条、数值有限、OFF精确恢复的检查。部署 adapter 必须将该字段传给 API，不能悄悄丢弃；不支持时应明确失败，不能退回碰撞后的字典并当作通过。

## 同次证据

- 窗口：`r82effects1126a3`；源码 `9e4aeaf59fdce19f87c5ce1a06445a3a63574621`；源码清单 SHA `6a983cbef27b7ef72935e89347ffc148190c295f73164be3939868e3171302f9`。
- 原件：`r82effects1126a3-under4/effect_18.json`，16,201,191字节，SHA256 `05b63724125bce23132b8a69be54ad91172c19b5920516ed9a1c03cee56a8dc2`。留在受控证据目录，不提交整个原件。
- scope0 的 baseline/fault/off，两个rank各12个execute/logits事件，每个logits为 `[1,248320]`。全量有限性检查的非有限元素数均为0；observer未截断并恢复了原方法。
- 首个请求原文 `Name a city with a river.`，实际prompt token `[657,264,3177,440,264,14367,13]`；CPU tokenizer 与运行中rank记录的SHA均为 `2f9613166692bc0508a1cb1cbcc48b18b6813745ef3671fee54b9ad4d369b586`。内部request ID的choice后缀、首步位置与该请求一致。
- 首步两rank top-5 ID为 `[236,250,104,255,248]`，对应logits `[21.375,20.75,20.5,19.875,19.75]`；实际tokenizer单独解码均为替代字符 `�`。vLLM增量解码后的API键均为空字符串；原件该位置只剩 `{"":-2.610393762588501}`。不以文本键个数推断词表候选数。
- 实际镜像内 `vllm/entrypoints/openai/completion/serving.py` SHA `2590281b465d8a9e31252bd92407661f3af9adeddf7bd52c80395cc9fbb9e19c`：`_create_completion_logprobs` 的字典推导以 `_get_decoded_token(...)` 为键，传入 `return_as_token_id`；completion请求协议提供 `return_tokens_as_token_ids`。
- 故障轮其它请求/位置也出现少于五个字符串键，已保留在原件。baseline与OFF没有此项缺失。故障输出可退化，本次不改变故障强度以规避工具问题。

## 修复验证

源码提交 `9464df0c2bd05eef00759f7cba359fee803a682e`。相关CPU 44项通过，包括同字串token的五个独立ID表示、每轮传递相同参数、缺项仍拒绝、OFF不精确仍失败，以及Batch只能按实际调度组成重试。442个公共机制和运行时文件与冻结r8逐字节一致。

修复后A2回归使用新窗口 `r82effects1126a4`，源码清单SHA `7ffa9de5a6fee00ab55b064cf5e64f179f6b2b6f483615a81638307c02c6cd60`。最终结果记录在本轮A2审查记录；旧失败原件不改写、不作为新源码通过证据。
