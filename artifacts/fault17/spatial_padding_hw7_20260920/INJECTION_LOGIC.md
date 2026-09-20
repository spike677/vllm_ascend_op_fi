# Fault17：原生 Conv3d 空间 padding 参数错误代理

## 1. 原理与定位

本实验模拟的是**算子调用合同中的几何参数错误**：输入图像、权重和计算实现不变，仅把真实 Conv3d 调用的空间 padding 从正确值改成错误值，让原生算子生成错误的视觉特征，再由模型原有下游计算产生回答。

故障族为 `OPERATOR_GEOMETRY_CORRUPTION`。本实例是空间 padding 参数偏差的效果级代理，不声称复现了历史 Conv2DTransposeD 的具体实现缺陷，也不是内存越界或硬件自然故障。

```text
相同图像和文本请求
  → 正常视觉预处理 / patch 输入
  → visual.patch_embed.proj
  → Conv3d 调用参数绑定
      B：padding=(0,0,0)
      F：padding=(0,7,7)
  → 同一个原生 Conv3d 执行
  → shape 不变、数值改变的 patch embedding
  → 原有视觉 / 语言模型计算
  → B 回答 white；F 回答 red
```

没有直接改写回答、logits、采样器、residual 或 KV cache；没有额外裁剪、平移图像或事后修复 Conv 输出。

## 2. 为什么输出 shape 不变，内容却会改变

本次实际输入与合同：

| 项目 | 值 |
| --- | --- |
| 输入 | BF16 `[1200,3,2,16,16]` |
| 权重 | `[1152,3,2,16,16]` |
| kernel / stride | 均为 `[2,16,16]` |
| dilation | `[1,1,1]` |
| B padding | `[0,0,0]` |
| F padding | `[0,7,7]` |
| B/F 输出 | BF16 `[1200,1152,1,1,1]` |

Conv3d 的 padding 顺序是时间、空间高、空间宽；本次不修改时间轴。空间单轴输出尺寸由当前合同计算为：

```text
out = floor((16 + 2p - (16 - 1) - 1) / 16) + 1
    = floor(2p / 16) + 1
```

因此 p=0 到 7 都输出一个位置；p=8 会输出两个位置，不属于本次保形配置。

但“输出一个位置”不代表卷积读取了相同内容。padding 从 0 改为 7 后，唯一空间卷积窗口的起点相对于原输入向前移动 7：部分输入位置被零填充替代，原图像 patch 的部分内容不再进入该窗口，权重与有效像素的对应关系也改变。于是无需改变 shape，就会得到不同的视觉特征。后续 `view` 可以正常完成，服务也可能 HTTP 200，但任务答案已经错误。

这不是把输出张量平移或裁剪后冒充参数错误；错误参数被原生算子真实消费。

## 3. 实现边界与保护

本轮执行代码的核查点为 `ServiceBinding.call()`，来源：
`fault8_17_readonly_audit/fault17_qwen_coverage/service_extension.py`。
该源码不包含在本 dump 包中，其 SHA256 为：
`347278dc8fdd34c2f1db8d0051b22ed095e4591f209e5caa633895c5fb37d42d`。

实现是在实验 worker 中对目标调用进行 method/function 参数绑定，不是修改 vendor kernel，也不是 LD_PRELOAD 内核注入。本次双 rank 使用相同配置。

执行顺序：

1. 确认目标是 `visual.patch_embed.proj` 的 `conv3d`，原 padding、kernel、stride、weight shape、dilation 均匹配。
2. 确认当前是绑定请求的首次完整 prefill，且没有第二次目标视觉调用；控制入口要求开发者模式，安装时检查 owner。
3. 同步 stream，记录输入、权重、bias 的摘要；仅复制/替换本次调用的 padding 参数，核对其他合同字段不变。
4. F 先在独立 storage 的输入/权重副本上执行同合同 scratch Conv，验证真实输出 shape、有限值以及后续 view。
5. 原生算子对模型的真实目标调用执行一次；同步完成后，确认真实结果与 scratch hash 一致、原输入/权重/bias 未被修改。
6. 将原生结果交回既有模型消费链，记录响应；结束后移除绑定并停止本轮实验服务。

因此日志中 live target 的 `native_calls=1` 不代表整个验证只执行过一次 Conv：F 还包含独立 scratch 执行。scratch 不能当作第二个模型 forward 命中。

该能力目前只对已验证合同成立，不能从本次结果推断任意 Conv、任意模型都支持保形静默错误。

## 4. 已观察到的效果

固定 VQAv2 validation question 42000 / image 42：

> What color are the gym shoes?
>
> Respond with only one color word.

| 项目 | B | F |
| --- | --- | --- |
| 完整正文 | `white` | `red` |
| 实际 token IDs | `[5782,151645]` | `[1151,151645]` |
| HTTP / finish_reason | 200 / stop | 200 / stop |
| completion tokens | 2 | 2 |
| rank0 Conv 输出 Norm | 450.0 | 446.0 |

这不是仅换一种等价措辞，而是颜色答案改变。此前固定题目的正确基线为 white。B/F 的输入、权重、bias 统计一致，实际 padding 与 Conv 输出统计不同；开启 msprobe 后的两组回答分别与此前无 msprobe 对应结果一致。

这些证据支持“几何参数错误 → 原生 Conv 特征改变 → 任务回答错误”。它们不定位模型内部具体如何把特征变化解释为 red，也不证明所有图像都会答错或 padding 偏差越大错误率必然越高。

## 5. msprobe 如何覆盖现场

配置为 `task=statistics`、`summary_mode=statistics`、`level=mix`，rank0/1，配置 steps 0–7。请求生成两个 token 后正常停止，所以实际只有 step0–1，每组共 12 个 dump/stack/construct JSON。

本版本 runner 的视觉编码先于常规 debugger.start。为捕获真正的视觉 Conv，本轮把已有 debugger 的首次 start 提前到视觉 encoder 入口，并合并同一 forward 随后的重复 start；原有 stop/step 和 decode start 保持不变。没有每一步重新安装故障 hook，也没有绕过 owner 检查。

现场包括 `Functional.conv3d.0.forward` 和视觉 patch embedding 模块记录。统计 dump 用于证明参数、shape、数值统计和调用关系，不是完整 tensor dump，不能从统计 JSON 恢复全部张量内容。

## 6. 恢复与证据范围

此前独立服务 B/F/R 验证已记录 `white → red → white` 的恢复链。本目录仅交付这次 **B/F** msprobe 数据，没有 R 或 Z；不能把历史恢复当作本压缩包独立包含的证据。

本轮结束后实验服务停止、资源锁释放。包内启动/控制/响应记录用于核对本次运行；版本、压缩包 SHA256 和解包方法见 [README](README.md) 与 [校验清单](archives.json)。

结论：本次证明的是**原生算子消费错误几何参数后，在 shape 合法且服务正常的情况下造成可见任务错误**，不是原始历史缺陷的物理根因复现。
