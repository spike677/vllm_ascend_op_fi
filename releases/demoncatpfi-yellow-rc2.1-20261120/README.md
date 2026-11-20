# 黄区测试部署 RC2.1

源码提交 `8c358961d9db16bd2bcbf6916bd8d9f5722d3a18`；基于 RC2 发行 `de17267b`；标签 `demoncatpfi-yellow-rc2.1-20261120`。RC1、RC1.1、RC2 原标签和包不变。

仅登记 18.api_half、39.output_proxy、22.wrong_row 为 YELLOW_TEST_DEPLOYMENT，证据等级 YELLOW_TEXT_ATTESTED，原回执留黄区。实际组合：A3（Ascend910_9382）、Qwen3-0.6B BF16、TP1/PP1/DP1、串行 eager、vLLM/vLLM-Ascend 0.18、torch2.9.0+cpu、torch_npu2.9.0.post1+gitee7ba04。CANN 按运行读取器观测的8.5.1登记；toolkit version.cfg报告8.5.0，二者来源差异保留。

- 39、22：summary显示同Worker两作用域。
- 18：两作用域的PID、进程开始身份、Worker开始身份及模型对象不同；仅登记两独立作用域，不声称同Worker。
- 22：**行级效果未进入 summary（工具缺口），以 hidden/logits 变化和 OFF/health 精确恢复为证；RC2.2 起 summary 工具输出 22 的 Q/K 或 rotary 行变化。**
- 18/22的hidden完整哈希恢复可由文本直接核对；logits改变和最终恢复还依赖摘要的变化/恢复声明，未收到完整logits数组。
- 安装检查仅收到文本摘录；原148719字节全文留黄区。RPC子类按可信部署包装接受：公开binding/permit API，无准入修改。
- 原始粘贴summary在本地仓库外逐字节保留；其SHA、大小和原回执规范化SHA写入登记表。本发行不提交含Worker身份的原始黄区文本。

[源码包](demoncatpfi-yellow-rc2.1-20261120.tar.gz)、[源码清单](source-review.json)、[安装核验](install-summary.json)、[发行清单](release-manifest.json)、[SHA256SUMS](SHA256SUMS)、[升级步骤](UPGRADE.md)。

源码包 SHA256：`782dcfe2f5fe4be41e65eeb1de595107aa5ceaeb39b5ec4e075f0f5aac962249`。
源码清单 SHA256：`0127383a2cad8c099b0101ab666cb8e404054eb4f9c373da6ac63aba63416d49`。

**登记版保留RC2的distribution版本号**，只改登记表；三个wheel均在本发行目录独立提供，不能只按版本号选包。升级必须进入新空target并按SHA核验，不能让旧同版本wheel命中缓存。285个安装后的Python源文件与818个Git源文件构成的源包逐字节核验通过。124 aarch64旧栈无设备容器安装检查通过，NPU用时0；17项原授权回归和18项精确组合正负检查通过。

不包含41、request_plan、TP2或28新实现；这些属于后续RC2.2/RC3。新栈、A2、TP2、现役Qwen3.8/DeepSeek不因本登记获准。product_closed仍为false。
