# 黄区工具补充件 r2

独立补充件；不覆盖r1、r5、r6及冻结标签，不授予产品资格。由负责人转交，不直接操作黄区。

- [ZIP](yellow-tools-20261123-r2/yellow-tools.zip)：123843字节，SHA256 `d71d70e0a4124d5550ff106f89979e8c5b257851af1caa2838c306423bb7bdf1`。
- [54文件manifest](yellow-tools-20261123-r2/manifest.json)、[外围校验清单](yellow-tools-20261123-r2/SHA256SUMS)、[独立CPU检查](yellow-tools-20261123-r2/cpu-check.json)：54项通过；14份既有r5提取文本重复verify均通过、逐字节相同。
- [真实A2只读回执提取检查](yellow-tools-20261123-r2/actual-alias-extract-check.json)：两次输出相同，每页不超过4096字节，逐页规范化SHA通过。原始张量不在包内。

新增15只读普查及有界文本提取、TP1 Batch诊断、r6补充提示及Native审查后的N批执行草稿。保留r1的DeepSeek noise/observe、Native构建提取和r5登记提取；r5审查器仍与r5原件逐字节相同。Native构建文本身份已审查通过，但运行时加载CANN及N批A3验收尚未完成，不能把旧r1说明中的“待构建身份”或本包中的草稿当作准入。

## 使用

Python >=3.10。纯提取/审查只用标准库；普查需要目标实例现有torch/torch_npu/vLLM。解压到独立TOOLS目录，先执行：

```sh
cd "$TOOLS"
sha256sum -c SHA256SUMS
python -m tools.summarize_alias_census "$PRIVATE/alias_census.json"
```

输出每行原样回传；原回执留黄区，只回传字节数和SHA。普查接线与start/stop见包内 `docs/worker_core_v1/alias15_census_handoff_20261123.md`。不要在黄区运行蓝区专用的probe主程序；现有Worker扩展只组合只读方法，不能替换现役启动参数或制造写入。`alias15_output_probe`目前只适配源码指纹匹配的0.26 Attention输出链，其他版本拒绝，不据此推广到DeepSeek。

Native、DeepSeek采集和提取命令继续使用包内 `yellow_tools_supplement_20261123.md`；Native最新审查与后续步骤以 `yellow_intake_20261123_r6_followup.md` 和 `r7_native_a3_draft_20261123.md` 为准。Qwen为CANN9.1，DeepSeek为CANN9.2.1，库不能混用。

Batch恢复提示以 `r6_a_supplement_20261123.md` 为准。r6的旧金丝雀代码不能因读了说明就视为支持新协议；串行B1维护修正在独立r6.1分支验收，本包不替换冻结r6 Runtime。A组0–7、B组所有步骤固定8–15；负责人已确认0、2、14号卡释放。

15 B档已批准；本包只交付只读工具，故障实现与A2产品验收在独立分支，尚未授予A3资格。
