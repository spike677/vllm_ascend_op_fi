# r6：33 的 A3 分阶段取证与验收

状态：A2/0.26/CANN9.1/Qwen3.8社区W8A8/TP2产品模式通过；A3未验收。r6包含方向A实现，但A3产品attach仍拒绝。不能改设备身份、用A2源码指纹或直接沿用512→256来绕过。以下由负责人转交，Codex不直接操作黄区。

## 1. 专用实例、原生只观测

在现役Qwen3.8权重、v0.26.0rc1-a3/CANN9.1.0的专用容器运行。TP1与TP2独立取证、独立登记；现有探针固定TP2，TP1需先按实际头数适配，不能用TP2数据代替。保留硬件原名、镜像digest、完整版本、config/量化描述/所有权重索引SHA、rank到逻辑设备映射。设定eager/mp、无prefix/异步/MTP/connector、纯文本、seed42、max_model_len=4096、max_num_batched_tokens=4096、max_num_seqs=1，单chunk的2304-token输入，4个生成token。不对现役服务安装观察钩子。

将r6源码挂载到`/source`、模型到`/model`、新私有证据目录到`/evidence`。以下路径均在容器内，源CANN环境按镜像原方式加载。`PREVIEW_SITE`为三wheel的独立安装目录。

```bash
mkdir -p /evidence/launch
g++ -std=c++17 -Wall -Wextra -shared -fPIC -O2 \
  -I/usr/local/Ascend/cann/include \
  /source/experiments/worker_core_v11_20261118/paged33_launch_capture.cpp \
  -ldl -o /evidence/liblaunch_capture.so
sha256sum /evidence/liblaunch_capture.so /source/experiments/worker_core_v11_20261118/paged33_launch_capture.cpp
export VLLM_WORKER_MULTIPROC_METHOD=spawn
DCAT_LAUNCH_CAPTURE_ALL=1 DCAT_LAUNCH_CAPTURE_DIR=/evidence/launch \
DCAT_CAPTURE_PRIVATE_QKV=1 LD_PRELOAD=/evidence/liblaunch_capture.so \
PYTHONPATH="$PREVIEW_SITE:/source" \
python /source/experiments/worker_core_v11_20261118/paged33_native_probe.py
PYTHONPATH=/source python -m tools.paged33_launch_inventory /evidence/launch \
  > /evidence/launch-inventory.json
```

观察器只复制host参数、调用原launch；不改参数、不加kernel。`ALL=1`用于发现A3未知key，最多8192个函数/launch每进程，参数单条≤64KiB。上限命中、截断、符号未拦截、没有记录均不得认定完整。它仅覆盖`aclrtBinaryGetFunctionByEntry`与`aclrtLaunchKernelWithHostArgs`，不保证其他launch API可见。profiler与direct FIA调用记录用来确认实际算子关联；每rank仅保存一个真实decode的固定Q/K/V作为初步检查点，不能声称全层控制已通过。

原始host参数含地址、QKV是完整私有张量，全部留黄区。inventory只输出key、launch核数、参数大小、文件SHA；**S2分组保持null，直到匹配实际源码/serializer完成解析**。`tools/summarize_paged33_launch.py`是A2特定布局解析器，不能拿它的常量解A3未知布局。A3需结合本机CANN9.1对应host/kernel源码及profiler算子关联，逐字段确认tiling key、实际核数、S2边界、P舍入、MM输出和LSE精度，记录源码SHA及字段偏移。若不能证明，回传“布局不可解析”，停止注入，不猜测。

## 2. 固定真实Q/K/V控制检查点（不改模型输出）

由上述解析结果生成每个rank的一份私有`observed-spans-rankN.json`。以下为字段格式，尖括号必须替换为实测值，**不是可直接运行的A3分组**：

```json
{
  "schema":"dcat.paged33.observed_spans.v1",
  "operator":"FIA_paged_decode",
  "arithmetic":"BF16_MM_FP32_OUT_PBF16_LSE_FP32",
  "qkv_sha256":"<对应private-qkv-rankN.pt的SHA256>",
  "launch_sha256":"<与该调用对应的launch文件SHA256>",
  "serializer_sha256":"<实际解析依据源码SHA256>",
  "spans":[[0, "<实测第一块结束>"], ["<下一块起点>", "<实际KV长度>"]]
}
```

`arithmetic`是待验证的明确假设，不能在实际精度不同的分支上强填。工具检查分块覆盖完整且每个key恰好一次，不提供512默认值；本地重算两次，分别与保存的原生输出逐位比对，每次≥99%，重复重算必须100%一致。每rank在映射后的实际设备执行；下面分别挂载/映射对应逻辑设备，不能把两rank都标成同一设备结果：

```bash
PYTHONPATH="$PREVIEW_SITE:/source" python -m tools.paged33_control_checkpoint \
  /evidence/private-qkv-rank0.pt --device 0 \
  --launch-review /evidence/observed-spans-rank0.json > /evidence/control-rank0.json
PYTHONPATH="$PREVIEW_SITE:/source" python -m tools.paged33_control_checkpoint \
  /evidence/private-qkv-rank1.pt --device 1 \
  --launch-review /evidence/observed-spans-rank1.json > /evidence/control-rank1.json
```

退出2或异常即停止；NaN/Inf、布局不符、无法对应调用、任何rank控制不足99%都不能进入下一步。≥99%仅是重算检查点，保留残余差异；不等于历史kernel完全复刻或注册许可。上述离线工具已做CPU契约测试，蓝区没有A3，不能声明其A3实测通过。

## 3. 解析审查后才开两作用域验收

r6现有运行adapter仍是固定A2分支。阶段1/2文本交负责人转回后，Codex核对实际A3布局：若相同也须新增对应部署身份；若不同则修改A3 adapter与合法分组生成、补CPU测试及候选源码，不能仅改manifest或改工具参数。**当前r6没有可直接打开A3 33的命令**，也没有“强制放行”选项；这是保留未验收能力关闭的边界。后继审查通过的候选使用以下统一契约，不改生命周期：

1. 缺许可、错误源码/镜像/模型SHA、未经审查分组和超范围长度负例必须拒绝/不注入，并报告原因。
2. 同一组Worker两个owner；每作用域先预热至连续两次输出全有限且精确相同（最多6轮）。验收模式按request_plan的normal→fault→off→health；每阶段默认900秒。
3. 故障阶段各rank所有符合条件的真实paged decode均执行；记录八项计数、实际S2分组、同一合法key集合、每次control≥99%、control→alternate与native→alternate各自变化、真实消费者、非目标区域和字节恢复。控制失败是eligible失败，不能当skip。
4. 产品模式先由可信部署安装binding/许可；宿主只attach→enable→status→disable→detach。真实用户HTTP请求至少三条不同长prompt，生成4token；长度/5token范围负例；长prompt金丝雀。每个owner独立完整恢复后才进入下一个owner。用户请求不插before/after回调。
5. 关闭后区域字节恢复、租约/钩子/inflight/owner零、新金丝雀hidden/logits精确恢复。第三方进程退出告警另记，不混为注入资源。不能满足恢复即隔离实例并停止下一作用域。
6. 原回执由`tools/review_continuous_public.py`（验收模式）或`tools/review_product_mode.py`（产品模式）独立审查。后者当前只复核A2 TP2候选格式，A3/TP1报告格式须随adapter一起审查；不得把A2标签改写为A3。

## 4. 文本回传

不要求summary全文。每个TP/rank分段回传：部署身份；观察器源码/库SHA；实际key/核数/S2分组与其解析来源SHA；capture完整性和缺失项；固定QKV文件SHA（不传张量）；control-rankN.json全文（很小）；两作用域每rank八计数、真实消费者、三项恢复、负例及耗时倍数；summary/原回执文件名、字节数、SHA。每段≤4KiB，超出按rank/作用域拆分，不截断字段。r5专用提取器仍只用于原r5七项登记，不用其结果为33发放资格。审查结论由负责人转交；未审查组合保持关闭。
