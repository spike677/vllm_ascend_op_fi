# 黄区统一工具补充件：Native、DeepSeek与文本复核

独立ZIP，不覆盖r5/r6目录、不修改冻结标签。已补齐漏发的`dsv4_noise_capture.py`与`dsv4_observe_probe.py`。包含精确依赖、组装工具、噪声review/文本提取、Native提取、方案B版登记verify和产品TP1/TP2文本审查。文件SHA、ZIP SHA和CPU记录见同批release清单。

解压到新TOOLS目录。Python≥3.10；capture/observe依赖专用镜像的torch（实机还需现有vLLM/torch_npu），review/extract只用标准库。ZIP内生成的最小namespace init仅供独立工具环境，不覆盖现役DemonCATPFI模块；不可用它的namespace替代已安装产品运行包。在只观测专用实例将TOOLS及其`experiments/worker_core_v11_20261118`加入PYTHONPATH。不创建公共任意callable接口。

## 1. Native：对现有产物提取，不重新构建

ff19425f回传已给出镜像、SDK版本、产物和回执SHA，但尚缺构建源码发行/提交、逐源文件SHA、编译器与工具提取文本。不要重新构建。仅在原产物目录运行本工具，路径可由操作者替换。

`runtime-identity.json`按以下规则填写：`image_id`为已回传完整本地Id；RepoDigests确实为空；`cann_runtime=null`，因为这是无设备构建，仅观测到SDK版本文件，未观测已加载运行时。source_release/source_commit填**当时实际构建使用的发行/提交**，不知道就null，不填当前r6冒充历史源码。SDK版本会从现有receipt自动提取。

```json
{
  "image_id":"sha256:1a9d274a0f509583fb8b020aad8c5ba53fa045e5162ecb3146c18452d30b6b8d",
  "repo_digests":[],
  "cann_runtime":null,
  "cann_runtime_source":"未观测：无设备构建；仅SDK版本文件报告9.1.0，不把文件版本当已加载运行时",
  "source_release":null,
  "source_commit":null
}
```

```bash
cd "$TOOLS"
PYTHONPATH="$TOOLS" python -m tools.extract_native_build_record \
  /private/existing-build/receipt.json /private/existing-build/libdcat_postwrite.so \
  --deployment-identity /private/runtime-identity.json
```

预期receipt：9057 bytes/SHA `ff19425f6b1e0460f39b3ca5f8a0f2b4b04b59a4aa9a329d957eff68ddc0a01e`；库29088 bytes/SHA `d9aa0d9488c5c1e8f7fca10c297e4968687ceb6a24866d8346bd33a58afce274`。工具读取实际字节，不将本页期望值写进结果。输出≤4096字节的一段JSON，缺项留null及issues。回传原样文本；它不是构建通过许可。Codex核对源SHA/编译器/recipe/SDK/导出后才进入41、26fia、20产品接线与A3步骤。

## 2. DeepSeek噪声标定

实例必须是DeepSeek V4-Flash-0731 **W8A8**、A3、v0.32.0rc1-a3、TP8+EP/DP1。先核对实际加载CANN为9.2.1并记录来源，不能仅看目录或拿Qwen9.1镜像。不是W4A8 NaN诊断，不是V4.1候选。先只跑一条采集验证每rank实际compute_logits可观测且4步完整；缺失就停止并回传原因，不复制driver数组填满8rank。

以下在现有可信验收驱动中执行；`llm`是已启动的专用实例，`prompt_ids`与`params`来自实际tokenizer及固定请求（≥2500 token、总长≤4096、单chunk、temperature=0、4生成token）。不需要/不允许用户新增请求钩子；这是一次性只观测驱动。

```python
from dsv4_noise_capture import capture
from tools.dsv4_capture_bundle import assemble_run
from fault_injection.dsv4_noise import digest

llm.collective_rpc(capture, args=("start",))
try:
    result = llm.generate([{"prompt_token_ids": prompt_ids}], params, use_tqdm=False)
    captured = llm.collective_rpc(capture, args=("stop",))
except BaseException:
    # 通过可信控制通道逐rank核对abort结果；已stop的rank可能已移除。
    # 清理失败不得开始下一请求。
    llm.collective_rpc(capture, args=("abort",))
    raise
run = assemble_run(captured,
    tokens=result[0].outputs[0].token_ids,
    prompt_token_ids=prompt_ids,
    single_chunk=actual_scheduler_single_chunk,
    sample_id=unique_actual_sample_id,
    process_identity=actual_worker_group_lifetime_identity)
```

最后三个变量必须由真实调度与进程生命周期取得，不填默认true或随意换ID把同一进程伪装成两个。每进程至少5次，同一Worker组采完后由部署方正常重启专用实例再5次；保存两组真实Worker PID/启动身份。run保留完整hidden/logits/top5，仅存黄区私有目录。host组装文件结构为`{"combination":实际部署身份,"calibration":[10个run]}`；部署字段见`fault_injection.dsv4_noise.TARGET`与identity函数，固定规则1.5不调整。正式四阶段后再扩展phases；标定不注入。

```bash
cd "$TOOLS"
PYTHONPATH="$TOOLS" python -m tools.review_dsv4_noise /private/calibration-input.json \
  --output /private/calibration-summary.json
PYTHONPATH="$TOOLS" python -m tools.extract_dsv4_record noise /private/calibration-summary.json
```

只回传最后命令的8个rank文本页，每页≤4096字节；完整summary/数组不回传。finite、token_stable、四步各指标最大值和上界、原summary字节/SHA均保留。CPU已实际执行capture→8rank×4步→两进程各5样本组装→review CLI→extract CLI，且重复提取字节一致；这只是CPU夹具，不是DeepSeek实测。

## 3. DeepSeek只观测

同一部署身份与长prompt。与噪声capture分开运行，避免边界owner冲突：

```python
from dsv4_observe_probe import observe
sources = llm.collective_rpc(observe, args=("source", {"paths": actual_source_paths}))
llm.collective_rpc(observe, args=("start",))
try:
    llm.generate([{"prompt_token_ids": prompt_ids}], params, use_tqdm=False)
finally:
    reports = llm.collective_rpc(observe, args=("stop",))
# sources、reports分别原样写入黄区私有JSON；不得改eligible或unknown计数。
```

actual_source_paths从包内`docs/worker_core_v1/dsv4_0731_source_review_20261122.json`的files条目取路径，读取的是已import包，不是把候选SHA复制为实测SHA。缺失为null。只观测后确认observer_removed=true、injected_calls=0。

11-23下午更正：r1 ZIP保持原字节；下一轮从开发线的`dsv4_0731_source_review_20261123.json`取files路径。新清单修正了旧候选SHA使用Windows CRLF字节的问题，改用同提交Git blob字节，并追加注册表/图相关核对路径。source动作本身无需更新；仍读取黄区实际文件原字节，不归一化黄区数据。使用原九路径已采到的正式source结果仍可按新候选SHA复核，不因蓝区清单错误要求重采同一份未变化文件。

```bash
PYTHONPATH="$TOOLS" python -m tools.extract_dsv4_record observe /private/observe-reports.json
PYTHONPATH="$TOOLS" python -m tools.extract_dsv4_record observe /private/source-reports.json
```

每rank/机制分页，原样回传全部页，包括unknown和不可观测项；0不等于不存在算子。W4A8 NaN诊断/decode计数继续独立回传，不改变W8A8目标。

## 4. 方案B登记verify与产品提取

```bash
PYTHONPATH="$TOOLS" python -m tools.extract_registration_record --verify /private/r5-extract.json
PYTHONPATH="$TOOLS" python -m tools.extract_product_record /private/product-receipt.json \
  --runtime-identity /private/product-runtime-identity.json
```

r5原提取文本不改。新版verify只取消image_digest非必需项，其他身份、计数、负向和SHA照旧；冻结r5审查器仍逐字节不变。产品提取按实际身份TP1/TP2核对，不复制rank；输出每scope/rank页，未通过明确review_error，不自动登记。产品runtime-identity须来自真实Worker部署观察；不能从本页或镜像标签补值。

## 5. 两条CANN路线

Qwen模型1：v0.26.0rc1-a3 / CANN9.1.0。DeepSeek模型2：v0.32.0rc1-a3 / CANN9.2.1。Native库不能跨路线共用；另一CANN上构建的库属于新组合，重新审查。该约定适用于r6后续步骤与后继发行，不改已冻结r6包。
