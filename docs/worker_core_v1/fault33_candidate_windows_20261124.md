# 33：公开源码候选与 A3 两个只读窗口

## r2勘误：本单必须配套r2采集器

2026-11-24完整报告只作为问题定位和可复用身份/金丝雀的文本证据，不算本单窗口一通过。用`fault33-evidence-candidate-20261124-r2`替代r1及黄区自改版本；冻结的r7.4/r7.4.1产品包不动。完整回传要求见`yellow_intake_20261124_fault33_full.md`。

r2新增`aclrtBinaryGetFunction`按名句柄登记，每条二进制旁保存`.bin.binding.json`。name-only的header临时key为0，审查输出key=null；它没有真实tiling key，不能进入候选解码。后到的ByEntry只更新之后的launch，早期记录不追补。候选CLI要求sidecar的rank进程内pid/serial/entry与bin一致；不能删除sidecar绕过检查。

初次三形状窗口必须`DCAT33_CAPTURE_ALL=1`，完整性检查要求`binding_metadata=true`，所有name-only调用单列。仅当目标FIA具有真实entry且与API唯一关联时才解码。unknown=0不替代这一门禁。过滤key模式不能当作全窗口普查。

在新的工具目录中，先运行`sha256sum -c SHA256SUMS`（补充件外层清单），再校验manifest中的逐文件SHA。附带的aarch64库是蓝区SDK构建参考，不自动获得A3资格；A3使用实际SDK重建：

```bash
python tools/build_fault33_capture.py \
  --source experiments/worker_core_v11_20261118/paged33_launch_capture_v2.cpp \
  --sdk "$ASCEND_HOME_PATH" --output "$WORK/capture-r2-build" \
  --source-commit "$(python -c 'import json; print(json.load(open("manifest.json"))["source_commit"])')"
```

输出目录必须不存在。保留build-receipt.json及其SHA，随后按下文既有可信RPC、LD_PRELOAD链、ARMED及flush步骤加载`$WORK/capture-r2-build/capture-v2-r2.so`，核对实际映射的库SHA。不得只改库文件名而继续加载旧库。两个窗口用不同实例和新目录，不删除上一轮原件。

2026-11-24负责人已批准五条标准。本单替代旧执行单中“必须证明二进制与源码一一对应”的前置要求；旧冻结补充件不改。**不把≥99%当作解码正确的充分条件**。原件留黄区，交负责人转发，蓝区不直接要求黄区运行。

范围：Qwen3.8 W8A8、A3、TP2、0.26.0rc1-a3、运行时CANN9.1.0；33保持关闭，不改模型输出。r7.4只含采集器，以下候选解码在r7.5开发线；没有完成适配或授予资格。

## 已核对的候选来源

公开仓库 `https://gitcode.com/cann/ops-transformer.git`，tag `v9.1.0`，commit `c77c0829f1f0f0b8f882db1d9b051a758b605472`。逐文件Git blob SHA见 `fault33_public_candidate_20261124.json`，不是Windows换行转换后的工作树SHA。

当前仅支持 `FAInferTilingData` 候选。已在124同版本SDK中无设备编译三组真实宏声明，得到coreNode=1248、splitNode=1040、FAInferTilingData=2488字节；布局见 `fault33_sdk_layout_candidate_20261124.json`。这只证明该SDK编译的候选布局，**不证明A3安装的算子选择了这个serializer**。还需在A3实际SDK复核布局，核对字段边界和实际分支。

源码入口有DAV3510、DAV5102和其他架构的不同分支；不可用“A3”名称直接推断分支。公开候选中的key `5100000000010200203` 是注册的一项；实际key不同就停止当前解码。sparseMode=3与MASK_SPEC的关系还依赖mask路径，不能仅用key末位3反推sparseMode。

## 窗口一：完整launch与三组真实形状

使用原 `fault33_a3_evidence_execution_20261124.md` 的编译、鉴权RPC、ARMED、同进程flush和完整性检查步骤。两rank的序号、PID与回包逐一关联。不补采已收到且未改变的9库身份，不单独补逐位金丝雀。

一个专用TP2实例、同一个采集窗口中串行发送三组固定token输入，输入长度分别128、512、2304，均生成4 token。记录实际decode KV长度；预期跨越128/512边界只是选例目的，**不得用预期长度填实际字段**。每组选择相同full-attention模块的首个真实decode调用，两个rank分别保留。若实际shape未形成三组独立长度则本门禁失败。

每个调用保存实际API形状、布局、dtype、Q/K/V维度、head数、block_size、block_table有效范围、实际序列长度、mask/sparseMode、调用序号。与launch匹配必须唯一；不能靠“最近序号”或预期key强配。

两rank均运行：

```bash
python tools/review_launch_capture_v2.py "$WORK/capture" --pid "$RANK_PID" --output "$WORK/capture-review.json"
```

要求complete、finalized为true，pending_writes和所有drop/error为0。旧7928条/最大8191的截断原件不能替代。窗口封存后不能在同进程重新ARMED；第二窗口使用新实例和目录。

## 离线候选检查（不占NPU）

在新目录取得并锁定候选，不下载/替换算子二进制：

```bash
git clone --depth 1 --branch v9.1.0 https://gitcode.com/cann/ops-transformer.git "$WORK/ops-candidate"
test "$(git -C "$WORK/ops-candidate" rev-parse HEAD)" = c77c0829f1f0f0b8f882db1d9b051a758b605472
python tools/fault33_candidate_schema.py "$WORK/ops-candidate" --output "$WORK/layout"
nm -D "$ASCEND_HOME_PATH/lib64/libregister.so" | c++filt > "$WORK/layout/register-symbols.txt"
# 先核对 GeLogError 的字符串ABI：124实际是旧ABI(0)。A3必须依据本机符号确认，
# __cxx11::basic_string 为1，旧 std::basic_string 为0；符号缺失则停止。
export CANN_CXX11_ABI=0  # 只在本机符号同样确认旧ABI后执行
c++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI="$CANN_CXX11_ABI" "$WORK/layout/layout_probe.cpp" \
  -I "$ASCEND_HOME_PATH/include" -I "$ASCEND_HOME_PATH/include/experiment" \
  -L "$ASCEND_HOME_PATH/lib64" -lregister -lc_sec -o "$WORK/layout/probe"
"$WORK/layout/probe" > "$WORK/layout/sdk-layout.json"
sha256sum "$ASCEND_HOME_PATH/include/register/tilingdata_base.h" "$ASCEND_HOME_PATH/lib64/libregister.so"
```

布局中的类型、数组长度、嵌套偏移、padding、总长度均要自洽。探针同时通过真实SDK setter给52个逻辑字段（数组覆盖所有元素）写不同标记，调用SaveToBuffer，核对序列化偏移及缓冲区两端守卫；124已通过，A3仍需本机运行。输出marker_roundtrip必须passed=true、fields=52、buffer_guards=true。初次无设备编译暴露了libc_sec依赖与旧字符串ABI，已经按实际库符号修正，未增加stub或替代宏。

`actual-api.json`逐调用只填写原始API观察推导的整数：numHeads、kvHeads、embeddingSize、embeddingSizeV、batch、blockSize、maxQSeqlen、maxKvSeqlen、sparseMode。缺失或无法解释就停止，不填profile默认值。holder索引须由实际placeholder表与指针边界审核确定，不能固定沿用A2的296偏移。

```bash
python tools/decode_fault33_candidate.py "$LAUNCH_BIN" \
  --layout "$WORK/layout/sdk-layout.json" --metadata "$ACTUAL_API_JSON" \
  --holder-index "$REVIEWED_HOLDER_INDEX" --reviewed-key "$ACTUAL_REVIEWED_KEY" \
  --output "$WORK/decoded-call.json"
```

对三组×两rank分别执行。shape_consistent必须为true；`serializer_validated=false`和`qualification_granted=false`保留。该工具主动拒绝越界、重叠、截断、未知类型、缺元数据、未审key；它不自动选择分支或证明key覆盖。

对候选key `5100000000010200203`，增加B1/Q1、GQA每组不超过128个Q头的分区自洽检查：

```bash
python tools/fault33_candidate_partitions.py "$WORK/decoded-call.json" \
  --output "$WORK/candidate-partitions.json"
```

它逐core重建每个KV head的合法key区间，并交叉核对splitInfo的head范围、split数量、LSE/O偏移与workspace字节数。512来自已锁定候选头文件的MAX_KV_STACK_LEN，并非A3实测常量；其他key和范围明确拒绝。四项CPU测试包括129/513/2305合成长度、尾块、缺口、错误分支和偏移，不计入三组实际shape门禁。输出始终installed_kernel_correspondence=false；公开host候选的自洽仍不能证明A3 kernel采用该算法。

## 窗口二：固定真实Q/K/V的控制重算

**须先收到窗口一并完成分组字段审查，才下达这一窗口的具体配置。** 当前仅新增上述窄范围的候选分区映射，尚无A3实际调用通过证据；不能让黄区自行填A2的512分组。

新实例重放相同三组输入，在实际调用边界只读保留各rank的Q、按真实block_table还原的合法K/V及完整原生输出。必须保留mask与序列边界，证明每个合法key恰好一次；私有张量不进入Git。旧单次 `_paged33_qkv_saved` 采集器不能冒充三形状采集。

审核后的每调用 `dcat.paged33.observed_spans.v1` 记录必须含：qkv_sha256、launch_sha256、serializer_sha256、实际spans（所有head一致时）或per_kv_head_spans（逐KV head列出）和算术假设。仅当源码及实际launch支持 `BF16_MM_FP32_OUT_PBF16_LSE_FP32` 时才运行现有控制工具，否则停下修适配，不试多个算法挑最高一致率。

```bash
python tools/paged33_control_checkpoint.py "$PRIVATE_QKV" \
  --launch-review "$OBSERVED_SPANS_JSON" --device "$LOCAL_LOGICAL_DEVICE" \
  --differences-output "$WORK/private-control-differences.json" > "$WORK/control.json"
```

每组、每rank均独立达到完整输出逐位一致率≥99%，两次控制重算精确稳定。控制工具现已输出完整私有差异清单：query/head/column、原生值/控制值及BF16原始位；summary仅记录清单SHA、字节数和条数。清单包含全部参与合并的分区。最终输出是这些分区的合并值，不能把某个输出差异归因于单一分区；single_partition_attribution保持null。私有清单不提交Git，不截断以掩盖不一致。3项CPU测试覆盖坐标、符号零和非有限值拒绝。

逐head控制重算与三形状采集接线现已实现，但尚无A3实测。设置 `DCAT_CAPTURE_LENGTHS='[129,513,2305]'` 运行 `paged33_native_probe.py` 时，会依次发送128/512/2304-token prompt，只保存每个rank首次匹配的三组真实decode调用。长度是候选取证矩阵，不是A3分组常量；窗口一不支持该矩阵时停止并重新审查。

新增 `tools/paged33_private_capture.py` 按真实block_table重建key集合、保留尾块、实际请求ID、API调用序号、文件字节数和SHA，私有目录必须新建。mask非空、KV量化、重复页面、越界、缺shape均拒绝；不能默默忽略mask。API序号不是launch序号，manifest中的launch_serial保持null，必须结合完整launch采集及真实调用关联审核后才能生成observed_spans。调用包装只覆盖实际走该Python API的路径；没有捕获到三组时complete=false，不用其他路径的张量补齐。

控制工具显式接受每个KV head独立分区，并核对GQA头组映射、完整key集合和分区覆盖。20项相关CPU测试通过，包括相同分区与旧控制逐位一致、不同head分区、缺口/重复、三形状保存、尾块与mask拒绝。CPU合成张量不计A3三形状证据。

窗口二仍等待A3真实coreInfo/splitInfo、实际调用与launch一一对应，以及算术假设审核。可交付采集工具，不应将该等待写成A3控制已通过；差异清单工具通过也不是A3控制通过。

## 放行与停止

五项全部成立才进入r7.5注入适配：①候选身份锁定；②真实字段边界验证；③至少三组实际形状×两rank自洽；④每组完整控制≥99%且定位不一致；⑤任一不通过停止并说明具体失败项。CPU合成三形状、A2 SDK布局和A2数值结果均不能替代A3门禁。

两owner注入、输出变化和恢复是适配后的独立验收，不在本单的只读窗口中执行。结论可能是当前候选不适用；不降低阈值、不忽略失败shape、不宣称历史默认kernel已复刻。
