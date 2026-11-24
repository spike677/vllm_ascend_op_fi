# N2：从旧 capture 补提，不启动 NPU

适用于 K3 审查的黄区 `1fecdc87…` 采集器。工具只读文件并向 stdout 输出；不补默认字段、不改原 capture、不生成假 r2 sidecar。r2 的新采集、ABI/布局探针步骤见 `fault33_candidate_windows_20261124.md`。补充 ZIP 中这些文件都在 `docs/worker_core_v1/`，工具在 `tools/`。

## 六条记录与字段

每组 prompt 128/512/2304，各 rank0/rank1，选择同一 full attention 模块的一个 Q1 paged decode。保留原窗口、rank→PID、模块名和 API 调用序号。

|字段|来源|缺失处理|
|---|---|---|
|记录号 PID/serial、原始 key、blockDim|80字节 capture header|格式/文件名不符拒绝|
|真实 launch key|ByEntry 记录及唯一调用关系|高位哈希/0 只保留 captured_key，launch_key=null|
|host 字节数和 SHA|原 capture host 区间|不能称为 tiling SHA|
|tiling 字节数和 SHA|审核的 placeholder 索引、serializer长度、host边界|未审定时 null，严禁默认 2488 或 A2 偏移|
|sparseMode、mask（包括明确的 absent）、Q/K/V、序列长度、实际 KV 长度|window1-result.json 对应实际 API 记录|未知 null；不把 prompt 长度当 KV 长度|
|block_table 有效范围|实际请求有效项、页号边界、对应 cache 容量|只知 tensor shape 不等于有效范围|
|API/launch 唯一对应|原 trace 的相关 ID 或同线程嵌套关系|不能取最近 serial；缺关系先退补|

`tools/extract_fault33_existing.py` 用 JSON Pointer 读取原文件字段。由于蓝区未持有完整 window1-result.json，不能预设其字段路径。黄区只映射实际路径；不存在就不填。映射文件不允许复制“预期值”替代源字段。完整原件仍留黄区。

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python tools/review_launch_capture_v2.py "$CAPTURE_DIR" --pid "$RANK0_PID" --output "$WORK/rank0-capture-review.json"
python tools/review_launch_capture_v2.py "$CAPTURE_DIR" --pid "$RANK1_PID" --output "$WORK/rank1-capture-review.json"
python tools/extract_fault33_existing.py --capture-root "$CAPTURE_DIR" \
  --api "$WORK/window1-result.json" --plan "$WORK/n2-map.json" > "$WORK/n2-extract.json"
# exit 2 = 字段未齐，输出仍有效；exit 0 也不代表 serializer/33 准入通过。
```

映射格式（以下仅结构示例，不可原样运行；calls 必须六条，路径需依据原件）：

```json
{"source_sha256":"1fecdc87bb4ea3f923d25d9d57e4bad83da41060d7c986bdc1c2635b0b24a4dc",
 "calls":[{"capture":"launch-<PID>-<SERIAL>.bin","pointers":{
   "rank":"/实际路径/rank","prompt_length":"/实际路径/prompt_length",
   "api":"/实际路径/api","module":"/实际路径/module","phase":"/实际路径/phase",
   "q_shape":"/实际路径/q_shape","k_shape":"/实际路径/k_shape","v_shape":"/实际路径/v_shape",
   "sparse_mode":"/实际路径/sparse_mode","mask":"/实际路径/mask",
   "actual_seq_lengths":"/实际路径/actual_seq_lengths","kv_length":"/实际路径/kv_length",
   "block_table_valid_range":"/实际路径/block_table_valid_range",
   "launch_relation":"/实际路径/launch_relation"}}]}
```

工具接受原件中的关系对象 `{pid,serial,unique:true,method:"recorded_correlation"}` 或 `method:"nested_trace"`。如果实际记录格式不同，先回传对应字段及来源，补适配；不得在原文件手填这个对象。tiling 未知时不填；已审核后可加 `tiling:{holder_index:<实际索引>,bytes:<实际长度>,boundary_evidence:<声明路径/文件SHA/审核记录>}`。工具还会检查 pointer 与 tiling 不重叠、未越出 host 区间。

回传六条提取记录、每 rank 完整性结论、原件/映射/提取 SHA、缺项原因。name-only 或关系缺失不自动要求全部重跑；先查旧 trace。原接口捕获不到的部分再安排针对性窗口。

## ABI / 本机布局（无设备）

使用同包 `fault33_candidate_windows_20261124.md` 的完整命令：锁定公开 commit；`tools/fault33_candidate_schema.py` 生成 layout_probe.cpp；对实际 `libregister.so` 执行 `nm -D | c++filt`，从 GeLogError 的 std::string 符号确认 `_GLIBCXX_USE_CXX11_ABI`，不能沿用 A2 的 0；链接 `-lregister -lc_sec`，运行探针。回传 SDK/header/lib SHA、编译器版本、ABI 符号、布局和 marker_roundtrip。符号缺失或 schema 与真实结构不符时停止。

此流程不改驱动、CANN、Docker或运行中的服务。窗口二控制重算仍待 N2 审查，未发最终分组参数。
