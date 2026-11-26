# d9aa Native 构建身份补齐（2026-11-26）

已生成 `dcat-native-build.json`，989 字节，SHA256：
`c61f1a27a78e27d65cb714f786a5f1015d64754550460027097ad0b04a973ea9`。

只对应库 SHA `d9aa0d9488c5c1e8f7fca10c297e4968687ceb6a24866d8346bd33a58afce274`。依据 `../r7_native_a2_20261123/build.json`（SHA `e9b36098fe658dc2de1ef3ba3e894cbc2dcfd6845771c926ed5c8f4764f643eb`）及同目录 `build_context.json`。原构建组合是 **A2 / CANN9.1.0**，不是重新进行的 A3 构建。供 Qwen 0.26 / CANN9.1.0 的 41、26fia、20 核对身份，不赋予 A3 产品资格。

核对了源码提交 `82c5b7f81b0155cec958aac9535a20209beb75d4` 的五个源文件、构建 recipe、编译器版本、SDK 版本文件及其 SHA、各构建命令成功状态和库 SHA。`source_release` 如实写为 `demoncatpfi-rc3-preview-20261123-r6-5-g82c5b7f8`，不冒写为后来冻结的 r7 标签。未重编、未修改库文件或冻结包。

蓝区可重现生成（需要完整 Git 历史及 Python 3.10+，在仓库根目录执行）：

```bash
python tools/generate_native_build_metadata.py \
  --receipt docs/worker_core_v1/r7_native_a2_20261123/build.json \
  --context docs/worker_core_v1/r7_native_a2_20261123/build_context.json \
  --output docs/worker_core_v1/native_build_metadata_d9aa_20261126/dcat-native-build.json
```

黄区无需重建。把本目录 JSON 作为独立身份文件放到 **Worker 实际加载的 d9aa 库旁边**。先核对库和 JSON，存在内容不同的同名文件时停止，不覆盖：

```bash
LIB=/实际部署/libdcat_postwrite.so
META=/本次交付/docs/worker_core_v1/native_build_metadata_d9aa_20261126/dcat-native-build.json
python - "$LIB" "$META" <<'PY'
import hashlib,sys
from pathlib import Path
lib,src=map(Path,sys.argv[1:]);raw=src.read_bytes()
assert hashlib.sha256(lib.read_bytes()).hexdigest()=='d9aa0d9488c5c1e8f7fca10c297e4968687ceb6a24866d8346bd33a58afce274'
assert hashlib.sha256(raw).hexdigest()=='c61f1a27a78e27d65cb714f786a5f1015d64754550460027097ad0b04a973ea9'
dst=lib.with_name('dcat-native-build.json')
if dst.exists():
    assert dst.read_bytes()==raw, 'EXISTING_METADATA_DIFFERS'
else:
    with dst.open('xb') as f: f.write(raw)
print(dst, hashlib.sha256(dst.read_bytes()).hexdigest())
PY
```

`LIB` 用部署文件中的真实路径，示例文件名不作为默认值。若容器目录只读，应在其对应部署挂载源放置，再确认 Worker 可读；不要覆盖冻结发行目录。

随后沿用当前可信身份观测入口，在 rank0、rank1 重新采集 `native_identity`：实际/声明库 SHA 都须为 d9aa…；`build_metadata_sha256` 为 c61f…；`build_metadata` 全文与本文件一致；运行时映射的 CANN 仍为9.1.0。旧回执中的 null 保留，不事后伪装为运行时已经观测到。三项仍等实际双 rank 身份及有效负例才能登记。该库不能借给 DeepSeek / CANN9.2.1 组合。

CPU 检查覆盖原证据重建、回执/源文件/recipe 篡改、缺元数据拒绝、匹配元数据通过及 CANN 不匹配拒绝；使用身份夹具不代表 A3 已放置文件。
