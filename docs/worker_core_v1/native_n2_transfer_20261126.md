# Native sidecar + N2 开发线工具交付

**开发线工具，不属于 r8 冻结包。** 独立解压，不覆盖r8安装目录，不升级wheel、不改许可SHA。不是参考适配器包，不授予33或Native的A3资格。由负责人转交，codex不直接联系黄区。无需NPU、无需为了本工具重启服务。

内容：根目录 `dcat-native-build.json`、`tools/` 下三个N2文件、空的包标记、说明、离线测试、校验器、manifest、SHA256SUMS。Python3.10+标准库足够，不需要r8源码、torch、tools-r1或Git。

| 文件 | 字节 | SHA256 |
|---|---:|---|
|dcat-native-build.json|989|c61f1a27a78e27d65cb714f786a5f1015d64754550460027097ad0b04a973ea9|
|tools/extract_fault33_launch_host.py|3601|93d0504f82cbbd7a247fe23803dbeef4c7706afcffc50938ff86341ffaa426df|
|tools/extract_fault33_existing.py|6480|2c84f71cae4ef00e6b98e94974e5bd57faa7f2edc5f144aa179a16ab4983606c|
|tools/review_launch_capture_v2.py|4013|66ee176da78971091b2e2108d7f296123bc6abeb5b909862f99a0fb38268743f|

四文件与开发提交57b6ca76完全一致，review_launch_capture_v2.py还与冻结r8一致。ZIP的完整SHA见发行目录DELIVERY-SHA256SUMS和本次交付索引。先在发行目录执行 `sha256sum -c DELIVERY-SHA256SUMS`，解压到新的专用目录，再执行：

```bash
cd /新解压/native-n2-tools-20261126-r1
sha256sum -c SHA256SUMS
python -I -B verify_bundle.py
```

校验器检查所有payload文件SHA，并在只加入解压包路径的隔离解释器中运行15项既有N2测试、一次CLI夹具两遍输出一致检查，校验sidecar固定身份。该测试只在临时目录写夹具，不触碰真实采集原件和推理状态；通过后打印JSON。

## Native：A0 8/9

库SHA必须为 `d9aa0d9488c5c1e8f7fca10c297e4968687ceb6a24866d8346bd33a58afce274`。本文件来自原A2/CANN9.1构建证据，不是新A3构建。运行下列命令，将元数据放在Worker实际加载的库旁；存在不同同名文件时停止，不覆盖：

```bash
ROOT=/新解压/native-n2-tools-20261126-r1
LIB=/Worker实际加载的Native库路径
python - "$LIB" "$ROOT/dcat-native-build.json" <<'PY'
import hashlib,sys
from pathlib import Path
lib,meta=map(Path,sys.argv[1:]);raw=meta.read_bytes()
assert hashlib.sha256(lib.read_bytes()).hexdigest()=='d9aa0d9488c5c1e8f7fca10c297e4968687ceb6a24866d8346bd33a58afce274'
assert hashlib.sha256(raw).hexdigest()=='c61f1a27a78e27d65cb714f786a5f1015d64754550460027097ad0b04a973ea9'
dst=lib.with_name('dcat-native-build.json')
if dst.exists():
    assert dst.read_bytes()==raw, 'EXISTING_METADATA_DIFFERS'
else:
    with dst.open('xb') as f: f.write(raw)
print(dst,hashlib.sha256(dst.read_bytes()).hexdigest())
PY
```

使用既有可信RPC在rank0、rank1重新observe_identity；回传实际库SHA、build_metadata_sha256=c61f…及完整build_metadata，确认实际映射CANN9.1.0。旧回执的null不改。详细原构建依据见 `docs/native-build.md`；其中“蓝区重现生成”命令需要开发仓库，不是黄区安装前置步骤。

## N2：CPU离线，不等负例、不占卡

```bash
cd /新解压/native-n2-tools-20261126-r1
export PYTHONPATH="$PWD"
python -B tools/extract_fault33_launch_host.py \
  --capture-root /已有六条原件共同根目录 \
  --api /真实API字段.json \
  --plan /n2-map.json
```

三个工具必须一起使用本目录版本。pointers指向真实API键；三形状×两rank仍要求六条；tiling不填写猜测值，保留null；blockDim取80字节header；block_table有效范围缺证就null。输出只到stdout，每行≤4096字节，完整host/holder分块运输，不修改原件。按行原样回传，不手拼N2事实。

退出0仅说明launch/host运输检查通过；`field_extraction_complete`、`ready_for_control_recompute`和`qualification_granted`仍false。不能据此进入33窗口二。详细schema和边界见 `docs/n2-launch-host.md`。

## 本轮负例与登记

第二次500已据执行路径定性为情况A，残留来自哪次操作仍待审计，见 `docs/owner-review.md`。黄区新实例各用例前后都保存逐rank审计、owner/generation、PID、RPC序号、返回体、退出码及同引擎正对照。

首次enable拒绝后exit5若在新实例不再出现，保留为r8.2待办；若再出现，codex依据同次运行证据做最小修复。r8.1不夹带代码修复。合格负例到达后，七项草稿才填negative_controls；Native三项再核对带sidecar双rank身份、各项完整验收和相关负例，满足后补入并冻结r8.1。当前不凭文字“通过”提前登记。Batch、under4、over4本轮不安排重跑。
