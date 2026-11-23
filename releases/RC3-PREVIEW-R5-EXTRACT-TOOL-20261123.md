# RC3 r5 独立提取工具补充件（2026-11-23）

本补充件单独发行，不改r5标签、wheel、源码包，不并入r5.1。r5.1仍只登记。未向黄区发送内容，由负责人转交。

[下载 ZIP](registration-extract-20261123-r1/registration-extract.zip) · [文件清单](registration-extract-20261123-r1/manifest.json) · [124实际提取复核](registration-extract-20261123-r1/actual-extraction-check.json)

ZIP SHA256：`890696443b8cde476818ac03afcaae94c520730a6981aceddde27067c3a98d81`。

| 文件 | SHA256 |
|---|---|
| tools/extract_registration_record.py | `5a6c545ee0a191adada0c6a5cbfc5c73746621a2be0c2190bdfea0c43c875a09` |
| tools/extract_dsv4_record.py | `33e9fdad80120e522606be098340ae163ebef17b13db10650a6d097f60b32b50` |
| 工具包中的 tools/review_continuous_public.py | `deb3c5c0d600f7bcae7d0997cfd883591ece87231ed8128e486edb5d823ffadc` |

审查器从冻结r5源码包 `tools/review_continuous_public.py` 逐字节复制，SHA为deb3c5c0d600f7bcae7d0997cfd883591ece87231ed8128e486edb5d823ffadc。当前开发线的审查器已有33扩展，因此工具在仓库内固定加载tools/_r5副本；独立ZIP内以原tools/review_continuous_public.py路径提供。运行时强制校验上述SHA，不静默使用开发线审查器。

## 环境与校验

Python 3.10+，仅标准库；不需要torch/NPU/vLLM、不需要r5源码目录。解压新目录并cd进去。需要读黄区本地summary和原回执；可选传r5的release-manifest.json以补齐有凭据的包身份。禁止覆盖r5源码目录。124实测为无设备的Python标准库过程。

```bash
unzip registration-extract.zip -d registration-extract
cd registration-extract
python -B -c "import hashlib,json,pathlib; r=pathlib.Path('.'); m=json.loads((r/'manifest.json').read_bytes()); assert all(len((r/f['path']).read_bytes())==f['bytes'] and hashlib.sha256((r/f['path']).read_bytes()).hexdigest()==f['sha256'] for f in m['files']); print('FILES_EXACT')"
```

## 14个组合的逐条命令

以下路径是布局示例，替换为实际现存路径；不需要搬动/修改原件。SUMMARY、RECEIPT每项必须对应同一次运行。`R5`指冻结发行目录。各命令输出一段JSON，原样粘贴，最多14段。完整summary和原件留黄区，只回传文件名、字节数和SHA。

```bash
EVIDENCE=/path/to/existing-evidence
R5=/path/to/demoncatpfi-rc3-preview-20261122-r5
python -B -m tools.extract_registration_record "$EVIDENCE/tp1/28/summary.json" "$EVIDENCE/tp1/28/continuous_public.json" --release-manifest "$R5/release-manifest.json"
python -B -m tools.extract_registration_record "$EVIDENCE/tp1/Batch/summary.json" "$EVIDENCE/tp1/Batch/continuous_public.json" --release-manifest "$R5/release-manifest.json"
python -B -m tools.extract_registration_record "$EVIDENCE/tp1/18/summary.json" "$EVIDENCE/tp1/18/continuous_public.json" --release-manifest "$R5/release-manifest.json"
python -B -m tools.extract_registration_record "$EVIDENCE/tp1/22/summary.json" "$EVIDENCE/tp1/22/continuous_public.json" --release-manifest "$R5/release-manifest.json"
python -B -m tools.extract_registration_record "$EVIDENCE/tp1/39op/summary.json" "$EVIDENCE/tp1/39op/continuous_public.json" --release-manifest "$R5/release-manifest.json"
python -B -m tools.extract_registration_record "$EVIDENCE/tp1/39mr/summary.json" "$EVIDENCE/tp1/39mr/continuous_public.json" --release-manifest "$R5/release-manifest.json"
python -B -m tools.extract_registration_record "$EVIDENCE/tp1/17/summary.json" "$EVIDENCE/tp1/17/continuous_public.json" --release-manifest "$R5/release-manifest.json"
python -B -m tools.extract_registration_record "$EVIDENCE/tp2/28/summary.json" "$EVIDENCE/tp2/28/continuous_public.json" --release-manifest "$R5/release-manifest.json"
python -B -m tools.extract_registration_record "$EVIDENCE/tp2/Batch/summary.json" "$EVIDENCE/tp2/Batch/continuous_public.json" --release-manifest "$R5/release-manifest.json"
python -B -m tools.extract_registration_record "$EVIDENCE/tp2/18/summary.json" "$EVIDENCE/tp2/18/continuous_public.json" --release-manifest "$R5/release-manifest.json"
python -B -m tools.extract_registration_record "$EVIDENCE/tp2/22/summary.json" "$EVIDENCE/tp2/22/continuous_public.json" --release-manifest "$R5/release-manifest.json"
python -B -m tools.extract_registration_record "$EVIDENCE/tp2/39op/summary.json" "$EVIDENCE/tp2/39op/continuous_public.json" --release-manifest "$R5/release-manifest.json"
python -B -m tools.extract_registration_record "$EVIDENCE/tp2/39mr/summary.json" "$EVIDENCE/tp2/39mr/continuous_public.json" --release-manifest "$R5/release-manifest.json"
python -B -m tools.extract_registration_record "$EVIDENCE/tp2/17/summary.json" "$EVIDENCE/tp2/17/continuous_public.json" --release-manifest "$R5/release-manifest.json"
```

本地验证：可将本条stdout重定向保存成自己的extract.json（工具本身只读、不写状态），然后：

```bash
python -B -m tools.extract_registration_record --verify /path/to/extract.json
```

`consistent=true`说明提取文本哈希、计数、复核与summary关联一致；`registration_ready`另外核对r5/A3身份。蓝区样例可以consistent=true而registration_ready=false，不能把A2录成A3。缺失字段为null，登记前必须补足；重跑审查失败在review字段明确列出，不隐去失败。字段顺序与规范化算法详见[回传协议](../docs/worker_core_v1/registration_text_return_20261123.md)。

## 实际检验

CPU：50项通过，0失败/0错误，其中test_registration_extract为11项。覆盖TP2大小/可重复性、摘要矛盾、缺字段null、失败输出、篡改、硬限、DeepSeek逐rank分页及缺失项、固定图片、r5审查器原字节。

124三项是r5沿用的早期A2/0.26/Qwen3.8 W8A8/TP2验收证据，不冒充重新以r5许可运行的回执。已有原回执不修改；用本补充件的冻结r5审查器在黄区式标准库环境重建对应summary，再运行提取。旧记录缺少后来新增的身份字段，这一缺失被保留，不能用于黄区登记。

| 机制 | 原回执窗口 | UTF-8含末尾换行 | 重复输出逐字节相同 | --verify |
|---|---|---:|---|---|
| 28 | a5 | 1172 | 是 | consistent=true；登记资格=false |
| Batch | a17 | 1223 | 是 | consistent=true；登记资格=false |
| 17 | vision17-a3 | 1132 | 是 | consistent=true；登记资格=false |

全部小于2048字节。目标上限2048，硬限4096：超过目标提示原因/长度，超过硬限拒绝输出，不删字段。以下为真实28样例（null即缺失，不是默认值）：

```json
{"conflicts":[],"d":["A2",null,"0.26.0","0.26.0rc1","2.10.0+cpu","2.10.0.post4",["9.1.0"],null,"191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",null,null,null,null,null,2,1,1,true,"torch.bfloat16",true],"devices":[[0,null],[1,null]],"extract_sha256":"14287f29763bfd908f8f0e32bd3d7ff211e3ad9ffcd303fbc5eeaaa9cf45ef26","files":[["local-summary-28-qwen.json",6224,"7935937b4facf1a17c12521cff0ce0ff874aa1909d0132ce10d5a8b00d8c91e5"],["continuous_public.json",114025689,"b0e626fb3afce62c0876ad28b795490c8049bd26f7a1da06ac32b4c81941d10e"]],"m":"28.fia.denominator_replay","p":[null,null,"44d766c31a2d8e57436445008779f34268b62f85afe515447b5a0c3d6d30723e"],"review":[true,null,true,true,"deb3c5c0d600f7bcae7d0997cfd883591ece87231ed8128e486edb5d823ffadc"],"s":[["worker-group-03cbbd5b98f940a69682af28aaffbcbc",[[0,[64,64,64,64,64,64,64,0],true,[true,true,true]],[1,[64,64,64,64,64,64,64,0],true,[true,true,true]]],[true,true],1.426533834776699],["worker-group-35ab95fb2b2c4ba3953385bbcbc69ee3",[[0,[64,64,64,64,64,64,64,0],true,[true,true,true]],[1,[64,64,64,64,64,64,64,0],true,[true,true,true]]],[true,true],1.3415848069907654]],"schema":"dcat.reg.extract.v1"}
```

## DeepSeek回传

```bash
python -B -m tools.extract_dsv4_record noise /path/noise-summary.json
python -B -m tools.extract_dsv4_record observe /path/observe.json
python -B -m tools.extract_dsv4_record observe /path/source-fingerprints.json
```

按rank/site分页，每行≤4096字节，全部page须回传，私有数组留黄区。新标定summary明确输出有限性/token稳定；旧summary缺字段保持null。缺失源码指纹不推定支持。

## 17默认图调整

开发线已统一vision_input.deterministic_image：64×64，R=4*x、G=4*y、B=4*((x+y)%64)。continuous_public_probe不再要求黄区--image；--image可选；产品金丝雀相同默认图，用户图片仍来自用户请求。像素SHA记录在回执。r5已有17结果按当时黄区固定图片方式审查，不要求重跑。此改动没有回写r5包。A3上TP2启动前设置VLLM_WORKER_MULTIPROC_METHOD=spawn，产品模式也一样，避免Invalid thread pool。

P0交付后继续P1产品迁移；新预览r6尚未冻结。
