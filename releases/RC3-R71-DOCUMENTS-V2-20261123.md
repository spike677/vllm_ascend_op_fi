# r7.1 文档补充件 v2

供负责人转交；六份文档，ZIP 内有逐文件 SHA256 清单。仅文档补充，不改 r7.1 标签、三 wheel 或源码包；r8 不随本包下发。

- ZIP：`r71-docs-20261123-v2/r71-docs-20261123-v2.zip`
- ZIP SHA256：`71ae235a679f1515aaaeee949ea16ca14cac8fa0014251d8ab67dbfa5f806fab`
- 总单统一黄区 `python -m demoncat.cli fault ...`，Native JSON / 可信 bootstrap 顺序已补齐。
- `capabilities()` 是当前 profile 的候选范围，enable 仍需门禁；旧 capabilities.py 不是产品资格目录。
- 总单修正重复 attach：capabilities 内部已做 census attach，随后直接 enable。
- N 批历史文件名和正文保留溯源；今晚统一 r7.1、TP2，以总单 v2 为准。
- A2 Native capabilities 实测回传单另行补充；本 ZIP 不伪造运行时输出。

解压后执行 `sha256sum -c SHA256SUMS` 应在 ZIP 同级目录先核验外包；内包逐项对照 manifest.json。链接到其他历史文件仅作溯源，未包含运行软件，安装仍用冻结 r7.1 完整发行包。
