# Fault26 validation evidence

This directory carries the validation material needed to audit the Fault26
developer-mode implementation without committing hundreds of megabytes of raw
JSONL as individual Git objects.

## Archive

`fault26_evidence_20260818.tar.gz` contains the complete local
`.fault26_evidence` tree captured through Phase 30:

- 92 files;
- 353,174,175 uncompressed bytes;
- read-only topology/lifetime traces;
- arm/query/disarm RPC responses;
- baseline, fault, and recovery model responses;
- bounded multi-hit and all-eligible-hit JSONL evidence.

The archive contains no SSH private key, password, authorization header, GitHub
token, model weights, or container image. No local compiled `.so`/`.o` artifact
was present when this evidence package was created; binaries remain reproducible
from the checked-in build scripts and C++ sources.

Archive SHA-256:

```text
BF3AF43C9A1886ECFF7D039C0CB59CB5D5C58465D18425E66F87678C0E67DE99
```

Restore from the repository root:

```bash
tar -xzf fault20_fast_validation/evidence/fault26_evidence_20260818.tar.gz
```

The directly reviewable Markdown reports remain under
`fault20_fast_validation/reports/`. In particular, the Phase 27, Phase 28/29,
and Phase 30 reports link each conclusion to the corresponding path inside this
archive.
