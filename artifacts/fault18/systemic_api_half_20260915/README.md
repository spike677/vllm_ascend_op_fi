# Fault18 baseline and fault msprobe dumps

These archives contain only the original dump directories from the 2026-09-15 systemic normalization-output gain experiment.

| Archive | Case | Files | Tensor files | Bytes |
| --- | --- | ---: | ---: | ---: |
| Fault18_Baseline_B_ON.zip | B_ON: normal baseline, msprobe ON | 33 | 30 | 1214865 |
| Fault18_Fault_F_ON.zip | F_ON: output gain 0.5, msprobe ON | 33 | 30 | 1218232 |

Each archive retains its B_ON or F_ON top-level directory, dump_config.json, step0/step1 rank0 debug.json, and .pt tensor files. Every archived file was SHA256-checked against its local source; file counts also match.

These are bounded explicit PrecisionDebugger.save samples, not a full automatic API trace. Sampling covers rank0; fault execution covered both ranks. Service logs, event JSONL, responses, source code, and other experimental groups are not included in this upload.

F_ON produced 1024 tokens with finish_reason=length and 509 occurrences of "bustling". This is a normalization-output effect proxy, not proof of native reduction nondeterminism. The full experiment separately verified recovery; these two archives alone are not the complete recovery evidence package.

## Archive SHA256

```text
ead12deb629d8f70735cb3fab02746385e118d98f4cc92ae3639f9da7cb1ab8f  Fault18_Baseline_B_ON.zip
8cdd00bb4bf731d79b3aedab450de9efc9fdb4c2023c29c0d4656bcf8e3b44ac  Fault18_Fault_F_ON.zip
```

Source snapshot: local research commit 3f6e17a6b205327f7f0075f7021e0f7fc23b4e50, experiments/fault18_determinism/deliveries/systemic_api_half_20260915/dumps/{B_ON,F_ON}. That separate commit is not included by this archive-only branch.
