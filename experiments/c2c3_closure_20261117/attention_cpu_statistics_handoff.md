# New attention CPU FP64 statistics capture

The actual reviewed msprobe processor source is
`e981aa6fe9afa1e7a1f0654db44166d7c199674b0727d8d7ffa40626b1074011`.
Its `get_stat_info` explicitly skips nonscalar floating Mean/Norm when
`precision=high` (lines 423–429). The old 28 records therefore remain valid
observations of that collector: 10 finite / 14 null fields, with no independent
first difference. This source finding does not rewrite their interpretation.

The opt-in `--statistics-backend cpu-fp64-v1` is a different collector identity
for a fresh B/F pair. It changes only the actual reviewed
`PytorchDataProcessor.get_stat_info` method in that process. Source SHA, compiled
code, globals and cleanup ownership are checked. Original package files,
statistics/mix configuration, record selection, stack, construct and JSON writer
are unchanged. The runtime replacement and adapter SHA are disclosed in
`collector_identity.json`, `statistics_backend.json`, result and launch receipts.

Complete nonempty dense nonscalar FP16/BF16/FP32/FP64 tensors are copied to CPU
in their original dtype, then converted on CPU to FP64 for Max/Min/Mean/L2 norm.
There is no NPU FP64 conversion, sampling, tensor scaling, injection-log access,
old JSON input or nonfinite replacement. Original bool/integer/complex/scalar,
empty/meta/NZ and unsupported handling remains, including null/NaN/Inf.
Each tensor is bounded by 128 MiB, total original bytes copied by 512 MiB, and
256 statistics calls per process. Overflow fails the experiment; it never
selects a partial tensor. For B8 FIA, one Q/K/V/output is 192 MiB total, so the
API plus module observations fit the 512 MiB total bound. CPU scalar reductions
pass through the original writer's `.item()` conversion without downcasting.

Only root runs the following commands, using a new integrated source snapshot
`S` below. The controller creates a new device-free `runc` container in the exact
CANN 9.1 image, with the actual collector package read-only. It executes a real
CPU model through PrecisionDebugger and checks the raw dump/stack/construct and
serialized FP64 values. There are no NPU devices, driver mounts or network.

```sh
R=/home/17-fault-c2c3-20261117
S=$R/source-NEW
python3 "$S/experiments/c2c3_closure_20261117/run_attention_statistics_preflight_124.py" \
  --source "$S" --out "$R/attention-statistics-cpu-a1"
```

After `CPU_COLLECTOR_AND_WRITER_VERIFIED`, create exactly four fresh cases; the
generator verifies the CPU receipt matches the adapter in `S`. It starts no
container and makes no NPU calls.

```sh
python3 "$S/experiments/c2c3_closure_20261117/prepare_attention_cpu_statistics_cases.py" \
  --source "$S" \
  --registered-legal /home/17-fault-engineering-20261117/fia-legal-registered-a1 \
  --registered-replay /home/17-fault-engineering-20261117/fia-replay-registered-a1 \
  --schema /home/17-fault-engineering-20261117/fia-schema-a1/schema \
  --observer "$R/attention-preflight-a4/cann910" \
  --cpu-preflight "$R/attention-statistics-cpu-a1/probe/result.json" \
  --card 5 --out "$R/attention28-cpu-fp64-a1-cases.json"
```

Root submits those four cases with its existing lock/resource queue into a new
`attention28-cpu-fp64-a1` directory. All four use the same in-container source,
config, collector and evidence paths, with only the mechanism/phase selection
changing. Actual final arguments, registered binary, launch and complete output
checks remain enabled. The fresh comparator receives only the new B/F raw
statistics, stack, construct and config, fixed rtol=1e-5 / atol=1e-8; null,
nonfinite or alignment failure remains a failure. No particular first difference
is predicted, and CPU tests or a new backend do not grant E5/C2/C3 or FA/Worker
qualification. Old B/F/OFF directories and receipts remain untouched.

Local CPU validation: six new backend tests plus existing fresh qualification
and launch tests (24 tests; one existing Windows symlink skip). They cover
complete noncontiguous data, FP64 cancellation, input immutability, exact original
null/scalar/bool handling, NaN/Inf, per-tensor/total budgets, source/globals
substitution, and refusal to overwrite a foreign method during cleanup. The
real-package CPU and new NPU results are pending root execution.
