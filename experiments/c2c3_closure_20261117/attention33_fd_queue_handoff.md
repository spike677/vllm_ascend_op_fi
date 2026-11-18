# Native FD stock/private gate

This handoff prepares root-owned work; it does not start a container or NPU.
The existing natural Worker driver serves as the platform preflight and full
baseline. There is no additional synthetic or modified-model preflight case.
Actual aic/FD/key must come from the new baseline host plus independently
observed registered ELF/function/final-argument launch, not from a CPU fixture.

The device-free `tnd33-fai-fd-build-a1` independently passed all 1,163 vendor
artifact hashes/sizes (50,574,274 bytes), the eight unique MIX 1:2 entries and
their registry, and a full CPU replay of the 24,576 saved SDK geometry rows.
The public serializer is 2,488 bytes. This establishes build consistency only.

| Artifact | SHA256 |
| --- | --- |
| completion.json | bdc2183ef5395844c9fc69dbb30b618b2200d9c3cbc85a39be8ebd35acaae472 |
| kernel ELF | 1e8604d4e453f4d6ba4d1cb9c821935b8d6ddfb2e0aad3c21f0eea692c84b7d8 |
| kernel metadata | 43c074b22feab7c09cc0bab0d8def54854ac4a38f388fbf56c4c21d84444e8e9 |
| binary summary registry | 2e2127e909d20e7b84b0e0ae579dc27a9a37d866cd87558dc850c09dfecb4190 |
| CPU geometry matrix | 5a535ad7967094093e485828dc03de289a7c4e843a9e0af6e8a57584cb10f5d6 |
| CPU compile receipt | c8a0ad1a0f88156aff981e1582588307b8a58ebc91ee56681d8edcb812d9e3d0 |
| public schema | 5c1ea1d6cd8ee289c4366faaafb1087307450239e848f4a90555d0a137c25d0a |

The prior stock result SHA is
`cfe31bd94721db16996e326d13fb3725cbced95d536a446addefa4fda12a1d17`.
Its source-a21 driver SHA `bbecf59e02a84944e2e34b4aef9f840944cd275024140aac402224cc1b7e8b90`
differs from source-a27/a42 `0acb9e5da88e8638cca54f3e4a5e3bc1c4806d593143cc647a56bb73428e749d`;
it also ran on card5. It is retained, not reused as this new strict bridge.
The new byte/finite/lineage reader successfully replayed all its 435 payload
files and 589,889,280 bytes read-only as a file-format regression, without
granting new admission or editing the old result.

## First root queue

Freeze the new driver/contract/bridge/generator together (source-a45 below is
the proposed next path). Runtime package code is unchanged by this commit,
so root's existing packages-a18 may be used. Driver identities, all installed
demoncat/fault_injection source/data files, original model weights/tokenizer,
fixed runner/sampling settings and reviewed card/env are recorded afresh in
each process. Hashing these inputs is CPU-only and precedes model construction.

```sh
R=/home/17-fault-c2c3-20261117
python3 -B "$R/source-a45/experiments/c2c3_closure_20261117/prepare_attention33_cases.py" \
  --source "$R/source-a45" --installed "$R/packages-a18/installed" \
  --build-root "$R/tnd33-fai-fd-build-a1/build/run" \
  --observer "$R/attention-preflight-a4/cann851" \
  --model /home/gyl/models/Qwen3-0.6B --stage baseline --card 2 \
  --out "$R/attention33-fd-stock-baseline-cases-a1.json"
```

This emits only `attention33-fd-natural-stock` and
`attention33-fd-natural-baseline`, each bounded to 1800 seconds/48GiB/16 CPUs.
Use root's current owned queue, lock, gate and cleanup policy. Do not run either
while another NPU job owns the resource. Both retain natural1024-token prefill,
first KV1025 decode, five phases, exact complete payload recovery and a final
healthy request. Private baseline records 28 actual calls in each of its four
native phases (112 launch checks); only its measured request is the 28-boundary
candidate denominator. All baseline keys must be the original
`5100000000010200103`, FD=true and native512, with actual observed aic and
native-derived active blocks. No F is emitted at this stage.

## Independent raw bridge and F release

After both cases finish, run the CPU-only comparator using the actual queue
output directory (the example assumes `attention33-fd-stock-baseline-a1`).

```sh
python3 -B "$R/source-a45/experiments/c2c3_closure_20261117/attention33_bridge.py" \
  --stock-run "$R/attention33-fd-stock-baseline-a1/attention33-fd-natural-stock/run" \
  --baseline-run "$R/attention33-fd-stock-baseline-a1/attention33-fd-natural-baseline/run" \
  --build-root "$R/tnd33-fai-fd-build-a1/build/run" \
  --out "$R/attention33-fd-stock-private-bridge-a1.json"
```

The comparator reads every original file, recomputes hashes and finiteness,
directly compares all435 complete payload pairs, checks all natural writer/
page/request bindings, then replays112 final launches from raw observer records
and registered binary files. It rechecks all52 public fields, all FD arrays and
workspace, actual aic, native FD key, actual page capacities and post-init
Norm/FIA-V3 library ownership. Physical addresses and page-allocation ordering
are not equality criteria, but each logical readback must bind its actual used
pages. No value, precision, shape, invalid/missing field, NaN, or valid KV
element is dropped. Its result remains C2=false/C3=false and does not read E5.

Only after a PASS, emit F with the same source, packages, model, card and build:

```sh
python3 -B "$R/source-a45/experiments/c2c3_closure_20261117/prepare_attention33_cases.py" \
  --source "$R/source-a45" --installed "$R/packages-a18/installed" \
  --build-root "$R/tnd33-fai-fd-build-a1/build/run" \
  --observer "$R/attention-preflight-a4/cann851" \
  --model /home/gyl/models/Qwen3-0.6B --stage fault --card 2 \
  --stock-run "$R/attention33-fd-stock-baseline-a1/attention33-fd-natural-stock/run" \
  --baseline-run "$R/attention33-fd-stock-baseline-a1/attention33-fd-natural-baseline/run" \
  --bridge "$R/attention33-fd-stock-private-bridge-a1.json" \
  --out "$R/attention33-fd-fault-cases-a1.json"
```

F preparation replays the raw bridge again; a copied PASS JSON is insufficient.
The Worker independently binds the bridge hash, actual-input contract, current
reviewer and fresh completion before model construction. Measured F must use
the new FD600... entry, while baseline/repeat/OFF use native510...; existing
same-Worker full recovery checks remain mandatory. Source/package/model changes
require a new stock/B bridge. A failed bridge remains FAILED; no silent reuse,
branch forcing or tensor-output correction is provided.

Public Coverage integration remains a separate next step. Exported native
facts can bind each of the28 actual measured module/scheduler boundaries, but
112 recovery-phase calls and24,576 CPU rows are not coverage denominators. The
public owner must provide independent denominator/Native proof and C3 group
recovery checks. Raw B/F collector statistics retain their separate fixed
rtol1e-5/atol1e-8 audit and missing/null/NaN rules.
