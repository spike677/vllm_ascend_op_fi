# Quant source55: metadata diagnostics and explicit collection

This increment fixes two Python boundaries and adds a bounded Native diagnostic.
It does **not** accept a new Native layout. In source55, rank-two input and Q
descriptors had zero geometry, while rank-one scale was complete. The old raw
View did not retain the actual stride/storage ranks, so the reason cannot be
uniquely reconstructed from that evidence. In particular, the separate FIA
NCL observation is not evidence of the quant ND descriptor layout.

The original `view()` rank/storage gate remains unchanged: rank 1 or 2, matching
view/stride/storage ranks and equal view/storage extents. No public Tensor value
fills a Native field. Unknown, null, mismatched or larger layouts still refuse
qualification. Fault arithmetic, owned outputs, reference/helper execution,
input invariance, rollback and original C1 library remain unchanged.

## Actual GWS metadata diagnostic

The new additive read exports are `dcat_dqg_metadata_state_v1`,
`dcat_dqg_metadata_v1` and `dcat_dqg_metadata_parent_v1`. Existing ABI layouts are
unchanged. A fresh SDK build is required: the binding verifies the new sizes
against both the real DSO and its build manifest. Local ctypes sizes are
snapshot 416, record 1304 and state 72 bytes; these are expectations awaiting
the fresh SDK receipt, not a claim that the SDK build has run.

One record captures all three original public GWS descriptors on their current
stack. It retains descriptor/data addresses, thread/device, actual parent token,
dtype/format/offset, all three getter counts, up to eight elements per array,
getter call/status masks, null/truncation flags and the exact refusal stage.
The copied diagnostic never supplies geometry to the execution engine. The
record links to the existing parent; actual helper/native events remain in the
original independent journal and do not become new model targets.

The fixed log retains the first 8192 public GWS records (10,682,368 static bytes),
with explicit total/retained/overflow counts. It cannot silently replace old
records. Its static diagnostic storage is reported separately from target-owned
resources. Each parent returned by `QuantGroupNative.raw_range()` includes its
matching `metadata_diagnostic`, so the existing full inventory artifact captures
the rejected layout as well as all 135 structural candidates. Reads copy retained
data only; they never revisit expired descriptors or call an SDK getter/device.
The NOLOAD initialization diagnostic also retains the new records and reconciles
them against actual public GWS entries. New tests cover rank 1/2/3, mismatched
storage/stride ranks, null arrays, failed/throwing getters, truncation, ownership
release, no late getters and bounded overflow. Native C++ tests require root's
SDK/host build; Python fixture tests are not Native acceptance.

Stages: 0 not attempted, 1 unavailable, 2 type, 3 format, 4 offset, 5 view shape,
6 strides, 7 storage shape, 8 raw address, 9 rank/null-array gate, 10 differing
storage extents, 11 invalid extent/type size, 12 complete extraction, 13 exception.
Stage 12 alone is not qualification. `called_mask` uses bits 0..6 in getter order;
array masks/counts use view/strides/storage order. An exception keeps the last
attempted getter and releases every allocated SDK array through RAII.

## Metadata row producer and public collection

The source-checked metadata builder now binds its actual common positions Tensor
at entry, observes the actual `aten.slice.Tensor` prefix `[0:8:1]` on dimension 0,
and requires that exact returned Tensor as `input_positions`. A padded source
such as `[272]` is allowed only through this observed operation and exact storage,
offset, stride, dtype, current version and write epoch. An optional actual
`aten.to.dtype` must be an identity int64/no-copy call on that same object.
Clones, changed indices, writes, replaced result objects and unknown producer
chains remain unverified. The actual rotary cache-index/copy/output chain must
belong to this same metadata builder and current request interval.

PyTorch's dispatcher can expose a newly returned view at version 0 before the
view wrapper attaches its source's version 6. Both snapshots are retained. The
final same-object view must have the still-unchanged source version and epoch;
source writes or a changed storage/extent cannot use this transition as an excuse.
CPU contracts exercise the padded source, source version 6, same-value writes,
wrong slices, equal-value clones and two successive inference-mode requests with
the profile/mode fully removed between requests.

`QuantProvider.query()` now reads state/report only. The public Host helper
`collect_replay()` sends existing action `advance` with exactly `{'collect': True}`.
It never advances the scheduler or performs a forward. The C1 probe explicitly
collects after host completion and then queries. A selection refusal is an action
failure with its original reason retained in `collection_errors`, rather than a
query error labelled `NATIVE_STATE_UNAVAILABLE`. Finish still explicitly collects.
Host/Runtime contracts cover unchanged repeated queries, exclusive collection
settings, failed collection before any F, retained owner and successful cleanup.

## Root-only next attempt

Use a new immutable source/package and a new output directory. No agent container,
SDK or NPU run has occurred for this increment. Root first builds without devices:

```sh
python3 "$SOURCE/experiments/c2c3_closure_20261117/providers_quant_build_group.py" \
  --source "$SOURCE" --source-commit "$COMMIT" \
  --image-id sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0 \
  --out /home/17-fault-c2c3-20261117/native-quant-group-metadata-a1
```

This runs 20 real-runtime C++ memory-double scenarios plus the existing core and
seven real-ELF loader contracts against the actual SDK headers. The build records
the actual new struct sizes and static diagnostic budget. Then use the existing
device-free managed load probe with the **new** build.json SHA. Neither a6 nor any
earlier deployment manifest matches this source.

Prepare one healthy inventory case, without `--single-site-c1`:

```sh
python3 "$SOURCE/experiments/c2c3_closure_20261117/providers_quant_prepare_inventory_case.py" \
  --group-abi --installed "$INSTALLED" \
  --quant /home/17-fault-c2c3-20261117/native-quant-group-metadata-a1 \
  --probe "$SOURCE/experiments/c2c3_closure_20261117/providers_quant_inventory_probe.py" \
  --source-commit "$COMMIT" \
  --image-id sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0 \
  --model /home/haoran/models/DeepSeek-V2-Lite-W8A8 --card 2 \
  --out /home/17-fault-c2c3-20261117/providers-quant-metadata-a1-cases.json
```

Root alone gates and runs this case. It executes the two original healthy B8
first-decode censuses, with all raw parent/helper/native calls and row refusals
retained. If Native public matching remains rejected, the result remains a failed
qualification and its raw diagnostic still explains the actual descriptor. The
next possible layout adaptation requires that new evidence; this increment must
not be advertised as a rank-two fix or a C1/C2/C3 result. All three remain false.
