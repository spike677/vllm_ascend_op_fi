# Quant group initial GWS resolution

Root's source51 diagnostic SHA
`00bcfd9d967823d94bcee297f16f474b713a83f62634fd9e55b6ccd388a301e9`
records a stable same-process snapshot from the failed LLM constructor. No
LLM, Core client or binding had been constructed. The group had one public
GWS parent, zero Native prepare/execute calls, one rejection, and last error
`Identity (-1)`. No owner, resources, pending call, overlap, quarantine or ABI
exception was present. Device was 0; the three tensor descriptors were nonzero
but all view metadata was empty. This is the `api.ready == false` branch,
before any actual vendor prepare or tensor getter call.

Root inspected the pinned image's real `libopapi.so`, SHA
`b3890625d3e0cc11ef32460083698ecaa8f702b5edd0169ea617e2e6bb76270c`.
Both DynamicQuantV2 symbols are defined there, but `CanUsePTACache` is absent
from that file's own symbol definitions. **This alone does not prove a null
`dlsym` result or the unique source51 rejection cause.** The earlier report's
missing-export attribution was too strong. The old a4 ELF also has preemptible
V2/GWS GLOB_DAT relocations. Their actual source51 runtime binding was not
captured, so symbol preemption remains unproven in that actual process.

Source52/a5 passed seven host-mock ELF cases, but its actual Worker still stopped
at its first GWS with no vendor execution. Original result SHA256
`8ca8ae0f4a44b075662a54ec51fb80c8388cd798550feaad54c612697da867cd`,
diagnostic `7bd5082092bc2762cd819bde511cc7b1c19589572ee00a0172b7d916215b6d55`.
An additional device-free probe of the exact real CANN libraries resolved every
required NN/ACL/V2 symbol. Cache advice is present through `DT_NEEDED` and belongs
to `libnnopbase.so`, not the `libopapi.so` handle's object. NN file SHA256:
`6dc3b4fcfcf57b10be0018e89f4edbc079f02b4bb0aaadb109df350939fd4c99`.
No operator or device function was called in that CPU probe. This identifies an
actual link dependency rejected by a5's optional-cache owner check; it does not
retroactively record source51's public-symbol binding.

The repair separates the required V2 functions from optional cache advice.
V2 always declines PTA caching. An unrelated API receives `false` when no
cache consultation export is available; a present, correctly owned export
is called normally. Both V2 functions must belong to the explicitly opened
operator DSO, identified by that handle's link_map and dladdr. A private local
anchor identifies the group DSO even when public symbols are preempted.
Self resolution, a missing required API, or required functions found only in a
foreign dependency refuse resolution. Foreign cache advice is ignored and never
called; it cannot authorize caching and cannot invalidate the required V2 APIs.
No linker-wide `-Bsymbolic` option is introduced. Owner, lease, lifetime,
observer, budget, execution, publication and recovery guards are unchanged.

Seven new loader contracts use the complete production DSO and separate ELF
stand-ins for NN metadata, ACL and the operator API. They never assign the
private `api` object or `ready` flag. The original operator DSO is globally
loaded first. Cases cover absent/present cache consultation, each missing V2
API, self resolution, foreign operator owner and foreign cache owner. Successful
cases (including ignored foreign cache advice) record one genuine call to each host stub through the full exported
GWS/execute path; negative cases record zero such calls. The NN stand-in uses
allocated shape arrays, and all device allocation/copy/synchronization stand-ins
assert if called. These are loader/control contracts, not CANN numerical tests.

The builder compiles the production DSO and driver against the actual SDK;
only explicit stand-ins use the mock declaration headers. It preserves all
seven JSON receipts in `loader-contract.jsonl`, binds their SHA into the build
receipt, and refuses an incomplete matrix. The ABI sizes and exports remain
unchanged, but changed production source requires a fresh DSO and deployment
manifest. The old a4 result and bundle remain immutable. No new Native or
C1/C2/C3 acceptance follows from local Python contracts or prepared C++ source.

Root-only next build command (this agent does not execute it):

```sh
python3 "$SOURCE/experiments/c2c3_closure_20261117/providers_quant_build_group.py" \
  --source "$SOURCE" --source-commit "$COMMIT" \
  --image-id sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0 \
  --out /home/17-fault-c2c3-20261117/native-quant-group-v1-a6
```

After actual SDK/loader success, use the resulting manifest SHA and new bundle
with the same reviewed C1 model, workload, runtime flags and resource bounds.
The initialization diagnostic remains enabled and verifies the new production
source hash. No old address or failed-call storage is reused for recovery.
