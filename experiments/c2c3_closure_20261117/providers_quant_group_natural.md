# Natural B8 entry for the isolated quant group ABI

The new `SharedScaleGroupInventorySession` observes two real first-decode B8
batches through the existing host-driven scheduler/embedding/logits hooks.
It loads the reviewed group DSO, never the old C1 backend, and does not install
a Native group owner. It automatically discovers current dynamic projections;
no layer name or chosen first projection is supplied by the caller.

The independent group Journal retains every actual model parent and every
typed GWS/execute event. A read-only public quant call must bind one parent and
four actual Native events, exact current input/Q/scale descriptors and storage,
thread/device, registration, stream and workspace. A reused executor address
requires a new parent/registration. Classified helper calls cannot masquerade
as those four events. Raw counters and complete rows must reconcile; unowned,
missing, failed or unknown-row calls remain in the evidence and block complete
qualification. The bounded range reader verifies the loaded mapping once per
snapshot while reading all records; it does not rescan every loaded DSO for
every individual event.

Current SDK evidence is `native-quant-group-v1-a4`, manifest SHA256
`8d8d02476a911b2b0251eb70b6af3e2cd45ea2498aa65d1beb1a316822f3012d`,
library SHA256
`b25807b379345c35139138b5eeb1b57d00b421594d88aa2b62dc6ebd8021f557`.
Root's actual device-free managed load/query passed in
`quant-group-managed-load-cpu-a2`, controller SHA256
`2f444d7804c4921ca276e724953331b78daa4f1e7a2921ed09621b2fac305f92`.
That verifies the real marker mapping, all ABI sizes and zero initial state;
it is not an NPU result.

The local 69-test run passed. The new CPU integration executes actual PyTorch
module/API/dispatcher hooks over an explicit Native-journal double. It checks
the two B8 generation phases, group-only loader, exact parent/native joins,
fresh registration on executor reuse, missing execute preservation, unowned
Native calls, hook removal and an opaque row permutation that remains unknown.
The double does not establish CANN numerical behavior or an actual NPU row map.

Root prepares the new case from the freshly installed source package:

```sh
python3 "$SOURCE/experiments/c2c3_closure_20261117/providers_quant_prepare_inventory_case.py" \
  --group-abi --installed "$INSTALLED" \
  --quant /home/17-fault-c2c3-20261117/native-quant-group-v1-a4 \
  --probe "$SOURCE/experiments/c2c3_closure_20261117/providers_quant_inventory_probe.py" \
  --source-commit "$COMMIT" \
  --image-id sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0 \
  --model /home/haoran/models/DeepSeek-V2-Lite-W8A8 --card 2 \
  --out /home/17-fault-c2c3-20261117/providers-quant-group-inventory-a1-cases.json
```

The script prepares a reviewed manifest only. Root owns the global queue,
actual image/card checks, controlled launch and cleanup. The case sets the
private group vendor path and explicitly disables legacy C1 mode. No provider
agent container/NPU launch is authorized by this handoff.

Row transfer still uses actual observed scheduler/embedding/RMSNorm lineage,
current storage/write epochs and the reviewed row-preserving operator chain.
Attention, opaque MoE routing/combine and unreviewed mutations do not inherit
labels from shape or equal values. The new raw trace supplies their actual
operator schema/objects for the next source-bound transfer adapters. All
structural candidates and all actual Native parents remain visible;
`native_qualified_targets=0`, `c2=false`, `c3=false` until the complete row set,
per-target reference/lossless/OFF, actual current-input helper/effect and
same-Worker group recovery are demonstrated.
