# Real operator probe

`native_probe.py` is one explicitly launched operator case per fresh Python
process. It neither starts containers nor launches other cases. The parent
controller selects the allowed device, owns process timeout/termination, and
must limit each experiment batch to at most eight cases.

Before process startup, preload `libfault39_native_v1.so`, put the isolated
metadata `libopapi.so` first in the worker library search path, and set
`VLLM_SERVER_DEV_MODE=1`. Use the exact CANN installation recorded in this build's
`version_profile.json`, with its environment script and matched libraries. The provider only
attaches to `CDLL(None)`; a late load cannot satisfy the probe.

Example arguments after the parent has prepared that environment:

```sh
python3 native_probe.py --stage discovery --output discovery.json
python3 native_probe.py --stage lossless --input-mask 1 --output lossless-self.json
python3 native_probe.py --stage lossless --input-mask 2 --output lossless-mat2.json
python3 native_probe.py --stage lossless --input-mask 3 --output lossless-dual.json
python3 native_probe.py --stage pattern --pattern half --output half.json
```

Each invocation first executes two actual `torch.matmul` calls on the same finite
BF16 ND inputs and output allocation. The mat2 input is a transpose view with
stride `[1,K]`. Selection requires two increasing real observations of a unique,
identical native signature, exact output repeatability, finite baseline output,
and real typed metadata/Run counter increments. If `torch.mm` takes a different
vendor path, the probe reports failure rather than claiming interception.

After stable discovery, P2 performs one real `mode=observer` Run with **no input
substitution**. It compares the three live PyTorch backing-storage pointers,
captured before and after the call, to the three native original and Run
addresses, requiring exact `[self, mat2, out]` order and unchanged addresses.
It also requires successful execution/synchronization/release, exact baseline
output equality, original input hashes, and the actual operator scope. Every
later selected Run repeats this independent pointer proof and checks which
input addresses changed according to its mask; output storage must not change.
Thus `formal_order_proven` alone never authorizes a substitution in this probe.

A lossless invocation executes its selected input mask only after discovery.
A pattern invocation first checks self-only, mat2-only and dual lossless cases
in sequence in that same process, then executes only its requested pattern.
Every selected case uses one bounded hit and waits for disable acknowledgment.
All runs use a scope named `operator-probe:<pid>:<case>`; this is actual operator
probe identity and is not a model request mapping.

The default historical shape is `M=1,K=12800,N=5120`. `--m`, `--k`, `--n`,
`--amplitude-policy historical58|auto`, and resource cap arguments support small
controlled cases. The defaults cap each storage at 256 MiB and a conservative
host/device working-set estimate at 1 GiB. K must be positive and even.

The JSON contains complete scalar output statistics, complete backing-storage
SHA-256 hashes, native snapshots, inspector conclusions, exact lossless/OFF
output equality, and explicit cleanup acknowledgment. It contains no tensor
payload. Original physical storage is hashed only after proving that the whole
allocation is the contiguous ND backing with zero offset. Any missing proof,
ambiguous target, failed stream/copy recovery, or nonzero lease count fails the
probe. Pattern success means the real controlled experiment executed and
recovered; the inspector separately states whether a range anomaly occurred.

Source inspection and `--help` are CPU-only. Running a valid probe initializes
the NPU and is reserved for the parent controller's authorized experiment.
`python3 test_native_probe_slots.py` runs CPU-only positive and negative tests
for the dynamic slot checker without importing torch or initializing an NPU.
