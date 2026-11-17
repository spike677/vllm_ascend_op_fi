# Public ACL postwrite adapter

`postwrite.cpp` provides a separate ABI for two controlled proxies. It does not
modify or replace the validated Fault39 MatMul arithmetic library.

| Provider | Observed boundary | Intervention |
|---|---|---|
| `Fault39OutputProxyProvider` | actual `aten.mm.default` returns a fresh finite BF16 tensor | write a prefix of 1/5/51/512 values to explicit +Inf, −Inf or qNaN bits |
| `Fault41TailColumnProvider` | actual `aten.index_copy_.default` writes a registered model buffer | flip the sign bit of the last N logical columns of the one actually written slot |

The Python provider runs two unchanged model forwards. It selects the first
eligible call in stable execution order, requiring a later value-consuming
operator on the same live storage. View-only operations do not count as
consumers. Layer names are diagnostic output, never configuration selectors.
All other calls run unchanged. The output, writer and consumer receipts are
derived from actual dispatch, not submitted by the caller.

The current scope is synchronous inference, one selected call, BF16 contiguous
base format, zero offset, and storage no larger than 1 MiB. Fault41 supports a
two-dimensional model-owned cache and a single actual int64 slot index. It has
no scheduler request mapping, paged KV adapter or natural cache OOB claim.
The MM path is a public Tensor/operator output proxy, not the previous OpCache
slot-2 implementation. Neither provider grants public Worker, pretrained-model,
`all_eligible` or `every_match` qualification.

Native code owns one lease at a time. It synchronizes the current device,
copies the original bytes to host, checks finiteness, writes selected BF16 bits
to the real tensor, and verifies the complete readback. Python retains the
actual tensor object while that lease exists. Native recovery rejects an
unexpected intervening writer, restores the healthy post-operator bytes, checks
all bytes, and only then releases ownership. Failed transfers/recovery retain
the lease. Healthy cache-writer updates are preserved; only the proxy mutation
is restored. Returning model outputs are copied for evidence before restoration.

`baseline` performs an identical byte writeback and must reproduce the two
unchanged forwards exactly. `fault` is admitted only once after that lossless
gate. `off` executes no mutation and must recover the original model output.
These are separate states from a build success or ABI load.

Build against the discovered CANN installation's public ACL header and
`libascendcl`, using the common Native Build Manager recipe. ABI exports are
`dcat_pw_abi/open/apply/restore/release/live`; supported ABI is 1. The caller
checks the actual compiled library SHA-256 before loading it. Unknown private
OpCache layouts are irrelevant to this backend and receive no implied support.

Real NPU entry: `experiments/engineering_20261117/postwrite_probe.py`.
It can collect raw msprobe `statistics/mix` with `--statistics`, leaves those
files in the selected evidence directory, and emits a mechanism-only receipt.
It does not publish raw data. `tests/test_postwrite_providers.py` uses actual
CPU Torch dispatch and an explicitly labeled CPU transport to verify selection,
value propagation, unchanged regions, recovery and refusal paths. CPU tests
do not establish Native acceptance.
