# Exact frozen Native20/Native39 metadata ownership audit

Both previously accepted native builds omit the required release of host
arrays returned by ACL tensor metadata getters. This is confirmed against
their actual build profiles and exact frozen source bytes, not inferred from
another branch. Their numerical effects, hit counts and healthy OFF artifacts
are preserved. Their previous resource counters do not establish that all
resources were released in the same Worker.

Root consequently suspended strict C2/C3 acceptance for these two mechanisms;
the strict count is temporarily 1/17, retaining only `18.api_half`. This review
does not rewrite historical receipts, alter source-a21, or reclassify their
numerical evidence. A corrected implementation needs new acceptance evidence.

## Source/build identity

Read-only SSH verified the existing manifests, profiles and seven actual
libraries. Each library hash matches its build manifest; the profile file hash
also matches. The three reviewed source files on `source-a21` match their
respective profile source entries and the exact Git bytes at
`5a8b54486161132b34a0af7d427a851519134a8f`.

| Scope | Actual build | Profile identity | Main library SHA-256 |
| --- | --- | --- | --- |
| `20.buffer_substitute`, `providers20-group-a1` | `/home/17-fault-c2c3-20261117/native20-group-a1` | `d68519cebd35d0d7e6053155b8ba77a835719434a0596eed6e684447629b4a86` | `0c3e3f5fc73401f7b6e9bcdba261c730cf8ba17d6389c90a3843399da6b3df30` |
| `39.matmul_range`, `providers39-a6` | `build39-a3/cache/2c83a5742eb393f7b7bd202f421b1b8abed5f42dc433e86dda23f40412f89ff7/attempt-a0743e472ccd4f5dbb32831f35c89123/native` under that same experiment root | `e31eefa42824b1960e7f2e1e58f3130dd4b07a054b6d3b15bfb2787885e75032` | `04f834ef2e479f2578ec2c33a7e895097a2eb2a5c0e619dea33a497c5bfbc07c` |

The build-manifest file hashes are respectively
`42b422b0adc0771e465f77eb104a117320d2913e7081a7a6af8bb6b90305c579`
and `4016161bff94bf54be7292cbfeb0ea7f7adb3a32ba13c739010cff2257d86974`.
The adjacent JSON records complete paths, proxy library hashes, getter/call
sites and normalized reviewed-span hashes.

## Confirmed ownership omissions

The official CANN 8.5
[aclGetViewShape reference](https://www.hiascend.com/doc_center/source/en/canncommercial/850/API/aolapi/operatorlist_00038.html)
states that the API allocates the returned array and its example releases
view dimensions, storage dimensions and strides with `delete[]`. The
[aclGetViewStrides reference](https://www.hiascend.com/doc_center/source/en/canncommercial/850/API/aolapi/operatorlist_00039.html)
likewise assigns release responsibility to the caller.

- **Native20:**
  `fault20_fast_validation/source/l4_metadata_lifecycle_proxy/libopapi_metadata_proxy.cpp`,
  SHA `0384a7609202d01b9224fd92374f7e0206a314586a10c7b7cda9812c7d54c9e6`.
  `capture_x1_metadata` allocates four getter arrays at lines 373–376;
  `capture_tensor_metadata` allocates three at lines 441–443. Both copy values
  into metadata and return without retaining or releasing the arrays, on
  success and on subsequent validation failures. The actual QMM GWS route
  calls `capture_qmm_metadata` at lines 913–915, which calls both helpers at
  lines 541–552. `log_tensor` also calls the three getters at lines 786–788;
  its printing/early-return paths need the same ownership correction.
- **Native39:** `fault39/native/metadata_capture.inc`, SHA
  `19a5ec47d9c70e39a2c9990711017bbe51e6c4b856dd1b230df41ac90256d1c3`.
  `capture_tensor_metadata` allocates three arrays at lines 59–61, copies their
  contents and returns at line 108 without releasing them. Early returns
  after a successful earlier getter have the same omission.
  `fault39/native/metadata_proxy.cpp`, SHA
  `622d9cff10abdd5e41c47b04daa251f7ef859fbd2b9901c501c973bcad53af77`,
  invokes that helper for each successful MatMul's self, mat2 and out at
  lines 69–70. Those proxy bytes are bound by the accepted build profile.

These are host-side metadata allocations. Successful stream drains or zero
device/owned-workspace counters cannot account for them. Worker termination
eventually reclaims process memory, but that is not same-Worker recovery.
No new heap measurement was performed and no exact historical leaked-byte
total is claimed.

## New quant shim review and required correction

The independent `bfe3a9bc` quant group shim has three material issues, sent to
its owner before any Native qualification: a failed actual execute was treated
as consuming its executor; C++ exceptions could skip manual clearing of the
Native stack-liveness flag; and its metadata getter arrays had the same
ownership omission. Its CPU getter mocks returned internal Tensor arrays,
masking the real SDK allocation contract. The owner is correcting all three.
The seven Python build-orchestration contracts passed locally with mocked
subprocesses; they do not validate these C++ lifetime paths or real NPU use.

Fresh CPU contracts must allocate getter arrays dynamically and prove balanced
release on success, partial getter failure, rejected geometry and exceptional
exit. Runtime contracts must retain executors/resources after a nonzero
execute result without verified consumption, and revoke the internal live
witness on every stack exit. Exact current source/build identities, per-target
effects, complete denominator, healthy OFF and same-Worker zero resources need
fresh acceptance after the fixes. No historical source or evidence was edited,
and this review started no NPU operation or container.
