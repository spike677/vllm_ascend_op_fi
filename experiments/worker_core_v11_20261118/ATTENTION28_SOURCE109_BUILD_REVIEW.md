# Source109: actual SDK build stopped at schema verification

Frozen source `17ee25b0f73267f7d842399d8f94e0e377ea1d65` was tested on124
in the pinned production image without NPU devices or network access.
All 160 selected production CPU tests passed with zero skips. The separate
TND28 C++ geometry/scope contract compiled and ran successfully, including
actual foreign-thread and non-target pass-through checks.

The fresh host package, private staging and actual public-schema compiler
all returned zero. The next verification step refused the build with
`TND28_ACTUAL_SDK_TYPED_FIELDS_DIFFER`. The builder expected 63 semantic
fields; the actual 1,576-byte serializer schema contains 65 fields.
No semantic field is missing. The two additional records are:

| Actual schema field | Offset | Width | Count | Type |
| --- | ---: | ---: | ---: | --- |
| `baseParams.FusedInferAttentionBaseParamsPH` | 81 | 1 | 7 | `uint8_t` |
| `prefixParams.FusedInferAttentionPrefixParamsPH` | 1537 | 1 | 7 | `uint8_t` |

The actual host dependency file identifies
`/usr/local/Ascend/cann-8.5.1/include/register/tilingdata_base.h`.
A separate device-free material process used the real decoder compiler flags
to resolve that header and its dependency closure. Its exact SHA256 is
`b2f7c0b08c95e4559b899bababf67cd4d3740780762c889450b089bfe09478a8`.
The repair must derive and validate these generated alignment fields from
the actual macro/source rules; it must not ignore arbitrary extra fields,
accept every name ending in `PH`, or reduce the full-schema checks.

The failed build did not enter kernel compilation or NPU execution.
Its owned container exited2 with `OOMKilled=false` and was removed.
This was a validator refusal, not another external Docker stop.
The failure and its original artifacts remain unchanged. A repaired build
needs a new source/attempt and its own complete result.

Originals:
`/home/17-fault-c2c3-20261117/tnd28-prefill-normal-build-a1`.
Key identities:

- Source manifest: `e0c7af868693b5c6f9159a1210638d0aea4e5dc3e82a45d321b774bab62cfd05`.
- Production CPU log: `6afffb49109b4f6c3824d989fbadd1e23351017215e91a59be2ec1e0a244e7d8`.
- Executed C++ contract: `736f6d772ac58da0b948ce0fff6bcc06a61afc2360d5d0e19eb5555a738b42a0`.
- Failed completion: `88ecf6184af758f1ef38e5fe92d3c4d886388789a23008e1071f86135c68814d`.
- Controller: `48cd2fe73e67bd76962b5d0b77881f92a0040672deb5936006e5fac5c93e6e1c`.
- Actual schema: `20dd6a361142aa9acd73f0787673e24660daef46846e5b362eb9306b15fe035c`.
- Schema compiler receipt: `b82a9c1eb16ae297b2a97874818bd8186638013b21e7134b96965cf4eecb76df`.
- Preserved schema failure materials: `18dd5eb1048518758f4d453b8d60d7e276a6393d713b8cef23e8ce590d10a01b`.
- Actual macro/dependency materials: `8deb119f22607b79f90b0752d8b070b6ef488b7a0251135ebc1ad0735bfd2b65`.

Loading, normal equivalence, F, C1/C2/C3 and E5 remain unqualified.
The follow-up source repair reconstructs the full pinned serializer layout,
including exactly these two padding fields, and checks the compiler's actual
macro-header dependency. It does not change the failed source109 result;
the next fresh SDK build must independently verify that repair.
That separate source110 attempt has now completed; see its
[build result and remaining NPU gates](ATTENTION28_SOURCE110_SDK_REVIEW.md).
No raw statistics, private tensors, compiled binaries or SDK header copies
are included in this publication.
