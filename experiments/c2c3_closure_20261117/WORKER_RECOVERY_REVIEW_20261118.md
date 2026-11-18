# Natural RoPE and KV recovery review

These new results use the standard Qwen3-0.6B TP1 inference Worker on124 card2,
Ascend910B4/A2, CANN8.5.1. They are separate from the public Worker Core source46
Native20 acceptance and do not qualify all models, CANN versions or platforms.
All original results and statistics remain on124; this publication contains
implementation, reviewers and summaries only.

## RoPE22

`rope22-natural-group-source46-card2-a1/natural-rope22-group/probe/result.json`
SHA256 `2e92fcb0e841ecb92be8c790160f9f11607fe48eb70000b006a562ac0d7ee497`
records28 C2 hits and56 C3 hits on28 natural module aliases. Source46 is
`11607cda4d30a402b4e69dc4318fd6ff927053a2`. Healthy repeat, all target output
digests, OFF and final same-Worker health recover; owned hooks are zero.

The independent reviewer checks the shared object's canonical observer path
against its actual module identity, all structural aliases, actual executing
ancestor, scheduler/request identity and all four tensor metadata records.
It does not equate two same-shaped tensors or discard mismatched paths. Its
initial strict path equality was inappropriate for one shared module registered
under28 parents; both original review attempts are retained.

The corrected review checks9259 conditions and passes C2 and the observed C3
execution/recovery chain. Review SHA256
`a4bc8c4efcd116cca449e00635350a13b5577805e40d3ee0ec29fb72a50a24e6`.
The original result did not serialize issuance/use of its live C2 capability.
The frozen source does require the original live object; that is source-bound
support, not direct observation of capability identity. Strict C3 independent
receipt acceptance remains incomplete. New public Core integration records this
separately; no JSON receipt is converted back into a live capability.

The mechanism is **legal positions selecting the wrong row**, not an integer
overflow reproduction. The real source-bound Triton call was observed; raw
device ABI payload and raw statistics were not collected in this group.

## Fault41

The fresh `postwrite41-recovery-source44-card2-a1` pair fixes the old collector
uninstall failure. It records28 C2 targets/hits and56 every-match hits across
two requests. Every target's complete valid K/V prefix, actual FIA view/base
storage relationship, logical pages, untouched range and OFF are independently
checked. The3D FIA view is verified against its4D cache allocation; it is not
required to have an identical shape. Both groups finish with zero owned hooks,
API patches and native leases, with same-Worker healthy output restored.

| Original file or independent review | SHA256 |
|---|---|
| Baseline result | `1e5de3916b625e4ca535f31fe42cb6cc76fba73ae24ca2e66ef627e62f3a441a` |
| Fault result | `9a34e2804923679dbff07817fee6c887b3b024765ae48517a2e1edea24493a6c` |
| C2 consistency review | `a48e0a35bad3a370c48055bf2a7c7e3d2d3bb7569fb34a196293d57d969fda64` |
| Complete C2/C3 valid-prefix review, a2 | `3d80c125fabe519b9f76d8b50cba0b9ad70ef654775c6e68a01d0d9bee6cc8e4` |

The independent B C2, F C2 and F C3 reviews each pass19 conditions. Baseline
has zero fault hits; F has28 and56. Scope, owner and target sets match across
groups and their generation changes. As with RoPE, original capability identity
is supported by the frozen live-C2 source gate but was not separately recorded.
Keep demonstrated C3 execution and recovery distinct from strict direct proof
receipt acceptance. The prior failed recovery is not relabelled as successful.

## Independent statistics are a different result

Only original B/F statistics/mix, collector identity, config, stack and module
relations enter the baseline-reference comparator (`rtol=1e-5`, `atol=1e-8`).
Injection counts, expected operators and complete-array reports do not select
the first numerical difference.

| Fault41 comparison | First observable result | Limitation |
|---|---|---|
| B self | No observable difference | Self-check alone is not repeat stability |
| B step0/step1 repeat | `_npu_reshape_and_cache.0` slot_indices Max174→190 | Real physical page allocation differs;100 earlier nulls |
| B/F step0 | `Functional.linear.1.forward`, output0 Max3.0625→2.96875 | Relative change3.06122449%;138 earlier nulls |
| B/F step1 | `Tensor.__setitem__.18.forward`, input0 Max20.375→20.5 | Once is exhausted;2060 earlier nulls; raw change retained |

Corresponding summary SHA256 values are
`21cbe5b68938f79b4954800f969509fd4d1ae81995bfc49f9c102c27bc778691`,
`79b56d7d3cfede58e51131c16047903560e6591b340231b5cbc8a12626f37a94`,
`e9b691bef451e140f0242153a807a690520ac65b33467d2892b2eb68b782b60f`,
and `765d36dccc7f568a632f4306cdfc51b16786539ed8a00c30b69e935b28c72af6`.
Step labels are actual capture requests, not arbitrary generated token labels.
Complete valid KV and FIA output recovery does not erase the raw step1 change
or make uninitialized/overwritten storage a stable statistic. C3 raw capture
is missing. These statistics do not independently identify the postwrite site
uniquely, and no whole-capture E5 acceptance is asserted.

## Fault39 output proxy remains separate

`postwrite39-autodiscovery-source44-card2-a1` baseline result
`f26689e9b05aa55c470cabfd81392d40f44c9f23ab661f910bf0c5679acf87c6`
is `FAILED_WITH_EVIDENCE`. Its preceding natural HF MODEL_LOCAL C1 qualified
one real MatMul output/consumer: one fault hit changes output and OFF restores
exactly. The all-target baseline found140 eligible chains but stopped with
`POSTWRITE_STRUCTURAL_BINDING_CHANGED` after its first collection window.
Native leases were zero; group recovery remained pending and retained ownership.
The separate F case was not started. This result grants no group acceptance.

CPU reproduction shows that msprobe's persistent instance forward wrapper
changes the structural ModuleList classifier. The repair uses the shared exact
source/code/closure-pinned collector adapter to recover the original bound
unimplemented forward. Foreign wrappers, source changes, wrong owners, unused
children and executable subclasses remain rejected. New runtime validation is
required; the existing source44 failure stays intact.
