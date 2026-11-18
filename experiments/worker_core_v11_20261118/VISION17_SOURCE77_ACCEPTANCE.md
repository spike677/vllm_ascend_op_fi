# Fault17: accepted single-projection Torch C2/C3 scope

Root accepts source77 against its predeclared
[Torch parameter contract](../c2c3_closure_20261117/V11_VISION17_GROUP_20261118.md).
This adds one C2 and one C3 mechanism: current totals are **C2 9/17, C3 8/17**.
This decision reviews existing originals; it is not a new run or a runtime
admission certificate. No Native/device requirement is added retroactively.

The scope is Qwen3-VL-2B, standard vLLM Worker, TP1, BF16, CANN8.5.1/A2,
one independently discovered patch16 Conv3d projection. Each of ten actual
single-image requests reached the encoder (901 tokens/3536 patches). Cache
reset alone was not accepted as evidence of execution.

| Requirement | Original evidence |
|---|---|
| Automatic selection and denominator | One eligible projection; actual upper/lower observers and scheduler image/grid binding |
| Only intervention | Real Torch Conv3d padding `[0,0,0]` to `[0,7,7]`; original operator executes once, unchanged input/weight/bias and output geometry |
| C2 | One isolated F/OFF pair, covering the complete eligible set of cardinality one |
| C3 | Original live C2 proof consumed in the same Worker/owner/scope, generation1 to2; two fresh F requests, two actual hits |
| Numerical propagation | Changed projection return, exact live first consumer and logits; repeat fault result stable |
| Recovery | C2 OFF, C3 OFF and post-cleanup ordinary health restore the complete recorded healthy result; owned resources released |
| Returned inference | Token and text unchanged; logits change is not a changed-answer claim |

The [original independent review](vision17_source77_original_review.json)
contains 1,010 execution checks and 29 source checks, zero issues, 25 rehashed
source files and 28 Host RPCs. Review SHA256:
`57dfe5c170dfef8a81e4ef927126ea81489432ef7ec46724fac10789ba01d1ca`.
Source commit `c0045f283de5af99628da9631d19d3ee9769d66d`; source manifest SHA256
`11c609ce87f0f7dd8c97b1ae008fd512f514bae46a6cec77c4d2a5d8f534bcf3`.
Original result SHA256:
`36f17fa31ca8b4b3a04e114baaacdc1b54baefcb162129e1b41b408d586a361a`.

The reader's serialized `public_qualification_granted/c2/c3=false` fields are
unchanged: a saved audit is not a live authority object. This separate root
decision records the accepted scope without rewriting that invariant.

Final Torch arguments are not ACLNN arguments or device launch evidence.
Native qualification, device payload, multi-target coverage and E5 remain
unestablished. Full-tensor hashes are runtime diagnostics, not independently
transported arrays or msprobe statistics. Other models, multiple projections,
other versions and yellow-zone integration need their own acceptance.
