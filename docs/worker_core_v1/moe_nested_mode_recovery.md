# MoE nested observer mode recovery

Independent CPU review of source66 (`239096ef`) reproduced a cleanup defect:
`NaturalMoETrace._profile` removed a returning frame from `pending` before
exiting its nested dispatch mode. An exit exception before the actual pop left
that mode in the real thread-local stack, but subsequent public `recover`,
`unregister`, and `detach` released the Runtime ticket and deployment reservation.
The ordinary CPU request could still return; this did not prove a zero stack.

The fix keeps an independent owned-mode ledger from before entry until actual
stack readback proves removal. Returned Python frames remain pending while
their mode remains owned. Cleanup retries only on the original thread with
the exact mode at the top; active frames, foreign profiles, and foreign modes
prevent release. An after-pop exception retires the absent handle but still
propagates the first failure. A before-pop exception retains the handle and
the public owner. No exception is converted into experiment success.

The failure receipt retains the original trace error after resource cleanup.
Reports expose retained mode/frame counts and profile state; a trace with any
recorded error remains incomplete. Public recovery requires no additional
model execution before the original reservation can be removed.

`test_core_moe_mode_cleanup.py` exercises real CPU Host/Runtime and dispatch
stacks, including before/after-pop exceptions, foreign top/profile/thread,
active frames, an exit returning without removal, and entry failures before
and after a real push. It verifies owner retention, safe retry, retained first
error in the cleanup receipt, and an ordinary same-Worker request afterward.

This changes implementation hashes and requires a new source scope. It does
not change source66 original artifacts or assert that its completed normal
8/18 protocols encountered this exception. CPU contracts grant no NPU or E5
qualification.
