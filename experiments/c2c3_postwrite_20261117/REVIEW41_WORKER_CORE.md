# Public Core41 source60 saved-evidence review

`review41_worker_core_result.py` reads the original result, all 43 RPC JSON
artifacts, and the complete Host receipt. It does not import Torch/vLLM, load a
Native library, run a subprocess, access a device, or modify an artifact. It
reuses `review41_group_receipts.review_document` for the existing complete valid
K/V prefix, all-target lossless/OFF and recovery checks. The in-memory schema
view refers to original records; it does not create a replacement result.

The source manifest and commit are pinned to source60; the Native bytes are
pinned to the existing reviewed `libdcat_postwrite.so`. Supply the final original
result SHA from the root operator. Run on an ordinary CPU process:

```sh
python3 -B /review/review41_worker_core_result.py \
  --result /home/17-fault-c2c3-20261117/worker-core41-source60-card2-a1/natural-kv41-worker-core/probe/result.json \
  --result-sha256 7547e859ccab85c0a36c9712f0b66a20a330ebfefb16ebaeca63c0daabb0796a \
  --source-root /home/17-fault-c2c3-20261117/source-a60 \
  --native-library /actual/reviewed/native/libdcat_postwrite.so \
  > /new-review-directory/review.json
```

Copy the two reviewer files together. The new reviewer verifies the reused
helper's SHA before loading it. Exit 0 means source-bound evidence reconciled;
exit 2 means an evidence gap or contradiction. Output stays compact and never
contains tensor arrays or full RPC bodies. Source and Native paths are read-only.
The original result and previous reviews must remain unchanged.

The root operator recorded the completed result as 111,486,889 bytes and the
Host receipt as 2,408,281,754 bytes, SHA
`e89cc0df828ec5d95674b11ce670bff8cd205950e08298739708811cbd6ccd6f`.
The loader retains one parsed Host history and compares each separately hashed
RPC file sequentially; it does not copy or serialize the whole history again.

The review checks:

- All 17 real protocol requests and the 18th ordinary enqueue after unregister,
  real driver/Worker request mapping, identical model/process identities and
  first-decode scheduler inputs. It reconstructs the actual attention target
  denominator from complete independent writer/consumer inventories.
- Actual 3D FIA K/V views joined to 4D caches by storage identity, offset,
  contiguous geometry and writer objects; complete valid logical K/V ranges,
  exact legal tail span, per-target lossless controls, Native effects and OFF.
- C2 once and C3 twice per target, every independent ticket, admission decision,
  effect/execution hash, discovery/candidate hash and target/Worker recovery.
- All original public RPC replies in order. Host ticket owner, Runtime internal
  owner and Native numeric owner remain distinct and explicitly joined.
- Original C2 proof issuance and C3 use, including report/scope/discovery/
  observation/recovery hashes, actual object identity, same owner and a later
  generation. Issuance must follow actual C2 health; use follows C3 lossless.
- Zero owned hooks, API links, Native/coverage leases and inflight calls before
  unregister, successful detach, trusted binding removal and same-Worker health.

Execution/recovery and proof-audit scopes are reported separately. Missing proof
receipts do not erase otherwise reconciled execution observations, but do not
establish independent proof inheritance. `status` and `contract_satisfied` are
not accepted as evidence; the latter remains a hashed field of the original C2
proof body and cannot be silently removed or rewritten.

Limits remain explicit: full prefix/root hashes are runtime receipts, not a new
hash of absent tensor bins. External model files are not reread. The structural
universe is enforced by the frozen source and joined to all actual observed
calls. The 18th request records Worker identity and generated tokens/text; its
full root tensor is not captured after observer removal. KV41 overwrites a legal
cache tail address. This report grants no out-of-bounds, E5, other mechanism,
portable model/device, or public qualification.

CPU contracts use the existing real Host/Runtime protocol fixture with an
explicit CPU transport. No CPU result becomes Native or natural-Worker evidence.
