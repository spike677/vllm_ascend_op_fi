# Source62 single typed FIA evidence review

`review_native26_fia_single_source62.py` reads the completed source62 originals.
It imports only the adjacent, SHA-pinned independent source54 reviewer and the
Python standard library. It does not import the injector, load a DSO, contact a
Worker or execute a device operation. The shared reviewer's QMM defaults retain
the source54 checks; explicit FIA parameters select the direct residual
destination and actual typed public-return rules.

Keep both reviewer Python files together in the new review directory. Read the
frozen source and Native build at their original paths; do not replace either
with the review checkout. Root can run the following on the evidence host:

```sh
python3 -I -B /path/to/review_native26_fia_single_source62.py \
  --evidence-root /home/17-fault-c2c3-20261117/fia26-single-c1-source62-card2-a1/providers26-v2-fia-smol-c1_single_effects \
  --source-root /home/17-fault-c2c3-20261117/source62 \
  --native-root /home/17-fault-c2c3-20261117/native26-fia-typed-group-a4 \
  > /new/review-directory/review.json
```

Exit 0 means the saved source-bound single-site evidence reconciles. Exit 2
means missing, inconsistent or refused evidence. The output is compact JSON;
it does not print tensor arrays, public object ledgers or whole Native traces.
The original result, inventory, serving receipt, source manifest, build and
review helper hashes are pinned. Native compile inputs, deployed Python source
hashes, metadata-array ownership and each recorded DSO are independently joined
to the frozen source/build. All files are read without alteration.

The review reconstructs the full 30-call denominator for all eight requests,
the twice-healthy resolved set and minimum-key selection. It joins each Native
callback to the actual request, module, producer, retained public return and
later consumer, and checks the first Native receipt through release. It checks
all original R/L/OFF/F/OFF leases, owner/generation, direct residual destination,
nonzero F effect, full residual restoration, logits-hash recovery and final
zero owned Native/Python resources. F may change the generated token; every
control, OFF and post-cleanup generation must match the healthy token.

The additional FIA layer checks complete typed NCL geometry and exact flat
storage span, current borrowed GWS descriptor, owned descriptor at GWS and
execute, consumer-time descriptor release and literal Python return identity.
The public ND label and Native NCL label must be explicitly joined by the
complete formals; neither a shape match nor a format allowlist is sufficient.
QMM allocation adjacency and its 32-byte tail shift do not apply to FIA.

It reopens all 60 stock output bins and five selected-mode output bins, checks
full length, raw SHA, tensor-header SHA and typed-record digest, then compares
the two stock occurrences and fresh nonfault owned results byte for byte.
Only paths recorded under `/evidence` map to the explicitly supplied evidence
root. The five selected writer hashes must also agree through the shared
receipt checks. There is no search for substitute or similarly named files.

The result remains source-bound C1 evidence. Native identity, exact device
readback, strong-reference duration and stream synchronization are original
runtime receipts; they are not a new live observation. Full residual, input and
logits arrays were not saved as bins and are not independently rehashed by this
reader. The later ordinary post-Python-cleanup request proves tokens, not a new
full-logits capture. The initial warmup typed records outside the eight request
windows have state counters but are not included in the per-window detailed
record audit. The report never grants C2, C3, E5, all-resolved-target or another
mechanism qualification. Separately reported E5 statistics have an earlier
slot-indices difference and incomparable preceding null rows; they are not
runtime-review inputs or proof of fault-only numerical localization.

CPU contracts use actual CPU public-return hooks plus explicit ABI packet
doubles, and saved-packet fixtures for the shared request/receipt checks. They
cover wrong storage spans, strides, call sequence, cloned objects, foreign
requests, extra launches, owned-descriptor release, wrong destination/executor,
changed bins, recovery failures and retained resources. Those contracts do not
claim CANN or NPU validation. The source54 default suite remains part of the
regression check.
