# Source108: normal build passed; private run incomplete

The new normal-only package covers all 28 discovered decode targets. Its stock
arm completed and passed independent original-file checks. Its private arm was
externally stopped before the last health stage finished. **The normal pair is
not accepted. No F, C2/C3 or E5 was established.** Source105's accepted single-site
function/OFF scope remains unchanged.

The [summary](attention33_source108_partial_summary.json) records original
hashes, event identity, stage counts and explicit negative qualification fields.
No private payload or new raw statistics is published.

## Build and actual execution

Frozen source: `b2339400e3959bcae7358b4a3ffd389eab724d0b`.
Variant: `tnd_page16_arch22_all_normal`.
Vendor: `dcat_f33_arch22_all_normal_transformer`.
This independent package retains M512/S2=512 and unchanged kernel arithmetic;
it does not enable the earlier single-site S2=256 policy.

The fresh CANN8.5.1/A2 build completed. The production image passed all 130
selected CPU tests with no skips; normal geometry covered 22,528 CPU cases.
The actual kernel, public POD schema and ten compiled keys are retained in
the completion receipt. These are build checks, not NPU acceptance.

Both actual arms used card5 with the existing lock and fresh device/allocation
checks. Card2 and unrelated services were not stopped. The 1,800-second case
limit was unchanged.

| Item | Stock | Private normal |
| --- | --- | --- |
| Original status | Complete, requires independent review | `STARTED` |
| Complete capture phases | 5 | 4; fifth has no `finish` |
| Decode targets per complete phase | 28 | 28 |
| Private normal host/POD records in original | 0 | 112 across four phases |
| Final ordinary post-clean health | Present | Missing |
| Exit code | 0 | 137 |
| Docker `OOMKilled` | false | false |
| Controller timeout | false | false |

The private container started at `2026-11-18T17:20:56.516151123Z` and exited
at `17:26:48.523265194Z`. Scoped Docker events preserve SIGTERM, ten seconds
later SIGKILL, exit137 and `stop`. The controller found an already-exited
container and removed only its owned container. This identifies an external
Docker stop, but not its caller; no particular SSH client, ATK queue or service
is attributed. Post-cleanup device checks passed. An idle snapshot does not
prevent future external cleanup.

## Independent review and reader boundary

The exact frozen source108 reader ran directly on both originals, without
rewriting controller status or filling missing fields. It made 123 checks and
returned exit2, `INCOMPLETE_OR_REJECTED_ALL_NORMAL_NO_F`. Its reported issue
was `baseline: KeyError('finish')`, at the unfinished fifth phase.

The stock summary independently accounts for 280 actual API calls, 140 decode
boundaries, five requests, full payloads and repeat/cleanup checks. The reader
did not produce a completed private summary or any cross-arm comparisons.
Four private captures exist, but the incomplete arm has no independent
acceptance. There is no stock/private normal-equivalence certificate.

The stock result is 65,705,149 bytes. The incomplete private result is
131,592,390 bytes, close to the reader's 134,217,728-byte limit. It embeds
repeated capture history; completing the fifth phase could exceed that bound.
This is a separate serialization constraint, not the cause of Docker stop.
Frozen source and original bytes are retained unchanged.

The new group implementation uses a separate source/package and per-phase
artifact references, with each original rehashed and read independently.
It must retain explicit file/total bounds and require complete phases. The
old partial result cannot qualify the new package, and no old frozen reader
or raw result is patched to make this run appear complete.

## Remaining work

The group needs its own complete normal pair, current-input admission,
isolated per-target F/OFF and full recovery within the unchanged budget.
No source105/source108 certificate transfers automatically. External cleanup
coordination is pending; expired host Docker-entry isolation was not restored.

Accepted totals remain **C2 9/17, C3 8/17** in their named scopes. Originals:
`/home/17-fault-c2c3-20261117/v11-33-all-normal-source108-card5-a1`.
Independent report:
`/home/17-fault-c2c3-20261117/v11-source108-original-review-incomplete-a1`.
