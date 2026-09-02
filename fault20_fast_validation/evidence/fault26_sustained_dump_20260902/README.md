# Fault26 Sustained Dump Evidence — 2026-09-02

## Scope

This evidence set contains curated msprobe statistics dumps for Fault26
sustained all-hit validation on the local vLLM Ascend model service.

## Fault model

FIA wrong-destination residual stomp:

1. Track AddRms residual producer/consumer topology.
2. When the relation gate is valid, write FIA output payload into residual
   storage.
3. Let the next AddRms consume the corrupted residual.
4. Restore residual storage after consumer.
5. Verify output degradation and recovery.

## Included runs

- `runs/nohit_dumpon/`
  - Fault26 runtime loaded.
  - Writer disabled.
  - msprobe dump enabled.

- `runs/allhit1024_dumpon/`
  - Sustained all-hit mode.
  - Target bound: 1024 hits.
  - Visible text degradation was observed in the corresponding runtime
    validation.

- `runs/allhit4096_dumpon/`
  - Sustained all-hit upper bound: 4096 hits.
  - Actual hit count was lower because the model emitted early EOS.
  - Early EOS collapse was observed in the corresponding runtime validation.

## Dump contents

Each run contains msprobe files for both ranks:

- `step0/rank0/construct.json`
- `step0/rank0/dump.json`
- `step0/rank0/stack.json`
- `step0/rank1/construct.json`
- `step0/rank1/dump.json`
- `step0/rank1/stack.json`

## Summaries

- `summaries/nohit_dumpon_dump_summary.json`
- `summaries/allhit1024_dumpon_dump_summary.json`
- `summaries/allhit4096_dumpon_dump_summary.json`

## Size

- 21 committed dump/summary files.
- Approximately 75.82 MiB in this evidence set.

## Status labels

- `FAULT26_SUSTAINED_ALLHIT_OUTPUT_DEGRADATION_PASS`
- `FAULT26_SUSTAINED_ALLHIT_EARLY_EOS_COLLAPSE_OBSERVED`
- `FAULT26_SUSTAINED_ALLHIT_RECOVERY_PASS`

## Boundary

This evidence demonstrates proxy-induced sustained FIA wrong-destination
residual stomp. It does not prove spontaneous natural OOB in the vendor kernel.
