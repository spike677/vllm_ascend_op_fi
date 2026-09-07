# Discovery

Discovery maps a semantic candidate to an environment-specific physical
binding. It may fingerprint symbols, shapes, ranks, phases, tensor slots, or
call relationships.

Environment-specific values such as `call_index=217` are discovery results,
not stable production target ABI. DemonCAT decides which logical fields are
exposed to users and passes binding options to this repository's backend.

V0 reads the existing native `qmm_run_trace` JSONL schema, normalizes it to an
`ObservedFact`, matches the capability-manifest snapshot, and emits/ranks the
shared `Candidate` contract. Direct GWS address matches remain diagnostic only;
automatic lossless validation requires the complete semantic-slot proof.
