# FA a2: independently verified baseline, rejected envelope comparison

The immutable `fa28-launch-short-a2` baseline records a successful actual
FlashAttentionScore launch and completed device synchronization. Its subsequent
Python validation fails because it compares the complete C++ JSON document
with the flat field map. Independent decoding confirms **all205 public fields
agree**. This is a partial baseline result, not B/F/OFF or model acceptance.

The audit selected the actual registered binary/function and argument slot,
then compared raw bytes; it did not select a launch using a requested profile:

* Registered ELF bytes match the fresh compiled kernel, SHA256
  `967265862973a761ae7de2c7b54600e246f067898cd2c97f297aa9a12e784eb0`.
  Function binding and launch use key22619644976, generation1, block_dim16,
  schedule_mode0. Registration, binding, launch and both post-launch device
  synchronization results are zero and belong to the same process/thread/call.
* The actual argument payload is1720 bytes. The compiled FA entry has25
  pointers; its mixed-core hardware prefix places the tiling pointer at byte200.
  The observed placeholder maps that slot to byte216. Those1464 actual argument
  bytes equal both the host raw record and`final-tiling.bin` byte-for-byte.
* Independent standard-library `struct` decoding uses the fresh public schema's
  offsets/types/counts, checks nonoverlap and bounds, and decodes205 fields.
  The192 opaque bytes remain preserved by complete raw-byte equality.
* Running the already-built C++ decoder on these saved bytes uses only the
  existing image's CPU loader/libraries. Its output has exactly the envelope
  keys`schema`, `raw_size`, `fields`; schema is`nes.fa.public_pod_values.v1`,
  raw_size1464. Its `fields` map equals the independently decoded map exactly.
  The entire envelope does not equal that flat map. The frozen source-a31
  validator contains the failing expression`json.loads(decoded.stdout) == pod`.

Actual baseline factors are **B8, M128, S2 base128, nRatio8**, with native
passthrough and no strategy selection applied. The original result remains
`FAILED_WITH_EVIDENCE`, `actual_npu_api_calls=1`, `calls=[]`; the audit does not
rewrite it as a successful whole run. No fault or OFF call was reached.

This review did not read private output tensor arrays, run a new NPU call,
start a container, modify the kernel/build, or change the original evidence.
It does not establish output correctness against stock, numerical propagation,
statistics first difference, all-model coverage, or C2/C3. Root's qualification
counts remain unchanged. A new attempt with the corrected verifier is needed
for the complete sequence; its outcome must be reported separately.

[agent7_fa_a2_partial_summary.json](agent7_fa_a2_partial_summary.json) is the
publishable summary, SHA256
`8d781b6484eea0b7fd6092cab838ef16ba66829ccc660705e771519aa7f8c94a`.
Original runtime/host records and the full decoder output remain under
`/home/17-fault-c2c3-20261117/fa28-launch-short-a2` and
`/home/17-fault-c2c3-20261117/agent7-fa28-a2-review-a1`.
The summary binds the original result, source manifest, validator, completion,
schema, decoder, runtime/host logs, raw tiling and registered binary hashes.
No raw tensor, kernel binary or original statistics are included in this commit.
