# Target validation

Validation consumes an accepted shared `Candidate` and invokes only a lossless
developer/backend adapter. It emits the shared `ValidationResult` and never
arms a destructive strategy.

V0 accepts the proven QMM temporary-buffer behavior: the original storage is
left untouched, a distinct temporary enters the real Run, the stream is
synchronized, and the temporary is freed. Hash fields remain `null` when the
legacy backend did not record content hashes; they are never fabricated.
