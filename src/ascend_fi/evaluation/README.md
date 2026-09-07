# Evaluation

Evaluation compares baseline, fault, and recovery runs using common metrics:
response/token divergence, non-finite propagation, hit/effect repeatability,
request health, and exact recovery.

Evaluation code may provide development runners and report generators. It does
not own the production injection CLI or controller lifecycle.
