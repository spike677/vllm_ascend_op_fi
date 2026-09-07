# Backends

Backends bind a fault or observation operation to one Ascend execution plane.
They resolve timing-local physical details such as symbols, tensor slots,
device addresses, streams, and ABI versions.

They do not define DemonCAT semantic positions or production CLI arguments.
During the transition, the canonical `aclnn` backend implementation remains in:

- `fault20_fast_validation/source/l4_buffer_substitute_allhit_v1/`
- `fault20_fast_validation/source/l4_metadata_lifecycle_proxy/`
- `fault20_fast_validation/source/fault26_readonly_trace_v1/`

Code moves here only after its build and runtime contracts can be preserved
without a second copy of the native implementation.
