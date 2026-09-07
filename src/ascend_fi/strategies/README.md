# Strategies

Strategies describe reusable value or storage transformations independently of
the operator that exposes the target. Examples already validated in the legacy
runtime include INT2/INT4-equivalent transforms, BF16 mantissa truncation,
finite extreme-value transforms, and explicit non-finite output assignment.

A strategy owns mutation semantics. It does not own symbol resolution, request
policy, production lifecycle control, or DemonCAT primitive registration.
