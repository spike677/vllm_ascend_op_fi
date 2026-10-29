# Native msprobe B/F statistical comparison

`analyze_msprobe_native.py` reads exactly four files for each arm: its collection config and the native `dump.json`, `stack.json`, `construct.json` for one step/rank. It never reads runtime snapshots, tensor binaries, injected-hit counters, expected fault locations, host tiling output or receipt claims. Run it separately for each corresponding step/rank.

```bash
python analyze_msprobe_native.py \
  --baseline B/observation/step0/rank0 --baseline-config B/config.json \
  --fault F/observation/step0/rank0 --fault-config F/config.json \
  --expected-sdpa-calls 28 --output native_step0_comparison.json
```

The output must be new. Both configs must request `task=statistics`, `level=mix`, `summary_mode=statistics`, `tensor_list=[]`; output directories may differ, other configuration must match. The analyzer checks the native API family, complete parsed source stack, construct ancestry, occurrence and recorded order. Different shapes or dtypes do not become a same-target comparison. The default coverage gate expects the 28 HF attention modules and one owned SDPA call per layer in this model's captured step; it does not specify a desired first-difference location.

Two observations are reported independently:

- `raw_first_recorded_difference_including_diagnostics` retains the first recorded statistical/metadata difference or sequence mismatch, including diagnostic operations.
- `first_observed_model_api_numerical_difference` and `first_observed_model_api_output_difference` exclude only explicit diagnostic source-function pairs. For the HF probe these are `hf_runtime_probe.py:record` and `identity`. The `observe` wrapper's real `self.original(...)` call remains in the model API trace. Exclusions include their original stacks and ordinals for audit.

Missing stacks, reordered/unmatched nodes or incompatible argument/config metadata suppress the complete model API first-difference claim. Matching four summary statistics never means equal elements. Null fields are unobservable, including null versus finite; earlier unavailable fields are counted before a later observed difference. Changed input summaries remain visible and prevent describing an output change as occurring with matching input summaries. Even all matching input summaries do not prove identical Q/K/V or masks.

Coverage distinguishes PyTorch SDPA, ATen flash attention, torch_npu fusion attention / FIA, and an explicitly named `FlashAttentionScore` record if one exists. SDPA alone is not a directly observed CANN kernel. No native-only result claims kernel-launch identity, legal-key coverage, precision policy, M/S1/S2 changes, non-SplitKV behavior, exact target-slice localization, or mechanism acceptance. Those require separate evidence. Record order is a hook-completion order, not a device execution timeline.

CLI exit 0 means the captured traces align and the declared HF SDPA/module coverage is complete. It is **not a fault acceptance result**; `fault_mechanism_accepted` and `tensor_equality_proven` always remain false. An aligned report with no observed difference may still contain unavailable statistics. Partial alignment or missing model coverage returns exit 1; malformed evidence raises a validation error.

`test_msprobe_native.py` uses small synthetic JSON fixtures solely to test the analyzer. Its negative cases cover empty statistics, incompatible batch shapes, changed inputs, reordered calls, missing stacks, construct cycles, capture-mode mismatch and incomplete model coverage. It also verifies that an early diagnostic difference remains in the raw result without replacing the later model API boundary, and that malformed control/runtime files are never consumed. The existing 87-node fault15 native statistics fixture was independently read as a schema compatibility check; its missing HF coverage is correctly reported. Neither test is fault28 NPU evidence.
