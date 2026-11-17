# TND33 stock/private API coexistence

The original `attention33-baseline-a1` failed during the unchanged Qwen profile
run at `post_attention_layernorm`: `aclnnAddRmsNormBias` and its workspace API
could not be found. No controlled decode was reached. That attempt remains
unchanged and unqualified; stock a1 is a separate successful observation.

The actual immutable 8.5.1 image
`sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0`
contains two different API resolution paths. Its
`csrc/aclnn_torch_adapter/op_api_common.h` SHA
`142023f0f0e682b3b79aeb987044ecbd9b7db0b63b20de967447ba6f9e3d6b16`
opens bare `libcust_opapi.so` and checks it before `libopapi.so`. The unchanged
vLLM extension's RUNPATH includes its stock custom API directory. Prepending the
private FIA `op_api/lib` to LD_LIBRARY_PATH defeats that RUNPATH. The private
package has only `libcust_opapi.so`, not `libopapi.so`; its missing Norm symbol
causes the misleading fallback error. Its SONAME is the same as the stock
custom library. Read-only image ELF inspection confirmed the stock library
exports both Norm symbols and has SHA
`de1633c2f1f41e505277697ee82f3ffba940f730bed9986c5aeff4c8514b713f`.

The new bootstrap runs before Torch/vLLM import in the dedicated child:

1. Verify the exact stock source/extension/API hashes and fresh private manifest.
2. Open the unchanged stock custom library by absolute path, then its bare name;
   dladdr must bind both Norm entry points to that exact stock ELF.
3. Open the private library by absolute path, retaining both live handles.
   dladdr must bind both FIA entry points to the fresh private ELF. Recheck that
   the bare name still selects stock and both library handles remain distinct.
4. Run the unchanged official platform `import_kernels` initializer. Its only
   action in this pinned source is registering its stock vendor environment.
   Require the exact resulting stock/private list, then order it private/stock
   for explicit custom package lookup. No private directory is added to the
   global library search path. No model operator has run at this point.

The NativeAdapter requires this same-process bootstrap and exact two-vendor
order, checks the environment for each eligible call, and records symbol
ownership separately from actual final kernel arguments. The original C++ Norm
adapter, Python model, profile run and natural cache writer are unchanged. OFF
still uses the original private-package policy, with exact stock/private payload
comparison required by the independent bridge. Library resolution alone does
not establish an actual Torch NPU FIA dispatch or any native qualification.

The first full no-device preflights preserved a native abort during the explicit
vLLM C++ extension import. `attention33-loading-cpu-a2` proves that actual symbol
ownership, Torch/vLLM platform import and the official vendor initializer had
already passed. The independent `attention33-loading-stock-control-a1`, with no
private preload and only the stock vendor environment, reproduces the same
`basic_string::_S_construct null not valid` abort in extension `create_module`.
This establishes that the abort is not specific to the private package; it does
not establish the lower-level C++ cause or a successful complete CPU import.
All failed attempts remain unchanged.

The minimal exact-image CPU preflight now verifies actual dlsym/dladdr ownership
before and after the unchanged official platform initializer. It explicitly
reports full extension import and native Worker initialization as unverified.
The controller uses `runc`, no devices/driver/network, and the same fresh build
read-only. Optional `--include-extension-import` and `--stock-extension-control`
retain the complete import diagnostics. No operator or model is run in CPU scope.

```sh
R=/home/17-fault-c2c3-20261117
S=$R/source-NEW
python3 "$S/experiments/c2c3_closure_20261117/run_attention33_loading_preflight_124.py" \
  --source "$S" --native "$R/tnd33-fai-build-a1/build/run" \
  --out "$R/attention33-loading-cpu-a3"
```

After `CPU_API_BINDINGS_AND_PLATFORM_ENV_VERIFIED`, root builds fresh installed
wheels including `paged33_loading.py`, then queues a new private baseline with
the same actual build/schema/observer hashes and ordinary Worker arguments.
The source mount and installed wheel must both include this change. Keep stock
a1 and failed baseline a1 unchanged. Do not run F until the new full baseline,
actual FIA launches and exact independent stock/private bridge pass. Existing
launch proof rejects any accidental stock FIA dispatch or wrong private entry.
The real device-backed Worker performs its complete ordinary initialization,
including the original C++ extension, Norm and profile run, before the probe
requires a new actual Norm/FIA binding recheck. No such initialization is skipped.

Local CPU contracts: 14 loading/TND/payload tests pass. Exact-image CPU loading
and new Worker execution remain pending root execution; no C1/C2/C3 grant.
