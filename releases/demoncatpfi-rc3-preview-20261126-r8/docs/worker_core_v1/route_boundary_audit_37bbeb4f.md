# Worker route and source-only handoff audit at 37bbeb4f

This is a bounded source audit of that exact commit. It does not refresh device
acceptance, inherit mechanism results, or describe later 8/18 work in progress.
No additional device run, package build or historical result rescan was used.

`worker_extension/registry.py` enumerates 17 canonical mechanisms and 13 callable
factories. A callable factory still requires a trusted deployment and current
target qualification. The four `WORKER_RPC_ADAPTATION_PENDING` entries are
`28.fa.legal_tiling`, `28.fia.legal_tiling`, `28.fia.denominator_replay`, and
`33.paged_decode`. Their separate implementations/diagnostics are not public
Worker factory acceptance.

| Route at the audited commit | Actual dispatch boundary and limitation |
| --- | --- |
| `8.restore_offset`, `15.alias_detach`, `17.spatial_padding`, `18.api_half`, `25.reduce_scatter` | `providers/tensor_modules.py:TensorModuleProvider._execute/_phase` calls trusted `binding.replay(self.model, phase)` during the synchronous RPC. These five factories are not a naturally enqueued standard-Worker serving adapter. |
| `20.buffer_substitute` | `providers/fault20.py` supports `Fault20ServingReplay`/`Fault20CoverageReplay`, and retains a `QualifiedNativeSession` fallback whose check/replay calls `binding.replay(self.model)`. Report the selected binding, rather than claiming every factory execution is natural Core. |
| `39.matmul_range` | `providers/fault39.py` distinguishes `Fault39ServingReplay`/`Fault39CoverageReplay` from `Fault39WorkerReplay`. The latter constructs `QualifiedFault39Backend` with a synchronous trusted replay callback. MODEL_LOCAL and separate replay acceptance cannot be transferred to every Worker factory execution. |
| `26.residual_stomp`, `26.fia_destination` | Existing `serving26.py` bindings are a different four-phase deployment. Source54/62 natural single-site experiments use `providers26_group_probe.py` callable helpers and `_demoncat_writer26_group_experiment`, not WorkerFaultClient/FaultConfig. The accompanying new Core26 branch adds real public Host wiring for a future source; it does not qualify those previous runs retroactively. |
| `22.wrong_row`, `41.tail_column`, `39.output_proxy`, `quant.batch_shared_scale` | Reviewed serving bindings expose distinct natural-call boundaries. Availability still does not prove the particular binding was deployed, every mechanism accepted, or C2/C3/E5 transferred. Output39 in particular has a new single-site C1 scope separate from MODEL_LOCAL coverage. |

End-user configuration is `FaultConfig(fault_id, variant)` plus optional
implementation version. `HostAdapter.route` consumes actual identities and
requires a trusted policy for multiple Workers; it does not guess rank zero.
Trusted deployment code supplies existing runtime objects, ABI/build identity,
prompt and evidence paths. The user does not need a layer name, rank, device,
raw address or native-load setting. This public surface is distinct from an
already integrated private framework: yellow-zone integration remains pending.

## Concrete handoff gaps and recommendations

* Keep `rpc_provider_count`, natural serving deployment availability and actual
  accepted scopes as separate fields. The five synchronous tensor factories and
  20/39 fallbacks make “13 natural Core routes” inaccurate at this commit.
* `describe_worker_provider().mechanism_boundary` comes from the historical
  catalog. In particular the 22 text can still describe lookup rows while its
  current route selects actual fused positions. Expose catalog history separately
  from the current binding boundary; do not let old 9.1/operator evidence grant
  a current 8.5.1 Worker admission.
* The three original Python distributions and the source-only exporter remain
  the handoff artifacts. Generated legacy HostContracts are not a single proof
  that all 17 Worker variants are implemented; use the actual Worker manifest
  and provider-specific deployment contracts alongside them.
* The generic `NativeBuildManager` 26 recipes invoke `native_build/build26.sh`
  and build legacy writer DSOs. Although `_F26_SOURCES` includes group source,
  the recipe does not produce `libdcat26_group_v2.so` or its typed profile.
  Current group-v2 and quant-group build/probe procedures live under the reviewed
  experiment tree. A source-only bundle must state that external build
  prerequisite, or later add a reviewed typed recipe; it must not claim the
  legacy recipe builds these newer deployments.
* The exporter intentionally excludes experiments. At the audited commit this
  also excludes current group single-site/typed-FIA handoff documents and probe
  harnesses, plus several non-`test_core_*` CPU fixtures. The new Core26 closure
  includes its exact reused fixture dependencies and this contract, while keeping
  experiments/history/raw arrays outside the bundle. Deployment harnesses remain
  an explicitly separate reviewed source snapshot; they are not silently copied.

No platform qualification inflation was found in the source-only compatibility
matrix: CANN 8.5.1/A2 is described as mechanism-specific prior evidence only;
CANN 9.x and A3 are `NOT_ACCEPTED_BY_THIS_BUNDLE`. Those declarations must remain
independent of factory counts, CPU tests, source copying and old catalog notes.
