# Source106 WeightNz observation build: CPU gate only

The new bounded read-only observer compiled against the actual CANN8.5.1 SDK
in the pinned production image on124. All19 existing container contracts and
28 new typed WeightNz contracts passed. Startup resolution of the original
functions also passed. This build did not execute an NPU operator, observe a
model request or grant C1/C2/C3.

The observer covers the actual23-argument
`aclnnGroupedMatmulWeightNzGetWorkspaceSize` and its execute function. It adds
named slots, current container registrations, group-list metadata, scalar
values, workspace/executor/stream and device observations to the existing
four-interface trace. Its limits are128 calls,32 pairs and1MiB of journal
storage. After successful startup admission, unknown observations remain
failures while the original typed function executes once with its original
return status. Startup resolution failure must refuse before the first model
call: that state does not guarantee original-function forwarding. Pending
executors prevent journal release. Original library mappings intentionally
remain resident until process exit; journal release does not unload them.
There is no injected fault in this observation scope.

| Identity | SHA256 or value |
|---|---|
| Source commit | `922ff7cb1d343a3e16f28d43776552b00123428d` |
| Source manifest | `2317f5947c73aa5ff7ae6644fd9b1ef26aaa36aab325d20557b870ff13db0bb4` |
| Production image | `sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0` |
| Build manifest | `6a2414e6e02a17d1abe06175a294b850f56908c24b31625b951708448996e620` |
| Build receipt | `4b09364f8c847aa750548b389af21706936ec3cf30331b36c08c9b5e3989920d` |
| Built observer | `51b0d92465232cbc26ad2d93416b028bd1601c3dcadc4282b25304512600526d` |
| Existing19 CPU contract receipt | `8af2b370210a9feec96956906212bdff6ff8d7a1e5c0c9781cf436b546190b89` |
| New28 CPU contract receipt | `be83aa2b1d03b823bf2926db4ee864974c9542d7c57ae3642821f5c2883f612c` |
| Original-resolution receipt | `98c39c84287e3dd4cb75454c4fb8b012b728e0299f78a486f5f2340815f13a1f` |

Original metadata and binaries remain at
`/home/17-fault-c2c3-20261117/quant-gmm2-trace-sdk-a1`.
Root rehashed the build/receipt and all13 build source dependencies against
the integration tree. The build container was confirmed absent after cleanup.
The no-device build used no explicit driver or operator calls; loader
constructors may probe device information, which is not counted as model
execution or runtime acceptance.

The next gate is a same-process standard Worker observation with startup-owned
handles, followed by independent review of the actual pairs and all135
projection calls. Interface observations cannot establish request-row
semantics by themselves. The current proven row count remains10/135.
