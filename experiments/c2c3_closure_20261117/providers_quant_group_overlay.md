# Managed quant load: scope identity checks to actual marker owners

Root's actual device-free `quant-group-managed-load-cpu-a1` stopped before
loading the group DSO. The guard compared every system executable mapping's
device/inode with `stat`, so normal container overlay mappings were treated as
changed quant libraries. The original failure remains at
`/home/17-fault-c2c3-20261117/quant-group-managed-load-cpu-a1/load-probe/result.json`;
controller SHA256
`2dfa3c2e3cfc3532837a58451e95eb29727d14431726ebc87b660d8c29afd825`.

Root's read-only follow-up found nine ordinary system mappings with
`/proc/self/maps` device `(253,2)` and `stat` overlay device `(0,47)`, with
matching inodes. `/proc/self/map_files` links were readable but their stat was
denied. Raw mapping data remain in `quant-mapping-overlay-cpu-a1/mappings.json`;
controller SHA256
`1c7c5da77784ae694d66b7f5441bb2de5040471dc3d42074e742105fb2b06e5e`.
No quant library was loaded and no device was exposed in either diagnostic.

The guard now collects executable ranges without statting every unrelated
file. It obtains existing handles with `RTLD_NOLOAD`, looks up the actual quant
markers, resolves each returned address with `dladdr`, and only then validates
its owning mapping. Actual marker owners still require an exact executable
address range, undeleted path, matching dladdr path, file inode/device and ELF
content hash. The open file and path are checked again after the read. An
unknown overlay identity for a quant library remains rejected. This does not
generalize system overlay observations into a quant-library identity waiver.

Six CPU fixture tests cover unrelated overlay/deleted mappings, exact marker
identity, marker device/inode mismatch, deleted marker, wrong address/path and
non-ELF content. The actual device-free container retry belongs to root; these
fixtures and the earlier host marker tests are not evidence of container
acceptance. The correction changes Python only and can be reviewed separately
from the pending empty-owner Native cancellation increment.
