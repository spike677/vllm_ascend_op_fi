# QMM physical-availability review

The complete reported matrix supports the existing refusal to reduce QMM C2
to the two stable sites. No runtime, eligibility predicate or Native mechanism
was changed by this review.

| Observations | Eligible-geometry counts | Union | Intersection | Variable |
| --- | --- | ---: | ---: | ---: |
| Two stock healthy requests | 20, 8 | 26 | 2 | 24 |
| Six observed non-F controls | 20, 8, 15, 20, 8, 20 | 30 | 2 | 28 |

The common ordinals are 83 and 103. Of the 24 stock transitions, 18 lose
adjacency and six gain it: 12/4 change between adjacency and reversed block
order, and six/two change between adjacency and intervening blocks. The six
controls expose four additional eligible-geometry ordinals: 30, 53, 73 and
113. They cannot be hidden by choosing a convenient healthy pair. Geometry
alone does not establish lossless qualification or permission to inject.

All 135 calls in all eight requests remain present: 1,080 calls total. Seven
requests each have 132 allocator observations and three missing observations,
giving 924 observation entries. The F request has 15 eligible, 117 positively
nonadjacent and three unobserved entries; it is excluded from the non-F set
calculations. The last health request retains 135 Native calls but no allocator
observations or retained callback target keys. It cannot be counted as 135
negative eligibility observations or used to establish a site-identity join.
All reported ordinals are zero-based Native call positions, not model layer
numbers.

There are 106 always-proven-nonadjacent sites across the two stock requests,
and 102 across the six controls. Three sites remain unobserved in both cases.
Those categories, the two stable sites and all variable sites stay separate.
The [dynamic-eligibility design](../../docs/worker_core_v1/writer26_qmm_dynamic_eligibility_design.md)
correctly retains the full 135-site universe. The current fixed eligibility
contract still refuses unresolved physical transitions; isolated F/OFF request
scheduling does not repair or bypass them.

## Reader corrections

Two future-input hazards were found and corrected in the read-only reader:

- Set comparison previously used ordinals while only reporting whether target
  keys/formals stayed equal. It now requires complete unique target identities,
  stable Native formal digests and unchanged ordering before producing set
  conclusions. Missing, duplicate, changed or reordered identities retain the
  matrix and counts, but produce unavailable sets and a failed alignment check.
- F exclusion previously trusted `row.plan.mode` without verifying the fixed
  eight-phase contract. The reader now verifies phase and mode, and neither
  the known F position nor a declared F can enter healthy/control calculations.

The existing atomic geometry helper is now SHA-pinned and both reader/helper
hashes are recorded. Empty observed-control selections remain explicitly
unobservable. No saved success flag or set intersection grants a live target
capability. Nine focused CPU tests pass, including identity and F-label
negative cases; CLI help and diff checks also pass.

## Evidence boundary

This independent review reread the local 2,121,520-byte upstream analysis
report, SHA256 `d921b2e5c28ffe317b76f4ada0cb6285173214f62b73aac4e6785bb22003f8ad`,
and recomputed its sets, transitions, counts and site alignment. All 135 keys
and stable formal identities align in each of the seven observed requests.
The corrected comparisons reproduce the published sets; the identified
reader hazards do not change the current report's numerical conclusions.

The upstream report states 4,445 checks and 924 raw allocator files rehashed.
This local review does **not** repeat those raw-file rehashes or the separate
C1 source/writer/consumer/recovery audit. The revised reader should be rerun
against the original 124 files to produce an amended full-original report.
No SSH, container, NPU or new injection was performed here.

The accompanying [compact summary](qmm26_availability_independent_summary.json)
contains hashes and aggregate counts only. It does not contain raw allocator
files, addresses or the complete matrix. C2, C3, E5 and new Native qualification
remain false.
