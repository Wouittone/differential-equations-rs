# v1.4.1 acceptance run

The final downstream acceptance harness was run on 2026-09-20 with three
independent rounds of nine timing samples and three independent rounds of nine
instrumented-memory samples per backend. Raw JSONL results and provenance
manifests are retained in the session acceptance artifact directory.

| Backend | Matched cases | Median migrated/upstream time | Median peak-live memory |
| --- | ---: | ---: | ---: |
| SatKit | 80 | 1.0468x | 1.0000x |
| Brahe | 68 | 0.9875x | 1.0656x |

The migrated paths completed the same cases as their upstream counterparts.
SatKit time ratios ranged from 0.9963x to 1.8948x; Brahe ranged from 0.9418x
to 1.0333x. These are whole-arc compatibility measurements, not isolated
stage-kernel measurements. Allocation bytes, retained bytes, peak live bytes,
and process memory remain separate metrics.

The library workspace targeted tests, formatting, clippy, downstream smoke
crates, tableau fixtures, and benchmark provenance checks passed. Full host
library suites still report one pre-existing numerical failure in each
downstream checkout (`satkit` BLS formulation equivalence and `brahe`
warm-start evaluation-count threshold); those failures are retained as
unresolved regressions rather than hidden by tolerance changes.
