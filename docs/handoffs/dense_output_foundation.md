# Phase 6 dense-output foundation

`solution.rs` now defines a crate-private `DenseSegment` seam and a checked
cubic `HermiteSegment` implementation with endpoint derivative data. It
validates dimensions/times and reproduces both endpoints exactly; midpoint
behavior is covered by a regression test. The existing recorder remains
linear until kernels provide endpoint derivative segments, so no public
trajectory behavior changes in this bounded foundation slice.

Validation:

```text
cargo fmt -- --check: pass
cargo test --all-targets: pass (87 unit/integration tests plus examples)
cargo clippy --all-targets -- -D warnings: pass
git diff --check: pass
```

Method-specific segment wiring and controller parity remain later Phase 6
waves after split/mass and family migrations.
