# Generated DP5 coefficient wave

This wave moves the existing native `Dp5` Dormand–Prince 5(4) tableau into
the deterministic generated coefficient fixture. The shared explicit driver,
FSAL lifecycle, adaptive controller, dense-output behavior, and public API are
unchanged; only coefficient ownership moves from `src/explicit_rk.rs` to
`src/generated_coefficients.rs`.

## Pinned source

The source algorithm is `DP5` in OrdinaryDiffEqLowOrderRK at pinned revision
`211142263781255a9aa2f910f6760b9f18ec29c8`:

- `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_tableaus.jl:1039-1128`,
  `DP5ConstantCacheActual`, supplies the stage coefficients and nodes;
- `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_perform_step.jl:654-711`,
  `perform_step!` supplies the FSAL stage and embedded defect lifecycle.

The generated Rust values retain the exact rational tableau used by the local
implementation. The pinned Julia source's decimal constants are equivalent
representations of those rational values. Dense interpolation coefficients
(`DP5_dense_ds`) are not copied because the Rust shared driver currently uses
its generic dense representation and does not expose upstream's lazy dense
stage path.

## Validation

`tests/generated_dp5.rs` checks endpoint accuracy and the FSAL RHS-work
invariant. The generated fixture unit test checks stage, weight, and defect
shapes plus consistency of the primary weights. Full Rust gates and both pinned
Julia checks pass before merge.

## Scope and limitations

No new algorithm is introduced and no numerical behavior is intentionally
changed. SDE, DAE, split/IMEX, external-wrapper, and specialized upstream
dense-output behavior remain out of scope.
