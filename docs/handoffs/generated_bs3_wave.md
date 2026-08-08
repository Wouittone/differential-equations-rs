# Generated BS3 coefficient wave

This wave migrates the existing native `Bs3` explicit Runge–Kutta tableau to
the deterministic generated coefficient fixture. The solver implementation,
shared driver, FSAL lifecycle, controller, dense output, and public API are
unchanged; only coefficient ownership moves from `src/explicit_rk.rs` to
`src/generated_coefficients.rs`.

## Pinned source

The source algorithm is `BS3` in OrdinaryDiffEqLowOrderRK at pinned revision
`211142263781255a9aa2f910f6760b9f18ec29c8`:

- `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_tableaus.jl`,
  `BS3ConstantCache` (tableau and embedded 2nd-order defect);
- `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_perform_step.jl`,
  `perform_step!` for the FSAL stage lifecycle.

The generated Rust values preserve the pinned nodes, strictly lower-triangular
rows, third-order weights, and embedded error weights exactly. No SDE, DAE,
split/IMEX, wrapper, or external solver behavior is included.

## Validation

The added integration test checks endpoint accuracy and the FSAL RHS-work
invariant. Full crate gates and both pinned Julia checks are run before merge.

## Scope and limitations

No new algorithm is introduced and no numerical behavior is intentionally
changed. Dense-output and controller behavior remain those of the frozen
explicit driver. Further generated migrations should follow one family at a
time with the same pinned-source and regression evidence.
