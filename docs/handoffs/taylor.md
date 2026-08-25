# Taylor-series solver family

The three pinned Taylor constructors are available without requiring callers to
replace the crate's plain `f64` RHS with an automatic-differentiation scalar.
The kernel builds the actual solution Taylor series coefficient by coefficient:
known state coefficients are continued to Chebyshev nodes, the corresponding
RHS coefficient is recovered through a pre-factorized interpolation matrix,
and the next solution coefficient follows from `u' = f(u,t)`.

- `ExplicitTaylor2` is the fixed second-order polynomial.
- `ExplicitTaylor::new(order)` supports orders 1 through 12 and adaptive step
  control from the first omitted coefficient.
- `ExplicitTaylorAdaptiveOrder::new(min, max)` compares nearby coefficient
  errors against their work estimates and changes order after accepted steps.

This is a Taylor-series construction, not a renamed Runge--Kutta method. Rust
tests cover configured design orders, fixed/adaptive control, and the fixed-only
contract. The Julia fixture compares second-, eighth-, and bounded adaptive-
order polynomials on an in-place vector problem so the pinned adaptive cache
starts at the same order as Rust.

Accepted Taylor coefficients also form the native dense polynomial used by
`save_at`, continuous-root localization, and retained `Solution::interpolate`
queries. Callback effects bound the owning polynomial at the pre-effect state.
