# Pinned-upstream explicit RK dense-output and controller audit

Summary:

This report is an implementation-ready source map for every public first-order
explicit Runge--Kutta algorithm currently exported by the Rust crate through
`src/explicit_rk.rs`, `src/tsit5.rs`, and `src/verner.rs`. It covers 22 methods:
Euler, Midpoint, Heun, Ralston, RK4, RKM, Ralston4, Alshina2, Alshina3, BS3,
DP5, OwrenZen3/4/5, BS5, SSPRK22/33/43, Tsit5, and Vern6/7/8/9.

The audit traced runtime dispatch rather than relying on documentation strings.
In particular:

- RK4 is adaptive upstream, using Shampine residual/defect control; endpoint-only
  classical RK4 is not adaptive parity.
- DP5, OwrenZen3/4/5, BS5, SSPRK22/33/43, Tsit5, and Vern6/7/8/9 have specialized
  dense formulas. The remaining methods dispatch to the core cubic-Hermite
  fallback.
- BS5 and all four Verner methods default to lazy interpolation. Their extra
  interpolation stages are not part of endpoint stepping and must be represented
  separately.
- The pinned generic Hermite fallback assumes `k[1]` and `k[2]` are endpoint
  derivatives. The Alshina2, Alshina3, and RKM cache layouts do not satisfy that
  assumption. This is pinned upstream behavior, not a license to silently replace
  it with a different interpolant.
- Every adaptive method uses the core PI controller by default. DP5 is the only
  method in scope with specialized PI gains.

Files changed:

- `docs/handoffs/explicit_upstream_audit.md`

Public APIs added:

- None.

## Evidence boundary and revision

Upstream source and revision:

- Repository: `SciML/OrdinaryDiffEq.jl`
- Read-only checkout: `D:/Source/_review/OrdinaryDiffEq.jl`
- Verified `git rev-parse HEAD`: `211142263781255a9aa2f910f6760b9f18ec29c8`
- All paths and line numbers below are relative to that checkout at that commit.
- Julia metaprogramming is followed to its source. `@OnDemandTableauExtract`
  consumers are cited together with the concrete tableau constructor from which
  fields are extracted.

Scope exclusions are exactly those in the overnight plan. This report does not
audit other low-order methods, extended SSP/low-storage methods, implicit or
second-order methods, SDE/DDE/BVP/PDE/steady-state behavior, DAE-only residual
behavior, or wrappers/composites such as `AutoTsit5`.

## Shared dispatch that applies to every method

### Adaptive/fixed classification and order metadata

`OrdinaryDiffEqAdaptiveAlgorithm` is the upstream opt-in to adaptivity; the
fallback `OrdinaryDiffEqAlgorithm` is fixed-only
(`lib/OrdinaryDiffEqCore/src/alg_utils.jl:290-299`). An adaptive algorithm can
also run with `adaptive=false`; a fixed-only algorithm rejects `adaptive=true`,
and fixed stepping requires `dt` or `tstops`
(`lib/OrdinaryDiffEqCore/src/solve.jl:277-283`).

All method orders in this report resolve through package `alg_utils.jl` files:

- low-order RK: `lib/OrdinaryDiffEqLowOrderRK/src/alg_utils.jl:1-27`
- SSPRK: `lib/OrdinaryDiffEqSSPRK/src/alg_utils.jl:1-33`
- Tsit5: `lib/OrdinaryDiffEqTsit5/src/alg_utils.jl:1-3`
- Verner: `lib/OrdinaryDiffEqVerner/src/alg_utils.jl:1-18`

Unless overridden, `alg_adaptive_order(alg) = alg_order(alg) - 1`
(`lib/OrdinaryDiffEqCore/src/alg_utils.jl:700-705`). None of the in-scope
methods overrides it.

### Default PI controller

All adaptive methods in scope use the generic `PIController(QT, alg)` default
(`lib/OrdinaryDiffEqCore/src/alg_utils.jl:708-719`). The common defaults are:

| field | adaptive default | source |
|---|---:|---|
| `qmin` | `1/5` | `lib/OrdinaryDiffEqCore/src/alg_utils.jl:378-399` |
| `qmax` | `10` | `lib/OrdinaryDiffEqCore/src/alg_utils.jl:401-420` |
| `gamma` | `9/10` | `lib/OrdinaryDiffEqCore/src/alg_utils.jl:779-791` |
| `qsteady_min`, `qsteady_max` | `1`, `1` | `lib/OrdinaryDiffEqCore/src/alg_utils.jl:804-843` |
| first-accepted-step `qmax` | `10000` | `lib/OrdinaryDiffEqCore/src/alg_utils.jl:848-857` |
| `qoldinit` | `1e-4` | `lib/OrdinaryDiffEqCore/src/integrators/controllers.jl:744-764` |
| `beta1` | `7/(10p)` | `lib/OrdinaryDiffEqCore/src/alg_utils.jl:758-777` |
| `beta2` | `2/(5p)` | `lib/OrdinaryDiffEqCore/src/alg_utils.jl:740-755` |

Here `p = alg_order(alg)`. DP5 alone overrides `beta2 = 0.04` and
`beta1 = 1/p - 3*beta2/4 = 0.17`
(`lib/OrdinaryDiffEqLowOrderRK/src/alg_utils.jl:37-39`). The scoped defaults are
therefore:

| methods | `beta1` | `beta2` |
|---|---:|---:|
| Midpoint, Heun, Ralston, Alshina2 | `0.35` | `0.20` |
| Alshina3, BS3, OwrenZen3, SSPRK43 | `7/30` | `2/15` |
| RK4, OwrenZen4 | `0.175` | `0.10` |
| BS5, OwrenZen5, Tsit5 | `0.14` | `0.08` |
| DP5 | `0.17` | `0.04` |
| Vern6 | `7/60` | `1/15` |
| Vern7 | `0.10` | `2/35` |
| Vern8 | `0.0875` | `0.05` |
| Vern9 | `7/90` | `2/45` |

Euler, RKM, Ralston4, SSPRK22, and SSPRK33 are fixed-only, so these adaptive
controller fields are not active for them.

The actual PI update is at
`lib/OrdinaryDiffEqCore/src/integrators/controllers.jl:795-837`:

- zero error sets inverse growth `q = 1/qmax`;
- otherwise `q11 = EEst^beta1` and
  `q = clamp((q11 / errold^beta2) / gamma, 1/qmax, 1/qmin)`;
- accepted `dtnew` is `dt/q`, with the inclusive steady band forcing `q=1`;
- `errold` becomes `max(EEst, qoldinit)`;
- rejection changes `dt` by
  `dt /= min(1/qmin, q11/gamma)` and does not update `errold`.

That convention is important: upstream's `q` is the inverse of the growth
factor commonly stored by Rust controllers.

### Initial-step selection

If an adaptive solve has no explicit `dt`, it uses the shared
Hairer--Norsett--Wanner estimator, not a method-specific starting rule. The
in-place path is `lib/OrdinaryDiffEqCore/src/initdt.jl:17-297`; the out-of-place
path is `lib/OrdinaryDiffEqCore/src/initdt.jl:346-458`.

For regular ODEs it constructs `sk = abstol + internalnorm(u0)*reltol`, computes
`d0 = norm(u0/sk)` and `d1 = norm(f0/sk)`, chooses
`dt0 = 1e-6` when either is below `1e-5` and otherwise
`dt0 = (d0/d1)/100`, evaluates an Euler trial and `f1`, estimates
`d2 = norm((f1-f0)/sk)/dt0`, and returns
`direction * max(dtmin, min(100dt0, dt1, dtmax))`, where
`dt1 = 10^(-(2+log10(max(d1,d2)))/p)` unless the curvature is tiny
(`lib/OrdinaryDiffEqCore/src/initdt.jl:200-297` and `:418-458`). `dtmin` is at
least the next float above `eps(t)`, and the small fallback is at least `1e-6`
(`:27-28`, `:356-357`). The initial derivative may reuse the preinitialized
FSAL buffer (`:71-75`).

### Dense dispatch

Every query first calls `SciMLBase.addsteps!` and then dispatches
`ode_interpolant` on the concrete cache
(`lib/OrdinaryDiffEqCore/src/dense/generic_dense.jl:202-225`). This is what makes
lazy BS5/Verner stages appear only when dense data is requested.

If there is no specialized cache method, `_ode_interpolant` chooses cubic
Hermite when at least two `k` vectors exist and linear interpolation otherwise
(`lib/OrdinaryDiffEqCore/src/dense/generic_dense.jl:1472-1515`). The Hermite
formula consumes exactly `k[1]` and `k[2]` as start/end derivatives
(`:1518-1573`) and supports state plus derivatives through order three. The
generic user-facing summary is `"3rd order Hermite"`; `dense=false` reports
linear (`lib/OrdinaryDiffEqCore/src/interp_func.jl:64-75`).

Specialized interpolation packages install an initial catch-all for their cache
unions and throw `DerivativeOrderNotPossibleError` outside explicitly supplied
derivative methods. Thus unsupported derivative orders must not fall through to
Hermite:

- low-order RK: `lib/OrdinaryDiffEqLowOrderRK/src/interpolants.jl:1-23`
- Tsit5: `lib/OrdinaryDiffEqTsit5/src/interpolants.jl:1-17`

### Limiters and cache invalidation

Every constructor in scope exposes `stage_limiter!` and `step_limiter!`, both
defaulting to `trivial_limiter!`; mutable-cache methods also expose a `thread`
choice defaulting to `Serial()`. Stage limiters are called after construction of
each stage state in the mutable `perform_step!` paths. The solve-level step
limiter is centralized and applied once to the accepted state
(`lib/OrdinaryDiffEqCore/src/integrators/integrator_utils.jl:175-183`).

After a callback/state modification, the core resizes `k`, forces eager
non-lazy interpolation stages to be recomputed, and marks FSAL for reevaluation
(`lib/OrdinaryDiffEqCore/src/integrators/integrator_interface.jl:54-78`). At the
step boundary, an FSAL method either copies `fsallast` to `fsalfirst` or calls
`reset_fsal!` after discontinuity/modification
(`lib/OrdinaryDiffEqCore/src/integrators/integrator_utils.jl:186-208` and
`:944-959`). Rejected explicit attempts leave their start derivative valid.

The generic FSAL trait is `true` for an `OrdinaryDiffEqAlgorithm`
(`lib/OrdinaryDiffEqCore/src/alg_utils.jl:73-81`). SSPRK22/33/43 explicitly set
it false (`lib/OrdinaryDiffEqSSPRK/src/alg_utils.jl:3-10`), as do Vern7/8/9
(`lib/OrdinaryDiffEqVerner/src/alg_utils.jl:1-3`). Tsit5, Vern6, and the scoped
low-order methods inherit true. This is a cache protocol trait, not always a
statement that the last mathematical RK stage equals the next first stage.

## Per-method source map

Names in the first column are Rust public names; parenthesized names are the
upstream spelling where it differs.

| Rust/upstream | definition; mode/order | coefficients and endpoint/error step | dense dispatch and cache facts |
|---|---|---|---|
| `Euler` | `lib/OrdinaryDiffEqLowOrderRK/src/algorithms.jl:1-20`; fixed, p=1 | caches `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_caches.jl:1-46`; formula and endpoint RHS `lib/OrdinaryDiffEqLowOrderRK/src/fixed_timestep_perform_step.jl:1-43` | generic Hermite; `k=[f0,f1]`. Endpoint RHS is done for interpolation/cache reuse even though endpoint stepping itself is one-stage Euler. |
| `Midpoint` | `lib/OrdinaryDiffEqLowOrderRK/src/algorithms.jl:58-70`; adaptive or fixed, p=2, Euler estimator | caches `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_caches.jl:139-158`; formula/error `lib/OrdinaryDiffEqLowOrderRK/src/fixed_timestep_perform_step.jl:166-238` | generic Hermite; interior stage storage is overwritten and `k=[f0,f1]`. |
| `Heun` | `lib/OrdinaryDiffEqLowOrderRK/src/algorithms.jl:30-42`; adaptive or fixed, p=2, Euler estimator | caches `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_caches.jl:57-128`; shared Heun/Ralston formulas/error `lib/OrdinaryDiffEqLowOrderRK/src/fixed_timestep_perform_step.jl:46-164` | generic Hermite; `k=[f0,f1]`. |
| `Ralston` | `lib/OrdinaryDiffEqLowOrderRK/src/algorithms.jl:44-56`; adaptive or fixed, p=2, Euler estimator | same cache/step sources as Heun; branch coefficients at `lib/OrdinaryDiffEqLowOrderRK/src/fixed_timestep_perform_step.jl:69-72`, error at `:87-95`/`:151-160` | generic Hermite; `k=[f0,f1]`. |
| `Rk4` (`RK4`) | `lib/OrdinaryDiffEqLowOrderRK/src/algorithms.jl:72-90`; adaptive or fixed, p=4 | caches `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_caches.jl:186-219`; RK4 stages at `lib/OrdinaryDiffEqLowOrderRK/src/fixed_timestep_perform_step.jl:240-262`/`:318-350`; Shampine two-point residual control at `:263-312`/`:351-410` | generic Hermite; `k=[f0,f1]`. Adaptive parity requires the two extra RHS defect probes at `sigma=(1/2 +/- sqrt(3)/6)` and `EEst=2.1342*max(e1,e2)`. |
| `Rkm` (`RKM`) | `lib/OrdinaryDiffEqLowOrderRK/src/algorithms.jl:279-299`; fixed, p=4 | recurrence/cache coefficients `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_caches.jl:1078-1149`; step `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_perform_step.jl:1072-1157` | no specialization: generic Hermite. Pinned caveat: `k[2]` is stage 2, while the separately computed endpoint derivative is `fsallast`; the fallback nevertheless reads `k[2]`. |
| `Ralston4` | `lib/OrdinaryDiffEqLowOrderRK/src/algorithms.jl:301-318`; fixed, p=4 | tableau `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_tableaus.jl:1968-2000`; cache `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_caches.jl:1698`; step `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_perform_step.jl:2213-2273` | generic Hermite; stored `k=[f0,f1]`. |
| `Alshina2` | `lib/OrdinaryDiffEqLowOrderRK/src/algorithms.jl:443-465`; adaptive or fixed, p=2 | tableau `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_tableaus.jl:1832-1852`; cache `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_caches.jl:1558`; step/error `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_perform_step.jl:1930-2007`; estimator is `dt*k1` | no specialization: generic Hermite. Pinned caveat: `k[2]=f(t+2dt/3,stage)`, not endpoint `f1`, and the method recomputes `k1` inside each attempt despite inheriting the true FSAL trait. |
| `Alshina3` | `lib/OrdinaryDiffEqLowOrderRK/src/algorithms.jl:467-489`; adaptive or fixed, p=3 | tableau `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_tableaus.jl:1854-1882`; cache `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_caches.jl:1603`; step/error `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_perform_step.jl:2010-2095`; estimator is `dt*(4/9)*k2` | no specialization: generic Hermite. Pinned caveat: fallback reads `k[2]` at `t+dt/2`, not the stored final stage `k[3]` at `t+3dt/4` nor an endpoint derivative. |
| `Bs3` (`BS3`) | `lib/OrdinaryDiffEqLowOrderRK/src/algorithms.jl:92-110`; adaptive or fixed, p=3 | tableau `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_tableaus.jl:1-61`; cache `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_caches.jl:250`; step `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_perform_step.jl:1-80` | generic Hermite; cache exposes start and endpoint/FSAL derivative as its two interpolation vectors. |
| `Dp5` (`DP5`) | `lib/OrdinaryDiffEqLowOrderRK/src/algorithms.jl:198-216`; adaptive or fixed, p=5 | tableau plus dense basis coefficients `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_tableaus.jl:1039-1194`; step/error and compressed four-vector dense basis `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_perform_step.jl:654-791` | specialized free order-4 interpolant. Evaluation is `y0+dt*(k1*theta+k2*theta(1-theta)+k3*theta^2(1-theta)+k4*theta^2(1-theta)^2)` at `lib/OrdinaryDiffEqLowOrderRK/src/interpolants.jl:25-90`; summary `lib/OrdinaryDiffEqLowOrderRK/src/interp_func.jl:34-42`. FSAL true. |
| `OwrenZen3` | `lib/OrdinaryDiffEqLowOrderRK/src/algorithms.jl:112-131`; adaptive or fixed, p=3 | tableau, embedded error, and `r` dense coefficients `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_tableaus.jl:63-133`; cache `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_caches.jl:311`; step `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_perform_step.jl:81-167` | specialized free order-3 polynomial using four stage vectors, `lib/OrdinaryDiffEqLowOrderRK/src/interpolants.jl:284-350`; summary `lib/OrdinaryDiffEqLowOrderRK/src/interp_func.jl:1-11`. FSAL true, no extra dense RHS. |
| `OwrenZen4` | `lib/OrdinaryDiffEqLowOrderRK/src/algorithms.jl:133-152`; adaptive or fixed, p=4 | tableau/error/`r` coefficients `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_tableaus.jl:135-273`; cache `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_caches.jl:361`; step `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_perform_step.jl:168-269` | specialized free order-4 polynomial using stages 1,3,4,5,6, `lib/OrdinaryDiffEqLowOrderRK/src/interpolants.jl:520-600`; summary `lib/OrdinaryDiffEqLowOrderRK/src/interp_func.jl:12-22`. FSAL true, no extra dense RHS. |
| `OwrenZen5` | `lib/OrdinaryDiffEqLowOrderRK/src/algorithms.jl:154-173`; adaptive or fixed, p=5 | tableau/error/`r` coefficients `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_tableaus.jl:275-511`; cache `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_caches.jl:415`; step `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_perform_step.jl:270-406` | specialized free order-5 polynomial using stages 1,3-8, `lib/OrdinaryDiffEqLowOrderRK/src/interpolants.jl:840-932`; summary `lib/OrdinaryDiffEqLowOrderRK/src/interp_func.jl:23-33`. FSAL true, no extra dense RHS. |
| `Bs5` (`BS5`) | `lib/OrdinaryDiffEqLowOrderRK/src/algorithms.jl:175-196`; adaptive or fixed, p=5, `lazy=Val(true)` default | complete endpoint, dual-estimator, extra-stage, and dense coefficient record `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_tableaus.jl:513-1038`; base and optional eager step `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_perform_step.jl:408-650`; `EEst=max(EEst1,EEst2)` at `:452-469`/`:586-612` | specialized lazy order-5 interpolant, evaluation `lib/OrdinaryDiffEqLowOrderRK/src/interpolants.jl:1299-1411`; summary `lib/OrdinaryDiffEqLowOrderRK/src/interp_func.jl:43-51`. Base `k` size 8; eager size 11. Three extra stages are computed only for an accepted/fixed step (`lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_perform_step.jl:481-516`/`:614-649`) or lazily by `lib/OrdinaryDiffEqLowOrderRK/src/low_order_rk_addsteps.jl:132-207` and `:499-601`. FSAL true. |
| `SspRk22` (`SSPRK22`) | `lib/OrdinaryDiffEqSSPRK/src/algorithms.jl:15-28`; fixed, p=2 | cache `lib/OrdinaryDiffEqSSPRK/src/ssprk_caches.jl:1-40`; Shu--Osher step `lib/OrdinaryDiffEqSSPRK/src/ssprk_perform_step.jl:1-49` | shared specialized free order-2 SSP interpolant `lib/OrdinaryDiffEqSSPRK/src/interpolants.jl:1-76`, `P(theta)=(1-theta^2)y0+theta^2*y1+dt*theta(1-theta)f0`; summary `lib/OrdinaryDiffEqSSPRK/src/interp_func.jl:1-14`. One `k` vector; start RHS is recomputed because FSAL=false. |
| `SspRk33` (`SSPRK33`) | `lib/OrdinaryDiffEqSSPRK/src/algorithms.jl:240-253`; fixed, p=3 | cache `lib/OrdinaryDiffEqSSPRK/src/ssprk_caches.jl:42-70`; Shu--Osher step `lib/OrdinaryDiffEqSSPRK/src/ssprk_perform_step.jl:121-176` | same shared order-2 SSP interpolant and no-FSAL policy as SSPRK22. Dense order is 2, not the main method order 3. |
| `SspRk43` (`SSPRK43`) | `lib/OrdinaryDiffEqSSPRK/src/algorithms.jl:72-101`; adaptive or fixed, p=3 | cache/constants `lib/OrdinaryDiffEqSSPRK/src/ssprk_caches.jl:771-832`; four-stage Shu--Osher step and embedded estimator `lib/OrdinaryDiffEqSSPRK/src/ssprk_perform_step.jl:755-849` | same shared order-2 SSP interpolant. FSAL=false. The constructor documentation mentions an optimized controller, but this pinned implementation has no method-specific controller override and dispatches to the generic PI controller. |
| `Tsit5` | `lib/OrdinaryDiffEqTsit5/src/algorithms.jl:1-22`; adaptive or fixed, p=5 | tableau/error `lib/OrdinaryDiffEqTsit5/src/tsit_tableaus.jl:7-230`; dense `r` coefficients `:238-320`; seven-stage step/error `lib/OrdinaryDiffEqTsit5/src/tsit_perform_step.jl:125-261` | specialized free order-4 polynomial over all seven stage derivatives, `lib/OrdinaryDiffEqTsit5/src/interpolants.jl:19-143`; summary `lib/OrdinaryDiffEqTsit5/src/interp_func.jl:1-11`. No extra dense RHS; FSAL true. |
| `Vern6` | `lib/OrdinaryDiffEqVerner/src/algorithms.jl:1-25`; adaptive or fixed, p=6, `lazy=Val(true)` | extra stages `lib/OrdinaryDiffEqVerner/src/verner_tableaus.jl:2-199`, interpolation coefficients `:200-382`, endpoint tableau `:383-584`; step/error/eager extras `lib/OrdinaryDiffEqVerner/src/verner_rk_perform_step.jl:1-110` and mutable path `:112-246` | specialized lazy order-6 interpolant `lib/OrdinaryDiffEqVerner/src/interpolants.jl:25-181`; summary `lib/OrdinaryDiffEqVerner/src/interp_func.jl:1-11`. Base/eager `k` sizes 9/12, three extra RHS stages; lazy addsteps `lib/OrdinaryDiffEqVerner/src/verner_addsteps.jl:1-79` and `:547-653`. FSAL true. |
| `Vern7` | `lib/OrdinaryDiffEqVerner/src/algorithms.jl:27-51`; adaptive or fixed, p=7, lazy default | extra stages `lib/OrdinaryDiffEqVerner/src/verner_tableaus.jl:585-797`, interpolation coefficients `:798-1067`, endpoint tableau `:1068-1369`; step/error/eager extras `lib/OrdinaryDiffEqVerner/src/verner_rk_perform_step.jl:247-383` and mutable path `:385-573` | specialized lazy order-7 interpolant `lib/OrdinaryDiffEqVerner/src/interpolants.jl:184-377`; summary `lib/OrdinaryDiffEqVerner/src/interp_func.jl:13-23`. Base/eager `k` sizes 10/16, six extra RHS stages; lazy addsteps `lib/OrdinaryDiffEqVerner/src/verner_addsteps.jl:80-196` and `:654-818`. FSAL=false. |
| `Vern8` | `lib/OrdinaryDiffEqVerner/src/algorithms.jl:53-77`; adaptive or fixed, p=8, lazy default | extra stages `lib/OrdinaryDiffEqVerner/src/verner_tableaus.jl:1370-1686`, interpolation coefficients `:1687-2066`, endpoint tableau `:2067-2525`; step/error/eager extras `lib/OrdinaryDiffEqVerner/src/verner_rk_perform_step.jl:574-765` and mutable path `:767-1014` | specialized lazy order-8 interpolant `lib/OrdinaryDiffEqVerner/src/interpolants.jl:380-666`; summary `lib/OrdinaryDiffEqVerner/src/interp_func.jl:25-35`. Base/eager `k` sizes 13/21, eight extra RHS stages; lazy addsteps `lib/OrdinaryDiffEqVerner/src/verner_addsteps.jl:197-354` and `:819-1044`. FSAL=false. |
| `Vern9` | `lib/OrdinaryDiffEqVerner/src/algorithms.jl:79-102`; adaptive or fixed, p=9, lazy default | extra stages `lib/OrdinaryDiffEqVerner/src/verner_tableaus.jl:2526-2985`, interpolation coefficients `:2986-3496`, endpoint tableau `:3497-4290`; 16 endpoint/error RHS values plus compressed `k` map at `lib/OrdinaryDiffEqVerner/src/verner_rk_perform_step.jl:1015-1128`, eager extras `:1130-1243`, mutable path `:1246-1547` | specialized lazy order-9 interpolant `lib/OrdinaryDiffEqVerner/src/interpolants.jl:669-1013` (higher derivative methods continue at `:1016-1425`); summary `lib/OrdinaryDiffEqVerner/src/interp_func.jl:37-47`. Base/eager `k` sizes 10/20: endpoint stepping computes 16 stages but stores only the ten vectors used by interpolation; ten extra dense RHS stages. Lazy addsteps `lib/OrdinaryDiffEqVerner/src/verner_addsteps.jl:355-546` and `:1045-1318`. FSAL=false. |

## Specialized dense-output record shapes

The generated coefficient schema should not force these methods into one flat
Butcher tableau. The following tagged shapes preserve upstream structure and
allow compile-time validation.

### `GenericHermite`

```text
GenericHermite {
    advertised_order: 3,
    derivative_orders: 0..=3,
    start_derivative_slot: 0,
    end_derivative_slot: 1,
    fallback_to_linear_if_missing: true,
    caveat: Option<&'static str>,
}
```

Use for Euler, Midpoint, Heun, Ralston, RK4, RKM, Ralston4, Alshina2,
Alshina3, and BS3. Preserve explicit caveats for RKM and both Alshina methods;
do not label their pinned cache layout as a mathematically valid endpoint
Hermite pair without a compliance decision.

### `FreeStagePolynomial`

```text
FreeStagePolynomial {
    order: usize,
    derivative_orders: RangeInclusive<usize>,
    stage_slots: &'static [usize],
    // row per referenced stage; ascending theta powers, exact/decimal literals
    power_coefficients: &'static [&'static [Coefficient]],
    leading_theta_power: &'static [usize],
    storage_transform: Identity | Dp5CompressedBasis,
}
```

Use for DP5, OwrenZen3/4/5, Tsit5, and the shared SSP interpolant. DP5 must
retain its four-vector transformed basis (`update`, `bspl`, two dense
combinations), not pretend those vectors are raw RK stages. The SSP polynomial
is endpoint/start-derivative based and shared across three methods, so represent
it once and reference it.

Invariants:

- every referenced slot is present after an accepted step;
- polynomial endpoint values reproduce `y0` and `y1` at theta 0 and 1;
- derivative order coverage matches actual dispatch (never generic-fallback);
- coefficient rows have the declared maximum degree/order;
- no RHS work is needed solely for evaluation.

### `LazyExtraStagePolynomial`

```text
LazyExtraStagePolynomial {
    order: usize,
    base_stage_count: usize,
    stored_base_slots: &'static [StageSource],
    extra_nodes: &'static [Coefficient],
    extra_rows: &'static [&'static [IndexedCoefficient]],
    interpolation_rows: &'static [&'static [Coefficient]],
    default_lazy: bool,
    eager_only_after_acceptance: bool,
    force_recompute_after_step_shortening: bool,
}
```

Use for BS5 and Vern6/7/8/9. `StageSource` must support a compressed mapping,
because Vern9 computes sixteen endpoint stages but stores only ten interpolation
inputs. Extra-stage rows are separate from endpoint rows and may depend on
previous extra stages. Validations must ensure topological ordering, node/row
count agreement, interpolation row count agreement, and no eager dense RHS on a
rejected attempt.

## Controller record shape and invariants

Controller metadata belongs beside, but not inside, a tableau:

```text
ExplicitControllerMetadata {
    method_order: usize,
    adaptive_order: usize,
    estimator: None | Embedded { count: usize } | Rk4Defect,
    default_controller: Fixed | Pi,
    beta1: Coefficient,
    beta2: Coefficient,
    gamma: Coefficient,
    qmin: Coefficient,
    qmax: Coefficient,
    qmax_first_step: Coefficient,
    qoldinit: Coefficient,
    qsteady: (Coefficient, Coefficient),
    zero_error_uses_qmax: bool,
    rejected_step_updates_history: bool,
}
```

Required invariants:

- fixed methods have no estimator and cannot enable adaptivity;
- adaptive methods declare the exact estimator count (BS5 has two; RK4 has a
  defect estimator rather than embedded weights);
- controller powers are generated from order metadata unless explicitly
  overridden (DP5);
- the implementation documents whether its stored factor is upstream `q` or
  the direct `dt` growth factor;
- rejection never overwrites accepted-error history;
- zero error is finite and uses maximum permitted growth;
- the first accepted step can use the upstream `qmax=10000` exception;
- `dtmin`, `dtmax`, direction, and endpoint/tstop clipping remain driver-owned.

## FSAL and dense-cache lifecycle requirements

1. Treat FSAL as a kernel/cache capability, not merely `last tableau row == b`.
   Several low-order methods obtain reusable endpoint derivatives through a
   separate RHS evaluation.
2. A rejected attempt keeps the derivative at the unchanged start state, but
   must discard candidate-only dense data.
3. BS5 and Verner eager interpolation stages run only after fixed-step completion
   or controller acceptance. Lazy mode computes them on the first dense query.
4. A callback that changes state invalidates FSAL and dense extras. A callback
   that shortens a step requires recomputation with the shortened `dt`; upstream
   explicitly forces this for non-lazy interpolants.
5. SSPRK22/33/43 and Vern7/8/9 recompute their start-stage data according to
   their non-FSAL step paths. Do not synthesize reuse from tableau resemblance.
6. Dense segments used for event location/save-at must represent the pre-effect
   accepted trajectory. Post-effect callback states are separate forced saves.

## Unsupported upstream options and current Rust differences

Numerical differences:

- Rust currently records `save_at` using linear interpolation in
  `src/solution.rs:61-93`; none of the method-specific or generic-Hermite dense
  behaviors above is represented.
- Rust's explicit controller is an I controller with direct factor
  `clamp(0.9*error^(-1/p), 0.2, 10)` and a previous-rejection growth cap
  (`src/explicit_rk.rs:839-891`, `:1078-1085`). It lacks upstream PI history,
  DP5 gains, first-step growth exception, steady band, and exact rejection rule.
- Rust's initial-step routine is similar in outline but lacks upstream `dtmin`,
  unit/type paths, constant-zone return, FSAL-aware initialization, and special
  fallback rules (`src/explicit_rk.rs:920-985`).
- Rust RK4 has no error weights and therefore rejects adaptive use, while
  upstream RK4 is adaptive through Shampine defect control.
- Rust fixed Euler does not compute the endpoint RHS solely for upstream
  Hermite/FSAL cache behavior.
- Rust BS5 has the two endpoint error estimators but omits stages 9--11 and its
  order-5 interpolant.
- Rust Verner tableaus implement endpoint stepping/error estimation but omit
  extra interpolation stages, lazy/eager selection, and dense polynomials.
- Rust marks Alshina2/3 and RKM according to mathematical tableau FSAL structure,
  while upstream's generic `isfsal`/cache protocol and dense storage are more
  idiosyncratic. Compliance must choose pinned behavior explicitly.

Upstream constructor/options not currently exposed by Rust:

- `stage_limiter!`, `step_limiter!`, and `thread` for every method;
- `lazy` for BS5 and Vern6/7/8/9;
- custom controller selection and `qmin`, `qmax`, `gamma`, PI gains,
  `qsteady_*`, `qoldinit`, `dtmin`, and first-step growth behavior;
- dense enable/disable, solution queries, component selection, and interpolant
  derivative queries;
- upstream in-place/out-of-place and generic scalar/array element-type support;
- composite-only stability/eigenvalue estimates made during DP5, Tsit5, and
  Verner stepping.

Allocation/performance impact:

- Report only; no runtime code changed.
- The implementation consequence is material: eager BS5/Vern interpolation
  adds 3/3/6/8/10 RHS calls respectively, but only on accepted/fixed steps.
  Lazy mode avoids those calls until interpolation is actually requested.
- DP5's compressed four-vector dense basis and Vern9's compressed ten-vector
  interpolation storage should be preserved to avoid unnecessary retained stage
  arrays.

Known limitations:

- This is a source audit, not a numerical validation run.
- The Alshina/RKM generic-Hermite mismatch is documented exactly as dispatched
  at the pinned revision. A follow-up compliance task should measure Julia query
  values before deciding whether Rust intentionally reproduces it or records a
  justified divergence.
- Only state dense evaluation is required by the current Rust parity phase;
  upstream derivative-query formulas are mapped by source, but their complete
  coefficient transcription is deferred to generation.
- Custom numeric types, mass matrices, and composite wrappers were inspected
  only where they affect the scoped controller/dense dispatch.

Rust tests:

- None required; report-only task.

Julia tests:

- None required; no Julia project or manifest was mutated.

Commands run:

- `Get-Content -Raw docs/OVERNIGHT_EXECUTION_PLAN.md`
- `Get-Content -Raw docs/AGENT_OPERATING_RULES.md`
- `git -C D:/Source/_review/OrdinaryDiffEq.jl rev-parse HEAD`
- extensive `rg -n --glob '*.jl' ...` dispatch/source searches in the pinned
  checkout
- line-numbered `Get-Content` inspection of every cited source range
- path/line verification script over every citation in this report
- `git diff --check`
- final `git status --short --branch`

Follow-up dependencies:

- Phase 4 coefficient-schema work should add the three dense record variants and
  the controller metadata record before transcribing coefficients.
- Phase 6 driver work must expose accepted-step dense segments, lazy stage
  materialization, and callback invalidation before claiming save-at/event parity.
- Julia compliance should add midpoint dense-query fixtures for every row in the
  method map, with explicit Alshina2/3 and RKM pinned-behavior probes.
- RK4 adaptive support must be implemented separately from a conventional
  embedded-tableau kernel.

Recommended next task:

Implement the explicit dense-schema types and deterministic generator inputs for
DP5, OwrenZen3/4/5, SSPRK22/33/43, and Tsit5 first (all require no extra RHS),
then add the shared lazy-extra-stage machinery for BS5 and Verner. In parallel,
add an explicit PI-controller unit-test fixture that checks DP5's override,
zero-error growth, rejection history, and the first-accepted-step `qmax` rule.
