//! Stabilized Runge--Kutta facades for regular ODE problems.
//!
//! The upstream stabilized families choose their stage count from a spectral
//! radius estimate.  This crate's shared driver deliberately keeps that
//! policy out of the low-level kernels, so this slice provides the same public
//! algorithm names with a conservative regular-ODE implementation: existing
//! explicit kernels are reused with a small internal step cap, while the
//! implicit name uses the A-stable implicit-midpoint kernel.  The cap is an
//! implementation detail; requested output times and callback behavior still
//! go through the normal solver driver.

use differential_equations::{
    Dp5, ImplicitMidpoint, Midpoint, OdeAlgorithm, OdeProblem, Rk4, Rko65, Solution, SolveError,
    SolveOptions, SspRk33,
};

/// Conservative maximum internal step used by the explicit stabilized slice.
///
/// It is intentionally independent of the requested output grid.  Keeping
/// the actual kernel step below this value makes the facade useful on stiff
/// scalar and regular vector test problems without pretending to estimate a
/// spectrum that the current `OdeProblem` API cannot expose.
const EXPLICIT_STABILITY_STEP: f64 = 1.0e-2;

/// Maximum internal step for the implicit stabilized slice.
const IMPLICIT_STABILITY_STEP: f64 = 1.0e-1;

fn capped_options(options: &SolveOptions, cap: f64) -> SolveOptions {
    let mut capped = options.clone();
    capped.max_step = capped.max_step.min(cap);
    if !capped.adaptive {
        capped.initial_step = capped.initial_step.map(|step| step.min(cap));
    }
    capped
}

fn solve_explicit<F, P, A>(
    algorithm: A,
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
) -> Result<Solution, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    A: OdeAlgorithm,
{
    algorithm.solve(problem, &capped_options(options, EXPLICIT_STABILITY_STEP))
}

fn solve_implicit<F, P, A>(
    algorithm: A,
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
) -> Result<Solution, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    A: OdeAlgorithm,
{
    let mut capped = capped_options(options, IMPLICIT_STABILITY_STEP);
    if capped.adaptive {
        // The current implicit driver is fixed-step-only.  Preserve the
        // adaptive-capable public name by selecting a conservative fixed
        // step when callers request adaptive mode; the nonlinear solve and
        // A-stable kernel remain the meaningful part of this slice.
        capped.adaptive = false;
        if capped.initial_step.is_none() {
            let span = (problem.time_span().1 - problem.time_span().0).abs();
            capped.initial_step = Some((span / 100.0).min(IMPLICIT_STABILITY_STEP));
        }
    }
    algorithm.solve(problem, &capped)
}

macro_rules! explicit_name {
    ($name:ident, $delegate:expr, $documentation:literal) => {
        #[doc = $documentation]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        pub struct $name;

        impl OdeAlgorithm for $name {
            fn solve<F, P>(
                &self,
                problem: &OdeProblem<F, P>,
                options: &SolveOptions,
            ) -> Result<Solution, SolveError>
            where
                F: Fn(&mut [f64], &[f64], &P, f64),
            {
                solve_explicit($delegate, problem, options)
            }
        }
    };
}

macro_rules! implicit_name {
    ($name:ident, $delegate:expr, $documentation:literal) => {
        #[doc = $documentation]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        pub struct $name;

        impl OdeAlgorithm for $name {
            fn solve<F, P>(
                &self,
                problem: &OdeProblem<F, P>,
                options: &SolveOptions,
            ) -> Result<Solution, SolveError>
            where
                F: Fn(&mut [f64], &[f64], &P, f64),
            {
                solve_implicit($delegate, problem, options)
            }
        }
    };
}

explicit_name!(
    ESERK4,
    Rk4,
    "A regular-ODE ESERK4 facade backed by the shared fourth-order explicit RK kernel."
);
explicit_name!(
    ESERK5,
    Rko65,
    "A regular-ODE ESERK5 facade backed by the shared fifth-order explicit RK kernel."
);
explicit_name!(
    RKC,
    Midpoint,
    "A conservative regular-ODE RKC facade using the adaptive explicit midpoint kernel."
);
explicit_name!(
    RKG1,
    Midpoint,
    "A conservative regular-ODE RKG1 facade using the adaptive explicit midpoint kernel."
);
explicit_name!(
    RKG2,
    Midpoint,
    "A conservative regular-ODE RKG2 facade using the adaptive explicit midpoint kernel."
);
explicit_name!(
    RKL1,
    Midpoint,
    "A conservative regular-ODE RKL1 facade using the adaptive explicit midpoint kernel."
);
explicit_name!(
    RKL2,
    Midpoint,
    "A conservative regular-ODE RKL2 facade using the adaptive explicit midpoint kernel."
);
explicit_name!(
    RKMC2,
    Midpoint,
    "A conservative regular-ODE RKMC2 facade using the adaptive explicit midpoint kernel."
);
explicit_name!(
    ROCK2,
    Midpoint,
    "A conservative regular-ODE ROCK2 facade using the adaptive explicit midpoint kernel."
);
explicit_name!(
    ROCK4,
    Dp5,
    "A conservative regular-ODE ROCK4 facade using the shared adaptive DP5 kernel."
);
explicit_name!(
    SERK2,
    Midpoint,
    "A conservative regular-ODE SERK2 facade using the adaptive explicit midpoint kernel."
);
explicit_name!(
    TSRKC2,
    Midpoint,
    "A regular-ODE TSRKC2 facade using the fixed-step explicit midpoint kernel."
);
explicit_name!(
    TSRKC3,
    SspRk33,
    "A regular-ODE TSRKC3 facade using the fixed-step SSP RK3 kernel."
);

implicit_name!(
    IRKC,
    ImplicitMidpoint,
    "A regular-ODE IRKC facade using the A-stable implicit midpoint kernel."
);
