mod array_adapters;
mod builder;
mod callbacks;
mod function;
mod jacobian;
mod ordinary;
mod split;

#[cfg(test)]
mod tests;

use crate::SolveError;
use crate::callback::{
    Callback, InitializationHook, LifecycleHook, PredictiveDomainPolicy, StepGuard,
};
pub use builder::OdeProblemBuilder;
pub use function::{MutableFunction, OdeFunction};
#[allow(unused_imports)] // Preserve the existing crate-internal facade path.
pub(crate) use jacobian::JacobianProvider;
use ndarray::IxDyn;

/// An initial-value ordinary differential equation problem.
///
/// In-place right-hand sides use the SciML calling convention
/// `f(du, u, p, t)` and overwrite every element of `du` with the derivative.
/// [`Self::from_array_out_of_place`] instead accepts `f(u, p, t)` returning
/// an ndarray with the state's shape. Both forms implement [`OdeFunction`].
pub struct OdeProblem<F, P> {
    pub(crate) rhs: F,
    initial_state: Vec<f64>,
    state_shape: IxDyn,
    time_span: (f64, f64),
    parameters: P,
    jacobian: Option<Box<JacobianFunction<P>>>,
    callbacks: Vec<Callback<P>>,
    initializers: Vec<InitializationHook<P>>,
    finalizers: Vec<Box<LifecycleHook<P>>>,
    step_guards: Vec<StepGuard<P>>,
    predictive_domains: Vec<PredictiveDomainPolicy<P>>,
}

type JacobianFunction<P> = dyn Fn(&mut [f64], &[f64], &P, f64);
type StepInterpolator<'a> = dyn FnMut(f64, &mut [f64]) -> Result<(), SolveError> + 'a;

/// A split/IMEX ODE representation retaining explicit and implicit components.
///
/// The representation is solver-neutral: kernels choose how to combine the
/// two components, while dimensions, parameters, and time semantics remain
/// shared and checked in one place.
/// Components can write derivatives in place or return ndarrays through
/// [`Self::from_array_out_of_place`].
#[allow(dead_code)]
pub struct SplitOdeProblem<FE, FI, P> {
    explicit: FE,
    implicit: FI,
    initial_state: Vec<f64>,
    state_shape: IxDyn,
    time_span: (f64, f64),
    parameters: P,
    implicit_jacobian: Option<Box<JacobianFunction<P>>>,
    callbacks: Vec<Callback<P>>,
    initializers: Vec<InitializationHook<P>>,
    finalizers: Vec<Box<LifecycleHook<P>>>,
    step_guards: Vec<StepGuard<P>>,
    predictive_domains: Vec<PredictiveDomainPolicy<P>>,
}
