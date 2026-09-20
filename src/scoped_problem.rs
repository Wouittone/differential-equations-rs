//! Borrowed and thread-eligible problem storage for reusable solvers.
//!
//! Unlike the original [`crate::OdeProblem`], these problems store hooks in
//! their concrete types. Captures need not be `'static`; `Send` and `Sync`
//! are inferred from the captures, without imposing either bound on sequential
//! callers. Mutable captures are exclusively borrowed during integration.
//!
//! Existing `OdeProblem` users can also place references in their parameter
//! context and let stateless callbacks access them, or construct a problem
//! inside each Rayon worker. The latter repeats problem setup; the scoped
//! representation can instead be constructed once and moved to an eligible
//! worker. Legacy constructors return `SolveError`; this API retains arbitrary
//! application payloads through `IntegrationError::Step(StepError::User(E))`.
use crate::stepping::{
    AdaptiveController, ExplicitRungeKuttaStepper, IntegrationError, IntegrationOutcome,
    Observation, ObserverAction, integrate_rk,
};

/// Default observer function type.
pub type ScopedObserver<E> = for<'a> fn(Observation<'a>) -> Result<ObserverAction, E>;
/// Default Jacobian function type; row-major matrix output.
pub type ScopedJacobian<E> = fn(f64, &[f64], &mut [f64]) -> Result<(), E>;
/// Default finalizer function type.
pub type ScopedFinalizer<E> = fn(IntegrationOutcome, &[f64]) -> Result<(), E>;

/// A lifetime-neutral first-order problem with concrete fallible hooks.
///
/// RHS and Jacobian use `(time, state, output)`, matching reusable steppers.
/// The Jacobian output is row-major `dimension × dimension`. Captured parameter
/// references replace a separate parameter object, including borrowed context.
/// An eligible problem is automatically `Send`/`Sync`; a closure capturing an
/// `Rc` is intentionally not transferable. Owning a `Sync` problem does not
/// allow simultaneous mutable solves: each solve still needs `&mut self`.
///
/// ```compile_fail
/// use differential_equations::ScopedOdeProblem;
/// use std::{rc::Rc, convert::Infallible};
/// let shared = Rc::new(1.0);
/// let problem = ScopedOdeProblem::new(move |_: f64, _: &[f64], out: &mut [f64]| {
///     out[0] = *shared;
///     Ok::<_, Infallible>(())
/// }, [0.0], (0.0, 1.0));
/// fn require_send<T: Send>(_: T) {}
/// require_send(problem);
/// ```
pub struct ScopedOdeProblem<
    F,
    E,
    O = ScopedObserver<E>,
    J = ScopedJacobian<E>,
    L = ScopedFinalizer<E>,
> {
    rhs: F,
    initial_state: Vec<f64>,
    time_span: (f64, f64),
    observer: O,
    jacobian: Option<J>,
    finalizer: L,
    error: std::marker::PhantomData<fn() -> E>,
}
fn observe<E>(_: Observation<'_>) -> Result<ObserverAction, E> {
    Ok(ObserverAction::Continue)
}
fn finalize<E>(_: IntegrationOutcome, _: &[f64]) -> Result<(), E> {
    Ok(())
}
impl<F, E> ScopedOdeProblem<F, E>
where
    F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), E>,
{
    /// Creates a problem with no-op observer/finalizer and no analytic Jacobian.
    pub fn new(rhs: F, initial_state: impl Into<Vec<f64>>, time_span: (f64, f64)) -> Self {
        Self {
            rhs,
            initial_state: initial_state.into(),
            time_span,
            observer: observe::<E>,
            jacobian: None,
            finalizer: finalize::<E>,
            error: std::marker::PhantomData,
        }
    }
}
impl<F, E, O, J, L> ScopedOdeProblem<F, E, O, J, L> {
    /// Initial state in contiguous component order.
    pub fn initial_state(&self) -> &[f64] {
        &self.initial_state
    }
    /// Initial and final integration times.
    pub fn time_span(&self) -> (f64, f64) {
        self.time_span
    }
    /// Attaches an observer borrowing local state, run after accepted steps.
    pub fn with_observer<N>(self, observer: N) -> ScopedOdeProblem<F, E, N, J, L>
    where
        N: FnMut(Observation<'_>) -> Result<ObserverAction, E>,
    {
        ScopedOdeProblem {
            rhs: self.rhs,
            initial_state: self.initial_state,
            time_span: self.time_span,
            observer,
            jacobian: self.jacobian,
            finalizer: self.finalizer,
            error: std::marker::PhantomData,
        }
    }
    /// Attaches a borrowed fallible row-major analytic Jacobian.
    pub fn with_jacobian<N>(self, jacobian: N) -> ScopedOdeProblem<F, E, O, N, L>
    where
        N: FnMut(f64, &[f64], &mut [f64]) -> Result<(), E>,
    {
        ScopedOdeProblem {
            rhs: self.rhs,
            initial_state: self.initial_state,
            time_span: self.time_span,
            observer: self.observer,
            jacobian: Some(jacobian),
            finalizer: self.finalizer,
            error: std::marker::PhantomData,
        }
    }
    /// Attaches a finalizer called after successful completion or observer stop.
    /// A finalizer failure is propagated as the original typed user error.
    pub fn with_finalizer<N>(self, finalizer: N) -> ScopedOdeProblem<F, E, O, J, N>
    where
        N: FnMut(IntegrationOutcome, &[f64]) -> Result<(), E>,
    {
        ScopedOdeProblem {
            rhs: self.rhs,
            initial_state: self.initial_state,
            time_span: self.time_span,
            observer: self.observer,
            jacobian: self.jacobian,
            finalizer,
            error: std::marker::PhantomData,
        }
    }
    /// Borrows RHS and analytic Jacobian together for a host-controlled stepper.
    /// Their concrete closures return the original application error `E`.
    pub fn functions_mut(&mut self) -> (&mut F, Option<&mut J>) {
        (&mut self.rhs, self.jacobian.as_mut())
    }
}
impl<F, E, O, J, L> ScopedOdeProblem<F, E, O, J, L>
where
    F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), E>,
    O: FnMut(Observation<'_>) -> Result<ObserverAction, E>,
    L: FnMut(IntegrationOutcome, &[f64]) -> Result<(), E>,
{
    /// Resets a reusable RK workspace to this problem and integrates it.
    ///
    /// The caller owns norm/controller/output policy and can reuse workspace
    /// allocation. This starts a fresh trajectory; use `integrate_rk` directly
    /// for continuation. Attached Jacobians are retained for implicit host
    /// steppers and are not evaluated by an explicit RK method.
    pub fn integrate<N>(
        &mut self,
        stepper: &mut ExplicitRungeKuttaStepper<'_>,
        controller: &mut AdaptiveController,
        requested_times: &[f64],
        maximum_attempts: usize,
        norm: &mut N,
    ) -> Result<IntegrationOutcome, IntegrationError<E>>
    where
        N: FnMut(&[f64], &[f64], &[f64]) -> Result<f64, E>,
    {
        stepper
            .reset(self.time_span.0, &self.initial_state)
            .map_err(crate::stepping::StepError::Solver)?;
        controller.reset(controller.next_step())?;
        let outcome = integrate_rk(
            stepper,
            controller,
            self.time_span.1,
            requested_times,
            maximum_attempts,
            &mut self.rhs,
            norm,
            &mut self.observer,
        )?;
        (self.finalizer)(outcome, stepper.state()).map_err(crate::stepping::StepError::User)?;
        Ok(outcome)
    }
}
