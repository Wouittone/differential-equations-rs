use crate::SolveError;
use crate::callback::{
    Callback, CallbackAction, CallbackOutcome, ContinuousCallback, DiscreteCallback, EventDirection,
};

/// An initial-value ordinary differential equation problem.
///
/// The right-hand side uses the in-place SciML calling convention
/// `f(du, u, p, t)`: it must overwrite every element of `du` with the
/// derivative at `(u, p, t)`.
pub struct OdeProblem<F, P> {
    pub(crate) rhs: F,
    initial_state: Vec<f64>,
    time_span: (f64, f64),
    parameters: P,
    jacobian: Option<Box<JacobianFunction<P>>>,
    callbacks: Vec<Callback<P>>,
}

type JacobianFunction<P> = dyn Fn(&mut [f64], &[f64], &P, f64);
type StepInterpolator<'a> = dyn FnMut(f64, &mut [f64]) -> Result<(), SolveError> + 'a;

/// A split/IMEX ODE representation retaining explicit and implicit components.
///
/// The representation is solver-neutral: kernels choose how to combine the
/// two components, while dimensions, parameters, and time semantics remain
/// shared and checked in one place.
#[allow(dead_code)]
pub struct SplitOdeProblem<FE, FI, P> {
    explicit: FE,
    implicit: FI,
    initial_state: Vec<f64>,
    time_span: (f64, f64),
    parameters: P,
}

#[allow(dead_code)]
impl<FE, FI, P> SplitOdeProblem<FE, FI, P> {
    pub fn new(
        explicit: FE,
        implicit: FI,
        initial_state: impl Into<Vec<f64>>,
        time_span: (f64, f64),
        parameters: P,
    ) -> Self {
        Self {
            explicit,
            implicit,
            initial_state: initial_state.into(),
            time_span,
            parameters,
        }
    }

    pub fn initial_state(&self) -> &[f64] {
        &self.initial_state
    }

    pub fn time_span(&self) -> (f64, f64) {
        self.time_span
    }

    pub fn parameters(&self) -> &P {
        &self.parameters
    }

    pub fn dimension(&self) -> usize {
        self.initial_state.len()
    }

    pub fn evaluate_explicit(&self, derivative: &mut [f64], state: &[f64], time: f64)
    where
        FE: Fn(&mut [f64], &[f64], &P, f64),
    {
        (self.explicit)(derivative, state, &self.parameters, time);
    }

    pub fn evaluate_implicit(&self, derivative: &mut [f64], state: &[f64], time: f64)
    where
        FI: Fn(&mut [f64], &[f64], &P, f64),
    {
        (self.implicit)(derivative, state, &self.parameters, time);
    }
}

/// A regular ODE with a constant nonsingular dense mass matrix `M*u' = f(u,t)`.
/// Singular/DAE residual initialization is intentionally not represented.
#[allow(dead_code)]
pub struct MassMatrixOdeProblem<F, P> {
    rhs: F,
    initial_state: Vec<f64>,
    time_span: (f64, f64),
    parameters: P,
    mass_matrix: Vec<f64>,
}

#[allow(dead_code)]
impl<F, P> MassMatrixOdeProblem<F, P> {
    pub fn new(
        rhs: F,
        initial_state: impl Into<Vec<f64>>,
        time_span: (f64, f64),
        parameters: P,
        mass_matrix: impl Into<Vec<f64>>,
    ) -> Result<Self, &'static str> {
        let initial_state = initial_state.into();
        if initial_state.is_empty() {
            return Err("mass-matrix ODE state must be non-empty");
        }
        let mass_matrix = mass_matrix.into();
        let expected = initial_state
            .len()
            .checked_mul(initial_state.len())
            .ok_or("mass-matrix dimension overflow")?;
        if mass_matrix.len() != expected || mass_matrix.iter().any(|value| !value.is_finite()) {
            return Err("mass matrix must be a finite square dense matrix");
        }
        Ok(Self {
            rhs,
            initial_state,
            time_span,
            parameters,
            mass_matrix,
        })
    }

    pub fn initial_state(&self) -> &[f64] {
        &self.initial_state
    }

    pub fn time_span(&self) -> (f64, f64) {
        self.time_span
    }

    pub fn parameters(&self) -> &P {
        &self.parameters
    }

    pub fn mass_matrix(&self) -> &[f64] {
        &self.mass_matrix
    }

    pub fn evaluate_rhs(&self, derivative: &mut [f64], state: &[f64], time: f64)
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        (self.rhs)(derivative, state, &self.parameters, time);
    }
}

#[allow(dead_code)]
pub(crate) struct JacobianProvider<'a, F, P> {
    problem: &'a OdeProblem<F, P>,
}

#[allow(dead_code)]
impl<'a, F, P> JacobianProvider<'a, F, P> {
    pub(crate) fn new(problem: &'a OdeProblem<F, P>) -> Self {
        Self { problem }
    }

    pub(crate) fn evaluate(&self, jacobian: &mut [f64], state: &[f64], time: f64) -> bool {
        self.problem.evaluate_jacobian(jacobian, state, time)
    }

    pub(crate) fn is_analytic(&self) -> bool {
        self.problem.has_jacobian()
    }
}

impl<F, P> OdeProblem<F, P> {
    /// Creates an ODE problem `du/dt = f(u, p, t)`.
    pub fn new(
        rhs: F,
        initial_state: impl Into<Vec<f64>>,
        time_span: (f64, f64),
        parameters: P,
    ) -> Self {
        Self {
            rhs,
            initial_state: initial_state.into(),
            time_span,
            parameters,
            jacobian: None,
            callbacks: Vec::new(),
        }
    }

    /// Supplies an analytic state Jacobian for implicit and Rosenbrock methods.
    ///
    /// The callback receives a row-major `dimension × dimension` output matrix
    /// and must overwrite every entry with `∂fᵢ/∂uⱼ` at `(state, parameters,
    /// time)`. Solvers that do not use a Jacobian ignore this callback.
    pub fn with_jacobian<J>(mut self, jacobian: J) -> Self
    where
        J: Fn(&mut [f64], &[f64], &P, f64) + 'static,
    {
        self.jacobian = Some(Box::new(jacobian));
        self
    }

    /// Adds a callback evaluated after every accepted step (and at the initial state).
    ///
    /// When `condition(state, parameters, time)` is true, `affect` may modify the
    /// state and may request termination. Multiple callbacks run in insertion order.
    pub fn with_discrete_callback<C, A>(mut self, condition: C, affect: A) -> Self
    where
        C: Fn(&[f64], &P, f64) -> bool + 'static,
        A: Fn(&mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.callbacks.push(Callback::Discrete(DiscreteCallback {
            condition: Box::new(condition),
            affect: Box::new(affect),
        }));
        self
    }

    /// Adds a zero-crossing callback that triggers in either direction.
    pub fn with_continuous_callback<C, A>(self, condition: C, affect: A) -> Self
    where
        C: Fn(&[f64], &P, f64) -> f64 + 'static,
        A: Fn(&mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.with_continuous_callback_direction(EventDirection::Any, condition, affect)
    }

    /// Adds a direction-filtered zero-crossing callback.
    ///
    /// A root is localized by bisection over the accepted step's state segment.
    /// The callback receives the localized time and interpolated state.
    pub fn with_continuous_callback_direction<C, A>(
        mut self,
        direction: EventDirection,
        condition: C,
        affect: A,
    ) -> Self
    where
        C: Fn(&[f64], &P, f64) -> f64 + 'static,
        A: Fn(&mut [f64], &P, f64) -> CallbackAction + 'static,
    {
        self.callbacks
            .push(Callback::Continuous(ContinuousCallback {
                condition: Box::new(condition),
                affect: Box::new(affect),
                direction,
            }));
        self
    }

    /// Returns the initial state.
    pub fn initial_state(&self) -> &[f64] {
        &self.initial_state
    }

    /// Returns `(start_time, end_time)`.
    pub fn time_span(&self) -> (f64, f64) {
        self.time_span
    }

    /// Returns whether an analytic state Jacobian was supplied.
    pub fn has_jacobian(&self) -> bool {
        self.jacobian.is_some()
    }

    /// Returns whether event callbacks were supplied.
    pub fn has_callbacks(&self) -> bool {
        !self.callbacks.is_empty()
    }

    /// Returns the problem parameters.
    pub fn parameters(&self) -> &P {
        &self.parameters
    }

    pub(crate) fn evaluate_jacobian(&self, jacobian: &mut [f64], state: &[f64], time: f64) -> bool {
        let Some(function) = &self.jacobian else {
            return false;
        };
        function(jacobian, state, &self.parameters, time);
        true
    }

    pub(crate) fn apply_initial_callbacks(
        &self,
        state: &mut [f64],
        time: f64,
    ) -> Result<CallbackOutcome, SolveError> {
        let mut outcome = CallbackOutcome::default();
        for callback in &self.callbacks {
            let Callback::Discrete(callback) = callback else {
                continue;
            };
            if (callback.condition)(state, &self.parameters, time) {
                outcome.invocations += 1;
                outcome.terminate =
                    (callback.affect)(state, &self.parameters, time) == CallbackAction::Terminate;
                ensure_finite_callback_state(state)?;
                if outcome.terminate {
                    break;
                }
            }
        }
        Ok(outcome)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_step_callbacks(
        &self,
        previous_state: &[f64],
        previous_time: f64,
        state: &mut [f64],
        time: &mut f64,
        state_before_effect: &mut [f64],
        event_tolerance: f64,
        mut interpolator: Option<&mut StepInterpolator<'_>>,
    ) -> Result<CallbackOutcome, SolveError> {
        if self.callbacks.is_empty() {
            return Ok(CallbackOutcome::default());
        }
        let mut outcome = CallbackOutcome::default();
        let mut root = None;

        for (index, callback) in self.callbacks.iter().enumerate() {
            let Callback::Continuous(callback) = callback else {
                continue;
            };
            let before = (callback.condition)(previous_state, &self.parameters, previous_time);
            let after = (callback.condition)(state, &self.parameters, *time);
            if !before.is_finite() || !after.is_finite() {
                return Err(SolveError::NonFiniteCallbackCondition);
            }
            if callback.direction.accepts(before, after) {
                let fraction = locate_root(
                    callback,
                    RootSegment {
                        previous_state,
                        previous_time,
                        state,
                        time: *time,
                    },
                    before,
                    state_before_effect,
                    &self.parameters,
                    event_tolerance,
                    &mut interpolator,
                )?;
                if root.is_none_or(|(_, earliest)| fraction < earliest) {
                    root = Some((index, fraction));
                }
            }
        }

        if let Some((index, fraction)) = root {
            let end_time = *time;
            let root_time = previous_time + fraction * (end_time - previous_time);
            if let Some(interpolator) = interpolator.as_mut() {
                interpolator(root_time, state_before_effect)?;
            } else {
                interpolate(state, previous_state, fraction, state_before_effect);
            }
            state.copy_from_slice(state_before_effect);
            *time = root_time;
            let Callback::Continuous(callback) = &self.callbacks[index] else {
                unreachable!();
            };
            outcome.invocations += 1;
            outcome.terminate =
                (callback.affect)(state, &self.parameters, *time) == CallbackAction::Terminate;
            ensure_finite_callback_state(state)?;
        }

        if !outcome.terminate {
            for callback in &self.callbacks {
                let Callback::Discrete(callback) = callback else {
                    continue;
                };
                if (callback.condition)(state, &self.parameters, *time) {
                    if outcome.invocations == 0 {
                        state_before_effect.copy_from_slice(state);
                    }
                    outcome.invocations += 1;
                    outcome.terminate = (callback.affect)(state, &self.parameters, *time)
                        == CallbackAction::Terminate;
                    ensure_finite_callback_state(state)?;
                    if outcome.terminate {
                        break;
                    }
                }
            }
        }
        Ok(outcome)
    }
}

struct RootSegment<'a> {
    previous_state: &'a [f64],
    previous_time: f64,
    state: &'a [f64],
    time: f64,
}

fn locate_root<P>(
    callback: &ContinuousCallback<P>,
    segment: RootSegment<'_>,
    before: f64,
    interpolation: &mut [f64],
    parameters: &P,
    event_tolerance: f64,
    interpolator: &mut Option<&mut StepInterpolator<'_>>,
) -> Result<f64, SolveError> {
    let mut left = 0.0;
    let mut right = 1.0;
    let mut left_value = before;
    for _ in 0..64 {
        let middle = 0.5 * (left + right);
        let middle_time = segment.previous_time + middle * (segment.time - segment.previous_time);
        if let Some(interpolator) = interpolator.as_mut() {
            interpolator(middle_time, interpolation)?;
        } else {
            interpolate(segment.state, segment.previous_state, middle, interpolation);
        }
        let value = (callback.condition)(interpolation, parameters, middle_time);
        if !value.is_finite() {
            return Err(SolveError::NonFiniteCallbackCondition);
        }
        if value == 0.0 {
            return Ok(middle);
        }
        if left_value.signum() == value.signum() {
            left = middle;
            left_value = value;
        } else {
            right = middle;
        }
        if (right - left) * (segment.time - segment.previous_time).abs() <= event_tolerance {
            break;
        }
    }
    Ok(0.5 * (left + right))
}

fn interpolate(state: &[f64], previous_state: &[f64], fraction: f64, output: &mut [f64]) {
    for ((output, previous), current) in output.iter_mut().zip(previous_state).zip(state) {
        *output = previous + fraction * (current - previous);
    }
}

fn ensure_finite_callback_state(state: &[f64]) -> Result<(), SolveError> {
    state
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteCallbackState)
}

#[cfg(test)]
mod tests {
    use super::{JacobianProvider, MassMatrixOdeProblem, OdeProblem, SplitOdeProblem};

    #[test]
    fn jacobian_provider_reports_analytic_callbacks() {
        let problem = OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
            vec![1.0],
            (0.0, 1.0),
            (),
        )
        .with_jacobian(|jacobian: &mut [f64], _: &[f64], _: &(), _: f64| jacobian[0] = 2.0);
        let provider = JacobianProvider::new(&problem);
        let mut jacobian = [0.0];
        assert!(provider.is_analytic());
        assert!(provider.evaluate(&mut jacobian, &[1.0], 0.0));
        assert_eq!(jacobian, [2.0]);
    }

    #[test]
    fn split_and_mass_representations_validate_dimensions() {
        let split = SplitOdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = -u[0],
            vec![1.0],
            (0.0, 1.0),
            (),
        );
        let mut explicit = [0.0];
        let mut implicit = [0.0];
        split.evaluate_explicit(&mut explicit, &[2.0], 0.0);
        split.evaluate_implicit(&mut implicit, &[2.0], 0.0);
        assert_eq!(explicit, [2.0]);
        assert_eq!(implicit, [-2.0]);

        let mass = MassMatrixOdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
            vec![1.0],
            (0.0, 1.0),
            (),
            vec![2.0],
        )
        .unwrap();
        assert_eq!(mass.mass_matrix(), &[2.0]);
        assert!(
            MassMatrixOdeProblem::new(
                |_: &mut [f64], _: &[f64], _: &(), _: f64| {},
                vec![1.0, 2.0],
                (0.0, 1.0),
                (),
                vec![1.0],
            )
            .is_err()
        );
    }
}
