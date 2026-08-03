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
}

type JacobianFunction<P> = dyn Fn(&mut [f64], &[f64], &P, f64);

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
}
