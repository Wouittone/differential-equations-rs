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
        }
    }

    /// Returns the initial state.
    pub fn initial_state(&self) -> &[f64] {
        &self.initial_state
    }

    /// Returns `(start_time, end_time)`.
    pub fn time_span(&self) -> (f64, f64) {
        self.time_span
    }

    /// Returns the problem parameters.
    pub fn parameters(&self) -> &P {
        &self.parameters
    }
}
