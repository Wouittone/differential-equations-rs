use crate::solvers::exponential::ExponentialAlgorithm;
use crate::{ConfigurationError, OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions};

/// A semilinear initial-value problem `u' = A u + g(u, p, t)`.
///
/// `A` is a constant dense row-major `dimension × dimension` operator.  The
/// exponential methods currently use a dense `f64` matrix-function backend;
/// matrix-free and Krylov actions are intentionally left for a future backend.
pub struct SemilinearOdeProblem<G, P> {
    nonlinear: G,
    linear_operator: Vec<f64>,
    initial_state: Vec<f64>,
    time_span: (f64, f64),
    parameters: P,
}

impl<G, P> SemilinearOdeProblem<G, P> {
    /// Constructs a checked semilinear problem.
    pub fn new(
        linear_operator: impl Into<Vec<f64>>,
        nonlinear: G,
        initial_state: impl Into<Vec<f64>>,
        time_span: (f64, f64),
        parameters: P,
    ) -> Result<Self, ConfigurationError> {
        let initial_state = initial_state.into();
        if initial_state.is_empty() {
            return Err(ConfigurationError::EmptyData {
                context: "semilinear ODE state",
            });
        }
        let linear_operator = linear_operator.into();
        let expected = initial_state.len().checked_mul(initial_state.len()).ok_or(
            ConfigurationError::DimensionOverflow {
                context: "semilinear ODE",
            },
        )?;
        if linear_operator.len() != expected {
            return Err(ConfigurationError::DimensionMismatch {
                context: "semilinear linear operator",
            });
        }
        if linear_operator.iter().any(|value| !value.is_finite()) {
            return Err(ConfigurationError::NonFiniteData {
                context: "semilinear linear operator",
            });
        }
        Ok(Self {
            nonlinear,
            linear_operator,
            initial_state,
            time_span,
            parameters,
        })
    }

    pub fn linear_operator(&self) -> &[f64] {
        &self.linear_operator
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

    pub fn evaluate_nonlinear(&self, output: &mut [f64], state: &[f64], time: f64)
    where
        G: Fn(&mut [f64], &[f64], &P, f64),
    {
        (self.nonlinear)(output, state, &self.parameters, time);
    }

    pub fn evaluate(&self, output: &mut [f64], state: &[f64], time: f64)
    where
        G: Fn(&mut [f64], &[f64], &P, f64),
    {
        (self.nonlinear)(output, state, &self.parameters, time);
        let dimension = state.len();
        for (row, output) in output.iter_mut().enumerate() {
            for (column, state) in state.iter().enumerate() {
                *output += self.linear_operator[row * dimension + column] * state;
            }
        }
    }
}

/// Solves a semilinear problem while preserving its exact linear/nonlinear split.
pub fn solve_exponential<G, P, A>(
    problem: &SemilinearOdeProblem<G, P>,
    algorithm: A,
    options: &SolveOptions,
) -> Result<Solution, SolveError>
where
    G: Fn(&mut [f64], &[f64], &P, f64),
    A: ExponentialAlgorithm,
{
    let total = |output: &mut [f64], state: &[f64], _: &(), time: f64| {
        problem.evaluate(output, state, time);
    };
    let operator = problem.linear_operator.clone();
    let ode = OdeProblem::new(total, problem.initial_state.clone(), problem.time_span, ())
        .with_jacobian(move |jacobian, _, _, _| jacobian.copy_from_slice(&operator));
    OdeAlgorithm::solve(&algorithm, &ode, options)
}
