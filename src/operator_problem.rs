/// A vector problem driven by a dense, possibly state- and time-dependent
/// linear operator: `u' = A(u, p, t) u`.
///
/// The operator callback overwrites a row-major `dimension × dimension`
/// buffer. This mirrors SciML's updateable `MatrixOperator` contract while
/// making dimensions explicit in Rust.
pub struct LinearOperatorProblem<O, P> {
    pub(crate) operator: O,
    pub(crate) initial_state: Vec<f64>,
    pub(crate) time_span: (f64, f64),
    pub(crate) parameters: P,
}

impl<O, P> LinearOperatorProblem<O, P> {
    pub fn new(
        operator: O,
        initial_state: impl Into<Vec<f64>>,
        time_span: (f64, f64),
        parameters: P,
    ) -> Result<Self, &'static str> {
        let initial_state = initial_state.into();
        if initial_state.is_empty() {
            return Err("linear operator state must be non-empty");
        }
        Ok(Self {
            operator,
            initial_state,
            time_span,
            parameters,
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

    pub fn dimension(&self) -> usize {
        self.initial_state.len()
    }

    pub fn evaluate_operator(&self, output: &mut [f64], state: &[f64], time: f64)
    where
        O: Fn(&mut [f64], &[f64], &P, f64),
    {
        (self.operator)(output, state, &self.parameters, time);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LieRepresentation {
    Vector,
    Matrix,
}

/// A homogeneous-space or matrix Lie-group problem driven by a dense Lie
/// algebra generator.
///
/// Use [`vector`](Self::vector) for actions `u <- exp(Ω)u`, and
/// [`matrix`](Self::matrix) for conjugation actions on a square matrix state.
pub struct LieGroupProblem<O, P> {
    pub(crate) operator: O,
    pub(crate) initial_state: Vec<f64>,
    pub(crate) time_span: (f64, f64),
    pub(crate) parameters: P,
    pub(crate) group_dimension: usize,
    pub(crate) representation: LieRepresentation,
}

impl<O, P> LieGroupProblem<O, P> {
    pub fn vector(
        operator: O,
        initial_state: impl Into<Vec<f64>>,
        time_span: (f64, f64),
        parameters: P,
    ) -> Result<Self, &'static str> {
        let initial_state = initial_state.into();
        if initial_state.is_empty() {
            return Err("Lie-group vector state must be non-empty");
        }
        let group_dimension = initial_state.len();
        Ok(Self {
            operator,
            initial_state,
            time_span,
            parameters,
            group_dimension,
            representation: LieRepresentation::Vector,
        })
    }

    pub fn matrix(
        operator: O,
        initial_matrix: impl Into<Vec<f64>>,
        dimension: usize,
        time_span: (f64, f64),
        parameters: P,
    ) -> Result<Self, &'static str> {
        let initial_state = initial_matrix.into();
        let expected = dimension
            .checked_mul(dimension)
            .ok_or("Lie-group matrix dimension overflow")?;
        if dimension == 0 || initial_state.len() != expected {
            return Err("Lie-group matrix state must be square and non-empty");
        }
        Ok(Self {
            operator,
            initial_state,
            time_span,
            parameters,
            group_dimension: dimension,
            representation: LieRepresentation::Matrix,
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

    pub fn group_dimension(&self) -> usize {
        self.group_dimension
    }

    pub fn is_matrix_state(&self) -> bool {
        self.representation == LieRepresentation::Matrix
    }

    pub fn evaluate_operator(&self, output: &mut [f64], state: &[f64], time: f64)
    where
        O: Fn(&mut [f64], &[f64], &P, f64),
    {
        (self.operator)(output, state, &self.parameters, time);
    }
}
