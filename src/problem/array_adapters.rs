use super::function::ArrayFunction;
use super::{OdeProblem, SplitOdeProblem};
use crate::OdeFunction;
use ndarray::{Array, ArrayView, ArrayViewMut, Dimension};

#[allow(dead_code)]
impl SplitOdeProblem<(), (), ()> {
    /// Constructs a split problem whose components return ndarray derivatives.
    ///
    /// Each returned array must have exactly the initial state's shape.
    /// Invalid shapes produce [`crate::SolveError::DerivativeShapeMismatch`]. Returning
    /// owned arrays may allocate on every evaluation; use [`Self::from_array`]
    /// when in-place evaluation is more suitable.
    pub fn from_array_out_of_place<FE, FI, P, D>(
        explicit: FE,
        implicit: FI,
        initial_state: Array<f64, D>,
        time_span: (f64, f64),
        parameters: P,
    ) -> SplitOdeProblem<impl OdeFunction<P>, impl OdeFunction<P>, P>
    where
        D: Dimension,
        FE: for<'a> Fn(ArrayView<'a, f64, D>, &P, f64) -> Array<f64, D>,
        FI: for<'a> Fn(ArrayView<'a, f64, D>, &P, f64) -> Array<f64, D>,
    {
        let shape = initial_state.raw_dim();
        let mut problem = SplitOdeProblem::new(
            ArrayFunction {
                rhs: explicit,
                shape: shape.clone(),
            },
            ArrayFunction {
                rhs: implicit,
                shape: shape.clone(),
            },
            initial_state.iter().copied().collect::<Vec<_>>(),
            time_span,
            parameters,
        );
        problem.state_shape = shape.into_dyn();
        problem
    }

    /// Constructs a split problem from ndarray-shaped functions and state.
    ///
    /// Both functions receive mutable/read-only ndarray views with the
    /// same dimensionality as `initial_state`. The generated adapters are
    /// monomorphized and expose contiguous slices only to numerical kernels.
    #[allow(clippy::type_complexity)] // Preserve monomorphized RHS adapters instead of boxing.
    pub fn from_array<FE, FI, P, D>(
        explicit: FE,
        implicit: FI,
        initial_state: Array<f64, D>,
        time_span: (f64, f64),
        parameters: P,
    ) -> SplitOdeProblem<
        impl Fn(&mut [f64], &[f64], &P, f64),
        impl Fn(&mut [f64], &[f64], &P, f64),
        P,
    >
    where
        D: Dimension,
        FE: for<'a, 'b> Fn(ArrayViewMut<'a, f64, D>, ArrayView<'b, f64, D>, &P, f64),
        FI: for<'a, 'b> Fn(ArrayViewMut<'a, f64, D>, ArrayView<'b, f64, D>, &P, f64),
    {
        let rhs_shape = initial_state.raw_dim();
        let state_shape = rhs_shape.clone().into_dyn();
        let initial_state = initial_state.iter().copied().collect();
        let explicit_shape = rhs_shape.clone();
        let implicit_shape = rhs_shape;
        let explicit = move |derivative: &mut [f64], state: &[f64], parameters: &P, time| {
            let derivative = ArrayViewMut::from_shape(explicit_shape.clone(), derivative)
                .expect("split derivative shape must match its contiguous storage");
            let state = ArrayView::from_shape(explicit_shape.clone(), state)
                .expect("split state shape must match its contiguous storage");
            explicit(derivative, state, parameters, time);
        };
        let implicit = move |derivative: &mut [f64], state: &[f64], parameters: &P, time| {
            let derivative = ArrayViewMut::from_shape(implicit_shape.clone(), derivative)
                .expect("split derivative shape must match its contiguous storage");
            let state = ArrayView::from_shape(implicit_shape.clone(), state)
                .expect("split state shape must match its contiguous storage");
            implicit(derivative, state, parameters, time);
        };
        SplitOdeProblem {
            explicit,
            implicit,
            initial_state,
            state_shape,
            time_span,
            parameters,
            implicit_jacobian: None,
            callbacks: Vec::new(),
            initializers: Vec::new(),
            finalizers: Vec::new(),
            step_guards: Vec::new(),
            predictive_domains: Vec::new(),
        }
    }
}

impl OdeProblem<(), ()> {
    /// Constructs an ODE problem whose function returns an ndarray derivative.
    ///
    /// Scalar (`arr0`), vector, and matrix states use one API. The returned
    /// derivative must have exactly the initial state's shape, otherwise the
    /// solve returns [`crate::SolveError::DerivativeShapeMismatch`]. Array layout need
    /// not be contiguous. Returning owned arrays may allocate each evaluation;
    /// [`Self::from_array`] remains the in-place alternative.
    ///
    /// ```
    /// use differential_equations::ndarray::{arr0, ArrayView0};
    /// use differential_equations::solvers::explicit::Tsit5;
    /// use differential_equations::{OdeProblem, SolveOptions, solve};
    /// let problem = OdeProblem::from_array_out_of_place(
    ///     |u: ArrayView0<'_, f64>, _: &(), _| -&u,
    ///     arr0(1.0), (0.0, 1.0), (),
    /// );
    /// let solution = solve(&problem, Tsit5, &SolveOptions::default())?;
    /// assert!(solution.state_shape().is_empty());
    /// assert!((solution.last_state()[0] - (-1.0_f64).exp()).abs() < 1e-3);
    /// # Ok::<(), differential_equations::SolveError>(())
    /// ```
    pub fn from_array_out_of_place<F, P, D>(
        rhs: F,
        initial_state: Array<f64, D>,
        time_span: (f64, f64),
        parameters: P,
    ) -> OdeProblem<impl OdeFunction<P>, P>
    where
        D: Dimension,
        F: for<'a> Fn(ArrayView<'a, f64, D>, &P, f64) -> Array<f64, D>,
    {
        let shape = initial_state.raw_dim();
        let mut problem = OdeProblem::new(
            ArrayFunction {
                rhs,
                shape: shape.clone(),
            },
            initial_state.iter().copied().collect::<Vec<_>>(),
            time_span,
            parameters,
        );
        problem.state_shape = shape.into_dyn();
        problem
    }

    /// Creates an ODE problem from an ndarray-shaped function and state.
    ///
    /// The function receives mutable/read-only ndarray views with the
    /// same dimensionality as `initial_state`. The generated adapter is
    /// monomorphized and exposes contiguous slices only to numerical kernels.
    #[allow(clippy::type_complexity)] // Preserve a monomorphized RHS adapter instead of boxing.
    pub fn from_array<F, P, D>(
        rhs: F,
        initial_state: Array<f64, D>,
        time_span: (f64, f64),
        parameters: P,
    ) -> OdeProblem<impl Fn(&mut [f64], &[f64], &P, f64), P>
    where
        D: Dimension,
        F: for<'a, 'b> Fn(ArrayViewMut<'a, f64, D>, ArrayView<'b, f64, D>, &P, f64),
    {
        let rhs_shape = initial_state.raw_dim();
        let state_shape = rhs_shape.clone().into_dyn();
        let initial_state = initial_state.iter().copied().collect();
        let rhs = move |derivative: &mut [f64], state: &[f64], parameters: &P, time| {
            let derivative = ArrayViewMut::from_shape(rhs_shape.clone(), derivative)
                .expect("derivative shape must match its contiguous storage");
            let state = ArrayView::from_shape(rhs_shape.clone(), state)
                .expect("state shape must match its contiguous storage");
            rhs(derivative, state, parameters, time);
        };
        OdeProblem {
            rhs,
            initial_state,
            state_shape,
            time_span,
            parameters,
            jacobian: None,
            callbacks: Vec::new(),
            initializers: Vec::new(),
            finalizers: Vec::new(),
            step_guards: Vec::new(),
            predictive_domains: Vec::new(),
        }
    }
}
