use super::OdeProblem;
use crate::OdeFunction;
use ndarray::{Array, ArrayView, ArrayViewMut, Dimension};

/// A type-state builder for an ndarray-backed [`OdeProblem`].
///
/// Start with [`OdeProblem::builder`], then name the initial state, time span,
/// and parameters in any order. Once the required state and time span are
/// present, finish with [`Self::build_with_in_place_rhs`] or
/// [`Self::build_with_out_of_place_rhs`]. Unit parameters are used by default.
///
/// The generic parameters are implementation details that track which values
/// have already been supplied; callers normally do not name this type.
#[derive(Clone, Copy, Debug, Default)]
pub struct OdeProblemBuilder<S = (), T = (), P = ()> {
    initial_state: S,
    time_span: T,
    parameters: P,
}

impl OdeProblemBuilder {
    pub(crate) const fn new() -> Self {
        Self {
            initial_state: (),
            time_span: (),
            parameters: (),
        }
    }
}

impl<S, T, P> OdeProblemBuilder<S, T, P> {
    /// Sets the scalar, vector, or matrix initial state.
    pub fn initial_state<D>(
        self,
        initial_state: Array<f64, D>,
    ) -> OdeProblemBuilder<Array<f64, D>, T, P>
    where
        D: Dimension,
    {
        OdeProblemBuilder {
            initial_state,
            time_span: self.time_span,
            parameters: self.parameters,
        }
    }

    /// Sets the inclusive integration interval `(start_time, end_time)`.
    pub fn time_span(self, time_span: (f64, f64)) -> OdeProblemBuilder<S, (f64, f64), P> {
        OdeProblemBuilder {
            initial_state: self.initial_state,
            time_span,
            parameters: self.parameters,
        }
    }

    /// Sets the value passed to every right-hand-side evaluation.
    pub fn parameters<Q>(self, parameters: Q) -> OdeProblemBuilder<S, T, Q> {
        OdeProblemBuilder {
            initial_state: self.initial_state,
            time_span: self.time_span,
            parameters,
        }
    }
}

impl<D, P> OdeProblemBuilder<Array<f64, D>, (f64, f64), P>
where
    D: Dimension,
{
    /// Builds a problem whose right-hand side writes into a shaped derivative view.
    ///
    /// The derivative and state views have the same dimensionality as the
    /// configured initial state. This is the allocation-free array evaluation
    /// path used by [`OdeProblem::from_array`].
    #[allow(clippy::type_complexity)] // Preserve the monomorphized ndarray adapter.
    pub fn build_with_in_place_rhs<F>(
        self,
        rhs: F,
    ) -> OdeProblem<impl Fn(&mut [f64], &[f64], &P, f64), P>
    where
        F: for<'a, 'b> Fn(ArrayViewMut<'a, f64, D>, ArrayView<'b, f64, D>, &P, f64),
    {
        OdeProblem::from_array(rhs, self.initial_state, self.time_span, self.parameters)
    }

    /// Builds a problem whose right-hand side returns a shaped derivative.
    ///
    /// The returned array must match the initial state's shape. Returning an
    /// owned array can allocate on every evaluation; prefer
    /// [`Self::build_with_in_place_rhs`] when allocations matter.
    pub fn build_with_out_of_place_rhs<F>(self, rhs: F) -> OdeProblem<impl OdeFunction<P>, P>
    where
        F: for<'a> Fn(ArrayView<'a, f64, D>, &P, f64) -> Array<f64, D>,
    {
        OdeProblem::from_array_out_of_place(
            rhs,
            self.initial_state,
            self.time_span,
            self.parameters,
        )
    }
}
