//! Acceleration functions for partitioned second-order problems.

use ndarray::{ArrayView, ArrayViewMut, Dimension};

use crate::SolveError;

/// A statically dispatched, fallible acceleration function `q'' = f(q', q, p, t)`.
///
/// Existing in-place closures `Fn(&mut [f64], &[f64], &[f64], &P, f64)`
/// implement this automatically. The ndarray out-of-place constructor uses
/// the same interface to report incompatible returned shapes without panicking.
pub trait SecondOrderFunction<P> {
    /// Overwrites every acceleration component at the supplied velocity,
    /// position, parameters, and time.
    ///
    /// All three buffers have the same length. Implementations must not change
    /// parameters affecting the equations during evaluation; use callbacks for
    /// such changes.
    fn evaluate(
        &self,
        acceleration: &mut [f64],
        velocity: &[f64],
        position: &[f64],
        parameters: &P,
        time: f64,
    ) -> Result<(), SolveError>;
}

impl<F, P> SecondOrderFunction<P> for F
where
    F: Fn(&mut [f64], &[f64], &[f64], &P, f64),
{
    fn evaluate(
        &self,
        acceleration: &mut [f64],
        velocity: &[f64],
        position: &[f64],
        parameters: &P,
        time: f64,
    ) -> Result<(), SolveError> {
        self(acceleration, velocity, position, parameters, time);
        Ok(())
    }
}

pub(super) struct ArrayAcceleration<F, D> {
    pub(super) function: F,
    pub(super) shape: D,
}

impl<F, P, D> SecondOrderFunction<P> for ArrayAcceleration<F, D>
where
    D: Dimension,
    F: for<'a, 'b, 'c> Fn(
        ArrayViewMut<'a, f64, D>,
        ArrayView<'b, f64, D>,
        ArrayView<'c, f64, D>,
        &P,
        f64,
    ) -> Result<(), SolveError>,
{
    fn evaluate(
        &self,
        acceleration: &mut [f64],
        velocity: &[f64],
        position: &[f64],
        parameters: &P,
        time: f64,
    ) -> Result<(), SolveError> {
        let velocity = ArrayView::from_shape(self.shape.clone(), velocity)
            .map_err(|_| SolveError::DerivativeShapeMismatch)?;
        let position = ArrayView::from_shape(self.shape.clone(), position)
            .map_err(|_| SolveError::DerivativeShapeMismatch)?;
        let acceleration = ArrayViewMut::from_shape(self.shape.clone(), acceleration)
            .map_err(|_| SolveError::DerivativeShapeMismatch)?;
        (self.function)(acceleration, velocity, position, parameters, time)
    }
}

/// Sequential mutable, fallible acceleration closure adapter.
/// Captures may borrow local data. Independent problems are needed per worker.
pub struct MutableAcceleration<F>(std::cell::RefCell<F>);
impl<F> MutableAcceleration<F> {
    pub(crate) fn new(function: F) -> Self {
        Self(std::cell::RefCell::new(function))
    }
    /// Recovers the closure and its accumulated state.
    pub fn into_inner(self) -> F {
        self.0.into_inner()
    }
}
impl<F, P> SecondOrderFunction<P> for MutableAcceleration<F>
where
    F: FnMut(&mut [f64], &[f64], &[f64], &P, f64) -> Result<(), SolveError>,
{
    fn evaluate(
        &self,
        acceleration: &mut [f64],
        velocity: &[f64],
        position: &[f64],
        parameters: &P,
        time: f64,
    ) -> Result<(), SolveError> {
        (self.0.borrow_mut())(acceleration, velocity, position, parameters, time)
    }
}
