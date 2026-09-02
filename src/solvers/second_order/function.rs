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
