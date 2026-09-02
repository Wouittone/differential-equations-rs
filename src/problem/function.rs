use ndarray::{Array, ArrayView, Dimension};

use crate::SolveError;

/// A statically dispatched, fallible ODE right-hand side.
///
/// Existing in-place closures `Fn(&mut [f64], &[f64], &P, f64)` implement this
/// automatically. Ndarray out-of-place constructors use this same interface
/// to validate returned shapes before copying derivatives into solver storage.
/// Custom implementations may return a [`SolveError`] without panicking.
pub trait OdeFunction<P> {
    /// Overwrites the entire derivative buffer at the supplied state and time.
    ///
    /// The derivative and state buffers have the same dimension. Implementations
    /// must not mutate parameters that affect the equations during evaluation;
    /// callbacks provide the supported mechanism for those changes.
    fn evaluate(
        &self,
        derivative: &mut [f64],
        state: &[f64],
        parameters: &P,
        time: f64,
    ) -> Result<(), SolveError>;
}

impl<F, P> OdeFunction<P> for F
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    fn evaluate(
        &self,
        derivative: &mut [f64],
        state: &[f64],
        parameters: &P,
        time: f64,
    ) -> Result<(), SolveError> {
        self(derivative, state, parameters, time);
        Ok(())
    }
}

pub(super) struct ArrayFunction<F, D> {
    pub(super) rhs: F,
    pub(super) shape: D,
}

impl<F, P, D> OdeFunction<P> for ArrayFunction<F, D>
where
    D: Dimension,
    F: for<'a> Fn(ArrayView<'a, f64, D>, &P, f64) -> Array<f64, D>,
{
    fn evaluate(
        &self,
        derivative: &mut [f64],
        state: &[f64],
        parameters: &P,
        time: f64,
    ) -> Result<(), SolveError> {
        let state = ArrayView::from_shape(self.shape.clone(), state)
            .map_err(|_| SolveError::DerivativeShapeMismatch)?;
        let result = (self.rhs)(state, parameters, time);
        if result.raw_dim() != self.shape || result.len() != derivative.len() {
            return Err(SolveError::DerivativeShapeMismatch);
        }
        // Iteration follows logical axis order even for a non-contiguous or
        // column-major result. Solvers always use contiguous row-major storage.
        for (output, value) in derivative.iter_mut().zip(result.iter()) {
            *output = *value;
        }
        Ok(())
    }
}
