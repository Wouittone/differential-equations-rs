use crate::SolveError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LinearError {
    DimensionOverflow { rows: usize, columns: usize },
    LengthMismatch { expected: usize, actual: usize },
    NonFiniteCoefficient,
    Singular,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StateLayout {
    dimension: usize,
}

impl StateLayout {
    /// Constructs a layout after the public solver entry point has validated
    /// that the state is non-empty.
    pub(crate) fn for_validated_state(dimension: usize) -> Self {
        debug_assert!(dimension > 0, "validated states are non-empty");
        Self { dimension }
    }

    pub(crate) fn dimension(self) -> usize {
        self.dimension
    }

    pub(crate) fn matrix_len(self) -> Result<usize, LinearError> {
        self.dimension
            .checked_mul(self.dimension)
            .ok_or(LinearError::DimensionOverflow {
                rows: self.dimension,
                columns: self.dimension,
            })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DenseLu {
    dimension: usize,
    factors: Vec<f64>,
    pivots: Vec<usize>,
}

impl DenseLu {
    pub(crate) fn factorize(layout: StateLayout, matrix: &[f64]) -> Result<Self, LinearError> {
        let expected = layout.matrix_len()?;
        if matrix.len() != expected {
            return Err(LinearError::LengthMismatch {
                expected,
                actual: matrix.len(),
            });
        }
        if matrix.iter().any(|value| !value.is_finite()) {
            return Err(LinearError::NonFiniteCoefficient);
        }
        let mut factors = matrix.to_vec();
        let mut pivots = vec![0; layout.dimension()];
        factorize(&mut factors, &mut pivots, layout.dimension())
            .map_err(|_| LinearError::Singular)?;
        Ok(Self {
            dimension: layout.dimension(),
            factors,
            pivots,
        })
    }

    pub(crate) fn solve(&self, right_hand_side: &mut [f64]) -> Result<(), LinearError> {
        if right_hand_side.len() != self.dimension {
            return Err(LinearError::LengthMismatch {
                expected: self.dimension,
                actual: right_hand_side.len(),
            });
        }
        solve_factorized(&self.factors, &self.pivots, right_hand_side, self.dimension);
        Ok(())
    }
}

pub(crate) fn factorize(
    matrix: &mut [f64],
    pivots: &mut [usize],
    dimension: usize,
) -> Result<(), SolveError> {
    let matrix_scale = matrix
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if !matrix_scale.is_finite() || matrix_scale == 0.0 {
        return Err(SolveError::SingularLinearSystem);
    }
    let singularity_threshold = f64::EPSILON * matrix_scale * dimension.max(1) as f64;
    for pivot_column in 0..dimension {
        let mut pivot_row = pivot_column;
        let mut pivot_magnitude = matrix[pivot_column * dimension + pivot_column].abs();
        for row in (pivot_column + 1)..dimension {
            let magnitude = matrix[row * dimension + pivot_column].abs();
            if magnitude > pivot_magnitude {
                pivot_magnitude = magnitude;
                pivot_row = row;
            }
        }
        if !pivot_magnitude.is_finite() || pivot_magnitude <= singularity_threshold {
            return Err(SolveError::SingularLinearSystem);
        }
        pivots[pivot_column] = pivot_row;
        if pivot_row != pivot_column {
            for column in 0..dimension {
                matrix.swap(
                    pivot_column * dimension + column,
                    pivot_row * dimension + column,
                );
            }
        }
        let pivot = matrix[pivot_column * dimension + pivot_column];
        for row in (pivot_column + 1)..dimension {
            let factor = matrix[row * dimension + pivot_column] / pivot;
            matrix[row * dimension + pivot_column] = factor;
            for column in (pivot_column + 1)..dimension {
                matrix[row * dimension + column] -=
                    factor * matrix[pivot_column * dimension + column];
            }
        }
    }
    Ok(())
}

pub(crate) fn solve_factorized(
    factorization: &[f64],
    pivots: &[usize],
    right_hand_side: &mut [f64],
    dimension: usize,
) {
    for (row, &pivot) in pivots.iter().enumerate() {
        if pivot != row {
            right_hand_side.swap(row, pivot);
        }
    }
    for row in 0..dimension {
        for column in 0..row {
            right_hand_side[row] -=
                factorization[row * dimension + column] * right_hand_side[column];
        }
    }
    for row in (0..dimension).rev() {
        for column in (row + 1)..dimension {
            right_hand_side[row] -=
                factorization[row * dimension + column] * right_hand_side[column];
        }
        right_hand_side[row] /= factorization[row * dimension + row];
    }
}

#[cfg(test)]
mod tests {
    use super::{DenseLu, LinearError, StateLayout, factorize, solve_factorized};

    #[test]
    fn pivoted_factorization_handles_row_exchange() {
        let mut matrix = vec![0.0, 2.0, 1.0, 1.0];
        let mut pivots = vec![0; 2];
        let mut right_hand_side = vec![4.0, 3.0];

        factorize(&mut matrix, &mut pivots, 2).unwrap();
        solve_factorized(&matrix, &pivots, &mut right_hand_side, 2);

        assert!((right_hand_side[0] - 1.0).abs() < 1.0e-14);
        assert!((right_hand_side[1] - 2.0).abs() < 1.0e-14);
    }

    #[test]
    fn factorization_is_invariant_under_uniform_matrix_scaling() {
        for scale in [1.0e-30, 1.0, 1.0e30] {
            let mut matrix = vec![2.0 * scale, scale, scale, 3.0 * scale];
            let mut pivots = vec![0; 2];
            factorize(&mut matrix, &mut pivots, 2).unwrap();

            let mut right_hand_side = vec![3.0 * scale, 4.0 * scale];
            solve_factorized(&matrix, &pivots, &mut right_hand_side, 2);
            assert!((right_hand_side[0] - 1.0).abs() < 1.0e-14);
            assert!((right_hand_side[1] - 1.0).abs() < 1.0e-14);
        }
    }

    #[test]
    fn dense_lu_solves() {
        let layout = StateLayout::for_validated_state(2);
        let lu = DenseLu::factorize(layout, &[0.0, 2.0, 1.0, 1.0]).unwrap();
        let mut rhs = [4.0, 3.0];
        lu.solve(&mut rhs).unwrap();
        assert!((rhs[0] - 1.0).abs() < 1.0e-14);
        assert!((rhs[1] - 2.0).abs() < 1.0e-14);
    }

    #[test]
    fn dense_lu_reports_nonfinite_and_singular_inputs() {
        let layout = StateLayout::for_validated_state(2);
        assert_eq!(
            DenseLu::factorize(layout, &[f64::NAN, 0.0, 0.0, 1.0]),
            Err(LinearError::NonFiniteCoefficient)
        );
        assert_eq!(
            DenseLu::factorize(layout, &[1.0, 2.0, 2.0, 4.0]),
            Err(LinearError::Singular)
        );
    }
}
