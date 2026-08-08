use crate::SolveError;

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LinearError {
    EmptyDimension,
    DimensionOverflow { rows: usize, columns: usize },
    LengthMismatch { expected: usize, actual: usize },
    NonFiniteCoefficient,
    Singular,
    Unfactorized,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StateLayout {
    dimension: usize,
}

#[allow(dead_code)]
impl StateLayout {
    pub(crate) fn new(dimension: usize) -> Result<Self, LinearError> {
        if dimension == 0 {
            return Err(LinearError::EmptyDimension);
        }
        Ok(Self { dimension })
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

    pub(crate) fn state<'a>(self, data: &'a [f64]) -> Result<StateRef<'a>, LinearError> {
        if data.len() != self.dimension {
            return Err(LinearError::LengthMismatch {
                expected: self.dimension,
                actual: data.len(),
            });
        }
        Ok(StateRef { layout: self, data })
    }

    pub(crate) fn state_mut<'a>(self, data: &'a mut [f64]) -> Result<StateMut<'a>, LinearError> {
        if data.len() != self.dimension {
            return Err(LinearError::LengthMismatch {
                expected: self.dimension,
                actual: data.len(),
            });
        }
        Ok(StateMut { layout: self, data })
    }

    pub(crate) fn matrix<'a>(self, data: &'a [f64]) -> Result<DenseMatrixRef<'a>, LinearError> {
        let expected = self.matrix_len()?;
        if data.len() != expected {
            return Err(LinearError::LengthMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(DenseMatrixRef {
            rows: self.dimension,
            columns: self.dimension,
            data,
        })
    }

    pub(crate) fn matrix_mut<'a>(
        self,
        data: &'a mut [f64],
    ) -> Result<DenseMatrixMut<'a>, LinearError> {
        let expected = self.matrix_len()?;
        if data.len() != expected {
            return Err(LinearError::LengthMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(DenseMatrixMut {
            rows: self.dimension,
            columns: self.dimension,
            data,
        })
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct StateRef<'a> {
    layout: StateLayout,
    data: &'a [f64],
}

#[allow(dead_code)]
impl<'a> StateRef<'a> {
    pub(crate) fn as_slice(self) -> &'a [f64] {
        self.data
    }

    pub(crate) fn dimension(self) -> usize {
        self.layout.dimension()
    }
}

#[allow(dead_code)]
pub(crate) struct StateMut<'a> {
    layout: StateLayout,
    data: &'a mut [f64],
}

#[allow(dead_code)]
impl<'a> StateMut<'a> {
    pub(crate) fn as_mut_slice(&mut self) -> &mut [f64] {
        self.data
    }

    pub(crate) fn dimension(&self) -> usize {
        self.layout.dimension()
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct DenseMatrixRef<'a> {
    rows: usize,
    columns: usize,
    data: &'a [f64],
}

#[allow(dead_code)]
impl<'a> DenseMatrixRef<'a> {
    pub(crate) fn rows(self) -> usize {
        self.rows
    }

    pub(crate) fn columns(self) -> usize {
        self.columns
    }

    pub(crate) fn as_slice(self) -> &'a [f64] {
        self.data
    }
}

#[allow(dead_code)]
pub(crate) struct DenseMatrixMut<'a> {
    rows: usize,
    columns: usize,
    data: &'a mut [f64],
}

#[allow(dead_code)]
impl<'a> DenseMatrixMut<'a> {
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn columns(&self) -> usize {
        self.columns
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [f64] {
        self.data
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DenseLu {
    dimension: usize,
    factors: Vec<f64>,
    pivots: Vec<usize>,
    revision: u64,
}

#[allow(dead_code)]
pub(crate) trait LinearOperator {
    fn dimension(&self) -> usize;
    fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), LinearError>;
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct IdentityOperator {
    layout: StateLayout,
}

#[allow(dead_code)]
impl IdentityOperator {
    pub(crate) fn new(layout: StateLayout) -> Self {
        Self { layout }
    }
}

#[allow(dead_code)]
impl LinearOperator for IdentityOperator {
    fn dimension(&self) -> usize {
        self.layout.dimension()
    }

    fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), LinearError> {
        self.layout.state(input)?;
        self.layout
            .state_mut(output)?
            .as_mut_slice()
            .copy_from_slice(input);
        Ok(())
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub(crate) struct DenseOperator {
    layout: StateLayout,
    matrix: Vec<f64>,
}

#[allow(dead_code)]
impl DenseOperator {
    pub(crate) fn new(layout: StateLayout, matrix: &[f64]) -> Result<Self, LinearError> {
        let view = layout.matrix(matrix)?;
        if view.as_slice().iter().any(|value| !value.is_finite()) {
            return Err(LinearError::NonFiniteCoefficient);
        }
        Ok(Self {
            layout,
            matrix: matrix.to_vec(),
        })
    }
}

#[allow(dead_code)]
impl LinearOperator for DenseOperator {
    fn dimension(&self) -> usize {
        self.layout.dimension()
    }

    fn apply(&self, input: &[f64], output: &mut [f64]) -> Result<(), LinearError> {
        self.layout.state(input)?;
        self.layout.state_mut(output)?;
        for (row, destination) in output.iter_mut().enumerate() {
            *destination = self.matrix[row * self.dimension()..(row + 1) * self.dimension()]
                .iter()
                .zip(input)
                .map(|(coefficient, value)| coefficient * value)
                .sum();
        }
        Ok(())
    }
}

#[allow(dead_code)]
impl DenseLu {
    pub(crate) fn factorize(
        layout: StateLayout,
        matrix: &[f64],
        revision: u64,
    ) -> Result<Self, LinearError> {
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
            revision,
        })
    }

    pub(crate) fn dimension(&self) -> usize {
        self.dimension
    }

    pub(crate) fn revision(&self) -> u64 {
        self.revision
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
        if !pivot_magnitude.is_finite() || pivot_magnitude <= f64::EPSILON {
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
    use super::{
        DenseLu, DenseOperator, IdentityOperator, LinearError, LinearOperator, StateLayout,
        factorize, solve_factorized,
    };

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
    fn checked_layout_rejects_wrong_lengths() {
        let layout = StateLayout::new(2).unwrap();
        assert_eq!(
            layout.state(&[1.0]).unwrap_err(),
            LinearError::LengthMismatch {
                expected: 2,
                actual: 1,
            }
        );
        assert_eq!(
            layout.matrix(&[1.0]).unwrap_err(),
            LinearError::LengthMismatch {
                expected: 4,
                actual: 1,
            }
        );
        assert_eq!(StateLayout::new(0), Err(LinearError::EmptyDimension));
    }

    #[test]
    fn dense_lu_tracks_revision_and_solves() {
        let layout = StateLayout::new(2).unwrap();
        let lu = DenseLu::factorize(layout, &[0.0, 2.0, 1.0, 1.0], 7).unwrap();
        let mut rhs = [4.0, 3.0];
        lu.solve(&mut rhs).unwrap();
        assert_eq!(lu.dimension(), 2);
        assert_eq!(lu.revision(), 7);
        assert!((rhs[0] - 1.0).abs() < 1.0e-14);
        assert!((rhs[1] - 2.0).abs() < 1.0e-14);
    }

    #[test]
    fn dense_lu_reports_nonfinite_and_singular_inputs() {
        let layout = StateLayout::new(2).unwrap();
        assert_eq!(
            DenseLu::factorize(layout, &[f64::NAN, 0.0, 0.0, 1.0], 0),
            Err(LinearError::NonFiniteCoefficient)
        );
        assert_eq!(
            DenseLu::factorize(layout, &[1.0, 2.0, 2.0, 4.0], 0),
            Err(LinearError::Singular)
        );
    }

    #[test]
    fn operators_apply_with_checked_dimensions() {
        let layout = StateLayout::new(2).unwrap();
        let identity = IdentityOperator::new(layout);
        let mut output = [0.0; 2];
        identity.apply(&[2.0, -1.0], &mut output).unwrap();
        assert_eq!(output, [2.0, -1.0]);
        assert_eq!(
            identity.apply(&[1.0], &mut output),
            Err(LinearError::LengthMismatch {
                expected: 2,
                actual: 1,
            })
        );

        let dense = DenseOperator::new(layout, &[2.0, 1.0, 0.0, 3.0]).unwrap();
        dense.apply(&[2.0, -1.0], &mut output).unwrap();
        assert_eq!(output, [3.0, -3.0]);
    }
}
