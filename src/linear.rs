use crate::SolveError;

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
    use super::{factorize, solve_factorized};

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
}
