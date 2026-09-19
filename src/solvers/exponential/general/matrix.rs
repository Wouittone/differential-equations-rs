pub(crate) fn matrix_exp(matrix: &[f64], n: usize) -> Vec<f64> {
    let norm = (0..n)
        .map(|row| {
            (0..n)
                .map(|column| matrix[row * n + column].abs())
                .sum::<f64>()
        })
        .fold(0.0, f64::max);
    let squarings = if norm <= 0.5 {
        0
    } else {
        (norm / 0.5).log2().ceil() as u32
    };
    let divisor = 2.0_f64.powi(squarings as i32);
    let scaled: Vec<_> = matrix.iter().map(|value| value / divisor).collect();
    let mut result = identity(n);
    let mut term = identity(n);
    for k in 1..=128 {
        term = mat_mul(&term, &scaled, n);
        for value in &mut term {
            *value /= k as f64;
        }
        let term_norm = term
            .iter()
            .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
        for (result, value) in result.iter_mut().zip(&term) {
            *result += value;
        }
        if term_norm
            <= f64::EPSILON
                * result
                    .iter()
                    .fold(1.0_f64, |maximum, value| maximum.max(value.abs()))
        {
            break;
        }
    }
    for _ in 0..squarings {
        result = mat_mul(&result, &result, n);
    }
    result
}

pub(crate) fn identity(n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        out[i * n + i] = 1.0;
    }
    out
}
pub(crate) fn mat_mul(left: &[f64], right: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            let value = left[i * n + k];
            for j in 0..n {
                out[i * n + j] += value * right[k * n + j];
            }
        }
    }
    out
}
pub(crate) fn mat_vec(matrix: &[f64], vector: &[f64]) -> Vec<f64> {
    let n = vector.len();
    (0..n)
        .map(|row| {
            (0..n)
                .map(|column| matrix[row * n + column] * vector[column])
                .sum()
        })
        .collect()
}
