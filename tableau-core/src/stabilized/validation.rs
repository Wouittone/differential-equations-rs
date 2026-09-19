use crate::{TableauError, approximately_equal};

use super::{RockRecurrence, RockRecurrenceStage};

pub(super) fn validate_serk2_order(
    degree: usize,
    subdivisions: usize,
    alpha: f64,
    weights: &[f64],
) -> Result<(), TableauError> {
    let weight_sum = weights.iter().sum::<f64>();
    if !approximately_equal(weight_sum, 1.0) {
        return Err(TableauError::new(format!(
            "SERK2 output weights must sum to one; found {weight_sum}"
        )));
    }

    let internal_degree = degree / subdivisions;
    let mut stage_rows = Vec::with_capacity(degree + 1);
    stage_rows.push(vec![0.0; degree]);
    let mut previous_one = vec![0.0; degree];
    let mut derivative = 0;
    for _ in 0..subdivisions {
        let mut next = previous_one.clone();
        next[derivative] += alpha;
        derivative += 1;
        stage_rows.push(next.clone());
        let mut previous_two = std::mem::replace(&mut previous_one, next);

        for _ in 2..=internal_degree {
            let mut next = previous_one
                .iter()
                .zip(&previous_two)
                .map(|(previous, previous_two)| 2.0 * previous - previous_two)
                .collect::<Vec<_>>();
            next[derivative] += 2.0 * alpha;
            derivative += 1;
            stage_rows.push(next.clone());
            previous_two = previous_one;
            previous_one = next;
        }
    }
    debug_assert_eq!(derivative, degree);
    debug_assert_eq!(stage_rows.len(), degree + 1);

    let mut equivalent_weights = vec![0.0; degree];
    for (weight, row) in weights.iter().skip(1).zip(stage_rows.iter().skip(1)) {
        for (equivalent, coefficient) in equivalent_weights.iter_mut().zip(row) {
            *equivalent += weight * coefficient;
        }
    }
    validate_explicit_rk_order(&stage_rows[..degree], &equivalent_weights, 2, "SERK2")
}

fn reconstruct_rock_rows(recurrence: &RockRecurrence, stage_count: usize) -> Vec<Vec<f64>> {
    let degree = recurrence.stages.len() + 1;
    let mut a = vec![vec![0.0; stage_count]; stage_count];
    a[1][0] = recurrence.first_stage;
    for (offset, recurrence_stage) in recurrence.stages.iter().enumerate() {
        let stage = offset + 2;
        let (previous_rows, current_rows) = a.split_at_mut(stage);
        let current = &mut current_rows[0];
        for ((value, previous), previous_two) in current
            .iter_mut()
            .zip(&previous_rows[stage - 1])
            .zip(&previous_rows[stage - 2])
        {
            *value =
                (1.0 + recurrence_stage.kappa) * previous - recurrence_stage.kappa * previous_two;
        }
        current[stage - 1] += recurrence_stage.mu;
    }
    debug_assert!(degree < stage_count);
    a
}

pub(super) fn validate_rock4_orders(
    recurrence: &RockRecurrence,
    finishing_a: &[Vec<f64>],
    b: &[f64],
    b_hat: &[f64],
) -> Result<(), TableauError> {
    let degree = recurrence.stages.len() + 1;
    let stage_count = degree + 5;
    let mut a = reconstruct_rock_rows(recurrence, stage_count);
    let base = a[degree].clone();
    for (finishing_stage, coefficients) in finishing_a.iter().enumerate().take(4).skip(1) {
        let stage = degree + finishing_stage;
        a[stage].copy_from_slice(&base);
        for (offset, coefficient) in coefficients.iter().enumerate() {
            a[stage][degree + offset] += coefficient;
        }
    }

    let mut primary = base.clone();
    for (offset, weight) in b.iter().enumerate() {
        primary[degree + offset] += weight;
    }
    a[degree + 4].copy_from_slice(&primary);

    let mut embedded = base;
    for (offset, weight) in b_hat.iter().enumerate() {
        embedded[degree + offset] += weight;
    }
    validate_explicit_rk_order(&a, &primary, 4, "ROCK4 primary")?;
    validate_explicit_rk_order(&a, &embedded, 3, "ROCK4 embedded")
}

fn validate_explicit_rk_order(
    a: &[Vec<f64>],
    b: &[f64],
    order: usize,
    label: &str,
) -> Result<(), TableauError> {
    let c = a
        .iter()
        .map(|row| row.iter().sum::<f64>())
        .collect::<Vec<_>>();
    let condition =
        |value: f64, expected: f64, name: &str| {
            approximately_equal(value, expected).then_some(()).ok_or_else(|| {
            TableauError::new(format!(
                "{label} violates order condition {name}: found {value}, expected {expected}"
            ))
        })
        };
    condition(b.iter().sum(), 1.0, "b·1")?;
    if order < 2 {
        return Ok(());
    }
    condition(dot(b, &c), 0.5, "b·c")?;
    if order < 3 {
        return Ok(());
    }
    let c_squared = c.iter().map(|value| value * value).collect::<Vec<_>>();
    let a_c = matrix_vector(a, &c);
    condition(dot(b, &c_squared), 1.0 / 3.0, "b·c²")?;
    condition(dot(b, &a_c), 1.0 / 6.0, "b·A·c")?;
    if order < 4 {
        return Ok(());
    }
    let c_cubed = c_squared
        .iter()
        .zip(&c)
        .map(|(squared, value)| squared * value)
        .collect::<Vec<_>>();
    let c_a_c = c.iter().zip(&a_c).map(|(c, ac)| c * ac).collect::<Vec<_>>();
    let a_c_squared = matrix_vector(a, &c_squared);
    let a_a_c = matrix_vector(a, &a_c);
    condition(dot(b, &c_cubed), 0.25, "b·c³")?;
    condition(dot(b, &c_a_c), 0.125, "b·C·A·c")?;
    condition(dot(b, &a_c_squared), 1.0 / 12.0, "b·A·c²")?;
    condition(dot(b, &a_a_c), 1.0 / 24.0, "b·A·A·c")
}

fn matrix_vector(matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
    matrix.iter().map(|row| dot(row, vector)).collect()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

pub(super) fn validate_order_two(
    first_stage: f64,
    stages: &[RockRecurrenceStage],
    finish_first: f64,
    finish_second: f64,
) -> Result<(), TableauError> {
    let degree = stages.len() + 1;
    let derivative_count = degree + 2;
    let mut previous_two = vec![0.0; derivative_count];
    let mut previous_one = vec![0.0; derivative_count];
    let mut next = vec![0.0; derivative_count];
    previous_one[0] = first_stage;

    let mut nodes = vec![0.0; derivative_count];
    nodes[1] = first_stage;
    for (offset, stage) in stages.iter().enumerate() {
        let derivative = offset + 1;
        for index in 0..derivative_count {
            next[index] =
                (1.0 + stage.kappa) * previous_one[index] - stage.kappa * previous_two[index];
        }
        next[derivative] += stage.mu;
        nodes[derivative + 1] = next.iter().sum();
        std::mem::swap(&mut previous_two, &mut previous_one);
        std::mem::swap(&mut previous_one, &mut next);
    }

    nodes[degree + 1] = nodes[degree] + finish_first;
    let mut weights = previous_one;
    weights[degree] += finish_first - finish_second;
    weights[degree + 1] += finish_first + finish_second;
    let weight_sum = weights.iter().sum::<f64>();
    if !approximately_equal(weight_sum, 1.0) {
        return Err(TableauError::new(format!(
            "ROCK2 weights must sum to one; found {weight_sum}"
        )));
    }
    let first_moment = weights
        .iter()
        .zip(nodes)
        .map(|(weight, node)| weight * node)
        .sum::<f64>();
    if !approximately_equal(first_moment, 0.5) {
        return Err(TableauError::new(format!(
            "ROCK2 weights and nodes must satisfy the second-order condition; found {first_moment}"
        )));
    }
    Ok(())
}
