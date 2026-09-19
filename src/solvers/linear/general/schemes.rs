use crate::solvers::exponential::{mat_mul, mat_vec, matrix_exp};
use crate::{SolveError, SolverStats};

use super::{OperatorResult, Scheme};

pub(super) struct StepResult {
    pub(super) state: Vec<f64>,
    pub(super) error: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn perform_step<E>(
    scheme: Scheme,
    n: usize,
    evaluate: &mut E,
    constant_operator: Option<&[f64]>,
    leapfrog_previous: Option<&[f64]>,
    state: &[f64],
    time: f64,
    step: f64,
    stats: &mut SolverStats,
) -> Result<StepResult, SolveError>
where
    E: FnMut(&mut [f64], &[f64], f64, &mut SolverStats) -> OperatorResult,
{
    let mut at = |u: &[f64], t: f64| {
        let mut matrix = vec![0.0; n * n];
        evaluate(&mut matrix, u, t, stats)?;
        Ok::<_, SolveError>(matrix)
    };
    let (state, error) = match scheme {
        Scheme::LieEuler => {
            let a = at(state, time)?;
            (exp_apply(&scale_matrix(&a, step), state), Vec::new())
        }
        Scheme::LinearExponential => {
            let a = constant_operator.ok_or(SolveError::InvalidTableau)?;
            (exp_apply(&scale_matrix(a, step), state), Vec::new())
        }
        Scheme::MagnusMidpoint => {
            let a = at(state, time + step / 2.0)?;
            (exp_apply(&scale_matrix(&a, step), state), Vec::new())
        }
        Scheme::MagnusLeapfrog => {
            if let Some(previous) = leapfrog_previous {
                let a = at(state, time)?;
                (
                    exp_apply(&scale_matrix(&a, 2.0 * step), previous),
                    Vec::new(),
                )
            } else {
                let a = at(state, time + step / 2.0)?;
                (exp_apply(&scale_matrix(&a, step), state), Vec::new())
            }
        }
        Scheme::Rkmk2 => (rkmk2(&mut at, state, time, step)?, Vec::new()),
        Scheme::Rkmk4 => (rkmk4(&mut at, state, time, step, n)?, Vec::new()),
        Scheme::LieRk4 => (lie_rk4(&mut at, state, time, step)?, Vec::new()),
        Scheme::Cg2 => (cg2(&mut at, state, time, step)?, Vec::new()),
        Scheme::Cg3 => (cg3(&mut at, state, time, step)?, Vec::new()),
        Scheme::Cg4a => (cg4a(&mut at, state, time, step)?, Vec::new()),
        Scheme::MagnusAdapt4 => magnus_adapt4(&mut at, state, time, step, n)?,
        Scheme::MagnusGauss4 => (magnus_gl4(&mut at, state, time, step, n, 1.0)?, Vec::new()),
        Scheme::MagnusGl4 => (magnus_gl4(&mut at, state, time, step, n, -1.0)?, Vec::new()),
        Scheme::MagnusGl6 => (magnus_gl6(&mut at, state, time, step, n)?, Vec::new()),
        Scheme::MagnusNc6 => (magnus_nc6(&mut at, state, time, step, n)?, Vec::new()),
        Scheme::MagnusGl8 => (magnus_gl8(&mut at, state, time, step, n)?, Vec::new()),
        Scheme::MagnusNc8 => (magnus_nc8(&mut at, state, time, step, n)?, Vec::new()),
    };
    Ok(StepResult { state, error })
}

type MatrixAt<'a> = dyn FnMut(&[f64], f64) -> Result<Vec<f64>, SolveError> + 'a;

fn rkmk2(at: &mut MatrixAt<'_>, u: &[f64], t: f64, h: f64) -> Result<Vec<f64>, SolveError> {
    let k1 = scale_matrix(&at(u, t)?, h);
    let y2 = exp_apply(&k1, u);
    let k2 = scale_matrix(&at(&y2, t)?, h);
    Ok(exp_apply(&linear_combination(&[(0.5, &k1), (0.5, &k2)]), u))
}

fn rkmk4(
    at: &mut MatrixAt<'_>,
    u: &[f64],
    t: f64,
    h: f64,
    n: usize,
) -> Result<Vec<f64>, SolveError> {
    let k1 = scale_matrix(&at(u, t)?, h);
    let y2 = exp_apply(&scale_matrix(&k1, 0.5), u);
    let k2 = scale_matrix(&at(&y2, t)?, h);
    let omega3 = linear_combination(&[(0.5, &k1), (-0.125, &commutator(&k1, &k2, n))]);
    let y3 = exp_apply(&omega3, u);
    let k3 = scale_matrix(&at(&y3, t)?, h);
    let y4 = exp_apply(&k3, u);
    let k4 = scale_matrix(&at(&y4, t)?, h);
    let bracket = commutator(&k1, &k4, n);
    let omega = linear_combination(&[
        (1.0 / 6.0, &k1),
        (1.0 / 3.0, &k2),
        (1.0 / 3.0, &k3),
        (1.0 / 6.0, &k4),
        (-1.0 / 12.0, &bracket),
    ]);
    Ok(exp_apply(&omega, u))
}

fn lie_rk4(at: &mut MatrixAt<'_>, u: &[f64], t: f64, h: f64) -> Result<Vec<f64>, SolveError> {
    let k1 = scale_matrix(&at(u, t)?, h);
    let y2 = exp_apply(&scale_matrix(&k1, 0.5), u);
    let k2 = scale_matrix(&at(&y2, t)?, h);
    let y3 = exp_apply(&scale_matrix(&k2, 0.5), u);
    let k3 = scale_matrix(&at(&y3, t)?, h);
    let omega4 = linear_combination(&[(1.0, &k3), (-0.5, &k1)]);
    let y4 = exp_apply(&omega4, &y2);
    let k4 = scale_matrix(&at(&y4, t)?, h);
    let first = linear_combination(&[
        (0.25, &k1),
        (1.0 / 6.0, &k2),
        (1.0 / 6.0, &k3),
        (-1.0 / 12.0, &k4),
    ]);
    let half = exp_apply(&first, u);
    let second = linear_combination(&[
        (-1.0 / 12.0, &k1),
        (1.0 / 6.0, &k2),
        (1.0 / 6.0, &k3),
        (0.25, &k4),
    ]);
    Ok(exp_apply(&second, &half))
}

fn cg2(at: &mut MatrixAt<'_>, u: &[f64], t: f64, h: f64) -> Result<Vec<f64>, SolveError> {
    let k1 = scale_matrix(&at(u, t)?, h);
    let y2 = exp_apply(&k1, u);
    let k2 = scale_matrix(&at(&y2, t)?, h);
    let right = exp_apply(&scale_matrix(&k2, 0.5), u);
    Ok(exp_apply(&scale_matrix(&k1, 0.5), &right))
}

fn cg3(at: &mut MatrixAt<'_>, u: &[f64], t: f64, h: f64) -> Result<Vec<f64>, SolveError> {
    let a = at(u, t)?;
    let v2 = exp_apply(&scale_matrix(&a, 0.75 * h), u);
    let b = at(&v2, t + 0.75 * h)?;
    let v3 = composed_actions(u, &[(119.0 / 216.0 * h, &b), (17.0 / 108.0 * h, &a)]);
    let c = at(&v3, t + 17.0 * h / 24.0)?;
    Ok(composed_actions(
        u,
        &[
            (24.0 / 17.0 * h, &c),
            (-2.0 / 3.0 * h, &b),
            (13.0 / 51.0 * h, &a),
        ],
    ))
}

fn cg4a(at: &mut MatrixAt<'_>, u: &[f64], t: f64, h: f64) -> Result<Vec<f64>, SolveError> {
    let a = at(u, t)?;
    let v2 = exp_apply(&scale_matrix(&a, 0.817_722_798_812_485_2 * h), u);
    let b = at(&v2, t + 0.817_722_798_812_485_2 * h)?;
    let v3 = composed_actions(
        u,
        &[
            (0.319_987_637_547_642_7 * h, &b),
            (0.065_986_426_355_602_2 * h, &a),
        ],
    );
    let c = at(&v3, t + 0.385_974_063_903_244_9 * h)?;
    let v4 = composed_actions(
        u,
        &[
            (0.921_441_719_446_494_6 * h, &c),
            (0.499_785_777_677_357_3 * h, &b),
            (-1.096_998_444_837_158_2 * h, &a),
        ],
    );
    let d = at(&v4, t + 0.324_229_052_286_693_7 * h)?;
    let v5 = composed_actions(
        u,
        &[
            (0.355_235_855_902_332_2 * h, &d),
            (0.239_095_837_230_732_6 * h, &c),
            (1.391_856_572_420_324_6 * h, &b),
            (-1.109_297_939_211_346_5 * h, &a),
        ],
    );
    let e = at(&v5, t + 0.876_890_326_342_042_9 * h)?;
    Ok(composed_actions(
        u,
        &[
            (0.332_219_559_106_837_4 * h, &e),
            (-0.190_714_256_550_588_9 * h, &d),
            (0.739_781_398_537_078 * h, &c),
            (-0.018_369_853_156_402 * h, &b),
            (0.137_083_152_063_075_5 * h, &a),
        ],
    ))
}

fn magnus_adapt4(
    at: &mut MatrixAt<'_>,
    u: &[f64],
    t: f64,
    h: f64,
    n: usize,
) -> Result<(Vec<f64>, Vec<f64>), SolveError> {
    let q1 = scale_matrix(&at(u, t)?, h);
    let y2 = scale_matrix(&q1, 0.5);
    let k2 = scale_matrix(&at(&exp_apply(&y2, u), t + h / 2.0)?, h);
    let q2 = linear_combination(&[(1.0, &k2), (-1.0, &q1)]);
    let y3 = linear_combination(&[(0.5, &q1), (0.25, &q2)]);
    let k3 = scale_matrix(&at(&exp_apply(&y3, u), t + h / 2.0)?, h);
    let q3 = linear_combination(&[(1.0, &k3), (-1.0, &k2)]);
    let y4 = linear_combination(&[(1.0, &q1), (1.0, &q2)]);
    let k4 = scale_matrix(&at(&exp_apply(&y4, u), t + h)?, h);
    let q4 = linear_combination(&[(1.0, &k4), (-2.0, &k2), (1.0, &q1)]);
    let bracket12 = commutator(&q1, &q2, n);
    let y5 = linear_combination(&[
        (0.5, &q1),
        (0.25, &q2),
        (1.0 / 3.0, &q3),
        (-1.0 / 24.0, &q4),
        (-1.0 / 48.0, &bracket12),
    ]);
    let k5 = scale_matrix(&at(&exp_apply(&y5, u), t + h / 2.0)?, h);
    let q5 = linear_combination(&[(1.0, &k5), (-1.0, &k2)]);
    let y6 = linear_combination(&[
        (1.0, &q1),
        (1.0, &q2),
        (2.0 / 3.0, &q3),
        (1.0 / 6.0, &q4),
        (-1.0 / 6.0, &bracket12),
    ]);
    let k6 = scale_matrix(&at(&exp_apply(&y6, u), t + h)?, h);
    let q6 = linear_combination(&[(1.0, &k6), (-2.0, &k2), (1.0, &q1)]);
    let inner = linear_combination(&[(1.0, &q2), (-1.0, &q3), (1.0, &q5), (0.5, &q6)]);
    let correction = commutator(&q1, &inner, n);
    let v4 = linear_combination(&[
        (1.0, &q1),
        (1.0, &q2),
        (2.0 / 3.0, &q5),
        (1.0 / 6.0, &q6),
        (-1.0 / 6.0, &correction),
    ]);
    let candidate = exp_apply(&v4, u);
    let embedded = exp_apply(&y6, u);
    let error = candidate
        .iter()
        .zip(&embedded)
        .map(|(a, b)| a - b)
        .collect();
    Ok((candidate, error))
}

fn magnus_gl4(
    at: &mut MatrixAt<'_>,
    u: &[f64],
    t: f64,
    h: f64,
    n: usize,
    commutator_sign: f64,
) -> Result<Vec<f64>, SolveError> {
    let offset = 3.0_f64.sqrt() / 6.0;
    let a1 = at(u, t + h * (0.5 - offset))?;
    let a2 = at(u, t + h * (0.5 + offset))?;
    let bracket = commutator(&a1, &a2, n);
    let omega = linear_combination(&[
        (h / 2.0, &a1),
        (h / 2.0, &a2),
        (commutator_sign * h * h * 3.0_f64.sqrt() / 12.0, &bracket),
    ]);
    Ok(exp_apply(&omega, u))
}

fn magnus_nc6(
    at: &mut MatrixAt<'_>,
    u: &[f64],
    t: f64,
    h: f64,
    n: usize,
) -> Result<Vec<f64>, SolveError> {
    let a0 = at(u, t)?;
    let a1 = at(u, t + h / 4.0)?;
    let a2 = at(u, t + h / 2.0)?;
    let a3 = at(u, t + 3.0 * h / 4.0)?;
    let a4 = at(u, t + h)?;
    let b0 = linear_combination(&[
        (7.0 / 90.0, &a0),
        (32.0 / 90.0, &a1),
        (12.0 / 90.0, &a2),
        (32.0 / 90.0, &a3),
        (7.0 / 90.0, &a4),
    ]);
    let b1 = linear_combination(&[
        (-3.5 / 90.0, &a0),
        (-8.0 / 90.0, &a1),
        (8.0 / 90.0, &a3),
        (3.5 / 90.0, &a4),
    ]);
    let b2 = linear_combination(&[
        (1.75 / 90.0, &a0),
        (2.0 / 90.0, &a1),
        (2.0 / 90.0, &a3),
        (1.75 / 90.0, &a4),
    ]);
    Ok(exp_apply(&magnus6_omega(&b0, &b1, &b2, h, n), u))
}

fn magnus_gl6(
    at: &mut MatrixAt<'_>,
    u: &[f64],
    t: f64,
    h: f64,
    n: usize,
) -> Result<Vec<f64>, SolveError> {
    let offset = (3.0_f64 / 20.0).sqrt();
    let a1 = at(u, t + h * (0.5 - offset))?;
    let a2 = at(u, t + h / 2.0)?;
    let a3 = at(u, t + h * (0.5 + offset))?;
    let b0 = linear_combination(&[(5.0 / 18.0, &a1), (8.0 / 18.0, &a2), (5.0 / 18.0, &a3)]);
    let b1 = linear_combination(&[
        (-15.0_f64.sqrt() / 36.0, &a1),
        (15.0_f64.sqrt() / 36.0, &a3),
    ]);
    let b2 = linear_combination(&[(1.0 / 24.0, &a1), (1.0 / 24.0, &a3)]);
    Ok(exp_apply(&magnus6_omega(&b0, &b1, &b2, h, n), u))
}

fn magnus6_omega(b0: &[f64], b1: &[f64], b2: &[f64], h: f64, n: usize) -> Vec<f64> {
    let inner2 = linear_combination(&[(1.5, b0), (-6.0, b2)]);
    let omega2 = scale_matrix(&commutator(b1, &inner2, n), h * h);
    let inner34 = linear_combination(&[(h / 2.0, b2), (-1.0 / 60.0, &omega2)]);
    let double = commutator(b0, &commutator(b0, &inner34, n), n);
    let mixed = commutator(b1, &omega2, n);
    linear_combination(&[
        (h, b0),
        (1.0, &omega2),
        (h * h, &double),
        (3.0 * h / 5.0, &mixed),
    ])
}

fn magnus_nc8(
    at: &mut MatrixAt<'_>,
    u: &[f64],
    t: f64,
    h: f64,
    n: usize,
) -> Result<Vec<f64>, SolveError> {
    let mut a = Vec::with_capacity(7);
    for j in 0..=6 {
        a.push(at(u, t + h * j as f64 / 6.0)?);
    }
    let s1 = linear_combination(&[(1.0, &a[0]), (1.0, &a[6])]);
    let s2 = linear_combination(&[(1.0, &a[1]), (1.0, &a[5])]);
    let s3 = linear_combination(&[(1.0, &a[2]), (1.0, &a[4])]);
    let r1 = linear_combination(&[(-1.0, &a[0]), (1.0, &a[6])]);
    let r2 = linear_combination(&[(-1.0, &a[1]), (1.0, &a[5])]);
    let r3 = linear_combination(&[(-1.0, &a[2]), (1.0, &a[4])]);
    let b0 = linear_combination(&[
        (41.0 / 840.0, &s1),
        (216.0 / 840.0, &s2),
        (27.0 / 840.0, &s3),
        (272.0 / 840.0, &a[3]),
    ]);
    let b1 = linear_combination(&[(20.5 / 840.0, &r1), (72.0 / 840.0, &r2), (4.5 / 840.0, &r3)]);
    let b2 = linear_combination(&[
        (10.25 / 840.0, &s1),
        (24.0 / 840.0, &s2),
        (0.75 / 840.0, &s3),
    ]);
    let b3 = linear_combination(&[
        (5.125 / 840.0, &r1),
        (8.0 / 840.0, &r2),
        (0.125 / 840.0, &r3),
    ]);
    Ok(exp_apply(&magnus8_omega(&b0, &b1, &b2, &b3, h, n), u))
}

fn magnus_gl8(
    at: &mut MatrixAt<'_>,
    u: &[f64],
    t: f64,
    h: f64,
    n: usize,
) -> Result<Vec<f64>, SolveError> {
    let root = (6.0_f64 / 5.0).sqrt();
    let v1 = 0.5 * ((3.0 + 2.0 * root) / 7.0).sqrt();
    let v2 = 0.5 * ((3.0 - 2.0 * root) / 7.0).sqrt();
    let w1 = 0.5 - (5.0_f64 / 6.0).sqrt() / 6.0;
    let w2 = 0.5 + (5.0_f64 / 6.0).sqrt() / 6.0;
    let a1 = at(u, t + h * (0.5 - v1))?;
    let a2 = at(u, t + h * (0.5 - v2))?;
    let a3 = at(u, t + h * (0.5 + v2))?;
    let a4 = at(u, t + h * (0.5 + v1))?;
    let s1 = linear_combination(&[(1.0, &a1), (1.0, &a4)]);
    let s2 = linear_combination(&[(1.0, &a2), (1.0, &a3)]);
    let r1 = linear_combination(&[(-1.0, &a1), (1.0, &a4)]);
    let r2 = linear_combination(&[(-1.0, &a2), (1.0, &a3)]);
    let b0 = linear_combination(&[(0.5 * w1, &s1), (0.5 * w2, &s2)]);
    let b1 = linear_combination(&[(0.5 * v1 * w1, &r1), (0.5 * v2 * w2, &r2)]);
    let b2 = linear_combination(&[(0.5 * v1 * v1 * w1, &s1), (0.5 * v2 * v2 * w2, &s2)]);
    let b3 = linear_combination(&[(0.5 * v1.powi(3) * w1, &r1), (0.5 * v2.powi(3) * w2, &r2)]);
    Ok(exp_apply(&magnus8_omega(&b0, &b1, &b2, &b3, h, n), u))
}

fn magnus8_omega(b0: &[f64], b1: &[f64], b2: &[f64], b3: &[f64], h: f64, n: usize) -> Vec<f64> {
    let q1 = commutator(&linear_combination(&[(-38.0 / 5.0, b0), (24.0, b2)]), b3, n);
    let q2 = commutator(
        &linear_combination(&[(63.0 / 5.0, b0), (-84.0, b2)]),
        &linear_combination(&[(-5.0 / 28.0, b1), (1.0, b3)]),
        n,
    );
    let inner3 = linear_combination(&[(1.0, b2), (h * 61.0 / 588.0, &q1), (-h / 12.0, &q2)]);
    let q3 = commutator(
        &linear_combination(&[(19.0 / 28.0, b0), (-15.0 / 7.0, b2)]),
        &commutator(b0, &inner3, n),
        n,
    );
    let q4 = commutator(
        b3,
        &linear_combination(&[(20.0 / 7.0, &q1), (10.0, &q2)]),
        n,
    );
    let q5 = commutator(
        &linear_combination(&[(-6025.0 / 4116.0, b0), (2875.0 / 343.0, b2)]),
        &commutator(b2, &q1, n),
        n,
    );
    let q6 = commutator(
        b3,
        &linear_combination(&[
            (20.0 / 7.0, &q3),
            (20.0 / 7.0, &q4),
            (820.0 * h / 189.0, &q5),
        ]),
        n,
    );
    let inner7 = linear_combination(&[(1.0, &q3), (-1.0 / 3.0, &q4), (h, &q5)]);
    let q7 = scale_matrix(&commutator(b0, &commutator(b0, &inner7, n), n), -1.0 / 42.0);
    linear_combination(&[
        (h, b0),
        (h * h, &q1),
        (h * h, &q2),
        (h.powi(3), &q3),
        (h.powi(3), &q4),
        (h.powi(4), &q5),
        (h.powi(4), &q6),
        (h.powi(5), &q7),
    ])
}

fn exp_apply(omega: &[f64], state: &[f64]) -> Vec<f64> {
    mat_vec(&matrix_exp(omega, state.len()), state)
}

fn composed_actions(state: &[f64], actions: &[(f64, &[f64])]) -> Vec<f64> {
    let mut result = state.to_vec();
    for (coefficient, matrix) in actions.iter().rev() {
        result = exp_apply(&scale_matrix(matrix, *coefficient), &result);
    }
    result
}

fn commutator(left: &[f64], right: &[f64], n: usize) -> Vec<f64> {
    linear_combination(&[
        (1.0, &mat_mul(left, right, n)),
        (-1.0, &mat_mul(right, left, n)),
    ])
}

fn scale_matrix(matrix: &[f64], factor: f64) -> Vec<f64> {
    matrix.iter().map(|value| factor * value).collect()
}

fn linear_combination(terms: &[(f64, &[f64])]) -> Vec<f64> {
    let mut output = vec![0.0; terms.first().map_or(0, |(_, matrix)| matrix.len())];
    for (coefficient, matrix) in terms {
        for (destination, value) in output.iter_mut().zip(*matrix) {
            *destination += coefficient * value;
        }
    }
    output
}

pub(super) fn transpose(matrix: &[f64], n: usize) -> Vec<f64> {
    let mut output = vec![0.0; n * n];
    for row in 0..n {
        for column in 0..n {
            output[column * n + row] = matrix[row * n + column];
        }
    }
    output
}
