//! Downstream smoke coverage for the minimal, renamed dependency.

use diffeq::ndarray::{
    ArrayView0, ArrayView1, ArrayView2, ArrayViewMut0, ArrayViewMut1, ArrayViewMut2, arr0, array,
};
use diffeq::{OdeProblem, SaveMode, SolveOptions, solve};

diffeq::tableau::define_explicit_rk_from_file!(
    pub DownstreamHeun,
    "resources/heun.json",
    crate = diffeq
);

fn options() -> SolveOptions {
    SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(0.01)
        .with_save(SaveMode::Endpoints)
}

/// Solves one componentwise decay equation with ndarray ranks zero through two.
pub fn solve_supported_shapes() -> [Vec<usize>; 3] {
    let scalar = OdeProblem::from_array(
        |mut derivative: ArrayViewMut0<'_, f64>, state: ArrayView0<'_, f64>, _: &(), _: f64| {
            derivative[[]] = -state[[]];
        },
        arr0(1.0),
        (0.0, 0.25),
        (),
    );
    let vector = OdeProblem::from_array(
        |mut derivative: ArrayViewMut1<'_, f64>, state: ArrayView1<'_, f64>, _: &(), _: f64| {
            derivative.zip_mut_with(&state, |derivative, state| *derivative = -*state);
        },
        array![1.0, 2.0],
        (0.0, 0.25),
        (),
    );
    let matrix = OdeProblem::from_array(
        |mut derivative: ArrayViewMut2<'_, f64>, state: ArrayView2<'_, f64>, _: &(), _: f64| {
            derivative.zip_mut_with(&state, |derivative, state| *derivative = -*state);
        },
        array![[1.0, 2.0], [3.0, 4.0]],
        (0.0, 0.25),
        (),
    );

    let scalar = solve(&scalar, DownstreamHeun, &options()).expect("scalar solve succeeds");
    let vector = solve(&vector, DownstreamHeun, &options()).expect("vector solve succeeds");
    let matrix = solve(&matrix, DownstreamHeun, &options()).expect("matrix solve succeeds");

    for solution in [&scalar, &vector, &matrix] {
        assert!(solution.last_state().iter().all(|value| value.is_finite()));
    }
    let scalar_endpoint = scalar.last_state()[0];
    for (endpoint, initial) in vector
        .last_state()
        .iter()
        .chain(matrix.last_state())
        .zip([1.0, 2.0, 1.0, 2.0, 3.0, 4.0])
    {
        assert!((*endpoint - initial * scalar_endpoint).abs() < 1.0e-12);
    }

    [
        scalar.state_shape().to_vec(),
        vector.state_shape().to_vec(),
        matrix.state_shape().to_vec(),
    ]
}

#[cfg(test)]
mod tests {
    use super::solve_supported_shapes;

    #[test]
    fn consumer_owned_tableau_preserves_every_supported_array_shape() {
        assert_eq!(solve_supported_shapes(), [vec![], vec![2], vec![2, 2]]);
    }
}
#[cfg(test)]
mod new_api {
    include!("../../new_api.rs");
}
