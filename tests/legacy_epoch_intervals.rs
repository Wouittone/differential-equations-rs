use differential_equations::solvers::{
    explicit::Tsit5,
    rosenbrock::Rodas4,
    second_order::{FineRkn4, Nystrom4, SecondOrderOdeProblem, solve_second_order},
};
use differential_equations::{OdeProblem, SaveMode, SolveOptions, solve};
#[test]
fn first_order_fixed_steps_follow_reported_epoch_interval() {
    for origin in [0., 1e9, 1e12] {
        for sign in [-1., 1.] {
            let p = OdeProblem::new(
                |d: &mut [f64], _: &[f64], _: &(), _: f64| d[0] = 1.,
                [0.],
                (origin, origin + sign),
                (),
            )
            .with_jacobian(|j: &mut [f64], _: &[f64], _: &(), _: f64| j[0] = 0.);
            let options = SolveOptions::new()
                .with_adaptive(false)
                .with_initial_step(0.1)
                .with_save(SaveMode::Endpoints);
            let rk = solve(&p, Tsit5, &options).unwrap();
            let ros = solve(&p, Rodas4, &options).unwrap();
            assert!(
                (rk.last_state()[0] - sign).abs() < 1e-12,
                "rk origin={origin} sign={sign} value={}",
                rk.last_state()[0]
            );
            assert!(
                (ros.last_state()[0] - sign).abs() < 1e-12,
                "ros origin={origin} sign={sign} value={}",
                ros.last_state()[0]
            );
        }
    }
}
#[test]
fn second_order_fixed_steps_follow_reported_epoch_interval() {
    for origin in [0., 1e9, 1e12] {
        for sign in [-1., 1.] {
            let p = SecondOrderOdeProblem::new(
                |a: &mut [f64], _: &[f64], _: &[f64], _: &(), _: f64| a[0] = 0.,
                [1.],
                [0.],
                (origin, origin + sign),
                (),
            );
            let options = SolveOptions::new()
                .with_adaptive(false)
                .with_initial_step(0.1)
                .with_save(SaveMode::Endpoints);
            let adaptive_family = solve_second_order(&p, FineRkn4, &options).unwrap();
            let fixed_family = solve_second_order(&p, Nystrom4, &options).unwrap();
            assert!(
                (adaptive_family.last_position()[0] - sign).abs() < 1e-12,
                "adaptive RKN origin={origin} sign={sign} value={}",
                adaptive_family.last_position()[0]
            );
            assert!(
                (fixed_family.last_position()[0] - sign).abs() < 1e-12,
                "fixed RKN origin={origin} sign={sign} value={}",
                fixed_family.last_position()[0]
            );
        }
    }
}
