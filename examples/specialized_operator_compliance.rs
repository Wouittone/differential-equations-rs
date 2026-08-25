use differential_equations::algorithms::rosenbrock::Rosenbrock23;
use differential_equations::solvers::exponential::rkip::{RKIP, solve_rkip};
use differential_equations::solvers::rosenbrock::amf::{
    AMF, AmfProblem, build_amf_function, solve_amf,
};
use differential_equations::solvers::stabilized::irkc::{IRKC, solve_irkc};
use differential_equations::{SaveMode, SemilinearOdeProblem, SolveOptions, SplitOdeProblem};

fn fixed(step: f64) -> SolveOptions {
    SolveOptions::new()
        .with_adaptive(false)
        .with_initial_step(step)
        .with_save(SaveMode::Endpoints)
}

fn main() {
    let amf_function = build_amf_function(
        1,
        |output: &mut [f64], state: &[f64], _: &(), _: f64| output[0] = -3.0 * state[0],
        |output: &mut [f64], _: &[f64], _: &(), _: f64| output[0] = -3.0,
        vec![vec![-1.0], vec![-2.0]],
        |factors: &mut [Vec<f64>], _: &[f64], _: &(), _: f64| {
            factors[0][0] = -1.0;
            factors[1][0] = -2.0;
        },
    )
    .unwrap();
    let amf_problem = AmfProblem::new(amf_function, vec![1.0], (0.0, 0.5), ()).unwrap();
    let amf = solve_amf(&amf_problem, AMF::new(Rosenbrock23), &fixed(0.01)).unwrap();
    println!("amf,{:.17e}", amf.last_state()[0]);

    let rkip_problem = SemilinearOdeProblem::new(
        vec![-2.0],
        |output: &mut [f64], _: &[f64], _: &(), _: f64| output[0] = 1.0,
        vec![1.0],
        (0.0, 1.0),
        (),
    )
    .unwrap();
    let rkip_algorithm = RKIP::new(0.1, 0.2, 2).unwrap();
    let rkip = solve_rkip(&rkip_problem, &rkip_algorithm, &fixed(0.1)).unwrap();
    println!("rkip,{:.17e}", rkip.last_state()[0]);

    let irkc_problem = SplitOdeProblem::new(
        |output: &mut [f64], state: &[f64], _: &(), _: f64| output[0] = -100.0 * state[0],
        |output: &mut [f64], state: &[f64], _: &(), _: f64| output[0] = -state[0],
        vec![1.0],
        (0.0, 0.1),
        (),
    )
    .with_implicit_jacobian(|jacobian, _, _, _| jacobian[0] = -1.0);
    let irkc = solve_irkc(
        &irkc_problem,
        IRKC::new().with_eigenvalue_estimate(100.0),
        &fixed(0.001),
    )
    .unwrap();
    println!("irkc,{:.17e}", irkc.last_state()[0]);
}
