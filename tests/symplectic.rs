use differential_equations::algorithms::*;
use differential_equations::*;
use std::error::Error as _;

type Acceleration = fn(&mut [f64], &[f64], &[f64], &(), f64);

fn oscillator() -> SecondOrderOdeProblem<Acceleration, ()> {
    fn acceleration(output: &mut [f64], _: &[f64], position: &[f64], _: &(), _: f64) {
        output[0] = -position[0];
    }
    SecondOrderOdeProblem::new(
        acceleration as Acceleration,
        vec![0.0],
        vec![1.0],
        (0.0, 1.0),
        (),
    )
}

fn options(step: f64) -> SolveOptions {
    SolveOptions {
        adaptive: false,
        initial_step: Some(step),
        max_step: step,
        save: SaveMode::Endpoints,
        ..SolveOptions::default()
    }
}

fn endpoint_error<A: SymplecticAlgorithm>(algorithm: A, step: f64) -> f64 {
    let problem = oscillator();
    let solution = solve_symplectic(&problem, algorithm, &options(step)).unwrap();
    (solution.last_position()[0] - 1.0_f64.cos()).hypot(solution.last_velocity()[0] + 1.0_f64.sin())
}

#[test]
fn all_recovered_names_expose_pinned_tableaus() {
    let methods = [
        ("PseudoVerletLeapfrog", PseudoVerletLeapfrog::tableau(), 2),
        ("McAte2", McAte2::tableau(), 2),
        ("Ruth3", Ruth3::tableau(), 3),
        ("McAte3", McAte3::tableau(), 3),
        ("CandyRoz4", CandyRoz4::tableau(), 4),
        ("McAte4", McAte4::tableau(), 4),
        ("CalvoSanz4", CalvoSanz4::tableau(), 5),
        ("McAte42", McAte42::tableau(), 5),
        ("McAte5", McAte5::tableau(), 6),
        ("Yoshida6", Yoshida6::tableau(), 8),
        ("KahanLi6", KahanLi6::tableau(), 10),
        ("McAte8", McAte8::tableau(), 16),
        ("KahanLi8", KahanLi8::tableau(), 18),
        ("SofSpa10", SofSpa10::tableau(), 36),
    ];
    for (name, tableau, stages) in methods {
        assert_eq!(tableau.stages(), stages, "{name} stage count");
        assert_eq!(tableau.a.len(), tableau.b.len());
        assert!(
            (tableau.a.iter().sum::<f64>() - 1.0).abs() < 5.0e-12,
            "{name} a sum"
        );
        assert!(
            (tableau.b.iter().sum::<f64>() - 1.0).abs() < 5.0e-12,
            "{name} b sum"
        );
    }
}

#[test]
fn compositions_have_their_expected_orders() {
    let ruth_ratio = endpoint_error(Ruth3, 0.05) / endpoint_error(Ruth3, 0.025);
    assert!(ruth_ratio > 7.0, "Ruth3 ratio was {ruth_ratio}");

    let yoshida_ratio = endpoint_error(Yoshida6, 0.1) / endpoint_error(Yoshida6, 0.05);
    assert!(yoshida_ratio > 35.0, "Yoshida6 ratio was {yoshida_ratio}");
}

#[test]
fn higher_order_method_preserves_bounded_oscillator_energy() {
    let problem = SecondOrderOdeProblem::new(
        |output: &mut [f64], _: &[f64], position: &[f64], _: &(), _: f64| {
            output[0] = -position[0];
        },
        vec![0.0],
        vec![1.0],
        (0.0, 200.0),
        (),
    );
    let solution = solve_symplectic(
        &problem,
        KahanLi6,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(0.1),
            max_step: 0.1,
            save: SaveMode::EveryStep,
            ..SolveOptions::default()
        },
    )
    .unwrap();
    let maximum_error = solution
        .times()
        .iter()
        .enumerate()
        .map(|(index, _)| {
            let q = solution.position(index).unwrap()[0];
            let v = solution.velocity(index).unwrap()[0];
            (0.5 * (q * q + v * v) - 0.5).abs()
        })
        .fold(0.0, f64::max);
    assert!(
        maximum_error < 0.01,
        "maximum energy error was {maximum_error}"
    );
    assert!(solution.rhs_evaluations() > 1_000);
}

#[test]
fn fixed_steps_work_backward_and_honor_save_at() {
    let problem = SecondOrderOdeProblem::new(
        |output: &mut [f64], _: &[f64], position: &[f64], _: &(), _: f64| {
            output[0] = -position[0];
        },
        vec![-1.0_f64.sin()],
        vec![1.0_f64.cos()],
        (1.0, 0.0),
        (),
    );
    let solution = solve_symplectic(
        &problem,
        Yoshida6,
        &SolveOptions {
            adaptive: false,
            initial_step: Some(0.01),
            max_step: 0.01,
            save_at: vec![0.75, 0.5, 0.0],
            ..SolveOptions::default()
        },
    )
    .unwrap();

    assert_eq!(solution.times(), &[0.75, 0.5, 0.0]);
    assert!((solution.last_position()[0] - 1.0).abs() < 1.0e-10);
    assert!(solution.last_velocity()[0].abs() < 1.0e-10);
}

#[test]
fn work_count_is_one_acceleration_evaluation_per_stage() {
    let solution = solve_symplectic(&oscillator(), McAte4, &options(0.25)).unwrap();

    assert_eq!(solution.rhs_evaluations(), 4 * McAte4::tableau().stages());
    assert_eq!(solution.dimension(), 1);
    assert_eq!(solution.position_values().len(), 2);
    assert_eq!(solution.velocity_values().len(), 2);
}

#[test]
fn wrapped_errors_preserve_their_source() {
    let error = SymplecticSolveError::from(SolveError::InvalidInitialStep);

    assert_eq!(
        error.to_string(),
        "the initial step must be finite and positive"
    );
    assert_eq!(
        error.source().unwrap().to_string(),
        SolveError::InvalidInitialStep.to_string()
    );
}
