#[path = "../src/symplectic.rs"]
mod symplectic;

use differential_equations::{SaveMode, SecondOrderOdeProblem, SolveOptions};
use symplectic::{SymplecticAlgorithm, solve_symplectic};

fn oscillator() -> SecondOrderOdeProblem<impl Fn(&mut [f64], &[f64], &[f64], &(), f64), ()> {
    SecondOrderOdeProblem::new(
        |output: &mut [f64], _: &[f64], position: &[f64], _: &(), _: f64| {
            output[0] = -position[0];
        },
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
        (
            "PseudoVerletLeapfrog",
            symplectic::PseudoVerletLeapfrog::tableau(),
            2,
        ),
        ("McAte2", symplectic::McAte2::tableau(), 2),
        ("Ruth3", symplectic::Ruth3::tableau(), 3),
        ("McAte3", symplectic::McAte3::tableau(), 3),
        ("CandyRoz4", symplectic::CandyRoz4::tableau(), 4),
        ("McAte4", symplectic::McAte4::tableau(), 4),
        ("CalvoSanz4", symplectic::CalvoSanz4::tableau(), 5),
        ("McAte42", symplectic::McAte42::tableau(), 5),
        ("McAte5", symplectic::McAte5::tableau(), 6),
        ("Yoshida6", symplectic::Yoshida6::tableau(), 8),
        ("KahanLi6", symplectic::KahanLi6::tableau(), 10),
        ("McAte8", symplectic::McAte8::tableau(), 16),
        ("KahanLi8", symplectic::KahanLi8::tableau(), 18),
        ("SofSpa10", symplectic::SofSpa10::tableau(), 36),
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
    let ruth_ratio =
        endpoint_error(symplectic::Ruth3, 0.05) / endpoint_error(symplectic::Ruth3, 0.025);
    assert!(ruth_ratio > 7.0, "Ruth3 ratio was {ruth_ratio}");

    let yoshida_ratio =
        endpoint_error(symplectic::Yoshida6, 0.1) / endpoint_error(symplectic::Yoshida6, 0.05);
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
        symplectic::KahanLi6,
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
