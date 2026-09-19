use differential_equations::ndarray::{ArrayView1, ArrayViewMut1, array};
use differential_equations::solvers::explicit::Tsit5;
use differential_equations::{OdeProblem, SaveMode, SolveOptions, solve};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let problem = OdeProblem::builder()
        .initial_state(array![1.0])
        .time_span((0.0, 1.0))
        .parameters(-2.0)
        .build_with_in_place_rhs(
            |mut derivative: ArrayViewMut1<'_, f64>,
             state: ArrayView1<'_, f64>,
             rate: &f64,
             _time: f64| {
                derivative[0] = rate * state[0];
            },
        );
    let options = SolveOptions::new()
        .with_tolerances(1.0e-9, 1.0e-9)
        .with_save(SaveMode::Endpoints);

    let solution = solve(&problem, Tsit5, &options)?;
    println!("u(1) = {}", solution.last_state()[0]);
    println!("accepted steps = {}", solution.stats().accepted_steps);
    Ok(())
}
