use differential_equations::ndarray::{ArrayView2, ArrayViewMut2, array};
use differential_equations::solvers::second_order::{
    SecondOrderFunction, SecondOrderOdeAlgorithm, SecondOrderOdeProblem, SecondOrderSolution,
    SecondOrderSolveError, solve_second_order,
};
use differential_equations::{
    OdeAlgorithm, OdeFunction, OdeProblem, Solution, SolutionConstructionError, SolveError,
    SolveOptions, SolverStats, solve,
};

#[derive(Clone, Copy)]
struct ExternalEuler;

impl OdeAlgorithm for ExternalEuler {
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        _: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: OdeFunction<P>,
    {
        let (start, end) = problem.time_span();
        let step = end - start;
        let dimension = problem.initial_state().len();
        let mut derivative = vec![0.0; dimension];
        problem.evaluate(&mut derivative, problem.initial_state(), start)?;

        let mut values = Vec::with_capacity(2 * dimension);
        values.extend_from_slice(problem.initial_state());
        values.extend(
            problem
                .initial_state()
                .iter()
                .zip(derivative)
                .map(|(state, derivative)| state + step * derivative),
        );
        let mut stats = SolverStats::default();
        stats.rhs_evaluations = 1;
        stats.accepted_steps = 1;
        Ok(Solution::from_saved(
            vec![start, end],
            values,
            problem.state_shape(),
            stats,
        )?)
    }
}

#[derive(Clone, Copy)]
struct ExternalSecondOrderEuler;

impl SecondOrderOdeAlgorithm for ExternalSecondOrderEuler {
    fn solve_validated<F, P>(
        &self,
        problem: &SecondOrderOdeProblem<F, P>,
        _: &SolveOptions,
    ) -> Result<SecondOrderSolution, SecondOrderSolveError>
    where
        F: SecondOrderFunction<P>,
    {
        let (start, end) = problem.time_span();
        let step = end - start;
        let dimension = problem.initial_position().len();
        let mut acceleration = vec![0.0; dimension];
        problem.evaluate_acceleration(
            &mut acceleration,
            problem.initial_velocity(),
            problem.initial_position(),
            start,
        )?;

        let mut velocities = Vec::with_capacity(2 * dimension);
        velocities.extend_from_slice(problem.initial_velocity());
        velocities.extend(
            problem
                .initial_velocity()
                .iter()
                .zip(&acceleration)
                .map(|(velocity, acceleration)| velocity + step * acceleration),
        );
        let mut positions = Vec::with_capacity(2 * dimension);
        positions.extend_from_slice(problem.initial_position());
        positions.extend(
            problem
                .initial_position()
                .iter()
                .zip(problem.initial_velocity())
                .map(|(position, velocity)| position + step * velocity),
        );
        let mut stats = SolverStats::default();
        stats.rhs_evaluations = 1;
        stats.accepted_steps = 1;
        Ok(SecondOrderSolution::from_saved(
            vec![start, end],
            velocities,
            positions,
            problem.state_shape(),
            stats,
        )?)
    }
}

#[test]
fn downstream_ordinary_algorithm_can_evaluate_and_return_a_shaped_solution() {
    let problem = OdeProblem::from_array(
        |mut derivative: ArrayViewMut2<'_, f64>, state: ArrayView2<'_, f64>, _: &(), _| {
            derivative.zip_mut_with(&state, |derivative, state| *derivative = -*state)
        },
        array![[1.0, 2.0], [3.0, 4.0]],
        (0.0, 0.25),
        (),
    );

    let solution = solve(&problem, ExternalEuler, &SolveOptions::default()).unwrap();

    assert_eq!(solution.state_shape(), &[2, 2]);
    assert_eq!(solution.last_state(), &[0.75, 1.5, 2.25, 3.0]);
    assert_eq!(solution.stats().rhs_evaluations, 1);
}

#[test]
fn downstream_second_order_algorithm_can_evaluate_and_return_a_shaped_solution() {
    let problem = SecondOrderOdeProblem::from_array(
        |mut acceleration: ArrayViewMut2<'_, f64>,
         _: ArrayView2<'_, f64>,
         _: ArrayView2<'_, f64>,
         _: &(),
         _| acceleration.fill(2.0),
        array![[0.0, 1.0], [2.0, 3.0]],
        array![[10.0, 20.0], [30.0, 40.0]],
        (0.0, 0.5),
        (),
    )
    .unwrap();

    let solution =
        solve_second_order(&problem, ExternalSecondOrderEuler, &SolveOptions::default()).unwrap();

    assert_eq!(solution.state_shape(), &[2, 2]);
    assert_eq!(solution.last_velocity(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(solution.last_position(), &[10.0, 20.5, 31.0, 41.5]);
    assert_eq!(solution.stats().rhs_evaluations, 1);
}

#[test]
fn saved_solution_constructors_reject_malformed_trajectories() {
    assert_eq!(
        Solution::from_saved(vec![0.0, 1.0], vec![1.0], &[1], SolverStats::default(),),
        Err(SolutionConstructionError::DimensionMismatch)
    );
    assert_eq!(
        SecondOrderSolution::from_saved(
            vec![0.0, 1.0, 0.5],
            vec![0.0; 3],
            vec![0.0; 3],
            &[1],
            SolverStats::default(),
        ),
        Err(SolutionConstructionError::NonMonotonicTimes)
    );
    assert_eq!(
        Solution::from_saved(vec![0.0], vec![f64::NAN], &[], SolverStats::default(),),
        Err(SolutionConstructionError::NonFiniteState)
    );
}

#[test]
fn public_evaluation_seams_check_dimensions_and_finiteness() {
    let ordinary = OdeProblem::new(
        |derivative: &mut [f64], _: &[f64], _: &(), _: f64| derivative.fill(f64::INFINITY),
        vec![1.0, 2.0],
        (0.0, 1.0),
        (),
    );
    assert_eq!(
        ordinary.evaluate(&mut [0.0], &[1.0, 2.0], 0.0),
        Err(SolveError::EvaluationDimensionMismatch)
    );
    assert_eq!(
        ordinary.evaluate(&mut [0.0, 0.0], &[1.0, 2.0], 0.0),
        Err(SolveError::NonFiniteDerivative)
    );

    let second_order = SecondOrderOdeProblem::new(
        |acceleration: &mut [f64], _: &[f64], _: &[f64], _: &(), _: f64| {
            acceleration.fill(f64::NAN)
        },
        vec![0.0, 0.0],
        vec![1.0, 2.0],
        (0.0, 1.0),
        (),
    );
    assert_eq!(
        second_order.evaluate_acceleration(&mut [0.0, 0.0], &[0.0], &[1.0, 2.0], 0.0),
        Err(SolveError::EvaluationDimensionMismatch)
    );
    assert_eq!(
        second_order.evaluate_acceleration(&mut [0.0, 0.0], &[0.0, 0.0], &[1.0, 2.0], 0.0,),
        Err(SolveError::NonFiniteDerivative)
    );
}
