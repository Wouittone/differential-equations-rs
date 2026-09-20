//! Borrowed, typed, output-free integration with reusable storage.
//! Run `cargo run --example reusable_integration --features serde` to include
//! versioned JSON round trips; it also runs without default features or serde.
use differential_equations::{
    InterpolationQuality, Solution, SolverStats,
    solvers::explicit::Tsit5,
    stepping::{
        AdaptiveController, Continuation, ControllerConfig, ExplicitRungeKuttaStepper, Observation,
        ObserverAction, StepError, integrate_rk,
    },
    tolerances::{ErrorNorm, ToleranceError, Tolerances},
};

#[derive(Debug, thiserror::Error)]
enum ApplicationError {
    #[error("force model {code} could not be evaluated")]
    Force { code: u32 },
    #[error(transparent)]
    Tolerance(#[from] ToleranceError),
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rate = -1.0; // Borrowed context; no 'static requirement.
    let tolerances = Tolerances::componentwise(vec![1e-10, 1e-12], vec![1e-9, 1e-9])?;
    let mut evaluations = 0;
    let mut saved = [[0.0; 2]; 3]; // Caller-owned output; no growing trajectory.
    let mut saved_index = 0;
    let mut stepper = ExplicitRungeKuttaStepper::new(Tsit5.tableau()?, 0.0, &[1.0, 0.0])?;
    let mut controller = AdaptiveController::new(ControllerConfig::proportional(5)?, 0.05)?;
    {
        // State[1] is sensitivity to the rate: d(y)/d(rate) = t*exp(rate*t).
        let mut rhs = |_: f64, y: &[f64], dy: &mut [f64]| {
            evaluations += 1;
            dy[0] = rate * y[0];
            dy[1] = y[0] + rate * y[1];
            Ok::<_, ApplicationError>(())
        };
        let mut norm = |old: &[f64], candidate: &[f64], error: &[f64]| {
            Ok::<_, ApplicationError>(tolerances.error_norm(
                old,
                candidate,
                error,
                ErrorNorm::Rms,
            )?)
        };
        integrate_rk(
            &mut stepper,
            &mut controller,
            1.0,
            &[0.0, 0.5, 1.0],
            10000,
            &mut rhs,
            &mut norm,
            &mut |observation: Observation<'_>| {
                if observation.requested {
                    saved[saved_index].copy_from_slice(observation.state);
                    saved_index += 1;
                }
                Ok(ObserverAction::Continue)
            },
        )?;
        // Moving both objects retains accepted state, derivative cache, controller
        // history, and actual next proposal. No problem setup or stage allocation.
        let mut continuation = Continuation {
            stepper,
            controller,
        };
        integrate_rk(
            &mut continuation.stepper,
            &mut continuation.controller,
            2.0,
            &[],
            10000,
            &mut rhs,
            &mut norm,
            &mut |_: Observation<'_>| Ok(ObserverAction::Continue),
        )?;
        assert!((continuation.stepper.state()[0] - (-2.0f64).exp()).abs() < 1e-8);
        assert!((continuation.stepper.state()[1] - 2.0 * (-2.0f64).exp()).abs() < 1e-8);
        stepper = continuation.stepper;
    }
    println!(
        "accepted final time={}, RHS evaluations={evaluations}",
        stepper.time()
    );

    // Application failures are returned directly. No NaN sentinel or external
    // error cell is needed, and accepted state remains available after failure.
    stepper.invalidate_derivative();
    let error = stepper
        .attempt(0.1, &mut |_: f64, _: &[f64], _: &mut [f64]| {
            Err(ApplicationError::Force { code: 42 })
        })
        .unwrap_err();
    match error {
        StepError::User(ApplicationError::Force { code }) => assert_eq!(code, 42),
        other => return Err(Box::new(other)),
    }
    assert_eq!(stepper.time(), 2.0);

    // Saved-only output explicitly advertises linear interpolation. Retaining
    // a native solution's dense output and export_data() instead preserves
    // method-specific polynomial segments and their quality through serde.
    let solution = Solution::from_saved(
        vec![0.0, 0.5, 1.0],
        saved.into_iter().flatten().collect(),
        &[2],
        SolverStats::default(),
    )?;
    assert_eq!(
        solution.interpolation_quality(0.25)?,
        InterpolationQuality::Linear
    );
    assert!(
        solution
            .try_interpolate_method_into(0.25, &mut [0.0; 2])
            .is_err()
    );
    #[cfg(feature = "serde")]
    {
        let json = serde_json::to_string(&solution)?;
        let restored: Solution = serde_json::from_str(&json)?;
        assert_eq!(restored.values(), solution.values());
        assert_eq!(
            restored.interpolation_quality(0.25)?,
            InterpolationQuality::Linear
        );
        println!("versioned trajectory JSON bytes={}", json.len());
    }
    Ok(())
}
