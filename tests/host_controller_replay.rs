//! Replay pinned downstream control formulas independently of solver kernels.
use differential_equations::stepping::{AdaptiveController, ControllerConfig, MinimumStepPolicy};

#[test]
fn satkit_pid_accept_reject_history_and_next_proposal_match() {
    for direction in [-1.0, 1.0] {
        let mut config = ControllerConfig::pid([0.7 / 5.0, 0.4 / 5.0, 0.1 / 5.0]);
        config.initial_error_history = [1e-4, 1e-4];
        config.error_history_floor = 1e-4;
        config.accept_equal = false;
        config.rejection_exponent = Some(0.7 / 5.0);
        config.repeated_rejection_maximum = Some(0.5);
        config.rejected_acceptance_maximum = config.maximum_factor;
        config.minimum_step = 1e-12;
        config.minimum_step_policy = MinimumStepPolicy::ForceAccept;
        let mut controller = AdaptiveController::new(config, direction * 10.0).unwrap();
        let (mut h, mut previous, mut older, mut rejections) =
            (direction * 10.0, 1e-4_f64, 1e-4_f64, 0);
        for error in [0.0_f64, 0.2, 30.0, 30.0, 2.0, 0.8, 1.0, 1e-6, 0.7] {
            let accepted = error < 1.0 || h.abs() <= 1e-12;
            let q = if accepted {
                let raw =
                    error.powf(0.7 / 5.0) / previous.powf(0.4 / 5.0) * older.powf(0.1 / 5.0) / 0.9;
                rejections = 0;
                older = previous;
                previous = error.max(1e-4);
                raw.clamp(0.1, 5.0)
            } else {
                rejections += 1;
                let raw = error.powf(0.7 / 5.0) / 0.9;
                if rejections > 1 {
                    raw.clamp(2.0, 5.0)
                } else {
                    raw.min(5.0)
                }
            };
            h /= q;
            let decision = controller.assess(controller.next_step(), error).unwrap();
            assert_eq!(decision.accepted, accepted);
            assert!(
                (decision.next_step - h).abs() < 2e-13 * h.abs().max(1.0),
                "{error}: {} vs {h}",
                decision.next_step
            );
            assert_eq!(controller.state().accepted_errors, [previous, older]);
            assert_eq!(controller.state().consecutive_rejections, rejections);
            let state = controller.state();
            let mut restored = AdaptiveController::new(config, direction).unwrap();
            restored.restore(state).unwrap();
            assert_eq!(restored.state(), state);
        }
    }
}

#[test]
fn brahe_distinct_accept_and_reject_exponents_match() {
    for direction in [-1.0, 1.0] {
        let mut config = ControllerConfig::proportional(5).unwrap();
        config.rejection_exponent = Some(0.25);
        config.rejected_acceptance_maximum = 10.0;
        config.minimum_step = 1e-12;
        config.maximum_step = 900.0;
        config.minimum_step_policy = MinimumStepPolicy::ForceAccept;
        let mut c = AdaptiveController::new(config, direction * 60.0).unwrap();
        let mut h = direction * 60.0;
        for error in [4.0_f64, 10.0, 0.9, 0.0, 0.01, 1.0, 1.1, 0.2] {
            let accept = error <= 1.0 || h.abs() <= 1e-12;
            let raw = if error == 0.0 {
                10.0
            } else {
                0.9 * (1.0 / error).powf(if accept { 0.2 } else { 0.25 })
            };
            let expected = direction * (h.abs() * raw.clamp(0.2, 10.0)).clamp(1e-12, 900.0);
            let decision = c.assess(h, error).unwrap();
            assert_eq!(decision.accepted, accept);
            assert!((decision.next_step - expected).abs() < 2e-13 * expected.abs().max(1.0));
            h = decision.next_step;
        }
    }
}
