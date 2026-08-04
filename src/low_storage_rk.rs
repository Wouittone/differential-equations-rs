//! Fixed-step two-register low-storage Runge--Kutta methods.
//!
//! This module implements the Williamson 2N recurrence used by the pinned
//! `OrdinaryDiffEqLowStorageRK` source. The numerical recurrence and stage
//! times are preserved. OrdinaryDiffEq's stage/step limiter, fused-array
//! `williamson_condition`, and threading configuration are not exposed.

// Preserve the pinned source's decimal coefficient literals exactly.
#![allow(clippy::excessive_precision)]

use std::marker::PhantomData;

use crate::integrator::{
    KernelCapabilities, StepEstimate, StepKernel, integrate as drive_integration,
};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

trait LowStorage2N {
    const A: &'static [f64];
    const B: &'static [f64];
    const C: &'static [f64];
}

macro_rules! method {
    ($name:ident, $coefficients:ident, $doc:literal, $a:expr, $b:expr, $c:expr) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        pub struct $name;

        struct $coefficients;

        impl LowStorage2N for $coefficients {
            const A: &'static [f64] = $a;
            const B: &'static [f64] = $b;
            const C: &'static [f64] = $c;
        }

        impl OdeAlgorithm for $name {
            fn solve<F, P>(
                &self,
                problem: &OdeProblem<F, P>,
                options: &SolveOptions,
            ) -> Result<Solution, SolveError>
            where
                F: Fn(&mut [f64], &[f64], &P, f64),
            {
                integrate::<F, P, $coefficients>(problem, options)
            }
        }
    };
}

method!(
    Ork256,
    Ork256Coefficients,
    "Five-stage, second-order low-storage method for wave propagation.",
    &[-1.0, -1.55798, -1.0, -0.45031],
    &[0.2, 0.83204, 0.6, 0.35394, 0.2],
    &[0.2, 0.2, 0.8, 0.8]
);

method!(
    CarpenterKennedy2N54,
    CarpenterKennedy2N54Coefficients,
    "Five-stage, fourth-order Carpenter--Kennedy 2N-storage method.",
    &[
        -567_301_805_773.0 / 1_357_537_059_087.0,
        -2_404_267_990_393.0 / 2_016_746_695_238.0,
        -3_550_918_686_646.0 / 2_091_501_179_385.0,
        -1_275_806_237_668.0 / 842_570_457_699.0,
    ],
    &[
        1_432_997_174_477.0 / 9_575_080_441_755.0,
        5_161_836_677_717.0 / 13_612_068_292_357.0,
        1_720_146_321_549.0 / 2_090_206_949_498.0,
        3_134_564_353_537.0 / 4_481_467_310_338.0,
        2_277_821_191_437.0 / 14_882_151_754_819.0,
    ],
    &[
        1_432_997_174_477.0 / 9_575_080_441_755.0,
        2_526_269_341_429.0 / 6_820_363_962_896.0,
        2_006_345_519_317.0 / 3_224_310_063_776.0,
        2_802_321_613_138.0 / 2_924_317_926_251.0,
    ]
);

method!(
    Shlddrk64,
    Shlddrk64Coefficients,
    "Six-stage, fourth-order low-dissipation and low-dispersion method.",
    &[-0.4919575, -0.8946264, -1.5526678, -3.4077973, -1.074264],
    &[0.1453095, 0.4653797, 0.4675397, 0.7795279, 0.3574327, 0.15],
    &[0.1453095, 0.3817422, 0.6367813, 0.7560744, 0.9271047]
);

method!(
    Dglddrk73C,
    Dglddrk73CCoefficients,
    "Seven-stage, third-order low-dissipation and low-dispersion method.",
    &[
        -0.808316387498383,
        -1.503407858773331,
        -1.053064525050744,
        -1.463149119280508,
        -0.659288128108783,
        -1.667891931891068,
    ],
    &[
        0.0119705267309784,
        0.8886897793820711,
        0.4578382089261419,
        0.5790045253338471,
        0.3160214638138484,
        0.2483525368264122,
        0.0677123095940884,
    ],
    &[
        0.0119705267309784,
        0.182317794036199,
        0.5082168062551849,
        0.653203122014859,
        0.853440138567825,
        0.998046608462379,
    ]
);

method!(
    Dglddrk84C,
    Dglddrk84CCoefficients,
    "Eight-stage, fourth-order low-dissipation and low-dispersion method.",
    &[
        -0.721296248227924,
        -0.0107733657161298,
        -0.516258469893097,
        -1.730100286632201,
        -5.200129304403076,
        0.783705894541642,
        -0.544583609433219,
    ],
    &[
        0.2165936736758085,
        0.1773950826411583,
        0.0180253861162329,
        0.0847347637254149,
        0.8129106974622483,
        1.90341603042276,
        0.1314841743399048,
        0.2082583170674149,
    ],
    &[
        0.2165936736758085,
        0.266034348753817,
        0.284005612252272,
        0.325126684378857,
        0.455514959918753,
        0.771321931710117,
        0.919902896453866,
    ]
);

method!(
    Dglddrk84F,
    Dglddrk84FCoefficients,
    "Eight-stage, fourth-order low-dissipation and low-dispersion method.",
    &[
        -0.5534431294501569,
        0.0106598757020349,
        -0.5515812888932,
        -1.885790377558741,
        -5.701295742793264,
        2.113903965664793,
        -0.533957882667528,
    ],
    &[
        0.0803793688273695,
        0.5388497458569843,
        0.0197497440903196,
        0.0991184129733997,
        0.7466920411064123,
        1.679584245618894,
        0.2433728067008188,
        0.1422730459001373,
    ],
    &[
        0.0803793688273695,
        0.321006425033843,
        0.340850182660466,
        0.385036482428547,
        0.50400524775341,
        0.657897756116854,
        0.9484087623348481,
    ]
);

method!(
    Ndblsrk124,
    Ndblsrk124Coefficients,
    "Twelve-stage, fourth-order low-storage method for advection-dominated problems.",
    &[
        -0.0923311242368072,
        -0.9441056581158819,
        -4.3271273247576394,
        -2.1557771329026072,
        -0.9770727190189062,
        -0.7581835342571139,
        -1.7977525470825499,
        -2.691566797270077,
        -4.6466798960268143,
        -0.1539613783825189,
        -0.5943293901830616,
    ],
    &[
        0.0650008435125904,
        0.0161459902249842,
        0.5758627178358159,
        0.1649758848361671,
        0.3934619494248182,
        0.0443509641602719,
        0.2074504268408778,
        0.6914247433015102,
        0.3766646883450449,
        0.0757190350155483,
        0.2027862031054088,
        0.2167029365631842,
    ],
    &[
        0.0650008435125904,
        0.0796560563081853,
        0.1620416710085376,
        0.2248877362907778,
        0.2952293985641261,
        0.3318332506149405,
        0.4094724050198658,
        0.6356954475753369,
        0.6806551557645497,
        0.714377371241835,
        0.9032588871651854,
    ]
);

method!(
    Ndblsrk134,
    Ndblsrk134Coefficients,
    "Thirteen-stage, fourth-order low-storage method for advection-dominated problems.",
    &[
        -0.6160178650170565,
        -0.4449487060774118,
        -1.0952033345276178,
        -1.2256030785959187,
        -0.2740182222332805,
        -0.0411952089052647,
        -0.179708489915356,
        -1.1771530652064288,
        -0.4078831463120878,
        -0.8295636426191777,
        -4.7895970584252288,
        -0.6606671432964504,
    ],
    &[
        0.0271990297818803,
        0.1772488819905108,
        0.0378528418949694,
        0.6086431830142991,
        0.21543139743161,
        0.2066152563885843,
        0.0415864076069797,
        0.0219891884310925,
        0.9893081222650993,
        0.0063199019859826,
        0.3749640721105318,
        1.6080235151003195,
        0.0961209123818189,
    ],
    &[
        0.0271990297818803,
        0.0952594339119365,
        0.1266450286591127,
        0.1825883045699772,
        0.3737511439063931,
        0.5301279418422206,
        0.5704177433952291,
        0.5885784947099155,
        0.6160769826246714,
        0.6223252334314046,
        0.6897593128753419,
        0.9126827615920843,
    ]
);

method!(
    Ndblsrk144,
    Ndblsrk144Coefficients,
    "Fourteen-stage, fourth-order low-storage method for advection-dominated problems.",
    &[
        -0.718801210867241,
        -0.778533117342157,
        -0.0053282796654044,
        -0.8552979934029281,
        -3.9564138245774565,
        -1.5780575380587385,
        -2.0837094552574054,
        -0.748333418276161,
        -0.7032861106563359,
        0.0013917096117681,
        -0.093207536963746,
        -0.9514200470875948,
        -7.1151571693922548,
    ],
    &[
        0.0367762454319673,
        0.3136296607553959,
        0.1531848691869027,
        0.0030097086818182,
        0.332629379064611,
        0.2440251405350864,
        0.3718879239592277,
        0.6204126221582444,
        0.1524043173028741,
        0.0760894927419266,
        0.0077604214040978,
        0.0024647284755382,
        0.0780348340049386,
        5.5059777270269628,
    ],
    &[
        0.0367762454319673,
        0.1249685262725025,
        0.2446177702277698,
        0.247614953107042,
        0.2969311120382472,
        0.3978149645802642,
        0.5270854589440328,
        0.6981269994175695,
        0.8190890835352128,
        0.8527059887098624,
        0.8604711817462826,
        0.8627060376969976,
        0.8734213127600976,
    ]
);

fn integrate<F, P, T>(
    problem: &OdeProblem<F, P>,
    options: &SolveOptions,
) -> Result<Solution, SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    T: LowStorage2N,
{
    validate_recurrence::<T>()?;
    drive_integration(
        problem,
        options,
        LowStorageKernel::<T>::new(problem.initial_state().len()),
    )
}

fn validate_recurrence<T: LowStorage2N>() -> Result<(), SolveError> {
    if T::A.len() + 1 != T::B.len() || T::A.len() != T::C.len() {
        return Err(SolveError::InvalidTableau);
    }
    Ok(())
}

struct LowStorageKernel<T> {
    derivative: Vec<f64>,
    residual: Vec<f64>,
    marker: PhantomData<fn() -> T>,
}

impl<T> LowStorageKernel<T> {
    fn new(dimension: usize) -> Self {
        Self {
            derivative: vec![0.0; dimension],
            residual: vec![0.0; dimension],
            marker: PhantomData,
        }
    }
}

impl<F, P, T> StepKernel<F, P> for LowStorageKernel<T>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
    T: LowStorage2N,
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::new(false, 1)
    }

    fn initialize(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: f64,
        _: &mut SolverStats,
    ) -> Result<(), SolveError> {
        // Stage zero evaluates the current derivative on every attempt.
        Ok(())
    }

    fn estimate_initial_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: f64,
        _: f64,
        _: f64,
        _: &mut [f64],
        _: &SolveOptions,
        _: &mut SolverStats,
    ) -> Result<f64, SolveError> {
        Err(SolveError::InitialStepRequired)
    }

    fn attempt_step(
        &mut self,
        problem: &OdeProblem<F, P>,
        state: &[f64],
        time: f64,
        step: f64,
        candidate: &mut [f64],
        _: &SolveOptions,
        stats: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        candidate.copy_from_slice(state);
        evaluate(problem, &mut self.derivative, state, time, stats)?;
        for ((residual, candidate), derivative) in self
            .residual
            .iter_mut()
            .zip(&mut *candidate)
            .zip(&self.derivative)
        {
            *residual = step * derivative;
            *candidate += T::B[0] * *residual;
        }
        for stage in 0..T::A.len() {
            evaluate(
                problem,
                &mut self.derivative,
                candidate,
                time + T::C[stage] * step,
                stats,
            )?;
            for ((residual, candidate), derivative) in self
                .residual
                .iter_mut()
                .zip(&mut *candidate)
                .zip(&self.derivative)
            {
                *residual = T::A[stage] * *residual + step * derivative;
                *candidate += T::B[stage + 1] * *residual;
            }
        }
        ensure_finite(candidate)?;
        Ok(StepEstimate::new(0.0))
    }

    fn accept_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: &[f64],
        _: f64,
        _: f64,
        _: bool,
        _: &mut SolverStats,
    ) -> Result<(), SolveError> {
        Ok(())
    }

    fn reject_step(&mut self) {}
}

fn evaluate<F, P>(
    problem: &OdeProblem<F, P>,
    derivative: &mut [f64],
    state: &[f64],
    time: f64,
    stats: &mut SolverStats,
) -> Result<(), SolveError>
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    (problem.rhs)(derivative, state, problem.parameters(), time);
    stats.rhs_evaluations += 1;
    ensure_finite(derivative)
}

fn ensure_finite(values: &[f64]) -> Result<(), SolveError> {
    values
        .iter()
        .all(|value| value.is_finite())
        .then_some(())
        .ok_or(SolveError::NonFiniteDerivative)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use super::{
        CarpenterKennedy2N54, Dglddrk73C, Dglddrk84C, Dglddrk84F, Ndblsrk124, Ndblsrk134,
        Ndblsrk144, Ork256, Shlddrk64, integrate,
    };
    use crate::{
        CallbackAction, OdeAlgorithm, OdeProblem, SaveMode, SolveError, SolveOptions, solve,
    };

    type TestRhs = fn(&mut [f64], &[f64], &(), f64);

    fn problem(time_span: (f64, f64), initial: f64) -> OdeProblem<TestRhs, ()> {
        fn rhs(du: &mut [f64], u: &[f64], _: &(), time: f64) {
            du[0] = u[0] + time;
        }
        OdeProblem::new(rhs, vec![initial], time_span, ())
    }

    fn options(step: f64) -> SolveOptions {
        SolveOptions {
            adaptive: false,
            initial_step: Some(step),
            save: SaveMode::Endpoints,
            ..SolveOptions::default()
        }
    }

    fn endpoint<A: OdeAlgorithm>(algorithm: A, step: f64) -> f64 {
        solve(&problem((0.0, 1.0), 1.0), algorithm, &options(step))
            .unwrap()
            .last_state()[0]
    }

    fn order<A: OdeAlgorithm + Copy>(algorithm: A) -> f64 {
        let exact = 2.0 * std::f64::consts::E - 2.0;
        let coarse = (endpoint(algorithm, 0.1) - exact).abs();
        let fine = (endpoint(algorithm, 0.05) - exact).abs();
        (coarse / fine).log2()
    }

    #[test]
    fn methods_recover_their_design_orders() {
        assert!(order(Ork256) > 1.9);
        assert!(order(Dglddrk73C) > 2.9);
        for (name, observed) in [
            ("CarpenterKennedy2N54", order(CarpenterKennedy2N54)),
            ("DGLDDRK84_C", order(Dglddrk84C)),
            ("DGLDDRK84_F", order(Dglddrk84F)),
            ("NDBLSRK124", order(Ndblsrk124)),
            ("NDBLSRK134", order(Ndblsrk134)),
            ("NDBLSRK144", order(Ndblsrk144)),
        ] {
            assert!(observed > 3.75, "{name} observed order was {observed}");
        }

        // The pinned upstream suite marks SHLDDRK64's order checks broken due
        // to the published coefficients' limited precision. Keep its exact
        // recurrence covered without asserting an order upstream cannot meet.
        assert!(endpoint(Shlddrk64, 0.01).is_finite());
    }

    #[test]
    fn callbacks_save_at_and_backward_integration_use_shared_semantics() {
        let backward = problem((1.0, 0.0), 2.0 * std::f64::consts::E - 2.0);
        let backward_options = SolveOptions {
            adaptive: false,
            initial_step: Some(0.01),
            save_at: vec![1.0, 0.5, 0.0],
            ..SolveOptions::default()
        };
        let solution = solve(&backward, CarpenterKennedy2N54, &backward_options).unwrap();
        assert_eq!(solution.times(), &[1.0, 0.5, 0.0]);
        assert!((solution.last_state()[0] - 1.0).abs() < 1.0e-8);

        let terminating = problem((0.0, 1.0), 1.0)
            .with_continuous_callback(|_, _, time| time - 0.5, |_, _, _| CallbackAction::Terminate);
        let solution = solve(&terminating, Dglddrk73C, &options(0.1)).unwrap();
        assert!((solution.times().last().unwrap() - 0.5).abs() < 1.0e-14);
        assert_eq!(solution.stats().callback_invocations, 1);
    }

    #[test]
    fn malformed_recurrence_is_rejected_before_driver_dispatch() {
        struct MalformedRecurrence;

        impl super::LowStorage2N for MalformedRecurrence {
            const A: &'static [f64] = &[0.0];
            const B: &'static [f64] = &[1.0];
            const C: &'static [f64] = &[0.0];
        }

        assert_eq!(
            integrate::<_, _, MalformedRecurrence>(&problem((0.0, 1.0), 1.0), &options(0.1))
                .unwrap_err(),
            SolveError::InvalidTableau
        );
    }

    #[test]
    fn terminating_callbacks_do_not_trigger_post_effect_rhs_work() {
        let rhs_calls = Rc::new(Cell::new(0));
        let rhs_counter = Rc::clone(&rhs_calls);
        let problem = OdeProblem::new(
            move |derivative: &mut [f64], state: &[f64], _: &(), _: f64| {
                rhs_counter.set(rhs_counter.get() + 1);
                derivative[0] = state[0];
            },
            vec![1.0],
            (0.0, 1.0),
            (),
        )
        .with_discrete_callback(
            |_, _, time| time >= 0.25,
            |_, _, _| CallbackAction::Terminate,
        );
        let solution = solve(&problem, Dglddrk73C, &options(0.25)).unwrap();
        assert_eq!(solution.stats().rhs_evaluations, 7);
        assert_eq!(rhs_calls.get(), 7);

        let initial_rhs_calls = Rc::new(Cell::new(0));
        let initial_rhs_counter = Rc::clone(&initial_rhs_calls);
        let initially_terminating = OdeProblem::new(
            move |derivative: &mut [f64], state: &[f64], _: &(), _: f64| {
                initial_rhs_counter.set(initial_rhs_counter.get() + 1);
                derivative[0] = state[0];
            },
            vec![1.0],
            (0.0, 1.0),
            (),
        )
        .with_discrete_callback(|_, _, _| true, |_, _, _| CallbackAction::Terminate);
        let solution = solve(&initially_terminating, Ork256, &options(0.25)).unwrap();
        assert_eq!(solution.stats().rhs_evaluations, 0);
        assert_eq!(initial_rhs_calls.get(), 0);
    }
}
