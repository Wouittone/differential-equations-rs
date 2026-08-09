use crate::{Dp5, OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions};

/// Automatic low-order Dormand--Prince composite facade.
///
/// OrdinaryDiffEq defines `AutoDP5(stiff_alg)` as
/// `AutoAlgSwitch(DP5(), stiff_alg)`, which dynamically switches between the
/// non-stiff DP5 method and the supplied stiff method.  The regular ODE driver
/// currently has no stiffness detector or in-flight algorithm-switch state, so
/// this facade preserves the native DP5 component's fixed/adaptive, callback,
/// and dense-output semantics while retaining the requested stiff component
/// for API compatibility.  Automatic switching is intentionally explicit as a
/// deferred capability rather than silently selecting an external wrapper.
#[derive(Clone, Debug, PartialEq)]
pub struct AutoDp5<A> {
    /// The stiff component requested by the upstream composite constructor.
    pub stiff_algorithm: A,
}

impl<A> AutoDp5<A> {
    /// Constructs an AutoDP5 facade around a stiff component.
    pub const fn new(stiff_algorithm: A) -> Self {
        Self { stiff_algorithm }
    }

    /// Returns the configured stiff component.
    pub const fn stiff_algorithm(&self) -> &A {
        &self.stiff_algorithm
    }
}

impl<A> OdeAlgorithm for AutoDp5<A> {
    fn solve<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        // Keep the field live so this remains a faithful configured facade;
        // switching will consume it once the driver supports that capability.
        let _ = &self.stiff_algorithm;
        Dp5.solve(problem, options)
    }
}

/// Uppercase acronym spelling used by the pinned Julia algorithm name.
#[allow(non_camel_case_types)]
pub type AutoDP5<A> = AutoDp5<A>;
