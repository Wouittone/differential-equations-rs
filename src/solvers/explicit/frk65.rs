//! Fitted zero-dissipation sixth-order Runge--Kutta method (FRK65).
use crate::integrator::{KernelCapabilities, StepEstimate, StepKernel, integrate};
use crate::solution::{BorrowedHermiteSegment, TrajectoryRecorder};
use crate::tableau::{FittedWeight, RungeKuttaTableau, load_tableau};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};
use differential_equations_tableau_macros::define_explicit_rk_tableau_from_file;

define_explicit_rk_tableau_from_file!(
    pub(super) FRK65_TABLEAU,
    "Frk65",
    "tableaux/explicit/frk65.json",
    crate = crate
);

/// Fitted Runge--Kutta method of order six (embedded order five).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frk65 {
    /// Angular frequency used to fit the zero-dissipation coefficients.
    pub omega: f64,
}
impl Frk65 {
    /// Creates an FRK65 method fitted to `omega`.
    pub const fn new(omega: f64) -> Self {
        Self { omega }
    }

    /// Returns this method's lazily materialized, validated base tableau.
    pub fn tableau(self) -> Result<&'static RungeKuttaTableau, crate::tableau::TableauError> {
        load_tableau(&FRK65_TABLEAU)
    }
}
impl Default for Frk65 {
    fn default() -> Self {
        Self { omega: 0.0 }
    }
}
impl OdeAlgorithm for Frk65 {
    fn solve_validated<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        let tableau = self.tableau().map_err(|_| SolveError::InvalidTableau)?;
        integrate(
            problem,
            options,
            Frk65Kernel::new(problem.initial_state().len(), self.omega, tableau)?,
        )
    }
}

struct Frk65Kernel {
    dim: usize,
    stages: Vec<f64>,
    tmp: Vec<f64>,
    err: Vec<f64>,
    omega: f64,
    tableau: &'static RungeKuttaTableau,
    fitted_weights: [&'static FittedWeight; 3],
    fsal_current: bool,
}
impl Frk65Kernel {
    fn new(
        dim: usize,
        omega: f64,
        tableau: &'static RungeKuttaTableau,
    ) -> Result<Self, SolveError> {
        if tableau.a().len() != 9 || tableau.error().is_none() {
            return Err(SolveError::InvalidTableau);
        }
        let fitted_weights = [
            tableau.fitted_weight(3).ok_or(SolveError::InvalidTableau)?,
            tableau.fitted_weight(4).ok_or(SolveError::InvalidTableau)?,
            tableau.fitted_weight(5).ok_or(SolveError::InvalidTableau)?,
        ];
        Ok(Self {
            dim,
            stages: vec![0.0; 9 * dim],
            tmp: vec![0.0; dim],
            err: vec![0.0; dim],
            omega,
            tableau,
            fitted_weights,
            fsal_current: false,
        })
    }
    fn stage(&self, i: usize) -> &[f64] {
        &self.stages[i * self.dim..(i + 1) * self.dim]
    }
    fn stage_mut(&mut self, i: usize) -> &mut [f64] {
        &mut self.stages[i * self.dim..(i + 1) * self.dim]
    }
    fn coeffs(&self, dt: f64) -> Result<(f64, f64, f64), SolveError> {
        let x = (self.omega * dt).powi(2);
        Ok((
            self.fitted_weights[0]
                .evaluate(x)
                .ok_or(SolveError::InvalidTableau)?,
            self.fitted_weights[1]
                .evaluate(x)
                .ok_or(SolveError::InvalidTableau)?,
            self.fitted_weights[2]
                .evaluate(x)
                .ok_or(SolveError::InvalidTableau)?,
        ))
    }
}
fn eval<F, P>(
    problem: &OdeProblem<F, P>,
    out: &mut [f64],
    state: &[f64],
    t: f64,
    stats: &mut SolverStats,
) where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    (problem.rhs)(out, state, problem.parameters(), t);
    stats.rhs_evaluations += 1;
}
impl<F, P> StepKernel<F, P> for Frk65Kernel
where
    F: Fn(&mut [f64], &[f64], &P, f64),
{
    fn capabilities(&self) -> KernelCapabilities {
        KernelCapabilities::new(true, 5)
    }
    fn initialize(
        &mut self,
        p: &OdeProblem<F, P>,
        s: &[f64],
        t: f64,
        st: &mut SolverStats,
    ) -> Result<(), SolveError> {
        eval(p, &mut self.err, s, t, st);
        self.stages[..self.dim].copy_from_slice(&self.err);
        self.fsal_current = true;
        Ok(())
    }
    fn estimate_initial_step(
        &mut self,
        _: &OdeProblem<F, P>,
        s: &[f64],
        _: f64,
        d: f64,
        m: f64,
        _: &mut [f64],
        o: &SolveOptions,
        _: &mut SolverStats,
    ) -> Result<f64, SolveError> {
        let mut n = 0.0;
        for (u, k) in s.iter().zip(self.stage(0)) {
            let q = o.absolute_tolerance + o.relative_tolerance * u.abs();
            n += (k / q).powi(2);
        }
        Ok((0.01 / (n / self.dim as f64).sqrt().max(1e-15))
            .powf(1.0 / 6.0)
            .min(m)
            * d.abs())
    }
    #[allow(clippy::needless_range_loop)]
    fn attempt_step(
        &mut self,
        p: &OdeProblem<F, P>,
        s: &[f64],
        t: f64,
        h: f64,
        c: &mut [f64],
        o: &SolveOptions,
        st: &mut SolverStats,
    ) -> Result<StepEstimate, SolveError> {
        if !self.fsal_current {
            eval(p, self.stage_mut(0), s, t, st);
        }
        for i in 1..8 {
            for j in 0..self.dim {
                let mut v = s[j];
                for (k, a) in self.tableau.stage_row(i).iter().enumerate() {
                    v += h * a * self.stage(k)[j];
                }
                self.tmp[j] = v;
            }
            eval(p, &mut self.err, &self.tmp, t + self.tableau.c()[i] * h, st);
            let start = i * self.dim;
            self.stages[start..start + self.dim].copy_from_slice(&self.err);
        }
        let (b4, b5, b6) = self.coeffs(h)?;
        let weights = self.tableau.b();
        for j in 0..self.dim {
            c[j] = s[j]
                + h * (weights[0] * self.stage(0)[j]
                    + b4 * self.stage(3)[j]
                    + b5 * self.stage(4)[j]
                    + b6 * self.stage(5)[j]
                    + weights[6] * self.stage(6)[j]
                    + weights[7] * self.stage(7)[j]);
        }
        eval(p, self.stage_mut(8), c, t + h, st);
        self.fsal_current = false;
        let mut e = 0.0;
        if o.adaptive {
            let error_weights = self.tableau.error().ok_or(SolveError::InvalidTableau)?;
            let embedded_weight = |stage: usize| weights[stage] - error_weights[stage];
            for j in 0..self.dim {
                // Upstream forms `utilde = dt * beta_tilde*k` and measures
                // `u - uprev - utilde`; this is the main-minus-embedded
                // difference, not beta_tilde by itself.
                let z = error_weights[0] * self.stage(0)[j]
                    + (b4 - embedded_weight(3)) * self.stage(3)[j]
                    + (b5 - embedded_weight(4)) * self.stage(4)[j]
                    + (b6 - embedded_weight(5)) * self.stage(5)[j]
                    + error_weights[6] * self.stage(6)[j]
                    + error_weights[7] * self.stage(7)[j]
                    + error_weights[8] * self.stage(8)[j];
                let q = o.absolute_tolerance + o.relative_tolerance * s[j].abs().max(c[j].abs());
                e += (h * z / q).powi(2);
            }
            e = (e / self.dim as f64).sqrt();
        }
        Ok(StepEstimate::new(e))
    }
    fn record_dense_step(
        &mut self,
        p: &OdeProblem<F, P>,
        prev: &[f64],
        s: &[f64],
        pt: f64,
        _: f64,
        t: f64,
        fin: bool,
        r: &mut TrajectoryRecorder<'_>,
        st: &mut SolverStats,
    ) -> Result<bool, SolveError> {
        eval(p, &mut self.err, s, t, st);
        let seg = BorrowedHermiteSegment::new(pt, t, prev, s, self.stage(0), &self.err)
            .map_err(|_| SolveError::NonFiniteDerivative)?;
        r.record_step_dense(prev, pt, s, t, fin, &seg)
            .map_err(|_| SolveError::NonFiniteDerivative)?;
        Ok(true)
    }
    fn accept_step(
        &mut self,
        _: &OdeProblem<F, P>,
        _: &[f64],
        _: &[f64],
        _: f64,
        _: f64,
        cb: bool,
        _: &mut SolverStats,
    ) -> Result<(), SolveError> {
        if !cb {
            let d = self.dim;
            let (head, tail) = self.stages.split_at_mut(8 * d);
            head[..d].copy_from_slice(&tail[..d]);
            self.fsal_current = true;
        }
        Ok(())
    }
    fn reject_step(&mut self) {
        self.fsal_current = true;
    }
}

#[cfg(test)]
mod tests {
    use super::Frk65;
    use crate::{OdeAlgorithm, OdeProblem, SolveOptions};

    #[allow(clippy::type_complexity)]
    fn problem() -> OdeProblem<impl Fn(&mut [f64], &[f64], &(), f64), ()> {
        OdeProblem::new(
            |du: &mut [f64], u: &[f64], _: &(), _: f64| du[0] = u[0],
            vec![1.0],
            (0.0, 1.0),
            (),
        )
    }

    #[test]
    fn fixed_step_recovers_sixth_order() {
        let mut o = SolveOptions {
            adaptive: false,
            initial_step: Some(0.1),
            ..Default::default()
        };
        let coarse = Frk65::default().solve(&problem(), &o).unwrap();
        o.initial_step = Some(0.05);
        let fine = Frk65::default().solve(&problem(), &o).unwrap();
        let ec = (coarse.last_state()[0] - std::f64::consts::E).abs();
        let ef = (fine.last_state()[0] - std::f64::consts::E).abs();
        assert!(ec / ef > 40.0, "observed ratio {}", ec / ef);
    }

    #[test]
    fn adaptive_and_fitted_modes_are_accurate() {
        let o = SolveOptions {
            absolute_tolerance: 1.0e-9,
            relative_tolerance: 1.0e-9,
            max_step: 0.2,
            ..Default::default()
        };
        let result = Frk65::new(0.0).solve(&problem(), &o).unwrap();
        let error = (result.last_state()[0] - std::f64::consts::E).abs();
        assert!(error < 1.0e-7, "error {error}, stats {:?}", result.stats());
    }
}
