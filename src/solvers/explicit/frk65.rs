//! Fitted zero-dissipation sixth-order Runge--Kutta method (FRK65).
use crate::integrator::{KernelCapabilities, StepEstimate, StepKernel, integrate};
use crate::solution::{BorrowedHermiteSegment, TrajectoryRecorder};
use crate::{OdeAlgorithm, OdeProblem, Solution, SolveError, SolveOptions, SolverStats};

/// Fitted Runge--Kutta method of order six (embedded order five).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frk65 {
    pub omega: f64,
}
impl Frk65 {
    pub const fn new(omega: f64) -> Self {
        Self { omega }
    }
}
impl Default for Frk65 {
    fn default() -> Self {
        Self { omega: 0.0 }
    }
}
impl OdeAlgorithm for Frk65 {
    fn solve<F, P>(
        &self,
        problem: &OdeProblem<F, P>,
        options: &SolveOptions,
    ) -> Result<Solution, SolveError>
    where
        F: Fn(&mut [f64], &[f64], &P, f64),
    {
        integrate(
            problem,
            options,
            Frk65Kernel::new(problem.initial_state().len(), self.omega),
        )
    }
}

const C: [f64; 9] = [
    0.0,
    1.0 / 89.0,
    34.0 / 377.0,
    51.0 / 377.0,
    14497158.0 / 33407747.0,
    9744566553.0 / 16002998914.0,
    330.0 / 383.0,
    1.0,
    1.0,
];
const A: [&[f64]; 9] = [
    &[],
    &[1.0 / 89.0],
    &[-38624.0 / 142129.0, 51442.0 / 142129.0],
    &[51.0 / 1508.0, 0.0, 153.0 / 1508.0],
    &[
        3259284578.0 / 3517556363.0,
        0.0,
        -69727055112.0 / 19553806387.0,
        36230363390.0 / 11788838981.0,
    ],
    &[
        -108363632681.0 / 45875676369.0,
        0.0,
        80902506271.0 / 8700424616.0,
        -120088218786.0 / 17139312481.0,
        4533285649.0 / 6676940598.0,
    ],
    &[
        7137368591.0 / 11299833148.0,
        0.0,
        -33088067061.0 / 10572251159.0,
        11481363823.0 / 3650030081.0,
        -4096673444.0 / 7349814937.0,
        9911918171.0 / 12847192605.0,
    ],
    &[
        8898824396.0 / 9828950919.0,
        0.0,
        25673454973.0 / 11497947835.0,
        -74239028301.0 / 15737704666.0,
        222688842816.0 / 44196813415.0,
        -105204445705.0 / 30575217706.0,
        8799291910.0 / 8966990271.0,
    ],
    &[
        1026331676.0 / 33222204855.0,
        0.0,
        0.0,
        1450675392.0 / 5936579813.0,
        4617877550.0 / 16762182457.0,
        1144867463.0 / 6520294355.0,
        1822809703.0 / 7599996644.0,
        79524953.0 / 2351253316.0,
    ],
];
const B1: f64 = 1026331676.0 / 33222204855.0;
const B7: f64 = 1822809703.0 / 7599996644.0;
const B8: f64 = 79524953.0 / 2351253316.0;
const ET: [f64; 9] = [
    413034411.0 / 13925408836.0,
    0.0,
    0.0,
    1865954212.0 / 7538591735.0,
    4451980162.0 / 16576017119.0,
    1157843020.0 / 6320223511.0,
    802708729.0 / 3404369569.0,
    -251398161.0 / 17050111121.0,
    1.0 / 20.0,
];
const D: [f64; 13] = [
    140209127.0 / 573775965.0,
    -8530039.0 / 263747097.0,
    -308551.0 / 104235790.0,
    233511.0 / 333733259.0,
    9126.0 / 184950985.0,
    22.0 / 50434083.0,
    19.0 / 424427471.0,
    -28711.0 / 583216934059.0,
    -3831531.0 / 316297807.0,
    551767.0 / 187698280.0,
    9205.0 / 210998423.0,
    -250.0 / 519462673.0,
    67.0 / 327513887.0,
];
const E: [f64; 11] = [
    437217689.0 / 1587032700.0,
    -15824413.0 / 592362279.0,
    -1563775.0 / 341846569.0,
    270497.0 / 369611210.0,
    -26623.0 / 453099487.0,
    -616297487849.0,
    -47682337.0 / 491732789.0,
    -4778275.0 / 287766311.0,
    641177.0 / 265376522.0,
    44633.0 / 291742143.0,
    611.0 / 223639880.0,
];
const F: [f64; 11] = [
    44861261.0 / 255495624.0,
    -11270940.0 / 352635157.0,
    -182222.0 / 232874507.0,
    164263.0 / 307215200.0,
    32184.0 / 652060417.0,
    -352.0 / 171021903.0,
    -18395427.0 / 101056291.0,
    -621686.0 / 139501937.0,
    2030024.0 / 612171255.0,
    -711049.0 / 7105160932.0,
    267.0 / 333462710.0,
];

struct Frk65Kernel {
    dim: usize,
    stages: Vec<f64>,
    tmp: Vec<f64>,
    err: Vec<f64>,
    omega: f64,
    fsal_current: bool,
}
impl Frk65Kernel {
    fn new(dim: usize, omega: f64) -> Self {
        Self {
            dim,
            stages: vec![0.0; 9 * dim],
            tmp: vec![0.0; dim],
            err: vec![0.0; dim],
            omega,
            fsal_current: false,
        }
    }
    fn stage(&self, i: usize) -> &[f64] {
        &self.stages[i * self.dim..(i + 1) * self.dim]
    }
    fn stage_mut(&mut self, i: usize) -> &mut [f64] {
        &mut self.stages[i * self.dim..(i + 1) * self.dim]
    }
    fn coeffs(&self, dt: f64) -> (f64, f64, f64) {
        let x = (self.omega * dt).powi(2);
        let p = |v: &[f64], n: usize, den: usize| {
            let mut z = v[n - 1];
            for j in (0..n - 1).rev() {
                z = v[j] + x * z;
            }
            z / (1.0 + x * v[den - 1])
        };
        (p(&D, 7, 13), p(&E, 6, 11), p(&F, 6, 11))
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
                for (k, a) in A[i].iter().enumerate() {
                    v += h * a * self.stage(k)[j];
                }
                self.tmp[j] = v;
            }
            eval(p, &mut self.err, &self.tmp, t + C[i] * h, st);
            let start = i * self.dim;
            self.stages[start..start + self.dim].copy_from_slice(&self.err);
        }
        let (b4, b5, b6) = self.coeffs(h);
        for j in 0..self.dim {
            c[j] = s[j]
                + h * (B1 * self.stage(0)[j]
                    + b4 * self.stage(3)[j]
                    + b5 * self.stage(4)[j]
                    + b6 * self.stage(5)[j]
                    + B7 * self.stage(6)[j]
                    + B8 * self.stage(7)[j]);
        }
        eval(p, self.stage_mut(8), c, t + h, st);
        self.fsal_current = false;
        let mut e = 0.0;
        if o.adaptive {
            for j in 0..self.dim {
                // Upstream forms `utilde = dt * beta_tilde*k` and measures
                // `u - uprev - utilde`; this is the main-minus-embedded
                // difference, not beta_tilde by itself.
                let z = (B1 - ET[0]) * self.stage(0)[j]
                    + (b4 - ET[3]) * self.stage(3)[j]
                    + (b5 - ET[4]) * self.stage(4)[j]
                    + (b6 - ET[5]) * self.stage(5)[j]
                    + (B7 - ET[6]) * self.stage(6)[j]
                    + (B8 - ET[7]) * self.stage(7)[j]
                    - ET[8] * self.stage(8)[j];
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
