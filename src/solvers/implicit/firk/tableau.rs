use std::f64::consts::PI;

#[derive(Clone, Copy)]
pub(super) enum Family {
    Radau,
    Gauss,
}

#[derive(Clone)]
pub(super) struct Tableau {
    pub(super) stages: usize,
    pub(super) order: usize,
    pub(super) a: Vec<f64>,
    pub(super) b: Vec<f64>,
    pub(super) c: Vec<f64>,
    // Ascending coefficients of each Lagrange cardinal polynomial.
    pub(super) lagrange: Vec<f64>,
}

impl Tableau {
    pub(super) fn generate(family: Family, stages: usize) -> Self {
        let c = match family {
            Family::Radau => radau_nodes(stages),
            Family::Gauss => gauss_nodes(stages),
        };
        let mut lagrange = vec![0.0; stages * stages];
        for j in 0..stages {
            let mut coefficients = vec![1.0];
            let mut denominator = 1.0;
            for k in 0..stages {
                if k == j {
                    continue;
                }
                denominator *= c[j] - c[k];
                let mut next = vec![0.0; coefficients.len() + 1];
                for (power, &value) in coefficients.iter().enumerate() {
                    next[power] -= c[k] * value;
                    next[power + 1] += value;
                }
                coefficients = next;
            }
            for power in 0..stages {
                lagrange[j * stages + power] = coefficients[power] / denominator;
            }
        }
        let mut a = vec![0.0; stages * stages];
        let mut b = vec![0.0; stages];
        for j in 0..stages {
            b[j] = integrated_cardinal(&lagrange, stages, j, 1.0);
            for i in 0..stages {
                a[i * stages + j] = integrated_cardinal(&lagrange, stages, j, c[i]);
            }
        }
        Self {
            stages,
            order: match family {
                Family::Radau => 2 * stages - 1,
                Family::Gauss => 2 * stages,
            },
            a,
            b,
            c,
            lagrange,
        }
    }

    pub(super) fn weights_at(&self, theta: f64, output: &mut [f64]) {
        for (j, weight) in output.iter_mut().enumerate().take(self.stages) {
            *weight = integrated_cardinal(&self.lagrange, self.stages, j, theta);
        }
    }
}

fn integrated_cardinal(coefficients: &[f64], stages: usize, row: usize, x: f64) -> f64 {
    let mut power = x;
    let mut value = 0.0;
    for degree in 0..stages {
        value += coefficients[row * stages + degree] * power / (degree + 1) as f64;
        power *= x;
    }
    value
}

fn legendre(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0);
    }
    let mut previous = 1.0;
    let mut current = x;
    for degree in 2..=n {
        let next = ((2 * degree - 1) as f64 * x * current - (degree - 1) as f64 * previous)
            / degree as f64;
        previous = current;
        current = next;
    }
    let derivative = n as f64 * (x * current - previous) / (x * x - 1.0);
    (current, derivative)
}

fn gauss_nodes(stages: usize) -> Vec<f64> {
    let mut nodes = Vec::with_capacity(stages);
    for k in 1..=stages {
        let mut x = (PI * (4 * k - 1) as f64 / (4 * stages + 2) as f64).cos();
        for _ in 0..30 {
            let (value, derivative) = legendre(stages, x);
            let next = x - value / derivative;
            if (next - x).abs() <= 4.0 * f64::EPSILON {
                x = next;
                break;
            }
            x = next;
        }
        nodes.push(0.5 * (x + 1.0));
    }
    nodes.sort_by(f64::total_cmp);
    nodes
}

fn radau_nodes(stages: usize) -> Vec<f64> {
    if stages == 1 {
        return vec![1.0];
    }
    let mut nodes = Vec::with_capacity(stages);
    for k in 0..stages {
        if k == 0 {
            nodes.push(1.0);
            continue;
        }
        let mut x = (2.0 * PI * k as f64 / (2 * stages - 1) as f64).cos();
        for _ in 0..40 {
            let (pn, dpn) = legendre(stages, x);
            let (pm, dpm) = legendre(stages - 1, x);
            let next = x - (pn - pm) / (dpn - dpm);
            if (next - x).abs() <= 8.0 * f64::EPSILON {
                x = next;
                break;
            }
            x = next.clamp(-1.0 + 1.0e-14, 1.0 - 1.0e-14);
        }
        nodes.push(0.5 * (x + 1.0));
    }
    nodes.sort_by(f64::total_cmp);
    nodes
}
