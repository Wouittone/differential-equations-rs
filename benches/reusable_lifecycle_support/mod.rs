//! Shared workload, RHS, and metadata helpers for the `reusable_lifecycle`
//! and `reusable_lifecycle_diagnostics` benchmark binaries.
//!
//! This module is included (via `#[path]`) into both binaries rather than
//! published as a library item, so each binary can independently choose
//! whether to install `StatsAlloc` as the global allocator: the timing
//! binary must not pay per-allocation counter overhead on every measured
//! iteration, while the diagnostics binary needs it to report allocation
//! counts.
use differential_equations::stepping::{
    AdaptiveController, ControllerConfig, ExplicitRungeKuttaStepper, ObserverAction, integrate_rk,
};
use std::convert::Infallible;

pub const CONTROLLER_TOLERANCE: f64 = 1.0e-10;

/// Compiler version captured by invoking the toolchain reported at build
/// time (falling back to `rustc` on `PATH`); environment variables such as
/// `RUSTC` are cargo-internal and are not propagated to this process.
pub fn compiler_version() -> String {
    std::process::Command::new(option_env!("RUSTC").unwrap_or("rustc"))
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|version| version.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Best-effort CPU model/identifier, tried through the mechanism available
/// on each platform, with an explicit `unknown` fallback rather than an
/// empty or misleading value when none of them succeed.
pub fn cpu_identifier() -> String {
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/cpuinfo") {
            if let Some(model) = contents
                .lines()
                .find(|line| line.starts_with("model name"))
                .and_then(|line| line.split_once(':'))
                .map(|(_, value)| value.trim().to_string())
                .filter(|value| !value.is_empty())
            {
                return model;
            }
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
        {
            if output.status.success() {
                if let Some(model) = String::from_utf8(output.stdout)
                    .ok()
                    .map(|value| value.trim().to_string())
                    .filter(|value| !value.is_empty())
                {
                    return model;
                }
            }
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Some(identifier) = std::env::var("PROCESSOR_IDENTIFIER")
            .ok()
            .filter(|value| !value.is_empty())
        {
            return identifier;
        }
    }
    "unknown".to_string()
}

/// Metadata printed once per benchmarked configuration. `output_policy`
/// documents that lane's own retention behavior rather than a single
/// process-wide constant, since the suite intentionally exercises several
/// different output policies (endpoint-only, endpoint-and-sampled, and
/// every-accepted-step retention).
pub fn benchmark_metadata(output_policy: &str) -> String {
    format!(
        "reusable_lifecycle crate={} version={} target={}-{} cpu={} compiler={} tolerance={} controller=proportional(5) output={}",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        std::env::consts::ARCH,
        std::env::consts::OS,
        cpu_identifier(),
        compiler_version(),
        CONTROLLER_TOLERANCE,
        output_policy,
    )
}

pub fn scalar_rhs(_: f64, state: &[f64], derivative: &mut [f64]) -> Result<(), Infallible> {
    derivative[0] = -state[0];
    Ok(())
}

pub fn scalar_norm(_: &[f64], _: &[f64], error: &[f64]) -> Result<f64, Infallible> {
    Ok(error[0].abs() / CONTROLLER_TOLERANCE)
}

pub fn orbit_velocity_rhs(
    gravitational_parameter: f64,
    drag: f64,
) -> impl FnMut(f64, &[f64], &mut [f64]) -> Result<(), Infallible> {
    move |_: f64, state: &[f64], derivative: &mut [f64]| {
        let (x, y, z, vx, vy, vz) = (state[0], state[1], state[2], state[3], state[4], state[5]);
        let radius = (x * x + y * y + z * z).sqrt().max(1.0e-12);
        let radius_cubed = radius * radius * radius;
        let speed = (vx * vx + vy * vy + vz * vz).sqrt();
        let acceleration_scale = gravitational_parameter / radius_cubed;
        derivative[0] = vx;
        derivative[1] = vy;
        derivative[2] = vz;
        derivative[3] = -acceleration_scale * x - drag * speed * vx;
        derivative[4] = -acceleration_scale * y - drag * speed * vy;
        derivative[5] = -acceleration_scale * z - drag * speed * vz;
        Ok(())
    }
}

pub fn orbit_norm(_: &[f64], _: &[f64], error: &[f64]) -> Result<f64, Infallible> {
    Ok(error
        .iter()
        .copied()
        .fold(0.0_f64, |worst, component| worst.max(component.abs()))
        / CONTROLLER_TOLERANCE)
}

pub fn run_arc<F, N>(
    stepper: &mut ExplicitRungeKuttaStepper<'_>,
    controller: &mut AdaptiveController,
    endpoint: f64,
    rhs: &mut F,
    norm: &mut N,
) where
    F: FnMut(f64, &[f64], &mut [f64]) -> Result<(), Infallible>,
    N: FnMut(&[f64], &[f64], &[f64]) -> Result<f64, Infallible>,
{
    integrate_rk(
        stepper,
        controller,
        endpoint,
        &[],
        100_000,
        rhs,
        norm,
        &mut |_| Ok::<_, Infallible>(ObserverAction::Continue),
    )
    .expect("reusable benchmark arc must solve");
}

/// Six-state initial condition shared by the orbit-workload lanes so setup
/// costs stay comparable across solver families. With `gravitational_parameter
/// = 1.0` and unit radius, the nonzero tangential velocity component
/// (`vy = 1.0`) makes this a genuine (drag-decaying) circular orbit rather
/// than radial free-fall.
pub const ORBIT_INITIAL_STATE: [f64; 6] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

/// High-accuracy reference endpoint for the velocity-coupled orbit, computed
/// once with a much tighter tolerance than the measured lanes. Diagnostics
/// compare their endpoint against this to report a meaningful error even
/// though the drag term has no closed-form solution.
///
/// This module is `#[path]`-included separately into both the
/// `reusable_lifecycle` (timing) and `reusable_lifecycle_diagnostics`
/// binaries; only the latter uses this helper, so it is allowed to be
/// unused in the former rather than duplicating the module per binary.
#[allow(dead_code)]
pub fn reference_orbit_endpoint(
    tableau: &differential_equations::tableau::RungeKuttaTableau,
    endpoint: f64,
    gravitational_parameter: f64,
    drag: f64,
) -> [f64; 6] {
    let mut state = ORBIT_INITIAL_STATE;
    let mut stepper = ExplicitRungeKuttaStepper::from_buffer(tableau, 0.0, &mut state).unwrap();
    let mut controller =
        AdaptiveController::new(ControllerConfig::proportional(5).unwrap(), 0.1).unwrap();
    let mut rhs = orbit_velocity_rhs(gravitational_parameter, drag);
    let mut norm = |_: &[f64], _: &[f64], error: &[f64]| -> Result<f64, Infallible> {
        Ok(error
            .iter()
            .copied()
            .fold(0.0_f64, |worst, component| worst.max(component.abs()))
            / 1.0e-13)
    };
    run_arc(&mut stepper, &mut controller, endpoint, &mut rhs, &mut norm);
    state
}
