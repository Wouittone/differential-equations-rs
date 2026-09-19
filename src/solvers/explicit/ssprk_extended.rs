//! Strong-stability-preserving Runge--Kutta methods, including fixed-step
//! families and the adaptive SSPRK432 embedded pair.
//!
//! The coefficients below are algebraically equivalent Butcher forms of the
//! Shu--Osher implementations in OrdinaryDiffEqSSPRK.  Keeping them in the
//! shared explicit RK engine gives these methods the same output and callback
//! handling as the other one-step solvers without pretending to expose
//! OrdinaryDiffEq's stage/step limiter or threading options.

// The transformed tableaus retain the full f64 results of combining the
// upstream decimal Shu--Osher coefficients.
#![allow(clippy::excessive_precision)]

mod dense;
mod prrk22;
mod prrk33;
mod prrk54;

#[cfg(test)]
mod tests;

macro_rules! fixed_ssprk {
    ($algorithm:ident, $path:literal) => {
        crate::tableau::define_explicit_rk_from_file!(pub $algorithm, $path, crate = crate);
    };
}

crate::tableau::define_explicit_rk_from_file!(
    pub SspRk432,
    "src/tableau/resources/explicit/ssp_rk432.json",
    crate = crate
);
crate::tableau::define_explicit_rk_from_file!(
    pub SspRk932,
    "src/tableau/resources/explicit/ssp_rk932.json",
    crate = crate
);

fixed_ssprk!(SspRk53, "src/tableau/resources/explicit/ssp_rk53.json");
fixed_ssprk!(
    SspRk53TwoN1,
    "src/tableau/resources/explicit/ssp_rk53_two_n1.json"
);

fixed_ssprk!(
    SspRk53TwoN2,
    "src/tableau/resources/explicit/ssp_rk53_two_n2.json"
);
fixed_ssprk!(SspRk53H, "src/tableau/resources/explicit/ssp_rk53_h.json");
fixed_ssprk!(SspRk63, "src/tableau/resources/explicit/ssp_rk63.json");
fixed_ssprk!(SspRk73, "src/tableau/resources/explicit/ssp_rk73.json");
fixed_ssprk!(SspRk83, "src/tableau/resources/explicit/ssp_rk83.json");
fixed_ssprk!(SspRk54, "src/tableau/resources/explicit/ssp_rk54.json");
fixed_ssprk!(SspRk104, "src/tableau/resources/explicit/ssp_rk104.json");

/// Parametric relaxation SSPRK22. The default `kappa = 0` is the standard
/// fixed-step two-stage SSPRK22 method; nonzero values apply the pinned
/// OrdinaryDiffEqSSPRK coefficient rescaling before each step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prrk22 {
    kappa: f64,
}

impl Default for Prrk22 {
    fn default() -> Self {
        Self { kappa: 0.0 }
    }
}

impl Prrk22 {
    /// Creates the method with relaxation parameter `kappa`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::ConfigurationError::InvalidParameter`] when `kappa`
    /// is not finite.
    pub fn new(kappa: f64) -> Result<Self, crate::ConfigurationError> {
        validate_relaxation(kappa)?;
        Ok(Self { kappa })
    }

    /// Returns the coefficient relaxation parameter.
    pub const fn kappa(&self) -> f64 {
        self.kappa
    }
}

#[allow(non_camel_case_types)]
/// SciML-compatible spelling of [`Prrk22`].
pub type pRRK22 = Prrk22;

/// Parametric relaxation SSPRK33. The default `kappa = 0` is the standard
/// fixed-step three-stage SSPRK33 method; nonzero values apply the pinned
/// OrdinaryDiffEqSSPRK coefficient rescaling before each step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prrk33 {
    kappa: f64,
}

impl Default for Prrk33 {
    fn default() -> Self {
        Self { kappa: 0.0 }
    }
}

impl Prrk33 {
    /// Creates the method with relaxation parameter `kappa`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::ConfigurationError::InvalidParameter`] when `kappa`
    /// is not finite.
    pub fn new(kappa: f64) -> Result<Self, crate::ConfigurationError> {
        validate_relaxation(kappa)?;
        Ok(Self { kappa })
    }

    /// Returns the coefficient relaxation parameter.
    pub const fn kappa(&self) -> f64 {
        self.kappa
    }
}

#[allow(non_camel_case_types)]
/// SciML-compatible spelling of [`Prrk33`].
pub type pRRK33 = Prrk33;

/// Parametric-relaxation SSPRK(5,4) of Spiteri and Ruuth.
///
/// This is the fixed-step `pRRK54` method from OrdinaryDiffEqSSPRK.  The
/// relaxation parameter is applied to the Shu--Osher coefficients at every
/// attempted step; `kappa = 0` is the ordinary SSPRK(5,4) method.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prrk54 {
    kappa: f64,
}

impl Default for Prrk54 {
    fn default() -> Self {
        Self { kappa: 0.0 }
    }
}

impl Prrk54 {
    /// Creates the method with relaxation parameter `kappa`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::ConfigurationError::InvalidParameter`] when `kappa`
    /// is not finite.
    pub fn new(kappa: f64) -> Result<Self, crate::ConfigurationError> {
        validate_relaxation(kappa)?;
        Ok(Self { kappa })
    }

    /// Returns the coefficient relaxation parameter.
    pub const fn kappa(&self) -> f64 {
        self.kappa
    }
}

fn validate_relaxation(kappa: f64) -> Result<(), crate::ConfigurationError> {
    if !kappa.is_finite() {
        return Err(crate::ConfigurationError::InvalidParameter {
            parameter: "pRRK relaxation parameter",
            reason: "the parameter must be finite",
        });
    }
    Ok(())
}

#[allow(non_camel_case_types)]
/// SciML-compatible spelling of [`Prrk54`].
pub type pRRK54 = Prrk54;
