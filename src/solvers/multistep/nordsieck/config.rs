use crate::ConfigurationError;

/// Adaptive fifth-order fixed-leading-coefficient Adams method in Nordsieck form.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AN5;

/// Equation family used by [`JVODE`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum JvodeMethod {
    /// Variable-order Adams corrector (orders 1 through 12).
    #[default]
    Adams,
    /// Variable-order BDF corrector (orders 1 through 5).
    Bdf,
}

/// Variable-order, variable-step Nordsieck integrator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JVODE {
    pub(super) method: JvodeMethod,
    pub(super) bias1: f64,
    pub(super) bias2: f64,
    pub(super) bias3: f64,
    pub(super) addon: f64,
    pub(super) minimum_factor: f64,
    pub(super) maximum_factor: f64,
    pub(super) steady_maximum: f64,
}

impl JVODE {
    /// Constructs JVODE for the selected equation family.
    pub const fn new(method: JvodeMethod) -> Self {
        Self {
            method,
            bias1: 6.0,
            bias2: 6.0,
            bias3: 10.0,
            addon: 1.0e-6,
            minimum_factor: 0.2,
            maximum_factor: 10.0,
            steady_maximum: 1.5,
        }
    }

    /// Constructs the variable-order Adams configuration.
    pub const fn adams() -> Self {
        Self::new(JvodeMethod::Adams)
    }

    /// Constructs the variable-order BDF configuration.
    pub const fn bdf() -> Self {
        Self::new(JvodeMethod::Bdf)
    }

    /// Returns the configured equation family.
    pub const fn method(&self) -> JvodeMethod {
        self.method
    }

    /// Returns the lower-order, current-order, and higher-order selection biases.
    pub const fn biases(&self) -> (f64, f64, f64) {
        (self.bias1, self.bias2, self.bias3)
    }

    /// Returns the minimum and maximum adaptive step factors.
    pub const fn step_factors(&self) -> (f64, f64) {
        (self.minimum_factor, self.maximum_factor)
    }

    /// Sets the lower-order, current-order, and higher-order selection biases.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigurationError::InvalidParameter`] unless every bias is
    /// finite and positive.
    pub fn with_biases(
        mut self,
        lower: f64,
        current: f64,
        higher: f64,
    ) -> Result<Self, ConfigurationError> {
        if !lower.is_finite()
            || !current.is_finite()
            || !higher.is_finite()
            || lower <= 0.0
            || current <= 0.0
            || higher <= 0.0
        {
            return Err(ConfigurationError::InvalidParameter {
                parameter: "JVODE order-selection biases",
                reason: "biases must be finite and positive",
            });
        }
        self.bias1 = lower;
        self.bias2 = current;
        self.bias3 = higher;
        Ok(self)
    }

    /// Sets the minimum and maximum adaptive step factors.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigurationError::InvalidBounds`] unless both factors are
    /// finite and positive and `minimum <= maximum`.
    pub fn with_step_factors(
        mut self,
        minimum: f64,
        maximum: f64,
    ) -> Result<Self, ConfigurationError> {
        if !minimum.is_finite() || !maximum.is_finite() || minimum <= 0.0 || maximum < minimum {
            return Err(ConfigurationError::InvalidBounds {
                context: "JVODE adaptive step factors",
                reason: "factors must be finite, positive, and ordered",
            });
        }
        self.minimum_factor = minimum;
        self.maximum_factor = maximum;
        Ok(self)
    }
}

impl Default for JVODE {
    fn default() -> Self {
        Self::adams()
    }
}

/// Configured Adams alias for [`JVODE`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct JvodeAdams;

/// Configured BDF alias for [`JVODE`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct JvodeBdf;

#[allow(non_camel_case_types)]
/// Exact OrdinaryDiffEq-compatible spelling alias for [`JvodeAdams`].
pub type JVODE_Adams = JvodeAdams;
#[allow(non_camel_case_types)]
/// Exact OrdinaryDiffEq-compatible spelling alias for [`JvodeBdf`].
pub type JVODE_BDF = JvodeBdf;
