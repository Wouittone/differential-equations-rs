use crate::tableau::{MisTableau, MriTableau};

/// Extrapolation sequence used by [`super::algorithms::MREEF`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MultirateSequence {
    /// `n_j = j`, matching upstream's `:harmonic` default.
    #[default]
    Harmonic,
    /// `n_j = 2^(j-1)`, matching upstream's `:romberg` option.
    Romberg,
}

#[derive(Clone, Copy)]
pub(super) enum Method {
    Mreef {
        m: usize,
        order: usize,
        sequence: MultirateSequence,
    },
    Mrab {
        m: usize,
        order: usize,
    },
    Mis {
        m: usize,
        tableau: &'static MisTableau,
    },
    Mri {
        m: usize,
        tableau: &'static MriTableau,
    },
}

impl Method {
    pub(super) fn controller_order(self) -> usize {
        match self {
            Self::Mreef { order, .. } | Self::Mrab { order, .. } => order,
            Self::Mis { tableau, .. } => tableau.order(),
            Self::Mri { tableau, .. } => tableau.order(),
        }
    }
}
