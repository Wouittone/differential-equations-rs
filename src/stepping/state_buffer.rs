use std::ops::{Deref, DerefMut};

/// Accepted state storage. Scratch candidates remain separate until acceptance.
#[derive(Debug)]
pub(super) enum StateBuffer<'a> {
    Owned(Vec<f64>),
    Borrowed(&'a mut [f64]),
}

impl Deref for StateBuffer<'_> {
    type Target = [f64];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(values) => values,
            Self::Borrowed(values) => values,
        }
    }
}

impl DerefMut for StateBuffer<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Owned(values) => values,
            Self::Borrowed(values) => values,
        }
    }
}
