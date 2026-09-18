pub(super) const SPECTRAL_ITERATIONS: usize = 12;
pub(super) const SPECTRAL_SAFETY: f64 = 1.2;
pub(super) const MAX_POLYNOMIAL_STAGES: usize = 200;
const MAX_RKMC2_STAGES: usize = 1_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum StabilizedFamily {
    Rkc,
    Rock2,
    Rock4,
    Serk2,
    Eserk4,
    Eserk5,
    Tsrkc2,
    Tsrkc3,
    Rkl1,
    Rkl2,
    Rkg1,
    Rkg2,
    Rkmc2,
}

impl StabilizedFamily {
    pub(super) const fn order(self) -> usize {
        match self {
            Self::Rkl1 | Self::Rkg1 => 1,
            Self::Rkc
            | Self::Rock2
            | Self::Serk2
            | Self::Tsrkc2
            | Self::Rkl2
            | Self::Rkg2
            | Self::Rkmc2 => 2,
            Self::Rock4 | Self::Eserk4 => 4,
            Self::Eserk5 => 5,
            Self::Tsrkc3 => 3,
        }
    }

    pub(super) fn stages(self, scaled_radius: f64) -> Option<usize> {
        let scaled_radius = scaled_radius.max(0.0);
        let stages = match self {
            Self::Rkc => ((1.54 * scaled_radius + 1.0).sqrt().floor() as usize + 1)
                .clamp(2, MAX_POLYNOMIAL_STAGES),
            Self::Rock2
            | Self::Rock4
            | Self::Serk2
            | Self::Eserk4
            | Self::Eserk5
            | Self::Tsrkc2
            | Self::Tsrkc3 => return None,
            Self::Rkl1 => {
                odd_stage_count((((1.0 + 4.0 * scaled_radius).sqrt() - 1.0) / 2.0).ceil() as usize)
            }
            Self::Rkl2 => {
                odd_stage_count((((9.0 + 8.0 * scaled_radius).sqrt() - 1.0) / 2.0).ceil() as usize)
            }
            Self::Rkg1 => (((9.0 + 16.0 * scaled_radius).sqrt() - 3.0) / 2.0)
                .ceil()
                .max(2.0) as usize,
            Self::Rkg2 => (((25.0 + 24.0 * scaled_radius).sqrt() - 3.0) / 2.0)
                .ceil()
                .max(3.0) as usize,
            Self::Rkmc2 => (-0.830_678_217_871_279_5
                + 1.854_788_782_583_655_3 * scaled_radius.powf(0.533_871_357_807_877))
            .ceil()
            .max(3.0) as usize,
        };
        Some(stages.min(if self == Self::Rkmc2 {
            MAX_RKMC2_STAGES
        } else {
            MAX_POLYNOMIAL_STAGES
        }))
    }
}

fn odd_stage_count(stages: usize) -> usize {
    let stages = stages.max(3);
    if stages % 2 == 0 { stages + 1 } else { stages }.min(MAX_POLYNOMIAL_STAGES - 1)
}
