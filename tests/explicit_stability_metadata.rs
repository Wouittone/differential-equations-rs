use differential_equations::solvers::explicit::{Dp5, Tsit5, Vern6, Vern7, Vern8, Vern9};

#[test]
fn canonical_explicit_tableaus_expose_pinned_sciml_stability_radii() {
    for (name, radius, expected) in [
        (
            "Dp5",
            Dp5.tableau().unwrap().real_stability_radius(),
            3.3066,
        ),
        (
            "Tsit5",
            Tsit5.tableau().unwrap().real_stability_radius(),
            3.5068,
        ),
        (
            "Vern6",
            Vern6.tableau().unwrap().real_stability_radius(),
            4.8553,
        ),
        (
            "Vern7",
            Vern7.tableau().unwrap().real_stability_radius(),
            4.64,
        ),
        (
            "Vern8",
            Vern8.tableau().unwrap().real_stability_radius(),
            5.8641,
        ),
        (
            "Vern9",
            Vern9.tableau().unwrap().real_stability_radius(),
            4.4762,
        ),
    ] {
        assert_eq!(radius, Some(expected), "{name} stability radius");
    }
}
