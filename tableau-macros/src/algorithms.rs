#[cfg(test)]
use crate::expansion::expand_multistep_source;
use crate::expansion::{emit_lazy_static, read_static_resource};
use crate::input::{MacroInput, StaticTableauInput};
use differential_equations_tableau_core::{
    RungeKuttaKind, parse_irkn_tableau, parse_low_storage_tableau, parse_rkn_tableau,
    parse_symplectic_tableau, parse_tableau,
};
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use std::path::PathBuf;
use syn::parse_macro_input;
#[cfg(test)]
use syn::{LitStr, parse_quote};

/// Defines a named fixed or adaptive RKN solver from one JSON resource.
///
/// The generated zero-sized type implements `SecondOrderOdeAlgorithm` and
/// exposes its independently lazy tableau through `tableau()`.
pub fn define_rkn_from_file(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as MacroInput);
    let result = read_algorithm_resource(&input)
        .and_then(|source| expand_rkn_algorithm_source(input, &source));
    match result {
        Ok(tokens) => tokens.into(),
        Err(error) => syn::Error::new(proc_macro2::Span::call_site(), error)
            .into_compile_error()
            .into(),
    }
}

/// Defines a named low-storage Runge--Kutta solver from one JSON resource.
///
/// The generated zero-sized type implements [`OdeAlgorithm`](https://docs.rs/differential-equations-rs/latest/differential_equations/trait.OdeAlgorithm.html)
/// and exposes its independently lazy tableau through `tableau()`.
pub fn define_low_storage_rk_from_file(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as MacroInput);
    let result = read_algorithm_resource(&input)
        .and_then(|source| expand_low_storage_rk_algorithm_source(input, &source));
    match result {
        Ok(tokens) => tokens.into(),
        Err(error) => syn::Error::new(proc_macro2::Span::call_site(), error)
            .into_compile_error()
            .into(),
    }
}

fn expand_low_storage_rk_algorithm_source(
    input: MacroInput,
    source: &str,
) -> Result<TokenStream2, String> {
    let tableau = parse_low_storage_tableau(source, &input.name.to_string())
        .map_err(|error| format!("invalid tableau `{}`: {error}", input.path.value()))?;
    let visibility = input.visibility;
    let name = input.name;
    let static_name = format_ident!("__{}_TABLEAU", name.to_string().to_uppercase());
    let source_path = input.path;
    let crate_path = input.crate_path;
    let description = tableau.description();
    Ok(quote! {
        static #static_name: #crate_path::tableau::LazyLowStorageRungeKuttaTableau =
            ::std::sync::LazyLock::new(|| {
                #crate_path::tableau::parse_low_storage_tableau(
                    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/", #source_path)),
                    stringify!(#name),
                )
            });

        #[doc = #description]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        #[allow(
            non_camel_case_types,
            reason = "preserve the published low-storage algorithm name"
        )]
        #visibility struct #name;

        impl #name {
            /// Returns the lazily initialized, compile-time-validated tableau.
            #visibility fn tableau(
                self,
            ) -> ::std::result::Result<
                &'static #crate_path::tableau::LowStorageRungeKuttaTableau,
                #crate_path::tableau::TableauError,
            > {
                #crate_path::tableau::load_tableau(&#static_name)
            }
        }

        impl #crate_path::OdeAlgorithm for #name {
            fn solve_validated<F, P>(
                &self,
                problem: &#crate_path::OdeProblem<F, P>,
                options: &#crate_path::SolveOptions,
            ) -> ::std::result::Result<#crate_path::Solution, #crate_path::SolveError>
            where
                F: #crate_path::OdeFunction<P>,
            {
                #crate_path::OdeAlgorithm::solve_validated(
                    &#crate_path::solvers::explicit::ResourceLowStorageRungeKutta::new(
                        &#static_name,
                    ),
                    problem,
                    options,
                )
            }
        }
    })
}

fn expand_rkn_algorithm_source(input: MacroInput, source: &str) -> Result<TokenStream2, String> {
    let tableau = parse_rkn_tableau(source, &input.name.to_string())
        .map_err(|error| format!("invalid tableau `{}`: {error}", input.path.value()))?;
    let visibility = input.visibility;
    let name = input.name;
    let static_name = format_ident!("__{}_TABLEAU", name.to_string().to_uppercase());
    let source_path = input.path;
    let crate_path = input.crate_path;
    let description = tableau.description();
    Ok(quote! {
        static #static_name: #crate_path::tableau::LazyRungeKuttaNystromTableau =
            ::std::sync::LazyLock::new(|| {
                #crate_path::tableau::parse_rkn_tableau(
                    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/", #source_path)),
                    stringify!(#name),
                )
            });

        #[doc = #description]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        #visibility struct #name;

        impl #name {
            /// Returns the lazily initialized, compile-time-validated tableau.
            #visibility fn tableau(
                self,
            ) -> ::std::result::Result<
                &'static #crate_path::tableau::RungeKuttaNystromTableau,
                #crate_path::tableau::TableauError,
            > {
                #crate_path::tableau::load_tableau(&#static_name)
            }
        }

        impl #crate_path::solvers::second_order::SecondOrderOdeAlgorithm for #name {
            fn solve_validated<F, P>(
                &self,
                problem: &#crate_path::solvers::second_order::SecondOrderOdeProblem<F, P>,
                options: &#crate_path::SolveOptions,
            ) -> ::std::result::Result<
                #crate_path::solvers::second_order::SecondOrderSolution,
                #crate_path::solvers::second_order::SecondOrderSolveError,
            >
            where
                F: #crate_path::solvers::second_order::SecondOrderFunction<P>,
            {
                #crate_path::solvers::second_order::SecondOrderOdeAlgorithm::solve_validated(
                    &#crate_path::solvers::second_order::ResourceRungeKuttaNystrom::new(
                        &#static_name,
                    ),
                    problem,
                    options,
                )
            }
        }
    })
}

pub(crate) fn expand_rkn_source(
    input: StaticTableauInput,
    source: &str,
) -> Result<TokenStream2, String> {
    parse_rkn_tableau(source, &input.method_name.value())
        .map_err(|error| format!("invalid tableau `{}`: {error}", input.path.value()))?;
    Ok(emit_lazy_static(
        input,
        quote!(LazyRungeKuttaNystromTableau),
        quote!(parse_rkn_tableau),
    ))
}

pub(crate) fn expand_irkn_source(
    input: StaticTableauInput,
    source: &str,
) -> Result<TokenStream2, String> {
    parse_irkn_tableau(source, &input.method_name.value())
        .map_err(|error| format!("invalid tableau `{}`: {error}", input.path.value()))?;
    Ok(emit_lazy_static(
        input,
        quote!(LazyIrknTableau),
        quote!(parse_irkn_tableau),
    ))
}

pub(crate) fn expand_low_storage_rk_source(
    input: StaticTableauInput,
    source: &str,
) -> Result<TokenStream2, String> {
    parse_low_storage_tableau(source, &input.method_name.value())
        .map_err(|error| format!("invalid tableau `{}`: {error}", input.path.value()))?;
    Ok(emit_lazy_static(
        input,
        quote!(LazyLowStorageRungeKuttaTableau),
        quote!(parse_low_storage_tableau),
    ))
}

#[cfg(test)]
mod low_storage_tests {
    use super::*;

    fn source() -> &'static str {
        r#"{"name":"Test2N","description":"Macro test","kind":"low-storage-runge-kutta","layout":"two-n","order":2,"A":["-1/2"],"b":["1/2",1],"c":["1/2"]}"#
    }

    #[test]
    fn static_expansion_embeds_only_validated_source() {
        let input: StaticTableauInput =
            syn::parse_str("pub TABLEAU, \"Test2N\", \"method.json\", crate = renamed").unwrap();
        let tokens = expand_low_storage_rk_source(input, source())
            .unwrap()
            .to_string();
        assert!(tokens.contains("LazyLowStorageRungeKuttaTableau"));
        assert!(tokens.contains("include_str"));
        assert!(tokens.contains("LazyLock"));
        assert!(tokens.contains("renamed"));
        assert!(!tokens.contains("0.5"));
        assert!(!tokens.contains("const "));
    }

    #[test]
    fn algorithm_expansion_is_one_resource_backed_solver_value() {
        let input = MacroInput {
            visibility: parse_quote!(pub),
            name: parse_quote!(Test2N),
            path: LitStr::new("method.json", proc_macro2::Span::call_site()),
            crate_path: parse_quote!(renamed),
        };
        let tokens = expand_low_storage_rk_algorithm_source(input, source())
            .unwrap()
            .to_string();
        assert!(tokens.contains("ResourceLowStorageRungeKutta"));
        assert!(tokens.contains("include_str"));
        assert!(tokens.contains("LazyLock"));
        assert!(tokens.contains("renamed"));
        assert!(!tokens.contains("0.5"));
        assert!(!tokens.contains("const "));
    }

    #[test]
    fn expansion_rejects_invalid_recurrence_with_resource_path() {
        let input: StaticTableauInput =
            syn::parse_str("TABLEAU, \"Test2N\", \"method.json\"").unwrap();
        let invalid = source().replace("\"c\":[\"1/2\"]", "\"c\":[0]");
        let error = expand_low_storage_rk_source(input, &invalid).unwrap_err();
        assert!(error.contains("method.json"), "{error}");
        assert!(error.contains("stage-row sum"), "{error}");
    }
}

pub(crate) fn expand_rosenbrock_source(
    input: StaticTableauInput,
    source: &str,
) -> Result<TokenStream2, String> {
    differential_equations_tableau_core::parse_rosenbrock_tableau(
        source,
        &input.method_name.value(),
    )
    .map_err(|error| format!("invalid tableau `{}`: {error}", input.path.value()))?;
    Ok(emit_lazy_static(
        input,
        quote!(LazyRosenbrockTableau),
        quote!(parse_rosenbrock_tableau),
    ))
}

fn read_algorithm_resource(input: &MacroInput) -> Result<String, String> {
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .ok_or("CARGO_MANIFEST_DIR is unavailable during macro expansion")?;
    let path = manifest_dir.join(input.path.value());
    std::fs::read_to_string(&path)
        .map_err(|error| format!("failed to read `{}`: {error}", path.display()))
}

pub(crate) fn expand_static(
    input: StaticTableauInput,
    expected_kind: RungeKuttaKind,
) -> Result<TokenStream2, String> {
    let source = read_static_resource(&input)?;
    expand_static_source(input, &source, expected_kind)
}

fn expand_static_source(
    input: StaticTableauInput,
    source: &str,
    expected_kind: RungeKuttaKind,
) -> Result<TokenStream2, String> {
    let tableau = parse_tableau(source, &input.method_name.value())
        .map_err(|error| format!("invalid tableau `{}`: {error}", input.path.value()))?;
    if tableau.kind() != expected_kind {
        let expected = match expected_kind {
            RungeKuttaKind::Explicit => "explicit",
            RungeKuttaKind::Implicit => "implicit",
        };
        return Err(format!(
            "tableau `{}` is not an {expected} Runge--Kutta method",
            input.path.value(),
        ));
    }

    Ok(emit_lazy_static(
        input,
        quote!(LazyTableau),
        quote!(parse_tableau),
    ))
}

#[cfg(test)]
mod implicit_tests {
    use super::*;

    #[test]
    fn implicit_expansion_validates_predictors_and_embeds_only_source() {
        let source = r#"{"name":"Pair","description":"Implicit pair","kind":"implicit-runge-kutta","order":1,"embedded_order":2,"A":[[0,0],["1/2","1/2"]],"b":[0,1],"c":[0,1],"error":["-1/2","1/2"],"stage_predictors":[[],[1]]}"#;
        let input = || {
            syn::parse_str::<StaticTableauInput>(
                "pub TABLEAU, \"Pair\", \"pair.json\", crate = renamed",
            )
            .unwrap()
        };
        let tokens = expand_static_source(input(), source, RungeKuttaKind::Implicit)
            .unwrap()
            .to_string();
        assert!(tokens.contains("include_str"));
        assert!(tokens.contains("LazyLock"));
        assert!(tokens.contains("renamed"));
        assert!(!tokens.contains("const "));
        for invalid in [
            source.replace("[[],[1]]", "[[],[1,0]]"),
            source.replace("[[],[1]]", "[[],[2]]"),
        ] {
            let error =
                expand_static_source(input(), &invalid, RungeKuttaKind::Implicit).unwrap_err();
            assert!(error.contains("pair.json"), "{error}");
            assert!(error.contains("stage_predictors"), "{error}");
        }
    }
}

/// Defines a symplectic composition from a validated JSON resource.
///
/// Like [`define_explicit_rk_from_file!`], accepts a visibility, method name,
/// resource path, and optional `crate = path`. The expansion contains an
/// `include_str!` resource and a per-method lazy tableau, not coefficient arrays.
pub fn define_symplectic_from_file(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as MacroInput);
    match expand_symplectic(input) {
        Ok(tokens) => tokens.into(),
        Err(error) => syn::Error::new(proc_macro2::Span::call_site(), error)
            .into_compile_error()
            .into(),
    }
}

fn expand_symplectic(input: MacroInput) -> Result<TokenStream2, String> {
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .ok_or("CARGO_MANIFEST_DIR is unavailable during macro expansion")?;
    let path = manifest_dir.join(input.path.value());
    let source = std::fs::read_to_string(&path)
        .map_err(|error| format!("failed to read `{}`: {error}", path.display()))?;
    expand_symplectic_source(input, &source)
}

fn expand_symplectic_source(input: MacroInput, source: &str) -> Result<TokenStream2, String> {
    parse_symplectic_tableau(source, &input.name.to_string())
        .map_err(|error| format!("invalid tableau `{}`: {error}", input.path.value()))?;
    let visibility = input.visibility;
    let name = input.name;
    let static_name = format_ident!("__{}_TABLEAU", name.to_string().to_uppercase());
    let source_path = input.path;
    let crate_path = input.crate_path;
    Ok(quote! {
        static #static_name: #crate_path::tableau::LazySymplecticTableau = ::std::sync::LazyLock::new(|| {
            #crate_path::tableau::parse_symplectic_tableau(
                include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/", #source_path)),
                stringify!(#name),
            )
        });
        #[doc = concat!("Resource-backed symplectic composition `", stringify!(#name), "`.")]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        #visibility struct #name;
        impl #name {
            /// Returns this method's lazily initialized tableau.
            pub fn tableau() -> ::std::result::Result<&'static #crate_path::tableau::SymplecticTableau, #crate_path::tableau::TableauError> {
                #crate_path::tableau::load_tableau(&#static_name)
            }
        }
        impl #crate_path::solvers::second_order::SymplecticAlgorithm for #name {
            fn tableau(&self) -> ::std::result::Result<&'static #crate_path::tableau::SymplecticTableau, #crate_path::tableau::TableauError> {
                #name::tableau()
            }
        }
    })
}

pub(crate) fn expand(input: MacroInput) -> Result<TokenStream2, String> {
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .ok_or("CARGO_MANIFEST_DIR is unavailable during macro expansion")?;
    let relative_path = input.path.value();
    let path = manifest_dir.join(&relative_path);
    let source = std::fs::read_to_string(&path)
        .map_err(|error| format!("failed to read `{}`: {error}", path.display()))?;
    let tableau = parse_tableau(&source, &input.name.to_string())
        .map_err(|error| format!("invalid tableau `{}`: {error}", path.display()))?;
    if tableau.kind() != RungeKuttaKind::Explicit {
        return Err(format!(
            "tableau `{}` is not an explicit Runge--Kutta method",
            path.display()
        ));
    }

    let visibility = input.visibility;
    let name = input.name;
    let static_name = format_ident!("__{}_TABLEAU", name.to_string().to_uppercase());
    let source_path = input.path;
    let crate_path = input.crate_path;
    let description = tableau.description();

    Ok(quote! {
        static #static_name: #crate_path::tableau::LazyTableau = ::std::sync::LazyLock::new(|| {
            #crate_path::tableau::parse_tableau(
                include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/", #source_path)),
                stringify!(#name),
            )
        });

        #[doc = #description]
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        #visibility struct #name;

        impl #name {
            #[doc = "Returns the lazily parsed, compile-time-validated tableau."]
            #visibility fn tableau(
                self,
            ) -> ::std::result::Result<
                &'static #crate_path::tableau::RungeKuttaTableau,
                #crate_path::tableau::TableauError,
            > {
                #crate_path::tableau::load_tableau(&#static_name)
            }
        }

        impl #crate_path::OdeAlgorithm for #name {
            fn solve_validated<F, P>(
                &self,
                problem: &#crate_path::OdeProblem<F, P>,
                options: &#crate_path::SolveOptions,
            ) -> Result<
                #crate_path::Solution,
                #crate_path::SolveError,
            >
            where
                F: #crate_path::OdeFunction<P>,
            {
                #crate_path::OdeAlgorithm::solve_validated(
                    &#crate_path::tableau::ResourceExplicitRungeKutta::new(&#static_name),
                    problem,
                    options,
                )
            }
        }
    })
}

#[cfg(test)]
mod symplectic_tests {
    use super::*;

    fn input() -> MacroInput {
        MacroInput {
            visibility: parse_quote!(pub),
            name: parse_quote!(TestComposition),
            path: LitStr::new("composition.json", proc_macro2::Span::call_site()),
            crate_path: parse_quote!(crate),
        }
    }

    const SOURCE: &str = r#"{"name":"TestComposition","description":"Macro test","kind":"symplectic-composition","order":2,"a":[1,0],"b":["1/2","1/2"]}"#;

    #[test]
    fn expansion_rejects_malformed_compositions_with_the_resource_path() {
        let source = SOURCE.replace("[1,0]", "[1]");
        let error = expand_symplectic_source(input(), &source).unwrap_err();
        assert!(error.contains("composition.json"), "{error}");
        assert!(error.contains("same non-zero stage count"), "{error}");
    }

    #[test]
    fn expansion_embeds_only_source_and_lazy_loading_not_coefficient_arrays() {
        let tokens = expand_symplectic_source(input(), SOURCE)
            .unwrap()
            .to_string();
        assert!(tokens.contains("include_str"));
        assert!(tokens.contains("LazyLock"));
        assert!(tokens.contains("parse_symplectic_tableau"));
        assert!(!tokens.contains("0.5"));
        assert!(!tokens.contains("const "));
    }
}

#[cfg(test)]
mod multistep_tests {

    use super::*;

    const SOURCE: &str = r#"{"name":"AB2","description":"Macro test","kind":"linear-multistep","order":2,"alpha":[1,-1,0],"beta":[0,"3/2","-1/2"]}"#;

    fn input() -> StaticTableauInput {
        syn::parse_str("pub FORMULA, \"AB2\", \"formula.json\", crate = renamed").unwrap()
    }

    #[test]
    fn multistep_expansion_embeds_source_not_coefficients() {
        let tokens = expand_multistep_source(input(), SOURCE)
            .unwrap()
            .to_string();
        assert!(tokens.contains("LazyMultistepTableau"));
        assert!(tokens.contains("include_str"));
        assert!(tokens.contains("parse_multistep_tableau"));
        assert!(tokens.contains("renamed"));
        assert!(!tokens.contains("1.5"));
        assert!(!tokens.contains("const "));
    }

    #[test]
    fn multistep_expansion_rejects_incorrect_order_conditions_with_the_path() {
        let invalid = SOURCE.replace("\"order\":2", "\"order\":3");
        let error = expand_multistep_source(input(), &invalid).unwrap_err();
        assert!(error.contains("formula.json"), "{error}");
        assert!(error.contains("order condition 3 failed"), "{error}");
    }

    #[test]
    fn multistep_expansion_validates_bdf_modifiers_before_runtime_use() {
        let source = r#"{"name":"AB2","description":"BDF macro test","kind":"backward-differentiation","order":2,"alpha":["3/2",-2,"1/2"],"beta":[1,0,0],"ndf_kappa":"-1/9"}"#;
        let tokens = expand_multistep_source(input(), source)
            .unwrap()
            .to_string();
        assert!(tokens.contains("include_str"));
        assert!(!tokens.contains("const "));
        let invalid = source.replace("\"-1/9\"", "1");
        let error = expand_multistep_source(input(), &invalid).unwrap_err();
        assert!(error.contains("formula.json"), "{error}");
        assert!(error.contains("NDF modifier"), "{error}");
    }
}

#[cfg(test)]
mod rosenbrock_tests {
    use super::*;

    const SOURCE: &str = r#"{"name":"Test","description":"Macro test","kind":"rosenbrock","order":2,"gamma":"1/2","A":[[0,0],[2,0]],"C":[[0,0],[-4,0]],"c":[0,1],"d":[0.5,-0.5],"b":[3,1],"btilde":[1,1],"H":[[1,0],[0,1]]}"#;

    fn input() -> StaticTableauInput {
        syn::parse_str("pub FORMULA, \"Test\", \"formula.json\", crate = renamed").unwrap()
    }

    #[test]
    fn rosenbrock_expansion_embeds_only_source_and_the_shared_parser() {
        let tokens = expand_rosenbrock_source(input(), SOURCE)
            .unwrap()
            .to_string();
        assert!(tokens.contains("LazyRosenbrockTableau"));
        assert!(tokens.contains("include_str"));
        assert!(tokens.contains("parse_rosenbrock_tableau"));
        assert!(tokens.contains("renamed"));
        assert!(!tokens.contains("0.5"));
        assert!(!tokens.contains("const "));
    }

    #[test]
    fn rosenbrock_validation_fails_during_macro_expansion() {
        for invalid in [
            SOURCE.replace("\"1/2\"", "0"),
            SOURCE.replace("\"1/2\"", "\"1/0\""),
            SOURCE.replace("[2,0]", "[2,1]"),
            SOURCE.replace("[0,1]]", "[0]]"),
            SOURCE.replace("\"btilde\":[1,1]", "\"btilde\":[1]"),
        ] {
            let error = expand_rosenbrock_source(input(), &invalid).unwrap_err();
            assert!(error.contains("formula.json"), "{error}");
        }
        let tokens =
            expand_rosenbrock_source(input(), &SOURCE.replace(",\"btilde\":[1,1]", "")).unwrap();
        assert!(tokens.to_string().contains("include_str"));
    }

    #[test]
    fn hybrid_validation_rejects_invalid_diagonals_at_compile_time() {
        let source = SOURCE
            .replace("\"rosenbrock\"", "\"hybrid-explicit-implicit\"")
            .replace("\"C\":[[0,0],[-4,0]]", "\"C\":[[0.5,0],[0,0.5]]")
            .replace("\"b\":[3,1]", "\"b\":[0.5,0.5]")
            .replace("\"btilde\":[1,1]", "\"btilde\":[1,-1]");
        let tokens = expand_rosenbrock_source(input(), &source)
            .unwrap()
            .to_string();
        assert!(tokens.contains("include_str"));
        assert!(!tokens.contains("0.5"));
        assert!(!tokens.contains("const "));
        let invalid = source.replace("\"C\":[[0.5,0],[0,0.5]]", "\"C\":[[0.5,0],[0,0]]");
        let error = expand_rosenbrock_source(input(), &invalid).unwrap_err();
        assert!(error.contains("formula.json"), "{error}");
        assert!(error.contains("must equal gamma"), "{error}");
    }
}

#[cfg(test)]
mod second_order_tests {
    use super::*;

    const RKN: &str = r#"{"name":"TestRkn","description":"RKN macro test","kind":"fixed-runge-kutta-nystrom","order":2,"A":[[0,0],["1/8",0]],"b":["1/2",0],"b_velocity":[0,1],"c":[0,"1/2"]}"#;
    const IRKN: &str = r#"{"name":"TestIrkn","description":"IRKN macro test","kind":"improved-runge-kutta-nystrom","order":3,"bootstrap_order":4,"bootstrap_seed":"previous-endpoint","velocity_history":["3/2","-1/2"],"c":["1/2"],"A":["1/8"],"velocity_weights":["2/3","5/6"],"history_weights":["1/3","5/12"]}"#;

    fn input(name: &str) -> StaticTableauInput {
        syn::parse_str(&format!(
            "pub TABLEAU, \"{name}\", \"second-order.json\", crate = renamed"
        ))
        .unwrap()
    }

    #[test]
    fn expansions_embed_only_source_and_use_the_shared_lazy_parsers() {
        let rkn = expand_rkn_source(input("TestRkn"), RKN)
            .unwrap()
            .to_string();
        assert!(rkn.contains("LazyRungeKuttaNystromTableau"));
        assert!(rkn.contains("parse_rkn_tableau"));
        assert!(rkn.contains("include_str"));
        assert!(rkn.contains("renamed"));
        assert!(!rkn.contains("0.125"));
        assert!(!rkn.contains("const "));

        let irkn = expand_irkn_source(input("TestIrkn"), IRKN)
            .unwrap()
            .to_string();
        assert!(irkn.contains("LazyIrknTableau"));
        assert!(irkn.contains("parse_irkn_tableau"));
        assert!(irkn.contains("include_str"));
        assert!(!irkn.contains("0.125"));
        assert!(!irkn.contains("const "));

        let algorithm_input: MacroInput =
            syn::parse_str("pub TestRkn, \"second-order.json\", crate = renamed").unwrap();
        let algorithm = expand_rkn_algorithm_source(algorithm_input, RKN)
            .unwrap()
            .to_string();
        assert!(algorithm.contains("ResourceRungeKuttaNystrom"));
        assert!(algorithm.contains("SecondOrderOdeAlgorithm"));
        assert!(algorithm.contains("include_str"));
        assert!(!algorithm.contains("0.125"));
        assert!(!algorithm.contains("const "));
    }

    #[test]
    fn malformed_second_order_resources_fail_during_expansion() {
        let rkn = RKN.replace("[\"1/8\",0]", "[\"1/8\",1]");
        let error = expand_rkn_source(input("TestRkn"), &rkn).unwrap_err();
        assert!(error.contains("second-order.json"), "{error}");
        assert!(error.contains("strictly lower triangular"), "{error}");

        let irkn = IRKN.replace("\"c\":[\"1/2\"]", "\"c\":[]");
        let error = expand_irkn_source(input("TestIrkn"), &irkn).unwrap_err();
        assert!(error.contains("second-order.json"), "{error}");
        assert!(error.contains("IRKN"), "{error}");
    }
}
