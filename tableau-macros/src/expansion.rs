//! Shared resource expansion primitives.

use crate::input::StaticTableauInput;
use differential_equations_tableau_core::parse_multistep_tableau;
use proc_macro2::TokenStream;
use quote::quote;
use std::path::PathBuf;

pub(crate) fn emit_lazy_static(
    input: StaticTableauInput,
    lazy_type: TokenStream,
    parser: TokenStream,
) -> TokenStream {
    let visibility = input.visibility;
    let static_name = input.static_name;
    let method_name = input.method_name;
    let source_path = input.path;
    let crate_path = input.crate_path;
    quote! {
        #[doc = concat!("Lazily parsed tableau `", #method_name, "`.")]
        #visibility static #static_name: #crate_path::tableau::#lazy_type =
            ::std::sync::LazyLock::new(|| {
                #crate_path::tableau::#parser(
                    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/", #source_path)),
                    #method_name,
                )
            });
    }
}

pub(crate) fn read_static_resource(input: &StaticTableauInput) -> Result<String, String> {
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .ok_or("CARGO_MANIFEST_DIR is unavailable during macro expansion")?;
    let path = manifest_dir.join(input.path.value());
    std::fs::read_to_string(&path)
        .map_err(|error| format!("failed to read `{}`: {error}", path.display()))
}

pub(crate) fn expand_multistep_source(
    input: StaticTableauInput,
    source: &str,
) -> Result<TokenStream, String> {
    parse_multistep_tableau(source, &input.method_name.value())
        .map_err(|error| format!("invalid tableau `{}`: {error}", input.path.value()))?;
    Ok(emit_lazy_static(
        input,
        quote!(LazyMultistepTableau),
        quote!(parse_multistep_tableau),
    ))
}
