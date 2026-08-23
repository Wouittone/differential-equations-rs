using Printf

const PINNED_REVISION = "211142263781255a9aa2f910f6760b9f18ec29c8"

length(ARGS) in (2, 3) || error(
    "usage: julia scripts/generate_stabilized_coefficients.jl <OrdinaryDiffEq checkout> <output.rs> [--check]",
)
check_only = length(ARGS) == 3 && ARGS[3] == "--check"
length(ARGS) == 2 || check_only || error("the only supported third argument is --check")
upstream = abspath(ARGS[1])
output = abspath(ARGS[2])
revision = readchomp(`git -c safe.directory=$upstream -C $upstream rev-parse HEAD`)
revision == PINNED_REVISION || error(
    "expected OrdinaryDiffEq revision $PINNED_REVISION, found $revision",
)
source = joinpath(upstream, "lib", "OrdinaryDiffEqStabilizedRK", "src")

mutable struct ROCK2ConstantCache{T, T2, Z}
    ms; fp1; fp2; recf; zprev::Z; mdeg; deg_index; start; min_stage; max_stage; eig_age
end
mutable struct ROCK4ConstantCache{T, T2, T3, T4, Z}
    ms; fpa; fpb; fpbe; recf; zprev::Z; mdeg; deg_index; start; min_stage; max_stage; eig_age
end
mutable struct SERK2ConstantCache{T, Z}
    ms; zprev::Z; Bᵢ; mdeg; start; internal_deg
end
mutable struct ESERK4ConstantCache{T, Z}
    ms; Cᵤ; Cₑ; zprev::Z; Bᵢ; mdeg; start; internal_deg
end
mutable struct ESERK5ConstantCache{T, Z}
    ms; Cᵤ; Cₑ; zprev::Z; Bᵢ; mdeg; start; internal_deg
end

include(joinpath(source, "rkc_tableaus_rock2.jl"))
include(joinpath(source, "rkc_tableaus_rock4.jl"))
include(joinpath(source, "rkc_tableaus_serk2.jl"))
include(joinpath(source, "rkc_tableaus_eserk4.jl"))
include(joinpath(source, "rkc_tableaus_eserk5.jl"))

rock2 = ROCK2ConstantCache(Float64, Float64, nothing)
rock4 = ROCK4ConstantCache(Float64, Float64, nothing)
serk2 = SERK2ConstantCache(Float64, nothing)
eserk4 = ESERK4ConstantCache(Float64, nothing)
eserk5 = ESERK5ConstantCache(Float64, nothing)

function rust_array(io, name, values; integer = false, signed = false)
    type = signed ? "i32" : integer ? "usize" : "f64"
    println(io, "pub(super) const $name: &[$type] = &[")
    for item in vec(collect(values))
        item_values = item isa Union{Tuple, AbstractArray} ? item : (item,)
        for value in item_values
            println(io, "    ", integer || signed ? string(value) : repr(Float64(value)), ",")
        end
    end
    println(io, "];\n")
end

mkpath(dirname(output))
generated_output = check_only ? tempname() : output
open(generated_output, "w") do io
    println(io, "//! Generated stabilized-method coefficient banks.")
    println(io, "//! Source: SciML/OrdinaryDiffEq.jl at `$PINNED_REVISION`.")
    println(io, "//! Regenerate with `scripts/generate_stabilized_coefficients.jl`.\n")
    rust_array(io, "ROCK2_DEGREES", rock2.ms; integer = true)
    rust_array(io, "ROCK2_FINISH_FIRST", rock2.fp1)
    rust_array(io, "ROCK2_FINISH_SECOND", rock2.fp2)
    rust_array(io, "ROCK2_RECURRENCE", rock2.recf)
    rust_array(io, "ROCK4_DEGREES", rock4.ms; integer = true)
    rust_array(io, "ROCK4_FINISH_A", rock4.fpa)
    rust_array(io, "ROCK4_FINISH_B", rock4.fpb)
    rust_array(io, "ROCK4_FINISH_ERROR", rock4.fpbe)
    rust_array(io, "ROCK4_RECURRENCE", rock4.recf)
    rust_array(io, "SERK2_DEGREES", serk2.ms; integer = true)
    rust_array(io, "SERK2_WEIGHTS", serk2.Bᵢ)
    rust_array(io, "ESERK4_DEGREES", eserk4.ms; integer = true)
    rust_array(io, "ESERK4_SOLUTION_COMBINATION", eserk4.Cᵤ; signed = true)
    rust_array(io, "ESERK4_ERROR_COMBINATION", eserk4.Cₑ; signed = true)
    rust_array(io, "ESERK4_WEIGHTS", eserk4.Bᵢ)
    rust_array(io, "ESERK5_DEGREES", eserk5.ms; integer = true)
    rust_array(io, "ESERK5_SOLUTION_COMBINATION", eserk5.Cᵤ; signed = true)
    rust_array(io, "ESERK5_ERROR_COMBINATION", eserk5.Cₑ; signed = true)
    rust_array(io, "ESERK5_WEIGHTS", eserk5.Bᵢ)
end
run(`rustfmt --edition 2024 $generated_output`)

if check_only
    isfile(output) || error("$output does not exist")
    generated = read(generated_output)
    existing = read(output)
    rm(generated_output; force = true)
    generated == existing || error("$output is stale; regenerate it from $PINNED_REVISION")
    println("Verified byte-stable $output from OrdinaryDiffEq $PINNED_REVISION")
else
    println("Wrote $output from OrdinaryDiffEq $PINNED_REVISION")
end
