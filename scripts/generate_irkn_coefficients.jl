const PINNED_REVISION = "211142263781255a9aa2f910f6760b9f18ec29c8"
const METHODS = (
    (
        name = "IRKN3",
        fields = ("bconst1", "bconst2", "c1", "a21", "b1", "b2", "bbar1", "bbar2"),
        order = 3,
        internal_stages = 1,
    ),
    (
        name = "IRKN4",
        fields = (
            "bconst1", "bconst2", "c1", "c2", "a21", "a32", "b1", "b2", "b3",
            "bbar1", "bbar2", "bbar3",
        ),
        order = 4,
        internal_stages = 2,
    ),
)

length(ARGS) in (2, 3) || error(
    "usage: julia scripts/generate_irkn_coefficients.jl <OrdinaryDiffEq checkout> <output.rs> [--check]",
)
check_only = length(ARGS) == 3 && ARGS[3] == "--check"
length(ARGS) == 2 || check_only || error("the only supported third argument is --check")
upstream = abspath(ARGS[1])
output = abspath(ARGS[2])
revision = readchomp(`git -c safe.directory=$upstream -C $upstream rev-parse HEAD`)
revision == PINNED_REVISION || error(
    "expected OrdinaryDiffEq revision $PINNED_REVISION, found $revision",
)

tableaus_path = joinpath(
    upstream, "lib", "OrdinaryDiffEqRKN", "src", "rkn_tableaus.jl",
)
source = read(tableaus_path, String)

function exact_rationals(method)
    signature = "function $(method.name)ConstantCache(T::Type, T2::Type)"
    start = findfirst(signature, source)
    isnothing(start) && error("could not find $signature in $tableaus_path")
    tail = source[last(start) + 1:end]
    finish = findfirst(r"(?m)^end\s*$", tail)
    isnothing(finish) && error("unterminated $signature in $tableaus_path")
    block = tail[1:first(finish) - 1]

    assignments = Dict{String, Tuple{Int, Int}}()
    for match in eachmatch(
            r"(?m)^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*convert\(T2?,\s*(-?\d+)\s*//\s*(\d+)\)\s*$",
            block,
        )
        assignments[match.captures[1]] = (
            parse(Int, match.captures[2]), parse(Int, match.captures[3]),
        )
    end
    found = Tuple(keys(assignments))
    missing = setdiff(method.fields, found)
    extra = setdiff(found, method.fields)
    isempty(missing) || error("missing $(method.name) coefficients: $(join(missing, ", "))")
    isempty(extra) || error("unexpected $(method.name) coefficients: $(join(extra, ", "))")

    constructor = "return $(method.name)ConstantCache($(join(method.fields, ", ")))"
    occursin(constructor, block) || error("unexpected $(method.name) constructor field order")
    return [assignments[field] for field in method.fields]
end

function rust_ratio(value)
    numerator, denominator = value
    return "$(numerator).0 / $(denominator).0"
end

function rust_array(io, name, values)
    println(io, "pub(super) const $name: [f64; $(length(values))] = [")
    for value in values
        println(io, "    $(rust_ratio(value)),")
    end
    println(io, "];\n")
end

mkpath(dirname(output))
generated_output = check_only ? tempname() : output
open(generated_output, "w") do io
    println(io, "//! Generated IRKN3/IRKN4 fixed-step coefficients.")
    println(io, "//! Source: SciML/OrdinaryDiffEq.jl at `$PINNED_REVISION`.")
    println(io, "//! Regenerate with `scripts/generate_irkn_coefficients.jl`.\n")

    for method in METHODS
        values = exact_rationals(method)
        coefficients = Dict(zip(method.fields, values))
        prefix = method.name
        println(io, "pub(super) const $(prefix)_ORDER: usize = $(method.order);")
        println(io, "pub(super) const $(prefix)_BOOTSTRAP_ORDER: usize = 4;")
        println(io, "pub(super) const $(prefix)_INTERNAL_STAGES: usize = $(method.internal_stages);")
        println(io, "pub(super) const $(prefix)_RETAINED_ENDPOINT_ACCELERATIONS: usize = 2;")
        println(io, "pub(super) const $(prefix)_RETAINED_INTERNAL_STAGES: usize = $(method.internal_stages);\n")
        rust_array(io, "$(prefix)_VELOCITY_HISTORY", [coefficients["bconst1"], coefficients["bconst2"]])
        rust_array(
            io,
            "$(prefix)_C",
            [coefficients["c$i"] for i in 1:method.internal_stages],
        )
        if method.name == "IRKN3"
            rust_array(io, "$(prefix)_A", [coefficients["a21"]])
        else
            # The only nonzero entries are a21 for stage 2 and a32 for stage 3.
            rust_array(io, "$(prefix)_A", [coefficients["a21"], coefficients["a32"]])
        end
        rust_array(
            io,
            "$(prefix)_VELOCITY_WEIGHTS",
            [coefficients["b$i"] for i in 1:(method.internal_stages + 1)],
        )
        rust_array(
            io,
            "$(prefix)_HISTORY_WEIGHTS",
            [coefficients["bbar$i"] for i in 1:(method.internal_stages + 1)],
        )
    end
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
