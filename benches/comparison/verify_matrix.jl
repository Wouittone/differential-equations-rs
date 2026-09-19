using Test

const EXPECTED_ALGORITHMS = Set([
    "Tsit5",
    "Midpoint",
    "Heun",
    "Ralston",
    "BS3",
    "DP5",
    "Euler",
    "RK4",
    "RKM",
    "Ralston4",
    "Alshina2",
    "Alshina3",
    "AB3",
    "AB4",
    "AB5",
    "ABM32",
    "ABM43",
    "ABM54",
    "SSPRK22",
    "SSPRK33",
    "SSPRK43",
    "ImplicitEuler",
    "ImplicitMidpoint",
    "Trapezoid",
    "Rosenbrock23",
    "TRBDF2",
    "Kvaerno5",
    "KenCarp5",
    "Rodas4P",
    "Rodas5P",
    "Rodas5Pr",
])

const STIFF_ALGORITHMS = Set([
    "ImplicitEuler",
    "ImplicitMidpoint",
    "Trapezoid",
    "Rosenbrock23",
    "TRBDF2",
    "Kvaerno5",
    "KenCarp5",
    "Rodas4P",
    "Rodas5P",
    "Rodas5Pr",
])

const EXPECTED_DIMENSIONS = Dict(
    algorithm => (algorithm in STIFF_ALGORITHMS ? 8 : 128) for
    algorithm in EXPECTED_ALGORITHMS
)

const HEADER = [
    "language",
    "algorithm",
    "dimension",
    "nanoseconds_per_solve",
    "bytes_allocated_per_solve",
    "allocations_per_solve",
    "rhs_evaluations_per_solve",
    "checksum",
]

function usage_error(message)
    error(
        "$message\nusage: julia verify_matrix.jl --rust RUST.csv --julia JULIA.csv " *
        "[--checksum-rtol VALUE] [--checksum-atol VALUE]",
    )
end

function parse_arguments(arguments)
    paths = Dict{String, String}()
    # The adaptive cases request 1e-7 from both solvers.  Permit two relative
    # tolerances for O(1) states and half an absolute tolerance when stiff
    # decay brings the expected endpoint close to zero.
    checksum_rtol = 2.0e-7
    checksum_atol = 5.0e-8
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        index == length(arguments) && usage_error("$argument requires a value")
        value = arguments[index + 1]
        if argument == "--rust" || argument == "--julia"
            paths[argument] = value
        elseif argument == "--checksum-rtol"
            checksum_rtol = parse(Float64, value)
        elseif argument == "--checksum-atol"
            checksum_atol = parse(Float64, value)
        else
            usage_error("unknown argument: $argument")
        end
        index += 2
    end
    haskey(paths, "--rust") || usage_error("--rust is required")
    haskey(paths, "--julia") || usage_error("--julia is required")
    isfinite(checksum_rtol) && checksum_rtol >= 0 ||
        usage_error("--checksum-rtol must be finite and non-negative")
    isfinite(checksum_atol) && checksum_atol >= 0 ||
        usage_error("--checksum-atol must be finite and non-negative")
    paths["--rust"], paths["--julia"], checksum_rtol, checksum_atol
end

function parse_float(value, field, source, line)
    try
        parse(Float64, value)
    catch
        error("$source:$line has invalid $field value: $value")
    end
end

function read_results(path, expected_language)
    lines = readlines(path)
    header_index = findfirst(line -> split(strip(line), ',') == HEADER, lines)
    isnothing(header_index) && error("$path does not contain the comparison CSV header")

    rows = Dict{String, NamedTuple}()
    for (offset, line) in enumerate(lines[(header_index + 1):end])
        line_number = header_index + offset
        isempty(strip(line)) && continue
        fields = split(strip(line), ',')
        length(fields) == length(HEADER) ||
            error("$path:$line_number has $(length(fields)) fields, expected $(length(HEADER))")
        language, algorithm = fields[1:2]
        language == expected_language ||
            error("$path:$line_number reports language $language, expected $expected_language")
        haskey(rows, algorithm) && error("$path contains duplicate row for $algorithm")
        dimension = try
            parse(Int, fields[3])
        catch
            error("$path:$line_number has invalid dimension: $(fields[3])")
        end
        rows[algorithm] = (
            dimension = dimension,
            nanoseconds = parse_float(fields[4], "nanoseconds", path, line_number),
            bytes = parse_float(fields[5], "allocated bytes", path, line_number),
            allocations = parse_float(fields[6], "allocation count", path, line_number),
            rhs_evaluations = parse_float(fields[7], "RHS evaluation count", path, line_number),
            checksum = parse_float(fields[8], "checksum", path, line_number),
        )
    end
    rows
end

function certify(rust_path, julia_path; checksum_rtol, checksum_atol)
    rust = read_results(rust_path, "rust")
    julia = read_results(julia_path, "julia")

    @testset "matched Rust/Julia benchmark certification" begin
        @test Set(keys(rust)) == EXPECTED_ALGORITHMS
        @test Set(keys(julia)) == EXPECTED_ALGORITHMS
        for algorithm in sort!(collect(EXPECTED_ALGORITHMS))
            @testset "$algorithm" begin
                rust_row = rust[algorithm]
                julia_row = julia[algorithm]
                @test rust_row.dimension == julia_row.dimension
                @test rust_row.dimension == EXPECTED_DIMENSIONS[algorithm]
                @test isfinite(rust_row.nanoseconds) && rust_row.nanoseconds > 0
                @test isfinite(julia_row.nanoseconds) && julia_row.nanoseconds > 0
                @test isnan(rust_row.bytes) || isfinite(rust_row.bytes) && rust_row.bytes >= 0
                @test isnan(julia_row.bytes) || isfinite(julia_row.bytes) && julia_row.bytes >= 0
                @test isnan(rust_row.allocations) ||
                      isfinite(rust_row.allocations) && rust_row.allocations >= 0
                @test isnan(julia_row.allocations) ||
                      isfinite(julia_row.allocations) && julia_row.allocations >= 0
                # Timings remain reports, not CI thresholds: shared runners are
                # too noisy for a cross-language speed gate.  Likewise, Julia
                # and Rust count finite-difference/Jacobian RHS work
                # differently.  Requiring finite positive metrics still makes
                # missing or malformed measurements fail certification.
                @test isfinite(rust_row.rhs_evaluations) && rust_row.rhs_evaluations > 0
                @test isfinite(julia_row.rhs_evaluations) && julia_row.rhs_evaluations > 0
                @test isfinite(rust_row.checksum)
                @test isfinite(julia_row.checksum)
                @test isapprox(
                    rust_row.checksum,
                    julia_row.checksum;
                    rtol = checksum_rtol,
                    atol = checksum_atol,
                )
            end
        end
    end

    println(
        "algorithm,dimension,rust_nanoseconds,julia_nanoseconds,rust_to_julia_time," *
        "rust_rhs,julia_rhs,rust_to_julia_rhs,checksum_absolute_difference"
    )
    for algorithm in sort!(collect(EXPECTED_ALGORITHMS))
        rust_row = rust[algorithm]
        julia_row = julia[algorithm]
        println(
            "$algorithm,$(rust_row.dimension),$(rust_row.nanoseconds),$(julia_row.nanoseconds)," *
            "$(rust_row.nanoseconds / julia_row.nanoseconds),$(rust_row.rhs_evaluations)," *
            "$(julia_row.rhs_evaluations)," *
            "$(rust_row.rhs_evaluations / julia_row.rhs_evaluations)," *
            "$(abs(rust_row.checksum - julia_row.checksum))"
        )
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    rust_path, julia_path, checksum_rtol, checksum_atol = parse_arguments(ARGS)
    certify(rust_path, julia_path; checksum_rtol, checksum_atol)
end
