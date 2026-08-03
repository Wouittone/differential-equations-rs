using Pkg
using TOML
using UUIDs: UUID

const UPSTREAM_URL = "https://github.com/SciML/OrdinaryDiffEq.jl.git"
const UPSTREAM_REVISION = "211142263781255a9aa2f910f6760b9f18ec29c8"
const PROJECT_DIRECTORY = @__DIR__

is_ordinary_diffeq(name) = startswith(name, "OrdinaryDiffEq")

function package_spec(name, uuid)
    PackageSpec(
        name = name,
        uuid = UUID(uuid),
        url = UPSTREAM_URL,
        rev = UPSTREAM_REVISION,
        subdir = "lib/$name",
    )
end

function manifest_entry(manifest, name)
    dependencies = get(manifest, "deps", nothing)
    dependencies isa Dict || error("Manifest.toml has no deps table; run this script without --check")
    entries = get(dependencies, name, nothing)
    entries === nothing && error("$name is absent from Manifest.toml; run this script without --check")
    entries isa Vector ? only(entries) : entries
end

function project_packages()
    project = TOML.parsefile(joinpath(PROJECT_DIRECTORY, "Project.toml"))
    Dict(
        name => uuid for (name, uuid) in project["deps"] if is_ordinary_diffeq(name)
    )
end

function manifest_packages(manifest)
    dependencies = get(manifest, "deps", Dict())
    Dict(
        name => manifest_entry(manifest, name)["uuid"] for name in keys(dependencies) if
            is_ordinary_diffeq(name)
    )
end

function pin_failures(manifest, packages)
    failures = String[]
    for (name, uuid) in sort!(collect(packages); by = first)
        entry = try
            manifest_entry(manifest, name)
        catch exception
            push!(failures, sprint(showerror, exception))
            continue
        end
        expected_subdir = "lib/$name"
        get(entry, "uuid", nothing) == uuid ||
            push!(failures, "$name has an unexpected UUID")
        get(entry, "repo-url", nothing) == UPSTREAM_URL ||
            push!(failures, "$name is not sourced from $UPSTREAM_URL")
        get(entry, "repo-rev", nothing) == UPSTREAM_REVISION ||
            push!(failures, "$name is not pinned to revision $UPSTREAM_REVISION")
        get(entry, "repo-subdir", nothing) == expected_subdir ||
            push!(failures, "$name is not sourced from monorepo subdirectory $expected_subdir")
    end
    failures
end

function check_pins()
    manifest_path = joinpath(PROJECT_DIRECTORY, "Manifest.toml")
    isfile(manifest_path) || error("$manifest_path is missing; run this script without --check")
    manifest = TOML.parsefile(manifest_path)
    packages = merge(manifest_packages(manifest), project_packages())
    isempty(packages) && error("the compliance project has no OrdinaryDiffEq dependencies")
    failures = pin_failures(manifest, packages)
    isempty(failures) || error(
        "Julia compliance environment is not pinned to the reference OrdinaryDiffEq revision:\n  - " *
            join(failures, "\n  - ") *
            "\nRun `julia --project=tests/julia tests/julia/pinned_environment.jl` to repair it.",
    )
    println("Verified $(length(packages)) OrdinaryDiffEq packages at $UPSTREAM_REVISION")
end

function setup_pins()
    Pkg.activate(PROJECT_DIRECTORY)
    packages = project_packages()
    for _ in 1:16
        Pkg.add([package_spec(name, uuid) for (name, uuid) in sort!(collect(packages); by = first)])
        manifest = TOML.parsefile(joinpath(PROJECT_DIRECTORY, "Manifest.toml"))
        resolved_packages = merge(manifest_packages(manifest), packages)
        if isempty(pin_failures(manifest, resolved_packages))
            Pkg.instantiate()
            check_pins()
            return
        end
        packages = resolved_packages
    end
    error(
        "Julia Pkg did not preserve exact monorepo revision/subdirectory metadata after " *
            "16 resolution passes; this Julia version may not support portable monorepo pins",
    )
end

if ARGS == ["--check"]
    check_pins()
elseif isempty(ARGS)
    setup_pins()
else
    error("usage: julia --project=tests/julia tests/julia/pinned_environment.jl [--check]")
end
