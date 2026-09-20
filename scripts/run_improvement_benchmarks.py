#!/usr/bin/env python3
"""Serial downstream acceptance runs with provenance for the local library.

The fork runners already validate their own sources and binary hashes. This
wrapper additionally freezes the library's tracked files, which Cargo path
dependencies would otherwise leave outside those runners' manifests. Build and
measurement phases are separate so all competing compilation can finish first.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def snapshot(root):
    status = git(root, "status", "--porcelain", "--untracked-files=no").decode().strip()
    if status:
        raise RuntimeError(f"Commit tracked changes before freezing {root}:\n{status}")
    paths = set(git(root, "ls-files", "-z").decode().split("\0"))
    for folder in ("src", "tableau-core/src", "tableau-macros/src"):
        paths.update(p.relative_to(root).as_posix() for p in (root / folder).rglob("*") if p.is_file())
    hashes = {}
    for relative in sorted(paths):
        path = root / relative
        if relative and path.is_file():
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"commit": git(root, "rev-parse", "HEAD").decode().strip(), "files_sha256": hashes}


def validate_library_dependency(host, library):
    manifest = tomllib.loads((host / "Cargo.toml").read_text(encoding="utf-8"))
    expected = tomllib.loads((library / "Cargo.toml").read_text(encoding="utf-8"))["package"]["version"]
    dependencies = [value for key, value in manifest.get("dependencies", {}).items()
                    if key == "differential-equations-rs" or
                    isinstance(value, dict) and value.get("package") == "differential-equations-rs"]
    if len(dependencies) != 1 or not isinstance(dependencies[0], dict):
        raise RuntimeError(f"{host} must explicitly depend on the candidate library")
    dependency = dependencies[0]
    if not dependency.get("path") or (host / dependency["path"]).resolve() != library:
        raise RuntimeError(f"{host} does not use the frozen local library path")
    if dependency.get("version") != "=" + expected:
        raise RuntimeError(f"{host} must pin candidate version ={expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["build", "measure"])
    parser.add_argument("--brahe", type=Path, required=True)
    parser.add_argument("--satkit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=9)
    args = parser.parse_args()
    if args.rounds < 3 or args.samples < 9:
        parser.error("final acceptance requires at least 3 rounds and 9 samples")
    library = Path(__file__).resolve().parent.parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance_path = output / "library-provenance.json"
    frozen = snapshot(library)
    if args.phase == "measure":
        if json.loads(provenance_path.read_text(encoding="utf-8")) != frozen:
            raise RuntimeError("Library changed after benchmark build; rebuild candidates")
    elif provenance_path.exists():
        raise RuntimeError("Preserve immutable results: select a fresh output directory")
    for name, root, runner in [
        ("brahe", args.brahe.resolve(), "benchmarks/run_ode_comparison.py"),
        ("satkit", args.satkit.resolve(), "benches/run_ode_comparison.py"),
    ]:
        validate_library_dependency(root, library)
        if snapshot(library) != frozen:
            raise RuntimeError("Library changed during acceptance run")
        cmd = [sys.executable, str(root / runner),
               "--build-only" if args.phase == "build" else "--run-only",
               "--output", str(output / name), "--rounds", str(args.rounds),
               "--samples", str(args.samples), "--memory-rounds", str(args.rounds),
               "--memory-samples", str(args.samples)]
        subprocess.run(cmd, cwd=root, check=True)
        if snapshot(library) != frozen:
            raise RuntimeError("Library changed during acceptance run; discard mixed evidence")
    if args.phase == "build":
        provenance_path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
