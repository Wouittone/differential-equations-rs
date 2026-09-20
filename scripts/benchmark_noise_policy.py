#!/usr/bin/env python3
"""Freeze descriptive timing thresholds from immutable 3x9 baseline samples.

This conservative variability band is not a confidence interval. Improvements
must exceed both a 3% practical floor and three robust baseline scatter units.
Final reports must also show candidate scatter and achieved accuracy; a noisy
candidate remains inconclusive even when its median crosses this baseline band.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def policy(root):
    values = {backend: defaultdict(dict) for backend in ("upstream", "migrated")}
    hashes = {}
    for backend in values:
        for path in sorted(root.glob(f"{backend}-time-*.jsonl")):
            round_id = int(path.stem.rsplit("-", 1)[1])
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                key = (row["case"], row["abs_tol"], row["rel_tol"])
                elapsed = row["elapsed_ns"]
                if row.get("instrumented") or elapsed <= 0:
                    raise ValueError(f"Invalid timing sample in {path}")
                sample_id = (round_id, row["sample"])
                if sample_id in values[backend][key]:
                    raise ValueError(f"Duplicate timing sample in {path}")
                values[backend][key][sample_id] = elapsed
    if values["upstream"].keys() != values["migrated"].keys() or not values["upstream"]:
        raise ValueError("Baseline case sets differ or are empty")
    cases = []
    for key, upstream in sorted(values["upstream"].items()):
        migrated = values["migrated"][key]
        if upstream.keys() != migrated.keys() or len(upstream) < 27:
            raise ValueError(f"Incomplete 3x9 baseline: {key}")
        logarithms = [math.log(migrated[k] / upstream[k]) for k in sorted(upstream)]
        center = statistics.median(logarithms)
        scatter = 1.4826 * statistics.median(abs(x-center) for x in logarithms)
        band = max(math.log(1.03), 3.0*scatter)
        cases.append({"case": key[0], "abs_tol": key[1], "rel_tol": key[2],
                      "samples": len(logarithms), "baseline_paired_median_ratio": math.exp(center),
                      "baseline_robust_log_scatter": scatter,
                      "minimum_relative_change_factor": math.exp(band)})
    return {"source_files_sha256": hashes, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brahe", type=Path, required=True)
    parser.add_argument("--satkit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"schema_version": 1, "policy": __doc__,
              "brahe": policy(args.brahe), "satkit": policy(args.satkit)}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({name: len(report[name]["cases"]) for name in ("brahe", "satkit")}))


if __name__ == "__main__":
    main()
