#!/usr/bin/env python3
"""Merge one-process measurements into JSONL and a flat TSV."""

from __future__ import annotations

import csv
import json
import pathlib
import sys


def benchmark_row(stdout_path: pathlib.Path) -> dict[str, str]:
    lines = stdout_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if line.startswith("language,") and index + 1 < len(lines):
            header = next(csv.reader([line]))
            values = next(csv.reader([lines[index + 1]]))
            return dict(zip(header, values, strict=False))
    return {}


def read_metrics(path: pathlib.Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    return rows[-1] if rows else {}


def read_metadata(metrics_path: pathlib.Path) -> dict[str, str]:
    case_stem = metrics_path.name.removesuffix(".metrics.tsv")
    base_stem = case_stem.rsplit("_sample", 1)[0]
    meta_path = metrics_path.with_name(f"{base_stem}.meta.tsv")
    if not meta_path.exists():
        return {}
    with meta_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    return rows[-1] if rows else {}


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: aggregate.py RESULTS_DIR", file=sys.stderr)
        return 2
    results = pathlib.Path(sys.argv[1]).resolve()
    records = []
    for metrics_path in sorted(results.rglob("*.metrics.tsv")):
        metrics = read_metrics(metrics_path)
        case_id = metrics.get("case_id", metrics_path.stem)
        stdout_path = metrics_path.with_name(f"{case_id}.stdout")
        record = {"case_id": case_id, **read_metadata(metrics_path), **metrics, **benchmark_row(stdout_path)}
        records.append(record)

    jsonl_path = results / "results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    fields = sorted({field for record in records for field in record})
    with (results / "results.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    print(f"wrote {len(records)} records to {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
