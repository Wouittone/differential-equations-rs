#!/usr/bin/env python3
"""Format tableau JSON without changing coefficient spellings."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias


@dataclass(frozen=True)
class RawNumber:
    """A JSON number retained in its original lexical representation."""

    text: str


JsonScalar: TypeAlias = None | bool | str | RawNumber
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def reject_non_finite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def read_json(path: Path) -> JsonValue:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_int=RawNumber,
        parse_float=RawNumber,
        parse_constant=reject_non_finite,
    )


def is_scalar(value: JsonValue) -> bool:
    return value is None or isinstance(value, (bool, str, RawNumber))


def format_scalar(value: JsonScalar) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, RawNumber):
        return value.text
    return json.dumps(value, ensure_ascii=False)


def format_value(value: JsonValue, indent: int = 0) -> str:
    if is_scalar(value):
        return format_scalar(value)

    if isinstance(value, list):
        if not value:
            return "[]"
        if all(is_scalar(item) for item in value):
            return "[" + ", ".join(format_scalar(item) for item in value) + "]"

        lines = ["["]
        for index, item in enumerate(value):
            rendered = format_value(item, indent + 2).splitlines()
            rendered[0] = " " * (indent + 2) + rendered[0]
            if index + 1 != len(value):
                rendered[-1] += ","
            lines.extend(rendered)
        lines.append(" " * indent + "]")
        return "\n".join(lines)

    if not value:
        return "{}"

    encoded_keys = [(json.dumps(key, ensure_ascii=False), item) for key, item in value.items()]
    key_width = max(len(key) for key, _ in encoded_keys)
    lines = ["{"]
    for index, (key, item) in enumerate(encoded_keys):
        rendered = format_value(item, indent + 2).splitlines()
        rendered[0] = " " * (indent + 2) + key.ljust(key_width) + " : " + rendered[0]
        if index + 1 != len(encoded_keys):
            rendered[-1] += ","
        lines.extend(rendered)
    lines.append(" " * indent + "}")
    return "\n".join(lines)


def tableau_paths(root: Path) -> list[Path]:
    paths = sorted((root / "tableaux").rglob("*.json"))
    paths.extend(
        path
        for path in (
            root / "examples" / "resources" / "file_heun.json",
            root / "tests" / "resources" / "file_heun.json",
        )
        if path.is_file()
    )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report files that differ instead of rewriting them",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    changed: list[Path] = []
    for path in tableau_paths(root):
        original = path.read_text(encoding="utf-8")
        formatted = format_value(read_json(path)) + "\n"
        if original == formatted:
            continue
        changed.append(path.relative_to(root))
        if not args.check:
            path.write_text(formatted, encoding="utf-8", newline="\n")

    if args.check and changed:
        for path in changed:
            print(path.as_posix())
        print(f"{len(changed)} tableau JSON file(s) require formatting", file=sys.stderr)
        return 1

    action = "would reformat" if args.check else "reformatted"
    print(f"{action} {len(changed)} of {len(tableau_paths(root))} tableau JSON files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
