#!/usr/bin/env python3
"""Balance sample-detail rows by carb_name using the smallest class count."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a balanced sample-details CSV by randomly selecting the "
            "minimum carb_name row count from every carb_name group."
        )
    )
    parser.add_argument(
        "-i",
        "--input-csv",
        required=True,
        type=Path,
        help="Input sample-details CSV containing a carb_name column.",
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        required=True,
        type=Path,
        help="Output balanced CSV path.",
    )
    parser.add_argument(
        "--carb-column",
        default="carb_name",
        help="Column to balance on. Default: carb_name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible row selection. Default: 42.",
    )
    parser.add_argument(
        "--sort-by-carb",
        action="store_true",
        help="Write output grouped by carb_name instead of preserving input-order positions.",
    )
    return parser.parse_args()


def read_rows(input_csv: Path, carb_column: str) -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")

    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV is empty: {input_csv}")
        if carb_column not in reader.fieldnames:
            raise ValueError(
                f"Column '{carb_column}' not found. Available columns: {', '.join(reader.fieldnames)}"
            )

        rows_by_carb: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in reader:
            rows_by_carb[row[carb_column]].append(row)

    if not rows_by_carb:
        raise ValueError(f"No data rows found in input CSV: {input_csv}")

    return reader.fieldnames, rows_by_carb


def select_balanced_rows(
    rows_by_carb: dict[str, list[dict[str, str]]],
    seed: int,
    sort_by_carb: bool,
) -> tuple[list[dict[str, str]], int]:
    min_count = min(len(rows) for rows in rows_by_carb.values())
    rng = random.Random(seed)

    selected_rows: list[dict[str, str]] = []
    carb_names = sorted(rows_by_carb) if sort_by_carb else rows_by_carb.keys()
    for carb_name in carb_names:
        rows = rows_by_carb[carb_name]
        selected_rows.extend(rng.sample(rows, min_count))

    if not sort_by_carb:
        selected_ids = {id(row) for row in selected_rows}
        selected_rows = [
            row
            for rows in rows_by_carb.values()
            for row in rows
            if id(row) in selected_ids
        ]

    return selected_rows, min_count


def write_rows(output_csv: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    fieldnames, rows_by_carb = read_rows(args.input_csv, args.carb_column)
    selected_rows, min_count = select_balanced_rows(
        rows_by_carb=rows_by_carb,
        seed=args.seed,
        sort_by_carb=args.sort_by_carb,
    )
    write_rows(args.output_csv, fieldnames, selected_rows)

    print(f"[INFO] Input CSV: {args.input_csv}")
    print(f"[INFO] Output CSV: {args.output_csv}")
    print(f"[INFO] Balance column: {args.carb_column}")
    print(f"[INFO] Random seed: {args.seed}")
    print(f"[INFO] Minimum rows per {args.carb_column}: {min_count}")
    print("[INFO] Original counts:")
    for carb_name in sorted(rows_by_carb):
        print(f"  {carb_name}: {len(rows_by_carb[carb_name])}")
    print("[INFO] Balanced counts:")
    for carb_name in sorted(rows_by_carb):
        print(f"  {carb_name}: {min_count}")
    print(f"[INFO] Total output rows: {len(selected_rows)}")


if __name__ == "__main__":
    main()
